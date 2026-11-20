"""The model clients: real Claude, and an offline fake that bills like the API.

`ClaudeModel.create()` takes the model per call, because routing conditions change it mid-attempt.
Caching is the API's automatic prompt caching (a top-level `cache_control`), switched off by
omitting it.

`FakeModel` solves each ticket from its ground truth and reports usage the way the API would,
from a small cache simulator: per model, the longest prefix it has seen in the last five minutes
is read, the rest written, and nothing is cached when caching is off. The dry run uses it to prove
the cost accounting and that each condition shapes the bill the way it should. Its numbers say
nothing about real models; --dry-run output is never written to results/.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Any

import anthropic

from bill.scenarios import Scenario

MAX_TOKENS = 8000


@dataclass
class Turn:
    content: list[Any]  # goes back into the transcript unchanged (thinking blocks included)
    blocks: list[dict]  # tool_use blocks: id, name, input
    stop_reason: str
    usage: dict[str, int] = field(default_factory=dict)
    text: str = ""


def usage_dict(u) -> dict[str, int]:
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read": u.cache_read_input_tokens or 0,
        "cache_write": u.cache_creation_input_tokens or 0,
    }


class ClaudeModel:
    """Claude through the Anthropic SDK; thinking at the model's default (adaptive), effort pinned."""

    def __init__(self, *, effort: str = "medium", client: anthropic.AsyncAnthropic | None = None):
        self.effort = effort
        self.client = client or anthropic.AsyncAnthropic(max_retries=6)

    async def create(
        self, *, model: str, system: str, messages: list[dict], tools: list[dict], caching: bool
    ) -> Turn:
        extra = {"cache_control": {"type": "ephemeral"}} if caching else {}
        r = await self.client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
            tools=tools,
            output_config={"effort": self.effort},
            **extra,
        )
        blocks = [
            {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
            for b in r.content
            if b.type == "tool_use"
        ]
        text = "".join(b.text for b in r.content if b.type == "text")
        return Turn(list(r.content), blocks, r.stop_reason or "", usage_dict(r.usage), text)

    async def summarise(
        self, *, model: str, system: str, prompt: str
    ) -> tuple[str, dict[str, int]]:
        r = await self.client.messages.create(
            model=model,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low"},
        )
        return "".join(b.text for b in r.content if b.type == "text"), usage_dict(r.usage)

    async def count_cache(self, *, model: str, text: str) -> dict[str, int]:
        """One cached call with `text` as the system prompt: the cache-minimum micro-check."""
        r = await self.client.messages.create(
            model=model,
            max_tokens=16,
            system=text,
            messages=[{"role": "user", "content": "Reply with OK."}],
            cache_control={"type": "ephemeral"},
        )
        return usage_dict(r.usage)


# --- the offline fake ------------------------------------------------------------------------

_ids = itertools.count(1)


def _tokens(obj: Any) -> int:
    return len(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o)))) // 4


def plan_for(s: Scenario) -> list[tuple[str, dict]]:
    """The calls an agent that knows the answer makes, in order."""
    plan: list[tuple[str, dict]] = [("get_ticket", {"ticket_id": s.ticket_id})]
    if not s.order_in_ticket:
        plan.append(("customer_orders", {"email": s.customer_email}))
    if s.kind == "status":
        plan += [
            ("order_status", {"order_id": s.order_id}),
            (
                "reply_to_customer",
                {"ticket_id": s.ticket_id, "message": "Here is where your order is."},
            ),
        ]
    else:
        plan.append(("check_return_eligibility", {"order_id": s.order_id, "as_of": s.today}))
        if s.outcome == "refund":
            plan.append(
                (
                    "issue_refund",
                    {"order_id": s.order_id, "amount_usd": s.total, "reason": "Inside the window"},
                )
            )
        elif s.outcome == "escalate":
            plan.append(
                ("escalate_to_finance", {"order_id": s.order_id, "reason": "Above USD 500"})
            )
        else:
            plan.append(
                (
                    "reply_to_customer",
                    {
                        "ticket_id": s.ticket_id,
                        "message": "Sorry, this return is outside the window.",
                    },
                )
            )
    if s.needs_engineering:
        plan.append(("route_to_engineering", {"ticket_id": s.ticket_id, "summary": s.bug or "bug"}))
    return plan


class FakeModel:
    def __init__(self, scenarios: list[Scenario]):
        self.by_ticket = {s.ticket_id: s for s in scenarios}
        self.seen: dict[
            str, list[str]
        ] = {}  # model -> prompts it has cached (as serialised blocks)

    def _bill(self, model: str, prompt: list[str], caching: bool) -> dict[str, int]:
        total = sum(len(p) // 4 for p in prompt)
        if not caching:
            return {"input_tokens": total, "output_tokens": 60, "cache_read": 0, "cache_write": 0}
        best = 0
        for old in self.seen.get(model, []):
            n = 0
            for a, b in zip(old, prompt, strict=False):
                if a != b:
                    break
                n += 1
            best = max(best, n)
        read = sum(len(p) // 4 for p in prompt[:best])
        self.seen.setdefault(model, []).append(prompt)
        return {
            "input_tokens": 0,
            "output_tokens": 60,
            "cache_read": read,
            "cache_write": total - read,
        }

    async def create(
        self, *, model: str, system: str, messages: list[dict], tools: list[dict], caching: bool
    ) -> Turn:
        prompt = [json.dumps(tools), system] + [json.dumps(m, default=str) for m in messages]
        usage = self._bill(model, prompt, caching)
        ticket_id = next(
            w.strip(".") for w in str(messages[0]["content"]).split() if w.startswith("T-B")
        )
        s = self.by_ticket[ticket_id]
        done = sum(
            1
            for m in messages
            if m["role"] == "assistant"
            for b in m["content"]
            if (b["type"] if isinstance(b, dict) else b.type) == "tool_use"
        )
        # under a history cap, the calls folded into the summary are counted from its marker
        text = str(messages[0]["content"])
        if "[calls so far: " in text:
            done += int(text.split("[calls so far: ")[1].split("]")[0])
        plan = plan_for(s)
        if done >= len(plan):
            return Turn([{"type": "text", "text": "Done."}], [], "end_turn", usage, "Done.")
        name, args = plan[done]
        block = {"type": "tool_use", "id": f"toolu_fake{next(_ids)}", "name": name, "input": args}
        return Turn([block], [block], "tool_use", usage)

    async def summarise(
        self, *, model: str, system: str, prompt: str
    ) -> tuple[str, dict[str, int]]:
        n = prompt.count("AGENT CALLED")
        prev = (
            int(prompt.split("[calls so far: ")[1].split("]")[0])
            if "[calls so far: " in prompt
            else 0
        )
        return f"Earlier steps handled. [calls so far: {prev + n}]", {
            "input_tokens": len(prompt) // 4,
            "output_tokens": 30,
            "cache_read": 0,
            "cache_write": 0,
        }
