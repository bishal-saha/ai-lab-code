"""The model clients: real Claude, and offline fakes for the dry run.

`ClaudeModel` makes tool-calling turns (automatic prompt caching on the growing transcript)
and plain summaries. The fakes test the plumbing without a key:

    oracle     solves every ticket and always keeps both standing instructions
    forgetful  solves every ticket, but keeps a standing instruction only while its text is
               literally in the window it was sent: the behaviour the article worries about
    lossy summariser  writes a generic summary that leaves the instructions out

The dry run uses them to prove the grader catches a forgotten instruction, and that only the
unpinned summary policy drops the instruction text from the window. Their numbers say nothing
about real models; --dry-run output is never written to results/.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import anthropic

from ticket_thirty.pricing import resolve
from ticket_thirty.scenarios import Scenario
from ticket_thirty.tools import AUDIT_REASON

MAX_TOKENS = 8000


@dataclass
class Turn:
    content: list[Any]  # goes back into the transcript unchanged (thinking blocks included)
    blocks: list[dict]  # normalised: tool_use blocks with id, name, input
    stop_reason: str
    usage: dict[str, int] = field(default_factory=dict)
    text: str = ""


def _usage(u) -> dict[str, int]:
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read": u.cache_read_input_tokens or 0,
        "cache_write": u.cache_creation_input_tokens or 0,
    }


class ClaudeModel:
    """Claude through the Anthropic SDK; thinking at the model's default (adaptive), effort pinned."""

    def __init__(
        self, model: str, *, effort: str = "medium", client: anthropic.AsyncAnthropic | None = None
    ):
        self.name = resolve(model)
        self.effort = effort
        self.client = client or anthropic.AsyncAnthropic(max_retries=6)

    async def create(self, *, system: str, messages: list[dict], tools: list[dict]) -> Turn:
        r = await self.client.messages.create(
            model=self.name,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
            tools=tools,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},  # the growing transcript is re-read every turn
        )
        blocks = [
            {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
            for b in r.content
            if b.type == "tool_use"
        ]
        text = "".join(b.text for b in r.content if b.type == "text")
        return Turn(list(r.content), blocks, r.stop_reason or "", _usage(r.usage), text)

    async def summarise(self, *, system: str, prompt: str) -> tuple[str, dict[str, int]]:
        r = await self.client.messages.create(
            model=self.name,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low"},
        )
        return "".join(b.text for b in r.content if b.type == "text"), _usage(r.usage)


# --- offline fakes ------------------------------------------------------------------------

_ids = itertools.count(1)


def _window_text(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        c = m["content"]
        if isinstance(c, str):
            parts.append(c)
            continue
        for b in c:
            if isinstance(b, dict):
                parts.append(
                    str(b.get("text", "")) + str(b.get("content", "")) + str(b.get("input", ""))
                )
            else:
                parts.append(getattr(b, "text", "") or "")
    return "\n".join(parts)


def _calls_so_far(messages: list[dict], ticket_id: str) -> list[str]:
    """Tool names already called on the current ticket (the turns after its announcement)."""
    names: list[str] = []
    for m in reversed(messages):
        if m["role"] == "user":
            content = (
                m["content"]
                if isinstance(m["content"], list)
                else [{"type": "text", "text": m["content"]}]
            )
            if any(
                b.get("type") == "text" and f"Next ticket: {ticket_id}" in b.get("text", "")
                for b in content
                if isinstance(b, dict)
            ):
                break
        if m["role"] == "assistant":
            names += [
                b["name"] if isinstance(b, dict) else b.name
                for b in m["content"]
                if (b["type"] if isinstance(b, dict) else b.type) == "tool_use"
            ]
    return names


class FakeModel:
    def __init__(self, behaviour: str, scenarios: list[Scenario]):
        assert behaviour in ("oracle", "forgetful")
        self.behaviour = behaviour
        self.name = "fake"
        self.by_ticket = {s.ticket_id: s for s in scenarios}

    def _plan(self, s: Scenario, freeze: bool, audit: bool) -> list[tuple[str, dict]]:
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
                        {
                            "order_id": s.order_id,
                            "amount_usd": s.total,
                            "reason": "Inside the window",
                        },
                    )
                )
            elif s.outcome == "escalate" and freeze:
                plan.append(
                    (
                        "reply_to_customer",
                        {
                            "ticket_id": s.ticket_id,
                            "message": "Your refund is delayed until Finance reopens on Monday.",
                        },
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
            plan.append(
                ("route_to_engineering", {"ticket_id": s.ticket_id, "summary": s.bug or "bug"})
            )
        if audit:
            plan.append(("flag_for_review", {"ticket_id": s.ticket_id, "reason": AUDIT_REASON}))
        return plan

    async def create(self, *, system: str, messages: list[dict], tools: list[dict]) -> Turn:
        window = _window_text(messages)
        ticket_id = window[window.rfind("Next ticket: ") + 13 :].split()[0]
        s = self.by_ticket[ticket_id]
        if self.behaviour == "oracle":
            freeze = audit = True
        else:
            freeze, audit = "frozen escalations" in window, AUDIT_REASON in window
        done = _calls_so_far(messages, ticket_id)
        plan = self._plan(s, freeze, audit)
        usage = {
            "input_tokens": len(window) // 4,
            "output_tokens": 40,
            "cache_read": 0,
            "cache_write": 0,
        }
        if len(done) >= len(plan):
            return Turn(
                [{"type": "text", "text": f"{ticket_id} done."}],
                [],
                "end_turn",
                usage,
                f"{ticket_id} done.",
            )
        name, args = plan[len(done)]
        block = {"type": "tool_use", "id": f"toolu_fake{next(_ids)}", "name": name, "input": args}
        return Turn([block], [block], "tool_use", usage)


class LossySummariser:
    """A generic summary that keeps the work done and drops the standing instructions."""

    name = "fake"

    async def summarise(self, *, system: str, prompt: str) -> tuple[str, dict[str, int]]:
        tickets = sorted(set(w.strip(".,") for w in prompt.split() if w.startswith("T-Q")))
        return (
            f"The agent is working through a support queue. Tickets handled so far: {', '.join(tickets)}.",
            {
                "input_tokens": len(prompt) // 4,
                "output_tokens": 30,
                "cache_read": 0,
                "cache_write": 0,
            },
        )
