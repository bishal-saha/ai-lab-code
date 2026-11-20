"""One attempt on one ticket, in a fresh window, under one condition: the unit the article prices.

The loop calls the model, runs its tool calls, and stops when the model ends its turn without a
tool call or the turn budget runs out. Before every call the condition decides the model, whether
caching is on, whether the system prompt carries a timestamp, and (under the history cap) which
exchanges the window keeps.

Two details keep the comparison fair:
- Thinking blocks are model-specific, so in the routing conditions each model sees its own earlier
  messages with their thinking and the other model's without. Each model's view of the history
  then only ever grows by appending, which is what its cache needs.
- Every model call, the cap's summaries included, is priced from the usage the API reports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from kestrel.services import KestrelData

from bill.conditions import HAIKU, Condition
from bill.grading import grade
from bill.ledger import Ledger
from bill.pricing import cost_usd
from bill.scenarios import REFUND_LIMIT_USD, Scenario
from bill.tools import Tools

MAX_TURNS = 12

SYSTEM = f"""You are a support agent for Kestrel Outfitters, an outdoor-gear retailer. Resolve the ticket you are given with the tools provided.

The returns policy:
- A ticket without an order id: find the order with customer_orders, using the customer's email from the ticket.
- A delivery question ("where is my order?"): look the order up with order_status and reply with where it is. No money moves.
- A return request: check it with check_return_eligibility (as of the date the customer wrote) before any money action, then take the one outcome the policy calls for:
  - inside the window and total USD {REFUND_LIMIT_USD} or less: issue_refund for the full order total;
  - inside the window and total above USD {REFUND_LIMIT_USD}: escalate_to_finance;
  - outside the window: reply_to_customer to explain that the return can't be accepted.
- A website or app bug the customer reports: route_to_engineering with a short summary.
- Anything these tools can't resolve: escalate_to_human.

When the ticket is finished, say so in one short line without calling a tool."""

SUMMARISER_SYSTEM = """You compact an AI support agent's working history on one ticket so it can continue with less context.
Write a short summary of the earlier steps below (what was looked up, what was found, what was done), folding in the previous summary if there is one. Keep ids, amounts, dates and decisions exact. Write only the summary."""


class BudgetExceeded(Exception):
    """The spending cap was reached; the job is abandoned and not recorded (resume reruns it)."""


class Budget:
    """A cap shared by every concurrent job, checked before each model call."""

    def __init__(self, max_usd: float):
        self.max_usd, self.spent = max_usd, 0.0

    def check(self) -> None:
        if self.spent >= self.max_usd:
            raise BudgetExceeded(f"cap USD {self.max_usd:.2f} reached")

    def add(self, cost: float) -> None:
        self.spent += cost


@dataclass
class Message:
    role: str
    content: list[Any]
    model: str = ""  # who wrote an assistant message


@dataclass
class Attempt:
    outcome: str
    violation: bool
    calls: list[dict[str, Any]] = field(default_factory=list)
    summaries: list[dict[str, Any]] = field(default_factory=list)
    ledger: list[dict[str, Any]] = field(default_factory=list)
    cost_usd: float = 0.0
    turn_budget_hit: bool = False


def _plain(block: Any) -> dict | None:
    kind = block["type"] if isinstance(block, dict) else block.type
    get = (lambda k: block[k]) if isinstance(block, dict) else (lambda k: getattr(block, k))
    if kind == "text":
        return {"type": "text", "text": get("text")} if get("text") else None
    if kind == "tool_use":
        return {"type": "tool_use", "id": get("id"), "name": get("name"), "input": get("input")}
    return None


def _as_seen_by(m: Message, model: str) -> dict:
    if m.role == "assistant" and m.model != model:
        blocks = [b for b in map(_plain, m.content) if b]
        return {"role": "assistant", "content": blocks or [{"type": "text", "text": "(working)"}]}
    return {"role": m.role, "content": m.content}


def _render(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        for b in m.content:
            b = b if isinstance(b, dict) else (_plain(b) or {})
            if b.get("type") == "tool_use":
                lines.append(f"AGENT CALLED {b['name']}({b['input']})")
            elif b.get("type") == "tool_result":
                lines.append(f"TOOL RESULT: {b['content']}")
            elif b.get("type") == "text":
                lines.append(f"{m.role.upper()}: {b['text']}")
    return "\n".join(lines)


def build_window(
    prompt: str, exchanges: list[list[Message]], cond: Condition, summary: str | None, model: str
) -> list[dict]:
    """The messages for the next call. An exchange is [assistant tool_use, user tool_result]."""
    kept = exchanges if cond.history_cap is None else exchanges[-cond.history_cap :]
    head = (
        prompt
        if not (summary and len(kept) < len(exchanges))
        else f"{prompt}\n\nEarlier steps on this ticket, summarised:\n{summary}"
    )
    msgs = [{"role": "user", "content": head}]
    for ex in kept:
        msgs += [_as_seen_by(m, model) for m in ex]
    return msgs


async def attempt(
    client,
    cond: Condition,
    s: Scenario,
    data: KestrelData,
    scenarios: list[Scenario],
    *,
    budget: Budget | None = None,
    job_tag: str = "",
) -> Attempt:
    ledger = Ledger()
    ledger.current = s.ticket_id
    tools = Tools(data, scenarios, ledger)
    schemas = tools.schemas()
    # The job tag makes every job's first message unique. Without it, every condition sends the
    # same prompt for the same ticket, and a condition run minutes after another read that one's
    # cache (seen in the first pilot). A retry keeps its job's tag: a real retry would resend the
    # same prompt, so its cache hits are fair.
    prompt = f"Please handle support ticket {s.ticket_id}." + (
        f" (job {job_tag})" if job_tag else ""
    )
    exchanges: list[list[Message]] = []
    summary: str | None = None
    folded = 0  # exchanges already in the summary
    result = Attempt(outcome="failed", violation=False)

    def charge(model: str, u: dict) -> float:
        cost = cost_usd(
            model, u["input_tokens"], u["output_tokens"], u["cache_read"], u["cache_write"]
        )
        result.cost_usd += cost
        if budget:
            budget.add(cost)
        return cost

    for step in range(MAX_TURNS):
        if cond.history_cap is not None and len(exchanges) - folded > cond.history_cap:
            drop = exchanges[folded : len(exchanges) - cond.history_cap]
            text = (
                (f"Previous summary:\n{summary}\n\n" if summary else "")
                + "Steps to fold in:\n"
                + _render([m for ex in drop for m in ex])
            )
            if budget:
                budget.check()
            summary, su = await client.summarise(model=HAIKU, system=SUMMARISER_SYSTEM, prompt=text)
            result.summaries.append(
                {"step": step, "folded": len(drop), "cost_usd": charge(HAIKU, su), **su}
            )
            folded += len(drop)
        model = cond.model_for(step)
        system = (
            f"Current time: {datetime.now(UTC).isoformat()}\n\n{SYSTEM}"
            if cond.timestamp
            else SYSTEM
        )
        messages = build_window(prompt, exchanges, cond, summary, model)
        if budget:
            budget.check()
        t0 = time.monotonic()
        t = await client.create(
            model=model, system=system, messages=messages, tools=schemas, caching=cond.caching
        )
        u = t.usage
        result.calls.append(
            {
                "step": step,
                "model": model,
                "stop_reason": t.stop_reason,
                **u,
                "window_tokens": u["input_tokens"] + u["cache_read"] + u["cache_write"],
                "cost_usd": charge(model, u),
                "seconds": round(time.monotonic() - t0, 2),
            }
        )
        reply = Message("assistant", t.content, model)
        if not t.blocks:
            break
        if t.stop_reason in ("max_tokens", "refusal"):  # every tool_use still needs a result
            exchanges.append(
                [
                    reply,
                    Message(
                        "user",
                        [
                            {
                                "type": "tool_result",
                                "tool_use_id": b["id"],
                                "content": "Not run: the response was cut off.",
                                "is_error": True,
                            }
                            for b in t.blocks
                        ],
                    ),
                ]
            )
            break
        results = []
        for b in t.blocks:
            text, is_error = tools.call(
                b["name"], b["input"] if isinstance(b["input"], dict) else {}
            )
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": b["id"],
                    "content": text,
                    **({"is_error": True} if is_error else {}),
                }
            )
        exchanges.append([reply, Message("user", results)])
    else:
        result.turn_budget_hit = True
    g = grade(s, ledger)
    result.outcome, result.violation = g.outcome, g.violation
    result.ledger = ledger.to_list()
    return result
