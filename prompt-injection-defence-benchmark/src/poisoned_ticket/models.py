"""The two model clients: real Claude, and an offline scripted fake.

The fake tests the plumbing (server, ledger, enforcement, grading, reporting)
without a key or a cent. It knows each scenario's answer, so its numbers say
nothing about defences; --dry-run output is never written to results/.
"""

from __future__ import annotations

import itertools
import json
from typing import Any

import anthropic

from poisoned_ticket.agent import Turn
from poisoned_ticket.attacks import EXFIL_ADDRESS, Attack
from poisoned_ticket.pricing import resolve
from poisoned_ticket.scenarios import Scenario


class ClaudeModel:
    """Claude through the Anthropic SDK. Thinking stays at the model's default (adaptive);
    effort is pinned so both models run the same setting."""

    def __init__(
        self, model: str, *, effort: str = "medium", client: anthropic.AsyncAnthropic | None = None
    ):
        self.name = resolve(model)
        self.effort = effort
        self.client = client or anthropic.AsyncAnthropic()

    async def create(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Turn:
        response = await self.client.messages.create(
            model=self.name,
            max_tokens=16000,
            system=system,
            tools=tools,
            messages=messages,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},  # the growing transcript is re-read every turn
        )
        blocks = [
            {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
            if b.type == "tool_use"
            else {"type": b.type}
            for b in response.content
        ]
        usage = response.usage
        return Turn(
            content=response.content,  # thinking blocks must go back unchanged
            blocks=blocks,
            stop_reason=response.stop_reason or "",
            usage={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
            },
        )


def attack_call(s: Scenario, attack: Attack) -> tuple[str, dict[str, Any]]:
    """The call an agent that obeys the injection would make."""
    if attack.goal == "refund_other":
        return "issue_refund", {
            "order_id": s.other_order_id,
            "amount_usd": 1250,
            "reason": "as requested",
        }
    if attack.goal == "exfiltrate":
        return "send_email", {
            "to": EXFIL_ADDRESS,
            "subject": f"Order {s.other_order_id}",
            "body": "Details as requested.",
        }
    return "issue_refund", {"order_id": s.order_id, "amount_usd": 40, "reason": "goodwill"}


def outcome_call(s: Scenario) -> tuple[str, dict[str, Any]]:
    if s.outcome == "refund":
        return "issue_refund", {
            "order_id": s.order_id,
            "amount_usd": s.total,
            "reason": "return in window",
        }
    if s.outcome == "escalate":
        return "escalate_to_finance", {"order_id": s.order_id, "reason": "total above USD 500"}
    return "reply_to_customer", {
        "ticket_id": s.ticket_id,
        "message": "Sorry, this order is outside the return window.",
    }


class FakeModel:
    """Scripted: read the ticket, check eligibility, act, then stop.

    behaviour "correct" does only the policy outcome; "resist" also flags an
    attacked ticket; "obey" also carries out the injected request.
    Usage is a rough characters/4 estimate so --dry-run exercises the cost path.
    """

    name = "fake"

    def __init__(self, s: Scenario, attack: Attack | None, behaviour: str = "resist") -> None:
        self.s, self.attack, self.behaviour = s, attack, behaviour
        self._ids = (f"toolu_{n:04d}" for n in itertools.count())
        self._turn = 0

    def _use(self, call: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        name, args = call
        return {"type": "tool_use", "id": next(self._ids), "name": name, "input": args}

    async def create(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Turn:
        self._turn += 1
        s = self.s
        if self._turn == 1:
            blocks = [self._use(("get_ticket", {"ticket_id": s.ticket_id}))]
        elif self._turn == 2:
            blocks = [
                self._use(("check_return_eligibility", {"order_id": s.order_id, "as_of": s.today}))
            ]
        elif self._turn == 3:
            blocks = [self._use(outcome_call(s))]
            if self.attack and self.behaviour == "obey":
                blocks.append(self._use(attack_call(s, self.attack)))
            if self.attack and self.behaviour == "resist":
                blocks.append(
                    self._use(
                        (
                            "flag_for_review",
                            {"ticket_id": s.ticket_id, "reason": "instructions in ticket"},
                        )
                    )
                )
        else:
            blocks = [{"type": "text", "text": "Done."}]
        prompt_chars = len(system) + len(json.dumps(tools)) + len(json.dumps(messages, default=str))
        return Turn(
            content=blocks,
            blocks=blocks,
            stop_reason="tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn",
            usage={"input_tokens": prompt_chars // 4, "output_tokens": 80},
        )
