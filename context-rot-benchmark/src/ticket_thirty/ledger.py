"""The ledger: every tool call of a run, reads included, tagged with the ticket it belongs to.

One run handles a queue of tickets in one window, so each call records the ticket the
agent was working on (set by the harness at each ticket boundary). Grading reads one
ticket's slice. Nothing here leaves the process.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

ACTION_TOOLS = {
    "issue_refund",
    "escalate_to_finance",
    "reply_to_customer",
    "route_to_engineering",
    "escalate_to_human",
    "flag_for_review",
}


@dataclass
class Call:
    seq: int
    ticket: str
    tool: str
    args: dict[str, Any]
    ok: bool = True
    error: str = ""


@dataclass
class Ledger:
    calls: list[Call] = field(default_factory=list)
    current: str = ""  # the ticket the agent is on, set by the harness

    def record(self, tool: str, args: dict[str, Any], *, ok: bool = True, error: str = "") -> None:
        self.calls.append(Call(len(self.calls), self.current, tool, dict(args), ok, error))

    def for_ticket(self, ticket: str) -> list[Call]:
        return [c for c in self.calls if c.ticket == ticket]

    def to_list(self) -> list[dict[str, Any]]:
        return [asdict(c) for c in self.calls]
