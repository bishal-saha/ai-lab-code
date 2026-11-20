"""The ledger: every tool call an episode makes, reads included, in order.

Reads are recorded so grading can check *order* (was the return window checked
before the refund?), not only what happened. A call that failed (a fault, a bad
argument) is recorded with `ok=False`; only ok actions count as something done.
Nothing here leaves the process.
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
}


@dataclass
class Call:
    seq: int
    tool: str
    args: dict[str, Any]
    ok: bool = True
    error: str = ""

    @property
    def is_action(self) -> bool:
        return self.tool in ACTION_TOOLS


@dataclass
class Ledger:
    calls: list[Call] = field(default_factory=list)

    def record(self, tool: str, args: dict[str, Any], *, ok: bool = True, error: str = "") -> None:
        self.calls.append(Call(len(self.calls), tool, dict(args), ok, error))

    def done(self, tool: str) -> list[Call]:
        """Successful calls of one tool, in order."""
        return [c for c in self.calls if c.tool == tool and c.ok]

    def actions(self) -> list[Call]:
        """Successful actions (no reads), in order."""
        return [c for c in self.calls if c.is_action and c.ok]

    def refunds(self) -> list[tuple[str, float]]:
        return [
            (c.args["order_id"], float(c.args["amount_usd"])) for c in self.done("issue_refund")
        ]

    def to_list(self) -> list[dict[str, Any]]:
        return [asdict(c) for c in self.calls]
