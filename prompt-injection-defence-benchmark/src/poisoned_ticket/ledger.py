"""The action ledger: every refund, escalation, reply, email and flag the agent attempts.

Nothing an agent does here leaves the process. Refused calls (server-side
enforcement) are recorded too, with `refused=True`, so the report can show what
the policy layer stopped; only accepted actions count as something that happened.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Action:
    tool: str
    args: dict[str, Any]
    refused: bool = False
    reason: str = ""


@dataclass
class Ledger:
    actions: list[Action] = field(default_factory=list)

    def record(
        self, tool: str, args: dict[str, Any], *, refused: bool = False, reason: str = ""
    ) -> None:
        self.actions.append(Action(tool, dict(args), refused, reason))

    def done(self, tool: str) -> list[Action]:
        """Accepted actions of one kind."""
        return [a for a in self.actions if a.tool == tool and not a.refused]

    def refunds(self) -> list[tuple[str, float]]:
        return [
            (a.args["order_id"], float(a.args["amount_usd"])) for a in self.done("issue_refund")
        ]

    def to_list(self) -> list[dict[str, Any]]:
        return [asdict(a) for a in self.actions]
