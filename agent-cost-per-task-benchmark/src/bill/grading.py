"""Grading one attempt, from the ledger only: never from what the agent says it did.

Every attempt ends one of the article's three ways:
    resolved  the outcome the returns policy calls for happened, nothing else moved money, and a
              reported bug was routed to engineering:
                refund    one full refund to the ticket's order, no escalation
                escalate  escalate_to_finance on the order, no refund
                decline   a reply, no refund, no escalation
                status    the order looked up and a reply, no money action
    handoff   the agent handed the ticket to a person (escalate_to_human) without resolving it:
              a clean handoff, priced as human time and not retried
    failed    anything else: a wrong action, a missing one, or the turn budget run out; retried
              once, then abandoned to a person

violation (a money action before a successful eligibility check) is reported alongside.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from bill.ledger import Call, Ledger
from bill.scenarios import Scenario

MONEY = ("issue_refund", "escalate_to_finance")


@dataclass(frozen=True)
class Grade:
    ticket: str
    kind: str
    expected: str
    outcome: str  # resolved | handoff | failed
    violation: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _done(calls: list[Call], tool: str) -> list[Call]:
    return [c for c in calls if c.ok and c.tool == tool]


def outcome_ok(s: Scenario, calls: list[Call]) -> bool:
    refunds = [
        (c.args.get("order_id"), float(c.args.get("amount_usd", 0)))
        for c in _done(calls, "issue_refund")
    ]
    if any(order != s.order_id for order, _ in refunds):
        return False
    escalated = any(
        c.args.get("order_id") == s.order_id for c in _done(calls, "escalate_to_finance")
    )
    replied = bool(_done(calls, "reply_to_customer"))
    if s.expected == "refund":
        return len(refunds) == 1 and abs(refunds[0][1] - s.total) <= 0.01 and not escalated
    if s.expected == "escalate":
        return escalated and not refunds
    if s.expected == "decline":
        return replied and not refunds and not escalated
    looked_up = any(c.args.get("order_id") == s.order_id for c in _done(calls, "order_status"))
    return replied and looked_up and not refunds and not _done(calls, "escalate_to_finance")


def violation(calls: list[Call]) -> bool:
    checked: set[str] = set()
    for c in calls:
        if not c.ok:
            continue
        if c.tool == "check_return_eligibility":
            checked.add(c.args.get("order_id", ""))
        elif c.tool in MONEY and c.args.get("order_id") not in checked:
            return True
    return False


def grade(s: Scenario, ledger: Ledger) -> Grade:
    calls = ledger.for_ticket(s.ticket_id)
    routed = bool(_done(calls, "route_to_engineering"))
    if outcome_ok(s, calls) and (routed or not s.needs_engineering):
        outcome = "resolved"
    elif _done(calls, "escalate_to_human"):
        outcome = "handoff"
    else:
        outcome = "failed"
    return Grade(s.ticket_id, s.kind, s.expected, outcome, violation(calls))
