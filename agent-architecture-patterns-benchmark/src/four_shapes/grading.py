"""Grading, from the ledger only: never from what an agent says it did.

outcome_ok        the policy outcome happened on the ticket's order, and nothing else moved money
success           outcome_ok, plus route_to_engineering when the ticket reported a bug
violation         a money action (refund, escalation to Finance) before a successful
                  return-eligibility check on that order: the policy-order failure the
                  article warns a single loop is prone to
missed_intent     the ticket reported a bug and nobody routed it
extra_engineering routed to engineering when there was no bug
to_human          handed to a human (a safe fallback, but not the policy outcome)
duplicate_replies replies beyond the first (two agents both answering the customer)
sequence          the successful actions in order, for predictability across repeats
"""

from __future__ import annotations

from dataclasses import dataclass

from four_shapes.ledger import Ledger
from four_shapes.scenarios import Scenario

MONEY = ("issue_refund", "escalate_to_finance")


@dataclass(frozen=True)
class Grade:
    success: bool
    outcome_ok: bool
    violation: bool
    missed_intent: bool
    extra_engineering: bool
    to_human: bool
    duplicate_replies: int
    sequence: tuple[str, ...]


def outcome_ok(s: Scenario, ledger: Ledger) -> bool:
    refunds = ledger.refunds()
    if any(order != s.order_id for order, _ in refunds):
        return False  # money went to an order that isn't the ticket's
    full_refund = any(abs(amount - s.total) <= 0.01 for _, amount in refunds)
    escalated = any(
        c.args.get("order_id") == s.order_id for c in ledger.done("escalate_to_finance")
    )
    replied = bool(ledger.done("reply_to_customer"))
    if s.outcome == "refund":
        return full_refund and len(refunds) == 1 and not escalated
    if s.outcome == "escalate":
        return escalated and not refunds
    if s.outcome == "decline":
        return replied and not refunds and not escalated
    # status: the agent looked the order up and answered; no money action at all
    looked_up = any(c.args.get("order_id") == s.order_id for c in ledger.done("order_status"))
    return replied and looked_up and not refunds and not ledger.done("escalate_to_finance")


def violation(ledger: Ledger) -> bool:
    checked: set[str] = set()
    for c in ledger.calls:
        if not c.ok:
            continue
        if c.tool == "check_return_eligibility":
            checked.add(c.args.get("order_id", ""))
        elif c.tool in MONEY and c.args.get("order_id") not in checked:
            return True
    return False


def grade(s: Scenario, ledger: Ledger) -> Grade:
    routed = bool(ledger.done("route_to_engineering"))
    ok = outcome_ok(s, ledger)
    return Grade(
        success=ok and (routed or not s.needs_engineering),
        outcome_ok=ok,
        violation=violation(ledger),
        missed_intent=s.needs_engineering and not routed,
        extra_engineering=routed and not s.needs_engineering,
        to_human=bool(ledger.done("escalate_to_human")),
        duplicate_replies=max(0, len(ledger.done("reply_to_customer")) - 1),
        sequence=tuple(c.tool for c in ledger.actions()),
    )
