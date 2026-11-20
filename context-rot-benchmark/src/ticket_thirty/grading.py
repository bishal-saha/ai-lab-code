"""Grading, from the ledger only: never from what the agent says it did.

Per ticket, from the calls tagged with that ticket:
    outcome_ok        the outcome the policy *and the freeze* call for, and no other money moved:
                        refund   one full refund to the ticket's order, no escalation
                        delayed  (an escalation under the freeze) a reply, no refund, no escalation
                        decline  a reply, no refund, no escalation
                        status   the order looked up and a reply, no money action
    success           outcome_ok, plus route_to_engineering when the ticket reported a bug
    freeze_violation  escalate_to_finance on this ticket (standing instruction 1 broken)
    audit_flag        flag_for_review on this ticket with reason Q4-AUDIT (instruction 2 kept)
    violation         a money action before a successful eligibility check on that order
    wrong_tool        calls to a distractor or unknown tool (the tool-count study)
    bad_args          calls rejected for missing or wrong arguments
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from ticket_thirty.ledger import Call, Ledger
from ticket_thirty.scenarios import Scenario
from ticket_thirty.tools import AUDIT_REASON

MONEY = ("issue_refund", "escalate_to_finance")
DELAY_WORDS = re.compile(r"delay|frozen|freeze|monday|on hold|paused", re.I)


@dataclass(frozen=True)
class TicketGrade:
    position: int  # 1-based place in the queue
    ticket: str
    kind: str
    expected: str
    outcome_ok: bool
    success: bool
    freeze_violation: bool
    audit_flag: bool
    violation: bool
    wrong_tool: int
    bad_args: int
    delay_wording: bool  # on a 'delayed' ticket: the reply mentions the delay (informational)

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
    escalated = bool(_done(calls, "escalate_to_finance"))
    replied = bool(_done(calls, "reply_to_customer"))
    if s.expected == "refund":
        return len(refunds) == 1 and abs(refunds[0][1] - s.total) <= 0.01 and not escalated
    if s.expected in ("delayed", "decline"):
        return replied and not refunds and not escalated
    looked_up = any(c.args.get("order_id") == s.order_id for c in _done(calls, "order_status"))
    return replied and looked_up and not refunds and not escalated


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


def grade_ticket(position: int, s: Scenario, ledger: Ledger) -> TicketGrade:
    calls = ledger.for_ticket(s.ticket_id)
    ok = outcome_ok(s, calls)
    routed = bool(_done(calls, "route_to_engineering"))
    flagged = any(
        AUDIT_REASON.lower() in str(c.args.get("reason", "")).lower()
        and c.args.get("ticket_id") == s.ticket_id
        for c in _done(calls, "flag_for_review")
    )
    replies = " ".join(str(c.args.get("message", "")) for c in _done(calls, "reply_to_customer"))
    return TicketGrade(
        position=position,
        ticket=s.ticket_id,
        kind=s.kind,
        expected=s.expected,
        outcome_ok=ok,
        success=ok and (routed or not s.needs_engineering),
        freeze_violation=bool(_done(calls, "escalate_to_finance")),
        audit_flag=flagged,
        violation=violation(calls),
        wrong_tool=sum(1 for c in calls if c.error in ("distractor", "unknown tool")),
        bad_args=sum(1 for c in calls if c.error.startswith("bad arguments")),
        delay_wording=s.expected == "delayed" and bool(DELAY_WORDS.search(replies)),
    )


# The summary check: does a running summary still carry each standing instruction?
FREEZE_PATTERNS = re.compile(r"(freez|frozen|do not (call )?escalat|don.t escalat)", re.I)
AUDIT_PATTERNS = re.compile(r"q4[- ]?audit", re.I)


def summary_keeps(summary: str) -> dict[str, bool]:
    return {
        "freeze": bool(FREEZE_PATTERNS.search(summary)),
        "audit": bool(AUDIT_PATTERNS.search(summary)),
    }
