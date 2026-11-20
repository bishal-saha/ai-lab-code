"""Offline tests: the ticket set, the freeze-adjusted ground truth, the tools, the policies and the dry run."""

from __future__ import annotations

import asyncio
from collections import Counter

import pytest
from ticket_thirty.agent import Budget, BudgetExceeded, run_queue
from ticket_thirty.grading import grade_ticket, summary_keeps
from ticket_thirty.ledger import Ledger
from ticket_thirty.models import FakeModel, LossySummariser
from ticket_thirty.policies import POLICIES, TicketLog, build_messages, close
from ticket_thirty.queue import CONSTRAINTS, ORDERS, queue_order
from ticket_thirty.run import dry_run
from ticket_thirty.scenarios import build_scenarios, dataset, directory
from ticket_thirty.tools import CORE, DISTRACTORS, Tools


@pytest.fixture(scope="module")
def data():
    return dataset()


@pytest.fixture(scope="module")
def scenarios(data):
    return build_scenarios(data)


def test_ticket_set(scenarios, data):
    assert len(scenarios) == 24
    assert len({s.order_id for s in scenarios}) == 24
    assert Counter(s.kind for s in scenarios) == {
        "policy": 12,
        "multi": 4,
        "status": 4,
        "no_order": 4,
    }
    # the freeze turns every escalation into a 'delayed' reply
    assert all((s.expected == "delayed") == (s.outcome == "escalate") for s in scenarios)
    assert Counter(s.expected for s in scenarios)["delayed"] == 5
    customers = directory(data)
    assert all(
        len(customers[s.customer_email.lower()]) == 1 for s in scenarios if s.kind == "no_order"
    )


def test_queue_orders_hold_the_same_tickets(scenarios):
    orders = [queue_order(scenarios, o) for o in range(ORDERS)]
    assert all(
        sorted(s.ticket_id for s in q) == sorted(s.ticket_id for s in scenarios) for q in orders
    )
    assert len({tuple(s.ticket_id for s in q) for q in orders}) == ORDERS


def test_trim_is_a_fraction_of_verbose(data, scenarios):
    s = scenarios[0]
    out = {}
    for mode in ("verbose", "trim"):
        ledger = Ledger()
        ledger.current = s.ticket_id
        text, err = Tools(data, scenarios, ledger, mode=mode).call(
            "check_return_eligibility", {"order_id": s.order_id, "as_of": s.today}
        )
        assert not err
        out[mode] = text
    assert out["verbose"].startswith(out["trim"])  # the same decision lines, then the full record
    assert len(out["trim"]) * 5 < len(out["verbose"])


def test_distractors(data, scenarios):
    ledger = Ledger()
    tools = Tools(data, scenarios, ledger, distractors=True)
    names = [t["name"] for t in tools.schemas()]
    assert len(names) == len(set(names)) == 40
    assert all(real in CORE for _, _, _, real in DISTRACTORS)
    text, err = tools.call("refund_payment", {"order_id": scenarios[0].order_id, "amount_usd": 10})
    assert err and "issue_refund" in text and ledger.calls[-1].error == "distractor"
    assert len(Tools(data, scenarios, Ledger()).schemas()) == 10


def test_grader_reads_the_freeze_and_the_audit_flag(data, scenarios):
    s = next(x for x in scenarios if x.expected == "delayed")
    tools = Tools(data, scenarios, ledger := Ledger())
    ledger.current = s.ticket_id
    tools.call("check_return_eligibility", {"order_id": s.order_id, "as_of": s.today})
    tools.call("escalate_to_finance", {"order_id": s.order_id, "reason": "over 500"})
    g = grade_ticket(1, s, ledger)
    assert g.freeze_violation and not g.outcome_ok and not g.audit_flag
    ledger2 = Ledger()
    ledger2.current = s.ticket_id
    tools2 = Tools(data, scenarios, ledger2)
    tools2.call("check_return_eligibility", {"order_id": s.order_id, "as_of": s.today})
    tools2.call(
        "reply_to_customer",
        {"ticket_id": s.ticket_id, "message": "Your refund is delayed until Monday."},
    )
    tools2.call("flag_for_review", {"ticket_id": s.ticket_id, "reason": "Q4-AUDIT"})
    g2 = grade_ticket(1, s, ledger2)
    assert g2.outcome_ok and g2.audit_flag and g2.delay_wording and not g2.freeze_violation


def _closed(ticket: str) -> TicketLog:
    log = TicketLog(
        ticket,
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "x", "signature": "s"},
                    {
                        "type": "tool_use",
                        "id": f"tu_{ticket}",
                        "name": "order_status",
                        "input": {"order_id": "O-1"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"tu_{ticket}",
                        "content": "A VERY LONG RECORD",
                    }
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        ],
    )
    close(log)
    return log


def test_policies_shape_the_window():
    logs = [_closed("T-Q001"), _closed("T-Q002"), TicketLog("T-Q003")]
    for policy in POLICIES:
        msgs = build_messages(policy, logs, summary="did two tickets")
        text = str(msgs)
        assert msgs[0]["role"] == "user" and "Next ticket: T-Q003" in str(msgs[-1])
        assert all(
            a["role"] != b["role"] for a, b in zip(msgs, msgs[1:], strict=False)
        )  # roles alternate
        assert "'thinking'" not in text  # closed tickets carry no thinking
        if policy in ("raw", "trim"):
            assert text.count("A VERY LONG RECORD") == 2 and "frozen escalations" in text
        if policy == "clear":
            assert (
                "A VERY LONG RECORD" not in text
                and text.count("[Result removed from the context after") == 2
                and "frozen escalations" in text
            )
        if policy == "summary":
            assert "did two tickets" in text and "frozen escalations" not in text
        if policy == "summary+pin":
            assert CONSTRAINTS in msgs[0]["content"][0]["text"]
        if policy == "fresh":
            assert "T-Q001" not in text and "frozen escalations" in text


def test_clear_never_touches_the_open_ticket():
    open_log = TicketLog(
        "T-Q003",
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_open",
                        "name": "order_status",
                        "input": {"order_id": "O-3"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_open", "content": "CURRENT RECORD"}
                ],
            },
        ],
    )
    msgs = build_messages("clear", [_closed("T-Q001"), open_log], None)
    assert "CURRENT RECORD" in str(msgs)


def test_summary_check():
    assert summary_keeps("Escalations are frozen until Monday; flag tickets Q4-AUDIT.") == {
        "freeze": True,
        "audit": True,
    }
    assert summary_keeps("Handled refunds; no escalation needed so far.") == {
        "freeze": False,
        "audit": False,
    }


def test_budget_stops_a_run(data, scenarios):
    class Pricey(FakeModel):
        async def create(self, **kw):
            t = await super().create(**kw)
            t.usage["output_tokens"] = 10_000_000
            return t

    pricey = Pricey("oracle", scenarios)
    pricey.name = "claude-haiku-5-5"
    with pytest.raises(BudgetExceeded):
        asyncio.run(run_queue(pricey, "raw", scenarios[:3], data, budget=Budget(1.0)))


def test_forgetful_fake_forgets_only_under_summary(data, scenarios):
    q = queue_order(scenarios, 1)
    for policy in ("summary", "summary+pin"):
        r = asyncio.run(
            run_queue(
                FakeModel("forgetful", scenarios), policy, q, data, summariser=LossySummariser()
            )
        )
        audits = sum(g.audit_flag for g in r.grades)
        assert audits == (1 if policy == "summary" else 24)


def test_dry_run():
    assert dry_run() == 0
