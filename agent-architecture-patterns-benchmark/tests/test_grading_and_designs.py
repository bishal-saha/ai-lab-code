"""The grader, plan validation, and every design end to end on the scripted fakes."""

import pytest
from four_shapes.architectures.planner import Step, validate
from four_shapes.grading import grade
from four_shapes.harness import ARCHITECTURES, run_episode
from four_shapes.ledger import Ledger
from four_shapes.models import FakeModel, outcome_call

REGISTERED = {
    "customer_orders",
    "order_status",
    "check_return_eligibility",
    "issue_refund",
    "escalate_to_finance",
    "reply_to_customer",
    "route_to_engineering",
    "escalate_to_human",
}


def _ledger(*calls):
    ledger = Ledger()
    for tool, args in calls:
        ledger.record(tool, args)
    return ledger


def test_refund_before_the_check_is_a_violation_but_after_is_not(scenarios):
    s = next(s for s in scenarios if s.outcome == "refund" and s.kind == "policy")
    check = ("check_return_eligibility", {"order_id": s.order_id, "as_of": s.today})
    assert grade(s, _ledger(check, outcome_call(s))) == grade(s, _ledger(check, outcome_call(s)))
    assert not grade(s, _ledger(check, outcome_call(s))).violation
    late = grade(s, _ledger(outcome_call(s), check))
    assert late.violation and late.success  # right outcome, wrong order: counted separately


def test_a_failed_check_does_not_count_as_checked(scenarios):
    s = next(s for s in scenarios if s.outcome == "refund")
    ledger = Ledger()
    ledger.record("check_return_eligibility", {"order_id": s.order_id}, ok=False, error="503")
    ledger.record(*outcome_call(s))
    assert grade(s, ledger).violation


def test_multi_intent_needs_the_bug_routed_and_status_needs_a_lookup(scenarios):
    m = next(s for s in scenarios if s.kind == "multi")
    check = ("check_return_eligibility", {"order_id": m.order_id})
    without = grade(m, _ledger(check, outcome_call(m)))
    assert without.outcome_ok and without.missed_intent and not without.success
    assert grade(
        m, _ledger(check, outcome_call(m), ("route_to_engineering", {"summary": "x"}))
    ).success
    st = next(s for s in scenarios if s.kind == "status")
    assert not grade(st, _ledger(("reply_to_customer", {"message": "hi"}))).success
    assert grade(
        st,
        _ledger(
            ("order_status", {"order_id": st.order_id}), ("reply_to_customer", {"message": "hi"})
        ),
    ).success


def test_money_to_another_order_fails_and_duplicate_replies_are_counted(scenarios):
    s = next(s for s in scenarios if s.outcome == "refund")
    other = next(x for x in scenarios if x.order_id != s.order_id)
    assert not grade(
        s, _ledger(outcome_call(s), ("issue_refund", {"order_id": other.order_id, "amount_usd": 5}))
    ).outcome_ok
    replies = _ledger(
        ("reply_to_customer", {"message": "a"}), ("reply_to_customer", {"message": "b"})
    )
    assert grade(s, replies).duplicate_replies == 1


@pytest.mark.parametrize(
    "steps, problem",
    [
        ([], "no steps"),
        ([Step("lookup_contract", "x")], "not a registered tool"),
        (
            [Step("issue_refund", "x"), Step("check_return_eligibility", "x")],
            "before check_return_eligibility",
        ),
        ([Step("escalate_to_finance", "x")], "before check_return_eligibility"),
    ],
)
def test_plan_validation_rejects_each_bad_shape(steps, problem):
    assert any(problem in p for p in validate(steps, REGISTERED))


def test_plan_validation_accepts_a_checked_plan_and_a_replan_after_a_check():
    assert (
        validate(
            [Step("check_return_eligibility", "x"), Step("issue_refund", "if open")], REGISTERED
        )
        == []
    )
    assert validate([Step("issue_refund", "x")], REGISTERED, checked_already=True) == []


@pytest.mark.parametrize("arch", list(ARCHITECTURES))
@pytest.mark.parametrize("condition", ["normal", "faults"])
async def test_oracle_resolves_every_ticket(api, scenarios, customers, arch, condition):
    for s in scenarios:
        ep = await run_episode(s, arch, condition, api, customers, FakeModel(s, "oracle"))
        assert ep.stop == "done" and ep.outcome_ok and not ep.violation, (arch, s.id, ep.error)
        if arch == "graph" and s.kind == "multi":
            assert ep.missed_intent  # one intent per ticket, by design
        else:
            assert ep.success, (arch, s.id)


async def test_sloppy_failures_are_caught(api, scenarios, customers):
    refund = next(s for s in scenarios if s.outcome == "refund" and s.kind == "policy")
    multi = next(s for s in scenarios if s.kind == "multi")
    for arch in ("loop", "multi"):
        assert (
            await run_episode(refund, arch, "normal", api, customers, FakeModel(refund, "sloppy"))
        ).violation
        assert (
            await run_episode(multi, arch, "normal", api, customers, FakeModel(multi, "sloppy"))
        ).missed_intent
    ep = await run_episode(refund, "planner", "normal", api, customers, FakeModel(refund, "sloppy"))
    assert ep.invalid_plans == 1 and ep.replans == 1 and ep.success
    assert not (
        await run_episode(refund, "graph", "normal", api, customers, FakeModel(refund, "sloppy"))
    ).violation


async def test_multi_agent_drops_a_handoff_order_that_isnt_the_customers(api, scenarios, customers):
    s = next(s for s in scenarios if s.kind == "policy")
    other = next(x for x in scenarios if x.customer_email != s.customer_email)

    class WrongOrder(FakeModel):
        def _json(self, role, meta):
            out = super()._json(role, meta)
            return {**out, "order_id": other.order_id} if role == "triage" else out

    ep = await run_episode(s, "multi", "normal", api, customers, WrongOrder(s, "oracle"))
    assert ep.handoff_invalid == 1 and ep.success  # dropped in code; billing found the right order
