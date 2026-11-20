"""Offline tests: the vendored planning model, the ticket set, the conditions, the accounting, the dry run."""

from __future__ import annotations

import asyncio
from collections import Counter

import pytest
from bill.agent import Message, build_window
from bill.conditions import CONDITIONS, HAIKU, SONNET, condition
from bill.grading import grade
from bill.ledger import Ledger
from bill.models import FakeModel
from bill.planning_model import Run, article_output, attempt_cost, fit, predict
from bill.run import dry_run, execute, jobs, recompute
from bill.scenarios import build_scenarios, dataset
from bill.tools import Tools

# The article's printed output, copied from "AI Agent Cost per Task".
ARTICLE = """attempt $0.249  per ticket $1.049  agent resolves 90.0%  baseline (append-only cache)
attempt $0.486  per ticket $1.362  agent resolves 90.0%  no prompt caching
attempt $0.335  per ticket $1.162  agent resolves 90.0%  history cap 4
attempt $0.227  per ticket $1.020  agent resolves 90.0%  phase routing: first 6 steps cheap
attempt $0.289  per ticket $1.101  agent resolves 90.0%  per-step switching, 50% cheap
attempt $0.340  per ticket $1.168  agent resolves 90.0%  per-step switching, 50% cheap, no caching
attempt $0.386  per ticket $1.230  agent resolves 90.0%  18 steps instead of 12
attempt $0.369  per ticket $1.207  agent resolves 90.0%  tool fee $0.02 per call
attempt $0.249  per ticket $1.573  agent resolves 84.5%  resolve 65%, fail 30%
attempt $0.249  per ticket $0.784  agent resolves 93.0%  3 attempts instead of 2"""


@pytest.fixture(scope="module")
def data():
    return dataset()


@pytest.fixture(scope="module")
def scenarios(data):
    return build_scenarios(data)


def test_vendored_model_reproduces_the_article():
    assert article_output() == ARTICLE.splitlines()


def test_ticket_set(data, scenarios):
    orders = {o.id: o for o in data.dataset.orders}
    assert len(scenarios) == 24 and len({s.order_id for s in scenarios}) == 24
    assert Counter(s.kind for s in scenarios) == {
        "policy": 12,
        "multi": 4,
        "status": 4,
        "no_order": 4,
    }
    # experiment 7's flaw: no ticket over an order that was already returned
    assert not any(orders[s.order_id].status.value == "returned" for s in scenarios)
    for s in scenarios:
        status = orders[s.order_id].status.value
        assert status == "delivered" if s.kind != "status" else status in ("delivered", "shipped")


def test_conditions_route_models():
    assert [condition("cache").model_for(i) for i in range(4)] == [SONNET] * 4
    assert [condition("haiku").model_for(i) for i in range(4)] == [HAIKU] * 4
    assert [condition("phase").model_for(i) for i in range(4)] == [HAIKU, HAIKU, SONNET, SONNET]
    assert [condition("switch").model_for(i) for i in range(4)] == [HAIKU, SONNET, HAIKU, SONNET]
    assert not condition("no-cache").caching and condition("cache-break").timestamp
    assert condition("cap").history_cap == 2


def _ex(i: int, model: str) -> list[Message]:
    return [
        Message(
            "assistant",
            [
                {"type": "thinking", "thinking": "t", "signature": "s"},
                {"type": "tool_use", "id": f"tu{i}", "name": "order_status", "input": {}},
            ],
            model,
        ),
        Message(
            "user", [{"type": "tool_result", "tool_use_id": f"tu{i}", "content": f"RESULT {i}"}]
        ),
    ]


def test_window_cap_and_cross_model_thinking():
    exs = [_ex(i, SONNET if i % 2 else HAIKU) for i in range(4)]
    full = build_window("Please handle T-B001.", exs, condition("cache"), None, SONNET)
    assert len(full) == 9 and all(
        a["role"] != b["role"] for a, b in zip(full, full[1:], strict=False)
    )
    # Sonnet sees thinking only in its own messages
    assert sum("'thinking'" in str(m) for m in full if m["role"] == "assistant") == 2
    capped = build_window(
        "Please handle T-B001.", exs, condition("cap"), "looked up the order", SONNET
    )
    assert len(capped) == 5 and "looked up the order" in capped[0]["content"]
    assert "RESULT 0" not in str(capped) and "RESULT 3" in str(capped)


def test_grader_outcomes(data, scenarios):
    s = next(x for x in scenarios if x.expected == "refund" and not x.needs_engineering)
    for calls, want in (
        (
            [
                ("check_return_eligibility", {"order_id": s.order_id, "as_of": s.today}),
                ("issue_refund", {"order_id": s.order_id, "amount_usd": s.total, "reason": "r"}),
            ],
            "resolved",
        ),
        (
            [
                ("check_return_eligibility", {"order_id": s.order_id, "as_of": s.today}),
                ("escalate_to_human", {"ticket_id": s.ticket_id, "reason": "r"}),
            ],
            "handoff",
        ),
        ([("check_return_eligibility", {"order_id": s.order_id, "as_of": s.today})], "failed"),
    ):
        ledger = Ledger()
        ledger.current = s.ticket_id
        tools = Tools(data, scenarios, ledger)
        for name, args in calls:
            tools.call(name, args)
        assert grade(s, ledger).outcome == want


def test_fit_and_predict():
    rows = [
        {"calls": [{"window_tokens": 4000 + 1500 * i, "output_tokens": 120} for i in range(5)]}
        for _ in range(3)
    ]
    base = fit(rows)
    assert (
        base.steps,
        base.prefix_tokens,
        base.tokens_added_per_step,
        base.output_tokens_per_step,
    ) == (5, 4000, 1500, 120)
    assert base.tool_cost_per_call == 0
    p = {c: predict(c, base) for c in CONDITIONS}
    assert (
        p["no-cache"] > p["cache"]
        and p["cache-break"] > p["no-cache"]
        and p["haiku"] < p["phase"] < p["cache"]
    )
    assert attempt_cost(Run()) == pytest.approx(0.249, abs=0.0005)


def test_costs_recompute_from_usage(scenarios):
    rows, spent = asyncio.run(execute(jobs(4, 1), FakeModel(scenarios), 1e9, 1, None, quiet=True))
    assert len(rows) == 4 * len(CONDITIONS)
    assert all(abs(recompute(r) - r["cost_usd"]) < 1e-9 for r in rows)
    assert abs(sum(r["cost_usd"] for r in rows) - spent) < 1e-9


def test_dry_run():
    assert dry_run() == 0
