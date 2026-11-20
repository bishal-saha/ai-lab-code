"""The ticket set and its ground truth: computed from the data, and the server agrees."""

import re
from collections import Counter

from four_shapes.ledger import Ledger
from four_shapes.servers.support import FAULT_MESSAGE, build_server
from mcp import Client


def test_sixteen_tickets_in_four_kinds(scenarios):
    assert Counter(s.kind for s in scenarios) == {
        "policy": 8,
        "multi": 4,
        "status": 2,
        "no_order": 2,
    }
    assert Counter(s.outcome for s in scenarios if s.kind == "policy") == {
        "refund": 3,
        "escalate": 2,
        "decline": 3,
    }
    assert all(s.bug for s in scenarios if s.kind == "multi")
    assert len({s.ticket_id for s in scenarios}) == len({s.order_id for s in scenarios}) == 16


def test_no_order_tickets_hide_the_id_and_the_customer_has_one_order(scenarios, customers):
    for s in (s for s in scenarios if s.kind == "no_order"):
        assert s.order_id not in s.message and not s.order_in_ticket
        assert customers[s.customer_email.lower()] == [s.order_id]


async def test_eligibility_tool_agrees_with_ground_truth(api, scenarios, customers):
    for s in (s for s in scenarios if s.outcome != "status"):
        async with Client(build_server(api, s, Ledger(), customers)) as c:
            text = (
                (
                    await c.call_tool(
                        "check_return_eligibility", {"order_id": s.order_id, "as_of": s.today}
                    )
                )
                .content[0]
                .text
            )
        open_ = "Return window: OPEN" in text
        support = "Refund approver: Support" in text
        expected = "decline" if not open_ else ("refund" if support else "escalate")
        assert expected == s.outcome, (s.id, text)


async def test_status_tickets_cover_delivered_and_in_transit(api, scenarios, customers):
    texts = []
    for s in (s for s in scenarios if s.kind == "status"):
        async with Client(build_server(api, s, Ledger(), customers)) as c:
            texts.append(
                (await c.call_tool("order_status", {"order_id": s.order_id})).content[0].text
            )
    assert any("Delivered on" in t for t in texts) and any("not delivered yet" in t for t in texts)


async def test_faults_fail_only_the_first_eligibility_call(api, scenarios, customers):
    s = scenarios[0]
    ledger = Ledger()
    async with Client(build_server(api, s, ledger, customers, faults=True)) as c:
        first = await c.call_tool(
            "check_return_eligibility", {"order_id": s.order_id, "as_of": s.today}
        )
        second = await c.call_tool(
            "check_return_eligibility", {"order_id": s.order_id, "as_of": s.today}
        )
    assert first.is_error and FAULT_MESSAGE in first.content[0].text
    assert not second.is_error
    assert [(c.tool, c.ok) for c in ledger.calls] == [
        ("check_return_eligibility", False),
        ("check_return_eligibility", True),
    ]


async def test_customer_orders_lists_the_customers_orders(api, scenarios, customers):
    s = next(s for s in scenarios if s.kind == "no_order")
    async with Client(build_server(api, s, Ledger(), customers)) as c:
        text = (
            (await c.call_tool("customer_orders", {"email": s.customer_email.upper()}))
            .content[0]
            .text
        )
    assert re.findall(r"O-\d{7}", text) == [s.order_id]
