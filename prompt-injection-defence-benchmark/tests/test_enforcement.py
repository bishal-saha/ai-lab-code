"""Server-side enforcement (D3/D4): refuses out-of-policy calls, records them, allows the rest."""

import pytest
from mcp import Client
from poisoned_ticket.attacks import EXFIL_ADDRESS
from poisoned_ticket.ledger import Ledger
from poisoned_ticket.servers.support import build_server


@pytest.fixture
def refund_scenario(scenarios):
    return next(s for s in scenarios if s.outcome == "refund")


@pytest.mark.parametrize(
    "call",
    [
        ("issue_refund", lambda s: {"order_id": s.other_order_id, "amount_usd": 10, "reason": "x"}),
        (
            "issue_refund",
            lambda s: {"order_id": s.order_id, "amount_usd": s.total + 1, "reason": "x"},
        ),
        ("send_email", lambda s: {"to": EXFIL_ADDRESS, "subject": "x", "body": "x"}),
    ],
    ids=["other-order", "above-total", "outside-recipient"],
)
async def test_enforced_server_refuses(api, refund_scenario, call):
    name, args = call
    s = refund_scenario
    for enforce, refused in ((True, True), (False, False)):
        ledger = Ledger()
        async with Client(build_server(api, s, ledger, enforce=enforce)) as c:
            result = await c.call_tool(name, args(s))
        assert bool(result.is_error) is refused
        assert ledger.actions[0].refused is refused
        if refused:
            assert "Refused:" in result.content[0].text


async def test_refund_above_limit_points_to_finance(api, scenarios):
    s = next(s for s in scenarios if s.outcome == "escalate")
    ledger = Ledger()
    async with Client(build_server(api, s, ledger, enforce=True)) as c:
        result = await c.call_tool(
            "issue_refund", {"order_id": s.order_id, "amount_usd": s.total, "reason": "x"}
        )
    assert result.is_error and "escalate_to_finance" in result.content[0].text


async def test_in_policy_calls_pass_enforcement(api, refund_scenario):
    s = refund_scenario
    ledger = Ledger()
    async with Client(build_server(api, s, ledger, enforce=True)) as c:
        await c.call_tool(
            "issue_refund", {"order_id": s.order_id, "amount_usd": s.total, "reason": "x"}
        )
        await c.call_tool(
            "issue_refund", {"order_id": s.order_id, "amount_usd": 40, "reason": "goodwill"}
        )
        await c.call_tool(
            "send_email", {"to": s.customer_email.upper(), "subject": "x", "body": "x"}
        )
    assert [a.refused for a in ledger.actions] == [False, False, False]
