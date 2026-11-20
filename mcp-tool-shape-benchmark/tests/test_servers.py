"""Both servers work, and the task-shaped tools agree with the independent ground truth.

The second part matters for fairness: if a task tool misapplied a policy, the
benchmark would be measuring a bug, not a tool design.
"""

import json
import re

from mcp import Client


async def call(server, name, args):
    async with Client(server) as client:
        result = await client.call_tool(name, args)
        return result.is_error, "\n".join(p.text for p in result.content)


async def tool_names(server):
    async with Client(server) as client:
        return [t.name for t in (await client.list_tools()).tools]


async def test_tool_surfaces(servers):
    assert await tool_names(servers["endpoints"]) == [
        "get_order", "get_order_shipment", "list_shipments", "get_stock",
        "get_ticket", "list_tickets", "list_policies", "get_policy",
    ]  # fmt: skip
    assert await tool_names(servers["tasks"]) == [
        "order_status", "check_return_eligibility", "ticket_context", "stock_check", "read_policy",
    ]  # fmt: skip


async def test_endpoint_tools_return_raw_api_json(servers, task_list):
    order_id = _ref(next(t for t in task_list if t.family == "delivery"))
    error, text = await call(servers["endpoints"], "get_order", {"order_id": order_id})
    assert not error
    assert json.loads(text)["order_id"] == order_id


async def test_endpoint_policies_include_the_outdated_one(servers):
    _, text = await call(servers["endpoints"], "list_policies", {})
    assert "shipping-policy-v2-OUTDATED" in json.loads(text)


async def test_errors_are_tool_errors_not_crashes(servers):
    error, text = await call(servers["endpoints"], "get_order", {"order_id": "O-NOPE"})
    assert error and "404" in text
    error, text = await call(servers["tasks"], "order_status", {"order_id": "O-NOPE"})
    assert error and "look like O-" in text
    error, text = await call(servers["tasks"], "read_policy", {"topic": "refunds"})
    assert error and "returns, shipping, expenses" in text


async def test_read_policy_never_serves_the_outdated_version(servers):
    _, text = await call(servers["tasks"], "read_policy", {"topic": "shipping"})
    assert "3 working days" in text and "v2" not in text


def _ref(task):
    return next(w.rstrip(".?,") for w in task.prompt.split() if w.startswith(("O-", "T-", "SKU-")))


def _delivery_facts(text):
    delivered = "Delivered: yes" in text
    late = "LATE" in text
    carrier = re.search(r"with (.+?) from", text)
    return delivered, late, carrier.group(1) if carrier else None


async def test_task_tools_match_ground_truth(servers, task_list):
    tasks_server = servers["tasks"]
    for task in task_list:
        want = task.expected
        if task.family == "delivery":
            _, text = await call(tasks_server, "order_status", {"order_id": _ref(task)})
            assert _delivery_facts(text) == (want["delivered"], want["late"], want["carrier"]), (
                task.id
            )
        elif task.family == "ticket":
            _, text = await call(tasks_server, "ticket_context", {"ticket_id": _ref(task)})
            assert want["order_id"] in text, task.id
            assert _delivery_facts(text)[:2] == (want["delivered"], want["late"]), task.id
        elif task.family == "returns":
            as_of = task.prompt[len("Today is ") : len("Today is ") + 10]
            _, text = await call(
                tasks_server, "check_return_eligibility", {"order_id": _ref(task), "as_of": as_of}
            )
            assert ("Return window: OPEN" in text) == want["eligible"], task.id
            assert f"Refund approver: {want['approver'].capitalize()}" in text, task.id
        elif task.family == "stock":
            _, text = await call(tasks_server, "stock_check", {"sku": _ref(task)})
            assert f"{want['total']} unit(s) in total" in text, task.id
            assert ("IN STOCK" in text) == want["in_stock"], task.id
