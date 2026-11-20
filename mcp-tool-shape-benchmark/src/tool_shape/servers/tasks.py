"""Server B, "tasks": a few tools shaped around the jobs a support agent actually does.

Same API underneath as the endpoints server. The difference is where the work
happens: these tools join records, apply Kestrel's written policies (return
windows, refund approval limits, "lost counts as late") and answer in short,
plain text, with errors that say what to do next. That moves business logic out
of the prompt and into code you can test, which is the trade the article argues
for. The cost is visible too: every rule here is code someone has to maintain
when the policy changes.
"""

from __future__ import annotations

from datetime import date, datetime

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from tool_shape.kestrel_api import ApiNotFound, KestrelAPI

# From returns-policy.md. Kept next to the tools that apply them.
RETURN_WINDOW_DAYS = {"store": 30, "trade": 14}
SUPPORT_REFUND_LIMIT_USD = 500
POLICY_DOCS = {
    "returns": "returns-policy",
    "shipping": "shipping-policy",
    "expenses": "expense-policy",
}


def _day(value: str | None) -> date | None:
    return datetime.fromisoformat(value).date() if value else None


def build_server(api: KestrelAPI) -> MCPServer:
    server = MCPServer(
        "kestrel-tasks",
        instructions="Customer-support tools for Kestrel Outfitters. Each tool answers one support question.",
    )

    def order_and_shipment(order_id: str) -> tuple[dict, dict | None]:
        try:
            order = api.get(f"/orders/{order_id}")
        except ApiNotFound:
            raise ToolError(
                f"No order {order_id!r}. Order ids look like O-0000123; check the id in the ticket or ask the customer."
            ) from None
        try:
            shipment = api.get(f"/orders/{order_id}/shipment")
        except ApiNotFound:
            shipment = None
        return order, shipment

    def describe(order: dict, shipment: dict | None) -> list[str]:
        customer = "trade customer" if order["channel"] == "trade" else "consumer"
        lines = [
            f"Order {order['order_id']} ({customer}), placed {order['placed_at'][:10]}, "
            f"status {order['status']}, total {order['total']:.2f} {order['currency']}."
        ]
        if shipment is None:
            lines.append(f"Not shipped yet. Promised date {order['promised_date']}.")
            return lines
        lines.append(
            f"Shipped {shipment['shipped_at'][:10]} with {shipment['carrier']} from {shipment['warehouse']}; "
            f"promised by {shipment['promised_by'][:10]}."
        )
        if shipment["status"] == "lost":
            lines.append(
                "Delivered: no. The parcel is lost, which counts as LATE under the shipping policy."
            )
        elif shipment["delivered_at"]:
            verdict = "LATE (after the promised date)" if shipment["late"] else "on time"
            lines.append(f"Delivered: yes, on {shipment['delivered_at'][:10]}, {verdict}.")
        else:
            lines.append("Delivered: no, still in transit.")
        return lines

    @server.tool()
    def order_status(order_id: str) -> str:
        """Everything about where an order is: order status and total, customer type,
        carrier, ship and promise dates, whether it was delivered and whether it was late
        (a lost parcel counts as late). Use this for any "where is my order" question."""
        order, shipment = order_and_shipment(order_id)
        return "\n".join(describe(order, shipment))

    @server.tool()
    def check_return_eligibility(order_id: str, as_of: str) -> str:
        """Applies the returns policy to an order for an unused-item return requested on
        as_of (YYYY-MM-DD): whether it is inside the return window (30 days from delivery
        for consumers, 14 for trade customers), how many days are left, and who must
        approve the refund (Support up to USD 500, Finance above)."""
        try:
            today = date.fromisoformat(as_of)
        except ValueError:
            raise ToolError(f"as_of must be a date like 2026-01-15, got {as_of!r}.") from None
        order, shipment = order_and_shipment(order_id)
        lines = describe(order, shipment)
        delivered = _day(shipment["delivered_at"]) if shipment else None
        if delivered is None:
            lines.append("Return window: not started; the order has not been delivered.")
            return "\n".join(lines)
        window = RETURN_WINDOW_DAYS[order["channel"]]
        used = (today - delivered).days
        if used <= window:
            lines.append(
                f"Return window: OPEN. Day {used} of {window}; {window - used} day(s) left."
            )
        else:
            lines.append(
                f"Return window: CLOSED. Day {used}, window was {window} days from delivery."
            )
        if order["currency"] != "USD":
            lines.append(
                f"Refund approver: the limit is USD {SUPPORT_REFUND_LIMIT_USD} and this order is in "
                f"{order['currency']}; convert before deciding between Support and Finance."
            )
        elif order["total"] <= SUPPORT_REFUND_LIMIT_USD:
            lines.append(
                f"Refund approver: Support (total is within USD {SUPPORT_REFUND_LIMIT_USD})."
            )
        else:
            lines.append(
                f"Refund approver: Finance (total is above USD {SUPPORT_REFUND_LIMIT_USD})."
            )
        lines.append("Damaged or wrong items follow different rules: see read_policy('returns').")
        return "\n".join(lines)

    @server.tool()
    def ticket_context(ticket_id: str) -> str:
        """A support ticket with what you need to answer it: subject, status, priority,
        the customer's first message, and the full status of the order it is about."""
        try:
            ticket = api.get(f"/tickets/{ticket_id}")
        except ApiNotFound:
            raise ToolError(f"No ticket {ticket_id!r}. Ticket ids look like T-000123.") from None
        first = ticket["messages"][0]["body"] if ticket["messages"] else "(no messages)"
        lines = [
            f"Ticket {ticket['ticket_id']}: {ticket['subject']!r}, {ticket['status']}, "
            f"{ticket['priority']} priority, via {ticket['channel']}.",
            f"Customer wrote: {first}",
        ]
        if ticket["order_id"]:
            order, shipment = order_and_shipment(ticket["order_id"])
            lines += ["About this order:", *describe(order, shipment)]
        else:
            lines.append("Not linked to an order.")
        return "\n".join(lines)

    @server.tool()
    def stock_check(sku: str) -> str:
        """Whether a product is in stock: units on hand per warehouse and in total."""
        try:
            stock = api.get(f"/stock/{sku}")
        except ApiNotFound:
            raise ToolError(f"No product {sku!r}. SKUs look like SKU-00042.") from None
        total = sum(w["on_hand"] for w in stock["warehouses"])
        per = ", ".join(f"{w['warehouse']}: {w['on_hand']}" for w in stock["warehouses"])
        state = "IN STOCK" if total else "OUT OF STOCK"
        return f"{stock['sku']} {stock['name']}: {state}, {total} unit(s) in total ({per})."

    @server.tool()
    def read_policy(topic: str) -> str:
        """The current policy document for a topic: returns, shipping or expenses.
        Superseded versions are never returned."""
        name = POLICY_DOCS.get(topic.strip().lower())
        if name is None:
            raise ToolError(f"Unknown topic {topic!r}. Use one of: {', '.join(POLICY_DOCS)}.")
        return api.get(f"/policies/{name}")["body"]

    return server
