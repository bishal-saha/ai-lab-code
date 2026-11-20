"""The support MCP server every architecture uses, built fresh for every episode.

Reads go to the Kestrel mock API (orders, shipments) and to a customer directory
(email -> order ids). The episode's own ticket (T-A0nn) is an overlay holding the
scenario's customer message. Every call, reads included, is written to the
episode's ledger in order; actions go nowhere else.

The architectures differ only in who decides the next call: they all see these
same tools (the multi-agent design hands each agent a subset).

With `faults=True` the first check_return_eligibility call of the episode fails
with a retryable error, so grading can see how each design handles a failure.
"""

from __future__ import annotations

from datetime import date, datetime

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from four_shapes.kestrel_api import ApiNotFound, KestrelAPI
from four_shapes.ledger import Ledger
from four_shapes.scenarios import REFUND_LIMIT_USD, Scenario

# From returns-policy.md, keyed by the order's channel.
RETURN_WINDOW_DAYS = {"store": 30, "trade": 14}
FAULT_MESSAGE = (
    "Service temporarily unavailable (503). The returns service is restarting; retry the same call."
)


def _day(value: str | None) -> date | None:
    return datetime.fromisoformat(value).date() if value else None


def render_ticket(s: Scenario) -> str:
    lines = [
        f"Ticket {s.ticket_id}: open, normal priority, via web form.",
        "Subject: " + ("Where is my order?" if s.kind == "status" else "Return and refund request"),
        f"Customer: {s.customer_name} <{s.customer_email}> ({s.customer_type})",
    ]
    if s.order_in_ticket:
        lines.append(f"Order: {s.order_id}")
    return "\n".join([*lines, "", f"Customer wrote on {s.today}:", s.message])


def build_server(
    api: KestrelAPI,
    s: Scenario,
    ledger: Ledger,
    customers: dict[str, list[str]],
    *,
    faults: bool = False,
) -> MCPServer:
    server = MCPServer(
        "kestrel-support",
        instructions="Support tools for Kestrel Outfitters: read tickets and orders, apply the returns policy, act.",
    )
    state = {"eligibility_calls": 0}

    def fail(tool: str, args: dict, message: str) -> None:
        ledger.record(tool, args, ok=False, error=message)
        raise ToolError(message)

    def order_and_shipment(tool: str, args: dict, order_id: str) -> tuple[dict, dict | None]:
        try:
            order = api.get(f"/orders/{order_id}")
        except ApiNotFound:
            fail(
                tool,
                args,
                f"No order {order_id!r}. Order ids look like O-0000123; check the ticket or customer_orders.",
            )
        try:
            shipment = api.get(f"/orders/{order_id}/shipment")
        except ApiNotFound:
            shipment = None
        return order, shipment

    def describe(order: dict, shipment: dict | None) -> list[str]:
        customer = "trade customer" if order["channel"] == "trade" else "consumer"
        lines = [
            f"Order {order['order_id']} ({customer}, {order['customer_id']}), placed {order['placed_at'][:10]}, "
            f"status {order['status']}, total {order['total']:.2f} {order['currency']}."
        ]
        if shipment is None:
            lines.append("Not shipped yet.")
        elif shipment["delivered_at"]:
            lines.append(f"Delivered on {shipment['delivered_at'][:10]} by {shipment['carrier']}.")
        else:
            lines.append(
                f"Shipped {shipment['shipped_at'][:10]} with {shipment['carrier']}; not delivered yet, "
                f"promised by {shipment['promised_by'][:10]}."
            )
        return lines

    @server.tool()
    def get_ticket(ticket_id: str) -> str:
        """Read a support ticket: status, customer, the order it is about (when given) and the customer's message."""
        args = {"ticket_id": ticket_id}
        if ticket_id != s.ticket_id:
            fail("get_ticket", args, f"No ticket {ticket_id!r}. You are handling {s.ticket_id}.")
        ledger.record("get_ticket", args)
        return render_ticket(s)

    @server.tool()
    def customer_orders(email: str) -> str:
        """List a customer's orders (id, date, total, status) by their email address."""
        args = {"email": email}
        ids = customers.get(email.strip().lower())
        if not ids:
            fail("customer_orders", args, f"No customer with email {email!r}.")
        ledger.record("customer_orders", args)
        rows = []
        for order_id in ids:
            o = api.get(f"/orders/{order_id}")
            rows.append(
                f"{o['order_id']}: placed {o['placed_at'][:10]}, total {o['total']:.2f} {o['currency']}, status {o['status']}"
            )
        return f"{len(rows)} order(s) for {email}:\n" + "\n".join(rows)

    @server.tool()
    def order_status(order_id: str) -> str:
        """An order's status, customer, total and delivery."""
        args = {"order_id": order_id}
        lines = describe(*order_and_shipment("order_status", args, order_id))
        ledger.record("order_status", args)
        return "\n".join(lines)

    @server.tool()
    def check_return_eligibility(order_id: str, as_of: str) -> str:
        """Apply the returns policy to an unused-item return requested on as_of (YYYY-MM-DD):
        whether it is inside the return window (30 days from delivery for consumers, 14 for
        trade customers) and who may approve the refund (Support up to USD 500, Finance above)."""
        args = {"order_id": order_id, "as_of": as_of}
        state["eligibility_calls"] += 1
        if faults and state["eligibility_calls"] == 1:
            fail("check_return_eligibility", args, FAULT_MESSAGE)
        try:
            today = date.fromisoformat(as_of)
        except ValueError:
            fail(
                "check_return_eligibility",
                args,
                f"as_of must be a date like 2026-01-15, got {as_of!r}.",
            )
        order, shipment = order_and_shipment("check_return_eligibility", args, order_id)
        lines = describe(order, shipment)
        delivered = _day(shipment["delivered_at"]) if shipment else None
        if delivered is None:
            lines.append("Return window: not started; the order has not been delivered.")
        else:
            window = RETURN_WINDOW_DAYS[order["channel"]]
            used = (today - delivered).days
            lines.append(
                f"Return window: OPEN. Day {used} of {window}."
                if used <= window
                else f"Return window: CLOSED. Day {used}; the window was {window} days from delivery."
            )
            approver = "Support" if order["total"] <= REFUND_LIMIT_USD else "Finance"
            lines.append(
                f"Refund approver: {approver} (Support may refund up to USD {REFUND_LIMIT_USD})."
            )
        ledger.record("check_return_eligibility", args)
        return "\n".join(lines)

    @server.tool()
    def issue_refund(order_id: str, amount_usd: float, reason: str) -> str:
        """Refund an amount in USD to an order's original payment method."""
        args = {"order_id": order_id, "amount_usd": amount_usd, "reason": reason}
        order_and_shipment("issue_refund", args, order_id)
        if amount_usd <= 0:
            fail("issue_refund", args, "amount_usd must be positive.")
        ledger.record("issue_refund", args)
        return f"Refund of USD {amount_usd:.2f} issued to order {order_id}."

    @server.tool()
    def escalate_to_finance(order_id: str, reason: str) -> str:
        """Hand a refund decision to the Finance team (refunds above USD 500)."""
        args = {"order_id": order_id, "reason": reason}
        order_and_shipment("escalate_to_finance", args, order_id)
        ledger.record("escalate_to_finance", args)
        return f"Escalated order {order_id} to Finance."

    @server.tool()
    def reply_to_customer(ticket_id: str, message: str) -> str:
        """Send a reply on a ticket to its customer."""
        ledger.record("reply_to_customer", {"ticket_id": ticket_id, "message": message})
        return f"Reply sent on {ticket_id}."

    @server.tool()
    def route_to_engineering(ticket_id: str, summary: str) -> str:
        """Report a website or app bug the customer described to the engineering team."""
        ledger.record("route_to_engineering", {"ticket_id": ticket_id, "summary": summary})
        return f"Bug report filed with engineering for {ticket_id}."

    @server.tool()
    def escalate_to_human(ticket_id: str, reason: str) -> str:
        """Hand the ticket to a human support agent when it can't be resolved with these tools."""
        ledger.record("escalate_to_human", {"ticket_id": ticket_id, "reason": reason})
        return f"Ticket {ticket_id} handed to a human agent."

    return server
