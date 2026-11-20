"""The support tools, called in-process, returning the full record an untrimmed API would.

Lookups return the order with line items and product details, the customer with their order and
contact history, and the shipment with its tracking events: about 1,060 tokens each. This is what
makes history grow with every step, the cost driver the article models.

Policy decisions (window open or closed, who approves) are computed in code and stated in every
lookup, so no condition makes the reasoning harder; the conditions change only what a step costs.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from typing import Any

from kestrel.services import KestrelData

from bill.ledger import Ledger
from bill.scenarios import REFUND_LIMIT_USD, Scenario

RETURN_WINDOW_DAYS = {"consumer": 30, "trade": 14}


def render_ticket(s: Scenario) -> str:
    lines = [
        f"Ticket {s.ticket_id}: open, normal priority, via web form.",
        "Subject: " + ("Where is my order?" if s.kind == "status" else "Return and refund request"),
        f"Customer: {s.customer_name} <{s.customer_email}> ({s.customer_type})",
    ]
    if s.order_in_ticket:
        lines.append(f"Order: {s.order_id}")
    return "\n".join([*lines, "", f"Customer wrote on {s.today}:", s.message])


def _schema(props: dict[str, str]) -> dict:
    return {
        "type": "object",
        "properties": {
            k: {"type": "number", "description": "Amount in USD."}
            if v == "number"
            else {"type": "string", "description": v}
            for k, v in props.items()
        },
        "required": list(props),
        "additionalProperties": False,
    }


TOOLS = {
    "get_ticket": (
        "Read a support ticket: customer, the order it is about (when given) and the customer's message.",
        {"ticket_id": "Ticket id, e.g. T-B001."},
    ),
    "customer_orders": (
        "List a customer's orders (id, date, total, status) by their email address.",
        {"email": "The customer's email address."},
    ),
    "order_status": (
        "An order's status, customer, total and delivery.",
        {"order_id": "Order id, e.g. O-0000123."},
    ),
    "check_return_eligibility": (
        "Apply the returns policy to an unused-item return requested on as_of (YYYY-MM-DD): whether it is inside the "
        "return window (30 days from delivery for consumers, 14 for trade customers) and who may approve the refund "
        "(Support up to USD 500, Finance above).",
        {"order_id": "Order id.", "as_of": "The date the customer asked, YYYY-MM-DD."},
    ),
    "issue_refund": (
        "Refund an amount in USD to an order's original payment method.",
        {"order_id": "Order id.", "amount_usd": "number", "reason": "Why."},
    ),
    "escalate_to_finance": (
        "Hand a refund decision to the Finance team (refunds above USD 500).",
        {"order_id": "Order id.", "reason": "Why."},
    ),
    "reply_to_customer": (
        "Send a reply on a ticket to its customer.",
        {"ticket_id": "Ticket id.", "message": "The reply."},
    ),
    "route_to_engineering": (
        "Report a website or app bug the customer described to the engineering team.",
        {"ticket_id": "Ticket id.", "summary": "The bug."},
    ),
    "escalate_to_human": (
        "Hand the ticket to a human support agent when it can't be resolved with these tools.",
        {"ticket_id": "Ticket id.", "reason": "Why."},
    ),
}


class ToolError(Exception):
    pass


class Tools:
    def __init__(self, data: KestrelData, scenarios: list[Scenario], ledger: Ledger) -> None:
        self.d, self.ledger = data, ledger
        self.by_ticket = {s.ticket_id: s for s in scenarios}
        self.orders = {o.id: o for o in data.dataset.orders}
        self.by_email: dict[str, list[str]] = {}
        for o in data.dataset.orders:
            self.by_email.setdefault(data.customers[o.customer_id].email.lower(), []).append(o.id)

    def schemas(self) -> list[dict]:
        return [
            {"name": n, "description": desc, "input_schema": _schema(p)}
            for n, (desc, p) in TOOLS.items()
        ]

    def call(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Run one tool call; returns (text, is_error). Every call lands in the ledger."""
        fn = getattr(self, f"_t_{name}", None)
        if name not in TOOLS or fn is None:
            self.ledger.record(name, args, ok=False, error="unknown tool")
            return f"Unknown tool {name!r}.", True
        try:
            text = fn(**args)
        except ToolError as err:
            self.ledger.record(name, args, ok=False, error=str(err))
            return str(err), True
        except TypeError as err:  # wrong or missing arguments
            self.ledger.record(name, args, ok=False, error=f"bad arguments: {err}")
            return f"Bad arguments for {name}: {err}", True
        self.ledger.record(name, args)
        return text, False

    # -- helpers -----------------------------------------------------------------------------
    def _order(self, order_id: str):
        o = self.orders.get(str(order_id).strip())
        if o is None:
            raise ToolError(
                f"No order {order_id!r}. Order ids look like O-0000123; check the ticket or customer_orders."
            )
        return o

    def _total(self, o) -> float:
        return sum(line.quantity * line.unit_price_cents for line in o.lines) / 100

    def _summary(self, o) -> list[str]:
        c = self.d.customers[o.customer_id]
        sh = self.d.shipments_by_order.get(o.id)
        kind = "trade customer" if c.type.value == "trade" else "consumer"
        lines = [
            f"Order {o.id} ({kind}, {c.id}), placed {o.placed_at.date()}, status {o.status.value}, total {self._total(o):.2f} {o.currency}."
        ]
        if sh is None:
            lines.append("Not shipped yet.")
        elif sh.delivered_at:
            lines.append(f"Delivered on {sh.delivered_at.date()} by {sh.carrier}.")
        else:
            lines.append(
                f"Shipped {sh.shipped_at.date()} with {sh.carrier}; not delivered yet, promised by {sh.promised_by.date()}."
            )
        return lines

    def _full_record(self, o) -> str:
        """Everything an untrimmed support API knows about the order."""
        c = self.d.customers[o.customer_id]
        sh = self.d.shipments_by_order.get(o.id)
        rng = random.Random(o.id)
        street = f"{rng.randint(10, 9999)} {rng.choice(['Alder', 'Birch', 'Cedar', 'Harbour', 'Juniper', 'Maple'])} {rng.choice(['Street', 'Road', 'Lane', 'Avenue'])}"
        lines = []
        for i, line in enumerate(o.lines, 1):
            p = self.d.products[line.sku]
            lines.append(
                {
                    "line": i,
                    "sku": line.sku,
                    "name": p.name,
                    "category": p.category,
                    "quantity": line.quantity,
                    "unit_price": round(line.unit_price_cents / 100, 2),
                    "line_total": round(line.quantity * line.unit_price_cents / 100, 2),
                    "weight_kg": p.weight_kg,
                    "tax_code": rng.choice(["P0000000", "PC040100", "PC040500"]),
                    "returnable": True,
                    "warehouse_bin": f"{rng.choice('ABCDEFG')}-{rng.randint(1, 40):02d}-{rng.randint(1, 9)}",
                }
            )
        others = [x for x in self.by_email.get(c.email.lower(), []) if x != o.id][:6]
        topics = [
            "sizing question",
            "delivery address change",
            "promo code not applied",
            "product care advice",
            "back-in-stock alert",
            "invoice copy request",
            "loyalty points balance",
            "gift wrap request",
        ]
        record: dict[str, Any] = {
            "order": {
                "order_id": o.id,
                "status": o.status.value,
                "placed_at": o.placed_at.isoformat(),
                "currency": o.currency,
                "channel": "trade" if c.type.value == "trade" else "store",
                "subtotal": self._total(o),
                "tax": 0.0,
                "shipping": 0.0,
                "total": self._total(o),
                "payment": {
                    "method": rng.choice(["visa", "mastercard", "amex", "paypal"]),
                    "last4": f"{rng.randint(0, 9999):04d}",
                    "captured_at": o.placed_at.isoformat(),
                    "risk_score": round(rng.random() * 0.2, 3),
                },
                "lines": lines,
                "notes": [],
                "gift": False,
                "source": rng.choice(["web", "app", "phone"]),
            },
            "customer": {
                "customer_id": c.id,
                "type": c.type.value,
                "name": c.name,
                "email": c.email,
                "country": c.country,
                "since": c.created_at.date().isoformat(),
                "marketing_opt_in": rng.random() < 0.5,
                "shipping_address": {
                    "line1": street,
                    "city": rng.choice(["Portland", "Denver", "Austin", "Madison", "Burlington"]),
                    "postcode": f"{rng.randint(10000, 99999)}",
                    "country": c.country,
                },
                "recent_orders": [
                    {
                        "order_id": x,
                        "placed_at": self.orders[x].placed_at.date().isoformat(),
                        "total": self._total(self.orders[x]),
                        "status": self.orders[x].status.value,
                        "lines": len(self.orders[x].lines),
                    }
                    for x in others
                ],
                "contact_history": [
                    {
                        "contact_id": f"K-{rng.randint(100000, 999999)}",
                        "at": (o.placed_at - timedelta(days=rng.randint(20, 400)))
                        .date()
                        .isoformat(),
                        "channel": rng.choice(["email", "chat", "phone"]),
                        "topic": topic,
                        "resolved": True,
                        "agent_note": f"Customer asked about {topic}; answered from the help centre article and closed.",
                    }
                    for topic in rng.sample(topics, 4)
                ],
            },
            "shipment": None,
        }
        if sh is not None:
            events, t = [], sh.shipped_at
            stops = [
                "Label created",
                "Picked up",
                "Departed origin facility",
                "In transit",
                "Arrived at regional hub",
                "Out for delivery",
            ]
            for i, label in enumerate(stops):
                events.append(
                    {
                        "at": (t + timedelta(hours=9 * i)).isoformat(),
                        "event": label,
                        "location": rng.choice(["Riverside", "Hub 4", "Hub 7", "Local depot"]),
                    }
                )
            if sh.delivered_at:
                events.append(
                    {
                        "at": sh.delivered_at.isoformat(),
                        "event": "Delivered",
                        "location": "Front door",
                        "signed_by": None,
                    }
                )
            record["shipment"] = {
                "shipment_id": sh.id,
                "carrier": sh.carrier,
                "warehouse": sh.warehouse,
                "status": sh.status.value,
                "tracking_number": f"{rng.randint(10**11, 10**12 - 1)}",
                "shipped_at": sh.shipped_at.isoformat(),
                "promised_by": sh.promised_by.isoformat(),
                "delivered_at": sh.delivered_at.isoformat() if sh.delivered_at else None,
                "service_level": rng.choice(["ground", "express"]),
                "events": events,
            }
        return json.dumps(record, indent=1)

    def _view(self, o, extra: list[str] | None = None) -> str:
        return (
            "\n".join(self._summary(o) + (extra or []))
            + "\n\nFull record:\n"
            + self._full_record(o)
        )

    # -- the tools ---------------------------------------------------------------------------
    def _t_get_ticket(self, ticket_id: str) -> str:
        s = self.by_ticket.get(ticket_id)
        if s is None or ticket_id != self.ledger.current:
            raise ToolError(
                f"No open ticket {ticket_id!r}. You are handling {self.ledger.current}."
            )
        return render_ticket(s)

    def _t_customer_orders(self, email: str) -> str:
        ids = self.by_email.get(email.strip().lower())
        if not ids:
            raise ToolError(f"No customer with email {email!r}.")
        rows = [
            f"{o.id}: placed {o.placed_at.date()}, total {self._total(o):.2f} {o.currency}, status {o.status.value}"
            for o in map(self._order, ids)
        ]
        return f"{len(rows)} order(s) for {email}:\n" + "\n".join(rows)

    def _t_order_status(self, order_id: str) -> str:
        return self._view(self._order(order_id))

    def _t_check_return_eligibility(self, order_id: str, as_of: str) -> str:
        o = self._order(order_id)
        try:
            today = date.fromisoformat(as_of)
        except ValueError:
            raise ToolError(f"as_of must be a date like 2026-01-15, got {as_of!r}.") from None
        sh = self.d.shipments_by_order.get(o.id)
        extra = []
        if sh is None or sh.delivered_at is None:
            extra.append("Return window: not started; the order has not been delivered.")
        else:
            window = RETURN_WINDOW_DAYS[self.d.customers[o.customer_id].type.value]
            used = (today - sh.delivered_at.date()).days
            extra.append(
                f"Return window: OPEN. Day {used} of {window}."
                if used <= window
                else f"Return window: CLOSED. Day {used}; the window was {window} days from delivery."
            )
            approver = "Support" if self._total(o) <= REFUND_LIMIT_USD else "Finance"
            extra.append(
                f"Refund approver: {approver} (Support may refund up to USD {REFUND_LIMIT_USD})."
            )
        return self._view(o, extra)

    def _t_issue_refund(self, order_id: str, amount_usd: float, reason: str) -> str:
        self._order(order_id)
        if amount_usd <= 0:
            raise ToolError("amount_usd must be positive.")
        return f"Refund of USD {amount_usd:.2f} issued to order {order_id}."

    def _t_escalate_to_finance(self, order_id: str, reason: str) -> str:
        self._order(order_id)
        return f"Escalated order {order_id} to Finance."

    def _t_reply_to_customer(self, ticket_id: str, message: str) -> str:
        return f"Reply sent on {ticket_id}."

    def _t_route_to_engineering(self, ticket_id: str, summary: str) -> str:
        return f"Bug report filed with engineering for {ticket_id}."

    def _t_escalate_to_human(self, ticket_id: str, reason: str) -> str:
        return f"Ticket {ticket_id} handed to a human agent."
