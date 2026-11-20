"""The 24 support tickets the agent works through in one window, and what the policy says to do.

Each scenario is a synthetic ticket (T-Q001 ... T-Q024) laid over a real order in
the Kestrel dataset. The correct outcome is computed here, from the dataset and
the written returns policy, never through the support server:

    inside the window and total <= USD 500  -> refund     (issue_refund, full total)
    inside the window and total >  USD 500  -> escalate   (escalate_to_finance)
    outside the window                      -> decline    (reply_to_customer)
    "where is my order?"                    -> status     (read order_status, reply)

Windows: 30 days from delivery for consumers, 14 for trade customers.

Four kinds (experiment 3's, extended so no ticket repeats in a 24-ticket queue):
    policy    12  one return request per ticket (5 refund, 4 escalate, 3 decline)
    multi      4  a return request plus a website bug, which also needs route_to_engineering
    status     4  a delivery question: no money moves (2 delivered, 2 in transit)
    no_order   4  a return request without the order id: find it with customer_orders
                  (refund, decline, delayed, decline)

The queue runs under a standing instruction: escalations to Finance are frozen, so a
ticket the policy would escalate must instead get a reply saying the refund is delayed
(`expected` below); see queue.py for the opening message.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta

from kestrel.data import generate
from kestrel.services import KestrelData

SEED = 2026
SCALE = 0.1  # 2,000 orders, 1,800 shipments, 300 tickets, 40 products
REFUND_LIMIT_USD = 500
WINDOW_DAYS = {"consumer": 30, "trade": 14}

# (customer type, days since delivery, total above USD 500)
POLICY = [
    ("consumer", 10, False),  # refund
    ("consumer", 20, False),  # refund
    ("consumer", 25, False),  # refund
    ("consumer", 5, False),  # refund
    ("consumer", 15, False),  # refund
    ("consumer", 10, True),  # escalate -> delayed under the freeze
    ("trade", 10, True),  # escalate (trade orders here are all above USD 500)
    ("consumer", 22, True),  # escalate
    ("trade", 6, True),  # escalate
    ("consumer", 45, False),  # decline
    ("consumer", 45, True),  # decline
    ("trade", 20, True),  # decline: inside a consumer's 30 days, outside trade's 14
]
MULTI = [
    ("consumer", 12, False),
    ("consumer", 22, False),
    ("consumer", 40, False),
    ("trade", 18, True),
]
# Only a few customers have exactly one order (lookups are by email, which some share), so
# two no-order tickets use orders above USD 500: one becomes "delayed" under the freeze.
NO_ORDER = [
    ("consumer", 8, False),
    ("consumer", 50, False),
    ("consumer", 12, True),
    ("consumer", 40, True),
]
BUGS = [
    "Also, the order tracking page on your website shows an error every time I open it.",
    "Separately, your app logs me out whenever I open my order history.",
    "One more thing: the size guide link on the product page opens a blank page.",
    "Also, checkout keeps rejecting my saved address, so I can't place a new order.",
]


@dataclass(frozen=True)
class Scenario:
    id: str
    kind: str  # policy | multi | status | no_order
    ticket_id: str
    today: str
    order_id: str
    customer_name: str
    customer_email: str
    customer_type: str
    total: float
    outcome: str  # refund | escalate | decline | status
    message: str
    order_in_ticket: bool = True
    bug: str | None = None

    @property
    def needs_engineering(self) -> bool:
        return self.bug is not None

    @property
    def expected(self) -> str:
        """The outcome under the escalation freeze: an escalation becomes a 'delayed' reply."""
        return "delayed" if self.outcome == "escalate" else self.outcome

    @property
    def refund_due(self) -> float:
        return self.total if self.outcome == "refund" else 0.0

    @property
    def prompt(self) -> str:
        return f"Today is {self.today}. Resolve support ticket {self.ticket_id}."


def dataset() -> KestrelData:
    return KestrelData(generate(SEED, SCALE))


def directory(d: KestrelData) -> dict[str, list[str]]:
    """Customer email -> their order ids: the CRM lookup behind the customer_orders tool."""
    out: dict[str, list[str]] = defaultdict(list)
    for o in sorted(d.dataset.orders, key=lambda o: o.id):
        out[d.customers[o.customer_id].email.lower()].append(o.id)
    return dict(out)


def _outcome(kind: str, days: int, total: float) -> str:
    if days > WINDOW_DAYS[kind]:
        return "decline"
    return "refund" if total <= REFUND_LIMIT_USD else "escalate"


def build_scenarios(d: KestrelData | None = None) -> list[Scenario]:
    d = d or dataset()
    rng = random.Random(SEED + 9)  # a different draw from the earlier experiments
    usd = sorted((o for o in d.dataset.orders if o.currency == "USD"), key=lambda o: o.id)
    shipped = {o.id: d.shipments_by_order[o.id] for o in usd if o.id in d.shipments_by_order}
    delivered = [o for o in usd if o.id in shipped and shipped[o.id].delivered_at]
    in_transit = [o for o in usd if o.id in shipped and not shipped[o.id].delivered_at]
    # Counted by email, which is what customer_orders looks up: synthetic customers can share one.
    orders_per_email: dict[str, int] = defaultdict(int)
    for o in d.dataset.orders:
        orders_per_email[d.customers[o.customer_id].email.lower()] += 1

    used: set[str] = set()
    out: list[Scenario] = []

    def pick(pool, kind: str, over: bool | None, single_order: bool = False):
        cands = [
            o
            for o in pool
            if o.id not in used
            and d.customers[o.customer_id].type.value == kind
            and (over is None or (o.total_cents > REFUND_LIMIT_USD * 100) == over)
            and (
                not single_order or orders_per_email[d.customers[o.customer_id].email.lower()] == 1
            )
        ]
        order = rng.choice(cands)
        used.add(order.id)
        return order

    def add(kind: str, order, today, outcome: str, message: str, **extra) -> None:
        c = d.customers[order.customer_id]
        n = len(out) + 1
        out.append(
            Scenario(
                id=f"{kind[0]}{sum(s.kind == kind for s in out) + 1}",
                kind=kind,
                ticket_id=f"T-Q{n:03d}",
                today=today.isoformat(),
                order_id=order.id,
                customer_name=c.name,
                customer_email=c.email,
                customer_type=c.type.value,
                total=order.total_cents / 100,
                outcome=outcome,
                message=message,
                **extra,
            )
        )

    def first(order) -> str:
        return d.customers[order.customer_id].name.split()[0]

    def returning(order, today_days: int, kind: str):
        on = shipped[order.id].delivered_at.date()
        return on + timedelta(days=today_days), _outcome(kind, today_days, order.total_cents / 100)

    for kind, days, over in POLICY:
        order = pick(delivered, kind, over)
        today, outcome = returning(order, days, kind)
        add(
            "policy",
            order,
            today,
            outcome,
            f"Hi, I'd like to return my order {order.id} for a refund. "
            f"The item is unused and still in its box. Thanks, {first(order)}",
        )

    for (kind, days, over), bug in zip(MULTI, BUGS, strict=True):
        order = pick(delivered, kind, over)
        today, outcome = returning(order, days, kind)
        add(
            "multi",
            order,
            today,
            outcome,
            f"Hi, I'd like to return my order {order.id} for a refund; "
            f"the item is unused. {bug} Thanks, {first(order)}",
            bug=bug,
        )

    for pool in (delivered, delivered, in_transit, in_transit):
        order = pick(pool, "consumer", None)
        s = shipped[order.id]
        today = (
            (s.delivered_at.date() + timedelta(days=3))
            if s.delivered_at
            else (s.shipped_at.date() + timedelta(days=2))
        )
        add(
            "status",
            order,
            today,
            "status",
            f"Hi, where is my order {order.id}? I haven't had an update "
            f"in a while. Thanks, {first(order)}",
        )

    for kind, days, over in NO_ORDER:
        order = pick(delivered, kind, over, single_order=True)
        today, outcome = returning(order, days, kind)
        add(
            "no_order",
            order,
            today,
            outcome,
            "Hi, I'd like to return the item I bought from you for a refund. "
            f"It's unused and still in its box. I don't have the order number to hand, sorry. Thanks, {first(order)}",
            order_in_ticket=False,
        )

    return out
