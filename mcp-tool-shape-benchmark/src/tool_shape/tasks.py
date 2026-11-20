"""The benchmark's 30 customer-support tasks and their ground truth.

Ground truth is computed here, straight from the generated dataset and the written
policies, never through either MCP server. So the task-shaped tools are checked
too: if one of them misapplies a policy, its score drops like anyone else's.

Every task asks for a few typed fields, which the agent submits through the
`submit_answer` tool and `grading.py` compares exactly. No LLM judge.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from kestrel.data import generate
from kestrel.services import KestrelData

SEED = 2026
SCALE = 0.1  # 2,000 orders, 1,800 shipments, 300 tickets, 40 products

# Field types become the submit_answer JSON schema for the task.
BOOL = {"type": "boolean"}
INT = {"type": "integer"}
STR = {"type": "string"}
APPROVER = {"type": "string", "enum": ["support", "finance"]}


@dataclass(frozen=True)
class Task:
    id: str
    family: str
    prompt: str
    fields: dict[str, dict[str, Any]]
    expected: dict[str, Any]


def dataset() -> KestrelData:
    return KestrelData(generate(SEED, SCALE))


def _delivery(d: KestrelData, rng: random.Random) -> list[Task]:
    shipments = sorted(d.dataset.shipments, key=lambda s: s.id)
    on_time = [s for s in shipments if s.delivered_at and not s.late]
    late = [s for s in shipments if s.delivered_at and s.late]
    transit = [s for s in shipments if s.status.value == "in_transit"]
    lost = [s for s in shipments if s.status.value == "lost"]
    picks = (
        rng.sample(on_time, 2) + rng.sample(late, 2) + rng.sample(transit, 1) + rng.sample(lost, 1)
    )
    return [
        Task(
            id=f"delivery-{i + 1}",
            family="delivery",
            prompt=f"A customer asks about order {s.order_id}. Has it been delivered, was it late, "
            "and which carrier is handling it?",
            fields={"delivered": BOOL, "late": BOOL, "carrier": STR},
            expected={
                "delivered": s.delivered_at is not None,
                "late": s.late,
                "carrier": s.carrier,
            },
        )
        for i, s in enumerate(picks)
    ]


def _returns(d: KestrelData, rng: random.Random) -> list[Task]:
    tasks: list[Task] = []
    delivered = sorted(
        (o for o in d.dataset.orders if o.currency == "USD"),
        key=lambda o: o.id,
    )
    delivered = [o for o in delivered if (s := d.shipments_by_order.get(o.id)) and s.delivered_at]
    # (customer type, days after delivery, total above USD 500). Days sit well clear
    # of each window's edge, except the deliberate trap: day 20 or 25 is inside a
    # consumer's 30 days but outside a trade customer's 14. Trade orders in this
    # dataset are all above USD 500, so trade refunds always need Finance.
    combos = [
        ("consumer", 10, False), ("consumer", 10, True),
        ("consumer", 45, False), ("consumer", 45, True),
        ("consumer", 20, False),
        ("trade", 10, True), ("trade", 20, True), ("trade", 25, True),
    ]  # fmt: skip
    for kind, days, over_limit in combos:
        pool = [
            o
            for o in delivered
            if d.customers[o.customer_id].type.value == kind
            and (o.total_cents > 50000) == over_limit
        ]
        order = rng.choice(pool)
        delivered_on = d.shipments_by_order[order.id].delivered_at.date()
        today = delivered_on + timedelta(days=days)
        window = 30 if kind == "consumer" else 14
        tasks.append(
            Task(
                id=f"returns-{len(tasks) + 1}",
                family="returns",
                prompt=f"Today is {today.isoformat()}. The customer on order {order.id} wants to return "
                "an unused item for a refund. Is the return inside the return window, and who would "
                "have to approve the refund (support or finance)?",
                fields={"eligible": BOOL, "approver": APPROVER},
                expected={
                    "eligible": days <= window,
                    "approver": "finance" if over_limit else "support",
                },
            )
        )
    return tasks


def _tickets(d: KestrelData, rng: random.Random) -> list[Task]:
    linked = sorted(
        (t for t in d.dataset.tickets if t.order_id in d.shipments_by_order),
        key=lambda t: t.id,
    )
    late = [t for t in linked if d.shipments_by_order[t.order_id].late]
    fine = [t for t in linked if not d.shipments_by_order[t.order_id].late]
    picks = rng.sample(late, 4) + rng.sample(fine, 4)
    tasks = []
    for i, t in enumerate(picks):
        s = d.shipments_by_order[t.order_id]
        tasks.append(
            Task(
                id=f"ticket-{i + 1}",
                family="ticket",
                prompt=f"Triage support ticket {t.id}. Which order is it about, has that order been "
                "delivered, and was its delivery late?",
                fields={"order_id": STR, "delivered": BOOL, "late": BOOL},
                expected={
                    "order_id": t.order_id,
                    "delivered": s.delivered_at is not None,
                    "late": s.late,
                },
            )
        )
    return tasks


def _stock(d: KestrelData, rng: random.Random) -> list[Task]:
    products = sorted(d.dataset.products, key=lambda p: p.sku)
    active = [p for p in products if p.active]
    inactive = [p for p in products if not p.active]
    picks = rng.sample(active, 3) + rng.sample(inactive, 1)
    tasks = []
    for i, p in enumerate(picks):
        total = sum(w["on_hand"] for w in d.stock(p.sku)["warehouses"])
        tasks.append(
            Task(
                id=f"stock-{i + 1}",
                family="stock",
                prompt=f"Is {p.sku} in stock in any warehouse, and how many units are on hand in total?",
                fields={"in_stock": BOOL, "total": INT},
                expected={"in_stock": total > 0, "total": total},
            )
        )
    return tasks


def _policy() -> list[Task]:
    # Answers come from the current documents in kestrel/docs. The outdated v2
    # shipping policy says 4 days for standard delivery; the current one says 3.
    questions = [
        ("What is the standard delivery promise, in working days, under the current shipping policy?",
         {"working_days": INT}, {"working_days": 3}),
        ("How many days after delivery do trade customers have to return unused items?",
         {"days": INT}, {"days": 14}),
        ("Up to what refund amount, in USD, can the Support team approve on its own?",
         {"usd": INT}, {"usd": 500}),
        ("An Express order was delivered late. Is the express fee refunded automatically?",
         {"refunded": BOOL}, {"refunded": True}),
    ]  # fmt: skip
    return [
        Task(id=f"policy-{i + 1}", family="policy", prompt=q, fields=f, expected=e)
        for i, (q, f, e) in enumerate(questions)
    ]


def build_tasks(d: KestrelData | None = None) -> list[Task]:
    d = d or dataset()
    rng = random.Random(SEED)
    return _delivery(d, rng) + _returns(d, rng) + _tickets(d, rng) + _stock(d, rng) + _policy()


def answer_tool(task: Task) -> dict[str, Any]:
    """The submit_answer tool for one task: its schema is the task's fields, strict."""
    return {
        "name": "submit_answer",
        "description": "Submit your final answer. Call this exactly once, when you are sure.",
        "input_schema": {
            "type": "object",
            "properties": task.fields,
            "required": list(task.fields),
            "additionalProperties": False,
        },
        "strict": True,
    }
