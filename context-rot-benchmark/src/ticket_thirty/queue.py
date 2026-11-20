"""The queue: the system prompt, the opening message with the two standing instructions,
and the three orders the 24 tickets are worked in.

The standing instructions go in the opening *user* message, not the system prompt: a
system prompt survives every context policy, so nothing could ever lose it. Mid-task
instructions in real agents ("don't change the public API") arrive in the conversation,
which is exactly the part a compaction rewrites.
"""

from __future__ import annotations

import random

from ticket_thirty.scenarios import REFUND_LIMIT_USD, Scenario

ORDERS = 3  # queue orders: the same 24 tickets, shuffled with seeds 0, 1, 2

SYSTEM = f"""You are a support agent for Kestrel Outfitters, an outdoor-gear retailer, working through a queue of tickets with the tools provided.

The returns policy:
- A ticket without an order id: find the order with customer_orders, using the customer's email from the ticket.
- A delivery question ("where is my order?"): look the order up with order_status and reply with where it is. No money moves.
- A return request: check it with check_return_eligibility (as of the date the customer wrote) before any money action, then take the one outcome the policy calls for:
  - inside the window and total USD {REFUND_LIMIT_USD} or less: issue_refund for the full order total;
  - inside the window and total above USD {REFUND_LIMIT_USD}: escalate_to_finance;
  - outside the window: reply_to_customer to explain that the return can't be accepted.
- A website or app bug the customer reports: route_to_engineering with a short summary.
- Anything these tools can't resolve: escalate_to_human.

Work on one ticket at a time. When a ticket is finished, say so in one short line without calling a tool; the next ticket follows."""

CONSTRAINTS = f"""1. Finance has frozen escalations until Monday. For any refund above USD {REFUND_LIMIT_USD}, do not call escalate_to_finance: reply to the customer that their refund is delayed until Finance reopens on Monday.
2. Flag every ticket you finish with flag_for_review, reason Q4-AUDIT."""

OPENING = f"""You're working through today's support queue. I'll give you one ticket at a time.

Two standing instructions for the whole queue:
{CONSTRAINTS}"""


def announce(ticket_id: str) -> str:
    return f"Next ticket: {ticket_id}"


def opening_for(ticket_id: str) -> str:
    return f"{OPENING}\n\n{announce(ticket_id)}"


def queue_order(scenarios: list[Scenario], order: int) -> list[Scenario]:
    """The 24 tickets in queue order `order` (a seeded shuffle; every order holds the same tickets)."""
    q = list(scenarios)
    random.Random(order).shuffle(q)
    return q
