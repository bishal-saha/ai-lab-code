"""Graph workflow: code owns every transition; the model only works inside nodes.

    read ticket (code) -> classify (LLM, one intent) -> resolve order (code)
      refund_request -> eligibility (code, retries a temporary failure) -> policy decision (code)
                        -> refund | escalate to Finance | decline -> draft reply (LLM) -> send (code)
      order_status   -> order_status (code) -> draft reply (LLM) -> send (code)
      bug_report     -> route_to_engineering (code) -> draft reply (LLM) -> send (code)
      other          -> escalate_to_human (code)

This is the article's design: a classifier built for one intent per ticket, with a
fallback edge to a human. A ticket with two intents takes one path; which one, and
what it costs, is what the multi-intent tickets measure.
"""

from __future__ import annotations

import re

from four_shapes.episode import POLICY, Context, Stop
from four_shapes.scenarios import REFUND_LIMIT_USD

INTENTS = ["refund_request", "order_status", "bug_report", "other"]
CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "order_id": {
            "type": "string",
            "description": "The order id in the ticket, like O-0000123, or empty",
        },
    },
    "required": ["intent", "order_id"],
    "additionalProperties": False,
}
DRAFT_SCHEMA = {
    "type": "object",
    "properties": {"message": {"type": "string"}},
    "required": ["message"],
    "additionalProperties": False,
}
CLASSIFY_SYSTEM = """You classify one Kestrel Outfitters support ticket into exactly one intent:
- refund_request: the customer wants to return an item for a refund
- order_status: the customer asks where an order is
- bug_report: the customer reports a website or app problem
- other: anything else
Also copy the order id from the ticket if there is one (like O-0000123); otherwise leave it empty."""
DRAFT_SYSTEM = f"""You write the reply to a Kestrel Outfitters customer, in two to four friendly sentences.
State only what the facts say happened; never promise anything else.

Context, the support policy:
{POLICY}"""
ORDER_ID = re.compile(r"\bO-\d{7}\b")
EMAIL = re.compile(r"<([^>]+@[^>]+)>")
TEMPORARY = "temporarily unavailable"


async def _reply(ctx: Context, ticket: str, facts: str) -> None:
    draft = await ctx.json(
        "draft", DRAFT_SYSTEM, f"Ticket:\n{ticket}\n\nWhat happened:\n{facts}", DRAFT_SCHEMA
    )
    await ctx.call("reply_to_customer", {"ticket_id": ctx.s.ticket_id, "message": draft["message"]})


async def run(ctx: Context) -> None:
    s = ctx.s
    ticket, err = await ctx.call("get_ticket", {"ticket_id": s.ticket_id})
    if err:
        raise Stop("breaker", ticket)

    label = await ctx.json("classify", CLASSIFY_SYSTEM, ticket, CLASSIFY_SCHEMA)
    intent = label["intent"] if label["intent"] in INTENTS else "other"
    ctx.ep.notes.append(f"intent={intent}")

    if intent == "other":
        await ctx.call(
            "escalate_to_human",
            {"ticket_id": s.ticket_id, "reason": "ticket outside the designed paths"},
        )
        return
    if intent == "bug_report":
        message = ticket.split("wrote on", 1)[-1].split(":", 1)[-1].strip()
        await ctx.call("route_to_engineering", {"ticket_id": s.ticket_id, "summary": message[:300]})
        await _reply(ctx, ticket, "The bug report was passed to the engineering team.")
        return

    # Resolve the order: the classifier's id if it looks real, else the customer's only order.
    order_id = label["order_id"] if ORDER_ID.fullmatch(label["order_id"] or "") else ""
    if not order_id:
        email = EMAIL.search(ticket)
        listing, err = await ctx.call("customer_orders", {"email": email.group(1) if email else ""})
        ids = [] if err else ORDER_ID.findall(listing)
        if len(ids) != 1:
            await ctx.call(
                "escalate_to_human", {"ticket_id": s.ticket_id, "reason": "can't tell which order"}
            )
            return
        order_id = ids[0]

    if intent == "order_status":
        status, err = await ctx.call("order_status", {"order_id": order_id})
        if err:
            await ctx.call("escalate_to_human", {"ticket_id": s.ticket_id, "reason": status})
            return
        await _reply(ctx, ticket, status)
        return

    # refund_request: the policy decision is code, from the eligibility tool's own words.
    for _ in range(3):
        check, err = await ctx.call(
            "check_return_eligibility", {"order_id": order_id, "as_of": s.today}
        )
        if not (err and TEMPORARY in check):
            break
    if err:
        await ctx.call("escalate_to_human", {"ticket_id": s.ticket_id, "reason": check})
        return
    total = float(re.search(r"total (\d+(?:\.\d+)?) ", check).group(1))
    if "Return window: OPEN" in check and total <= REFUND_LIMIT_USD:
        done, _ = await ctx.call(
            "issue_refund",
            {"order_id": order_id, "amount_usd": total, "reason": "Return inside the window"},
        )
    elif "Return window: OPEN" in check:
        done, _ = await ctx.call(
            "escalate_to_finance",
            {
                "order_id": order_id,
                "reason": f"Refund of USD {total:.2f} is above USD {REFUND_LIMIT_USD}",
            },
        )
    else:
        done = "The return was declined: " + check.splitlines()[-2 if "approver" in check else -1]
    await _reply(ctx, ticket, f"{check}\n{done}")
