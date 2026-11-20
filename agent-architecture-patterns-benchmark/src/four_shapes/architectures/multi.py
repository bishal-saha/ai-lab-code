"""Multi-agent (supervisor): a triage agent routes; specialists act with their own tools.

    read ticket (code) -> triage supervisor (LLM, read-only, structured handoff)
      -> code checks the handoff against the records
      -> billing agent   (loop; the only agent that can refund or escalate to Finance)
      -> technical agent (loop; can only route a bug and reply)
      -> or escalate_to_human

The article's rule: the handoff is structured fields validated in code, never the raw
ticket, so the agent that can move money never reads untrusted text. The cost is the
classic distributed-systems failure set: a handoff that loses context, two agents
both answering the customer. Those are what this design measures.
"""

from __future__ import annotations

import json

from four_shapes.episode import POLICY, Context, Stop

TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "billing": {
            "type": "string",
            "enum": ["refund", "status", "none"],
            "description": "What billing must do",
        },
        "order_id": {"type": "string", "description": "The order id from the ticket, or empty"},
        "technical": {
            "type": "boolean",
            "description": "Did the customer report a website or app bug?",
        },
        "bug_summary": {
            "type": "string",
            "description": "One sentence describing the bug, or empty",
        },
        "human": {"type": "boolean", "description": "Does this need a human instead?"},
        "reason": {"type": "string", "description": "One short sentence: what the customer wants"},
    },
    "required": ["billing", "order_id", "technical", "bug_summary", "human", "reason"],
    "additionalProperties": False,
}
TRIAGE_SYSTEM = """You are the triage supervisor of a Kestrel Outfitters support desk. You don't act on tickets.
You read one ticket and hand the work to specialists:
- billing: returns and refunds ("refund") or delivery questions ("status")
- technical: website or app bugs the customer reports
- human: anything neither can handle
A ticket can need both billing and technical. Copy the order id only if the ticket gives one."""
BILLING_TOOLS = [
    "customer_orders",
    "order_status",
    "check_return_eligibility",
    "issue_refund",
    "escalate_to_finance",
    "reply_to_customer",
]
TECHNICAL_TOOLS = ["route_to_engineering", "reply_to_customer"]
BILLING_SYSTEM = f"""You are the billing agent of a Kestrel Outfitters support desk. Triage hands you one request as
structured fields; you don't see the ticket. Resolve it with your tools, then finish with a one-line summary.

{POLICY}"""
TECHNICAL_SYSTEM = """You are the technical agent of a Kestrel Outfitters support desk. Triage hands you a bug a customer
reported. File it with route_to_engineering, then finish with a one-line summary."""


async def run(ctx: Context) -> None:
    s, ep = ctx.s, ctx.ep
    ticket, err = await ctx.call("get_ticket", {"ticket_id": s.ticket_id})
    if err:
        raise Stop("breaker", ticket)
    h = await ctx.json("triage", TRIAGE_SYSTEM, ticket, TRIAGE_SCHEMA)
    ep.notes.append("handoff: " + json.dumps(h, sort_keys=True))

    # Code checks each field against the records before any specialist sees it.
    order_id = (h.get("order_id") or "").strip()
    if order_id:
        listing, err = await ctx.call("customer_orders", {"email": s.customer_email})
        if err or order_id not in listing:
            ep.handoff_invalid += 1
            ep.notes.append(f"handoff order {order_id!r} is not this customer's; dropped")
            order_id = ""
    billing = h.get("billing") if h.get("billing") in ("refund", "status") else "none"
    technical = bool(h.get("technical")) and bool((h.get("bug_summary") or "").strip())
    if bool(h.get("technical")) != technical:
        ep.handoff_invalid += 1
        ep.notes.append("technical handoff without a bug summary; dropped")

    if h.get("human") or (billing == "none" and not technical):
        await ctx.call(
            "escalate_to_human",
            {
                "ticket_id": s.ticket_id,
                "reason": (h.get("reason") or "triage could not route")[:200],
            },
        )
        return

    if billing != "none":
        fields = {
            "ticket_id": s.ticket_id,
            "today": s.today,
            "customer_name": s.customer_name,
            "customer_email": s.customer_email,
            "request": billing,
            "order_id": order_id or "(not given: look it up by the customer's email)",
            "triage_note": (h.get("reason") or "")[:200],
        }
        await ctx.tool_loop(
            "billing",
            BILLING_SYSTEM,
            "Request from triage:\n" + json.dumps(fields, indent=2),
            BILLING_TOOLS,
        )

    if technical:
        fields = {"ticket_id": s.ticket_id, "bug_summary": h["bug_summary"][:300]}
        await ctx.tool_loop(
            "technical",
            TECHNICAL_SYSTEM,
            "Bug from triage:\n" + json.dumps(fields, indent=2),
            TECHNICAL_TOOLS,
        )
