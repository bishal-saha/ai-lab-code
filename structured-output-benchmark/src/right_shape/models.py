"""The four ways to ask Claude for an invoice record, and an offline fake.

    prompt  instructions only: a JSON example in the system prompt, parsed defensively
    tool    a `record_invoice` tool whose input schema is the record schema (not strict)
    strict  the same tool with `strict: true`: sampling constrained to schema-valid input
    so      structured outputs: the schema in `output_config.format`, constrained decoding

Each call returns the record (or None if nothing parseable came back), the raw text,
the stop reason and the token usage. `feedback` turns a call into the validated
condition's one retry: the previous record and the rule errors go back to the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import anthropic

from right_shape.invoices import Invoice
from right_shape.pricing import resolve
from right_shape.schemas import parse

SYSTEM = "You extract invoice data for the Kestrel Outfitters accounts team. Record the fields of the one document the user gives you, exactly as it states them."
EXAMPLE = {
    "invoice_number": "INV-2024-0000123",
    "document_type": "invoice",
    "invoice_date": "2024-06-01",
    "due_date": "2024-07-01",
    "currency": "USD",
    "subtotal": 100.0,
    "discount": 0.0,
    "tax": 8.25,
    "shipping": 9.95,
    "total": 118.2,
    "line_items": [
        {"description": "Example item", "quantity": 1, "unit_price": 100.0, "amount": 100.0}
    ],
}
TOOL = "record_invoice"


@dataclass
class Reply:
    record: dict | None
    raw: str
    stop_reason: str
    usage: dict[str, int]


class ClaudeModel:
    def __init__(
        self, model: str, *, effort: str = "medium", client: anthropic.AsyncAnthropic | None = None
    ):
        self.name = resolve(model)
        self.effort = effort
        self.client = client or anthropic.AsyncAnthropic()

    async def extract(
        self,
        inv: Invoice,
        mechanism: str,
        schema: dict,
        feedback: tuple[dict, list[str]] | None = None,
    ) -> Reply:
        messages: list[dict[str, Any]] = [{"role": "user", "content": inv.text}]
        if feedback:
            previous, errors = feedback
            messages += [
                {"role": "assistant", "content": json.dumps(previous)},
                {
                    "role": "user",
                    "content": "That record failed these checks:\n- "
                    + "\n- ".join(errors)
                    + "\nRe-read the document and give the corrected record.",
                },
            ]
        kwargs: dict[str, Any] = {"output_config": {"effort": self.effort}}
        system = SYSTEM
        if mechanism == "prompt":
            system += f"\n\nReply with only a JSON object in exactly this shape, and nothing else:\n{json.dumps(EXAMPLE, indent=2)}\nOmit due_date if the document doesn't state one."
        elif mechanism in ("tool", "strict"):
            tool = {
                "name": TOOL,
                "description": "Record the fields of one invoice or credit note.",
                "input_schema": schema,
            }
            if mechanism == "strict":
                tool["strict"] = True
            kwargs["tools"] = [tool]
            system += f" Call {TOOL} once with the record."
        elif mechanism == "so":
            kwargs["output_config"]["format"] = {"type": "json_schema", "schema": schema}
        r = await self.client.messages.create(
            model=self.name, max_tokens=8000, system=system, messages=messages, **kwargs
        )
        text = "".join(getattr(b, "text", "") for b in r.content if b.type == "text")
        if mechanism in ("tool", "strict"):
            call = next((b for b in r.content if b.type == "tool_use" and b.name == TOOL), None)
            record = dict(call.input) if call else None
            raw = json.dumps(call.input) if call else text
        else:
            record, raw = parse(text), text
        u = r.usage
        return Reply(
            record,
            raw,
            r.stop_reason or "",
            {
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cache_read": u.cache_read_input_tokens or 0,
                "cache_write": u.cache_creation_input_tokens or 0,
            },
        )


def oracle_record(inv: Invoice, schema: dict) -> dict:
    """The record a perfect reader returns, in the given schema's shape."""
    enum = schema["properties"]["currency"]["enum"]
    rec = {
        "invoice_number": inv.invoice_number,
        "document_type": inv.document_type,
        "invoice_date": inv.invoice_date,
        "due_date": inv.due_date,
        "currency": inv.currency
        if inv.currency in enum
        else ("OTHER" if "OTHER" in enum else "USD"),
        "subtotal": inv.subtotal,
        "discount": inv.discount,
        "tax": inv.tax,
        "shipping": inv.shipping,
        "total": inv.total,
        "line_items": [
            {
                "description": ln.description,
                "quantity": ln.quantity,
                "unit_price": ln.unit_price,
                "amount": ln.amount,
            }
            for ln in inv.lines
        ],
    }
    if "notes" in schema["properties"]:
        rec = {"notes": "Amount due taken from the last line of the totals.", **rec}
    if inv.due_date is None and "due_date" not in schema["required"]:
        rec.pop("due_date")
    return rec


def mistaken_record(inv: Invoice, schema: dict) -> dict:
    """The plausible mistake each trap invites, in a schema-valid shape."""
    rec = oracle_record(inv, schema)
    if inv.trap == "subtotal":
        rec["total"] = inv.subtotal
    elif inv.trap == "discount":
        rec["total"] = round(inv.subtotal + inv.tax + inv.shipping, 2)
    elif inv.trap == "words":
        digits = f"{inv.total:.2f}"
        rec["total"] = (
            float(digits[1] + digits[0] + digits[2:]) if digits[0] != digits[1] else inv.total + 9
        )
    elif inv.trap == "date":
        y, m, d = inv.invoice_date.split("-")
        rec["invoice_date"] = f"{y}-{d}-{m}"
    elif inv.trap in ("no_due_date", "credit_note") and "due_date" in schema["properties"]:
        rec["due_date"] = (date.fromisoformat(inv.invoice_date) + timedelta(days=30)).isoformat()
        if inv.trap == "credit_note":
            rec["document_type"] = "invoice"
    elif inv.trap == "currency":
        rec["currency"] = "USD"
    return rec


class FakeModel:
    """Offline stand-in. "oracle" answers right; "mistake" makes each trap's mistake, then fixes it on retry."""

    name = "fake"

    def __init__(self, behaviour: str = "oracle"):
        self.behaviour = behaviour

    async def extract(self, inv: Invoice, mechanism: str, schema: dict, feedback=None) -> Reply:
        rec = (
            oracle_record(inv, schema)
            if self.behaviour == "oracle" or feedback
            else mistaken_record(inv, schema)
        )
        return Reply(
            rec,
            json.dumps(rec),
            "end_turn",
            {"input_tokens": len(inv.text) // 4 + 300, "output_tokens": 250},
        )
