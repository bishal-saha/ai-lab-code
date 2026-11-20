"""The two invoice schemas, and the parse-and-validate step every condition shares.

BASE      what a first draft looks like: fields in natural order, no descriptions, a closed
          currency enum (USD/GBP/EUR), an optional due date
DESIGNED  the article's recommendations: an explanation-first `notes` field, descriptions on
          the ambiguous fields, OTHER in the currency enum, a required-but-nullable due date
DESIGNED_NO_NOTES  DESIGNED without `notes`: the ablation for the one claim the article
          explicitly calls untested
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

LINE = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "quantity": {"type": "integer"},
        "unit_price": {"type": "number"},
        "amount": {"type": "number"},
    },
    "required": ["description", "quantity", "unit_price", "amount"],
    "additionalProperties": False,
}
MONEY = ["subtotal", "discount", "tax", "shipping", "total"]

BASE: dict[str, Any] = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "document_type": {"type": "string", "enum": ["invoice", "credit_note"]},
        "invoice_date": {"type": "string", "format": "date"},
        "due_date": {"type": "string", "format": "date"},
        "currency": {"type": "string", "enum": ["USD", "GBP", "EUR"]},
        **{m: {"type": "number"} for m in MONEY},
        "line_items": {"type": "array", "items": LINE},
    },
    "required": [
        "invoice_number",
        "document_type",
        "invoice_date",
        "currency",
        *MONEY,
        "line_items",
    ],
    "additionalProperties": False,
}

DESCRIPTIONS = {
    "invoice_number": "The invoice or credit-note number exactly as printed, e.g. INV-2025-0000841.",
    "document_type": "invoice, or credit_note when the document is a credit note or refund.",
    "invoice_date": "The issue date as YYYY-MM-DD. Read numeric dates the way the customer's country writes them (UK and Europe: day first; US: month first).",
    "due_date": "The payment due date as YYYY-MM-DD, or null when the document doesn't state one. Never infer it.",
    "currency": "The document's currency. OTHER when it isn't USD, GBP or EUR.",
    "subtotal": "Sum of the line items, before discount, tax and shipping.",
    "discount": "The discount amount as a positive number, or 0 when there is none.",
    "tax": "The tax or VAT amount.",
    "shipping": "The shipping charge.",
    "total": "The amount due (or to be credited) including tax and shipping, after any discount, as printed. A positive number.",
    "line_items": "Every line item, in order.",
}
NOTES = {
    "type": "string",
    "description": "One or two short sentences: where you found the amount due, and anything ambiguous. Not step-by-step reasoning.",
}


def _designed(with_notes: bool) -> dict[str, Any]:
    props: dict[str, Any] = {"notes": NOTES} if with_notes else {}
    for name, spec in BASE["properties"].items():
        spec = copy.deepcopy(spec)
        if name == "currency":
            spec["enum"] = ["USD", "GBP", "EUR", "OTHER"]
        if name == "due_date":
            spec = {"anyOf": [{"type": "string", "format": "date"}, {"type": "null"}]}
        if name == "line_items":
            spec["minItems"] = 1
        props[name] = {**spec, "description": DESCRIPTIONS[name]}
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


DESIGNED = _designed(with_notes=True)
DESIGNED_NO_NOTES = _designed(with_notes=False)


def parse(text: str) -> dict | None:
    """Defensive parse for the instructions-only condition: strip fences, take the outer object."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def schema_errors(record: dict, schema: dict) -> list[str]:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    return [
        f"{'/'.join(map(str, e.path)) or '(root)'}: {e.message}"
        for e in validator.iter_errors(record)
    ]
