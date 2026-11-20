"""Field-by-field grading against the invoice's true values.

    fields       invoice_number, document_type, invoice_date, due_date, currency, subtotal, total,
                 line_count: each right or wrong
    all_correct  every field right
    valid_but_wrong  schema-valid and at least one field wrong: the failure constrained decoding
                 can't see, and the one the article is about

Currency on a JPY or CAD invoice is right when the record says OTHER (the designed
schema's way to say "not one of ours") or names the real currency; the base schema's
closed enum can do neither, so it can only be wrong there. A missing due date is
right when the record leaves it out or sets it to null; any date is a fabrication.
"""

from __future__ import annotations

from right_shape.invoices import Invoice

FIELDS = [
    "invoice_number",
    "document_type",
    "invoice_date",
    "due_date",
    "currency",
    "subtotal",
    "total",
    "line_count",
]


def _num(a, b, currency: str) -> bool:
    try:
        return abs(float(a) - float(b)) <= (0.5 if currency == "JPY" else 0.005)
    except (TypeError, ValueError):
        return False


def fields(inv: Invoice, r: dict | None) -> dict[str, bool]:
    if not r:
        return dict.fromkeys(FIELDS, False)
    cur = str(r.get("currency", "")).upper()
    return {
        "invoice_number": str(r.get("invoice_number", "")).strip().upper() == inv.invoice_number,
        "document_type": str(r.get("document_type", "")).lower() == inv.document_type,
        "invoice_date": r.get("invoice_date") == inv.invoice_date,
        "due_date": r.get("due_date") in (None, "")
        if inv.due_date is None
        else r.get("due_date") == inv.due_date,
        "currency": cur in ("OTHER", inv.currency)
        if inv.currency in ("JPY", "CAD")
        else cur == inv.currency,
        "subtotal": _num(r.get("subtotal"), inv.subtotal, inv.currency),
        "total": _num(r.get("total"), inv.total, inv.currency),
        "line_count": len(r.get("line_items") or []) == len(inv.lines),
    }
