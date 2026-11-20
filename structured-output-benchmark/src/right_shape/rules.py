"""Business-rule validation: what a schema can't say.

A schema can say `total` is a number; it can't say the number is right. These
checks can catch some wrong numbers, because an invoice's numbers have to agree
with each other. Each returns plain-language errors, which the `so+d+v` condition
sends back to the model for one retry. OTHER currency is not an error: it is
routed to a person.
"""

from __future__ import annotations

TOLERANCE = 0.011


def check(r: dict) -> list[str]:
    errors = []
    lines = r.get("line_items") or []
    for i, ln in enumerate(lines, start=1):
        if abs(ln["quantity"] * ln["unit_price"] - ln["amount"]) > TOLERANCE * max(
            1, ln["quantity"]
        ):
            errors.append(
                f"line {i}: quantity x unit price ({ln['quantity']} x {ln['unit_price']}) doesn't equal its amount ({ln['amount']})"
            )
    if lines and abs(sum(ln["amount"] for ln in lines) - r["subtotal"]) > TOLERANCE * len(lines):
        errors.append(
            f"the line items add up to {sum(ln['amount'] for ln in lines):.2f}, not the subtotal {r['subtotal']}"
        )
    expected = r["subtotal"] - r["discount"] + r["tax"] + r["shipping"]
    if abs(expected - r["total"]) > TOLERANCE * 3:
        errors.append(
            f"subtotal - discount + tax + shipping = {expected:.2f}, but the total is {r['total']}"
        )
    if r.get("due_date") and r.get("invoice_date") and r["due_date"] < r["invoice_date"]:
        errors.append(
            f"the due date {r['due_date']} is before the invoice date {r['invoice_date']}"
        )
    return errors


def needs_review(r: dict) -> bool:
    return r.get("currency") == "OTHER"
