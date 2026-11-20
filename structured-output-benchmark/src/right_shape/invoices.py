"""120 invoices rendered from real Kestrel orders, each with its true values and one trap.

Line items, quantities, prices, currencies, customers and dates come from the
Kestrel Outfitters dataset (seed 2026); totals are computed here, so every field a
model extracts has a known right answer. Four layouts (a clean table, a forwarded
email, a noisy OCR scan, prose) and eight trap types, 15 invoices each:

    subtotal       the order value is printed big at the top; the amount due is at the bottom
    discount       a pre-discount total comes first, then the discount, then the amount due
    currency       a partner invoice in JPY or CAD: the base schema's closed enum can't say it
    no_due_date    no due date is printed: the right answer is "none", not an invented date
    credit_note    a credit note, amounts shown as refunds
    date           03/04/2025 on a UK invoice: day first
    words          the amount due appears only in words
    none           a control invoice

    uv run python -m right_shape.invoices --write   # refresh invoices/ (committed, so they can be read)
"""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

from kestrel.data import generate
from kestrel.services import KestrelData

SEED = 2026
SCALE = 0.1
ROOT = Path(__file__).resolve().parents[2]
TRAPS = ["subtotal", "discount", "currency", "no_due_date", "credit_note", "date", "words", "none"]
LAYOUTS = ["table", "email", "ocr", "prose"]
TAX = {
    "USD": ("Sales tax", 0.0825),
    "GBP": ("VAT", 0.20),
    "EUR": ("VAT", 0.21),
    "JPY": ("Consumption tax", 0.10),
    "CAD": ("GST", 0.05),
}
SHIPPING = {"USD": 9.95, "GBP": 7.95, "EUR": 8.95, "JPY": 1500.0, "CAD": 12.95}
SYMBOL = {"USD": "$", "GBP": "£", "EUR": "€", "JPY": "¥", "CAD": "C$"}
RATE = {"JPY": 150.0, "CAD": 1.36}  # from USD, for partner invoices
UNIT_WORDS = {
    "USD": ("dollars", "cents"),
    "GBP": ("pounds", "pence"),
    "EUR": ("euros", "cents"),
    "JPY": ("yen", ""),
    "CAD": ("Canadian dollars", "cents"),
}


@dataclass(frozen=True)
class Line:
    description: str
    quantity: int
    unit_price: float
    amount: float


@dataclass(frozen=True)
class Invoice:
    id: str
    trap: str
    layout: str
    invoice_number: str
    document_type: str  # invoice | credit_note
    invoice_date: str  # ISO
    due_date: str | None  # ISO, or None when not printed
    currency: str  # USD | GBP | EUR | JPY | CAD
    subtotal: float
    discount: float
    tax: float
    shipping: float
    total: float
    customer: str
    country: str
    lines: tuple[Line, ...] = field(default_factory=tuple)
    text: str = ""

    def truth(self) -> dict:
        d = asdict(self)
        d.pop("text")
        return d


# ─── Money and words ──────────────────────────────────────────────────────────

ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
TENS = ["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else "-" + ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return ONES[n // 100] + " hundred" + ("" if rest == 0 else " and " + words(rest))
    if n < 1_000_000:
        rest = n % 1000
        return (
            words(n // 1000)
            + " thousand"
            + ("" if rest == 0 else (" and " if rest < 100 else " ") + words(rest))
        )
    rest = n % 1_000_000
    return words(n // 1_000_000) + " million" + ("" if rest == 0 else " " + words(rest))


def amount_in_words(value: float, currency: str) -> str:
    major_name, minor_name = UNIT_WORDS[currency]
    major, minor = int(value), round((value - int(value)) * 100)
    text = f"{words(major)} {major_name}"
    if minor and minor_name:
        text += f" and {words(minor)} {minor_name}"
    return text


def money(value: float, currency: str) -> str:
    if currency == "JPY":
        return f"¥{value:,.0f}"
    return f"{SYMBOL[currency]}{value:,.2f}"


def fmt_date(d: date, country: str, style: str = "long") -> str:
    if style == "numeric":
        return d.strftime("%d/%m/%Y") if country != "US" else d.strftime("%m/%d/%Y")
    return d.strftime("%-d %B %Y") if country != "US" else d.strftime("%B %-d, %Y")


# ─── Rendering ────────────────────────────────────────────────────────────────


def _totals_block(inv: Invoice, sym=money) -> list[str]:
    c = inv.currency
    out = [f"Subtotal: {sym(inv.subtotal, c)}"]
    if inv.discount:
        out.append(f"Discount (10%): -{sym(inv.discount, c)}")
    out += [
        f"{TAX[c][0]} ({TAX[c][1] * 100:g}%): {sym(inv.tax, c)}",
        f"Shipping: {sym(inv.shipping, c)}",
    ]
    label = "Amount to be credited" if inv.document_type == "credit_note" else "Amount due"
    out.append(
        f"{label}: {amount_in_words(inv.total, c)}"
        if inv.trap == "words"
        else f"{label}: {sym(inv.total, c)}"
    )
    return out


def render(inv: Invoice, d_invoice: date, d_due: date | None) -> str:
    c, kind = inv.currency, ("CREDIT NOTE" if inv.document_type == "credit_note" else "INVOICE")
    issued = fmt_date(d_invoice, inv.country, "numeric" if inv.trap == "date" else "long")
    # On a date-trap invoice every date is numeric, so the due date can't give the issue date away.
    due_style = "numeric" if inv.trap == "date" else "long"
    due = (
        f"Payment due: {fmt_date(d_due, inv.country, due_style)}"
        if d_due
        else "Payment terms: see your account agreement"
    )
    if inv.document_type == "credit_note":
        due = "Refund to original payment method"
    seller = (
        "Kestrel Outfitters Japan KK (partner)"
        if c == "JPY"
        else "Kestrel Outfitters Canada (partner)"
        if c == "CAD"
        else "Kestrel Outfitters Ltd"
    )
    head = []
    if inv.trap == "subtotal":
        head = [f"ORDER VALUE  {money(inv.subtotal, c)}", ""]
    if inv.trap == "discount":
        head = [f"Total before discount: {money(inv.subtotal, c)}", ""]
    items = [
        f"{ln.description} | {ln.quantity} | {money(ln.unit_price, c)} | {money(ln.amount, c)}"
        for ln in inv.lines
    ]
    totals = _totals_block(inv)
    if inv.layout == "table":
        body = [
            kind,
            seller,
            f"{kind.title()} no. {inv.invoice_number}",
            f"Date: {issued}",
            f"Bill to: {inv.customer} ({inv.country})",
            "",
            *head,
            "Item | Qty | Unit price | Amount",
            *items,
            "",
            *totals,
            "",
            due if d_due or inv.document_type == "credit_note" else "Thank you for your business.",
        ]
    elif inv.layout == "email":
        body = [
            "---------- Forwarded message ----------",
            f"From: accounts@{'kestrel-' + ('jp' if c == 'JPY' else 'ca' if c == 'CAD' else 'outfitters')}.example",
            f"Subject: Your {kind.lower()} {inv.invoice_number}",
            "",
            f"Hi {inv.customer.split()[0]},",
            "",
            f"Please find your {kind.lower()} below. Reference {inv.invoice_number}, issued {issued}.",
            "",
            *head,
            *items,
            "",
            *totals,
            "",
            due if d_due or inv.document_type == "credit_note" else "",
            "Kind regards,",
            f"Accounts, {seller}",
        ]
    elif inv.layout == "ocr":

        def noisy(s: str) -> str:  # scanner noise in the words; numbers stay intact
            return (
                s.replace("  ", " ")
                .replace("Subtotal", "Sub total")
                .replace("Item", "ltem")
                .replace("Shipping", "Shippng")
            )

        body = [
            f"{kind}   {seller}",
            f"No.  {inv.invoice_number}     Date   {issued}",
            f"Customer   {inv.customer}  /  {inv.country}",
            "",
            *head,
            *[noisy(x).replace(" | ", "    ") for x in items],
            "",
            *[noisy(x) for x in totals],
            due if d_due or inv.document_type == "credit_note" else "",
        ]
    else:  # prose
        listing = "; ".join(
            f"{ln.quantity} x {ln.description} at {money(ln.unit_price, c)} ({money(ln.amount, c)})"
            for ln in inv.lines
        )
        body = [
            f"{seller}: {kind.lower()} {inv.invoice_number}, issued {issued} to {inv.customer} ({inv.country}).",
            "",
            *head,
            f"This {kind.lower()} covers {listing}.",
            " ".join(t + "." for t in totals),
            due if d_due or inv.document_type == "credit_note" else "",
        ]
    return "\n".join(body).strip() + "\n"


# ─── Building ─────────────────────────────────────────────────────────────────


def dataset() -> KestrelData:
    return KestrelData(generate(SEED, SCALE))


def build(d: KestrelData | None = None, n_per_trap: int = 15) -> list[Invoice]:
    d = d or dataset()
    rng = random.Random(SEED + 7)
    products = {p.sku: p.name for p in d.dataset.products}
    orders = sorted((o for o in d.dataset.orders if 1 <= len(o.lines) <= 4), key=lambda o: o.id)
    used: set[str] = set()
    out: list[Invoice] = []
    for t_i, trap in enumerate(TRAPS):
        for k in range(n_per_trap):

            def ok(o, trap=trap):
                if o.id in used:
                    return False
                placed = o.placed_at.date()
                if trap == "date":  # an ambiguous day/month on a UK invoice
                    return o.currency == "GBP" and placed.day <= 12 and placed.day != placed.month
                if trap == "currency":
                    return o.currency == "USD"
                return True

            order = rng.choice([o for o in orders if ok(o)])
            used.add(order.id)
            customer = d.customers[order.customer_id]
            currency = order.currency
            country = (
                {"USD": "US", "GBP": "GB", "EUR": "DE"}[currency]
                if customer.country not in ("US", "GB")
                else customer.country
            )
            if trap == "currency":
                currency, country = rng.choice(["JPY", "CAD"]), "JP"
                country = "JP" if currency == "JPY" else "CA"
            fx = RATE.get(currency, 1.0)
            dec = 0 if currency == "JPY" else 2
            lines = []
            for ln in order.lines:
                unit = round(ln.unit_price_cents / 100 * fx, dec)
                lines.append(
                    Line(products[ln.sku], ln.quantity, unit, round(unit * ln.quantity, dec))
                )
            subtotal = round(sum(ln.amount for ln in lines), dec)
            discount = round(subtotal * 0.10, dec) if trap == "discount" else 0.0
            tax = round((subtotal - discount) * TAX[currency][1], dec)
            shipping = SHIPPING[currency]
            total = round(subtotal - discount + tax + shipping, dec)
            d_invoice = order.placed_at.date()
            d_due = (
                None if trap in ("no_due_date", "credit_note") else d_invoice + timedelta(days=30)
            )
            inv = Invoice(
                id=f"inv-{len(out) + 1:03d}",
                trap=trap,
                layout=LAYOUTS[(t_i + k) % len(LAYOUTS)],
                invoice_number=f"{'CN' if trap == 'credit_note' else 'INV'}-{d_invoice.year}-{order.id[2:]}",
                document_type="credit_note" if trap == "credit_note" else "invoice",
                invoice_date=d_invoice.isoformat(),
                due_date=d_due.isoformat() if d_due else None,
                currency=currency,
                subtotal=subtotal,
                discount=discount,
                tax=tax,
                shipping=shipping,
                total=total,
                customer=customer.name,
                country=country,
                lines=tuple(lines),
            )
            out.append(Invoice(**{**inv.__dict__, "text": render(inv, d_invoice, d_due)}))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write the invoices to invoices/")
    args = parser.parse_args(argv)
    invoices = build()
    print(f"{len(invoices)} invoices")
    if args.write:
        folder = ROOT / "invoices"
        folder.mkdir(exist_ok=True)
        for inv in invoices:
            (folder / f"{inv.id}-{inv.trap}.txt").write_text(inv.text)
        print(f"written to {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
