"""The invoices are deterministic, add up, and hold their own answers."""

from collections import Counter

from right_shape.invoices import amount_in_words, build, dataset, money, words


def test_deterministic_and_balanced(invoices):
    again = build(dataset())
    assert [i.text for i in again] == [i.text for i in invoices]
    assert set(Counter(i.trap for i in invoices).values()) == {15}
    assert len({i.invoice_number for i in invoices}) == len(invoices)


def test_every_invoice_adds_up(invoices):
    for inv in invoices:
        tol = 0.5 if inv.currency == "JPY" else 0.011
        assert abs(sum(ln.amount for ln in inv.lines) - inv.subtotal) <= tol * len(inv.lines)
        assert abs(inv.subtotal - inv.discount + inv.tax + inv.shipping - inv.total) <= tol * 3


def test_the_text_holds_the_answer(invoices):
    for inv in invoices:
        assert inv.invoice_number in inv.text
        shown = (
            amount_in_words(inv.total, inv.currency)
            if inv.trap == "words"
            else money(inv.total, inv.currency)
        )
        assert shown in inv.text, inv.id
        assert (inv.trap == "credit_note") == ("CREDIT NOTE" in inv.text.upper())


def test_traps_are_what_they_say(invoices):
    for inv in invoices:
        if inv.trap == "subtotal":
            assert inv.text.index("ORDER VALUE") < inv.text.index("Amount due")
        if inv.trap == "currency":
            assert inv.currency in ("JPY", "CAD")
        if inv.trap in ("no_due_date", "credit_note"):
            assert inv.due_date is None and "Payment due" not in inv.text
        if inv.trap == "date":
            day, month = int(inv.invoice_date[8:]), int(inv.invoice_date[5:7])
            assert inv.country == "GB" and day <= 12 and day != month
            assert f"{day:02d}/{month:02d}/{inv.invoice_date[:4]}" in inv.text
        if inv.trap == "words":
            assert money(inv.total, inv.currency) not in inv.text


def test_words():
    assert words(412) == "four hundred and twelve"
    assert words(1330) == "one thousand three hundred and thirty"
    assert amount_in_words(412.6, "GBP") == "four hundred and twelve pounds and sixty pence"
