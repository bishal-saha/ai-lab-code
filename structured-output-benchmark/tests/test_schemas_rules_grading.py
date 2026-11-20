"""Schemas accept right answers, rules catch inconsistent ones, the grader catches wrong values."""

import asyncio

import pytest
from right_shape import rules
from right_shape.grading import fields
from right_shape.models import FakeModel, mistaken_record, oracle_record
from right_shape.run import CONDITIONS, run_one
from right_shape.schemas import BASE, DESIGNED, DESIGNED_NO_NOTES, parse, schema_errors


@pytest.mark.parametrize("schema", [BASE, DESIGNED, DESIGNED_NO_NOTES])
def test_oracle_records_are_schema_valid_and_pass_the_rules(invoices, schema):
    for inv in invoices:
        rec = oracle_record(inv, schema)
        assert schema_errors(rec, schema) == [], inv.id
        assert rules.check(rec) == [], inv.id


def test_designed_schema_shape():
    assert (
        list(DESIGNED["properties"])[0] == "notes"
        and "notes" not in DESIGNED_NO_NOTES["properties"]
    )
    assert (
        "OTHER" in DESIGNED["properties"]["currency"]["enum"]
        and "OTHER" not in BASE["properties"]["currency"]["enum"]
    )
    assert "due_date" in DESIGNED["required"] and "due_date" not in BASE["required"]
    assert schema_errors({**oracle_record_stub(), "due_date": None}, DESIGNED) == []


def oracle_record_stub():
    return {
        "notes": "",
        "invoice_number": "X",
        "document_type": "invoice",
        "invoice_date": "2025-01-01",
        "due_date": None,
        "currency": "USD",
        "subtotal": 1.0,
        "discount": 0.0,
        "tax": 0.0,
        "shipping": 0.0,
        "total": 1.0,
        "line_items": [{"description": "a", "quantity": 1, "unit_price": 1.0, "amount": 1.0}],
    }


def test_rules_flag_each_arithmetic_mistake_and_pass_clean(invoices):
    for inv in invoices:
        if inv.trap in ("subtotal", "discount", "words"):
            assert rules.check(mistaken_record(inv, DESIGNED)), inv.id
    rec = oracle_record_stub()
    assert rules.check(rec) == []
    assert any(
        "before the invoice date" in e for e in rules.check({**rec, "due_date": "2024-12-01"})
    )
    assert any(
        "line 1" in e
        for e in rules.check(
            {
                **rec,
                "line_items": [
                    {"description": "a", "quantity": 2, "unit_price": 1.0, "amount": 1.0}
                ],
            }
        )
    )


def test_grader_catches_a_transposed_total_and_a_fabricated_due_date(invoices):
    inv = next(i for i in invoices if i.trap == "words")
    assert not fields(inv, mistaken_record(inv, DESIGNED))["total"]
    nodue = next(i for i in invoices if i.trap == "no_due_date")
    assert fields(nodue, oracle_record(nodue, DESIGNED))["due_date"]  # null is right
    assert fields(nodue, oracle_record(nodue, BASE))["due_date"]  # omitted is right
    assert not fields(nodue, mistaken_record(nodue, DESIGNED))[
        "due_date"
    ]  # an invented date is wrong
    cur = next(i for i in invoices if i.trap == "currency")
    assert (
        fields(cur, oracle_record(cur, DESIGNED))["currency"]
        and not fields(cur, oracle_record(cur, BASE))["currency"]
    )


def test_defensive_parse():
    assert parse('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse('Here you go: {"a": 1} hope that helps') == {"a": 1}
    assert parse("no json here") is None


def test_validated_condition_retries_and_repairs(invoices):
    inv = next(i for i in invoices if i.trap == "subtotal")
    row = asyncio.run(run_one(inv, "so+d+v", FakeModel("mistake")))
    assert (
        row["retried"]
        and row["rule_errors"]
        and not row["first_all_correct"]
        and row["all_correct"]
    )
    plain = asyncio.run(run_one(inv, "so+d", FakeModel("mistake")))
    assert not plain["retried"] and plain["valid_but_wrong"]
    assert set(CONDITIONS) == {"prompt", "tool", "strict", "so", "so+d", "so+d-n", "so+d+v"}
