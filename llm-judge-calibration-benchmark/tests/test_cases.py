"""Labels follow construction: every fail reply carries its defect, no pass reply does."""

import re
from collections import Counter
from datetime import date

from judge_the_judge.cases import LIMIT, WINDOW, build


def _outcome(c):
    days = (date.fromisoformat(c.today) - date.fromisoformat(c.delivered)).days
    if days > WINDOW[c.customer_type]:
        return "decline"
    return "refund" if c.total <= LIMIT else "escalate"


def test_mix_and_determinism(cases, pairs):
    assert Counter(c.label for c in cases) == {"PASS": 160, "FAIL": 40}
    assert set(Counter(c.fail_type for c in cases if c.fail_type).values()) == {8}
    assert Counter(c.length for c in cases) == {"short": 100, "long": 100}
    assert Counter(p.kind for p in pairs) == {"correct_vs_wrong": 50, "short_vs_long": 50}
    again, _ = build()
    assert [c.reply for c in again] == [c.reply for c in cases]


def test_outcome_matches_the_order(cases):
    for c in cases:
        assert _outcome(c) == c.outcome, c.id


def test_pass_replies_state_the_outcome_and_nothing_more(cases):
    for c in (c for c in cases if c.label == "PASS"):
        assert "credit" not in c.reply.lower()
        if c.outcome == "refund":
            assert f"USD {c.total:.2f}" in c.reply and "refunded" in c.reply
        if c.outcome == "escalate":
            assert "Finance" in c.reply and "refunded" not in c.reply
        if c.outcome == "decline":
            assert "can't accept" in c.reply and f"{WINDOW[c.customer_type]}-day" in c.reply


def test_fail_replies_carry_their_defect(cases):
    for c in (c for c in cases if c.label == "FAIL"):
        amounts = [float(a.replace(",", "")) for a in re.findall(r"USD ([\d,]+\.\d\d)", c.reply)]
        if c.fail_type == "wrong_amount":
            assert (
                c.outcome == "refund" and amounts and all(abs(a - c.total) > 0.005 for a in amounts)
            )
        if c.fail_type in ("wrong_outcome", "wrong_window"):
            assert c.outcome == "decline" and "refunded" in c.reply
        if c.fail_type == "wrong_window":
            assert c.customer_type == "trade" and "30-day" in c.reply
        if c.fail_type == "extra_promise":
            assert "USD 40 credit" in c.reply
        if c.fail_type == "missing_fact":
            assert (
                c.outcome == "escalate"
                and "Finance" not in c.reply
                and "refund" not in c.reply.lower()
            )


def test_long_and_short_share_the_decision(pairs):
    for p in (p for p in pairs if p.kind == "short_vs_long"):
        assert len(p.b.reply) > 2 * len(p.a.reply)
        decisive = [
            s for s in p.a.reply.split(". ") if "USD" in s or "window" in s or "Finance" in s
        ]
        assert decisive and all(s.split(",")[-1].strip()[:30] in p.b.reply for s in decisive)
