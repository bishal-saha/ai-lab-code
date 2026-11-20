"""The corpus is deterministic, and every question's answer sits where it says."""

import re
from collections import Counter

from cut_here.corpus import build, dataset


def test_corpus_is_deterministic(docs):
    again, _, _ = build(dataset())
    assert [(d.id, d.text) for d in again] == [(d.id, d.text) for d in docs]


def test_question_mix_and_spans(docs, questions):
    by_id = {d.id: d for d in docs}
    assert Counter(q.qtype for q in questions) == {"table": 120, "heading": 32, "prose": 52}
    assert len({q.id for q in questions}) == len(questions)
    for q in questions:
        span = by_id[q.doc].text[q.start : q.end]
        assert q.answer in span, (q.id, span)


def test_heading_questions_sit_under_their_carrier_and_region(docs, questions):
    text = next(d for d in docs if d.id == "shipping-rates.md").text

    def nearest(level, pos):
        return re.findall(rf"^{'#' * level} (.+)$", text[:pos], re.M)[-1]

    for q in (q for q in questions if q.qtype == "heading"):
        assert (nearest(2, q.start), nearest(3, q.start)) == q.context


def test_table_context_is_absent_from_the_row_itself(docs, questions):
    """The disambiguating context lives outside the row: that is what makes these questions hard."""
    by_id = {d.id: d for d in docs}
    for q in (q for q in questions if q.qtype in ("table", "heading")):
        row = by_id[q.doc].text[q.start : q.end]
        assert not all(c in row for c in q.context), q.id
