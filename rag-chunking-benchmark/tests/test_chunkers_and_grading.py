"""Chunkers keep their offsets honest; S keeps table headers; S+H adds the path; the grader is strict."""

import pytest
from cut_here.chunkers import STRATEGIES
from cut_here.chunkers.base import Chunk, count
from cut_here.chunkers.structure import structure
from cut_here.corpus import Doc
from cut_here.grading import answer_correct, first_rank, hit


@pytest.mark.parametrize("name", [n for n in STRATEGIES if n != "SEM"])
def test_chunk_text_is_its_source_spans(docs, name):
    by_id = {d.id: d for d in docs}
    for c in STRATEGIES[name](docs):
        assert c.text == "\n".join(by_id[c.doc].text[a:b] for a, b in c.spans)


def test_fixed_overlap_repeats_tokens_and_fixed_covers_the_whole_doc(docs):
    long_doc = [d for d in docs if d.id.startswith("warehouses/")]
    plain, overlap = STRATEGIES["F256"](long_doc), STRATEGIES["F256o"](long_doc)
    assert len(overlap) > len(plain)
    for d in long_doc:
        spans = sorted(s for c in plain if c.doc == d.id for s in c.spans)
        assert spans[0][0] == 0 and spans[-1][1] >= len(d.text.rstrip()) - 1


def test_structure_repeats_the_table_header_in_every_row_piece():
    rows = "\n".join(f"| SKU-{i:05d} | Product {i} | {i} |" for i in range(300))
    doc = Doc(
        "w.md",
        "W",
        f"# W warehouse\n\n## Stock levels\n\n| SKU | Product | Units on hand |\n|---|---|---|\n{rows}\n",
    )
    pieces = [c for c in structure([doc], 400, "S", heading_path=False) if "SKU-0" in c.text]
    assert len(pieces) > 1
    assert all(
        c.text.splitlines()[0 if not c.text.startswith("##") else 1].startswith("| SKU | Product")
        for c in pieces
    )
    assert all(count(c.text) <= 400 for c in pieces)


def test_heading_path_prefix(docs):
    sh = [
        c
        for c in STRATEGIES["S+H"](docs)
        if c.doc == "shipping-rates.md" and "### Europe" in c.text
    ]
    assert sh and all(
        c.embed_text.startswith("Shipping rates and delivery times > ")
        and " > Europe\n\n" in c.embed_text
        for c in sh
    )
    plain = [
        c for c in STRATEGIES["S"](docs) if c.doc == "shipping-rates.md" and "### Europe" in c.text
    ]
    assert all(c.embed_text == c.text for c in plain)


def test_grader_rejects_the_right_row_without_its_context(docs, questions):
    q = next(q for q in questions if q.qtype == "table" and q.doc.startswith("products/"))
    row_only = Chunk("x", q.doc, ((q.start, q.end),), "row", "| Warranty | 3 years |")
    with_name = Chunk(
        "y", q.doc, ((q.start, q.end),), "row", f"{q.context[0]} > Specifications\n\nrow"
    )
    other_doc = Chunk("z", "products/other.md", ((q.start, q.end),), "row", f"{q.context[0]}")
    assert not hit(row_only, q) and hit(with_name, q) and not hit(other_doc, q)
    assert first_rank([row_only, with_name], q) == 2
    assert first_rank([row_only, with_name], q, need_context=False) == 1


def test_answer_matching():
    assert answer_correct("2–3 business days", "It takes 2-3 business days.")
    assert answer_correct("USD 14.95", "usd 14.95")
    assert not answer_correct("3 years", "13 years")
    assert answer_correct("3 years", "The warranty is 3 years.")
    assert not answer_correct("IPX4", "not found")
