"""Grading, from offsets and strings only.

A retrieved chunk is a **hit** for a question when it covers the answer span *and*
its embedded text carries the question's context (the product, the warehouse, the
carrier and region). A chunk with the right row but no product name is not an
answer a reader can use: that is the failure the article describes.

    rank          1-based position of the first hit in the top 10, or None
    span_rank     1-based position of the first chunk that covers the span, context or not
    miss reason   "no span" (the passage never came back) or "no context" (it came back
                  without what disambiguates it)
"""

from __future__ import annotations

import re

from cut_here.chunkers.base import Chunk
from cut_here.questions import Question


def hit(chunk: Chunk, q: Question) -> bool:
    return (
        chunk.doc == q.doc
        and chunk.covers(q.start, q.end)
        and all(c in chunk.embed_text for c in q.context)
    )


def first_rank(chunks: list[Chunk], q: Question, *, need_context: bool = True) -> int | None:
    for i, c in enumerate(chunks, start=1):
        if c.doc == q.doc and c.covers(q.start, q.end) and (not need_context or hit(c, q)):
            return i
    return None


def miss_reason(rank: int | None, span_rank: int | None, k: int) -> str | None:
    if rank is not None and rank <= k:
        return None
    return "no context" if span_rank is not None and span_rank <= k else "no span"


def normalise(text: str) -> str:
    text = text.lower().replace("–", "-").replace("—", "-")
    return re.sub(r"[^a-z0-9.\-() ]+", " ", re.sub(r"\s+", " ", text)).strip()


def answer_correct(expected: str, given: str) -> bool:
    """Exact match after normalising: the expected value must appear as written in the answer,
    as a whole token run ("3 years" does not match "13 years")."""
    pattern = r"(?<![a-z0-9])" + re.escape(normalise(expected)) + r"(?![a-z0-9])"
    return re.search(pattern, normalise(given)) is not None
