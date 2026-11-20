"""Recursive character splitting: try paragraph breaks, then line breaks, then sentences, then
spaces, until every piece fits under the limit; then merge neighbours back up to it.

The usual upgrade on fixed-size: prose comes out better, but a table row is just another line.
"""

from __future__ import annotations

from cut_here.chunkers.base import Chunk, count, make
from cut_here.corpus import Doc

SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split(text: str, start: int, end: int, limit: int, seps: list[str]) -> list[tuple[int, int]]:
    if count(text[start:end]) <= limit or not seps:
        return [(start, end)]
    sep, rest = seps[0], seps[1:]
    if sep not in text[start:end]:
        return _split(text, start, end, limit, rest)
    pieces, at = [], start
    while True:
        i = text.find(sep, at, end)
        if i < 0:
            pieces.append((at, end))
            break
        pieces.append((at, i + len(sep)))  # the separator stays with the piece before it
        at = i + len(sep)
    parts = [p for a, b in pieces if b > a for p in _split(text, a, b, limit, rest)]
    merged: list[tuple[int, int]] = []
    for a, b in parts:
        if merged and count(text[merged[-1][0] : b]) <= limit:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def recursive(docs: list[Doc], limit: int, name: str) -> list[Chunk]:
    out = []
    for doc in docs:
        spans = [
            (a, b)
            for a, b in _split(doc.text, 0, len(doc.text), limit, SEPARATORS)
            if doc.text[a:b].strip()
        ]
        out += [make(name, doc.id, n, [span], doc.text) for n, span in enumerate(spans)]
    return out
