"""Semantic chunking: embed each sentence (and each table row), then start a new chunk
wherever the similarity between neighbours drops into the doc's lowest quartile, or the
chunk would pass the size limit.

It finds topic shifts without any notion of a heading, a table header or a code fence,
and it embeds every sentence of the corpus at index time.
"""

from __future__ import annotations

import re

import numpy as np

from cut_here.chunkers.base import Chunk, count, make
from cut_here.corpus import Doc

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def units(text: str) -> list[tuple[int, int]]:
    """Sentences of prose and single lines of tables and headings, as spans."""
    out, pos = [], 0
    for line in text.splitlines(keepends=True):
        start, body = pos, line.rstrip("\n")
        pos += len(line)
        if not body.strip():
            continue
        if body.lstrip().startswith(("|", "#")):
            out.append((start, start + len(body)))
            continue
        at = start
        for part in SENTENCE_END.split(body):
            i = text.index(part, at)
            out.append((i, i + len(part)))
            at = i + len(part)
    return out


def semantic(docs: list[Doc], limit: int, name: str, percentile: float = 25.0) -> list[Chunk]:
    from cut_here.index import embed

    out = []
    for doc in docs:
        spans = units(doc.text)
        vectors = embed([doc.text[a:b] for a, b in spans])
        sims = (vectors[:-1] * vectors[1:]).sum(axis=1)
        cut_below = np.percentile(sims, percentile) if len(sims) else 0.0
        groups, current = [], [spans[0]]
        for (a, b), sim in zip(spans[1:], sims, strict=True):
            grown = doc.text[current[0][0] : b]
            if sim < cut_below or count(grown) > limit:
                groups.append(current)
                current = []
            current.append((a, b))
        groups.append(current)
        out += [
            make(name, doc.id, n, [(g[0][0], g[-1][1])], doc.text) for n, g in enumerate(groups)
        ]
    return out
