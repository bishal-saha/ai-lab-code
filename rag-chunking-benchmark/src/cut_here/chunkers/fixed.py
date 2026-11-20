"""Fixed-size chunking: cut every `size` tokens, optionally repeating `overlap` tokens across each cut.

Deterministic and cheap; blind to sentences, sections and tables.
"""

from __future__ import annotations

from cut_here.chunkers.base import Chunk, make, tokenizer
from cut_here.corpus import Doc


def fixed(docs: list[Doc], size: int, overlap: int, name: str) -> list[Chunk]:
    out = []
    step = size - overlap
    for doc in docs:
        offsets = tokenizer().encode(doc.text, add_special_tokens=False).offsets
        for n, i in enumerate(range(0, len(offsets), step)):
            window = offsets[i : i + size]
            out.append(make(name, doc.id, n, [(window[0][0], window[-1][1])], doc.text))
            if i + size >= len(offsets):
                break
    return out
