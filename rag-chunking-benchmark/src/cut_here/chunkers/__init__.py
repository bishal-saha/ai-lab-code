"""The eight chunking strategies, by id. Each takes the corpus and returns chunks."""

from __future__ import annotations

from collections.abc import Callable

from cut_here.chunkers.base import Chunk
from cut_here.chunkers.fixed import fixed
from cut_here.chunkers.recursive import recursive
from cut_here.chunkers.semantic import semantic
from cut_here.chunkers.structure import structure
from cut_here.corpus import Doc

SECTION_LIMIT = 400  # tokens; leaves room for a heading path or context under the embedder's 512

STRATEGIES: dict[str, Callable[[list[Doc]], list[Chunk]]] = {
    "F256": lambda docs: fixed(docs, 256, 0, "F256"),
    "F256o": lambda docs: fixed(docs, 256, 128, "F256o"),
    "F512o": lambda docs: fixed(
        docs, 510, 77, "F512o"
    ),  # 510: the embedder's 512 minus its two special tokens
    "R512": lambda docs: recursive(docs, 510, "R512"),
    "S": lambda docs: structure(docs, SECTION_LIMIT, "S", heading_path=False),
    "S+H": lambda docs: structure(docs, SECTION_LIMIT, "S+H", heading_path=True),
    "SEM": lambda docs: semantic(docs, SECTION_LIMIT, "SEM"),
}
CONTEXTUAL = "S+H+C"  # S+H plus an LLM-written context per chunk: built in contextual.py (paid)
ALL = [*STRATEGIES, CONTEXTUAL]
