"""Contextual retrieval (S+H+C): S+H chunks with a short LLM-written context prepended.

As Anthropic describes it, a model reads the whole document and writes a sentence or
two situating each chunk; the context is prepended before embedding and BM25. The
cost lands at index time: one model call per chunk. Contexts are cached in
cache/contexts.jsonl (keyed by document and chunk text), so a rerun is free.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from cut_here.chunkers.base import Chunk
from cut_here.corpus import Doc

NAME = "S+H+C"


def key(chunk: Chunk) -> str:
    return hashlib.sha256(f"{chunk.doc}\n{chunk.text}".encode()).hexdigest()[:16]


def load(cache: Path) -> dict[str, dict]:
    if not cache.exists():
        return {}
    rows = (json.loads(line) for line in cache.read_text().splitlines() if line.strip())
    return {r["key"]: r for r in rows}


async def contextualize(
    docs: list[Doc],
    chunks: list[Chunk],
    model,
    cache: Path,
    *,
    max_usd: float,
    concurrency: int = 8,
) -> tuple[float, int]:
    """Write a context for every chunk not yet in the cache. Returns (USD spent, chunks done)."""
    from cut_here.models import Usage

    done = load(cache)
    todo = [c for c in chunks if key(c) not in done]
    by_doc = {d.id: d.text for d in docs}
    usage, spent, gate = Usage(), 0.0, asyncio.Semaphore(concurrency)
    cache.parent.mkdir(parents=True, exist_ok=True)

    async def one(c: Chunk) -> None:
        nonlocal spent
        async with gate:
            if spent >= max_usd:
                return
            text, u = await model.context(by_doc[c.doc], c.text)
            spent += usage.add(model.name, u)
            with cache.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "key": key(c),
                            "doc": c.doc,
                            "context": text.strip(),
                            "model": model.name,
                            **u,
                        }
                    )
                    + "\n"
                )

    await asyncio.gather(*(one(c) for c in todo))
    return spent, len(todo)


def apply(chunks: list[Chunk], cache: Path) -> list[Chunk]:
    """S+H chunks -> S+H+C chunks. Raises if a context is missing (run --contextualize first)."""
    contexts = load(cache)
    missing = [c.id for c in chunks if key(c) not in contexts]
    if missing:
        raise LookupError(f"{len(missing)} chunks have no context yet; run --contextualize")
    return [
        Chunk(
            c.id.replace("S+H:", f"{NAME}:", 1),
            c.doc,
            c.spans,
            c.text,
            f"{contexts[key(c)]['context']}\n\n{c.embed_text}",
        )
        for c in chunks
    ]
