"""Three retrievers over one set of chunks: dense, BM25, and a hybrid of the two.

    dense   BAAI/bge-small-en-v1.5 through fastembed (ONNX, CPU, free to rerun); cosine on
            normalised vectors, queries embedded with the model's query instruction
    bm25    rank-bm25 over lower-cased word tokens
    hybrid  reciprocal-rank fusion of the two (k = 60)

Each returns chunk indices, best first.
"""

from __future__ import annotations

import re
from functools import cache

import numpy as np
from rank_bm25 import BM25Okapi

from cut_here.chunkers.base import EMBED_MODEL, Chunk

RETRIEVERS = ["dense", "bm25", "hybrid"]
RRF_K = 60
WORD = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")


@cache
def embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(EMBED_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    vectors = np.array(list(embedder().embed(texts, batch_size=64)), dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def embed_queries(texts: list[str]) -> np.ndarray:
    vectors = np.array(list(embedder().query_embed(texts)), dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


class Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectors = embed([c.embed_text for c in chunks])
        self.bm25 = BM25Okapi([words(c.embed_text) for c in chunks])

    def search(self, queries: list[str], retriever: str, k: int = 10) -> list[list[int]]:
        dense = bm25 = None
        if retriever in ("dense", "hybrid"):
            dense = [list(np.argsort(-row)) for row in embed_queries(queries) @ self.vectors.T]
        if retriever in ("bm25", "hybrid"):
            bm25 = [list(np.argsort(-self.bm25.get_scores(words(q)))) for q in queries]
        if retriever == "dense":
            return [r[:k] for r in dense]
        if retriever == "bm25":
            return [r[:k] for r in bm25]
        fused = []
        for d, b in zip(dense, bm25, strict=True):
            score: dict[int, float] = {}
            for ranking in (d, b):
                for rank, i in enumerate(ranking[:100]):
                    score[i] = score.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
            fused.append(sorted(score, key=lambda i: -score[i])[:k])
        return fused
