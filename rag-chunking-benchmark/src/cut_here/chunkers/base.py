"""What every chunker returns, and the token counter they all share.

A chunk keeps the source spans it covers (offsets into its document), the text a
reader sees, and the text that gets embedded and indexed (`embed_text`), which a
strategy may prefix with a heading path or an LLM-written context. Grading checks
spans, so a chunk only "has" an answer if it really covers that part of the source.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from tokenizers import Tokenizer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    spans: tuple[tuple[int, int], ...]  # source offsets covered, in order
    text: str  # what a reader of the chunk sees
    embed_text: str  # what is embedded and indexed (text, maybe with a prefix)

    def covers(self, start: int, end: int) -> bool:
        return any(s <= start and end <= e for s, e in self.spans)


@cache
def tokenizer() -> Tokenizer:
    """The embedding model's own tokenizer (a private copy, so turning truncation off for
    counting never changes how the embedder truncates)."""
    from fastembed import TextEmbedding

    shared = TextEmbedding(EMBED_MODEL).model.tokenizer
    tok = Tokenizer.from_str(shared.to_str())
    tok.no_truncation()
    tok.no_padding()
    return tok


def count(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False).ids)


def make(
    strategy: str, doc: str, n: int, spans: list[tuple[int, int]], source: str, prefix: str = ""
) -> Chunk:
    text = "\n".join(source[s:e] for s, e in spans)
    return Chunk(
        f"{strategy}:{doc}:{n}", doc, tuple(spans), text, f"{prefix}{text}" if prefix else text
    )
