"""Embeddings: Voyage (voyage-4-lite, 1,024 dimensions), cached on disk; and a fake for offline runs.

The cache (`cache/embeddings.npz`) is keyed by input type and a hash of the text and committed with
the results, so a rerun of the studies needs no Voyage key. The fake embedder is a hashing-trick
bag of words: similar texts land close together, which is enough for the tests and the dry run.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import httpx
import numpy as np

MODEL = "voyage-4-lite"
DIM = 1024
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache" / "embeddings.npz"


def _key(text: str, input_type: str) -> str:
    return f"{input_type}:{hashlib.sha256(text.encode()).hexdigest()[:24]}"


def fake_embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            out[i, h % DIM] += 1.0 if (h >> 20) & 1 else -1.0
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    return out


class Embedder:
    def __init__(self, *, fake: bool = False):
        self.fake = fake
        self.tokens = 0
        self.cache: dict[str, np.ndarray] = {}
        if not fake and CACHE.exists():
            data = np.load(CACHE)
            self.cache = {k: data[k] for k in data.files}

    def embed(self, texts: list[str], input_type: str) -> np.ndarray:
        if self.fake:
            return fake_embed(texts)
        missing = [t for t in dict.fromkeys(texts) if _key(t, input_type) not in self.cache]
        if missing:
            key = os.environ["VOYAGE_API_KEY"]
            base = os.environ.get("VOYAGE_BASE_URL", "https://api.voyageai.com/v1").rstrip("/")
            with httpx.Client(timeout=120) as http:
                for i in range(0, len(missing), 128):
                    batch = missing[i : i + 128]
                    r = http.post(
                        f"{base}/embeddings",
                        headers={"Authorization": f"Bearer {key}"},
                        json={"input": batch, "model": MODEL, "input_type": input_type},
                    )
                    r.raise_for_status()
                    body = r.json()
                    self.tokens += body["usage"]["total_tokens"]
                    for t, d in zip(
                        batch, sorted(body["data"], key=lambda x: x["index"]), strict=True
                    ):
                        self.cache[_key(t, input_type)] = np.asarray(
                            d["embedding"], dtype=np.float32
                        )
            CACHE.parent.mkdir(exist_ok=True)
            np.savez_compressed(CACHE, **self.cache)
        return np.stack([self.cache[_key(t, input_type)] for t in texts])
