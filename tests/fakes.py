"""Shared fakes for the RAG evaluation tests (deterministic, no network)."""

from __future__ import annotations

import hashlib

import numpy as np


def unit_vector(text: str, dim: int) -> np.ndarray:
    """Deterministic L2-normalized vector derived from *text* content."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


class FakeEmbedder:
    """Stand-in for ``EmbeddingClient``: content-hashed unit vectors.

    Like the real client, output rows are L2-normalized, so inner products in
    the flat index equal cosine similarity and a chunk scores 1.0 against its
    own query vector. Records every call so tests can assert batching.
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.vstack([unit_vector(t, self.dim) for t in texts]).astype(np.float32)
