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


class FakeServiceEmbedder(FakeEmbedder):
    """``FakeEmbedder`` shaped like ``EmbeddingClient`` for ``RAGService``.

    Adds the attributes the service reads (``model``, ``backend``,
    ``dimension``, single-text ``embed``); vectors are the same deterministic
    sha256-seeded unit vectors, so rankings match what a bare store would give.
    """

    model = "fake-embed"
    backend = "fake"

    @property
    def dimension(self) -> int:
        return self.dim

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]


class MarkerServiceEmbedder:
    """Two orthogonal basis vectors shaped like ``EmbeddingClient``.

    Text mentioning "alpha" gets the first basis vector, everything else the
    second - so a query about alpha ranks every alpha chunk at exactly 1.0 and
    every other chunk at exactly 0.0 (same marker rule as the store-level
    ``MarkerEmbedder`` in ``tests/test_eval_runner.py``, plus the attributes
    ``RAGService`` reads).
    """

    model = "marker-embed"
    backend = "fake"

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.v_alpha = np.zeros(dim, dtype=np.float32)
        self.v_alpha[0] = 1.0
        self.v_beta = np.zeros(dim, dtype=np.float32)
        self.v_beta[1] = 1.0

    @property
    def dimension(self) -> int:
        return self.dim

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.vstack(
            [self.v_alpha if "alpha" in t.lower() else self.v_beta for t in texts]
        )
