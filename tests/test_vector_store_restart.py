"""Restart behaviour for app/services/rag/vector_store.py: embedding rebuild."""

from __future__ import annotations

import numpy as np

from app.services.rag.vector_store import VectorStore
from tests.test_vector_store_behaviour import _save_store, _vec


class TestEnsureEmbeddings:
    def test_restarted_store_starts_without_embeddings(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)

        reloaded = VectorStore(4, persist_dir=persist, embedding_model="model-a")

        assert reloaded.size == 3
        assert reloaded._embeddings == []

    def test_ensure_embeddings_rebuilds_one_row_per_chunk(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)
        reloaded = VectorStore(4, persist_dir=persist, embedding_model="model-a")

        reloaded._ensure_embeddings()

        assert len(reloaded._embeddings) == 3
        for i, row in enumerate(reloaded._embeddings):
            assert np.allclose(row, _vec(i))

    def test_ensure_embeddings_is_a_noop_when_already_built(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)
        reloaded = VectorStore(4, persist_dir=persist, embedding_model="model-a")
        reloaded._ensure_embeddings()
        first = reloaded._embeddings

        reloaded._ensure_embeddings()

        assert reloaded._embeddings is first
