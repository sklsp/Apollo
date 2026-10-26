"""Restart behaviour for app/services/rag/vector_store.py: embedding rebuild."""

from __future__ import annotations

import numpy as np

from app.services.rag.vector_store import VectorStore
from tests.test_vector_store_behaviour import _chunk, _save_store, _vec


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


class TestRemoveAfterRestart:
    def test_remove_then_second_restart_keeps_the_survivor(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)
        reloaded = VectorStore(4, persist_dir=persist, embedding_model="model-a")

        removed = reloaded.remove_by_doc_id("doc_a")

        assert removed == 2
        assert reloaded.size == 1
        results = reloaded.search(_vec(2), top_k=5)
        assert [r.doc_id for r in results] == ["doc_b"]

        reloaded.save()
        again = VectorStore(4, persist_dir=persist, embedding_model="model-a")
        assert again.size == 1


class TestAddAfterRestart:
    def test_add_then_remove_keeps_chunks_and_embeddings_aligned(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)
        store = VectorStore(4, persist_dir=persist, embedding_model="model-a")

        store.add(np.vstack([_vec(3)]), [_chunk("doc_new", 0)])
        removed = store.remove_by_doc_id("doc_a")

        assert removed == 2
        assert store.size == 2
        results = store.search(_vec(3), top_k=5)
        assert results[0].doc_id == "doc_new"
        assert len(store._embeddings) == len(store._chunks)
