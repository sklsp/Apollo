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


class TestTwoRestartCycles:
    def test_two_full_restart_cycles_keep_the_store_consistent(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)

        s1 = VectorStore(4, persist_dir=persist, embedding_model="model-a")
        assert s1.remove_by_doc_id("doc_a") == 2
        s1.save()

        s2 = VectorStore(4, persist_dir=persist, embedding_model="model-a")
        s2.add(np.vstack([_vec(3)]), [_chunk("doc_c", 0)])
        assert s2.remove_by_doc_id("doc_b") == 1
        s2.save()

        s3 = VectorStore(4, persist_dir=persist, embedding_model="model-a")
        assert s3.size == 1
        assert s3.indexed_doc_ids == {"doc_c"}
        results = s3.search(_vec(3), top_k=5)
        assert [r.doc_id for r in results] == ["doc_c"]


class TestEveryPublicMethodAfterRestart:
    def test_size(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)
        store = VectorStore(4, persist_dir=persist, embedding_model="model-a")
        assert store.size == 3

    def test_indexed_doc_ids(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)
        store = VectorStore(4, persist_dir=persist, embedding_model="model-a")
        assert store.indexed_doc_ids == {"doc_a", "doc_b"}

    def test_search(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)
        store = VectorStore(4, persist_dir=persist, embedding_model="model-a")
        results = store.search(_vec(0), top_k=1)
        assert [r.doc_id for r in results] == ["doc_a"]

    def test_add(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)
        store = VectorStore(4, persist_dir=persist, embedding_model="model-a")
        store.add(np.vstack([_vec(3)]), [_chunk("doc_c", 0)])
        assert store.size == 4

    def test_remove_by_doc_id(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)
        store = VectorStore(4, persist_dir=persist, embedding_model="model-a")
        assert store.remove_by_doc_id("doc_a") == 2

    def test_clear(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)
        store = VectorStore(4, persist_dir=persist, embedding_model="model-a")
        store.clear()
        assert store.size == 0

    def test_save(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)
        store = VectorStore(4, persist_dir=persist, embedding_model="model-a")
        store.add(np.vstack([_vec(3)]), [_chunk("doc_c", 0)])
        store.save()
        reloaded = VectorStore(4, persist_dir=persist, embedding_model="model-a")
        assert reloaded.size == 4
