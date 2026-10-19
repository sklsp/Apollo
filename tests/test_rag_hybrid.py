"""Hybrid retrieval service wiring (portfolio items 30-35).

Item 30 pins that a restarted ``RAGService`` rebuilds its BM25 keyword index
from every stored chunk, so keyword search works across restarts without any
separate persistence. Later items in this file cover the RAG_HYBRID setting,
fused queries and keeping the index in sync with indexing operations.
"""

from __future__ import annotations

import logging

import pytest

from app.services.rag.service import RAGService
from tests.fakes import FakeServiceEmbedder


@pytest.fixture
def persist_dir(tmp_path) -> str:
    return str(tmp_path / "rag")


class TestKeywordIndexStartup:
    def test_rebuilt_from_all_saved_chunks_after_restart(self, persist_dir):
        first = RAGService(embedding_client=FakeServiceEmbedder(), persist_dir=persist_dir)
        first.add_document(
            "The quarterly report covers revenue and gross margins.", {"doc_id": "report"}
        )
        first.add_document("Deployment runbook: restart the worker service.", {"doc_id": "ops"})

        second = RAGService(embedding_client=FakeServiceEmbedder(), persist_dir=persist_dir)

        assert second.keyword_index is not None
        # Every stored chunk made it into the rebuilt keyword index.
        assert second.keyword_index.size == second.chunk_count
        hits = second.keyword_index.search("runbook", top_k=5)
        assert [chunk.doc_id for chunk in hits] == ["ops"]

    def test_fresh_persist_dir_gives_empty_keyword_index(self, persist_dir):
        service = RAGService(embedding_client=FakeServiceEmbedder(), persist_dir=persist_dir)

        assert service.keyword_index is not None
        assert service.keyword_index.size == 0
        assert service.keyword_index.search("anything") == []

    def test_no_persistence_means_no_keyword_index_until_store_exists(self):
        # Empty persist dir disables persistence entirely; the store (and with
        # it the keyword index) only materializes on first upload.
        service = RAGService(embedding_client=FakeServiceEmbedder(), persist_dir="")

        assert service.keyword_index is None


class TestHybridModeSetting:
    def test_defaults_to_auto(self):
        # Config default; assumes RAG_HYBRID is not exported in the test env.
        service = RAGService(embedding_client=FakeServiceEmbedder(), persist_dir="")

        assert service.hybrid_mode == "auto"

    def test_normalizes_case_and_whitespace(self, set_setting):
        set_setting("rag_hybrid", "  VECTOR ")
        service = RAGService(embedding_client=FakeServiceEmbedder(), persist_dir="")

        assert service.hybrid_mode == "vector"

    def test_unknown_value_falls_back_to_auto_with_warning(
        self, set_setting, caplog
    ):
        set_setting("rag_hybrid", "banana")
        service = RAGService(embedding_client=FakeServiceEmbedder(), persist_dir="")

        with caplog.at_level(logging.WARNING):
            assert service.hybrid_mode == "auto"
        assert any(
            "RAG_HYBRID" in record.getMessage() for record in caplog.records
        )


class TestQueryHybrid:
    def test_empty_index_returns_empty(self):
        service = RAGService(embedding_client=FakeServiceEmbedder(), persist_dir="")

        assert service.query_hybrid("anything") == []

    def test_keyword_match_outranks_vector_only_chunk(self, persist_dir):
        # "zebra" appears only in doc a, so a is rank 1 on keywords and must
        # come out on top after fusion no matter how the fake embedder ranks
        # the two chunks by cosine similarity.
        service = RAGService(embedding_client=FakeServiceEmbedder(), persist_dir=persist_dir)
        service.add_document("The zebra migration runs every Sunday night.", {"doc_id": "a"})
        service.add_document("Quarterly revenue grew across all regions.", {"doc_id": "b"})

        results = service.query_hybrid("zebra", top_k=2)

        assert [chunk.doc_id for chunk in results] == ["a", "b"]

    def test_fused_scores_are_rrf_scale_and_sorted(self, persist_dir):
        service = RAGService(embedding_client=FakeServiceEmbedder(), persist_dir=persist_dir)
        service.add_document("The zebra migration runs every Sunday night.", {"doc_id": "a"})
        service.add_document("Quarterly revenue grew across all regions.", {"doc_id": "b"})

        results = service.query_hybrid("zebra", top_k=2)

        assert len(results) == 2
        scores = [chunk.score for chunk in results]
        # RRF with k=60: a single rank-1 hit is worth 1/61, and the maximum
        # possible value (rank 1 on both lists) is 2/61.
        assert all(0 < score <= 2 / 61 for score in scores)
        assert scores == sorted(scores, reverse=True)

    def test_respects_top_k(self, persist_dir):
        service = RAGService(embedding_client=FakeServiceEmbedder(), persist_dir=persist_dir)
        texts = {
            "a": "The zebra migration runs every Sunday night.",
            "b": "Quarterly revenue grew across all regions.",
            "c": "Zebra stripes are a camouflage pattern.",
        }
        for doc_id, text in texts.items():
            service.add_document(text, {"doc_id": doc_id})

        results = service.query_hybrid("zebra", top_k=2)

        assert len(results) == 2

    def test_degrades_toquery_vector_when_keyword_index_missing(self, persist_dir):
        service = RAGService(embedding_client=FakeServiceEmbedder(), persist_dir=persist_dir)
        service.add_document("The zebra migration runs every Sunday night.", {"doc_id": "a"})

        vector_only = service.query("zebra", top_k=1)
        service._keyword_index = None  # simulate a missing keyword index

        results = service.query_hybrid("zebra", top_k=1)

        assert [chunk.doc_id for chunk in results] == [vector_only[0].doc_id]


class TestKeywordIndexSync:
    def test_add_document_updates_keyword_index_immediately(self, persist_dir):
        # No restart needed: the keyword index tracks indexing live.
        service = RAGService(embedding_client=FakeServiceEmbedder(), persist_dir=persist_dir)

        service.add_document("The zebra migration runs every Sunday night.", {"doc_id": "a"})

        assert [chunk.doc_id for chunk in service.keyword_index.search("zebra")] == ["a"]
        assert service.keyword_index.size == service.chunk_count

    def test_reindexing_changed_content_keeps_only_new_chunks(self, persist_dir):
        service = RAGService(embedding_client=FakeServiceEmbedder(), persist_dir=persist_dir)
        service.add_document("The zebra migration runs every Sunday night.", {"doc_id": "a"})

        service.add_document("The quantum rollout ships on Friday morning.", {"doc_id": "a"})

        assert service.keyword_index.search("zebra") == []
        assert [chunk.doc_id for chunk in service.keyword_index.search("quantum")] == ["a"]
        assert service.keyword_index.size == service.chunk_count

    def test_remove_document_updates_keyword_index(self, persist_dir):
        service = RAGService(embedding_client=FakeServiceEmbedder(), persist_dir=persist_dir)
        service.add_document("The zebra migration runs every Sunday night.", {"doc_id": "a"})
        service.add_document("Quarterly revenue grew across all regions.", {"doc_id": "b"})

        removed = service.remove_document("a")

        assert removed == 1
        assert service.keyword_index.search("zebra") == []
        assert [chunk.doc_id for chunk in service.keyword_index.search("revenue")] == ["b"]
        assert service.keyword_index.size == service.chunk_count

    def test_clear_resets_keyword_index(self, persist_dir):
        service = RAGService(embedding_client=FakeServiceEmbedder(), persist_dir=persist_dir)
        service.add_document("The zebra migration runs every Sunday night.", {"doc_id": "a"})

        service.clear()

        assert service.keyword_index.size == 0
        assert service.keyword_index.search("zebra") == []


class TestQueryModeRouting:
    def test_auto_mode_fuses_by_default(self, persist_dir):
        # RAG_HYBRID defaults to auto, so the plain query path is already hybrid.
        service = RAGService(embedding_client=FakeServiceEmbedder(), persist_dir=persist_dir)
        service.add_document("The zebra migration runs every Sunday night.", {"doc_id": "a"})
        service.add_document("Quarterly revenue grew across all regions.", {"doc_id": "b"})

        results = service.query("zebra", top_k=2)

        assert [chunk.doc_id for chunk in results] == ["a", "b"]
        # Fused scores are RRF-scale, not raw cosine similarity.
        assert all(0 < chunk.score <= 2 / 61 for chunk in results)

    def test_vector_mode_ignores_keywords(self, persist_dir, set_setting):
        set_setting("rag_hybrid", "vector")
        service = RAGService(embedding_client=FakeServiceEmbedder(), persist_dir=persist_dir)
        service.add_document("The zebra migration runs every Sunday night.", {"doc_id": "a"})

        results = service.query("zebra", top_k=2)

        assert results == service.query_vector("zebra", 2)

    def test_keyword_mode_ignores_vectors(self, persist_dir, set_setting):
        set_setting("rag_hybrid", "keyword")
        service = RAGService(embedding_client=FakeServiceEmbedder(), persist_dir=persist_dir)
        service.add_document("The zebra migration runs every Sunday night.", {"doc_id": "a"})

        results = service.query("zebra", top_k=2)

        assert results == service.keyword_index.search("zebra", top_k=2)

    def test_query_debug_reports_retrieval_mode(self, persist_dir, set_setting):
        set_setting("rag_hybrid", "keyword")
        service = RAGService(embedding_client=FakeServiceEmbedder(), persist_dir=persist_dir)

        assert service.query_debug("anything")["retrieval_mode"] == "keyword"




