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

