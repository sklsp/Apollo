"""Service-level eval tests (portfolio item 22).

``index_dataset`` and ``run_retrieval_on_service`` measure the production
``RAGService`` end to end - its chunking defaults, incremental indexing and
query path - using deterministic fakes shaped like ``EmbeddingClient``.
"""

from __future__ import annotations

import pytest

from app.core.config import settings
from app.services.rag.service import RAGService
from evals.dataset import EvalDataset, load_dataset
from evals.metrics import summarize
from evals.runner import index_dataset, run_retrieval_on_service
from tests.fakes import FakeServiceEmbedder, MarkerServiceEmbedder


@pytest.fixture
def service(tmp_path):
    """RAGService wired to temp persistence and the deterministic fake client."""
    return RAGService(
        embedding_client=FakeServiceEmbedder(), persist_dir=str(tmp_path / "rag")
    )


class TestIndexDataset:
    def test_stores_every_document(self, service, synthetic_dataset):
        total = index_dataset(service, synthetic_dataset)

        assert total > 0
        assert total == service.chunk_count
        assert set(service.document_versions()) == synthetic_dataset.doc_ids

    def test_is_incremental_on_unchanged_content(self, service, synthetic_dataset):
        first = index_dataset(service, synthetic_dataset)
        second = index_dataset(service, synthetic_dataset)

        assert first > 0
        assert second == 0
        assert service.chunk_count == first


class TestRunRetrievalOnService:
    def test_one_result_per_question_in_dataset_order(self, service, synthetic_dataset):
        index_dataset(service, synthetic_dataset)

        results = run_retrieval_on_service(synthetic_dataset, service, top_k=3)

        assert [r.question_id for r in results] == [q.id for q in synthetic_dataset.questions]
        assert all(len(r.retrieved_doc_ids) <= 3 for r in results)
        assert results[0].relevant_doc_ids == ("alpha",)

    def test_marker_query_scores_perfectly_through_the_service(self, tmp_path, synthetic_dataset):
        """The fixture question is about alpha; every alpha chunk ranks at 1.0."""
        service = RAGService(
            embedding_client=MarkerServiceEmbedder(), persist_dir=str(tmp_path / "rag")
        )
        index_dataset(service, synthetic_dataset)

        results = run_retrieval_on_service(synthetic_dataset, service, top_k=4)
        summary = summarize(results, k=1)

        assert results[0].recall_at(1) == 1.0
        assert results[0].hit_at(1) is True
        assert summary.mrr == pytest.approx(1.0)
        assert summary.mean_recall_at_k == pytest.approx(1.0)

    def test_top_k_defaults_to_production_setting(self, service, synthetic_dataset, set_setting):
        index_dataset(service, synthetic_dataset)
        assert service.chunk_count >= 2
        set_setting("rag_top_k", 2)

        results = run_retrieval_on_service(synthetic_dataset, service)

        assert all(len(r.retrieved_doc_ids) == 2 for r in results)

    def test_explicit_top_k_overrides_setting(self, service, synthetic_dataset, set_setting):
        index_dataset(service, synthetic_dataset)
        set_setting("rag_top_k", 1)

        results = run_retrieval_on_service(synthetic_dataset, service, top_k=3)

        assert all(len(r.retrieved_doc_ids) == 3 for r in results)

    def test_empty_index_yields_empty_retrievals(self, service, synthetic_dataset):
        results = run_retrieval_on_service(synthetic_dataset, service, top_k=2)

        assert all(r.retrieved_doc_ids == () for r in results)

    def test_empty_dataset_raises(self, service):
        with pytest.raises(ValueError, match="no questions"):
            run_retrieval_on_service(EvalDataset(docs=[], questions=[]), service)


class TestCommittedCorpus:
    def test_runs_every_committed_question_through_the_service(self, tmp_path):
        dataset = load_dataset()
        service = RAGService(
            embedding_client=FakeServiceEmbedder(), persist_dir=str(tmp_path / "rag")
        )
        index_dataset(service, dataset)

        results = run_retrieval_on_service(dataset, service)

        assert len(results) == 48
        by_id = {r.question_id: r for r in results}
        for question in dataset.questions:
            retrieved = by_id[question.id]
            assert retrieved.relevant_doc_ids == tuple(question.source_docs)
            assert 1 <= len(retrieved.retrieved_doc_ids) <= settings.rag_top_k
