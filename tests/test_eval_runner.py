"""Retrieval runner tests (portfolio item 21).

``run_retrieval`` pairs every question with its ground truth; these tests use
the shared synthetic dataset plus a marker-based embedder whose two orthogonal
vectors make rankings fully deterministic. One test pins the run against the
committed corpus itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.config import settings
from evals.dataset import EvalDataset, load_dataset
from evals.index import build_index
from evals.metrics import summarize
from evals.runner import retrieve_doc_ids, run_retrieval
from tests.fakes import FakeEmbedder


class MarkerEmbedder:
    """Maps each text to one of two orthogonal unit vectors.

    Text mentioning "alpha" gets the first basis vector, everything else the
    second - so a query about alpha ranks every alpha chunk at exactly 1.0 and
    every other chunk at exactly 0.0, making rankings deterministic without
    relying on hash collisions.
    """

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.v_alpha = np.zeros(dim, dtype=np.float32)
        self.v_alpha[0] = 1.0
        self.v_beta = np.zeros(dim, dtype=np.float32)
        self.v_beta[1] = 1.0

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.vstack(
            [self.v_alpha if "alpha" in t.lower() else self.v_beta for t in texts]
        )


class TestRetrieveDocIds:
    def test_self_query_ranks_own_doc_first(self, synthetic_dataset):
        result = build_index(synthetic_dataset, MarkerEmbedder())
        alpha_text = next(d.text for d in synthetic_dataset.docs if d.doc_id == "alpha")

        doc_ids = retrieve_doc_ids(result.store, MarkerEmbedder(), alpha_text, top_k=3)

        assert isinstance(doc_ids, tuple)
        assert doc_ids[0] == "alpha"

    def test_repeats_are_preserved_in_the_raw_list(self, synthetic_dataset):
        """Two beta chunks both score 1.0 for a beta query - both occupy slots."""
        result = build_index(synthetic_dataset, MarkerEmbedder())
        assert result.chunks_per_doc["beta"] >= 2

        doc_ids = retrieve_doc_ids(result.store, MarkerEmbedder(), "BETA?", top_k=2)

        assert doc_ids == ("beta", "beta")

    def test_top_k_larger_than_store_returns_every_chunk(self, synthetic_dataset):
        result = build_index(synthetic_dataset, FakeEmbedder())

        doc_ids = retrieve_doc_ids(result.store, FakeEmbedder(), "anything", top_k=100)

        assert len(doc_ids) == result.num_chunks

    def test_non_positive_top_k_raises(self, synthetic_dataset):
        result = build_index(synthetic_dataset, FakeEmbedder())
        with pytest.raises(ValueError, match="top_k must be positive"):
            retrieve_doc_ids(result.store, FakeEmbedder(), "q", top_k=0)

    def test_wrong_dimension_embedder_raises(self, synthetic_dataset):
        result = build_index(synthetic_dataset, MarkerEmbedder(dim=8))
        with pytest.raises(ValueError, match="shape"):
            retrieve_doc_ids(result.store, MarkerEmbedder(dim=7), "q", top_k=2)


class TestRunRetrieval:
    def test_one_result_per_question_in_dataset_order(self, synthetic_dataset):
        result = build_index(synthetic_dataset, FakeEmbedder())

        results = run_retrieval(
            synthetic_dataset, result.store, FakeEmbedder(), top_k=3
        )

        assert [r.question_id for r in results] == [q.id for q in synthetic_dataset.questions]
        assert all(len(r.retrieved_doc_ids) <= 3 for r in results)
        assert results[0].relevant_doc_ids == ("alpha",)

    def test_results_are_scoreable_and_perfect_for_marker_query(self, synthetic_dataset):
        result = build_index(synthetic_dataset, MarkerEmbedder())

        # The fixture question is about alpha; the marker embedder puts every
        # alpha chunk at rank 1 with score 1.0.
        results = run_retrieval(
            synthetic_dataset, result.store, MarkerEmbedder(), top_k=4
        )
        summary = summarize(results, k=1)

        assert results[0].recall_at(1) == 1.0
        assert results[0].hit_at(1) is True
        assert summary.mrr == pytest.approx(1.0)
        assert summary.mean_recall_at_k == pytest.approx(1.0)

    def test_top_k_defaults_to_production_setting(self, synthetic_dataset, set_setting):
        result = build_index(synthetic_dataset, FakeEmbedder())
        assert result.num_chunks >= 3
        set_setting("rag_top_k", 2)

        results = run_retrieval(synthetic_dataset, result.store, FakeEmbedder())

        # No explicit top_k: the production setting (2) must be what was used.
        assert all(len(r.retrieved_doc_ids) == 2 for r in results)

    def test_explicit_top_k_overrides_setting(self, synthetic_dataset, set_setting):
        result = build_index(synthetic_dataset, FakeEmbedder())
        set_setting("rag_top_k", 1)

        results = run_retrieval(
            synthetic_dataset, result.store, FakeEmbedder(), top_k=3
        )

        assert all(len(r.retrieved_doc_ids) == 3 for r in results)

    def test_empty_dataset_raises(self):
        from app.services.rag.vector_store import VectorStore

        with pytest.raises(ValueError, match="no questions"):
            run_retrieval(EvalDataset(docs=[], questions=[]), VectorStore(8), FakeEmbedder())


class TestCommittedCorpus:
    def test_runs_every_committed_question(self):
        dataset = load_dataset()
        result = build_index(dataset, FakeEmbedder())

        results = run_retrieval(
            dataset, result.store, FakeEmbedder(), top_k=settings.rag_top_k
        )

        assert len(results) == 48
        by_id = {r.question_id: r for r in results}
        for question in dataset.questions:
            retrieved = by_id[question.id]
            assert retrieved.relevant_doc_ids == tuple(question.source_docs)
            assert 1 <= len(retrieved.retrieved_doc_ids) <= settings.rag_top_k

