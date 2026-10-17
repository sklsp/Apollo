"""Retrieval metrics for the RAG evaluation (portfolio item 19)."""

from __future__ import annotations

import pytest

from evals.metrics import (
    QuestionResult,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    summarize,
)


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b"], {"a", "b"}, 5) == 1.0

    def test_partial_recall(self):
        # One of the two relevant docs is in the cutoff: recall = 1/2.
        assert recall_at_k(["a", "x"], {"a", "b"}, 5) == pytest.approx(0.5)

    def test_no_hits(self):
        assert recall_at_k(["x", "y"], {"a"}, 5) == 0.0

    def test_cutoff_excludes_later_hits(self):
        # Relevant doc exists but only at rank 6; k=5 must not see it.
        assert recall_at_k(["x"] * 5 + ["a"], {"a"}, 5) == 0.0
        assert recall_at_k(["x"] * 5 + ["a"], {"a"}, 6) == 1.0

    def test_repeated_doc_counts_once(self):
        # One of two relevant docs, repeated to fill the cutoff: still 0.5.
        assert recall_at_k(["a", "a", "a"], {"a", "b"}, 3) == pytest.approx(0.5)

    def test_empty_retrieval_scores_zero(self):
        assert recall_at_k([], {"a"}, 5) == 0.0

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_raises(self, k):
        with pytest.raises(ValueError, match="k must be positive"):
            recall_at_k(["a"], {"a"}, k)

    def test_empty_relevance_set_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            recall_at_k(["a"], set(), 5)


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "b"], {"a", "b"}) == 1.0

    def test_second_position(self):
        assert reciprocal_rank(["x", "a"], {"a"}) == pytest.approx(0.5)

    def test_no_relevant_results(self):
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_empty_retrieval(self):
        assert reciprocal_rank([], {"a"}) == 0.0

    def test_repeats_of_irrelevant_doc_shift_first_hit(self):
        # Raw ranks: three irrelevant chunks occupy slots, so the hit is rank 4.
        assert reciprocal_rank(["x", "x", "x", "a"], {"a"}) == pytest.approx(0.25)


class TestHitAtK:
    def test_hit_within_cutoff(self):
        assert hit_at_k(["x", "a"], {"a"}, 2) is True

    def test_miss_beyond_cutoff(self):
        assert hit_at_k(["x"] * 3 + ["a"], {"a"}, 3) is False

    def test_no_hits(self):
        assert hit_at_k(["x"], {"a"}, 5) is False

    @pytest.mark.parametrize("k", [0, -2])
    def test_non_positive_k_raises(self, k):
        with pytest.raises(ValueError, match="k must be positive"):
            hit_at_k(["a"], {"a"}, k)


class TestQuestionResult:
    def test_methods_delegate_to_functions(self):
        result = QuestionResult("q01", ("x", "a"), ("a", "b"))
        assert result.recall_at(2) == pytest.approx(0.5)
        assert result.reciprocal_rank() == pytest.approx(0.5)
        assert result.hit_at(2) is True


class TestSummarize:
    @pytest.fixture
    def results(self):
        return [
            QuestionResult("q1", ("a", "b"), ("a",)),          # recall 1, rr 1, hit
            QuestionResult("q2", ("x", "a"), ("a", "b")),      # recall .5, rr .5, hit
            QuestionResult("q3", ("y",), ("z",)),              # recall 0, rr 0, miss
        ]

    def test_aggregates(self, results):
        summary = summarize(results, k=2)
        assert summary.num_questions == 3
        assert summary.k == 2
        assert summary.mean_recall_at_k == pytest.approx(0.5)
        assert summary.mrr == pytest.approx(0.5)
        assert summary.hit_rate_at_k == pytest.approx(2 / 3)

    def test_default_k_is_five(self, results):
        assert summarize(results).k == 5

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="zero question results"):
            summarize([], k=5)
