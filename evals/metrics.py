"""Retrieval metrics for the RAG evaluation (portfolio item 19).

The unit of scoring is the *document*: a question is answered well when the
retriever surfaces chunks from its ground-truth ``source_docs``. Metrics are
computed on the raw ranked chunk list exactly as the pipeline would consume
it - the top-*k* cutoff applies to chunks, so repeated chunks from one
document do occupy slots - and a document is then counted at most once when
scoring that cutoff.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass


def recall_at_k(
    retrieved_doc_ids: Sequence[str], relevant_doc_ids: Collection[str], k: int
) -> float:
    """Fraction of relevant documents present in the top-*k* results.

    The cutoff applies to the raw ranked list (repeats occupy slots); within
    the cutoff each document counts once. Returns a value in [0, 1]; 1.0
    means every ground-truth document was retrieved within the cutoff.
    Raises ``ValueError`` for an empty relevance set or a non-positive *k*.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    relevant = set(relevant_doc_ids)
    if not relevant:
        raise ValueError("relevant_doc_ids must be non-empty")
    top_k_docs = set(retrieved_doc_ids[:k])
    return len(top_k_docs & relevant) / len(relevant)


def reciprocal_rank(
    retrieved_doc_ids: Sequence[str], relevant_doc_ids: Collection[str]
) -> float:
    """1/rank of the first result from a relevant document (ranks start at 1).

    Ranks are positions in the raw list, so earlier repeats of an irrelevant
    document push the first hit down. Returns 0.0 when no result is relevant.
    """
    relevant = set(relevant_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def hit_at_k(
    retrieved_doc_ids: Sequence[str], relevant_doc_ids: Collection[str], k: int
) -> bool:
    """Whether at least one of the top-*k* results comes from a relevant doc."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    relevant = set(relevant_doc_ids)
    return any(doc_id in relevant for doc_id in retrieved_doc_ids[:k])


@dataclass(frozen=True)
class QuestionResult:
    """The retrieval outcome for one question, ready to be scored."""

    question_id: str
    retrieved_doc_ids: tuple[str, ...]  # ranked, best first
    relevant_doc_ids: tuple[str, ...]   # ground truth from the dataset

    def recall_at(self, k: int) -> float:
        return recall_at_k(self.retrieved_doc_ids, self.relevant_doc_ids, k)

    def reciprocal_rank(self) -> float:
        return reciprocal_rank(self.retrieved_doc_ids, self.relevant_doc_ids)

    def hit_at(self, k: int) -> bool:
        return hit_at_k(self.retrieved_doc_ids, self.relevant_doc_ids, k)


@dataclass(frozen=True)
class MetricsSummary:
    """Aggregate retrieval quality over a set of question results."""

    num_questions: int
    k: int
    mean_recall_at_k: float
    mrr: float
    hit_rate_at_k: float


def summarize(results: Sequence[QuestionResult], k: int = 5) -> MetricsSummary:
    """Average the per-question metrics over *results*.

    ``mrr`` is the mean reciprocal rank; ``hit_rate_at_k`` is the fraction of
    questions with at least one relevant document in the top-*k*. Raises
    ``ValueError`` when there are no results to summarize.
    """
    if not results:
        raise ValueError("cannot summarize zero question results")
    n = len(results)
    return MetricsSummary(
        num_questions=n,
        k=k,
        mean_recall_at_k=sum(r.recall_at(k) for r in results) / n,
        mrr=sum(r.reciprocal_rank() for r in results) / n,
        hit_rate_at_k=sum(1.0 if r.hit_at(k) else 0.0 for r in results) / n,
    )
