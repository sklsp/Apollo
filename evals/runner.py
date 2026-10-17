"""Retrieval runner for the RAG evaluation (portfolio item 21).

Runs every question in an ``EvalDataset`` through a built index and produces
one scored-ready ``QuestionResult`` per question. The embedder is injected so
tests can use a deterministic fake; the CLI passes the real
``EmbeddingClient``.
"""

from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.services.rag.vector_store import VectorStore
from evals.dataset import EvalDataset
from evals.index import BatchEmbedder
from evals.metrics import QuestionResult


def retrieve_doc_ids(
    store: VectorStore, embedder: BatchEmbedder, query_text: str, *, top_k: int = 4
) -> tuple[str, ...]:
    """Embed *query_text* and return the ranked doc ids of its top-k chunks.

    The list is raw - one entry per chunk, repeats allowed - matching how the
    metrics consume it: the top-k cutoff applies to chunks, not documents.
    Raises ``ValueError`` for a non-positive *top_k* or an embedder result that
    is not exactly one row of the store's dimension.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    vectors = np.asarray(embedder.embed_batch([query_text]), dtype=np.float32)
    if vectors.shape != (1, store.dimension):
        raise ValueError(
            f"embedder returned shape {vectors.shape}, expected (1, {store.dimension})"
        )
    hits = store.search(vectors[0], top_k=top_k)
    return tuple(hit.doc_id for hit in hits)


def run_retrieval(
    dataset: EvalDataset,
    store: VectorStore,
    embedder: BatchEmbedder,
    *,
    top_k: int | None = None,
) -> list[QuestionResult]:
    """Run every question through retrieval and pair it with its ground truth.

    ``top_k`` defaults to the production setting (``rag_top_k``), so an eval
    run measures what the pipeline actually serves. Results come back in
    dataset order; each is ready for ``summarize`` or per-question scoring.
    Raises:
        ValueError: if the dataset has no questions, or *top_k* is non-positive.
    """
    if top_k is None:
        top_k = settings.rag_top_k
    if not dataset.questions:
        raise ValueError("dataset has no questions to run")

    results: list[QuestionResult] = []
    for question in dataset.questions:
        retrieved = retrieve_doc_ids(store, embedder, question.question, top_k=top_k)
        results.append(
            QuestionResult(
                question_id=question.id,
                retrieved_doc_ids=retrieved,
                relevant_doc_ids=tuple(question.source_docs),
            )
        )
    return results
