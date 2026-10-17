"""Retrieval runner for the RAG evaluation (portfolio items 21-22).

Runs every question in an ``EvalDataset`` through a built index and produces
one scored-ready ``QuestionResult`` per question. The embedder is injected so
tests can use a deterministic fake; the CLI passes the real
``EmbeddingClient``. Item 22 adds the same run against a live ``RAGService``,
so an eval can measure the production pipeline (its chunking defaults and
query path) instead of a bare store + embedder pair.
"""

from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.services.rag.service import RAGService
from app.services.rag.vector_store import RetrievedChunk, VectorStore
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


def index_dataset(service: RAGService, dataset: EvalDataset) -> int:
    """Index every corpus document into *service* through its production path.

    Uses ``add_document`` with the service's own chunking defaults so the eval
    measures exactly what the app indexes. Returns the total number of chunks
    stored (0 for documents whose content was already current).
    """
    total = 0
    for doc in dataset.docs:
        total += service.add_document(
            doc.text, {"doc_id": doc.doc_id, "filename": doc.title}
        )
    return total


def run_retrieval_on_service(
    dataset: EvalDataset,
    service: RAGService,
    *,
    top_k: int | None = None,
    mode: str | None = None,
) -> list[QuestionResult]:
    """Run every question through the real ``RAGService``.

    Same contract as :func:`run_retrieval` (dataset order, raw chunk ranks,
    ground truth from ``source_docs``), but retrieval goes through the
    production service instead of a bare store + embedder pair. An empty index
    yields empty retrievals rather than an error, mirroring the app.

    ``mode`` selects the retrieval path under test - "auto" (fused vector +
    keyword), "vector", or "keyword". It defaults to the production
    ``RAG_HYBRID`` setting so a plain run measures what the app serves; pass
    an explicit value to compare modes side by side.

    Raises:
        ValueError: if the dataset has no questions or *mode* is not one of
            auto/vector/keyword.
    """
    if top_k is None:
        top_k = settings.rag_top_k
    if not dataset.questions:
        raise ValueError("dataset has no questions to run")
    if mode is None:
        mode = service.hybrid_mode
    if mode not in ("auto", "vector", "keyword"):
        raise ValueError(
            f"unknown retrieval mode {mode!r}; expected auto, vector or keyword"
        )

    results: list[QuestionResult] = []
    for question in dataset.questions:
        hits = _retrieve_on_service(service, question.question, top_k, mode)
        retrieved = tuple(hit.doc_id for hit in hits)
        results.append(
            QuestionResult(
                question_id=question.id,
                retrieved_doc_ids=retrieved,
                relevant_doc_ids=tuple(question.source_docs),
            )
        )
    return results


def _retrieve_on_service(
    service: RAGService, text: str, top_k: int, mode: str
) -> list[RetrievedChunk]:
    """One retrieval through the requested mode of *service*."""
    if mode == "vector":
        return service.query_vector(text, top_k=top_k)
    if mode == "keyword":
        index = service.keyword_index
        return index.search(text, top_k=top_k) if index is not None else []
    return service.query_hybrid(text, top_k=top_k)
