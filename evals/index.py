"""Build a retrieval index from the eval corpus (portfolio item 20).

Reuses the app's production chunker (:func:`app.services.rag.chunking.chunk_text`)
and vector store (:class:`app.services.rag.vector_store.VectorStore`) so the
evaluation measures exactly what the RAG pipeline does. The embedder is
injected: tests pass a deterministic fake, and the runner (item 21) passes the
real :class:`~app.services.rag.embeddings.EmbeddingClient`.

The built store is in-memory only - evaluation never touches ``DATA_DIR``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.core.config import settings
from app.services.rag.chunking import chunk_text
from app.services.rag.vector_store import StoredChunk, VectorStore
from evals.dataset import EvalDataset


class BatchEmbedder(Protocol):
    """Anything that can embed a batch of texts (e.g. ``EmbeddingClient``)."""

    def embed_batch(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class IndexBuildResult:
    """A fresh in-memory index over the corpus plus build diagnostics."""

    store: VectorStore
    num_chunks: int
    chunks_per_doc: dict[str, int]


def build_index(
    dataset: EvalDataset,
    embedder: BatchEmbedder,
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> IndexBuildResult:
    """Chunk every corpus document and index the chunks in a fresh store.

    Chunking parameters default to the app's production settings
    (``rag_chunk_size`` / ``rag_chunk_overlap``), so by default the evaluation
    measures exactly what the RAG pipeline does; pass explicit values to test
    other configurations. All chunks are embedded in a single batch call, in
    document order, and the embedding dimension is inferred from the output.

    Raises:
        ValueError: empty dataset, a document that yields no chunks, or an
            embedder result whose shape does not match the chunk count.
    """
    if chunk_size is None:
        chunk_size = settings.rag_chunk_size
    if overlap is None:
        overlap = settings.rag_chunk_overlap
    if not dataset.docs:
        raise ValueError("dataset has no documents")

    chunks_by_doc: dict[str, list[str]] = {}
    for doc in dataset.docs:
        chunks = chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            raise ValueError(f"document {doc.doc_id!r} produced no chunks")
        chunks_by_doc[doc.doc_id] = chunks

    all_chunks = [text for doc in dataset.docs for text in chunks_by_doc[doc.doc_id]]
    vectors = np.asarray(embedder.embed_batch(all_chunks), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(all_chunks):
        raise ValueError(
            f"embedder returned shape {vectors.shape} for {len(all_chunks)} chunks"
        )

    store = VectorStore(dimension=vectors.shape[1])
    stored: list[StoredChunk] = []
    for doc in dataset.docs:
        for chunk_index, text in enumerate(chunks_by_doc[doc.doc_id]):
            stored.append(
                StoredChunk(
                    doc_id=doc.doc_id,
                    filename=doc.doc_id,
                    chunk_index=chunk_index,
                    text=text,
                )
            )
    store.add(vectors, stored)

    return IndexBuildResult(
        store=store,
        num_chunks=len(stored),
        chunks_per_doc={doc_id: len(chunks) for doc_id, chunks in chunks_by_doc.items()},
    )
