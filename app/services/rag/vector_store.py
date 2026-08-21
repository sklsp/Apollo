"""Persistent FAISS vector store for RAG chunks.

Layout under ``DATA_DIR/rag/``::

    rag/index.faiss        flat inner-product index over normalized vectors
    rag/chunks.json        chunk metadata, one record per vector row
    rag/meta.json          index version, embedding model, dimension, counts

Design constraints (local-first, single user):

* **Atomic saves.** The index and metadata are written via temp-file + replace,
  so a crash mid-save cannot corrupt the previous good state.
* **Self-describing.** ``meta.json`` records the embedding model and dimension;
  loading an index built with a different model is refused rather than silently
  mixing incompatible vectors.
* **Recoverable.** If the FAISS file is missing or unreadable but chunk
  metadata survives, the store quarantines the bad files and reports the loss —
  the caller (RAGService) decides to re-embed from persisted documents.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.core.paths import read_json, write_json_atomic

logger = logging.getLogger(__name__)

INDEX_VERSION = 1


class IndexIncompatibleError(Exception):
    """The stored index does not match the current embedding configuration."""


@dataclass
class StoredChunk:
    """A chunk stored in the vector index."""

    doc_id: str
    filename: str
    chunk_index: int
    text: str
    content_hash: str = ""  # hash of the source document this chunk came from


@dataclass
class RetrievedChunk:
    """A chunk returned from similarity search."""

    doc_id: str
    filename: str
    chunk_index: int
    text: str
    score: float


@dataclass
class IndexMeta:
    """Everything needed to know whether a stored index is usable."""

    version: int = INDEX_VERSION
    embedding_model: str = ""
    dimension: int = 0
    vector_count: int = 0
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexMeta:
        return cls(
            version=int(data.get("version", 0)),
            embedding_model=str(data.get("embedding_model", "")),
            dimension=int(data.get("dimension", 0)),
            vector_count=int(data.get("vector_count", 0)),
            updated_at=float(data.get("updated_at", 0.0)),
        )


class VectorStore:
    """FAISS-backed persistent vector store with chunk metadata.

    Works in-memory while running; :meth:`save` flushes to disk and the
    constructor reloads whatever was saved, so documents survive application
    restarts without re-embedding.
    """

    def __init__(
        self,
        dimension: int,
        persist_dir: str | Path | None = None,
        embedding_model: str = "",
    ) -> None:
        self.dimension = dimension
        self.embedding_model = embedding_model
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self._chunks: list[StoredChunk] = []
        self._embeddings: list[np.ndarray] = []
        self._index = faiss.IndexFlatIP(dimension)
        self._dirty = False

        if self.persist_dir is not None:
            self._load()

    # ---------- persistence ----------

    @property
    def _index_path(self) -> Path:
        assert self.persist_dir is not None
        return self.persist_dir / "index.faiss"

    @property
    def _chunks_path(self) -> Path:
        assert self.persist_dir is not None
        return self.persist_dir / "chunks.json"

    @property
    def _meta_path(self) -> Path:
        assert self.persist_dir is not None
        return self.persist_dir / "meta.json"

    def _load(self) -> bool:
        """Load a saved index. Returns True when usable state was restored."""
        assert self.persist_dir is not None
        meta_raw = read_json(self._meta_path, default=None)
        if not isinstance(meta_raw, dict):
            # No metadata means no (complete) index was ever saved.
            return False

        meta = IndexMeta.from_dict(meta_raw)

        if meta.version != INDEX_VERSION:
            logger.warning(
                "[RAG] Index version %s != %s; starting fresh",
                meta.version, INDEX_VERSION,
            )
            self._quarantine()
            return False

        if meta.embedding_model != self.embedding_model:
            raise IndexIncompatibleError(
                f"Saved index was built with embedding model "
                f"'{meta.embedding_model}' but the current model is "
                f"'{self.embedding_model}'. Clear the RAG index to rebuild it."
            )

        if meta.dimension != self.dimension:
            raise IndexIncompatibleError(
                f"Saved index has dimension {meta.dimension} but the current "
                f"model produces {self.dimension}."
            )

        chunks_raw = read_json(self._chunks_path, default=None)
        if not isinstance(chunks_raw, list) or len(chunks_raw) != meta.vector_count:
            logger.warning("[RAG] Chunk metadata missing/mismatched; rebuilding")
            self._quarantine()
            return False

        try:
            index = faiss.read_index(str(self._index_path))
        except (RuntimeError, OSError) as exc:
            logger.warning("[RAG] Could not read FAISS index (%s); quarantining", exc)
            self._quarantine()
            return False

        if index.ntotal != meta.vector_count:
            logger.warning(
                "[RAG] Index holds %d vectors but metadata says %d; quarantining",
                index.ntotal, meta.vector_count,
            )
            self._quarantine()
            return False

        if index.d != self.dimension:
            logger.warning("[RAG] Index dimension mismatch; quarantining")
            self._quarantine()
            return False

        chunks: list[StoredChunk] = []
        for raw in chunks_raw:
            if not isinstance(raw, dict) or "doc_id" not in raw:
                continue
            chunks.append(StoredChunk(
                doc_id=str(raw["doc_id"]),
                filename=str(raw.get("filename", "")),
                chunk_index=int(raw.get("chunk_index", 0)),
                text=str(raw.get("text", "")),
                content_hash=str(raw.get("content_hash", "")),
            ))

        if len(chunks) != meta.vector_count:
            logger.warning("[RAG] Chunk records incomplete; quarantining")
            self._quarantine()
            return False

        # Success: adopt the loaded state.
        self._index = index
        self._chunks = chunks
        self._embeddings = []  # rebuilt lazily only when remove() needs them
        self._dirty = False
        logger.info("[RAG] Restored index: %d vectors from %s",
                    meta.vector_count, self.persist_dir)
        return True

    def save(self) -> None:
        """Flush index + metadata atomically. Cheap enough to call per change."""
        if self.persist_dir is None or not self._dirty:
            return
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(self._index_path))
        write_json_atomic(
            self._chunks_path,
            [asdict(chunk) for chunk in self._chunks],
        )
        write_json_atomic(
            self._meta_path,
            IndexMeta(
                embedding_model=self.embedding_model,
                dimension=self.dimension,
                vector_count=self._index.ntotal,
            ).to_dict(),
        )
        self._dirty = False

    def _quarantine(self) -> None:
        """Move unusable index files aside instead of deleting them outright."""
        assert self.persist_dir is not None
        quarantine = self.persist_dir / "quarantined"
        quarantine.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        for path in (self._index_path, self._chunks_path, self._meta_path):
            if path.is_file():
                try:
                    shutil.move(str(path), str(quarantine / f"{path.name}.{stamp}"))
                except OSError as exc:
                    logger.warning("[RAG] Could not quarantine %s: %s", path.name, exc)
        logger.warning("[RAG] Quarantined unusable index files in %s", quarantine)

    # ---------- vector operations ----------

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def indexed_doc_ids(self) -> set[str]:
        return {chunk.doc_id for chunk in self._chunks}

    def add(self, embeddings: np.ndarray, chunks: list[StoredChunk]) -> None:
        """Add embeddings and their metadata to the store."""
        if embeddings.size == 0:
            return
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Embedding count must match chunk count")
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, got {embeddings.shape[1]}"
            )

        self._index.add(embeddings.astype(np.float32))
        for row, chunk in zip(embeddings, chunks, strict=True):
            self._embeddings.append(np.asarray(row, dtype=np.float32))
            self._chunks.append(chunk)
        self._dirty = True

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[RetrievedChunk]:
        """Return the most similar chunks for a query embedding."""
        if self._index.ntotal == 0:
            return []

        vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(vector, k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            results.append(
                RetrievedChunk(
                    doc_id=chunk.doc_id,
                    filename=chunk.filename,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=float(score),
                )
            )
        return results

    def remove_by_doc_id(self, doc_id: str) -> int:
        """Remove all chunks for a document and rebuild the index."""
        remaining_chunks: list[StoredChunk] = []
        remaining_embeddings: list[np.ndarray] = []
        removed = 0

        for chunk, embedding in zip(self._chunks, self._embeddings, strict=True):
            if chunk.doc_id == doc_id:
                removed += 1
                continue
            remaining_chunks.append(chunk)
            remaining_embeddings.append(embedding)

        if removed:
            self._rebuild(remaining_chunks, remaining_embeddings)
            self._dirty = True
        return removed

    def clear(self) -> None:
        """Remove all stored chunks."""
        self._chunks.clear()
        self._embeddings.clear()
        self._index = faiss.IndexFlatIP(self.dimension)
        self._dirty = True
        if self.persist_dir is not None:
            for path in (self._index_path, self._chunks_path, self._meta_path):
                path.unlink(missing_ok=True)

    def _rebuild(
        self,
        chunks: list[StoredChunk],
        embeddings: list[np.ndarray],
    ) -> None:
        """Rebuild the FAISS index from stored chunk metadata."""
        self._chunks = chunks
        self._embeddings = embeddings
        self._index = faiss.IndexFlatIP(self.dimension)
        if embeddings:
            matrix = np.vstack(embeddings).astype(np.float32)
            self._index.add(matrix)
