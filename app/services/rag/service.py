"""RAG orchestration service.

The vector index is persistent: chunks, vectors and index metadata live under
``DATA_DIR/rag/`` and are reloaded on startup, so a restart does not re-embed
the knowledge base. Indexing is incremental — a document is only embedded when
its content hash differs from the version already in the index.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from app.core.config import settings
from app.services.rag.chunking import chunk_text
from app.services.rag.embeddings import EmbeddingClient
from app.services.rag.vector_store import (
    IndexIncompatibleError,
    RetrievedChunk,
    StoredChunk,
    VectorStore,
)

logger = logging.getLogger(__name__)


def content_hash(text: str) -> str:
    """Stable hash of document content, used to detect real changes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RAGService:
    """Index documents and retrieve relevant context for chat."""

    def __init__(
        self,
        embedding_client: EmbeddingClient | None = None,
        persist_dir: str | None = None,
    ) -> None:
        self.embedding_client = embedding_client or EmbeddingClient()
        # Persisted under DATA_DIR/rag by default; tests pass a temp dir.
        self.persist_dir = persist_dir if persist_dir is not None else str(
            settings.data_dir.rstrip("/\\") + "/rag"
        )
        self._store: VectorStore | None = None
        # doc_id -> {"hash": ..., "chunks": N, "model": ..., "indexed_at": ...}
        self._doc_versions: dict[str, dict[str, Any]] = {}
        self._load_store()

    # ---------- state ----------

    @property
    def chunk_count(self) -> int:
        return self._store.size if self._store else 0

    def document_versions(self) -> dict[str, dict[str, Any]]:
        """Which version of each document is currently indexed."""
        return {doc_id: dict(info) for doc_id, info in self._doc_versions.items()}

    def is_indexed_current(self, doc_id: str, text: str) -> bool:
        """True when this exact content is already indexed with the current model."""
        info = self._doc_versions.get(doc_id)
        if not info:
            return False
        return (
            info.get("hash") == content_hash(text)
            and info.get("model") == self.embedding_client.model
        )

    def status(self) -> dict[str, Any]:
        """Developer/debug view: what is indexed and with which configuration."""
        store = self._store
        return {
            "persisted": self.persist_dir is not None,
            "chunks": store.size if store else 0,
            "documents": len(self._doc_versions),
            "embedding_model": self.embedding_client.model,
            "backend": self.embedding_client.backend,
            "dimension": store.dimension if store else None,
            "indexed_documents": [
                {
                    "doc_id": doc_id,
                    "content_hash": info["hash"][:12],
                    "chunks": info["chunks"],
                    "embedding_model": info["model"],
                    "indexed_at": info["indexed_at"],
                }
                for doc_id, info in sorted(self._doc_versions.items())
            ],
        }

    # ---------- indexing ----------

    def add_document(
        self,
        text: str,
        metadata: dict[str, Any],
        *,
        force: bool = False,
    ) -> int:
        """Chunk, embed, and index a document — incrementally.

        When the same ``doc_id`` is already indexed with the identical content
        hash and embedding model, nothing is re-embedded and 0 is returned.
        Pass ``force=True`` to re-index regardless.

        Args:
            text: Full document text.
            metadata: Must include ``doc_id``; ``filename`` or ``source`` is used for display.
            force: Re-embed even when the content is unchanged.

        Returns:
            Number of chunks indexed (0 when an unchanged doc was skipped).
        """
        doc_id = str(metadata.get("doc_id", metadata.get("id", "unknown")))
        filename = str(metadata.get("filename", metadata.get("source", doc_id)))

        if not force and self.is_indexed_current(doc_id, text):
            logger.info("[RAG] %s unchanged since last index; skipping re-embed", doc_id)
            return 0

        # Content changed (or forced): drop the stale vectors first so we never
        # mix old and new versions of the same document.
        if self._store is not None:
            removed = self._store.remove_by_doc_id(doc_id)
            if removed:
                logger.info("[RAG] %s changed; dropped %d stale chunks", doc_id, removed)

        chunks = chunk_text(
            text,
            chunk_size=settings.rag_chunk_size,
            overlap=settings.rag_chunk_overlap,
        )
        if not chunks:
            self._record_version(doc_id, text, 0)
            self._save()
            return 0

        embeddings = self.embedding_client.embed_batch(chunks)
        self._ensure_store(embeddings.shape[1])

        stored_chunks = [
            StoredChunk(
                doc_id=doc_id,
                filename=filename,
                chunk_index=index,
                text=chunk,
                content_hash=content_hash(text),
            )
            for index, chunk in enumerate(chunks)
        ]
        assert self._store is not None
        self._store.add(embeddings, stored_chunks)
        self._record_version(doc_id, text, len(chunks))
        self._save()
        return len(chunks)

    def query(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a question."""
        if not self._store or self._store.size == 0:
            return []

        k = top_k or settings.rag_top_k
        query_embedding = self.embedding_client.embed(question)
        return self._store.search(query_embedding, top_k=k)

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks for LLM prompt injection."""
        if not chunks:
            return ""

        parts: list[str] = []
        for chunk in chunks:
            parts.append(
                f"[{chunk.filename} | chunk {chunk.chunk_index} | score {chunk.score:.3f}]\n"
                f"{chunk.text}"
            )
        return "\n\n".join(parts)

    def remove_document(self, doc_id: str) -> int:
        """Remove all indexed chunks for a document."""
        if not self._store:
            return 0
        removed = self._store.remove_by_doc_id(doc_id)
        if removed:
            self._doc_versions.pop(doc_id, None)
            self._save()
        return removed

    def clear(self) -> None:
        """Clear the entire vector index."""
        if self._store:
            self._store.clear()
        self._doc_versions.clear()
        self._save()

    # ---------- internals ----------

    def _load_store(self) -> None:
        """Restore the persisted index at startup, tolerating corruption."""
        if not self.persist_dir:
            return

        try:
            dimension = self.embedding_client.dimension
        except Exception as exc:  # noqa: BLE001 - no backend available yet
            logger.warning(
                "[RAG] Could not probe embedding dimension (%s); "
                "index will load lazily on first upload", exc,
            )
            return

        try:
            self._store = VectorStore(
                dimension,
                persist_dir=self.persist_dir,
                embedding_model=self.embedding_client.model,
            )
        except IndexIncompatibleError:
            # Model/dimension changed: refuse to mix embeddings. Quarantine
            # the old files and start clean rather than serving garbage.
            logger.warning("[RAG] Stored index incompatible; quarantining")
            self._quarantine_files()
            self._store = VectorStore(
                dimension,
                persist_dir=self.persist_dir,
                embedding_model=self.embedding_client.model,
            )
            self.index_rebuilt = True

        # Derive per-document version records from chunk metadata so
        # incremental indexing works across restarts.
        self._rebuild_version_map()

    def _quarantine_files(self) -> None:
        from pathlib import Path
        import shutil

        root = Path(self.persist_dir)
        quarantine = root / "quarantined"
        quarantine.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        for name in ("index.faiss", "chunks.json", "meta.json"):
            path = root / name
            if path.is_file():
                try:
                    shutil.move(str(path), str(quarantine / f"{name}.{stamp}"))
                except OSError:
                    pass

    def _rebuild_version_map(self) -> None:
        """Derive doc versions from stored chunks after a restart."""
        if self._store is None:
            return
        seen: dict[str, dict[str, Any]] = {}
        for chunk in self._store._chunks:
            info = seen.setdefault(chunk.doc_id, {
                "hash": chunk.content_hash,
                "chunks": 0,
                "model": self.embedding_client.model,
                "indexed_at": "",
            })
            info["chunks"] += 1
        self._doc_versions = seen

    def _record_version(self, doc_id: str, text: str, chunk_count: int) -> None:
        self._doc_versions[doc_id] = {
            "hash": content_hash(text),
            "chunks": chunk_count,
            "model": self.embedding_client.model,
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        }

    def _save(self) -> None:
        if self._store is not None:
            try:
                self._store.save()
            except OSError as exc:
                logger.warning("[RAG] Could not save index: %s", exc)

    def _ensure_store(self, dimension: int) -> None:
        if self._store is None:
            self._store = VectorStore(
                dimension,
                persist_dir=self.persist_dir,
                embedding_model=self.embedding_client.model,
            )
        elif self._store.dimension != dimension:
            raise ValueError(
                "Embedding dimension changed; clear the RAG index before re-indexing "
                f"(expected {self._store.dimension}, got {dimension})"
            )
