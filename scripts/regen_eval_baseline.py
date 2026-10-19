"""Regenerate the committed RAG eval baseline (portfolio item 37).

The CI regression gate (``tests/test_eval_regression.py``) compares live
retrieval against ``evals/baselines/vector_service.json``. Run this script to
re-baseline after an *intentional* change to chunking, embeddings or retrieval
logic:

    python -m scripts.regen_eval_baseline

It pins the sentence-transformers fallback backend (Ollama is never probed), so
the baseline is reproducible on any machine with the requirements installed.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from app.clients.ollama_client import OllamaClient
from app.core.config import settings
from app.core.exceptions import OllamaServiceError
from app.services.rag.embeddings import EmbeddingClient
from app.services.rag.service import RAGService
from evals.dataset import load_dataset
from evals.report import build_report, write_report
from evals.runner import index_dataset, run_retrieval_on_service

BASELINE_PATH = Path(__file__).resolve().parents[1] / "evals" / "baselines" / "vector_service.json"
TOP_K = 4


class _OllamaDown(OllamaClient):
    """``OllamaClient`` that always reports unreachable, forcing the fallback backend."""

    def embed(self, *, model: str, inputs: list[str]) -> list[list[float]]:
        raise OllamaServiceError("Ollama disabled for baseline generation", status_code=503)


def main() -> int:
    dataset = load_dataset()
    client = EmbeddingClient(ollama_client=_OllamaDown())
    persist_dir = Path(tempfile.mkdtemp(prefix="apollo-baseline-"))
    try:
        service = RAGService(embedding_client=client, persist_dir=str(persist_dir))
        num_chunks = index_dataset(service, dataset)
        results = run_retrieval_on_service(dataset, service, top_k=TOP_K, mode="vector")
    finally:
        shutil.rmtree(persist_dir, ignore_errors=True)

    config = {
        "mode": "service",
        "retrieval_mode": "vector",
        "embedding_backend": client.backend,
        "embedder_model": (
            client.model if client.backend == "ollama" else client.fallback_model
        ),
        "top_k": TOP_K,
        "num_docs": len(dataset.docs),
        "num_chunks": num_chunks,
        "chunk_size": settings.rag_chunk_size,
        "overlap": settings.rag_chunk_overlap,
    }
    report = build_report(results, k=TOP_K, config=config)
    path = write_report(report, BASELINE_PATH)

    summary = report["summary"]
    print(f"baseline written: {path}")
    print(
        f"recall@{TOP_K}={summary['mean_recall_at_k']:.4f} "
        f"MRR={summary['mrr']:.4f} hit_rate@{TOP_K}={summary['hit_rate_at_k']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
