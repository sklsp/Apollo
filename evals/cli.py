"""Command-line entry point for the RAG evaluation (portfolio item 24).

Runs the committed corpus and question set through retrieval and writes a JSON
report (item 23). Two modes:

* index mode (default): build an in-memory index with ``build_index``; the
  chunking parameters can be overridden on the command line.
* service mode (``--service``): index into a real ``RAGService`` and query it,
  measuring the production pipeline end to end. The service uses a temporary
  persist dir - an eval never touches ``DATA_DIR``.

Usage:
    python -m evals.cli [--top-k N] [--chunk-size N] [--overlap N]
                        [--out PATH] [--service]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.services.rag.embeddings import EmbeddingClient
from app.services.rag.service import RAGService
from evals.dataset import DatasetError, load_dataset
from evals.index import build_index
from evals.report import build_report, write_report
from evals.runner import index_dataset, run_retrieval, run_retrieval_on_service

DEFAULT_REPORT = Path(__file__).resolve().parent / "reports" / "rag_eval.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.cli",
        description="Run the RAG retrieval evaluation and write a JSON report.",
    )
    parser.add_argument("--top-k", type=int, default=None,
                        help=f"retrieval cutoff (default: settings.rag_top_k = {settings.rag_top_k})")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="index mode only; default: settings.rag_chunk_size")
    parser.add_argument("--overlap", type=int, default=None,
                        help="index mode only; default: settings.rag_chunk_overlap")
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT,
                        help=f"report path (default: {DEFAULT_REPORT})")
    parser.add_argument("--service", action="store_true",
                        help="measure the production RAGService instead of a bare index")
    return parser


def run(argv: list[str] | None = None, *, client: EmbeddingClient | None = None) -> int:
    """Execute one eval run; returns a process exit code (0 on success)."""
    args = build_parser().parse_args(argv)
    if args.top_k is not None and args.top_k <= 0:
        print("error: --top-k must be positive", file=sys.stderr)
        return 1

    client = client or EmbeddingClient()
    try:
        dataset = load_dataset()
        top_k = args.top_k if args.top_k is not None else settings.rag_top_k
        config: dict[str, Any] = {
            "mode": "service" if args.service else "index",
            "embedder_model": client.model,
            "top_k": top_k,
            "num_docs": len(dataset.docs),
        }

        if args.service:
            persist_dir = Path(tempfile.mkdtemp(prefix="apollo-eval-"))
            try:
                service = RAGService(embedding_client=client, persist_dir=str(persist_dir))
                config["num_chunks"] = index_dataset(service, dataset)
                results = run_retrieval_on_service(dataset, service, top_k=top_k)
            finally:
                shutil.rmtree(persist_dir, ignore_errors=True)
        else:
            chunk_size = args.chunk_size if args.chunk_size is not None else settings.rag_chunk_size
            overlap = (args.overlap if args.overlap is not None
                       else settings.rag_chunk_overlap)
            result = build_index(dataset, client, chunk_size=chunk_size, overlap=overlap)
            results = run_retrieval(dataset, result.store, client, top_k=top_k)
            config["chunk_size"] = chunk_size
            config["overlap"] = overlap
            config["num_chunks"] = result.num_chunks

        # Read after indexing so the backend probe is free (already embedded).
        config["embedding_backend"] = client.backend
        if client.backend == "sentence-transformers":
            # Ollama was unavailable; record what actually produced the vectors.
            config["embedder_model"] = client.fallback_model
        report = build_report(results, k=top_k, config=config)
        path = write_report(report, args.out)
    except (DatasetError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    summary = report["summary"]
    print(
        f"{summary['num_questions']} questions | "
        f"recall@{top_k}={summary['mean_recall_at_k']:.3f} "
        f"MRR={summary['mrr']:.3f} hit_rate={summary['hit_rate_at_k']:.3f}"
    )
    print(f"report: {path}")
    return 0


def main(
    argv: list[str] | None = None, *, client: EmbeddingClient | None = None
) -> int:
    return run(argv, client=client)


if __name__ == "__main__":
    raise SystemExit(main())
