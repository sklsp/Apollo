"""Command-line entry point for the RAG evaluation (portfolio item 24).

Runs the committed corpus and question set through retrieval and writes a JSON
report (item 23). Three modes:

* index mode (default): build an in-memory index with ``build_index``; the
  chunking parameters can be overridden on the command line.
* service mode (``--service``): index into a real ``RAGService`` and query it,
  measuring the production pipeline end to end. The service uses a temporary
  persist dir - an eval never touches ``DATA_DIR``.
* grid mode (``--grid``, item 27): run every cell of the chunking grid and
  write one row per cell.

Usage:
    python -m evals.cli [--top-k N] [--chunk-size N] [--overlap N]
                        [--out PATH] [--service | --grid]
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
from evals.grid import DEFAULT_CHUNKING_GRID, run_grid, write_grid_report
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
    parser.add_argument("--grid", action="store_true",
                        help=f"run every cell of the chunking grid "
                             f"(default: {DEFAULT_CHUNKING_GRID}) and write one row per cell")
    return parser


def run(argv: list[str] | None = None, *, client: EmbeddingClient | None = None) -> int:
    """Execute one eval run; returns a process exit code (0 on success)."""
    args = build_parser().parse_args(argv)
    if args.top_k is not None and args.top_k <= 0:
        print("error: --top-k must be positive", file=sys.stderr)
        return 1
    if args.grid and (args.service or args.chunk_size is not None or args.overlap is not None):
        print("error: --grid cannot be combined with --service, --chunk-size or --overlap",
              file=sys.stderr)
        return 1

    client = client or EmbeddingClient()
    try:
        dataset = load_dataset()
        top_k = args.top_k if args.top_k is not None else settings.rag_top_k
        mode = "grid" if args.grid else ("service" if args.service else "index")
        config: dict[str, Any] = {
            "mode": mode,
            "embedder_model": client.model,
            "top_k": top_k,
            "num_docs": len(dataset.docs),
        }

        if args.grid:
            cells = run_grid(dataset, client, top_k=top_k)
        elif args.service:
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

        if args.grid:
            path = write_grid_report(cells, args.out, top_k=top_k, config=config)
            for cell in cells:
                s = cell.summary
                print(
                    f"{cell.chunk_size}/{cell.overlap} -> {cell.num_chunks} chunks | "
                    f"recall@{top_k}={s.mean_recall_at_k:.3f} MRR={s.mrr:.3f} "
                    f"hit_rate={s.hit_rate_at_k:.3f}"
                )
        else:
            report = build_report(results, k=top_k, config=config)
            path = write_report(report, args.out)
            summary = report["summary"]
            print(
                f"{summary['num_questions']} questions | "
                f"recall@{top_k}={summary['mean_recall_at_k']:.3f} "
                f"MRR={summary['mrr']:.3f} hit_rate={summary['hit_rate_at_k']:.3f}"
            )
    except (DatasetError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"report: {path}")
    return 0


def main(
    argv: list[str] | None = None, *, client: EmbeddingClient | None = None
) -> int:
    return run(argv, client=client)


if __name__ == "__main__":
    raise SystemExit(main())
