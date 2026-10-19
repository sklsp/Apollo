"""Chunking-parameter grid search for the RAG evaluation (portfolio item 26).

Runs the full evaluation once per ``(chunk_size, overlap)`` cell so retrieval
quality can be compared across chunk granularities. Each cell builds its own
index with :func:`evals.index.build_index` and scores every question at a fixed
cutoff; results come back as one row per cell in grid order. Item 27 runs this
live and records the numbers.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings
from evals.dataset import EvalDataset
from evals.index import BatchEmbedder, build_index
from evals.metrics import MetricsSummary, summarize
from evals.runner import run_retrieval

#: Default grid: production defaults (800/150) plus finer and coarser chunks.
DEFAULT_CHUNKING_GRID: tuple[tuple[int, int], ...] = (
    (400, 75),
    (600, 120),
    (800, 150),
    (1200, 200),
)


@dataclass(frozen=True)
class GridCell:
    """The evaluated outcome of one chunking configuration."""

    chunk_size: int
    overlap: int
    num_chunks: int
    summary: MetricsSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "num_chunks": self.num_chunks,
            "num_questions": self.summary.num_questions,
            "k": self.summary.k,
            "mean_recall_at_k": round(self.summary.mean_recall_at_k, 4),
            "mrr": round(self.summary.mrr, 4),
            "hit_rate_at_k": round(self.summary.hit_rate_at_k, 4),
        }


def run_grid(
    dataset: EvalDataset,
    embedder: BatchEmbedder,
    *,
    grid: Sequence[tuple[int, int]] | None = None,
    top_k: int | None = None,
) -> list[GridCell]:
    """Evaluate every ``(chunk_size, overlap)`` cell in *grid*.

    Each cell gets a fresh index built with its own chunking parameters and is
    scored over the whole question set at cutoff *top_k* (default: the
    production setting). Cells are evaluated in grid order; an invalid cell or
    a build failure raises before any later cell runs.
    Raises:
        ValueError: on an empty grid, a non-positive *top_k*, or a cell whose
            parameters do not satisfy ``0 <= overlap < chunk_size``.
    """
    if grid is None:
        grid = DEFAULT_CHUNKING_GRID
    if top_k is None:
        top_k = settings.rag_top_k
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if not grid:
        raise ValueError("grid has no cells to run")
    for chunk_size, overlap in grid:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                f"invalid cell ({chunk_size}, {overlap}): need 0 <= overlap < chunk_size"
            )

    cells: list[GridCell] = []
    for chunk_size, overlap in grid:
        built = build_index(dataset, embedder, chunk_size=chunk_size, overlap=overlap)
        results = run_retrieval(dataset, built.store, embedder, top_k=top_k)
        cells.append(
            GridCell(chunk_size, overlap, built.num_chunks, summarize(results, k=top_k))
        )
    return cells


def write_grid_report(
    cells: list[GridCell], path: Path | str, *, top_k: int, config: dict[str, Any] | None = None
) -> Path:
    """Write the grid results as deterministic JSON (same rules as item 23).

    The file records *top_k* once at the top level, an optional echoed *config*
    block (embedder backend/model, ...), and one row per cell in run order;
    parent directories are created. Returns the resolved path.
    Raises ``ValueError`` when there are no cells to write.
    """
    if not cells:
        raise ValueError("cannot write a grid report with zero cells")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"schema_version": 1}
    if config is not None:
        payload["config"] = dict(config)
    payload["top_k"] = top_k
    payload["cells"] = [cell.to_dict() for cell in cells]
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return target.resolve()
