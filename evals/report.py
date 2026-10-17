"""Report generation for the RAG evaluation (portfolio item 23).

Turns a list of ``QuestionResult`` into a JSON-serializable report: one
aggregate summary plus every question's retrieval and per-question metrics.
The output is deterministic - no timestamps, floats rounded to four decimals -
so committed reports diff cleanly and the CI regression gate (item 37) can
compare them field by field.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import settings
from evals.metrics import QuestionResult, summarize


def _round(value: float) -> float:
    return round(float(value), 4)


def build_report(
    results: list[QuestionResult],
    *,
    k: int | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable report from *results*.

    ``k`` is the cutoff used for recall/hit-rate and defaults to the production
    setting (``rag_top_k``), matching what the runner retrieves. ``config`` is
    echoed verbatim into the report so each file records how it was produced
    (embedder model, chunking parameters, ...). Raises ``ValueError`` when
    there are no results or *k* is non-positive.
    """
    if k is None:
        k = settings.rag_top_k
    summary = summarize(results, k=k)

    questions = [
        {
            "id": result.question_id,
            "retrieved_doc_ids": list(result.retrieved_doc_ids),
            "relevant_doc_ids": list(result.relevant_doc_ids),
            "recall_at_k": _round(result.recall_at(k)),
            "reciprocal_rank": _round(result.reciprocal_rank()),
            "hit_at_k": result.hit_at(k),
        }
        for result in results
    ]
    return {
        "schema_version": 1,
        "config": dict(config or {}),
        "summary": {
            "num_questions": summary.num_questions,
            "k": k,
            "mean_recall_at_k": _round(summary.mean_recall_at_k),
            "mrr": _round(summary.mrr),
            "hit_rate_at_k": _round(summary.hit_rate_at_k),
        },
        "questions": questions,
    }


def write_report(report: dict[str, Any], path: Path | str) -> Path:
    """Write *report* as pretty-printed JSON to *path*, creating parent dirs.

    Returns the resolved path so callers can report where the file landed.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return target.resolve()
