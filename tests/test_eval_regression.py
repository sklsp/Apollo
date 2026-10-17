"""CI regression gate for RAG retrieval quality (portfolio item 37).

Runs the real service-mode pipeline against the committed dataset with the
sentence-transformers backend pinned (Ollama is never probed), and asserts that
retrieval quality has not regressed relative to
``evals/baselines/vector_service.json``. The baseline was measured in portfolio
item 35 (vector mode, top_k=4, production chunking): recall@4 = 0.9896,
MRR = 0.9757, hit_rate@4 = 1.0.

Re-baseline with ``python -m scripts.regen_eval_baseline`` after an intentional
change to chunking, embeddings or retrieval logic. The comparison is one-sided:
improvements pass; any metric dropping by more than float noise, or a question
losing its hit, fails the gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.clients.ollama_client import OllamaClient
from app.core.config import settings
from app.core.exceptions import OllamaServiceError
from app.services.rag.embeddings import EmbeddingClient
from app.services.rag.service import RAGService
from evals.dataset import load_dataset
from evals.report import build_report
from evals.runner import index_dataset, run_retrieval_on_service

BASELINE_PATH = Path(__file__).resolve().parents[1] / "evals" / "baselines" / "vector_service.json"
TOP_K = 4
# Well below the granularity of a single question flipping (1/48 ≈ 0.02), so this
# absorbs float noise only, never a real regression.
TOLERANCE = 1e-3

SUMMARY_METRICS = ("mean_recall_at_k", "mrr", "hit_rate_at_k")


class _OllamaDown(OllamaClient):
    """``OllamaClient`` that always reports unreachable, forcing the fallback backend."""

    def embed(self, *, model: str, inputs: list[str]) -> list[list[float]]:
        raise OllamaServiceError("Ollama disabled in the regression gate", status_code=503)


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def current_report(tmp_path_factory: pytest.TempPathFactory) -> dict:
    dataset = load_dataset()
    client = EmbeddingClient(ollama_client=_OllamaDown())
    persist_dir = tmp_path_factory.mktemp("rag-regression")
    service = RAGService(embedding_client=client, persist_dir=str(persist_dir))
    index_dataset(service, dataset)
    results = run_retrieval_on_service(dataset, service, top_k=TOP_K, mode="vector")
    return build_report(
        results,
        k=TOP_K,
        config={
            "mode": "service",
            "retrieval_mode": "vector",
            "embedding_backend": client.backend,
            "embedder_model": (
                client.model if client.backend == "ollama" else client.fallback_model
            ),
            "top_k": TOP_K,
        },
    )


def test_baseline_file_is_valid(baseline: dict) -> None:
    assert baseline["schema_version"] == 1
    summary = baseline["summary"]
    for metric in SUMMARY_METRICS:
        assert isinstance(summary[metric], float)
    assert summary["num_questions"] > 0
    assert len(baseline["questions"]) == summary["num_questions"]


def test_gate_configuration_matches_baseline(
    current_report: dict, baseline: dict
) -> None:
    config = current_report["config"]
    # Like-for-like comparison: the gate must run on the same backend and mode as
    # the baseline, with production chunking.
    assert config["embedding_backend"] == "sentence-transformers"
    assert config["retrieval_mode"] == "vector"
    assert config["embedding_backend"] == baseline["config"]["embedding_backend"]
    assert current_report["summary"]["k"] == baseline["summary"]["k"]
    assert settings.rag_chunk_size == baseline["config"]["chunk_size"], (
        "Production chunking changed; re-baseline with `python -m scripts.regen_eval_baseline`"
    )
    assert settings.rag_chunk_overlap == baseline["config"]["overlap"], (
        "Production chunking changed; re-baseline with `python -m scripts.regen_eval_baseline`"
    )


def test_no_summary_metric_regression(current_report: dict, baseline: dict) -> None:
    current = current_report["summary"]
    pinned = baseline["summary"]
    assert current["num_questions"] == pinned["num_questions"], (
        "The eval dataset changed; re-baseline with `python -m scripts.regen_eval_baseline`"
    )
    for metric in SUMMARY_METRICS:
        assert current[metric] >= pinned[metric] - TOLERANCE, (
            f"{metric} regressed from {pinned[metric]:.4f} to {current[metric]:.4f}; "
            "re-baseline with `python -m scripts.regen_eval_baseline` if the change is intentional"
        )


def test_no_question_lost_its_hit(current_report: dict, baseline: dict) -> None:
    pinned_hits = {q["id"]: q["hit_at_k"] for q in baseline["questions"]}
    current_questions = current_report["questions"]
    assert {q["id"] for q in current_questions} == set(pinned_hits), (
        "The eval dataset changed; re-baseline with `python -m scripts.regen_eval_baseline`"
    )
    lost = [q["id"] for q in current_questions if pinned_hits[q["id"]] and not q["hit_at_k"]]
    assert not lost, f"These questions lost their hit@{TOP_K}: {lost}"
