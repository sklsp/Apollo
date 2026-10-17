"""Report generation tests (portfolio item 23)."""

from __future__ import annotations

import json

import pytest

from app.core.config import settings
from evals.metrics import QuestionResult, summarize
from evals.report import build_report, write_report


def _results() -> list[QuestionResult]:
    return [
        # alpha is rank 2 of the top-3: recall@3 = 1/2, rr = 0.5, hit True
        QuestionResult(
            question_id="q1",
            retrieved_doc_ids=("gamma", "alpha", "delta"),
            relevant_doc_ids=("alpha", "beta"),
        ),
        # nothing relevant at all: recall 0, rr 0, no hit
        QuestionResult(
            question_id="q2",
            retrieved_doc_ids=("epsilon", "zeta"),
            relevant_doc_ids=("theta",),
        ),
    ]


class TestBuildReport:
    def test_summary_matches_summarize(self):
        report = build_report(_results(), k=3)

        expected = summarize(_results(), k=3)
        assert report["summary"]["num_questions"] == 2
        assert report["summary"]["k"] == 3
        assert report["summary"]["mean_recall_at_k"] == pytest.approx(
            round(expected.mean_recall_at_k, 4), abs=1e-9
        )
        assert report["summary"]["mrr"] == pytest.approx(round(expected.mrr, 4), abs=1e-9)
        assert report["summary"]["hit_rate_at_k"] == pytest.approx(
            round(expected.hit_rate_at_k, 4), abs=1e-9
        )

    def test_per_question_metrics_and_order(self):
        report = build_report(_results(), k=3)

        questions = report["questions"]
        assert [q["id"] for q in questions] == ["q1", "q2"]
        first, second = questions
        assert first["retrieved_doc_ids"] == ["gamma", "alpha", "delta"]
        assert first["relevant_doc_ids"] == ["alpha", "beta"]
        assert first["recall_at_k"] == pytest.approx(0.5)
        assert first["reciprocal_rank"] == pytest.approx(0.5)
        assert first["hit_at_k"] is True
        assert second["recall_at_k"] == 0.0
        assert second["reciprocal_rank"] == 0.0
        assert second["hit_at_k"] is False

    def test_k_defaults_to_production_setting(self, set_setting):
        set_setting("rag_top_k", 2)

        report = build_report(_results())

        assert report["summary"]["k"] == settings.rag_top_k == 2

    def test_config_is_echoed_verbatim(self):
        config = {"embedder": "fake-embed", "chunk_size": 800, "overlap": 150}

        report = build_report(_results(), k=3, config=config)

        assert report["config"] == config

    def test_output_is_deterministic(self):
        first = build_report(_results(), k=3)
        second = build_report(_results(), k=3)

        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    def test_empty_results_raise(self):
        with pytest.raises(ValueError, match="zero question results"):
            build_report([], k=3)


class TestWriteReport:
    def test_writes_valid_json_with_trailing_newline(self, tmp_path):
        target = tmp_path / "reports" / "nested" / "run.json"

        written = write_report(build_report(_results(), k=3), target)

        assert written == target.resolve()
        text = target.read_text(encoding="utf-8")
        assert text.endswith("\n")
        loaded = json.loads(text)
        assert loaded["summary"]["num_questions"] == 2

    def test_round_trip_preserves_content(self, tmp_path):
        report = build_report(_results(), k=3, config={"embedder": "fake"})
        target = tmp_path / "run.json"

        write_report(report, target)

        assert json.loads(target.read_text(encoding="utf-8")) == report
