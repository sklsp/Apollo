"""CLI entry-point tests (portfolio item 24).

The embedder client is injected so these run without Ollama; the real
``EmbeddingClient`` is only constructed by ``main`` in production runs.
"""

from __future__ import annotations

import json

import pytest

from app.core.config import settings
from evals.cli import build_parser, main, run
from tests.fakes import FakeServiceEmbedder


@pytest.fixture(autouse=True)
def _fake_client():
    return FakeServiceEmbedder()


class TestParser:
    def test_defaults(self):
        args = build_parser().parse_args([])

        assert args.top_k is None
        assert args.chunk_size is None
        assert args.overlap is None
        assert args.service is False
        assert args.grid is False
        assert str(args.out).endswith("rag_eval.json")

    def test_flags(self, tmp_path):
        out = tmp_path / "r.json"
        args = build_parser().parse_args(
            ["--top-k", "7", "--chunk-size", "500", "--overlap", "90", "--out", str(out), "--service"]
        )

        assert (args.top_k, args.chunk_size, args.overlap) == (7, 500, 90)
        assert args.service is True
        assert args.out == out


class TestRun:
    def test_index_mode_writes_report(self, _fake_client, tmp_path):
        out = tmp_path / "reports" / "index.json"

        code = run(["--top-k", "3", "--out", str(out)], client=_fake_client)

        assert code == 0
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["config"]["mode"] == "index"
        assert report["config"]["embedder_model"] == "fake-embed"
        assert report["config"]["embedding_backend"] == "fake"
        assert report["summary"]["num_questions"] == 48
        assert report["summary"]["k"] == 3

    def test_service_mode_writes_report(self, _fake_client, tmp_path):
        out = tmp_path / "service.json"

        code = run(["--top-k", "2", "--out", str(out), "--service"], client=_fake_client)

        assert code == 0
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["config"]["mode"] == "service"
        assert report["summary"]["num_questions"] == 48
        # service mode uses the production chunking defaults, not CLI overrides
        assert "chunk_size" not in report["config"]

    def test_chunk_overrides_are_recorded(self, _fake_client, tmp_path):
        out = tmp_path / "chunks.json"

        code = run(
            ["--top-k", "2", "--chunk-size", "500", "--overlap", "90", "--out", str(out)],
            client=_fake_client,
        )

        assert code == 0
        config = json.loads(out.read_text(encoding="utf-8"))["config"]
        assert (config["chunk_size"], config["overlap"]) == (500, 90)

    def test_top_k_defaults_to_production_setting(self, _fake_client, tmp_path, set_setting):
        out = tmp_path / "default.json"
        set_setting("rag_top_k", 2)

        code = run(["--out", str(out)], client=_fake_client)

        assert code == 0
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["summary"]["k"] == settings.rag_top_k == 2

    def test_invalid_top_k_fails_cleanly(self, _fake_client, tmp_path):
        out = tmp_path / "bad.json"

        code = run(["--top-k", "0", "--out", str(out)], client=_fake_client)

        assert code == 1
        assert not out.exists()

    def test_hybrid_mode_requires_service_flag(self, _fake_client, tmp_path):
        out = tmp_path / "hybrid.json"

        code = run(["--hybrid-mode", "vector", "--out", str(out)], client=_fake_client)

        assert code == 1
        assert not out.exists()

    def test_hybrid_mode_recorded_in_report(self, _fake_client, tmp_path):
        out = tmp_path / "hybrid.json"

        code = run(
            ["--top-k", "2", "--out", str(out), "--service", "--hybrid-mode", "keyword"],
            client=_fake_client,
        )

        assert code == 0
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["config"]["mode"] == "service"
        assert report["config"]["retrieval_mode"] == "keyword"

    def test_service_mode_records_default_retrieval_mode(self, _fake_client, tmp_path):
        out = tmp_path / "default-mode.json"

        code = run(["--top-k", "2", "--out", str(out), "--service"], client=_fake_client)

        assert code == 0
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["config"]["retrieval_mode"] == settings.rag_hybrid


    def test_fallback_backend_records_actual_model(self, tmp_path):
        """When Ollama is down and sentence-transformers serves the vectors, the
        report must name the model that actually ran."""

        class FallbackClient(FakeServiceEmbedder):
            backend = "sentence-transformers"
            fallback_model = "all-MiniLM-L6-v2"

        out = tmp_path / "fallback.json"

        code = run(["--top-k", "3", "--out", str(out)], client=FallbackClient())

        assert code == 0
        config = json.loads(out.read_text(encoding="utf-8"))["config"]
        assert config["embedding_backend"] == "sentence-transformers"
        assert config["embedder_model"] == "all-MiniLM-L6-v2"


    def test_grid_mode_writes_one_row_per_cell(self, _fake_client, tmp_path):
        out = tmp_path / "grid.json"

        code = run(["--top-k", "3", "--out", str(out), "--grid"], client=_fake_client)

        assert code == 0
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["config"]["mode"] == "grid"
        assert report["top_k"] == 3
        assert [c["chunk_size"] for c in report["cells"]] == [400, 600, 800, 1200]
        assert all(c["num_questions"] == 48 and c["k"] == 3 for c in report["cells"])

    def test_grid_rejects_combined_flags(self, _fake_client, tmp_path):
        out = tmp_path / "bad.json"

        code = run(["--grid", "--service", "--out", str(out)], client=_fake_client)

        assert code == 1
        assert not out.exists()


class TestMain:
    def test_main_returns_run_code(self, _fake_client, tmp_path):
        out = tmp_path / "main.json"

        code = main(["--top-k", "2", "--out", str(out)], client=_fake_client)

        assert code == 0
        assert json.loads(out.read_text(encoding="utf-8"))["summary"]["num_questions"] == 48
