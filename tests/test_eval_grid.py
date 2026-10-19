"""Chunking grid tests (portfolio item 26)."""

from __future__ import annotations

import json

import pytest

from app.core.config import settings
from evals.grid import DEFAULT_CHUNKING_GRID, GridCell, run_grid, write_grid_report
from evals.metrics import MetricsSummary
from tests.fakes import FakeEmbedder


class TestRunGrid:
    def test_default_grid_returns_one_cell_per_entry_in_order(self, synthetic_dataset):
        cells = run_grid(synthetic_dataset, FakeEmbedder(), top_k=3)

        assert len(cells) == len(DEFAULT_CHUNKING_GRID)
        assert [(c.chunk_size, c.overlap) for c in cells] == list(DEFAULT_CHUNKING_GRID)
        assert all(c.num_chunks > 0 for c in cells)
        assert all(c.summary.num_questions == 1 for c in cells)

    def test_production_defaults_are_a_cell(self):
        assert (settings.rag_chunk_size, settings.rag_chunk_overlap) in DEFAULT_CHUNKING_GRID

    def test_custom_grid_and_top_k_are_recorded(self, synthetic_dataset):
        cells = run_grid(
            synthetic_dataset, FakeEmbedder(), grid=[(500, 100), (900, 180)], top_k=2
        )

        assert [(c.chunk_size, c.overlap) for c in cells] == [(500, 100), (900, 180)]
        assert all(c.summary.k == 2 for c in cells)

    def test_top_k_defaults_to_production_setting(self, synthetic_dataset, set_setting):
        set_setting("rag_top_k", 4)

        cells = run_grid(synthetic_dataset, FakeEmbedder(), grid=[(800, 150)])

        assert cells[0].summary.k == settings.rag_top_k == 4

    def test_invalid_cells_raise(self, synthetic_dataset):
        for bad in [(0, 0), (100, -1), (100, 100)]:
            with pytest.raises(ValueError, match="invalid cell"):
                run_grid(synthetic_dataset, FakeEmbedder(), grid=[bad], top_k=2)

    def test_empty_grid_raises(self, synthetic_dataset):
        with pytest.raises(ValueError, match="no cells"):
            run_grid(synthetic_dataset, FakeEmbedder(), grid=[], top_k=2)

    def test_non_positive_top_k_raises(self, synthetic_dataset):
        with pytest.raises(ValueError, match="top_k must be positive"):
            run_grid(synthetic_dataset, FakeEmbedder(), grid=[(800, 150)], top_k=0)


class TestGridCell:
    def test_to_dict_has_all_fields_with_rounded_metrics(self):
        cell = GridCell(
            chunk_size=800,
            overlap=150,
            num_chunks=7,
            summary=MetricsSummary(
                num_questions=3, k=5, mean_recall_at_k=0.987654, mrr=0.5, hit_rate_at_k=1.0
            ),
        )

        assert cell.to_dict() == {
            "chunk_size": 800,
            "overlap": 150,
            "num_chunks": 7,
            "num_questions": 3,
            "k": 5,
            "mean_recall_at_k": 0.9877,
            "mrr": 0.5,
            "hit_rate_at_k": 1.0,
        }


class TestWriteGridReport:
    def test_writes_valid_json_in_cell_order(self, tmp_path):
        cells = [
            GridCell(400, 75, 9, _summary()),
            GridCell(800, 150, 6, _summary()),
        ]

        written = write_grid_report(cells, tmp_path / "grid" / "run.json", top_k=5)

        assert written == (tmp_path / "grid" / "run.json").resolve()
        payload = json.loads(written.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert payload["top_k"] == 5
        assert [c["chunk_size"] for c in payload["cells"]] == [400, 800]

    def test_round_trip_matches_to_dict(self, tmp_path):
        cells = run_grid(_dataset(), FakeEmbedder(), grid=[(600, 120)], top_k=3)
        target = tmp_path / "grid.json"

        write_grid_report(cells, target, top_k=3)

        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["cells"] == [cell.to_dict() for cell in cells]

    def test_empty_cells_raise(self, tmp_path):
        with pytest.raises(ValueError, match="zero cells"):
            write_grid_report([], tmp_path / "grid.json", top_k=5)


def _summary():
    return MetricsSummary(
        num_questions=1, k=3, mean_recall_at_k=0.75, mrr=0.5, hit_rate_at_k=1.0
    )


def _dataset():
    """A minimal two-doc dataset so the round-trip test is self-contained."""
    from evals.dataset import CorpusDoc, EvalDataset, Question

    return EvalDataset(
        docs=(
            CorpusDoc(doc_id="a", title="A", text="alpha " * 60),
            CorpusDoc(doc_id="b", title="B", text="beta " * 120),
        ),
        questions=(Question(id="q1", question="what is alpha?", source_docs=("a",), key_facts=()),),
    )
