"""Training run history: one persistent record per LoRA training attempt.

Answers "which dataset and settings created this LoRA?" without duplicating
any data — records reference the project, dataset and job by ID.

Storage: ``<project>/runs.json`` inside each project folder, so deleting a
project removes its history atomically (no cross-folder cleanup needed).
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.core.paths import read_json, write_json_atomic

logger = logging.getLogger(__name__)


@dataclass
class TrainingRun:
    """Everything about one training attempt worth remembering."""

    id: str
    project_id: str
    started_at: str
    status: str = "running"          # running | completed | failed | cancelled
    completed_at: str | None = None

    # Configuration actually used (the generated YAML's key values).
    config: dict[str, Any] = field(default_factory=dict)
    base_model: str = ""
    trigger_word: str = ""
    arch: str = ""

    # Dataset state at training time.
    dataset_image_count: int = 0
    dataset_captioned_count: int = 0

    # Hardware snapshot when the run started.
    hardware: dict[str, Any] = field(default_factory=dict)

    # Preflight verdict at launch time.
    preflight_verdict: str | None = None

    # Outcome.
    job_id: str | None = None
    lora_filename: str | None = None
    lora_size_bytes: int | None = None
    error: str | None = None
    final_loss: float | None = None
    total_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunHistory:
    """Per-project run records stored in ``<project>/runs.json``."""

    def __init__(self, projects_service: Any) -> None:
        self.projects = projects_service

    def _path(self, project_id: str) -> Path:
        return self.projects.project_dir(project_id) / "runs.json"

    def list_runs(self, project_id: str) -> list[dict[str, Any]]:
        data = read_json(self._path(project_id), default=[])
        runs = [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
        runs.sort(key=lambda r: r.get("started_at", ""), reverse=True)
        return runs

    def get_run(self, project_id: str, run_id: str) -> dict[str, Any] | None:
        for run in self.list_runs(project_id):
            if run.get("id") == run_id:
                return run
        return None

    def create_run(
        self,
        project_id: str,
        *,
        config: dict[str, Any],
        base_model: str,
        trigger_word: str,
        arch: str,
        dataset_image_count: int,
        dataset_captioned_count: int,
        hardware: dict[str, Any],
        preflight_verdict: str | None,
        job_id: str,
    ) -> dict[str, Any]:
        run = TrainingRun(
            id=uuid.uuid4().hex[:12],
            project_id=project_id,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            config=config,
            base_model=base_model,
            trigger_word=trigger_word,
            arch=arch,
            dataset_image_count=dataset_image_count,
            dataset_captioned_count=dataset_captioned_count,
            hardware=hardware,
            preflight_verdict=preflight_verdict,
            job_id=job_id,
        ).to_dict()

        runs = self.list_runs(project_id)
        runs.insert(0, run)
        self._save(project_id, runs)
        logger.info("[RUN HISTORY] %s: created run %s", project_id, run["id"])
        return run

    def update_run(self, project_id: str, run_id: str, **fields: Any) -> dict[str, Any] | None:
        runs = self.list_runs(project_id)
        for run in runs:
            if run.get("id") != run_id:
                continue
            for key, value in fields.items():
                if key in TrainingRun.__dataclass_fields__:
                    run[key] = value
            self._save(project_id, runs)
            return run
        return None

    def latest_run(self, project_id: str) -> dict[str, Any] | None:
        runs = self.list_runs(project_id)
        return runs[0] if runs else None

    def _save(self, project_id: str, runs: list[dict[str, Any]]) -> None:
        # Keep the file bounded; a project rarely exceeds a handful of runs.
        write_json_atomic(self._path(project_id), runs[:50])
