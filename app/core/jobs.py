"""Lightweight background job tracking for long-running local work.

Shared by ComfyUI generation and LoRA training. Deliberately stdlib-only: a
local single-user app does not need Redis or Celery, and a broker would be one
more service that has to be running for the app to boot.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from app.core.paths import read_json, write_json_atomic

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    """A unit of background work and everything the UI needs to render it."""

    id: str
    type: str
    status: JobStatus = JobStatus.QUEUED
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    completed_at: str | None = None
    progress: float | None = None
    message: str | None = None
    error: str | None = None
    outputs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["elapsed_seconds"] = _elapsed_seconds(self)
        return data


def _elapsed_seconds(job: Job) -> float | None:
    """Seconds spent working, or None when the job has not started yet."""
    if not job.started_at:
        return None
    end = job.completed_at or _now()
    try:
        start = datetime.fromisoformat(job.started_at)
        finish = datetime.fromisoformat(end)
    except ValueError:
        return None
    return round(max((finish - start).total_seconds(), 0.0), 1)


class JobStore:
    """Thread-safe job registry with a worker pool and JSON persistence.

    ``max_workers`` is small on purpose: ComfyUI and AI-Toolkit both want the
    whole GPU, so running several at once would only cause OOM.
    """

    def __init__(self, persist_path: str | None = None, max_workers: int = 2) -> None:
        self._jobs: dict[str, Job] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="job"
        )
        self._persist_path = persist_path
        self._load()

    # ---------- persistence ----------

    def _load(self) -> None:
        if not self._persist_path:
            return
        stored = read_json(self._persist_path, default=[])
        if not isinstance(stored, list):
            return
        for raw in stored:
            try:
                raw = dict(raw)
                raw["status"] = JobStatus(raw.get("status", "failed"))
                job = Job(**raw)
            except (TypeError, ValueError) as exc:
                logger.warning("[JOBS] Skipping unreadable job record: %s", exc)
                continue
            # Nothing survives a restart mid-flight; be honest about it rather
            # than leaving a job spinning at "running" forever.
            if job.status in (JobStatus.RUNNING, JobStatus.QUEUED):
                job.status = JobStatus.FAILED
                job.error = "Interrupted by application restart"
                job.completed_at = _now()
            self._jobs[job.id] = job

    def _save(self) -> None:
        if not self._persist_path:
            return
        try:
            # Keep the file bounded; oldest-first, most recent 200 kept.
            recent = sorted(self._jobs.values(), key=lambda j: j.created_at)[-200:]
            write_json_atomic(self._persist_path, [job.to_dict() for job in recent])
        except OSError as exc:
            logger.warning("[JOBS] Could not persist jobs: %s", exc)

    # ---------- api ----------

    def create(self, job_type: str, metadata: dict[str, Any] | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], type=job_type, metadata=metadata or {})
        with self._lock:
            self._jobs[job.id] = job
            self._save()
        return job

    def submit(
        self,
        job_type: str,
        work: Callable[[Job], Any],
        metadata: dict[str, Any] | None = None,
    ) -> Job:
        """Register a job and run ``work(job)`` on the pool. Returns immediately."""
        job = self.create(job_type, metadata)
        self._executor.submit(self._run, job.id, work)
        return job

    def _run(self, job_id: str, work: Callable[[Job], Any]) -> None:
        job = self.get(job_id)
        if job is None:
            return
        if job_id in self._cancelled:
            self.update(job_id, status=JobStatus.CANCELLED, completed_at=_now())
            return

        self.update(job_id, status=JobStatus.RUNNING, started_at=_now())
        try:
            work(job)
        except Exception as exc:  # noqa: BLE001 - a worker must never kill the pool
            logger.exception("[JOBS] Job %s (%s) failed", job_id, job.type)
            self.update(
                job_id,
                status=JobStatus.FAILED,
                error=str(exc),
                completed_at=_now(),
            )
            return

        # A worker may already have set a terminal status (e.g. cancelled).
        current = self.get(job_id)
        if current and current.status not in TERMINAL_STATUSES:
            self.update(
                job_id,
                status=JobStatus.COMPLETED,
                progress=100.0,
                completed_at=_now(),
            )

    def update(self, job_id: str, **fields: Any) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for key, value in fields.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            if job.status in TERMINAL_STATUSES and job.completed_at is None:
                job.completed_at = _now()
            self._save()
            return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, job_type: str | None = None, limit: int = 50) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        if job_type:
            jobs = [job for job in jobs if job.type == job_type]
        jobs.sort(key=lambda job: job.created_at, reverse=True)
        return jobs[:limit]

    def cancel(self, job_id: str) -> bool:
        """Mark a job cancelled. Workers cooperate via :meth:`is_cancelled`."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in TERMINAL_STATUSES:
                return False
            self._cancelled.add(job_id)
            job.status = JobStatus.CANCELLED
            job.completed_at = _now()
            self._save()
            return True

    def is_cancelled(self, job_id: str) -> bool:
        return job_id in self._cancelled

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)
