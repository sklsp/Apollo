"""Behaviour tests for app/core/jobs.py (Job dataclass and JobStore)."""

from __future__ import annotations

import json
import threading
import time

import pytest


def _wait_terminal(store, job_id, timeout=2.0):
    """Poll until the job reaches a terminal status; fail if it does not."""
    from app.core.jobs import TERMINAL_STATUSES

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = store.get(job_id)
        if job is not None and job.status in TERMINAL_STATUSES:
            return job
        time.sleep(0.01)
    pytest.fail(f"job {job_id} did not reach a terminal status within {timeout}s")


def _write_jobs_file(path, records):
    path.write_text(json.dumps(records), encoding="utf-8")


# ---------- Job.to_dict / elapsed_seconds ------------------------------------


class TestJobElapsedSeconds:
    def test_elapsed_between_start_and_finish(self):
        from app.core.jobs import Job

        job = Job(
            id="j1",
            type="generate",
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:00:05.500000+00:00",
        )

        assert job.to_dict()["elapsed_seconds"] == 5.5

    def test_elapsed_clamps_negative_to_zero(self):
        from app.core.jobs import Job

        job = Job(
            id="j2",
            type="generate",
            started_at="2026-01-01T00:00:10+00:00",
            completed_at="2026-01-01T00:00:00+00:00",
        )

        assert job.to_dict()["elapsed_seconds"] == 0.0

    def test_elapsed_none_when_not_started(self):
        from app.core.jobs import Job

        data = Job(id="j3", type="caption").to_dict()

        assert data["elapsed_seconds"] is None
        assert data["status"] == "queued"

    def test_elapsed_none_for_unparseable_timestamps(self):
        from app.core.jobs import Job

        job = Job(id="j4", type="generate", started_at="not-a-date")

        assert job.to_dict()["elapsed_seconds"] is None


# ---------- persistence: load --------------------------------------------------


class TestPersistenceLoad:
    def test_load_restores_terminal_jobs_and_drops_unknown_fields(self, tmp_path):
        from app.core.jobs import JobStore

        path = tmp_path / "jobs.json"
        _write_jobs_file(
            path,
            [
                {
                    "id": "done1",
                    "type": "generate",
                    "status": "completed",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "started_at": "2026-01-01T00:00:01+00:00",
                    "completed_at": "2026-01-01T00:00:05+00:00",
                    "progress": 100.0,
                    "message": None,
                    "error": None,
                    "outputs": ["/out/a.png"],
                    "metadata": {"prompt": "a cat"},
                    "elapsed_seconds": 9.9,  # derived field: must be dropped
                },
                {
                    "id": "stop1",
                    "type": "train",
                    "status": "cancelled",
                    "created_at": "2026-01-01T00:01:00+00:00",
                    "started_at": None,
                    "completed_at": "2026-01-01T00:01:02+00:00",
                    "progress": None,
                    "message": None,
                    "error": None,
                    "outputs": [],
                    "metadata": {},
                },
            ],
        )

        store = JobStore(persist_path=str(path))
        done = store.get("done1")
        stopped = store.get("stop1")

        assert done is not None and done.status.value == "completed"
        assert done.outputs == ["/out/a.png"]
        assert done.metadata == {"prompt": "a cat"}
        # the stored 9.9 was dropped; to_dict re-derives it from the timestamps
        assert done.to_dict()["elapsed_seconds"] == 4.0
        assert stopped is not None and stopped.status.value == "cancelled"

    def test_load_marks_interrupted_jobs_failed(self, tmp_path):
        from app.core.jobs import JobStore

        path = tmp_path / "jobs.json"
        _write_jobs_file(
            path,
            [
                {
                    "id": "run1",
                    "type": "generate",
                    "status": "running",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "started_at": "2026-01-01T00:00:01+00:00",
                    "completed_at": None,
                    "progress": 40.0,
                    "message": "halfway",
                    "error": None,
                    "outputs": [],
                    "metadata": {},
                },
                {
                    "id": "wait1",
                    "type": "caption",
                    "status": "queued",
                    "created_at": "2026-01-01T00:00:02+00:00",
                    "started_at": None,
                    "completed_at": None,
                    "progress": None,
                    "message": None,
                    "error": None,
                    "outputs": [],
                    "metadata": {},
                },
            ],
        )

        store = JobStore(persist_path=str(path))

        for job_id in ("run1", "wait1"):
            job = store.get(job_id)
            assert job.status.value == "failed"
            assert job.error == "Interrupted by application restart"
            assert job.completed_at is not None

    def test_load_skips_unreadable_records(self, tmp_path):
        from app.core.jobs import JobStore

        path = tmp_path / "jobs.json"
        _write_jobs_file(
            path,
            [
                {
                    "id": "ok1",
                    "type": "generate",
                    "status": "failed",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "started_at": None,
                    "completed_at": "2026-01-01T00:00:03+00:00",
                    "progress": None,
                    "message": None,
                    "error": "old error",
                    "outputs": [],
                    "metadata": {},
                },
                {"id": "bad1", "type": "x", "status": "bogus"},  # ValueError
                "just a string",  # TypeError in dict(raw)
            ],
        )

        store = JobStore(persist_path=str(path))

        jobs = store.list()
        assert [job.id for job in jobs] == ["ok1"]
        assert jobs[0].error == "old error"

    def test_load_ignores_non_list_file(self, tmp_path):
        from app.core.jobs import JobStore

        path = tmp_path / "jobs.json"
        _write_jobs_file(path, {"id": "not-a-list"})

        store = JobStore(persist_path=str(path))

        assert store.list() == []


# ---------- persistence: save --------------------------------------------------


class TestPersistenceSave:
    def test_save_keeps_only_most_recent_two_hundred(self, tmp_path):
        from app.core.jobs import JobStore

        path = tmp_path / "jobs.json"
        store = JobStore(persist_path=str(path))
        created = [store.create("generate") for _ in range(205)]

        stored = json.loads(path.read_text(encoding="utf-8"))
        assert len(stored) == 200
        assert [job["id"] for job in stored] == [job.id for job in created[5:]]

    def test_save_swallows_oserror_and_keeps_job_in_memory(self, tmp_path):
        from app.core.jobs import JobStore

        # A regular file where the parent directory should be makes every
        # write raise NotADirectoryError (an OSError).
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        store = JobStore(persist_path=str(blocker / "jobs.json"))

        job = store.create("generate")

        assert store.get(job.id) is not None
        assert not (blocker / "jobs.json").exists()



# ---------- submit / worker lifecycle ------------------------------------------


class TestSubmitLifecycle:
    def test_submit_completes_work_with_progress_100(self):
        from app.core.jobs import JobStore

        store = JobStore()

        def work(job):
            store.update(job.id, progress=50.0, message="working")

        job = store.submit("generate", work)
        finished = _wait_terminal(store, job.id)

        assert finished.status.value == "completed"
        assert finished.progress == 100.0
        assert finished.started_at is not None
        assert finished.completed_at is not None
        assert isinstance(finished.to_dict()["elapsed_seconds"], float)

    def test_failed_work_records_error_message(self):
        from app.core.jobs import JobStore

        store = JobStore()

        def work(job):
            raise RuntimeError("boom")

        job = store.submit("train", work)
        finished = _wait_terminal(store, job.id)

        assert finished.status.value == "failed"
        assert finished.error == "boom"

    def test_cancelled_queued_job_never_runs(self):
        from app.core.jobs import JobStore

        store = JobStore(max_workers=1)
        gate = threading.Event()
        ran_second = False

        def first_work(job):
            gate.wait(2.0)  # hold the only worker slot

        def second_work(job):
            nonlocal ran_second
            ran_second = True

        try:
            first = store.submit("generate", first_work)
            second = store.submit("caption", second_work)
            assert store.is_cancelled(second.id) is False
            assert store.cancel(second.id) is True
            assert store.is_cancelled(second.id) is True
        finally:
            gate.set()

        _wait_terminal(store, first.id)
        finished = _wait_terminal(store, second.id)

        assert ran_second is False
        assert finished.status.value == "cancelled"
        assert finished.error is None
        assert store.get(first.id).status.value == "completed"

    def test_worker_terminal_status_is_not_overwritten(self):
        from app.core.jobs import JobStatus, JobStore

        store = JobStore()

        def work(job):
            # cooperative cancellation: the worker itself goes terminal
            store.update(job.id, status=JobStatus.CANCELLED, progress=50.0)

        job = store.submit("generate", work)
        finished = _wait_terminal(store, job.id)

        assert finished.status.value == "cancelled"
        assert finished.progress == 50.0


# ---------- update / list / cancel ----------------------------------------------


class TestUpdateListCancel:
    def test_update_unknown_job_returns_none(self):
        from app.core.jobs import JobStore

        store = JobStore()

        assert store.update("nope", message="x") is None
        assert store.list() == []

    def test_update_terminal_status_fills_completed_at(self):
        from app.core.jobs import JobStatus, JobStore

        store = JobStore()
        job = store.create("generate")
        assert job.completed_at is None

        updated = store.update(job.id, status=JobStatus.COMPLETED)

        assert updated.status.value == "completed"
        assert updated.completed_at is not None

    def test_list_filters_by_type_and_limit(self):
        from app.core.jobs import JobStore

        store = JobStore()
        generated = []
        captioned = []
        for _ in range(2):
            time.sleep(0.002)  # keep created_at values distinct
            generated.append(store.create("generate"))
        for _ in range(3):
            time.sleep(0.002)
            captioned.append(store.create("caption"))

        by_type = store.list(job_type="caption")
        assert [job.id for job in by_type] == [job.id for job in reversed(captioned)]

        limited = store.list(limit=2)
        assert len(limited) == 2
        assert limited[0].id == captioned[-1].id
        assert limited[1].id == captioned[-2].id

    def test_cancel_unknown_and_terminal_jobs(self):
        from app.core.jobs import JobStatus, JobStore

        store = JobStore()
        job = store.create("generate")
        store.update(job.id, status=JobStatus.COMPLETED)

        assert store.cancel("nope") is False
        assert store.is_cancelled("nope") is False
        assert store.cancel(job.id) is False  # already terminal
        assert store.get(job.id).status.value == "completed"
