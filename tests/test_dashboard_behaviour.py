"""Behaviour tests for app/services/dashboard.py (build_dashboard and probes)."""

from __future__ import annotations

import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests


# ---------- fakes -----------------------------------------------------------


class FakeDocumentService:
    def __init__(self, documents=None, error=None):
        self._documents = list(documents or [])
        self._error = error

    def get_all_documents(self):
        if self._error is not None:
            raise self._error
        return list(self._documents)


class FakeRagService:
    def __init__(self, chunk_count=0, backend="ollama", model="nomic-embed-text"):
        self.chunk_count = chunk_count
        self.embedding_client = SimpleNamespace(backend=backend, model=model)


class FakeImage:
    def __init__(self, caption):
        self.caption = caption


class FakeProjectService:
    def __init__(self, projects=None, images_by_project=None, error=None):
        self._projects = list(projects or [])
        self._images = dict(images_by_project or {})
        self._error = error

    def list_projects(self):
        if self._error is not None:
            raise self._error
        return list(self._projects)

    def list_images(self, project_id):
        return list(self._images.get(project_id, []))


class FakeTrainingService:
    def __init__(self, runs_by_project=None):
        self._runs = dict(runs_by_project or {})

    def list_runs(self, project_id):
        return list(self._runs.get(project_id, []))


class FakeComfyuiService:
    def __init__(self, generated=None, error=None):
        self._generated = list(generated or [])
        self._error = error
        self.limit_calls = []

    def list_generated(self, limit=50):
        self.limit_calls.append(limit)
        if self._error is not None:
            raise self._error
        return list(self._generated)


class BrokenJobStore:
    def __init__(self, error):
        self._error = error

    def list(self, limit=50):
        raise self._error


def make_project(project_id, name="P", trained_lora_path=None):
    return SimpleNamespace(
        id=project_id,
        name=name,
        arch="sdxl",
        trigger_word="p3r5on",
        training_status="idle",
        trained_lora_path=trained_lora_path,
    )


def healthy_fakes():
    return dict(
        document_service=FakeDocumentService(),
        rag_service=FakeRagService(),
        project_service=FakeProjectService(),
        training_service=FakeTrainingService(),
        comfyui_service=FakeComfyuiService(),
        job_store=None,  # filled per test (real JobStore)
    )


def build(fakes):
    from app.services.dashboard import build_dashboard

    return build_dashboard(**fakes)


@pytest.fixture(autouse=True)
def offline_probes(monkeypatch):
    """build_dashboard always probes Ollama/ComfyUI; keep it off the network."""
    calls = []

    def fake_get(url, timeout=None):
        calls.append((url, timeout))
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(requests, "get", fake_get)
    return calls


# ---------- documents section -----------------------------------------------


class TestDocumentsSection:
    def test_reports_counts_and_embedding_info(self, offline_probes):
        fakes = healthy_fakes()
        fakes["document_service"] = FakeDocumentService(documents=["d1", "d2"])
        fakes["rag_service"] = FakeRagService(
            chunk_count=7, backend="sentence-transformers", model="all-MiniLM-L6-v2"
        )

        data = build(fakes)

        assert data["documents"] == {
            "count": 2,
            "indexed_chunks": 7,
            "embedding_backend": "sentence-transformers",
            "embedding_model": "all-MiniLM-L6-v2",
        }

    def test_error_message_is_truncated_to_120_chars(self, offline_probes):
        fakes = healthy_fakes()
        fakes["document_service"] = FakeDocumentService(error=RuntimeError("x" * 300))

        data = build(fakes)

        assert data["documents"] == {"error": "x" * 120}
        # the other sections are still built
        assert set(data) >= {"datasets", "generations", "jobs", "health"}


# ---------- datasets section -------------------------------------------------


class TestDatasetsSection:
    def test_aggregates_images_captions_and_runs(self, offline_probes):
        p1 = make_project("p1", name="Alpha", trained_lora_path="/loras/alpha.safetensors")
        p2 = make_project("p2", name="Beta")
        images = {
            "p1": [FakeImage(""), FakeImage("   "), FakeImage("a cat")],
            "p2": [FakeImage("one"), FakeImage("two")],
        }
        runs = {
            "p1": [{"id": "r2", "status": "completed"}, {"id": "r1", "status": "failed"}],
        }
        fakes = healthy_fakes()
        fakes["project_service"] = FakeProjectService(projects=[p1, p2], images_by_project=images)
        fakes["training_service"] = FakeTrainingService(runs_by_project=runs)

        data = build(fakes)

        datasets = data["datasets"]
        assert datasets["project_count"] == 2
        assert datasets["image_count"] == 5
        assert datasets["lora_count"] == 1
        alpha, beta = datasets["projects"]
        assert alpha == {
            "id": "p1",
            "name": "Alpha",
            "arch": "sdxl",
            "trigger_word": "p3r5on",
            "training_status": "idle",
            "image_count": 3,
            "captioned_count": 1,
            "run_count": 2,
            "latest_run": {"id": "r2", "status": "completed"},
            "trained_lora_path": "/loras/alpha.safetensors",
        }
        assert beta["image_count"] == 2
        assert beta["captioned_count"] == 2
        assert beta["run_count"] == 0
        assert beta["latest_run"] is None
        assert beta["trained_lora_path"] is None

    def test_error_reports_truncated_message(self, offline_probes):
        fakes = healthy_fakes()
        fakes["project_service"] = FakeProjectService(error=RuntimeError("d" * 200))

        data = build(fakes)

        assert data["datasets"] == {"error": "d" * 120}


# ---------- generations section ----------------------------------------------


class TestGenerationsSection:
    def test_passes_limit_12_and_lists_recent(self, offline_probes):
        generated = [{"file": f"g{i}.png"} for i in range(3)]
        comfyui = FakeComfyuiService(generated=generated)
        fakes = healthy_fakes()
        fakes["comfyui_service"] = comfyui

        data = build(fakes)

        assert comfyui.limit_calls == [12]
        assert data["generations"]["recent_count"] == 3
        assert data["generations"]["recent"] == generated

    def test_error_reports_message(self, offline_probes):
        fakes = healthy_fakes()
        fakes["comfyui_service"] = FakeComfyuiService(error=RuntimeError("comfyui down"))

        data = build(fakes)

        assert data["generations"] == {"error": "comfyui down"}


# ---------- jobs section ------------------------------------------------------


class TestJobsSection:
    def test_active_jobs_are_queued_or_running_only(self, offline_probes):
        from app.core.jobs import JobStatus, JobStore

        store = JobStore()
        queued = store.create("caption")
        running = store.create("generate")
        store.update(
            running.id, status=JobStatus.RUNNING, progress=40.0, message="halfway"
        )
        done = store.create("train")
        store.update(done.id, status=JobStatus.COMPLETED)

        fakes = healthy_fakes()
        fakes["job_store"] = store

        data = build(fakes)

        jobs = data["jobs"]
        assert jobs["active_count"] == 2
        # newest first: running (created second), then queued
        assert jobs["active"][0] == {
            "id": running.id,
            "type": "generate",
            "status": "running",
            "progress": 40.0,
            "message": "halfway",
            "elapsed_seconds": None,  # never started (no started_at)
            "created_at": running.created_at,
        }
        assert jobs["active"][1]["id"] == queued.id
        assert jobs["active"][1]["status"] == "queued"
        assert jobs["active"][1]["progress"] is None
        assert jobs["active"][1]["message"] is None

    def test_recent_is_capped_at_six(self, offline_probes):
        from app.core.jobs import JobStore

        store = JobStore()
        created = [store.create("generate") for _ in range(8)]
        newest_first = sorted(created, key=lambda job: job.created_at, reverse=True)

        fakes = healthy_fakes()
        fakes["job_store"] = store

        data = build(fakes)

        recent_ids = [entry["id"] for entry in data["jobs"]["recent"]]
        assert len(recent_ids) == 6
        assert recent_ids == [job.id for job in newest_first[:6]]

    def test_error_reports_message(self, offline_probes):
        fakes = healthy_fakes()
        fakes["job_store"] = BrokenJobStore(RuntimeError("jobs broken"))

        data = build(fakes)

        assert data["jobs"] == {"error": "jobs broken"}


# ---------- health section ----------------------------------------------------


class TestHealthSection:
    def test_probe_reports_online_for_status_below_500(self, offline_probes):
        from app.services.dashboard import _probe

        result = _probe("http://ollama.test/api/version")

        assert result == {"online": True, "url": "http://ollama.test/api/version"}
        assert offline_probes == [("http://ollama.test/api/version", 2.0)]

    def test_probe_reports_offline_for_server_error_status(self, monkeypatch):
        from app.services.dashboard import _probe

        monkeypatch.setattr(
            requests, "get", lambda url, timeout=None: SimpleNamespace(status_code=503)
        )

        assert _probe("http://comfy.test/system_stats") == {
            "online": False,
            "url": "http://comfy.test/system_stats",
        }

    def test_probe_reports_offline_when_connection_fails(self, monkeypatch):
        from app.services.dashboard import _probe

        def fake_get(url, timeout=None):
            raise requests.ConnectionError("connection refused")

        monkeypatch.setattr(requests, "get", fake_get)

        assert _probe("http://down.test/api/version") == {
            "online": False,
            "url": "http://down.test/api/version",
        }

    def test_health_uses_configured_urls_and_toolkit_state(
        self, tmp_settings, set_setting, offline_probes
    ):
        set_setting("ollama_base_url", "http://ollama.test")
        set_setting("comfyui_base_url", "http://comfy.test")

        data = build(healthy_fakes())

        assert data["health"]["ollama"] == {
            "online": True,
            "url": "http://ollama.test/api/version",
        }
        assert data["health"]["comfyui"] == {
            "online": True,
            "url": "http://comfy.test/system_stats",
        }
        # tmp_settings leaves the toolkit unconfigured (empty path)
        assert data["health"]["ai_toolkit"] == {"configured": False, "detail": "not configured"}

    def test_health_reports_configured_toolkit_path(
        self, tmp_settings, set_setting, offline_probes
    ):
        toolkit = Path(tmp_settings.data_dir) / "ai-toolkit"
        toolkit.mkdir(parents=True)
        (toolkit / "run.py").write_text("print('hi')\n", encoding="utf-8")
        interpreter = Path(tmp_settings.data_dir) / "python.exe"
        interpreter.write_text("", encoding="utf-8")
        set_setting("ai_toolkit_path", str(toolkit))
        set_setting("ai_toolkit_python", str(interpreter))

        data = build(healthy_fakes())

        assert data["health"]["ai_toolkit"] == {
            "configured": True,
            "detail": str(toolkit),
        }


# ---------- generated_at -------------------------------------------------------


class TestGeneratedAt:
    def test_is_utc_timestamp(self, offline_probes):
        data = build(healthy_fakes())

        stamp = data["generated_at"]
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", stamp) is not None
        assert stamp.startswith(time.strftime("%Y-%m-%d", time.gmtime()))
