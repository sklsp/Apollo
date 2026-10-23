"""Behavioural tests for app/api/comfyui_routes.py."""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.api.comfyui_routes import _http_error, jobs_router, router
from app.core.exceptions import ComfyUIServiceError, ServiceError, WorkflowError
from app.core.jobs import TERMINAL_STATUSES, Job, JobStatus
from app.core.paths import UnsafePathError


class FakeWorkflowInfo:
    def __init__(self, data: dict) -> None:
        self._data = dict(data)

    def to_dict(self) -> dict:
        return dict(self._data)


def _job(job_id: str, status: JobStatus = JobStatus.QUEUED, **overrides) -> Job:
    data = {
        "id": job_id,
        "type": "generation",
        "status": status,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    data.update(overrides)
    return Job(**data)


class FakeComfyUIService:
    def __init__(self) -> None:
        self.status_payload = {
            "connected": True,
            "base_url": "http://127.0.0.1:8188",
            "version": "0.3.45",
            "python_version": "3.12.4",
            "devices": [{"name": "NVIDIA GeForce RTX 4090", "vram_gb": 24}],
            "checkpoints": ["sd_xl_base.safetensors"],
            "loras": ["cat_lora.safetensors"],
            "error": None,
        }
        self.workflows = [
            FakeWorkflowInfo({
                "id": "txt2img",
                "name": "SDXL txt2img",
                "description": "Base text to image",
                "arch": "sdxl",
                "inputs": {"prompt": {"3": "text"}},
                "supported_inputs": ["prompt"],
                "node_count": 7,
            }),
            FakeWorkflowInfo({
                "id": "img2img",
                "name": "SDXL img2img",
                "description": "",
                "arch": "sdxl",
                "inputs": {},
                "supported_inputs": [],
                "node_count": 9,
            }),
        ]
        self.save_calls: list[dict] = []
        self.save_error: Exception | None = None
        self.load_result: tuple[dict, dict] = (
            {
                "1": {"class_type": "KSampler"},
                "2": {"class_type": "CheckpointLoaderSimple"},
            },
            {"prompt": {"1": "seed"}},
        )
        self.load_error: Exception | None = None
        self.delete_result = True
        self.delete_error: Exception | None = None
        self.submit_result = _job("gen-new")
        self.submit_calls: list[tuple] = []
        self.submit_error: Exception | None = None
        self.validate_result: dict = {}
        self.validate_calls: list[tuple] = []
        self.validate_error: Exception | None = None
        self.generated: list[dict] = []
        self.list_error: Exception | None = None
        self.list_calls: list[int] = []
        self.lora_result: dict = {}
        self.lora_calls: list[tuple] = []
        self.lora_error: Exception | None = None
        self.image_bytes = b""
        self.read_calls: list[str] = []
        self.read_error: Exception | None = None

    def status(self) -> dict:
        return dict(self.status_payload)

    def list_workflows(self) -> list[FakeWorkflowInfo]:
        return list(self.workflows)

    def save_workflow(
        self,
        name: str,
        graph: dict,
        description: str = "",
        arch: str = "generic",
        inputs: dict | None = None,
    ) -> FakeWorkflowInfo:
        self.save_calls.append({
            "name": name,
            "graph": graph,
            "description": description,
            "arch": arch,
            "inputs": inputs,
        })
        if self.save_error is not None:
            raise self.save_error
        mapping = inputs or {}
        return FakeWorkflowInfo({
            "id": name.lower().replace(" ", "_"),
            "name": name,
            "description": description,
            "arch": arch,
            "inputs": mapping,
            "supported_inputs": sorted(mapping),
            "node_count": len(graph),
        })

    def load_workflow(self, workflow_id: str) -> tuple[dict, dict]:
        if self.load_error is not None:
            raise self.load_error
        return self.load_result

    def delete_workflow(self, workflow_id: str) -> bool:
        if self.delete_error is not None:
            raise self.delete_error
        return self.delete_result

    def submit_generation(self, workflow_id: str, params: dict) -> Job:
        self.submit_calls.append((workflow_id, params))
        if self.submit_error is not None:
            raise self.submit_error
        return self.submit_result

    def validate_workflow_request(self, workflow_id: str, params: dict) -> dict:
        self.validate_calls.append((workflow_id, params))
        if self.validate_error is not None:
            raise self.validate_error
        return self.validate_result

    def list_generated(self, limit: int = 50) -> list[dict]:
        self.list_calls.append(limit)
        if self.list_error is not None:
            raise self.list_error
        return list(self.generated)

    def prepare_lora_test(
        self,
        lora_filename: str,
        workflow_id: str | None = None,
        prompt: str | None = None,
        trigger_word: str | None = None,
    ) -> dict:
        self.lora_calls.append((lora_filename, workflow_id, prompt, trigger_word))
        if self.lora_error is not None:
            raise self.lora_error
        return self.lora_result

    def read_generated(self, filename: str) -> bytes:
        self.read_calls.append(filename)
        if self.read_error is not None:
            raise self.read_error
        return self.image_bytes


class FakeJobStore:
    """Mirrors JobStore.get/list/cancel semantics with fixed timestamps."""

    def __init__(self, jobs: list[Job]) -> None:
        self.jobs = {job.id: job for job in jobs}
        self.list_calls: list[tuple] = []
        self.cancel_calls: list[str] = []

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self, job_type: str | None = None, limit: int = 50) -> list[Job]:
        self.list_calls.append((job_type, limit))
        jobs = [j for j in self.jobs.values() if not job_type or j.type == job_type]
        jobs.sort(key=lambda job: job.created_at, reverse=True)
        return jobs[:limit]

    def cancel(self, job_id: str) -> bool:
        self.cancel_calls.append(job_id)
        job = self.jobs.get(job_id)
        if job is None or job.status in TERMINAL_STATUSES:
            return False
        job.status = JobStatus.CANCELLED
        job.completed_at = "2026-01-01T00:00:10+00:00"
        return True


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router)
    app.include_router(jobs_router)
    app.state.comfyui_service = FakeComfyUIService()
    app.state.job_store = FakeJobStore([
        _job("gen-1"),
        _job(
            "train-1",
            type="training",
            status=JobStatus.COMPLETED,
            started_at="2026-01-01T00:00:05+00:00",
            completed_at="2026-01-01T00:00:35+00:00",
        ),
    ])
    return TestClient(app)


QUEUED_JOB_RESPONSE = {
    "id": "gen-1",
    "type": "generation",
    "status": "queued",
    "created_at": "2026-01-01T00:00:00+00:00",
    "started_at": None,
    "completed_at": None,
    "elapsed_seconds": None,
    "progress": None,
    "message": None,
    "error": None,
    "outputs": [],
    "metadata": {},
}

DONE_JOB_RESPONSE = {
    "id": "train-1",
    "type": "training",
    "status": "completed",
    "created_at": "2026-01-01T00:00:00+00:00",
    "started_at": "2026-01-01T00:00:05+00:00",
    "completed_at": "2026-01-01T00:00:35+00:00",
    "elapsed_seconds": 30.0,
    "progress": None,
    "message": None,
    "error": None,
    "outputs": [],
    "metadata": {},
}


# ============================================
# CONNECTION
# ============================================


def test_status_returns_service_payload(client):
    resp = client.get("/comfyui/status")
    assert resp.status_code == 200
    assert resp.json() == {
        "connected": True,
        "base_url": "http://127.0.0.1:8188",
        "version": "0.3.45",
        "python_version": "3.12.4",
        "devices": [{"name": "NVIDIA GeForce RTX 4090", "vram_gb": 24}],
        "checkpoints": ["sd_xl_base.safetensors"],
        "loras": ["cat_lora.safetensors"],
        "error": None,
    }


def test_status_reports_offline_without_raising(client):
    service = client.app.state.comfyui_service
    service.status_payload.update(connected=False, version=None, error="Connection refused")
    resp = client.get("/comfyui/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["connected"] is False
    assert body["error"] == "Connection refused"


def test_test_connection_returns_same_payload(client):
    resp = client.post("/comfyui/test")
    assert resp.status_code == 200
    body = resp.json()
    assert body["connected"] is True
    assert body["base_url"] == "http://127.0.0.1:8188"


# ============================================
# WORKFLOWS
# ============================================


def test_list_workflows_maps_summaries_and_count(client):
    resp = client.get("/comfyui/workflows")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert body["workflows"][0] == {
        "id": "txt2img",
        "name": "SDXL txt2img",
        "description": "Base text to image",
        "arch": "sdxl",
        "inputs": {"prompt": {"3": "text"}},
        "supported_inputs": ["prompt"],
        "node_count": 7,
    }
    assert body["workflows"][1]["id"] == "img2img"


def test_list_workflows_empty(client):
    client.app.state.comfyui_service.workflows = []
    resp = client.get("/comfyui/workflows")
    assert resp.status_code == 200
    assert resp.json() == {"workflows": [], "count": 0}


def test_import_workflow_passes_all_fields(client):
    service = client.app.state.comfyui_service
    payload = {
        "name": "Flux dev",
        "workflow": {"3": {"class_type": "KSampler"}},
        "description": "Flux pipeline",
        "arch": "flux",
        "inputs": {"prompt": {"6": "text"}},
    }
    resp = client.post("/comfyui/workflows", json=payload)
    assert resp.status_code == 200
    assert service.save_calls == [{
        "name": "Flux dev",
        "graph": {"3": {"class_type": "KSampler"}},
        "description": "Flux pipeline",
        "arch": "flux",
        "inputs": {"prompt": {"6": "text"}},
    }]
    assert resp.json() == {
        "id": "flux_dev",
        "name": "Flux dev",
        "description": "Flux pipeline",
        "arch": "flux",
        "inputs": {"prompt": {"6": "text"}},
        "supported_inputs": ["prompt"],
        "node_count": 1,
    }


def test_import_workflow_applies_defaults(client):
    service = client.app.state.comfyui_service
    resp = client.post(
        "/comfyui/workflows",
        json={"name": "Minimal", "workflow": {"1": {"class_type": "KSampler"}}},
    )
    assert resp.status_code == 200
    call = service.save_calls[0]
    assert call["description"] == ""
    assert call["arch"] == "generic"
    assert call["inputs"] is None
    assert resp.json() == {
        "id": "minimal",
        "name": "Minimal",
        "description": "",
        "arch": "generic",
        "inputs": {},
        "supported_inputs": [],
        "node_count": 1,
    }


def test_import_workflow_maps_service_errors(client):
    service = client.app.state.comfyui_service
    payload = {"name": "Broken", "workflow": {}}

    service.save_error = WorkflowError(
        "bad graph", status_code=400, detail="Node 3 has no class_type")
    resp = client.post("/comfyui/workflows", json=payload)
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Node 3 has no class_type"

    service.save_error = ComfyUIServiceError(
        "ComfyUI unreachable", status_code=502, detail="Connection refused")
    resp = client.post("/comfyui/workflows", json=payload)
    assert resp.status_code == 502
    assert resp.json()["detail"] == "Connection refused"


def test_get_workflow_returns_graph_mapping_and_nodes(client):
    resp = client.get("/comfyui/workflows/txt2img")
    assert resp.status_code == 200
    assert resp.json() == {
        "id": "txt2img",
        "workflow": {
            "1": {"class_type": "KSampler"},
            "2": {"class_type": "CheckpointLoaderSimple"},
        },
        "inputs": {"prompt": {"1": "seed"}},
        "nodes": [
            {"id": "1", "class_type": "KSampler"},
            {"id": "2", "class_type": "CheckpointLoaderSimple"},
        ],
    }


def test_get_workflow_maps_errors(client):
    service = client.app.state.comfyui_service

    service.load_error = WorkflowError(
        "no such workflow", status_code=404, detail="Workflow 'ghost' not found")
    resp = client.get("/comfyui/workflows/ghost")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Workflow 'ghost' not found"

    service.load_error = UnsafePathError("../outside/graph.json")
    resp = client.get("/comfyui/workflows/evil")
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Invalid path"


def test_delete_workflow_found_missing_and_unsafe(client):
    service = client.app.state.comfyui_service

    resp = client.delete("/comfyui/workflows/txt2img")
    assert resp.status_code == 200
    assert resp.json() == {"message": "Workflow txt2img deleted"}

    service.delete_result = False
    resp = client.delete("/comfyui/workflows/ghost")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Workflow 'ghost' not found"

    service.delete_error = UnsafePathError("../outside/graph.json")
    resp = client.delete("/comfyui/workflows/evil")
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Invalid path"


# ============================================
# GENERATION
# ============================================


def test_generate_queues_job_and_strips_none_params(client):
    service = client.app.state.comfyui_service
    resp = client.post(
        "/comfyui/generate", json={"workflow_id": "txt2img", "prompt": "a cat", "seed": 42})
    assert resp.status_code == 200
    assert service.submit_calls == [("txt2img", {"prompt": "a cat", "seed": 42})]
    body = resp.json()
    assert body["id"] == "gen-new"
    assert body["status"] == "queued"
    assert body["elapsed_seconds"] is None


def test_generate_rejects_out_of_range_steps(client):
    service = client.app.state.comfyui_service
    resp = client.post("/comfyui/generate", json={"workflow_id": "txt2img", "steps": 0})
    assert resp.status_code == 422
    assert service.submit_calls == []


def test_generate_maps_errors(client):
    service = client.app.state.comfyui_service

    service.submit_error = ComfyUIServiceError(
        "ComfyUI unreachable", status_code=503, detail="ComfyUI is not running")
    resp = client.post("/comfyui/generate", json={"workflow_id": "txt2img"})
    assert resp.status_code == 503
    assert resp.json()["detail"] == "ComfyUI is not running"

    service.submit_error = UnsafePathError("../outside/graph.json")
    resp = client.post("/comfyui/generate", json={"workflow_id": "evil"})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Invalid path"


def test_validate_generation_reports_gaps(client):
    service = client.app.state.comfyui_service
    service.validate_result = {
        "ok": False,
        "checks": [{"field": "checkpoint", "problem": "not installed"}],
    }
    resp = client.post(
        "/comfyui/validate-generation", json={"workflow_id": "txt2img", "prompt": "x"})
    assert resp.status_code == 200
    assert resp.json() == {
        "ok": False,
        "checks": [{"field": "checkpoint", "problem": "not installed"}],
    }
    assert service.validate_calls == [("txt2img", {"prompt": "x"})]


def test_validate_generation_maps_errors(client):
    service = client.app.state.comfyui_service
    service.validate_error = WorkflowError(
        "no such workflow", status_code=404, detail="Workflow 'ghost' not found")
    resp = client.post("/comfyui/validate-generation", json={"workflow_id": "ghost"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Workflow 'ghost' not found"


def test_list_generated_clamps_limit_and_counts(client):
    service = client.app.state.comfyui_service
    service.generated = [{"filename": "out-1.png"}, {"filename": "out-2.jpg"}]

    resp = client.get("/comfyui/generated", params={"limit": 999})
    assert resp.status_code == 200
    assert resp.json() == {
        "images": [{"filename": "out-1.png"}, {"filename": "out-2.jpg"}],
        "count": 2,
    }

    client.get("/comfyui/generated", params={"limit": 0})
    client.get("/comfyui/generated")
    assert service.list_calls == [200, 1, 50]


def test_list_generated_maps_errors(client):
    service = client.app.state.comfyui_service
    service.list_error = ServiceError(
        "disk gone", status_code=502, detail="Output folder missing")
    resp = client.get("/comfyui/generated")
    assert resp.status_code == 502
    assert resp.json()["detail"] == "Output folder missing"


# ============================================
# LORA TESTS
# ============================================


def test_lora_test_passes_all_fields(client):
    service = client.app.state.comfyui_service
    service.lora_result = {"job_id": "train-1", "status": "queued"}
    resp = client.get(
        "/comfyui/test-lora",
        params={
            "lora": "cat_lora.safetensors",
            "workflow_id": "txt2img",
            "prompt": "a cat",
            "trigger_word": "cat",
        },
    )
    assert resp.status_code == 200
    assert service.lora_calls == [("cat_lora.safetensors", "txt2img", "a cat", "cat")]
    assert resp.json() == {"job_id": "train-1", "status": "queued"}


def test_lora_test_defaults_and_error(client):
    service = client.app.state.comfyui_service

    resp = client.get("/comfyui/test-lora", params={"lora": "cat.safetensors"})
    assert resp.status_code == 200
    assert service.lora_calls[0] == ("cat.safetensors", None, None, None)

    service.lora_error = WorkflowError(
        "no such workflow", status_code=404, detail="Workflow 'ghost' not found")
    resp = client.get("/comfyui/test-lora", params={"lora": "cat.safetensors"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Workflow 'ghost' not found"


# ============================================
# JOBS ROUTER
# ============================================


def test_get_job_returns_full_response(client):
    resp = client.get("/jobs/gen-1")
    assert resp.status_code == 200
    assert resp.json() == QUEUED_JOB_RESPONSE

    resp = client.get("/jobs/train-1")
    assert resp.status_code == 200
    assert resp.json() == DONE_JOB_RESPONSE


def test_get_job_missing(client):
    resp = client.get("/jobs/nope")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Job nope not found"


def test_list_jobs_filters_sorts_and_limits(client):
    store = client.app.state.job_store

    resp = client.get("/jobs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert [job["id"] for job in body["jobs"]] == ["gen-1", "train-1"]
    assert store.list_calls[-1] == (None, 50)

    resp = client.get("/jobs", params={"job_type": "training", "limit": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert [job["id"] for job in body["jobs"]] == ["train-1"]
    assert store.list_calls[-1] == ("training", 1)


def test_cancel_job_marks_cancelled(client):
    store = client.app.state.job_store
    resp = client.delete("/jobs/gen-1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "gen-1"
    assert body["status"] == "cancelled"
    assert body["completed_at"] == "2026-01-01T00:00:10+00:00"
    assert body["elapsed_seconds"] is None  # never started
    assert store.cancel_calls == ["gen-1"]


def test_cancel_job_missing_returns_404(client):
    resp = client.delete("/jobs/nope")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Job nope not found"


def test_cancel_finished_job_returns_409(client):
    resp = client.delete("/jobs/train-1")
    assert resp.status_code == 409
    assert resp.json()["detail"] == "Job train-1 has already finished"


def test_get_generation_job_via_comfyui_prefix(client):
    resp = client.get("/comfyui/jobs/gen-1")
    assert resp.status_code == 200
    assert resp.json() == QUEUED_JOB_RESPONSE

    resp = client.get("/comfyui/jobs/nope")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Job nope not found"


# ============================================
# GENERATED IMAGES
# ============================================


def test_generated_image_streams_bytes_with_content_type(client):
    service = client.app.state.comfyui_service
    service.image_bytes = b"\x89PNG fake bytes"
    resp = client.get("/comfyui/images/out-1.png")
    assert resp.status_code == 200
    assert resp.content == b"\x89PNG fake bytes"
    assert resp.headers["content-type"] == "image/png"
    assert service.read_calls == ["out-1.png"]


def test_generated_image_jpeg_content_type(client):
    client.app.state.comfyui_service.image_bytes = b"jpeg-bytes"
    resp = client.get("/comfyui/images/out-2.jpg")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"


def test_generated_image_maps_errors(client):
    service = client.app.state.comfyui_service

    service.read_error = UnsafePathError("../outside/secret.png")
    resp = client.get("/comfyui/images/evil.png")
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Invalid path"

    service.read_error = ServiceError(
        "file gone", status_code=502, detail="Image 'ghost.png' not found")
    resp = client.get("/comfyui/images/ghost.png")
    assert resp.status_code == 502
    assert resp.json()["detail"] == "Image 'ghost.png' not found"


# ============================================
# _http_error HELPER
# ============================================


def test_http_error_uses_exception_status_and_detail():
    exc = WorkflowError("bad", status_code=418, detail="teapot")
    err = _http_error(exc)
    assert isinstance(err, HTTPException)
    assert err.status_code == 418
    assert err.detail == "teapot"


def test_http_error_defaults_to_502_and_message():
    exc = ComfyUIServiceError("ComfyUI unreachable")
    err = _http_error(exc)
    assert isinstance(err, HTTPException)
    assert err.status_code == 502
    assert err.detail == "ComfyUI unreachable"


def test_http_error_falls_back_to_500_for_unknown():
    exc = RuntimeError("boom")
    err = _http_error(exc)
    assert isinstance(err, HTTPException)
    assert err.status_code == 500
    assert err.detail == "Unexpected error"
