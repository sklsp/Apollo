"""HTTP-level tests for the ComfyUI and LoRA endpoints.

These check routing, status codes, and that failures come back as clean JSON
detail messages rather than tracebacks.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import ComfyUIServiceError
from app.services.comfyui_service import ComfyUIService
from app.services.lora_dataset_service import LoRAProjectService
from app.services.lora_training_service import LoRATrainingService
from tests.conftest import png_bytes
from tests.test_comfyui import StubClient, sample_graph


@pytest.fixture
def client(tmp_settings):
    """App wired to temp storage and a stub ComfyUI, with no live services."""
    from app.main import create_app

    class StubLLM:
        def health_check(self):
            return {"status": "ok", "ollama_reachable": True, "model": "llama3.2"}

        def chat(self, prompt, model=None):
            return f"echo:{prompt}"

        def list_models(self):
            return ["llama3.2"]

    app = create_app(service=StubLLM())
    app.state.comfyui_service = ComfyUIService(
        client=StubClient(),
        workflow_dir=tmp_settings.comfyui_workflow_dir,
        output_dir=tmp_settings.generated_dir,
        jobs=app.state.job_store,
    )
    app.state.lora_project_service = LoRAProjectService(data_dir=tmp_settings.lora_data_dir)
    app.state.lora_training_service = LoRATrainingService(
        projects=app.state.lora_project_service, jobs=app.state.job_store
    )
    with TestClient(app) as test_client:
        yield test_client


# ============================================
# EXISTING FUNCTIONALITY MUST STILL WORK
# ============================================


class TestNoRegression:
    def test_chat_still_works(self, client):
        response = client.post("/chat", json={"prompt": "hello"})
        assert response.status_code == 200
        # The route assembles system prompt + history + question before calling
        # the LLM, so the stub echoes the whole assembled prompt.
        body = response.json()
        assert body["response"].startswith("echo:")
        assert "USER QUESTION:\nhello" in body["response"]
        assert body["session_id"] == "default"

    def test_health_still_works(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_documents_and_prompts_still_work(self, client):
        assert client.get("/documents").status_code == 200
        assert client.get("/prompts").status_code == 200

    def test_document_upload_still_accepts_txt(self, client):
        response = client.post(
            "/documents/upload",
            files={"file": ("notes.txt", b"hello world", "text/plain")},
        )
        # 200 when embeddings are available, 502/503 when Ollama is down.
        # Either way it must not 404 or 500.
        assert response.status_code in (200, 502, 503)

    def test_frontend_is_served(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "AI Workspace" in response.text


# ============================================
# COMFYUI ENDPOINTS
# ============================================


class TestComfyUIEndpoints:
    def test_status_endpoint(self, client):
        body = client.get("/comfyui/status").json()
        assert body["connected"] is True
        assert body["version"] == "0.33.1"

    def test_status_never_500s_when_comfyui_is_down(self, client):
        client.app.state.comfyui_service.client = StubClient(
            fail=ComfyUIServiceError("down", status_code=503, detail="refused")
        )
        response = client.get("/comfyui/status")
        assert response.status_code == 200
        assert response.json()["connected"] is False

    def test_workflow_import_list_and_delete(self, client):
        created = client.post("/comfyui/workflows", json={
            "name": "Imported Flow", "workflow": sample_graph(), "description": "test",
        })
        assert created.status_code == 200
        workflow_id = created.json()["id"]
        assert "prompt" in created.json()["supported_inputs"]

        listing = client.get("/comfyui/workflows").json()
        assert listing["count"] == 1

        detail = client.get(f"/comfyui/workflows/{workflow_id}").json()
        assert len(detail["nodes"]) == 7

        assert client.delete(f"/comfyui/workflows/{workflow_id}").status_code == 200
        assert client.get("/comfyui/workflows").json()["count"] == 0

    def test_importing_ui_format_returns_400_not_500(self, client):
        response = client.post("/comfyui/workflows", json={
            "name": "bad", "workflow": {"nodes": [], "links": []},
        })
        assert response.status_code == 400
        assert "Export (API)" in response.json()["detail"]

    def test_generating_with_unknown_workflow_is_404(self, client):
        response = client.post("/comfyui/generate", json={"workflow_id": "ghost"})
        assert response.status_code == 404

    def test_missing_generated_image_is_404(self, client):
        assert client.get("/comfyui/images/nothing.png").status_code == 404

    def test_generated_image_path_traversal_is_blocked(self, client):
        response = client.get("/comfyui/images/..%2F..%2F..%2Fwindows%2Fwin.ini")
        assert response.status_code in (400, 404)
        assert b"[fonts]" not in response.content

    def test_unknown_job_is_404(self, client):
        assert client.get("/comfyui/jobs/nope").status_code == 404


# ============================================
# JOBS
# ============================================


class TestJobEndpoints:
    def test_list_is_empty_initially(self, client):
        assert client.get("/jobs").json()["count"] == 0

    def test_job_lifecycle_is_visible(self, client):
        job = client.app.state.job_store.create("test_job")
        listing = client.get("/jobs").json()
        assert listing["count"] == 1
        assert listing["jobs"][0]["status"] == "queued"

        fetched = client.get(f"/jobs/{job.id}").json()
        assert fetched["id"] == job.id

        cancelled = client.delete(f"/jobs/{job.id}")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"

        # Cancelling a finished job is a conflict, not a crash.
        assert client.delete(f"/jobs/{job.id}").status_code == 409

    def test_filter_by_type(self, client):
        client.app.state.job_store.create("type_a")
        client.app.state.job_store.create("type_b")
        assert client.get("/jobs?job_type=type_a").json()["count"] == 1


# ============================================
# LORA ENDPOINTS
# ============================================


class TestLoRAEndpoints:
    def test_project_crud(self, client):
        created = client.post("/loras/projects", json={
            "name": "API Project", "arch": "sdxl", "trigger_word": "tok",
        })
        assert created.status_code == 200
        project_id = created.json()["id"]

        assert client.get("/loras/projects").json()["count"] == 1
        assert client.get(f"/loras/projects/{project_id}").json()["name"] == "API Project"

        updated = client.put(f"/loras/projects/{project_id}", json={"trigger_word": "newtok"})
        assert updated.json()["trigger_word"] == "newtok"

        assert client.delete(f"/loras/projects/{project_id}").status_code == 200
        assert client.get("/loras/projects").json()["count"] == 0

    def test_unknown_project_is_404(self, client):
        assert client.get("/loras/projects/ghost").status_code == 404

    def test_image_upload_and_caption_flow(self, client):
        project_id = client.post("/loras/projects", json={"name": "P"}).json()["id"]

        uploaded = client.post(
            f"/loras/projects/{project_id}/images",
            files=[("files", ("a.png", png_bytes(), "image/png")),
                   ("files", ("b.png", png_bytes(), "image/png"))],
        )
        assert uploaded.status_code == 200
        assert uploaded.json()["count"] == 2

        captioned = client.put(
            f"/loras/projects/{project_id}/captions/a", json={"caption": "tok, a thing"}
        )
        assert captioned.status_code == 200
        assert captioned.json()["caption_edited"] is True

        images = client.get(f"/loras/projects/{project_id}/images").json()
        assert images["count"] == 2

        file_response = client.get(f"/loras/projects/{project_id}/images/a/file")
        assert file_response.status_code == 200
        assert file_response.headers["content-type"] == "image/png"

        assert client.delete(f"/loras/projects/{project_id}/images/a").status_code == 200
        assert client.get(f"/loras/projects/{project_id}/images").json()["count"] == 1

    def test_rejected_upload_returns_400(self, client):
        project_id = client.post("/loras/projects", json={"name": "P"}).json()["id"]
        response = client.post(
            f"/loras/projects/{project_id}/images",
            files={"files": ("evil.exe", b"MZ\x00\x00", "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "Unsupported image type" in response.json()["detail"]

    def test_toolkit_status_reports_unconfigured(self, client):
        body = client.get("/loras/toolkit/status").json()
        assert body["configured"] is False
        assert "sdxl" in body["supported_archs"]

    def test_training_without_toolkit_is_503(self, client):
        project_id = client.post("/loras/projects", json={"name": "P"}).json()["id"]
        response = client.post(f"/loras/projects/{project_id}/train", json={"steps": 10})
        assert response.status_code == 503
        assert "AI_TOOLKIT_PATH" in response.json()["detail"]

    def test_stopping_nothing_is_409(self, client):
        project_id = client.post("/loras/projects", json={"name": "P"}).json()["id"]
        assert client.post(f"/loras/projects/{project_id}/stop").status_code == 409

    def test_training_status_of_fresh_project(self, client):
        project_id = client.post("/loras/projects", json={"name": "P"}).json()["id"]
        body = client.get(f"/loras/projects/{project_id}/training").json()
        assert body["status"] == "not_started"
        assert body["running"] is False
        assert body["progress"] is None

    def test_library_is_empty_initially(self, client):
        assert client.get("/loras").json()["count"] == 0
