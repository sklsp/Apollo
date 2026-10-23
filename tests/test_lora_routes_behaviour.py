"""Behaviour tests for the LoRA API routes (app/api/lora_routes.py).

Complements test_lora.py (service level) and test_api_generation.py (full-app
smoke): these drive the router itself with a minimal app that wires only the
LoRA dependencies, so every route handler runs against real services on temp
data dirs. Ollama, ComfyUI, and the AI Toolkit are intentionally absent - the
tests assert how the API degrades without them (skipped captions, honest
hardware reports, fail-fast jobs) instead of mocking the services out.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.api.lora_routes import _http_error, router as lora_router
from app.core.config import settings
from app.core.exceptions import ServiceError
from app.core.jobs import JobStore, JobStatus
from app.core.paths import UnsafePathError
from app.services.hardware import GPUInfo, HardwareInfo
from app.services.lora_dataset_service import LoRAProjectService
from app.services.lora_training_service import TRAINING_PRESETS, LoRATrainingService
from tests.conftest import png_bytes


def _fake_detect_hardware(comfyui_base_url=None):  # noqa: ARG001 - mirrors the real signature
    """Deterministic stand-in for GPU probing (see ``lora_env``)."""
    return HardwareInfo(
        gpus=[GPUInfo(vendor="nvidia", name="Test GPU", vram_total_mb=8 * 1024,
                      vram_free_mb=7 * 1024, source="test-fake")],
        cpu_name="Test CPU",
        cpu_cores=8,
        ram_total_mb=32 * 1024,
        ram_available_mb=16 * 1024,
        disk_free_gb=100.0,
        cuda_available=True,
    )


@pytest.fixture
def lora_env(tmp_settings, monkeypatch):
    """Minimal FastAPI app with only the LoRA router and real services."""
    # Real GPU detection shells out to nvidia-smi (~2s per call on Windows) and
    # depends on this machine's accelerators. The routes under test only need a
    # HardwareInfo back, so pin it; hardware.py is covered by its own tests.
    monkeypatch.setattr("app.services.hardware.detect_hardware", _fake_detect_hardware)

    projects = LoRAProjectService(data_dir=tmp_settings.lora_data_dir)
    jobs = JobStore()
    training = LoRATrainingService(projects=projects, jobs=jobs)

    app = FastAPI()
    app.include_router(lora_router)
    app.state.lora_project_service = projects
    app.state.lora_training_service = training
    app.state.job_store = jobs

    return SimpleNamespace(client=TestClient(app), projects=projects,
                           training=training, jobs=jobs)


def _create_project(client: TestClient, name="Route Character", **extra):
    response = client.post("/loras/projects", json={"name": name, **extra})
    assert response.status_code == 200, response.text
    return response.json()


def _upload_image(client: TestClient, project_id: str, filename="a.png"):
    response = client.post(
        f"/loras/projects/{project_id}/images",
        files=[("files", (filename, png_bytes(), "image/png"))],
    )
    assert response.status_code == 200, response.text
    return response.json()


def _caption_image(client: TestClient, project_id: str, image_id: str):
    response = client.put(
        f"/loras/projects/{project_id}/captions/{image_id}",
        json={"caption": "a red test subject"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _wait_for_job(jobs: JobStore, job_id: str, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    while True:
        job = jobs.get(job_id)
        if job is not None and job.status in (
            JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED
        ):
            return job
        assert time.monotonic() < deadline, f"job {job_id} did not finish: {job}"
        time.sleep(0.1)


# ============================================
# LIBRARY AND TOOLKIT
# ============================================


class TestLibraryAndToolkit:
    def test_lora_files_alias_matches_library(self, lora_env):
        client = lora_env.client
        library = client.get("/loras")
        files = client.get("/loras/files")
        assert library.status_code == 200 and files.status_code == 200
        assert files.json()["count"] == library.json()["count"]

    def test_training_presets_endpoint_returns_documented_set(self, lora_env):
        response = lora_env.client.get("/loras/training/presets")
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == len(TRAINING_PRESETS)
        ids = [preset["id"] for preset in body["presets"]]
        assert set(ids) == set(TRAINING_PRESETS)
        for preset in body["presets"]:
            assert preset["label"] and preset["description"]
            assert isinstance(preset["values"], dict)

    def test_hardware_presets_reports_tier_and_variants(self, lora_env):
        response = lora_env.client.get("/loras/training/hardware-presets")
        assert response.status_code == 200
        body = response.json()
        assert body["hardware"]["tier"] in {"unknown", "small", "medium", "large"}
        for preset_id, variants in body["presets"].items():
            # Each preset carries a display label plus the three variant tiers.
            assert {"conservative", "balanced", "quality"} <= set(variants), preset_id
            # The conservative variant must always fit on the smallest GPU.
            assert variants["conservative"]["batch_size"] == 1

    def test_hardware_info_endpoint_shape(self, lora_env):
        response = lora_env.client.get("/loras/hardware")
        assert response.status_code == 200
        body = response.json()
        for key in ("gpus", "accelerator", "cpu_cores", "ram_total_mb",
                    "disk_free_gb", "cuda_available"):
            assert key in body, f"missing {key}"
        assert isinstance(body["gpus"], list)

    def test_toolkit_status_reports_unconfigured(self, lora_env):
        response = lora_env.client.get("/loras/toolkit/status")
        assert response.status_code == 200
        body = response.json()
        # No AI_TOOLKIT_PATH in the test environment: honest "not configured".
        assert body["configured"] is False
        assert body["error"]
        assert isinstance(body["supported_archs"], list) and body["supported_archs"]


# ============================================
# PROJECTS
# ============================================


class TestProjects:
    def test_create_project_with_blank_name_is_400(self, lora_env):
        response = lora_env.client.post("/loras/projects", json={"name": "   "})
        assert response.status_code == 400

    def test_delete_missing_project_is_404(self, lora_env):
        response = lora_env.client.delete("/loras/projects/ghost")
        assert response.status_code == 404

    def test_update_project_persists_fields(self, lora_env):
        client = lora_env.client
        project_id = _create_project(client)["id"]
        updated = client.put(
            f"/loras/projects/{project_id}",
            json={"name": "Renamed", "description": "fresh", "arch": "sd15"},
        )
        assert updated.status_code == 200
        fetched = client.get(f"/loras/projects/{project_id}").json()
        assert fetched["name"] == "Renamed"
        assert fetched["description"] == "fresh"
        assert fetched["arch"] == "sd15"

    def test_list_projects_returns_every_created_project(self, lora_env):
        client = lora_env.client
        first = _create_project(client, name="First")["id"]
        second = _create_project(client, name="Second")["id"]

        response = client.get("/loras/projects")
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 2
        assert {project["id"] for project in body["projects"]} == {first, second}

    def test_get_missing_project_is_404(self, lora_env):
        assert lora_env.client.get("/loras/projects/ghost").status_code == 404

    def test_update_missing_project_is_404(self, lora_env):
        response = lora_env.client.put(
            "/loras/projects/ghost", json={"name": "Nope"}
        )
        assert response.status_code == 404

    def test_delete_project_removes_it(self, lora_env):
        client = lora_env.client
        project_id = _create_project(client)["id"]

        response = client.delete(f"/loras/projects/{project_id}")
        assert response.status_code == 200
        assert "deleted" in response.json()["message"].lower()
        # The directory is gone, so every follow-up call 404s.
        assert client.get(f"/loras/projects/{project_id}").status_code == 404

    def test_delete_project_traversal_is_rejected(self, lora_env):
        # Percent-encoded ".." survives the HTTP layer; safe_join must reject
        # it before any filesystem access escapes the data directory.
        response = lora_env.client.delete("/loras/projects/%2e%2e")
        assert response.status_code == 400
        assert "invalid path" in response.json()["detail"].lower()


# ============================================
# DATASET
# ============================================


class TestDataset:
    def test_upload_mixed_files_keeps_valid_and_reports_rejected(self, lora_env):
        client = lora_env.client
        project_id = _create_project(client)["id"]
        response = client.post(
            f"/loras/projects/{project_id}/images",
            files=[("files", ("a.png", png_bytes(), "image/png")),
                   ("files", ("notes.txt", b"not an image", "text/plain"))],
        )
        # One bad file must not discard the good one.
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["count"] == 1
        listed = client.get(f"/loras/projects/{project_id}/images").json()
        assert [image["filename"] for image in listed["images"]] == ["a.png"]

    def test_upload_all_invalid_is_400_with_reasons(self, lora_env):
        client = lora_env.client
        project_id = _create_project(client)["id"]
        response = client.post(
            f"/loras/projects/{project_id}/images",
            files=[("files", ("notes.txt", b"not an image", "text/plain"))],
        )
        assert response.status_code == 400
        assert "notes.txt" in response.json()["detail"]

    def test_image_file_endpoint_serves_bytes_and_media_type(self, lora_env):
        client = lora_env.client
        project_id = _create_project(client)["id"]
        uploaded = _upload_image(client, project_id)
        image_id = uploaded["images"][0]["id"]
        response = client.get(f"/loras/projects/{project_id}/images/{image_id}/file")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/png")
        assert response.content == png_bytes()

    def test_delete_missing_image_is_404(self, lora_env):
        client = lora_env.client
        project_id = _create_project(client)["id"]
        response = client.delete(f"/loras/projects/{project_id}/images/ghost")
        assert response.status_code == 404

    def test_validate_dataset_reports_quality_score(self, lora_env):
        client = lora_env.client
        project_id = _create_project(client)["id"]
        uploaded = _upload_image(client, project_id)
        image_id = uploaded["images"][0]["id"]
        _caption_image(client, project_id, image_id)

        response = client.get(f"/loras/projects/{project_id}/validate")
        assert response.status_code == 200
        body = response.json()
        assert body["total_images"] == 1
        assert body["valid_images"] >= 1
        assert isinstance(body["score"], int) and 0 <= body["score"] <= 100
        assert isinstance(body["findings"], list)

    def test_list_images_missing_project_is_404(self, lora_env):
        response = lora_env.client.get("/loras/projects/ghost/images")
        assert response.status_code == 404

    def test_image_file_unknown_image_is_404(self, lora_env):
        client = lora_env.client
        project_id = _create_project(client)["id"]
        response = client.get(f"/loras/projects/{project_id}/images/ghost/file")
        assert response.status_code == 404

    def test_delete_image_removes_it(self, lora_env):
        client = lora_env.client
        project_id = _create_project(client)["id"]
        image_id = _upload_image(client, project_id)["images"][0]["id"]

        response = client.delete(f"/loras/projects/{project_id}/images/{image_id}")
        assert response.status_code == 200
        listed = client.get(f"/loras/projects/{project_id}/images").json()
        assert listed["count"] == 0

    def test_validate_missing_project_is_404(self, lora_env):
        response = lora_env.client.get("/loras/projects/ghost/validate")
        assert response.status_code == 404

    def test_delete_image_traversal_project_is_rejected(self, lora_env):
        # Percent-encoded ".." survives the HTTP layer; safe_join must reject
        # it before any filesystem access escapes the data directory.
        response = lora_env.client.delete("/loras/projects/%2e%2e/images/whatever")
        assert response.status_code == 400
        assert "invalid path" in response.json()["detail"].lower()


# ============================================
# CAPTIONS (Ollama intentionally down)
# ============================================


class TestCaptions:
    def test_generate_captions_on_empty_project_is_400(self, lora_env):
        client = lora_env.client
        project_id = _create_project(client)["id"]
        response = client.post(
            f"/loras/projects/{project_id}/generate-captions", json={}
        )
        assert response.status_code == 400

    def test_generate_captions_without_ollama_skips_images(self, lora_env):
        client = lora_env.client
        project_id = _create_project(client)["id"]
        uploaded = _upload_image(client, project_id)
        image_id = uploaded["images"][0]["id"]

        response = client.post(
            f"/loras/projects/{project_id}/generate-captions", json={}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        # Ollama is not running in the test environment: every image must be
        # reported as skipped with a reason, never silently dropped.
        assert body["generated"] == 0
        assert body["skipped"] == 1
        result = body["results"][0]
        assert result["image_id"] == image_id
        assert result["skipped"] is True
        assert result["reason"]
        assert body["model"] == settings.vision_model

    def test_caption_job_unknown_project_is_404(self, lora_env):
        response = lora_env.client.post(
            "/loras/projects/ghost/caption-jobs", json={}
        )
        assert response.status_code == 404

    def test_caption_job_empty_dataset_fails_fast_with_400(self, lora_env):
        client = lora_env.client
        project_id = _create_project(client)["id"]
        response = client.post(
            f"/loras/projects/{project_id}/caption-jobs", json={}
        )
        assert response.status_code == 400
        assert "no images" in response.json()["detail"].lower()

    def test_caption_job_runs_to_completion_with_metadata(self, lora_env):
        client = lora_env.client
        project_id = _create_project(client)["id"]
        _upload_image(client, project_id)

        response = client.post(
            f"/loras/projects/{project_id}/caption-jobs", json={}
        )
        assert response.status_code == 200, response.text
        job_id = response.json()["id"]

        job = _wait_for_job(lora_env.jobs, job_id)
        # Ollama is down: the batch degrades to all-skipped and still completes.
        assert job.status is JobStatus.COMPLETED
        assert job.metadata["generated"] == 0
        assert job.metadata["skipped"] == 1
        assert "Captioned 0" in job.message

    def test_update_caption_missing_project_is_404(self, lora_env):
        response = lora_env.client.put(
            "/loras/projects/ghost/captions/some-image", json={"caption": "x"}
        )
        assert response.status_code == 404

    def test_caption_job_reports_progress_through_job_store(self, lora_env, monkeypatch):
        """The job's progress callback must flow into the JobStore.

        Ollama is absent here, so a real batch never reports progress (every
        image skips before the callback fires); stub only the batch method to
        prove the route wires progress_cb into store updates. The batch
        algorithm itself belongs to lora_dataset_service's own tests.
        """
        client = lora_env.client
        project_id = _create_project(client)["id"]
        _upload_image(client, project_id)

        calls: list[tuple[int, int]] = []

        def fake_batch(project_id_, progress_cb=None, cancel_check=None,  # noqa: ARG001
                       model=None, overwrite=False):
            for done in (1, 2):
                if progress_cb:
                    progress_cb(done, 2)
                calls.append((done, 2))
            return [{"image_id": "img", "caption": "a caption", "skipped": False}]

        monkeypatch.setattr(lora_env.projects, "generate_captions_batch", fake_batch)

        response = client.post(f"/loras/projects/{project_id}/caption-jobs")
        assert response.status_code == 200, response.text
        job = _wait_for_job(lora_env.jobs, response.json()["id"])

        assert calls == [(1, 2), (2, 2)]
        assert job.progress == 100.0
        assert job.metadata["generated"] == 1
        assert job.metadata["skipped"] == 0


# ============================================
# TRAINING (AI Toolkit intentionally absent)
# ============================================


class TestTraining:
    def test_preflight_returns_structured_report(self, lora_env):
        client = lora_env.client
        project_id = _create_project(client)["id"]
        uploaded = _upload_image(client, project_id)
        image_id = uploaded["images"][0]["id"]
        _caption_image(client, project_id, image_id)

        response = client.post(
            f"/loras/projects/{project_id}/preflight", json={}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["verdict"] in {"ok", "heavy", "risky", "unsupported", "blocked"}
        names = {check["name"] for check in body["checks"]}
        assert {"Dataset", "Captions", "Base model"} <= names
        assert isinstance(body["summary"], str) and body["summary"]

    def test_write_config_endpoint_persists_yaml_without_training(self, lora_env):
        client = lora_env.client
        project_id = _create_project(client)["id"]
        uploaded = _upload_image(client, project_id)
        image_id = uploaded["images"][0]["id"]
        _caption_image(client, project_id, image_id)

        response = client.post(
            f"/loras/projects/{project_id}/config", json={"steps": 500}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["path"] == "training.yml"
        assert "500" in body["config"]

        config_path = Path(lora_env.projects.config_dir(project_id)) / "training.yml"
        assert config_path.is_file()
        # Previewing the config must not mark the project as training.
        fetched = client.get(f"/loras/projects/{project_id}").json()
        assert fetched["training_status"] in (None, "not_started", "idle", "stopped")

    def test_list_runs_endpoint_returns_empty_history(self, lora_env):
        client = lora_env.client
        project_id = _create_project(client)["id"]
        response = client.get(f"/loras/projects/{project_id}/runs")
        assert response.status_code == 200
        body = response.json()
        assert body["runs"] == [] and body["count"] == 0

    def test_training_log_tails_and_clamps_lines(self, lora_env):
        client = lora_env.client
        project_id = _create_project(client)["id"]

        # No log file yet: empty string, not an error.
        response = client.get(f"/loras/projects/{project_id}/training/log")
        assert response.status_code == 200
        assert response.json()["log"] == ""

        log_path = Path(lora_env.projects.output_dir(project_id)) / "training.log"
        log_path.write_text("\n".join(f"step {i}" for i in range(10)), encoding="utf-8")

        tail = client.get(
            f"/loras/projects/{project_id}/training/log", params={"lines": 3}
        ).json()["log"]
        assert tail.splitlines() == ["step 7", "step 8", "step 9"]

        # lines=0 clamps to 1; huge values clamp to the cap, not an error.
        one = client.get(
            f"/loras/projects/{project_id}/training/log", params={"lines": 0}
        ).json()["log"]
        assert one.splitlines() == ["step 9"]
        all_lines = client.get(
            f"/loras/projects/{project_id}/training/log", params={"lines": 99999}
        ).json()["log"]
        assert len(all_lines.splitlines()) == 10

    def test_preflight_missing_project_is_404(self, lora_env):
        response = lora_env.client.post("/loras/projects/ghost/preflight", json={})
        assert response.status_code == 404

    def test_runs_missing_project_is_404(self, lora_env):
        response = lora_env.client.get("/loras/projects/ghost/runs")
        assert response.status_code == 404

    def test_training_log_missing_project_is_404(self, lora_env):
        response = lora_env.client.get("/loras/projects/ghost/training/log")
        assert response.status_code == 404

    def test_write_config_missing_project_is_404(self, lora_env):
        response = lora_env.client.post(
            "/loras/projects/ghost/config", json={"steps": 500}
        )
        assert response.status_code == 404

    def test_start_training_without_toolkit_fails_with_503(self, lora_env):
        client = lora_env.client
        project_id = _create_project(client)["id"]
        response = client.post(f"/loras/projects/{project_id}/train", json={})
        assert response.status_code == 503
        assert "toolkit" in response.json()["detail"].lower()

    def test_stop_training_with_nothing_running_is_409(self, lora_env):
        client = lora_env.client
        project_id = _create_project(client)["id"]
        response = client.post(f"/loras/projects/{project_id}/stop")
        assert response.status_code == 409

    def test_training_status_reports_idle_for_fresh_project(self, lora_env):
        client = lora_env.client
        project_id = _create_project(client)["id"]
        response = client.get(f"/loras/projects/{project_id}/training")
        assert response.status_code == 200
        body = response.json()
        assert body["running"] is False
        assert body["progress"] is None

    def test_training_status_missing_project_is_404(self, lora_env):
        response = lora_env.client.get("/loras/projects/ghost/training")
        assert response.status_code == 404


# ============================================
# ERROR MAPPING
# ============================================


class TestHttpErrorMapper:
    def test_http_error_mapper_covers_all_branches(self):
        # ServiceError keeps its own status code and detail.
        mapped = _http_error(ServiceError("toolkit missing", status_code=503))
        assert isinstance(mapped, HTTPException)
        assert mapped.status_code == 503
        assert mapped.detail == "toolkit missing"

        # Unsafe paths are rejected without echoing the attempted path.
        mapped = _http_error(UnsafePathError("../etc/passwd"))
        assert mapped.status_code == 400
        assert mapped.detail == "Invalid path"

        # Plain ValueErrors surface their message; anything else is a 500.
        mapped = _http_error(ValueError("bad value"))
        assert (mapped.status_code, mapped.detail) == (400, "bad value")
        mapped = _http_error(RuntimeError("boom"))
        assert (mapped.status_code, mapped.detail) == (500, "Unexpected error")



