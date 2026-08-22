"""Integration-pass tests: run history, provenance, workflow validation, dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.run_history import RunHistory
from app.services.dashboard import build_dashboard


# ============================================
# TRAINING RUN HISTORY
# ============================================


class TestRunHistory:
    def test_create_and_read_back(self, project_service, project):
        history = RunHistory(project_service)
        run = history.create_run(
            project.id,
            config={"steps": 1500, "lora_rank": 32},
            base_model="stabilityai/stable-diffusion-xl-base-1.0",
            trigger_word="p3r5on",
            arch="sdxl",
            dataset_image_count=12,
            dataset_captioned_count=12,
            hardware={"accelerator": "cuda", "vram_total_mb": 12288},
            preflight_verdict="ok",
            job_id="job123",
        )
        assert run["status"] == "running"
        assert run["config"]["steps"] == 1500

        # Read back through a fresh instance (restart survival).
        fresh = RunHistory(project_service)
        runs = fresh.list_runs(project.id)
        assert len(runs) == 1
        assert runs[0]["id"] == run["id"]
        assert runs[0]["hardware"]["vram_total_mb"] == 12288

    def test_update_run_outcome(self, project_service, project):
        history = RunHistory(project_service)
        run = history.create_run(
            project.id, config={}, base_model="m", trigger_word="t", arch="sdxl",
            dataset_image_count=1, dataset_captioned_count=1,
            hardware={}, preflight_verdict=None, job_id="j1",
        )
        updated = history.update_run(
            project.id, run["id"], status="completed",
            lora_filename="out.safetensors", lora_size_bytes=123456,
            final_loss=0.082, total_steps=2000,
        )
        assert updated["status"] == "completed"
        assert updated["final_loss"] == pytest.approx(0.082)

        latest = history.latest_run(project.id)
        assert latest["lora_filename"] == "out.safetensors"

    def test_runs_are_newest_first(self, project_service, project):
        import time as time_mod

        history = RunHistory(project_service)
        first = history.create_run(
            project.id, config={}, base_model="m", trigger_word="", arch="sdxl",
            dataset_image_count=0, dataset_captioned_count=0, hardware={},
            preflight_verdict=None, job_id="old",
        )
        time_mod.sleep(1.1)  # timestamps have second resolution
        second = history.create_run(
            project.id, config={}, base_model="m", trigger_word="", arch="sdxl",
            dataset_image_count=0, dataset_captioned_count=0, hardware={},
            preflight_verdict=None, job_id="new",
        )
        runs = history.list_runs(project.id)
        assert [r["id"] for r in runs] == [second["id"], first["id"]]

    def test_history_lives_inside_project_folder(self, project_service, project):
        """Deleting the project removes its history — no orphan records."""
        history = RunHistory(project_service)
        history.create_run(
            project.id, config={}, base_model="m", trigger_word="", arch="sdxl",
            dataset_image_count=0, dataset_captioned_count=0, hardware={},
            preflight_verdict=None, job_id="j",
        )
        runs_file = Path(project_service.project_dir(project.id)) / "runs.json"
        assert runs_file.is_file()

        project_service.delete_project(project.id)
        assert not runs_file.exists()

    def test_find_run_by_job(self, project_service, project):
        history = RunHistory(project_service)
        history.create_run(
            project.id, config={}, base_model="m", trigger_word="", arch="sdxl",
            dataset_image_count=0, dataset_captioned_count=0, hardware={},
            preflight_verdict=None, job_id="special-job",
        )
        match = next(r for r in history.list_runs(project.id)
                     if r.get("job_id") == "special-job")
        history.update_run(project.id, match["id"], status="failed", error="OOM")
        run = history.get_run(project.id, match["id"])
        assert run["error"] == "OOM"


# ============================================
# GENERATION PROVENANCE
# ============================================


class TestGenerationProvenance:
    def _service(self, tmp_settings):
        from app.services.comfyui_service import ComfyUIService

        return ComfyUIService(
            workflow_dir=tmp_settings.comfyui_workflow_dir,
            output_dir=tmp_settings.generated_dir,
        )

    def test_provenance_sidecar_written_for_outputs(self, tmp_settings):
        from app.core.jobs import JobStore

        service = self._service(tmp_settings)
        store = JobStore()
        job = store.submit("comfyui_generation", lambda j: None, metadata={
            "workflow_id": "sdxl_txt2img",
            "seed": 4242,
            "prompt": "a cup",
            "lora_name": None,
            "checkpoint": "sdxl.safetensors",
        })
        import time as time_mod
        time_mod.sleep(0.2)  # let the no-op work finish

        service._write_provenance(job, ["abc_0.png"])

        sidecar = Path(tmp_settings.generated_dir) / "abc_0.provenance.json"
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        assert record["seed"] == 4242
        assert record["prompt"] == "a cup"
        assert record["workflow_id"] == "sdxl_txt2img"
        assert record["image"] == "abc_0.png"
        assert record["created_at"]

    def test_list_generated_attaches_provenance(self, tmp_settings):
        from app.core.jobs import JobStore

        service = self._service(tmp_settings)
        out = Path(tmp_settings.generated_dir)
        (out / "img_0.png").write_bytes(b"fakepng")

        store = JobStore()
        job = store.submit("comfyui_generation", lambda j: None, metadata={
            "workflow_id": "wf", "seed": 7, "prompt": "hello",
        })
        import time as time_mod
        time_mod.sleep(0.2)
        service._write_provenance(job, ["img_0.png"])

        entries = service.list_generated()
        match = next(e for e in entries if e["filename"] == "img_0.png")
        assert match["provenance"]["seed"] == 7
        assert match["provenance"]["prompt"] == "hello"

    def test_list_generated_without_provenance_still_works(self, tmp_settings):
        service = self._service(tmp_settings)
        out = Path(tmp_settings.generated_dir)
        (out / "orphan.png").write_bytes(b"x")
        entries = service.list_generated()
        assert entries[0]["filename"] == "orphan.png"
        assert "provenance" not in entries[0]


# ============================================
# WORKFLOW PRE-VALIDATION
# ============================================


class TestWorkflowValidation:
    def _graph(self):
        return {
            "4": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": "sdxl.safetensors"}},
            "5": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
            "3": {"class_type": "KSampler", "inputs": {
                "seed": 1, "steps": 20, "cfg": 7.0, "sampler_name": "euler",
                "scheduler": "normal", "denoise": 1.0, "model": ["4", 0],
                "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage", "inputs": {"f": "x", "images": ["8", 0]}},
            "10": {"class_type": "LoraLoader", "inputs": {
                "lora_name": "", "strength_model": 1.0, "strength_clip": 1.0,
                "model": ["4", 0], "clip": ["4", 1]}},
        }

    class OfflineStub:
        base_url = "http://stub:8188"

        def ping(self):
            from app.core.exceptions import ComfyUIServiceError
            raise ComfyUIServiceError("down", status_code=503, detail="refused")

        def list_checkpoints(self):
            return []

        def list_loras(self):
            return []

    class OnlineStub(OfflineStub):
        def ping(self):
            return {"system": {}, "devices": []}

        def list_checkpoints(self):
            return ["sdxl.safetensors"]

        def list_loras(self):
            return ["my.safetensors"]

    def _service(self, tmp_settings, client):
        from app.services.comfyui_service import ComfyUIService

        service = ComfyUIService(
            client=client,
            workflow_dir=tmp_settings.comfyui_workflow_dir,
            output_dir=tmp_settings.generated_dir,
        )
        service.save_workflow("flow", self._graph())
        return service

    def test_missing_workflow_fails_fast(self, tmp_settings):
        service = self._service(tmp_settings, self.OnlineStub())
        report = service.validate_workflow_request("nope", {})
        assert report["valid"] is False
        assert any(c["name"] == "Workflow" and not c["passed"]
                   for c in report["checks"])

    def test_offline_comfyui_is_reported_before_queueing(self, tmp_settings):
        service = self._service(tmp_settings, self.OfflineStub())
        report = service.validate_workflow_request("flow", {"prompt": "x"})
        assert report["valid"] is False
        conn = next(c for c in report["checks"] if c["name"] == "ComfyUI connection")
        assert not conn["passed"]
        assert "not reachable" in conn["detail"]

    def test_missing_checkpoint_is_caught(self, tmp_settings):
        service = self._service(tmp_settings, self.OnlineStub())
        report = service.validate_workflow_request(
            "flow", {"checkpoint": "ghost.safetensors"})
        assert report["valid"] is False
        assert any("not installed" in c["detail"] for c in report["checks"])

    def test_missing_lora_is_caught(self, tmp_settings):
        service = self._service(tmp_settings, self.OnlineStub())
        report = service.validate_workflow_request(
            "flow", {"lora_name": "ghost.safetensors"})
        assert report["valid"] is False
        assert any(c["name"] == "LoRA" and not c["passed"] for c in report["checks"])

    def test_valid_request_passes_all_checks(self, tmp_settings):
        service = self._service(tmp_settings, self.OnlineStub())
        report = service.validate_workflow_request("flow", {
            "prompt": "a cat", "checkpoint": "sdxl.safetensors",
            "lora_name": "my.safetensors",
        })
        assert report["valid"] is True
        assert all(c["passed"] for c in report["checks"])

    def test_unmapped_input_is_flagged(self, tmp_settings):
        service = self._service(tmp_settings, self.OnlineStub())
        graph = self._graph()
        del graph["10"]  # remove LoRA node -> lora_name unmapped
        service.save_workflow("plain", graph)
        report = service.validate_workflow_request(
            "plain", {"lora_name": "my.safetensors"})
        assert report["valid"] is False
        assert any("does not map" in c["detail"] for c in report["checks"])


# ============================================
# DASHBOARD
# ============================================


class TestDashboard:
    def _build(self, document_service, rag_service, project_service,
               training_service, comfyui_service, job_store):
        return build_dashboard(
            document_service=document_service,
            rag_service=rag_service,
            project_service=project_service,
            training_service=training_service,
            comfyui_service=comfyui_service,
            job_store=job_store,
        )

    def test_dashboard_shape_with_empty_state(self, tmp_settings):
        from app.core.jobs import JobStore
        from app.services.comfyui_service import ComfyUIService
        from app.services.document_service import DocumentService
        from app.services.lora_dataset_service import LoRAProjectService
        from app.services.lora_training_service import LoRATrainingService
        from app.services.rag.service import RAGService

        data = self._build(
            DocumentService(),
            RAGService(persist_dir=str(tmp_settings.data_dir + "/rag")),
            LoRAProjectService(data_dir=tmp_settings.lora_data_dir),
            LoRATrainingService(),
            ComfyUIService(output_dir=tmp_settings.generated_dir),
            JobStore(),
        )
        assert set(data) >= {"documents", "datasets", "generations", "jobs", "health"}
        assert data["documents"]["count"] == 0
        assert data["health"]["ollama"]["online"] in (True, False)

    def test_dashboard_reflects_real_counts(self, tmp_settings):
        from app.core.jobs import JobStore
        from app.services.comfyui_service import ComfyUIService
        from app.services.document_service import DocumentService
        from app.services.lora_dataset_service import LoRAProjectService
        from app.services.lora_training_service import LoRATrainingService
        from app.services.rag.service import RAGService
        from tests.test_workspace_features import make_png

        docs = DocumentService()
        docs.store_document("content here", source="a.txt")

        projects = LoRAProjectService(data_dir=tmp_settings.lora_data_dir)
        project = projects.create_project("Dash Test")
        projects.add_image(project.id, "a.png", make_png())

        data = self._build(
            docs,
            RAGService(persist_dir=str(tmp_settings.data_dir + "/rag")),
            projects,
            LoRATrainingService(projects=projects),
            ComfyUIService(output_dir=tmp_settings.generated_dir),
            JobStore(),
        )
        assert data["documents"]["count"] == 1
        assert data["datasets"]["project_count"] == 1
        assert data["datasets"]["projects"][0]["name"] == "Dash Test"

    def test_dashboard_survives_a_broken_subsystem(self, tmp_settings):
        """One failing subsystem must not blank the whole dashboard."""
        from app.core.jobs import JobStore
        from app.services.comfyui_service import ComfyUIService
        from app.services.document_service import DocumentService
        from app.services.lora_dataset_service import LoRAProjectService
        from app.services.lora_training_service import LoRATrainingService
        from app.services.rag.service import RAGService

        class BrokenDocs:
            def get_all_documents(self):
                raise RuntimeError("boom")

        data = self._build(
            BrokenDocs(),
            RAGService(persist_dir=str(tmp_settings.data_dir + "/rag")),
            LoRAProjectService(data_dir=tmp_settings.lora_data_dir),
            LoRATrainingService(),
            ComfyUIService(output_dir=tmp_settings.generated_dir),
            JobStore(),
        )
        assert "error" in data["documents"], "broken section reports its error"
        assert data["health"], "rest of dashboard still built"


# ============================================
# RAG DEBUG VIEW
# ============================================


class TestRagDebug:
    def test_debug_query_shows_full_pipeline(self, tmp_path):
        from tests.test_production_hardening import FakeEmbeddingClient
        from app.services.rag.service import RAGService

        service = RAGService(FakeEmbeddingClient(), persist_dir=str(tmp_path / "rag"))
        service.add_document("Debug target content. " * 10,
                             {"doc_id": "d1", "filename": "debug.txt"})

        view = service.query_debug("debug target")
        assert view["query"] == "debug target"
        assert view["retrieved"], "must show retrieved chunks"
        assert all("score" in chunk and "text" in chunk for chunk in view["retrieved"])
        assert "debug target" in view["context"].lower() or view["context"]

    def test_debug_query_on_empty_index_is_honest(self, tmp_path):
        from tests.test_production_hardening import FakeEmbeddingClient
        from app.services.rag.service import RAGService

        service = RAGService(FakeEmbeddingClient(), persist_dir=str(tmp_path / "rag"))
        view = service.query_debug("anything")
        assert view["retrieved"] == []
        assert view["context"] == ""
        assert view["index_chunks"] == 0
