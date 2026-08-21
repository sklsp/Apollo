"""Dataset validation, training presets, caption jobs, and LoRA test setup."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from app.services.dataset_validation import validate_dataset
from app.services.lora_training_service import LoRATrainingService
from tests.conftest import png_bytes


def make_png(width: int = 512, height: int = 512, color=(200, 30, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def make_gradient_png(width: int = 512, height: int = 512, seed: int = 0) -> bytes:
    """A textured image so perceptual hashes differ between files."""
    buffer = io.BytesIO()
    image = Image.new("RGB", (width, height))
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            shade = (x + y + seed * 40) % 256
            for dy in range(8):
                for dx in range(8):
                    if x + dx < width and y + dy < height:
                        image.putpixel((x + dx, y + dy), (shade, shade // 2, 255 - shade))
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ============================================
# DATASET VALIDATION
# ============================================


class TestDatasetValidation:
    def _read(self, store: dict[str, bytes]):
        return lambda image_id: store[image_id]

    def test_clean_dataset_scores_100(self):
        store = {
            "a": make_gradient_png(seed=0),
            "b": make_gradient_png(seed=1),
            "c": make_gradient_png(seed=2),
        }
        images = [
            {"id": key, "filename": f"{key}.png", "caption": f"a {key} subject"}
            for key in store
        ]
        report = validate_dataset(images, self._read(store)).to_dict()
        assert report["score"] == 100
        assert report["valid_images"] == 3
        assert report["issue_count"] == 0
        assert report["captioned"] == 3

    def test_corrupted_image_is_flagged(self):
        store = {"a": b"this is not a png at all"}
        images = [{"id": "a", "filename": "a.png", "caption": "caption"}]
        report = validate_dataset(images, self._read(store)).to_dict()
        assert report["invalid_images"] == 1
        assert any(f["kind"] == "invalid" for f in report["findings"])
        assert report["score"] < 100

    def test_exact_duplicates_are_detected_across_formats(self):
        # Same pixels, different encodes — must still match.
        buffer = io.BytesIO()
        Image.new("RGB", (300, 300), (10, 20, 30)).save(buffer, format="PNG")
        png_data = buffer.getvalue()
        image = Image.open(io.BytesIO(png_data))
        jpeg_buffer = io.BytesIO()
        image.save(jpeg_buffer, format="JPEG")

        store = {"a": png_data, "b": jpeg_buffer.getvalue()}
        images = [
            {"id": "a", "filename": "a.png", "caption": "one"},
            {"id": "b", "filename": "b.jpg", "caption": "two"},
        ]
        report = validate_dataset(images, self._read(store)).to_dict()
        assert report["duplicates"] >= 1

    def test_tiny_images_flag_extreme_resolution(self):
        store = {"tiny": make_png(64, 64)}
        images = [{"id": "tiny", "filename": "tiny.png", "caption": "x"}]
        report = validate_dataset(images, self._read(store)).to_dict()
        assert any(f["kind"] == "extreme_resolution" for f in report["findings"])

    def test_missing_captions_are_counted(self):
        store = {"a": make_png()}
        images = [{"id": "a", "filename": "a.png", "caption": ""}]
        report = validate_dataset(images, self._read(store)).to_dict()
        assert report["missing_captions"] == 1
        assert report["score"] == 95

    def test_suspicious_captions_are_flagged(self):
        store = {"a": make_png()}
        images = [{"id": "a", "filename": "a.png",
                   "caption": "As an AI language model I cannot see images"}]
        report = validate_dataset(images, self._read(store)).to_dict()
        assert any(f["kind"] == "suspicious_caption" for f in report["findings"])

    def test_average_resolution_is_reported(self):
        store = {"a": make_png(256, 512), "b": make_png(768, 512)}
        images = [
            {"id": "a", "filename": "a.png", "caption": "x"},
            {"id": "b", "filename": "b.png", "caption": "y"},
        ]
        report = validate_dataset(images, self._read(store)).to_dict()
        assert report["average_resolution"] == [512, 512]

    def test_score_never_goes_below_zero(self):
        store = {key: b"garbage" for key in ("a", "b", "c", "d")}
        images = [{"id": key, "filename": f"{key}.png", "caption": ""}
                  for key in store]
        report = validate_dataset(images, self._read(store)).to_dict()
        assert report["score"] == 0

    def test_service_endpoint_shape(self, project_service, project):
        project_service.add_image(project.id, "ok.png", make_png())
        project_service.set_caption(project.id, "ok", "p3r5on, portrait")
        project_service.add_image(project.id, "broken.png", b"not an image")

        report = project_service.validate_dataset(project.id)
        assert report["total_images"] == 2
        assert report["invalid_images"] == 1
        assert 0 <= report["score"] <= 100


# ============================================
# TRAINING PRESETS
# ============================================


class TestTrainingPresets:
    def test_presets_are_well_formed(self, project_service):
        service = LoRATrainingService(projects=project_service)
        presets = service.presets()

        assert len(presets) >= 4
        for preset in presets:
            assert preset["id"] and preset["label"]
            assert len(preset["description"]) > 20, "presets must explain themselves"
            assert preset["values"], "preset must carry actual values"

    def test_preset_values_fit_the_config_schema(self, project_service):
        from app.models.lora_schemas import TrainingConfigRequest

        service = LoRATrainingService(projects=project_service)
        for preset in service.presets():
            # Unknown or out-of-range values would raise here.
            TrainingConfigRequest(**preset["values"])

    def test_character_preset_has_expected_defaults(self, project_service):
        service = LoRATrainingService(projects=project_service)
        character = next(p for p in service.presets() if p["id"] == "character")
        assert character["values"]["lora_rank"] == 16
        assert character["values"]["steps"] >= 1000


# ============================================
# CAPTION BACKGROUND JOB
# ============================================


class TestCaptionJob:
    def test_batch_caption_reports_progress_and_respects_cancel(
        self, project_service, project
    ):
        class StubOllama:
            def generate_with_images(self, **kwargs):
                return "generated caption text"

        project_service.ollama_client = StubOllama()
        for index in range(3):
            project_service.add_image(project.id, f"img{index}.png", png_bytes())

        progress_calls: list[tuple[int, int]] = []

        def stop_after_two() -> bool:
            return len(progress_calls) >= 2

        results = project_service.generate_captions_batch(
            project.id,
            progress_cb=lambda done, total: progress_calls.append((done, total)),
            cancel_check=stop_after_two,
        )

        assert len(progress_calls) == 2
        assert progress_calls[-1] == (2, 3)
        generated = [r for r in results if not r.get("skipped")]
        assert len(generated) == 2

    def test_batch_caption_matches_sync_results(
        self, project_service, project
    ):
        class StubOllama:
            def generate_with_images(self, **kwargs):
                return "same caption"

        project_service.ollama_client = StubOllama()
        project_service.add_image(project.id, "a.png", png_bytes())

        sync_result = project_service.generate_captions(project.id)
        project_service.add_image(project.id, "b.png", png_bytes())
        batch_result = project_service.generate_captions_batch(project.id)

        assert sync_result[0]["caption"] == batch_result[0]["caption"]
        assert batch_result[1]["skipped"] is False


# ============================================
# LORA TEST-IN-COMFYUI INTEGRATION
# ============================================


class TestLoraTestIntegration:
    def _service(self, tmp_settings, client):
        from app.services.comfyui_service import ComfyUIService

        return ComfyUIService(
            client=client,
            workflow_dir=tmp_settings.comfyui_workflow_dir,
            output_dir=tmp_settings.generated_dir,
        )

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
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "t", "images": ["8", 0]}},
            "10": {"class_type": "LoraLoader", "inputs": {
                "lora_name": "", "strength_model": 1.0, "strength_clip": 1.0,
                "model": ["4", 0], "clip": ["4", 1]}},
        }

    class ConnectedStub:
        base_url = "http://stub:8188"

        def ping(self):
            return {"system": {}, "devices": []}

        def list_checkpoints(self):
            return ["sdxl.safetensors"]

        def list_loras(self):
            return ["my_lora.safetensors"]

    def test_prepare_returns_prefilled_request(self, tmp_settings):
        service = self._service(tmp_settings, self.ConnectedStub())
        service.save_workflow("with_lora", self._graph())

        result = service.prepare_lora_test(
            "my_lora.safetensors", trigger_word="p3r5on"
        )
        request = result["generation_request"]
        assert request["workflow_id"] == "with_lora"
        assert request["lora_name"] == "my_lora.safetensors"
        assert request["prompt"].startswith("p3r5on")
        assert result["connected"] is True
        assert result["warnings"] == []

    def test_prepare_warns_when_comfyui_cannot_see_the_lora(self, tmp_settings):
        service = self._service(tmp_settings, self.ConnectedStub())
        service.save_workflow("with_lora", self._graph())

        result = service.prepare_lora_test("other.safetensors")
        assert any("not in ComfyUI" in warning for warning in result["warnings"])

    def test_prepare_fails_without_a_lora_workflow(self, tmp_settings):
        service = self._service(tmp_settings, self.ConnectedStub())
        with pytest.raises(Exception, match="No saved workflow has a LoRA node"):
            service.prepare_lora_test("x.safetensors")

    def test_prepare_rejects_a_workflow_without_a_lora_node(self, tmp_settings):
        service = self._service(tmp_settings, self.ConnectedStub())
        graph = self._graph()
        del graph["10"]
        service.save_workflow("plain", graph)

        with pytest.raises(Exception, match="no LoRA node"):
            service.prepare_lora_test("x.safetensors", workflow_id="plain")


# ============================================
# PERSISTENCE ACROSS RESTARTS
# ============================================


class TestPersistence:
    def test_documents_survive_a_restart(self, tmp_settings):
        from app.services.document_service import DocumentService

        first = DocumentService()
        doc_id = first.store_document("hello world content", source="notes.txt")
        first.store_document("second document", source="other.txt")
        first.delete_document(doc_id)

        # A brand-new instance simulates an application restart.
        second = DocumentService()
        docs = second.get_all_documents()
        assert len(docs) == 1
        assert docs[0]["source"] == "other.txt"
        assert second.get_document(doc_id) is None

    def test_sessions_survive_a_restart(self, tmp_settings):
        from app.services.memory_service import MemoryService

        first = MemoryService()
        first.add_message("session_a", "user", "hi there")
        first.add_message("session_a", "assistant", "hello!")
        first.set_use_documents("session_a", False)

        second = MemoryService()
        history = second.get_history("session_a")
        assert [m["content"] for m in history] == ["hi there", "hello!"]
        assert second.get_use_documents("session_a") is False

    def test_clear_history_removes_it_from_disk(self, tmp_settings):
        from app.services.memory_service import MemoryService

        first = MemoryService()
        first.add_message("s1", "user", "message")
        first.clear_history("s1")

        second = MemoryService()
        assert second.get_history("s1") == []

    def test_corrupt_persistence_file_does_not_crash_startup(self, tmp_settings):
        from app.core.persistence import JsonPersist
        from app.services.memory_service import MemoryService

        persist = JsonPersist("sessions.json")
        persist.path.write_text("{ not valid json !!", encoding="utf-8")

        service = MemoryService()  # must not raise
        assert service.get_session_ids() == []
