"""ComfyUI client, workflow handling, and generation submission.

All tests run against a stub client — none of them require ComfyUI to be
running, and none of them fabricate a successful generation.
"""

from __future__ import annotations

import json

import pytest
import requests

from app.core.exceptions import ComfyUIServiceError, WorkflowError
from app.services.comfyui_service import (
    ComfyUIService,
    infer_mapping,
    inject_inputs,
    validate_graph,
)


def sample_graph(with_lora: bool = False) -> dict:
    model = ["10", 0] if with_lora else ["4", 0]
    clip = ["10", 1] if with_lora else ["4", 1]
    graph = {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl.safetensors"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "positive", "clip": clip}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "negative", "clip": clip}},
        "3": {"class_type": "KSampler", "inputs": {
            "seed": 1, "steps": 20, "cfg": 7.0, "sampler_name": "euler",
            "scheduler": "normal", "denoise": 1.0, "model": model,
            "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "test", "images": ["8", 0]}},
    }
    if with_lora:
        graph["10"] = {"class_type": "LoraLoader", "inputs": {
            "lora_name": "", "strength_model": 1.0, "strength_clip": 1.0,
            "model": ["4", 0], "clip": ["4", 1]}}
    return graph


class StubClient:
    """Minimal stand-in for ComfyUIClient."""

    base_url = "http://stub:8188"

    def __init__(self, *, history_sequence=None, fail=None, image=b"PNGDATA"):
        self.history_sequence = list(history_sequence or [])
        self.fail = fail
        self.image = image
        self.submitted: list[dict] = []
        self.queued = 0

    def ping(self):
        if self.fail:
            raise self.fail
        return {"system": {"comfyui_version": "0.33.1", "python_version": "3.12"},
                "devices": [{"name": "cuda:0", "vram_total": 12, "vram_free": 6}]}

    def list_checkpoints(self):
        return ["sdxl.safetensors"]

    def list_loras(self):
        return ["style.safetensors"]

    def queue_prompt(self, workflow):
        if self.fail:
            raise self.fail
        self.submitted.append(workflow)
        self.queued += 1
        return "prompt-123"

    def history(self, prompt_id):
        return self.history_sequence.pop(0) if self.history_sequence else {}

    def view_image(self, filename, subfolder="", folder_type="output"):
        return self.image


@pytest.fixture
def service(tmp_settings):
    return ComfyUIService(
        client=StubClient(),
        workflow_dir=tmp_settings.comfyui_workflow_dir,
        output_dir=tmp_settings.generated_dir,
    )


# ============================================
# CONNECTION
# ============================================


class TestConnection:
    def test_status_reports_connected_details(self, service):
        status = service.status()
        assert status["connected"] is True
        assert status["version"] == "0.33.1"
        assert status["checkpoints"] == ["sdxl.safetensors"]

    def test_connection_failure_degrades_gracefully(self, tmp_settings):
        service = ComfyUIService(
            client=StubClient(fail=ComfyUIServiceError("unreachable", status_code=503,
                                                       detail="connection refused")),
            workflow_dir=tmp_settings.comfyui_workflow_dir,
            output_dir=tmp_settings.generated_dir,
        )
        status = service.status()
        assert status["connected"] is False
        assert "refused" in status["error"]

    def test_timeout_is_surfaced_as_a_typed_error(self, monkeypatch):
        from app.clients.comfyui_client import ComfyUIClient

        def timeout(*args, **kwargs):
            raise requests.Timeout("timed out")

        monkeypatch.setattr(requests, "request", timeout)
        with pytest.raises(ComfyUIServiceError) as info:
            ComfyUIClient().ping()
        assert info.value.status_code == 504

    def test_connection_error_is_surfaced_as_503(self, monkeypatch):
        from app.clients.comfyui_client import ComfyUIClient

        def refuse(*args, **kwargs):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(requests, "request", refuse)
        with pytest.raises(ComfyUIServiceError) as info:
            ComfyUIClient().ping()
        assert info.value.status_code == 503

    def test_malformed_json_response_is_rejected(self, monkeypatch):
        from app.clients.comfyui_client import ComfyUIClient

        class BadResponse:
            status_code = 200
            content = b"not json"

            def json(self):
                raise ValueError("bad json")

        monkeypatch.setattr(requests, "request", lambda *a, **k: BadResponse())
        with pytest.raises(ComfyUIServiceError, match="malformed"):
            ComfyUIClient().ping()


# ============================================
# WORKFLOW VALIDATION
# ============================================


class TestWorkflowValidation:
    def test_valid_api_graph_passes(self):
        assert validate_graph(sample_graph())

    @pytest.mark.parametrize("bad", [{}, [], "string", 42, None])
    def test_non_object_graphs_are_rejected(self, bad):
        with pytest.raises(WorkflowError):
            validate_graph(bad)

    def test_ui_format_export_gets_a_specific_message(self):
        with pytest.raises(WorkflowError, match="Export \\(API\\)"):
            validate_graph({"nodes": [{"id": 1}], "links": []})

    def test_node_without_class_type_is_rejected(self):
        with pytest.raises(WorkflowError, match="class_type"):
            validate_graph({"1": {"inputs": {}}})

    def test_node_without_inputs_is_rejected(self):
        with pytest.raises(WorkflowError, match="inputs"):
            validate_graph({"1": {"class_type": "KSampler"}})


# ============================================
# MAPPING & INJECTION
# ============================================


class TestMapping:
    def test_prompts_are_resolved_through_sampler_links(self):
        mapping = infer_mapping(sample_graph())
        assert mapping["prompt"] == {"node": "6", "field": "text"}
        assert mapping["negative_prompt"] == {"node": "7", "field": "text"}

    def test_sampler_and_latent_fields_are_detected(self):
        mapping = infer_mapping(sample_graph())
        assert mapping["seed"]["node"] == "3"
        assert mapping["steps"]["node"] == "3"
        assert mapping["cfg"]["node"] == "3"
        assert mapping["width"]["node"] == "5"
        assert mapping["checkpoint"] == {"node": "4", "field": "ckpt_name"}

    def test_lora_node_is_detected_when_present(self):
        mapping = infer_mapping(sample_graph(with_lora=True))
        assert mapping["lora_name"]["node"] == "10"
        assert mapping["lora_strength_model"]["field"] == "strength_model"

    def test_no_lora_key_when_workflow_has_no_lora_node(self):
        assert "lora_name" not in infer_mapping(sample_graph())


class TestInjection:
    def test_values_land_on_the_mapped_nodes(self):
        graph = sample_graph()
        mapping = infer_mapping(graph)
        result = inject_inputs(graph, mapping, {
            "prompt": "a cat", "negative_prompt": "blurry",
            "seed": 999, "steps": 8, "cfg": 3.5, "width": 512, "height": 768,
        })
        assert result["6"]["inputs"]["text"] == "a cat"
        assert result["7"]["inputs"]["text"] == "blurry"
        assert result["3"]["inputs"]["seed"] == 999
        assert result["3"]["inputs"]["steps"] == 8
        assert result["5"]["inputs"]["width"] == 512
        assert result["5"]["inputs"]["height"] == 768

    def test_original_graph_is_not_mutated(self):
        graph = sample_graph()
        before = json.dumps(graph, sort_keys=True)
        inject_inputs(graph, infer_mapping(graph), {"prompt": "changed"})
        assert json.dumps(graph, sort_keys=True) == before

    def test_none_values_leave_the_workflow_default(self):
        graph = sample_graph()
        result = inject_inputs(graph, infer_mapping(graph), {"steps": None})
        assert result["3"]["inputs"]["steps"] == 20

    def test_unmapped_parameters_are_ignored(self):
        graph = sample_graph()
        result = inject_inputs(graph, infer_mapping(graph), {"lora_name": "x.safetensors"})
        assert "10" not in result

    def test_mapping_pointing_at_a_missing_node_is_an_error(self):
        with pytest.raises(WorkflowError, match="not in the workflow"):
            inject_inputs(sample_graph(), {"prompt": {"node": "999", "field": "text"}},
                          {"prompt": "x"})

    def test_node_ids_are_never_assumed(self):
        """A graph with unusual ids works purely through its mapping."""
        graph = {
            "sampler_a": {"class_type": "KSampler", "inputs": {
                "seed": 0, "steps": 10, "cfg": 7.0,
                "positive": ["text_pos", 0], "negative": ["text_neg", 0]}},
            "text_pos": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
            "text_neg": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        }
        mapping = infer_mapping(graph)
        assert mapping["prompt"]["node"] == "text_pos"
        result = inject_inputs(graph, mapping, {"prompt": "works"})
        assert result["text_pos"]["inputs"]["text"] == "works"


# ============================================
# WORKFLOW LIBRARY
# ============================================


class TestWorkflowLibrary:
    def test_save_then_load_roundtrip(self, service):
        info = service.save_workflow("My Test Flow", sample_graph(), description="demo")
        assert info.node_count == 7
        graph, mapping = service.load_workflow(info.id)
        assert graph["4"]["class_type"] == "CheckpointLoaderSimple"
        assert mapping["prompt"]["node"] == "6"

    def test_saved_id_is_filesystem_safe(self, service):
        info = service.save_workflow("../../evil name!.json", sample_graph())
        assert "/" not in info.id and ".." not in info.id

    def test_missing_workflow_is_404(self, service):
        with pytest.raises(WorkflowError) as exc:
            service.load_workflow("does_not_exist")
        assert exc.value.status_code == 404

    def test_invalid_import_is_rejected(self, service):
        with pytest.raises(WorkflowError):
            service.save_workflow("bad", {"nodes": [], "links": []})

    def test_unknown_mapped_input_is_rejected(self, service):
        with pytest.raises(WorkflowError, match="Unknown mapped inputs"):
            service.save_workflow("x", sample_graph(),
                                  inputs={"not_a_real_input": {"node": "6", "field": "text"}})

    def test_delete_removes_graph_and_mapping(self, service):
        info = service.save_workflow("temp", sample_graph())
        assert service.delete_workflow(info.id) is True
        assert service.delete_workflow(info.id) is False
        assert service.list_workflows() == []

    def test_map_files_are_not_listed_as_workflows(self, service):
        service.save_workflow("one", sample_graph())
        assert [w.id for w in service.list_workflows()] == ["one"]


# ============================================
# GENERATION
# ============================================


class TestGeneration:
    def test_seed_is_randomised_when_omitted(self, service):
        service.save_workflow("flow", sample_graph())
        _, seed_a = service.build_graph("flow", {"prompt": "x"})
        _, seed_b = service.build_graph("flow", {"prompt": "x"})
        assert 0 <= seed_a < 2**32
        assert seed_a != seed_b, "two omitted seeds should differ"

    def test_negative_seed_is_treated_as_random(self, service):
        service.save_workflow("flow", sample_graph())
        _, seed = service.build_graph("flow", {"seed": -1})
        assert seed >= 0

    def test_explicit_seed_is_respected(self, service):
        service.save_workflow("flow", sample_graph())
        graph, seed = service.build_graph("flow", {"seed": 4242})
        assert seed == 4242
        assert graph["3"]["inputs"]["seed"] == 4242

    def test_unselected_lora_is_neutralised_by_zero_strength(self, service):
        service.save_workflow("lora_flow", sample_graph(with_lora=True))
        graph, _ = service.build_graph("lora_flow", {"prompt": "x"})
        assert graph["10"]["inputs"]["strength_model"] == 0.0
        assert graph["10"]["inputs"]["strength_clip"] == 0.0

    def test_selected_lora_is_injected(self, service):
        service.save_workflow("lora_flow", sample_graph(with_lora=True))
        graph, _ = service.build_graph("lora_flow", {
            "lora_name": "mine.safetensors",
            "lora_strength_model": 0.8,
            "lora_strength_clip": 0.7,
        })
        assert graph["10"]["inputs"]["lora_name"] == "mine.safetensors"
        assert graph["10"]["inputs"]["strength_model"] == 0.8
        assert graph["10"]["inputs"]["strength_clip"] == 0.7

    def test_successful_run_saves_outputs(self, tmp_settings):
        client = StubClient(history_sequence=[
            {},
            {"status": {"status_str": "success", "completed": True},
             "outputs": {"9": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}}},
        ])
        service = ComfyUIService(client=client,
                                 workflow_dir=tmp_settings.comfyui_workflow_dir,
                                 output_dir=tmp_settings.generated_dir)
        service.save_workflow("flow", sample_graph())
        graph, _ = service.build_graph("flow", {"prompt": "x"})

        from app.core.jobs import Job
        job = service.jobs.create("comfyui_generation")
        service._run_generation(job, graph)

        saved = service.jobs.get(job.id).outputs
        assert len(saved) == 1
        assert service.read_generated(saved[0]) == b"PNGDATA"

    def test_execution_error_is_raised(self, tmp_settings):
        client = StubClient(history_sequence=[
            {"status": {"status_str": "error", "completed": False,
                        "messages": [["execution_error", "node 3 exploded"]]}},
        ])
        service = ComfyUIService(client=client,
                                 workflow_dir=tmp_settings.comfyui_workflow_dir,
                                 output_dir=tmp_settings.generated_dir)
        service.save_workflow("flow", sample_graph())
        graph, _ = service.build_graph("flow", {})
        job = service.jobs.create("comfyui_generation")

        with pytest.raises(ComfyUIServiceError, match="execution error"):
            service._run_generation(job, graph)

    def test_generation_timeout(self, tmp_settings, set_setting):
        set_setting("comfyui_generation_timeout", 0.05)
        set_setting("comfyui_poll_interval", 0.01)

        service = ComfyUIService(client=StubClient(history_sequence=[]),
                                 workflow_dir=tmp_settings.comfyui_workflow_dir,
                                 output_dir=tmp_settings.generated_dir)
        service.save_workflow("flow", sample_graph())
        graph, _ = service.build_graph("flow", {})
        job = service.jobs.create("comfyui_generation")

        with pytest.raises(ComfyUIServiceError) as exc:
            service._run_generation(job, graph)
        assert exc.value.status_code == 504

    def test_generated_image_read_is_traversal_safe(self, service):
        from app.core.paths import UnsafePathError
        with pytest.raises((UnsafePathError, WorkflowError)):
            service.read_generated("../../../../windows/win.ini")
