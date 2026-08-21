"""LoRA projects, datasets, captions, config generation, and training control.

Training tests drive a *fake* trainer script rather than the real AI Toolkit:
they verify the process management (start, stream, progress parse, stop, exit
code handling), not that a GPU can train a model.
"""

from __future__ import annotations

import sys
import textwrap
import time
from pathlib import Path

import pytest
import yaml

from app.core.exceptions import AIToolkitError, LoRAProjectError, OllamaServiceError
from app.services.lora_dataset_service import LoRAProjectService, _clean_caption
from app.services.lora_training_service import (
    AIToolkitProcess,
    LoRATrainingService,
    build_training_config,
)
from tests.conftest import png_bytes


@pytest.fixture
def project(project_service):
    return project_service.create_project(
        "Test Character", description="demo", arch="sdxl", trigger_word="p3r5on"
    )


# ============================================
# PROJECTS
# ============================================


class TestProjects:
    def test_creation_builds_the_expected_folders(self, project_service, project):
        root = Path(project_service.data_dir) / project.id
        assert (root / "dataset").is_dir()
        assert (root / "config").is_dir()
        assert (root / "output").is_dir()
        assert (root / "project.json").is_file()

    def test_id_is_slugified_and_unique(self, project_service):
        a = project_service.create_project("My Cool LoRA!")
        b = project_service.create_project("My Cool LoRA!")
        assert a.id != b.id
        assert a.id.startswith("my_cool_lora")
        assert "/" not in a.id and "!" not in a.id

    def test_blank_name_is_rejected(self, project_service):
        with pytest.raises(LoRAProjectError):
            project_service.create_project("   ")

    def test_metadata_roundtrips(self, project_service, project):
        loaded = project_service.get_project(project.id)
        assert loaded.name == "Test Character"
        assert loaded.trigger_word == "p3r5on"
        assert loaded.arch == "sdxl"

    def test_missing_project_is_404(self, project_service):
        with pytest.raises(LoRAProjectError) as exc:
            project_service.get_project("nope")
        assert exc.value.status_code == 404

    def test_traversal_project_id_is_refused(self, project_service):
        from app.core.paths import UnsafePathError
        with pytest.raises((UnsafePathError, LoRAProjectError)):
            project_service.get_project("../../etc")

    def test_update_and_list(self, project_service, project):
        project_service.update_project(project.id, trigger_word="newtrigger")
        assert project_service.get_project(project.id).trigger_word == "newtrigger"
        assert len(project_service.list_projects()) == 1

    def test_delete_removes_the_directory(self, project_service, project):
        assert project_service.delete_project(project.id) is True
        assert project_service.delete_project(project.id) is False
        assert project_service.list_projects() == []


# ============================================
# DATASET
# ============================================


class TestDataset:
    def test_upload_creates_image_and_caption_file(self, project_service, project):
        image = project_service.add_image(project.id, "photo.png", png_bytes())
        dataset = Path(project_service.dataset_dir(project.id))
        assert (dataset / "photo.png").is_file()
        assert (dataset / "photo.txt").is_file(), "AI Toolkit needs a matching .txt"
        assert image.id == "photo"

    def test_duplicate_names_do_not_overwrite(self, project_service, project):
        first = project_service.add_image(project.id, "a.png", png_bytes())
        second = project_service.add_image(project.id, "a.png", png_bytes(80))
        assert first.filename != second.filename
        assert len(project_service.list_images(project.id)) == 2

    @pytest.mark.parametrize("bad", ["virus.exe", "notes.txt", "doc.pdf", "run.py"])
    def test_disallowed_extensions_are_refused(self, project_service, project, bad):
        with pytest.raises(ValueError, match="Unsupported image type"):
            project_service.add_image(project.id, bad, png_bytes())

    def test_traversal_filename_is_confined_to_the_dataset(self, project_service, project):
        image = project_service.add_image(project.id, "../../../evil.png", png_bytes())
        stored = Path(project_service.dataset_dir(project.id)) / image.filename
        assert stored.is_file()
        assert ".." not in image.filename

    def test_oversized_upload_is_refused(self, project_service, project, set_setting):
        set_setting("max_image_upload_mb", 0.0001)
        with pytest.raises(ValueError, match="limit"):
            project_service.add_image(project.id, "big.png", png_bytes(5000))

    def test_delete_removes_image_and_caption(self, project_service, project):
        project_service.add_image(project.id, "x.png", png_bytes())
        assert project_service.delete_image(project.id, "x") is True
        dataset = Path(project_service.dataset_dir(project.id))
        assert not (dataset / "x.png").exists()
        assert not (dataset / "x.txt").exists()

    def test_read_image_returns_bytes_and_type(self, project_service, project):
        content = png_bytes()
        project_service.add_image(project.id, "y.png", content)
        data, media_type = project_service.read_image(project.id, "y")
        assert data == content
        assert media_type == "image/png"


# ============================================
# CAPTIONS
# ============================================


class TestCaptions:
    def test_setting_a_caption_writes_the_txt_file(self, project_service, project):
        project_service.add_image(project.id, "a.png", png_bytes())
        project_service.set_caption(project.id, "a", "p3r5on, smiling, studio light")

        caption_file = Path(project_service.dataset_dir(project.id)) / "a.txt"
        assert caption_file.read_text(encoding="utf-8") == "p3r5on, smiling, studio light"

    def test_manual_edits_are_flagged(self, project_service, project):
        project_service.add_image(project.id, "a.png", png_bytes())
        project_service.set_caption(project.id, "a", "mine")
        image = project_service.list_images(project.id)[0]
        assert image.caption_edited is True
        assert image.caption == "mine"

    def test_editing_a_missing_image_is_404(self, project_service, project):
        with pytest.raises(LoRAProjectError) as exc:
            project_service.set_caption(project.id, "ghost", "x")
        assert exc.value.status_code == 404

    def test_ai_captions_use_ollama_and_prepend_trigger(self, project_service, project):
        class StubOllama:
            def __init__(self):
                self.calls = 0

            def generate_with_images(self, *, model, prompt, images, timeout=None):
                self.calls += 1
                return "a woman in a red coat, city street, overcast"

        project_service.ollama_client = StubOllama()
        project_service.add_image(project.id, "a.png", png_bytes())

        results = project_service.generate_captions(project.id)
        assert results[0]["skipped"] is False
        assert results[0]["caption"].startswith("p3r5on,")
        assert project_service.ollama_client.calls == 1

        saved = (Path(project_service.dataset_dir(project.id)) / "a.txt").read_text()
        assert "red coat" in saved

    def test_ai_captions_never_clobber_manual_edits(self, project_service, project):
        class StubOllama:
            def generate_with_images(self, **kwargs):
                return "machine generated"

        project_service.ollama_client = StubOllama()
        project_service.add_image(project.id, "a.png", png_bytes())
        project_service.set_caption(project.id, "a", "hand written caption")

        results = project_service.generate_captions(project.id)
        assert results[0]["skipped"] is True
        assert results[0]["reason"] == "manually edited"

        saved = (Path(project_service.dataset_dir(project.id)) / "a.txt").read_text()
        assert saved == "hand written caption"

    def test_overwrite_flag_does_replace_manual_edits(self, project_service, project):
        class StubOllama:
            def generate_with_images(self, **kwargs):
                return "machine generated"

        project_service.ollama_client = StubOllama()
        project_service.add_image(project.id, "a.png", png_bytes())
        project_service.set_caption(project.id, "a", "hand written")

        results = project_service.generate_captions(project.id, overwrite=True)
        assert results[0]["skipped"] is False
        assert "machine generated" in results[0]["caption"]

    def test_offline_ollama_degrades_without_losing_captions(self, project_service, project):
        class DeadOllama:
            def generate_with_images(self, **kwargs):
                raise OllamaServiceError("Ollama is unreachable", status_code=503,
                                         detail="connection refused")

        project_service.ollama_client = DeadOllama()
        project_service.add_image(project.id, "a.png", png_bytes())

        results = project_service.generate_captions(project.id)
        assert results[0]["skipped"] is True
        assert "refused" in results[0]["reason"]

    def test_captioning_an_empty_dataset_is_an_error(self, project_service, project):
        with pytest.raises(LoRAProjectError, match="no images"):
            project_service.generate_captions(project.id)

    @pytest.mark.parametrize("raw,expected_start", [
        ("Here is a caption: a dog", "a caption: a dog"),
        ("This is a red car, sunny", "a red car, sunny"),
        ('"quoted caption"', "quoted caption"),
        ("  spaced   out   text ", "spaced out text"),
    ])
    def test_caption_boilerplate_is_stripped(self, raw, expected_start):
        assert _clean_caption(raw) == expected_start


# ============================================
# TRAINING CONFIG
# ============================================


class TestTrainingConfig:
    def test_matches_ai_toolkit_v0_12_schema(self):
        config = build_training_config(
            project_name="my_lora", dataset_path="/data/ds", output_path="/data/out",
            arch="sdxl", trigger_word="p3r5on", steps=1500, lora_rank=32,
        )
        assert config["job"] == "extension"
        process = config["config"]["process"][0]
        assert process["type"] == "sd_trainer"
        assert process["training_folder"] == "/data/out"
        assert process["trigger_word"] == "p3r5on"
        assert process["network"] == {"type": "lora", "linear": 32, "linear_alpha": 32}
        assert process["datasets"][0]["folder_path"] == "/data/ds"
        assert process["datasets"][0]["caption_ext"] == "txt"
        assert process["train"]["steps"] == 1500
        assert process["model"]["arch"] == "sdxl"

    def test_explicit_alpha_is_kept_separate_from_rank(self):
        config = build_training_config(
            project_name="x", dataset_path="d", output_path="o",
            lora_rank=64, lora_alpha=16,
        )
        network = config["config"]["process"][0]["network"]
        assert network["linear"] == 64 and network["linear_alpha"] == 16

    @pytest.mark.parametrize("arch,expected", [
        ("sdxl", "stabilityai/stable-diffusion-xl-base-1.0"),
        ("flux", "black-forest-labs/FLUX.1-dev"),
        ("qwen_image", "Qwen/Qwen-Image"),
    ])
    def test_arch_defaults(self, arch, expected):
        config = build_training_config(project_name="x", dataset_path="d",
                                       output_path="o", arch=arch)
        assert config["config"]["process"][0]["model"]["name_or_path"] == expected

    def test_unknown_arch_falls_back_without_crashing(self):
        config = build_training_config(project_name="x", dataset_path="d",
                                       output_path="o", arch="not_a_real_arch")
        assert config["config"]["process"][0]["model"]["arch"] == "not_a_real_arch"
        assert config["config"]["process"][0]["model"]["name_or_path"]

    def test_sampling_is_disabled_by_default(self):
        config = build_training_config(project_name="x", dataset_path="d", output_path="o")
        process = config["config"]["process"][0]
        assert "sample" not in process
        assert process["train"]["disable_sampling"] is True

    def test_sampling_can_be_enabled(self):
        config = build_training_config(project_name="x", dataset_path="d",
                                       output_path="o", sample_every=100)
        assert config["config"]["process"][0]["sample"]["sample_every"] == 100

    def test_config_serialises_to_yaml(self):
        config = build_training_config(project_name="x", dataset_path="d", output_path="o")
        reloaded = yaml.safe_load(yaml.safe_dump(config, sort_keys=False))
        assert reloaded == config

    def test_write_config_requires_images(self, project_service, project):
        service = LoRATrainingService(projects=project_service)
        with pytest.raises(LoRAProjectError, match="at least one training image"):
            service.write_config(project.id, {})

    def test_write_config_requires_captions(self, project_service, project):
        project_service.add_image(project.id, "a.png", png_bytes())
        service = LoRATrainingService(projects=project_service)
        with pytest.raises(LoRAProjectError, match="No captions"):
            service.write_config(project.id, {})

    def test_write_config_produces_a_readable_file(self, project_service, project):
        project_service.add_image(project.id, "a.png", png_bytes())
        project_service.set_caption(project.id, "a", "p3r5on, portrait")

        service = LoRATrainingService(projects=project_service)
        path = service.write_config(project.id, {"steps": 800, "lora_rank": 8})

        assert path.name == "training.yml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["config"]["process"][0]["train"]["steps"] == 800
        assert config["config"]["process"][0]["network"]["linear"] == 8


# ============================================
# TOOLKIT AVAILABILITY
# ============================================


class TestToolkitAvailability:
    def test_unconfigured_toolkit_reports_clearly(self, project_service):
        service = LoRATrainingService(projects=project_service)
        status = service.status()
        assert status["configured"] is False
        assert "AI_TOOLKIT_PATH" in status["error"]

    def test_training_without_toolkit_raises_503(self, project_service, project):
        service = LoRATrainingService(projects=project_service)
        with pytest.raises(AIToolkitError) as exc:
            service.start_training(project.id, {})
        assert exc.value.status_code == 503

    def test_configured_requires_files_to_actually_exist(self, set_setting):
        from app.core.config import settings
        set_setting("ai_toolkit_path", "W:/does/not/exist")
        set_setting("ai_toolkit_python", "W:/does/not/exist/python.exe")
        assert settings.ai_toolkit_configured is False


# ============================================
# PROCESS MANAGEMENT (fake trainer, real subprocess)
# ============================================


def write_fake_trainer(tmp_path: Path, body: str) -> Path:
    """Create a stand-in for AI Toolkit's run.py that we can control."""
    script = tmp_path / "run.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    return script


class TestProcessManagement:
    def test_progress_is_parsed_from_output(self, tmp_path, set_setting):
        write_fake_trainer(tmp_path, """
            import sys, time
            for step in range(1, 6):
                print(f"  {step * 100}/500 [00:0{step}<00:10, loss=0.1{step}]")
                sys.stdout.flush()
                time.sleep(0.05)
            print("done")
        """)
        set_setting("ai_toolkit_path", str(tmp_path))
        set_setting("ai_toolkit_python", sys.executable)

        trainer = AIToolkitProcess(tmp_path / "cfg.yml", tmp_path / "train.log")
        trainer.start()
        trainer.stream()
        trainer.process.wait(timeout=15)

        snapshot = trainer.snapshot()
        assert snapshot["total_steps"] == 500
        assert snapshot["current_step"] == 500
        assert snapshot["loss"] == pytest.approx(0.15, abs=0.01)
        assert trainer.progress() == 100.0
        assert "500/500" in (tmp_path / "train.log").read_text()

    def test_progress_is_none_when_output_is_unparseable(self, tmp_path, set_setting):
        write_fake_trainer(tmp_path, """
            print("Loading model weights...")
            print("Some opaque message with no step counter")
        """)
        set_setting("ai_toolkit_path", str(tmp_path))
        set_setting("ai_toolkit_python", sys.executable)

        trainer = AIToolkitProcess(tmp_path / "cfg.yml", tmp_path / "train.log")
        trainer.start()
        trainer.stream()
        trainer.process.wait(timeout=15)

        # Honest fallback rather than an invented percentage.
        assert trainer.progress() is None
        assert trainer.snapshot()["total_steps"] == 0

    def test_failing_trainer_reports_a_nonzero_exit_code(self, tmp_path, set_setting):
        write_fake_trainer(tmp_path, """
            import sys
            print("CUDA out of memory")
            sys.exit(1)
        """)
        set_setting("ai_toolkit_path", str(tmp_path))
        set_setting("ai_toolkit_python", sys.executable)

        trainer = AIToolkitProcess(tmp_path / "cfg.yml", tmp_path / "train.log")
        trainer.start()
        trainer.stream()
        trainer.process.wait(timeout=15)

        assert trainer.process.returncode == 1
        assert "out of memory" in trainer.snapshot()["last_line"]

    def test_stop_terminates_only_our_process(self, tmp_path, set_setting):
        write_fake_trainer(tmp_path, """
            import time
            for i in range(600):
                print(f"{i}/600 [running]", flush=True)
                time.sleep(0.5)
        """)
        set_setting("ai_toolkit_path", str(tmp_path))
        set_setting("ai_toolkit_python", sys.executable)

        trainer = AIToolkitProcess(tmp_path / "cfg.yml", tmp_path / "train.log")
        trainer.start()
        pid = trainer.process.pid
        time.sleep(1.0)
        assert trainer.is_running()

        assert trainer.stop(timeout=15) is True
        assert not trainer.is_running()
        assert trainer.process.pid == pid, "stop must target the PID we started"

        # Stopping again is a no-op, not an error.
        assert trainer.stop() is False

    def test_start_without_configuration_raises(self, tmp_path, tmp_settings):
        # tmp_settings blanks AI_TOOLKIT_* so this holds even when the developer
        # running the suite has a real toolkit configured in their .env.
        trainer = AIToolkitProcess(tmp_path / "cfg.yml", tmp_path / "train.log")
        with pytest.raises(AIToolkitError, match="not configured"):
            trainer.start()


# ============================================
# LORA LIBRARY DISCOVERY
# ============================================


def make_safetensors(path: Path, payload: dict | None = None) -> Path:
    import json

    header = json.dumps(payload or {"__metadata__": {"format": "pt"}}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\x00" * 64)
    return path


class TestLibrary:
    def test_trained_lora_is_discovered(self, project_service, project):
        output = Path(project_service.output_dir(project.id))
        make_safetensors(output / "my_character.safetensors")

        library = project_service.list_library()
        assert len(library) == 1
        assert library[0]["name"] == "my_character"
        assert library[0]["project_id"] == project.id
        assert library[0]["trigger_word"] == "p3r5on"

    def test_renamed_non_safetensors_is_excluded(self, project_service, project):
        output = Path(project_service.output_dir(project.id))
        (output / "fake.safetensors").write_bytes(b"just some text, definitely not tensors")
        assert project_service.list_library() == []

    def test_comfyui_folder_is_also_scanned(self, project_service, tmp_path, set_setting):
        comfy = tmp_path / "comfy_loras"
        make_safetensors(comfy / "downloaded_style.safetensors")
        set_setting("comfyui_lora_dir", str(comfy))

        library = project_service.list_library()
        assert [entry["name"] for entry in library] == ["downloaded_style"]
        assert library[0]["source"] == "comfyui"
        assert library[0]["training_status"] == "external"

    def test_find_trained_lora_picks_the_newest(self, project_service, project):
        output = Path(project_service.output_dir(project.id))
        old = make_safetensors(output / "step500.safetensors")
        import os
        os.utime(old, (1_600_000_000, 1_600_000_000))
        make_safetensors(output / "step1000.safetensors")

        service = LoRATrainingService(projects=project_service)
        assert service.find_trained_lora(project.id).name == "step1000.safetensors"

    def test_find_trained_lora_returns_none_when_absent(self, project_service, project):
        service = LoRATrainingService(projects=project_service)
        assert service.find_trained_lora(project.id) is None
