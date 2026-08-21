"""Ostris AI Toolkit orchestration.

AI Toolkit is treated strictly as an external local process: this module writes
a config it understands and runs its ``run.py``. None of its source is vendored.

Config shape verified against AI Toolkit **v0.12.26**
(``config/examples/train_lora_*.yaml``). If you upgrade the toolkit and the
schema moves, this is the one file that needs updating.
"""

from __future__ import annotations

import atexit
import logging
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from app.core.config import settings
from app.core.exceptions import AIToolkitError, LoRAProjectError
from app.core.jobs import Job, JobStatus, JobStore
from app.core.paths import is_safetensors, safe_join
from app.services.lora_dataset_service import LoRAProjectService

logger = logging.getLogger(__name__)

# AI Toolkit prints tqdm-style progress; both forms appear across versions.
_STEP_PATTERNS = (
    re.compile(r"(\d+)\s*/\s*(\d+)\s*\[", re.ASCII),   # "  350/2000 [01:22<..."
    re.compile(r"step[:\s]+(\d+)\s*/\s*(\d+)", re.I),
)
_LOSS_PATTERN = re.compile(r"loss[=:\s]+([0-9]*\.?[0-9]+)", re.I)

# Sensible per-architecture defaults. Only what is actually needed today —
# new architectures get an entry here rather than branches sprinkled elsewhere.
ARCH_DEFAULTS: dict[str, dict[str, Any]] = {
    "sdxl": {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "resolution": [768, 1024],
        "quantize": False,
        "noise_scheduler": "ddpm",
        "dtype": "bf16",
        "sample_sampler": "ddpm",
    },
    "flux": {
        "base_model": "black-forest-labs/FLUX.1-dev",
        "resolution": [512, 768, 1024],
        "quantize": True,
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "sample_sampler": "flowmatch",
    },
    "qwen_image": {
        "base_model": "Qwen/Qwen-Image",
        "resolution": [512, 768, 1024],
        "quantize": True,
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "sample_sampler": "flowmatch",
    },
    "krea2": {
        "base_model": "krea/krea-2",
        "resolution": [512, 768, 1024],
        "quantize": True,
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "sample_sampler": "flowmatch",
    },
}
DEFAULT_ARCH = "sdxl"

# Reusable starting points. Every value here is something AI Toolkit accepts;
# users can override any of it per-run from the training UI.
TRAINING_PRESETS: dict[str, dict[str, Any]] = {
    "character": {
        "label": "Character LoRA",
        "description": (
            "A specific person, pet or character. Moderate rank for detail, "
            "medium step count, low learning rate to avoid overfitting a "
            "small dataset."
        ),
        "values": {
            "steps": 2000,
            "learning_rate": 1e-4,
            "batch_size": 1,
            "lora_rank": 16,
            "save_every": 250,
            "resolution": [512, 768, 1024],
        },
    },
    "style": {
        "label": "Style LoRA",
        "description": (
            "An art style or aesthetic applied across many subjects. Higher "
            "rank captures stylistic texture; needs more varied images."
        ),
        "values": {
            "steps": 3000,
            "learning_rate": 1e-4,
            "batch_size": 1,
            "lora_rank": 32,
            "save_every": 250,
            "resolution": [768, 1024],
        },
    },
    "product": {
        "label": "Product LoRA",
        "description": (
            "A specific object (product, prop, vehicle) shot consistently. "
            "Low rank keeps it focused and small; short runs usually suffice."
        ),
        "values": {
            "steps": 1500,
            "learning_rate": 8e-5,
            "batch_size": 1,
            "lora_rank": 8,
            "save_every": 250,
            "resolution": [768, 1024],
        },
    },
    "concept": {
        "label": "Concept LoRA",
        "description": (
            "An abstract idea or visual motif rather than one subject. "
            "Balanced settings with more steps for generalisation."
        ),
        "values": {
            "steps": 2500,
            "learning_rate": 1e-4,
            "batch_size": 2,
            "lora_rank": 16,
            "save_every": 500,
            "resolution": [512, 768],
        },
    },
    "general": {
        "label": "General Purpose",
        "description": (
            "AI Toolkit's own example defaults. A safe starting point when "
            "you are not sure what your dataset needs yet."
        ),
        "values": {
            "steps": 2000,
            "learning_rate": 1e-4,
            "batch_size": 1,
            "lora_rank": 16,
            "save_every": 250,
            "resolution": [512, 1024],
        },
    },
}


def build_training_config(
    *,
    project_name: str,
    dataset_path: str,
    output_path: str,
    arch: str = DEFAULT_ARCH,
    base_model: str | None = None,
    trigger_word: str = "",
    steps: int = 2000,
    learning_rate: float = 1e-4,
    batch_size: int = 1,
    resolution: list[int] | None = None,
    lora_rank: int = 16,
    lora_alpha: int | None = None,
    save_every: int = 250,
    max_saves_to_keep: int = 4,
    sample_every: int = 0,
    sample_prompts: list[str] | None = None,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Build an AI Toolkit v0.12.26 training config as a plain dict."""
    defaults = ARCH_DEFAULTS.get(arch, ARCH_DEFAULTS[DEFAULT_ARCH])

    process: dict[str, Any] = {
        "type": "sd_trainer",
        "training_folder": output_path,
        "device": device,
        "network": {
            "type": "lora",
            "linear": lora_rank,
            "linear_alpha": lora_alpha if lora_alpha is not None else lora_rank,
        },
        "save": {
            "dtype": "float16",
            "save_every": save_every,
            "max_step_saves_to_keep": max_saves_to_keep,
            "push_to_hub": False,
        },
        "datasets": [
            {
                "folder_path": dataset_path,
                "caption_ext": "txt",
                "caption_dropout_rate": 0.05,
                "shuffle_tokens": False,
                "cache_latents_to_disk": True,
                "resolution": resolution or defaults["resolution"],
            }
        ],
        "train": {
            "batch_size": batch_size,
            "steps": steps,
            "gradient_accumulation_steps": 1,
            "train_unet": True,
            "train_text_encoder": False,
            "gradient_checkpointing": True,
            "noise_scheduler": defaults["noise_scheduler"],
            "optimizer": "adamw8bit",
            "lr": learning_rate,
            "dtype": defaults["dtype"],
        },
        "model": {
            "name_or_path": base_model or defaults["base_model"],
            "arch": arch,
            "quantize": defaults["quantize"],
        },
    }

    if trigger_word:
        process["trigger_word"] = trigger_word

    if sample_every and sample_every > 0:
        process["sample"] = {
            "sampler": defaults["sample_sampler"],
            "sample_every": sample_every,
            "width": 1024,
            "height": 1024,
            "prompts": sample_prompts or [f"{trigger_word or 'a person'}, portrait photo"],
            "neg": "",
            "seed": 42,
            "walk_seed": True,
            "guidance_scale": 4,
            "sample_steps": 20,
        }
    else:
        # Sampling mid-training costs VRAM most local GPUs would rather spend
        # on the actual training run.
        process["train"]["disable_sampling"] = True

    return {
        "job": "extension",
        "config": {"name": project_name, "process": [process]},
        "meta": {"name": "[name]", "version": "1.0"},
    }


class AIToolkitProcess:
    """Owns exactly one AI Toolkit subprocess and its log tail.

    Deliberately narrow: it starts, reads, and stops *its own* PID. It never
    searches for python processes to kill.
    """

    def __init__(self, config_path: Path, log_path: Path) -> None:
        self.config_path = config_path
        self.log_path = log_path
        self.process: subprocess.Popen | None = None
        self.current_step = 0
        self.total_steps = 0
        self.last_loss: float | None = None
        self.last_line = ""
        self._lock = threading.Lock()

    def start(self) -> None:
        if not settings.ai_toolkit_configured:
            raise AIToolkitError(
                "Ostris AI Toolkit is not configured. Set AI_TOOLKIT_PATH and "
                "AI_TOOLKIT_PYTHON in your .env file."
            )

        command = [
            settings.ai_toolkit_python,
            "run.py",
            str(self.config_path),
        ]
        logger.info("[LORA TRAINING] Launching: %s", " ".join(command))

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        creation_flags = 0
        if sys.platform == "win32":
            # Own process group, so stopping this run cannot signal our own
            # server process or anything else on the machine.
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

        try:
            self.process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                command,
                cwd=settings.ai_toolkit_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
                creationflags=creation_flags,
            )
        except OSError as exc:
            raise AIToolkitError(
                "Could not start AI Toolkit", detail=str(exc)
            ) from exc

        _REGISTRY.add(self)

    def stream(self) -> None:
        """Read stdout to completion, tee-ing to the log and parsing progress."""
        if self.process is None or self.process.stdout is None:
            return
        with open(self.log_path, "a", encoding="utf-8") as log_file:
            for line in self.process.stdout:
                log_file.write(line)
                log_file.flush()
                self._parse(line)

    def _parse(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return
        with self._lock:
            self.last_line = stripped[:500]
            for pattern in _STEP_PATTERNS:
                match = pattern.search(stripped)
                if match:
                    self.current_step = int(match.group(1))
                    self.total_steps = int(match.group(2))
                    break
            loss = _LOSS_PATTERN.search(stripped)
            if loss:
                try:
                    self.last_loss = float(loss.group(1))
                except ValueError:
                    pass

    def progress(self) -> float | None:
        """Percentage complete, or None when the output gave us nothing to go on."""
        with self._lock:
            if self.total_steps > 0:
                return round(min(self.current_step / self.total_steps * 100, 100.0), 1)
        return None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "loss": self.last_loss,
                "last_line": self.last_line,
            }

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self, timeout: float = 15.0) -> bool:
        """Terminate this run's process tree, and only this one."""
        if self.process is None or self.process.poll() is not None:
            _REGISTRY.discard(self)
            return False

        pid = self.process.pid
        logger.info("[LORA TRAINING] Stopping PID %s", pid)

        if sys.platform == "win32":
            # AI Toolkit spawns dataloader children; taskkill /T on our exact
            # PID takes the tree without touching unrelated processes.
            subprocess.run(  # noqa: S603 - fixed argv, pid is an int we own
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                check=False,
            )
        else:
            self.process.terminate()

        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
        _REGISTRY.discard(self)
        return True


# Every live trainer, so a server shutdown never orphans a GPU process.
_REGISTRY: set[AIToolkitProcess] = set()


@atexit.register
def _stop_all_trainers() -> None:
    for trainer in list(_REGISTRY):
        try:
            trainer.stop(timeout=5)
        except Exception:  # noqa: BLE001 - best effort during interpreter exit
            pass


class LoRATrainingService:
    """Turns a LoRA project into a running AI Toolkit job and tracks it."""

    def __init__(
        self,
        projects: LoRAProjectService | None = None,
        jobs: JobStore | None = None,
    ) -> None:
        self.projects = projects or LoRAProjectService()
        self.jobs = jobs or JobStore(persist_path=settings.jobs_file)
        self._active: dict[str, AIToolkitProcess] = {}

    # ---------- availability ----------

    def status(self) -> dict[str, Any]:
        configured = settings.ai_toolkit_configured
        return {
            "configured": configured,
            "path": settings.ai_toolkit_path or None,
            "python": settings.ai_toolkit_python or None,
            "supported_archs": sorted(ARCH_DEFAULTS),
            "error": None
            if configured
            else "Set AI_TOOLKIT_PATH and AI_TOOLKIT_PYTHON to enable LoRA training.",
        }

    def presets(self) -> list[dict[str, Any]]:
        """Reusable training starting points, with human-readable explanations."""
        return [
            {
                "id": preset_id,
                "label": preset["label"],
                "description": preset["description"],
                "values": dict(preset["values"]),
            }
            for preset_id, preset in TRAINING_PRESETS.items()
        ]

    # ---------- config ----------

    def write_config(self, project_id: str, options: dict[str, Any]) -> Path:
        """Generate training.yml for a project and persist the chosen options."""
        project = self.projects.get_project(project_id)
        dataset = self.projects.dataset_dir(project_id)
        output = self.projects.output_dir(project_id)

        images = self.projects.list_images(project_id)
        if not images:
            raise LoRAProjectError("Add at least one training image before training")
        missing = [image.id for image in images if not image.caption]
        if len(missing) == len(images):
            raise LoRAProjectError(
                "No captions found. Write captions, or generate them with Ollama, "
                "before starting training."
            )

        config = build_training_config(
            project_name=project.id,
            dataset_path=str(dataset),
            output_path=str(output),
            arch=options.get("arch") or project.arch,
            base_model=options.get("base_model") or project.base_model,
            trigger_word=options.get("trigger_word", project.trigger_word),
            steps=int(options.get("steps", 2000)),
            learning_rate=float(options.get("learning_rate", 1e-4)),
            batch_size=int(options.get("batch_size", 1)),
            resolution=options.get("resolution"),
            lora_rank=int(options.get("lora_rank", 16)),
            lora_alpha=options.get("lora_alpha"),
            save_every=int(options.get("save_every", 250)),
            sample_every=int(options.get("sample_every", 0)),
        )

        config_path = safe_join(self.projects.config_dir(project_id), "training.yml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

        self.projects.update_project(
            project_id,
            training_config=config["config"]["process"][0],
            arch=config["config"]["process"][0]["model"]["arch"],
            base_model=config["config"]["process"][0]["model"]["name_or_path"],
        )
        logger.info("[LORA TRAINING] Wrote config %s", config_path)
        return config_path

    # ---------- run ----------

    def start_training(self, project_id: str, options: dict[str, Any]) -> Job:
        if not settings.ai_toolkit_configured:
            raise AIToolkitError(
                "Ostris AI Toolkit is not configured. Set AI_TOOLKIT_PATH and "
                "AI_TOOLKIT_PYTHON in your .env file."
            )
        if project_id in self._active and self._active[project_id].is_running():
            raise LoRAProjectError("Training is already running for this project", status_code=409)

        config_path = self.write_config(project_id, options)
        log_path = safe_join(self.projects.output_dir(project_id), "training.log")

        job = self.jobs.submit(
            "lora_training",
            lambda job: self._run(job, project_id, config_path, log_path),
            metadata={"project_id": project_id, "config_path": str(config_path)},
        )
        self.projects.update_project(
            project_id, training_status="queued", training_job_id=job.id
        )
        return job

    def _run(self, job: Job, project_id: str, config_path: Path, log_path: Path) -> None:
        trainer = AIToolkitProcess(config_path, log_path)
        self._active[project_id] = trainer

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"=== Training started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n",
            encoding="utf-8",
        )

        trainer.start()
        self.projects.update_project(project_id, training_status="running")
        self.jobs.update(job.id, message="Training in progress", progress=0.0)

        reader = threading.Thread(target=trainer.stream, daemon=True)
        reader.start()

        while trainer.is_running():
            if self.jobs.is_cancelled(job.id):
                trainer.stop()
                self.projects.update_project(project_id, training_status="cancelled")
                logger.info("[LORA TRAINING] %s cancelled", project_id)
                return
            snapshot = trainer.snapshot()
            progress = trainer.progress()
            self.jobs.update(
                job.id,
                progress=progress,
                # Honest fallback: no reliable step info means no fake percentage.
                message=(
                    f"Step {snapshot['current_step']}/{snapshot['total_steps']}"
                    if snapshot["total_steps"]
                    else "Training in progress"
                ),
                metadata={**job.metadata, **snapshot},
            )
            time.sleep(2)

        reader.join(timeout=5)
        exit_code = trainer.process.returncode if trainer.process else -1
        self._active.pop(project_id, None)

        if exit_code != 0:
            self.projects.update_project(project_id, training_status="failed")
            raise AIToolkitError(
                f"AI Toolkit exited with code {exit_code}",
                status_code=500,
                detail=trainer.snapshot()["last_line"] or "See training.log for details.",
            )

        lora_path = self.find_trained_lora(project_id)
        if lora_path is None:
            self.projects.update_project(project_id, training_status="completed_no_output")
            raise AIToolkitError(
                "Training finished but no .safetensors file was found in the output folder",
                status_code=500,
            )

        self.projects.update_project(
            project_id, training_status="completed", trained_lora_path=str(lora_path)
        )
        self.jobs.update(
            job.id, outputs=[lora_path.name], message=f"LoRA ready: {lora_path.name}"
        )
        logger.info("[LORA TRAINING] %s completed -> %s", project_id, lora_path.name)

    def stop_training(self, project_id: str) -> bool:
        trainer = self._active.get(project_id)
        if trainer is None or not trainer.is_running():
            return False

        project = self.projects.get_project(project_id)
        if project.training_job_id:
            self.jobs.cancel(project.training_job_id)
        trainer.stop()
        self._active.pop(project_id, None)
        self.projects.update_project(project_id, training_status="cancelled")
        return True

    # ---------- introspection ----------

    def find_trained_lora(self, project_id: str) -> Path | None:
        """Newest valid safetensors in the project's output folder."""
        output = self.projects.output_dir(project_id)
        candidates = [
            path for path in output.rglob("*.safetensors") if is_safetensors(path)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def training_status(self, project_id: str) -> dict[str, Any]:
        project = self.projects.get_project(project_id)
        trainer = self._active.get(project_id)
        job = self.jobs.get(project.training_job_id) if project.training_job_id else None

        info: dict[str, Any] = {
            "project_id": project_id,
            "status": project.training_status,
            "job_id": project.training_job_id,
            "running": bool(trainer and trainer.is_running()),
            "progress": None,
            "current_step": 0,
            "total_steps": 0,
            "loss": None,
            "message": "Not started",
            "trained_lora_path": project.trained_lora_path,
        }

        if trainer:
            info.update(trainer.snapshot())
            info["progress"] = trainer.progress()
            info["message"] = (
                f"Step {info['current_step']}/{info['total_steps']}"
                if info["total_steps"]
                else "Training in progress"
            )
        if job:
            info["status"] = job.status.value
            info["message"] = job.message or info["message"]
            info["error"] = job.error
            if job.progress is not None:
                info["progress"] = job.progress
        return info

    def read_log(self, project_id: str, tail_lines: int = 200) -> str:
        log_path = safe_join(self.projects.output_dir(project_id), "training.log")
        if not log_path.is_file():
            return ""
        # ponytail: whole-file read. Training logs are a few MB at most; switch to
        # a seek-from-end reader only if they ever get large enough to matter.
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-tail_lines:])
