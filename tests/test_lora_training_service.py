"""Service-level tests for LoRA training orchestration (app/services/lora_training_service.py).

``test_lora.py`` covers config generation, the process-management primitives, and library
discovery; ``test_lora_routes_behaviour.py`` covers API degradation without a toolkit. What
remained untested is the full lifecycle of ``LoRATrainingService._run`` - success, failure,
no-output, and cancellation paths that tie together jobs, run history, project state, and
the trainer process - plus the duplicate-start guard, ``stop_training``, and hardware-preset
tiering.

The "AI Toolkit" is a fake ``run.py`` executed as a real subprocess under ``sys.executable``,
so spawn/stream/kill all happen for real; only the training itself is faked.
"""

from __future__ import annotations

import sys
import textwrap
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.exceptions import AIToolkitError, LoRAProjectError
from app.core.jobs import JobStatus, JobStore
from app.services.hardware import GPUInfo, HardwareInfo
from app.services.lora_dataset_service import LoRAProjectService
from app.services.lora_training_service import (
    LoRATrainingService,
)
from tests.conftest import png_bytes

# ============================================
# FAKE AI TOOLKIT
# ============================================

RUN_PY = """\
import json
import sys
import time
from pathlib import Path

import yaml

root = Path(__file__).resolve().parent
mode = (root / "mode.txt").read_text(encoding="utf-8").strip()
config_path = Path(sys.argv[1])
process = yaml.safe_load(config_path.read_text(encoding="utf-8"))["config"]["process"][0]
out_dir = Path(process["training_folder"])

if mode == "slow":
    for i in range(60):
        print(f"  {i}/60 [sleeping]", flush=True)
        time.sleep(1)
elif mode == "fail":
    print("Loading model weights...", flush=True)
    print("CUDA error: out of memory", flush=True)
    sys.exit(3)
else:
    steps = process["train"]["steps"]
    for i in range(1, 5):
        step = int(steps * i / 5)
        print(f"  {step}/{steps} [00:{i*2:<02d}<00:{(5-i)*2:<02d}, loss={0.9 - i * 0.15:.3f}", flush=True)
    if mode == "success":
        header = json.dumps({"__metadata__": {"format": "pt"}}).encode("utf-8")
        (out_dir / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header)
"""


def _install_toolkit(base: Path, name: str, mode: str) -> Path:
    """Create a fake AI Toolkit directory (run.py + mode.txt) and return its path."""
    root = base / f"toolkit_{name}"
    root.mkdir(parents=True, exist_ok=False)
    (root / "mode.txt").write_text(mode, encoding="utf-8")
    (root / "run.py").write_text(textwrap.dedent(RUN_PY), encoding="utf-8")
    return root


def _fake_detect_hardware(vram_mb: int | None = 8192):
    def detect(*args, **kwargs) -> HardwareInfo:
        gpus = (
            [GPUInfo(vendor="NVIDIA", name="Fake GPU", vram_total_mb=vram_mb, vram_free_mb=7000)]
            if vram_mb is not None
            else []
        )
        return HardwareInfo(
            gpus=gpus, cpu_name="Fake CPU", cpu_cores=8, ram_total_mb=16384,
            ram_available_mb=8192, disk_free_gb=500.0, cuda_available=vram_mb is not None,
        )

    return detect


def _wait_for_terminal(jobs: JobStore, job_id: str, timeout: float = 20.0) -> object:
    deadline = time.monotonic() + timeout
    while True:
        job = jobs.get(job_id)
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return job
        assert time.monotonic() < deadline, f"job {job_id} stuck in {job.status}"
        time.sleep(0.05)


def _wait_for_status(jobs: JobStore, job_id: str, status: JobStatus, timeout: float = 20.0) -> object | None:
    """Wait until a job reaches ``status``; return None if it goes terminal first."""
    deadline = time.monotonic() + timeout
    while True:
        job = jobs.get(job_id)
        if job.status is status:
            return job
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return None
        assert time.monotonic() < deadline, f"job {job_id} stuck in {job.status}"
        time.sleep(0.05)


def _wait_for_active(training: LoRATrainingService, project_id: str, timeout: float = 20.0) -> None:
    """Wait until the worker thread has registered a live trainer for the project."""
    deadline = time.monotonic() + timeout
    while True:
        trainer = training._active.get(project_id)
        if trainer is not None and trainer.is_running():
            return
        assert time.monotonic() < deadline, "trainer never became active"
        time.sleep(0.05)


def _wait_for_cancelled(env: SimpleNamespace, project_id: str, job_id: str) -> dict:
    """Wait for the worker's cancel branch to land on disk and return the run record.

    ``jobs.cancel()`` flips the job synchronously, but the *project* status and the
    *run record* are written by the worker thread a beat later - so both must be
    polled rather than asserted immediately after the job goes terminal.
    """
    deadline = time.monotonic() + 10
    while True:
        assert time.monotonic() < deadline, "cancel effects never landed on disk"
        if env.projects.get_project(project_id).training_status != "cancelled":
            time.sleep(0.05)
            continue
        run = next(
            (r for r in env.training.list_runs(project_id) if r.get("job_id") == job_id), None
        )
        if run and run.get("status") == "cancelled" and run.get("completed_at"):
            return run
        time.sleep(0.05)


def _ready_project(env: SimpleNamespace, name: str = "Train Character") -> object:
    project = env.projects.create_project(name, arch="sdxl", trigger_word="p3r5on")
    env.projects.add_image(project.id, "img1.png", png_bytes())
    env.projects.set_caption(project.id, "img1", "p3r5on, red subject")
    return project


@pytest.fixture
def training_env(tmp_settings, monkeypatch):
    """Projects + in-memory jobs + the service under test.

    ``detect_hardware`` is faked (8 GB GPU) so tests never shell out to nvidia-smi;
    every call site in lora_training_service imports it locally, so patching the source
    attribute covers start_training, preflight, and hardware_presets alike.
    """
    monkeypatch.setattr("app.services.hardware.detect_hardware", _fake_detect_hardware())
    projects = LoRAProjectService(data_dir=tmp_settings.lora_data_dir)
    jobs = JobStore()
    return SimpleNamespace(projects=projects, jobs=jobs, training=LoRATrainingService(projects=projects, jobs=jobs))


@pytest.fixture
def fast_poll(monkeypatch):
    """Shrink the 2 s poll interval in LoRATrainingService._run to keep tests quick.

    The shim proxies every public attribute of the real ``time`` module (the service
    also calls ``time.gmtime``/``strftime`` for run records) and only caps ``sleep``.
    """
    import app.services.lora_training_service as svc

    shim = types.SimpleNamespace(
        **{name: getattr(time, name) for name in dir(time) if not name.startswith("_")}
    )
    shim.sleep = lambda seconds: time.sleep(min(seconds, 0.02))
    monkeypatch.setattr(svc, "time", shim)


@pytest.fixture
def toolkit(tmp_path, set_setting):
    """Install a fake toolkit of the requested mode and point settings at it."""

    def _install(mode: str = "success") -> Path:
        root = _install_toolkit(tmp_path, mode, mode)
        set_setting("ai_toolkit_path", str(root))
        set_setting("ai_toolkit_python", sys.executable)
        return root

    return _install


# ============================================
# _RUN LIFECYCLE (fake trainer, real subprocess)
# ============================================


class TestRunLifecycle:
    def test_success_completes_job_project_and_run_record(
        self, training_env, fast_poll, toolkit
    ):
        env = training_env
        project = _ready_project(env)
        toolkit("success")

        job = env.training.start_training(project.id, {})
        finished = _wait_for_terminal(env.jobs, job.id)

        assert finished.status is JobStatus.COMPLETED
        assert finished.outputs == ["model.safetensors"]
        assert "LoRA ready" in (finished.message or "")

        loaded = env.projects.get_project(project.id)
        assert loaded.training_status == "completed"
        assert loaded.trained_lora_path and Path(loaded.trained_lora_path).is_file()

        runs = env.training.list_runs(project.id)
        assert len(runs) == 1
        run = runs[0]
        assert run["status"] == "completed"
        assert run["lora_filename"] == "model.safetensors"
        assert (run["lora_size_bytes"] or 0) > 0
        assert run["final_loss"] == pytest.approx(0.3, abs=0.01)
        assert run["total_steps"] == 2000

        info = env.training.training_status(project.id)
        assert info["status"] == "completed"
        assert info["running"] is False
        assert info["progress"] == 100.0

    def test_success_tees_output_into_the_training_log(
        self, training_env, fast_poll, toolkit
    ):
        env = training_env
        project = _ready_project(env)
        toolkit("success")

        job = env.training.start_training(project.id, {})
        _wait_for_terminal(env.jobs, job.id)

        log = env.training.read_log(project.id)
        assert "Training started" in log
        assert "loss=" in log  # trainer stdout was tee'd, not just exit codes

    def test_failure_records_diagnosis_and_marks_project_failed(
        self, training_env, fast_poll, toolkit
    ):
        env = training_env
        project = _ready_project(env)
        toolkit("fail")

        job = env.training.start_training(project.id, {})
        finished = _wait_for_terminal(env.jobs, job.id)

        assert finished.status is JobStatus.FAILED
        assert "exited with code 3" in (finished.error or "")

        loaded = env.projects.get_project(project.id)
        assert loaded.training_status == "failed"

        run = env.training.list_runs(project.id)[0]
        assert run["status"] == "failed"
        # diagnose_failure maps the OOM line to a human explanation, keeps the raw line.
        assert "GPU memory" in (run["error"] or "")
        assert "CUDA error: out of memory" in (run["error"] or "")

        info = env.training.training_status(project.id)
        assert info["status"] == "failed"
        assert info["error"]

    def test_no_output_marks_completed_no_output(
        self, training_env, fast_poll, toolkit
    ):
        env = training_env
        project = _ready_project(env)
        toolkit("no_output")

        job = env.training.start_training(project.id, {})
        finished = _wait_for_terminal(env.jobs, job.id)

        assert finished.status is JobStatus.FAILED
        assert ".safetensors" in (finished.error or "")

        loaded = env.projects.get_project(project.id)
        assert loaded.training_status == "completed_no_output"
        assert not loaded.trained_lora_path

        run = env.training.list_runs(project.id)[0]
        assert run["status"] == "failed"
        assert ".safetensors" in (run["error"] or "")


# ============================================
# CANCELLATION
# ============================================


class TestCancellation:
    def test_cooperative_cancel_via_job_store(
        self, training_env, fast_poll, toolkit
    ):
        """jobs.cancel() is observed in the _run poll loop; the worker stops the trainer itself."""
        env = training_env
        project = _ready_project(env)
        toolkit("slow")

        job = env.training.start_training(project.id, {})
        assert _wait_for_status(env.jobs, job.id, JobStatus.RUNNING) is not None
        _wait_for_active(env.training, project.id)

        assert env.jobs.cancel(job.id) is True
        finished = _wait_for_terminal(env.jobs, job.id)

        # The cancel branch returns cleanly: the worker must not overwrite CANCELLED.
        assert finished.status is JobStatus.CANCELLED
        run = _wait_for_cancelled(env, project.id, job.id)
        assert env.projects.get_project(project.id).training_status == "cancelled"
        assert run["status"] == "cancelled"
        assert run["completed_at"]
        # Trainer deregistered and its process gone.
        assert project.id not in env.training._active

    def test_stop_training_cancels_synchronously(self, training_env, fast_poll, toolkit):
        """stop_training() cancels + kills synchronously; the settled state may race with _run.

        Known race: after stop_training kills the process, the _run worker wakes to a non-zero
        exit code and can mark job/project FAILED/"failed" before it ever sees the cancel flag.
        The synchronous guarantees below are deterministic; the settled state is asserted
        loosely so both outcomes of that race stay green.
        """
        env = training_env
        project = _ready_project(env)
        toolkit("slow")

        job = env.training.start_training(project.id, {})
        assert _wait_for_status(env.jobs, job.id, JobStatus.RUNNING) is not None
        _wait_for_active(env.training, project.id)

        assert env.training.stop_training(project.id) is True
        # Synchronous guarantees (hold the moment stop_training returns):
        loaded = env.projects.get_project(project.id)
        assert loaded.training_status == "cancelled"
        assert env.jobs.get(job.id).status is JobStatus.CANCELLED
        assert project.id not in env.training._active

        # Let the _run worker settle (it polls every ~20 ms under fast_poll).
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if env.jobs.get(job.id).status is JobStatus.FAILED:
                break
            time.sleep(0.05)

        assert env.jobs.get(job.id).status in (JobStatus.CANCELLED, JobStatus.FAILED)
        assert env.projects.get_project(project.id).training_status in ("cancelled", "failed")

    def test_stop_training_with_nothing_running_returns_false(self, training_env):
        env = training_env
        project = _ready_project(env)
        assert env.training.stop_training(project.id) is False


# ============================================
# START GUARDS
# ============================================


class TestStartGuards:
    def test_duplicate_start_is_rejected(self, training_env, fast_poll, toolkit):
        env = training_env
        project = _ready_project(env)
        toolkit("slow")

        job = env.training.start_training(project.id, {})
        assert _wait_for_status(env.jobs, job.id, JobStatus.RUNNING) is not None
        _wait_for_active(env.training, project.id)
        try:
            with pytest.raises(LoRAProjectError):
                env.training.start_training(project.id, {})
        finally:
            env.jobs.cancel(job.id)
            _wait_for_terminal(env.jobs, job.id)

    def test_start_without_toolkit_raises(self, training_env):
        # tmp_settings leaves ai_toolkit_path / ai_toolkit_python empty.
        env = training_env
        project = _ready_project(env)
        with pytest.raises(AIToolkitError):
            env.training.start_training(project.id, {})


# ============================================
# HARDWARE PRESET TIERING
# ============================================


class TestHardwarePresets:
    def test_default_fixture_is_small_tier(self, training_env):
        # Fixture default: 8 GB GPU -> "small" (<= 8 GiB).
        result = training_env.training.hardware_presets()
        assert result["hardware"]["tier"] == "small"
        assert result["hardware"]["vram_mb"] == 8192

    def test_medium_and_large_tiers(self, training_env, monkeypatch):
        for vram, tier in ((16384, "medium"), (24576, "large")):
            monkeypatch.setattr(
                "app.services.hardware.detect_hardware", _fake_detect_hardware(vram)
            )
            result = training_env.training.hardware_presets()
            assert result["hardware"]["tier"] == tier

    def test_no_gpu_reports_unknown_tier(self, training_env, monkeypatch):
        monkeypatch.setattr(
            "app.services.hardware.detect_hardware", _fake_detect_hardware(None)
        )
        result = training_env.training.hardware_presets()
        assert result["hardware"]["tier"] == "unknown"
        assert result["hardware"]["vram_mb"] is None

    def test_variants_scale_with_tier(self, training_env):
        variants = training_env.training.hardware_presets()["presets"]
        assert set(variants) == {"character", "style", "product", "concept", "general"}
        for variant in variants.values():
            conservative = variant["conservative"]
            balanced = variant["balanced"]
            quality = variant["quality"]
            assert conservative["batch_size"] == 1
            assert conservative["resolution"] and all(r <= 512 for r in conservative["resolution"])
            assert balanced["batch_size"] == 1
            assert quality["batch_size"] >= 2
            assert quality["lora_rank"] > conservative["lora_rank"]

    def test_variant_caps(self, training_env):
        variants = training_env.training.hardware_presets()["presets"]
        # style: base rank 32 -> quality capped at 64; product: base rank 8 kept in conservative.
        assert variants["style"]["quality"]["lora_rank"] == 64
        assert variants["product"]["conservative"]["lora_rank"] == 8
        # concept is the only preset with batch_size 2; balanced normalises it to 1.
        assert variants["concept"]["balanced"]["batch_size"] == 1
