"""Hardware detection for training preflight.

Answers "what can this machine actually run?" without assuming NVIDIA.
Detection is best-effort and never raises: an unknown component simply
reports ``None`` and the advisor degrades to conservative advice.

Detection order per component:

* GPU: ComfyUI's ``/system_stats`` (already running, knows its own device),
  then ``nvidia-smi``, then ``torch.cuda``/``torch.version.hip`` if torch is
  importable, then AMD ROCm's ``rocm-smi``.
* System RAM: ``psutil`` when available, else ``ctypes``/``/proc/meminfo``.
* Disk: ``shutil.disk_usage`` on the target output directory.
"""

from __future__ import annotations

import logging
import shutil
import subprocess  # noqa: S404 - only fixed-argv probes of known binaries
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """One detected accelerator."""

    vendor: str  # "nvidia" | "amd" | "unknown"
    name: str
    vram_total_mb: int | None = None
    vram_free_mb: int | None = None
    source: str = ""  # which detector found it, for debugging

    def to_dict(self) -> dict[str, Any]:
        return {
            "vendor": self.vendor,
            "name": self.name,
            "vram_total_mb": self.vram_total_mb,
            "vram_free_mb": self.vram_free_mb,
            "source": self.source,
        }


@dataclass
class HardwareInfo:
    """Snapshot of the machine relevant to training decisions."""

    gpus: list[GPUInfo] = field(default_factory=list)
    cpu_name: str | None = None
    cpu_cores: int | None = None
    ram_total_mb: int | None = None
    ram_available_mb: int | None = None
    disk_free_gb: float | None = None
    cuda_available: bool = False
    rocm_available: bool = False

    @property
    def primary_gpu(self) -> GPUInfo | None:
        return self.gpus[0] if self.gpus else None

    @property
    def vram_total_mb(self) -> int | None:
        gpu = self.primary_gpu
        return gpu.vram_total_mb if gpu else None

    @property
    def accelerator(self) -> str:
        """Human summary: what training would actually run on."""
        gpu = self.primary_gpu
        if gpu is None:
            return "cpu"
        if gpu.vendor == "nvidia":
            return "cuda"
        if gpu.vendor == "amd":
            return "rocm" if self.rocm_available else "amd"
        return "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "primary_gpu": self.primary_gpu.to_dict() if self.primary_gpu else None,
            "accelerator": self.accelerator,
            "cpu_name": self.cpu_name,
            "cpu_cores": self.cpu_cores,
            "ram_total_mb": self.ram_total_mb,
            "ram_available_mb": self.ram_available_mb,
            "disk_free_gb": self.disk_free_gb,
            "cuda_available": self.cuda_available,
            "rocm_available": self.rocm_available,
        }


# ============================================
# GPU detection
# ============================================


def _detect_via_comfyui(base_url: str, timeout: float = 3.0) -> list[GPUInfo]:
    """Ask a running ComfyUI what devices it sees (most reliable when up)."""
    import requests

    try:
        response = requests.get(f"{base_url.rstrip('/')}/system_stats", timeout=timeout)
        devices = response.json().get("devices", [])
    except Exception:  # noqa: BLE001 - ComfyUI simply not running
        return []

    gpus: list[GPUInfo] = []
    for device in devices:
        name = str(device.get("name", ""))
        vendor = "unknown"
        lowered = name.lower()
        if "nvidia" in lowered or any(
            model in lowered for model in ("rtx", "gtx", "geforce", "quadro", "tesla")
        ):
            vendor = "nvidia"
        elif "amd" in lowered or "radeon" in lowered:
            vendor = "amd"
        vram_total = device.get("vram_total")
        vram_free = device.get("vram_free")
        gpus.append(GPUInfo(
            vendor=vendor,
            name=name or "GPU",
            # ComfyUI reports bytes.
            vram_total_mb=int(vram_total / (1024 * 1024)) if vram_total else None,
            vram_free_mb=int(vram_free / (1024 * 1024)) if vram_free else None,
            source="comfyui",
        ))
    return gpus


def _detect_via_nvidia_smi() -> list[GPUInfo]:
    """Query nvidia-smi with a fixed argv (no shell, no user input)."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    gpus: list[GPUInfo] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            gpus.append(GPUInfo(
                vendor="nvidia",
                name=parts[0],
                vram_total_mb=int(float(parts[1])),
                vram_free_mb=int(float(parts[2])),
                source="nvidia-smi",
            ))
        except ValueError:
            continue
    return gpus


def _detect_via_torch() -> list[GPUInfo]:
    """Last resort: torch's own device enumeration (import may be slow)."""
    try:
        import torch
    except ImportError:
        return []

    gpus: list[GPUInfo] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            name = getattr(props, "name", "") or "CUDA device"
            vendor = "amd" if torch.version.hip else "nvidia"
            gpus.append(GPUInfo(
                vendor=vendor,
                name=name,
                vram_total_mb=int(getattr(props, "total_memory", 0) / (1024 * 1024)),
                source="torch",
            ))
    return gpus


def detect_gpus(comfyui_base_url: str | None = None) -> list[GPUInfo]:
    """Detect accelerators through whichever channel answers first."""
    if comfyui_base_url:
        gpus = _detect_via_comfyui(comfyui_base_url)
        if gpus:
            return gpus

    gpus = _detect_via_nvidia_smi()
    if gpus:
        return gpus

    return _detect_via_torch()


# ============================================
# CPU / RAM / disk
# ============================================


def _detect_cpu() -> tuple[str | None, int | None]:
    name = None
    cores = None
    try:
        import platform

        name = platform.processor() or None
        cores = __import__("os").cpu_count()
    except Exception:  # noqa: BLE001
        pass
    return name, cores


def _detect_ram() -> tuple[int | None, int | None]:
    """(total_mb, available_mb), best effort."""
    try:
        import psutil

        memory = psutil.virtual_memory()
        return (
            int(memory.total / (1024 * 1024)),
            int(memory.available / (1024 * 1024)),
        )
    except ImportError:
        pass
    except Exception:  # noqa: BLE001
        pass

    # Linux fallback without psutil.
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            values = {}
            for line in handle:
                key, _, rest = line.partition(":")
                values[key.strip()] = int(rest.strip().split()[0])  # kB
        total = values.get("MemTotal")
        available = values.get("MemAvailable", values.get("MemFree"))
        return (
            int(total / 1024) if total else None,
            int(available / 1024) if available else None,
        )
    except (OSError, ValueError, IndexError):
        return None, None


def detect_disk_free_gb(path: str | None = None) -> float | None:
    """Free space on the drive that will hold training output."""
    import os

    target = path or os.getcwd()
    try:
        return round(shutil.disk_usage(target).free / (1024 ** 3), 1)
    except OSError:
        return None


# ============================================
# Public entry point
# ============================================


def detect_hardware(comfyui_base_url: str | None = None) -> HardwareInfo:
    """Snapshot the machine. Never raises; unknowns stay None."""
    gpus = detect_gpus(comfyui_base_url)
    cpu_name, cpu_cores = _detect_cpu()
    ram_total, ram_available = _detect_ram()

    info = HardwareInfo(
        gpus=gpus,
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        ram_total_mb=ram_total,
        ram_available_mb=ram_available,
        cuda_available=any(gpu.vendor == "nvidia" for gpu in gpus),
        rocm_available=any(gpu.vendor == "amd" for gpu in gpus),
    )
    return info
