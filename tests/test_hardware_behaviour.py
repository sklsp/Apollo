"""Behaviour tests for app/services/hardware.py (training preflight detection).

Detection is best-effort by design: every detector degrades to an empty list
or ``None`` instead of raising, and the advisor then gives conservative advice.
These tests pin down each degradation path and each response parse with
literal expected values. Fakes are injected via monkeypatch so no ComfyUI,
network or GPU is required; the real nvidia-smi subprocess path is only
exercised on machines that have a GPU (all of its parse branches are covered
by fakes here).
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from types import SimpleNamespace

from app.services.hardware import (
    GPUInfo,
    HardwareInfo,
    _detect_cpu,
    _detect_ram,
    _detect_via_comfyui,
    _detect_via_nvidia_smi,
    _detect_via_torch,
    detect_disk_free_gb,
    detect_gpus,
    detect_hardware,
)

MODULE = "app.services.hardware"


def _fake_response(payload):
    """A stand-in for requests.Response: only .json() is used."""
    return SimpleNamespace(status_code=200, json=lambda: payload)


# ============================================
# Dataclasses
# ============================================


class TestGPUInfo:
    def test_to_dict_round_trips_all_fields(self):
        gpu = GPUInfo(
            vendor="nvidia",
            name="NVIDIA GeForce RTX 4090",
            vram_total_mb=24576,
            vram_free_mb=18432,
            source="comfyui",
        )
        assert gpu.to_dict() == {
            "vendor": "nvidia",
            "name": "NVIDIA GeForce RTX 4090",
            "vram_total_mb": 24576,
            "vram_free_mb": 18432,
            "source": "comfyui",
        }


class TestHardwareInfo:
    def test_empty_snapshot_is_cpu_with_no_vram(self):
        info = HardwareInfo()
        assert info.primary_gpu is None
        assert info.vram_total_mb is None
        assert info.accelerator == "cpu"

    def test_accelerator_cuda_for_nvidia_primary(self):
        info = HardwareInfo(gpus=[GPUInfo(vendor="nvidia", name="RTX 4090")])
        assert info.primary_gpu.name == "RTX 4090"
        assert info.accelerator == "cuda"

    def test_accelerator_rocm_only_when_flag_set(self):
        amd = GPUInfo(vendor="amd", name="RX 7900 XTX")
        assert HardwareInfo(gpus=[amd]).accelerator == "amd"
        flagged = HardwareInfo(gpus=[amd], rocm_available=True)
        assert flagged.accelerator == "rocm"

    def test_accelerator_unknown_vendor(self):
        info = HardwareInfo(gpus=[GPUInfo(vendor="unknown", name="M3 Ultra")])
        assert info.accelerator == "unknown"

    def test_to_dict_includes_primary_and_flags(self):
        gpu = GPUInfo(
            vendor="nvidia",
            name="RTX 4090",
            vram_total_mb=24576,
            vram_free_mb=None,
            source="torch",
        )
        info = HardwareInfo(
            gpus=[gpu],
            cpu_name="Ryzen 9",
            cpu_cores=16,
            ram_total_mb=32768,
            ram_available_mb=16384,
            disk_free_gb=512.0,
            cuda_available=True,
        )
        assert info.to_dict() == {
            "gpus": [
                {
                    "vendor": "nvidia",
                    "name": "RTX 4090",
                    "vram_total_mb": 24576,
                    "vram_free_mb": None,
                    "source": "torch",
                }
            ],
            "primary_gpu": {
                "vendor": "nvidia",
                "name": "RTX 4090",
                "vram_total_mb": 24576,
                "vram_free_mb": None,
                "source": "torch",
            },
            "accelerator": "cuda",
            "cpu_name": "Ryzen 9",
            "cpu_cores": 16,
            "ram_total_mb": 32768,
            "ram_available_mb": 16384,
            "disk_free_gb": 512.0,
            "cuda_available": True,
            "rocm_available": False,
        }


# ============================================
# ComfyUI detector
# ============================================


class TestComfyUIDetection:
    def test_parses_nvidia_device_and_converts_bytes_to_mb(self, monkeypatch):
        payload = {
            "devices": [
                {
                    "name": "NVIDIA GeForce RTX 4090",
                    # ComfyUI reports bytes: 24 GiB total, 18 GiB free.
                    "vram_total": 25769803776,
                    "vram_free": 19327352832,
                }
            ]
        }
        monkeypatch.setattr("requests.get", lambda *a, **k: _fake_response(payload))

        gpus = _detect_via_comfyui("http://127.0.0.1:8188")

        assert gpus == [
            GPUInfo(
                vendor="nvidia",
                name="NVIDIA GeForce RTX 4090",
                vram_total_mb=24576,
                vram_free_mb=18432,
                source="comfyui",
            )
        ]

    def test_parses_amd_by_radeon_name(self, monkeypatch):
        payload = {"devices": [{"name": "AMD Radeon RX 7900 XTX"}]}
        monkeypatch.setattr("requests.get", lambda *a, **k: _fake_response(payload))

        gpus = _detect_via_comfyui("http://127.0.0.1:8188")

        assert [gpu.vendor for gpu in gpus] == ["amd"]
        assert gpus[0].name == "AMD Radeon RX 7900 XTX"

    def test_unknown_vendor_and_missing_vram_stay_none(self, monkeypatch):
        payload = {"devices": [{"name": ""}, {"name": "Apple M3 Ultra"}]}
        monkeypatch.setattr("requests.get", lambda *a, **k: _fake_response(payload))

        gpus = _detect_via_comfyui("http://127.0.0.1:8188")

        assert len(gpus) == 2
        # Empty name falls back to the literal "GPU".
        assert gpus[0].name == "GPU"
        assert gpus[0].vendor == "unknown"
        assert gpus[1].vram_total_mb is None
        assert gpus[1].vram_free_mb is None

    def test_empty_devices_list_returns_empty(self, monkeypatch):
        monkeypatch.setattr("requests.get", lambda *a, **k: _fake_response({"devices": []}))
        assert _detect_via_comfyui("http://127.0.0.1:8188") == []

    def test_connection_error_returns_empty(self, monkeypatch):
        import requests as real_requests

        def raise_conn(*args, **kwargs):
            raise real_requests.ConnectionError("refused")

        monkeypatch.setattr("requests.get", raise_conn)
        assert _detect_via_comfyui("http://127.0.0.1:8188") == []

    def test_malformed_json_returns_empty(self, monkeypatch):
        def broken_json():
            raise ValueError("No JSON object could be decoded")

        response = SimpleNamespace(status_code=200, json=broken_json)
        monkeypatch.setattr("requests.get", lambda *a, **k: response)
        assert _detect_via_comfyui("http://127.0.0.1:8188") == []

    def test_url_strips_trailing_slash_and_passes_timeout(self, monkeypatch):
        seen = {}

        def record(url, **kwargs):
            seen["url"] = url
            seen.update(kwargs)
            return _fake_response({"devices": []})

        monkeypatch.setattr("requests.get", record)
        _detect_via_comfyui("http://127.0.0.1:8188/", timeout=1.5)

        assert seen["url"] == "http://127.0.0.1:8188/system_stats"
        assert seen["timeout"] == 1.5


# ============================================
# nvidia-smi detector
# ============================================


class TestNvidiaSmiDetection:
    def test_missing_binary_returns_empty(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)
        assert _detect_via_nvidia_smi() == []

    def test_parses_csv_output_into_gpu_info(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "C:\\WINDOWS\\system32\\nvidia-smi.EXE")
        result = SimpleNamespace(stdout="NVIDIA GeForce RTX 4090, 24564, 18321\n")
        monkeypatch.setattr("subprocess.run", lambda *a, **k: result)

        gpus = _detect_via_nvidia_smi()

        assert gpus == [
            GPUInfo(
                vendor="nvidia",
                name="NVIDIA GeForce RTX 4090",
                vram_total_mb=24564,
                vram_free_mb=18321,
                source="nvidia-smi",
            )
        ]

    def test_skips_malformed_lines_but_keeps_valid_ones(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "C:\\WINDOWS\\system32\\nvidia-smi.EXE")
        result = SimpleNamespace(
            stdout=(
                "NVIDIA A100 80GB, 81920, 70000\n"
                "garbage line without commas\n"
                "BROKEN GPU, not-a-number, also-bad\n"
            )
        )
        monkeypatch.setattr("subprocess.run", lambda *a, **k: result)

        gpus = _detect_via_nvidia_smi()

        assert len(gpus) == 1
        assert gpus[0].name == "NVIDIA A100 80GB"
        assert (gpus[0].vram_total_mb, gpus[0].vram_free_mb) == (81920, 70000)

    def test_oserror_returns_empty(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "C:\\WINDOWS\\system32\\nvidia-smi.EXE")

        def raise_oserror(*args, **kwargs):
            raise OSError("access denied")

        monkeypatch.setattr("subprocess.run", raise_oserror)
        assert _detect_via_nvidia_smi() == []

    def test_timeout_returns_empty(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "C:\\WINDOWS\\system32\\nvidia-smi.EXE")

        def raise_timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)

        monkeypatch.setattr("subprocess.run", raise_timeout)
        assert _detect_via_nvidia_smi() == []



# ============================================
# torch detector (fake module injected into sys.modules)
# ============================================


def _install_fake_torch(monkeypatch, available, hip=None, devices=()):
    fake = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: available,
            device_count=lambda: len(devices),
            get_device_properties=lambda index: devices[index],
        ),
        version=SimpleNamespace(hip=hip),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)


class TestTorchDetection:
    def test_import_error_returns_empty(self, monkeypatch):
        # None in sys.modules makes `import torch` raise ImportError.
        monkeypatch.setitem(sys.modules, "torch", None)
        assert _detect_via_torch() == []

    def test_cuda_unavailable_returns_empty(self, monkeypatch):
        _install_fake_torch(monkeypatch, available=False)
        assert _detect_via_torch() == []

    def test_enumerates_nvidia_devices_with_vram(self, monkeypatch):
        # 80 GiB in bytes -> 81920 MB.
        device = SimpleNamespace(name="NVIDIA A100 80GB", total_memory=85899345920)
        _install_fake_torch(monkeypatch, available=True, hip=None, devices=[device])

        gpus = _detect_via_torch()

        assert gpus == [
            GPUInfo(
                vendor="nvidia",
                name="NVIDIA A100 80GB",
                vram_total_mb=81920,
                vram_free_mb=None,
                source="torch",
            )
        ]

    def test_hip_build_reports_amd_vendor(self, monkeypatch):
        device = SimpleNamespace(name="AMD Instinct MI300X", total_memory=4294967296)
        _install_fake_torch(monkeypatch, available=True, hip="6.3", devices=[device])

        gpus = _detect_via_torch()

        assert gpus[0].vendor == "amd"
        # 4 GiB -> 4096 MB.
        assert gpus[0].vram_total_mb == 4096

    def test_missing_name_and_memory_attrs_use_defaults(self, monkeypatch):
        _install_fake_torch(monkeypatch, available=True, hip=None, devices=[SimpleNamespace()])

        gpus = _detect_via_torch()

        assert gpus[0].name == "CUDA device"
        assert gpus[0].vram_total_mb == 0


# ============================================
# detect_gpus orchestration (fallback order)
# ============================================


class TestDetectGpus:
    def test_comfyui_wins_when_it_answers(self, monkeypatch):
        comfy = [GPUInfo(vendor="nvidia", name="from-comfyui")]
        monkeypatch.setattr(f"{MODULE}._detect_via_comfyui", lambda url: comfy)

        called = {}

        def boom():
            called["yes"] = True
            return []

        monkeypatch.setattr(f"{MODULE}._detect_via_nvidia_smi", boom)

        assert detect_gpus("http://127.0.0.1:8188") == comfy
        # nvidia-smi must not be consulted once ComfyUI answered.
        assert "yes" not in called

    def test_falls_back_to_nvidia_smi_when_comfyui_empty(self, monkeypatch):
        smi = [GPUInfo(vendor="nvidia", name="from-smi")]
        monkeypatch.setattr(f"{MODULE}._detect_via_comfyui", lambda url: [])
        monkeypatch.setattr(f"{MODULE}._detect_via_nvidia_smi", lambda: smi)

        assert detect_gpus("http://127.0.0.1:8188") == smi

    def test_falls_back_to_torch_when_others_empty(self, monkeypatch):
        torch_gpu = [GPUInfo(vendor="nvidia", name="from-torch")]
        monkeypatch.setattr(f"{MODULE}._detect_via_comfyui", lambda url: [])
        monkeypatch.setattr(f"{MODULE}._detect_via_nvidia_smi", lambda: [])
        monkeypatch.setattr(f"{MODULE}._detect_via_torch", lambda: torch_gpu)

        assert detect_gpus("http://127.0.0.1:8188") == torch_gpu

    def test_without_url_skips_comfyui_entirely(self, monkeypatch):
        smi = [GPUInfo(vendor="nvidia", name="from-smi")]

        def must_not_run(url):
            raise AssertionError("comfyui detector should not run without a URL")

        monkeypatch.setattr(f"{MODULE}._detect_via_comfyui", must_not_run)
        monkeypatch.setattr(f"{MODULE}._detect_via_nvidia_smi", lambda: smi)

        assert detect_gpus() == smi



# ============================================
# CPU / RAM / disk
# ============================================


class TestCpuRamDisk:
    def test_detect_cpu_returns_real_machine_values(self):
        name, cores = _detect_cpu()
        # os.cpu_count is the source of truth here, not the code under test.
        assert cores == os.cpu_count()
        assert isinstance(cores, int) and cores > 0
        assert name is None or isinstance(name, str)

    def test_detect_cpu_swallows_platform_errors(self, monkeypatch):
        def broken_processor():
            raise RuntimeError("no processor info")

        monkeypatch.setattr("platform.processor", broken_processor)
        # The exception aborts the whole try block: both values stay None.
        assert _detect_cpu() == (None, None)

    def test_detect_ram_with_psutil_converts_bytes_to_mb(self, monkeypatch):
        fake_psutil = SimpleNamespace(
            virtual_memory=lambda: SimpleNamespace(total=16 * 1024**3, available=8 * 1024**3)
        )
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        assert _detect_ram() == (16384, 8192)

    def test_detect_ram_psutil_crash_falls_back_to_meminfo(self, monkeypatch):
        # psutil is importable but its call blows up: the except-Exception
        # branch must swallow it and fall through to /proc/meminfo.
        def broken_virtual_memory():
            raise RuntimeError("psutil exploded")

        fake_psutil = SimpleNamespace(virtual_memory=broken_virtual_memory)
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
        meminfo = "MemTotal:       8192000 kB\nMemAvailable:    4096000 kB\n"
        real_open = open

        def fake_open(path, *args, **kwargs):
            if str(path) == "/proc/meminfo":
                return io.StringIO(meminfo)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)

        assert _detect_ram() == (8000, 4000)

    def test_detect_ram_meminfo_parse_without_psutil(self, monkeypatch):
        # psutil absent -> ImportError path; /proc/meminfo faked for Windows.
        monkeypatch.setitem(sys.modules, "psutil", None)
        meminfo = (
            "MemTotal:       16384000 kB\n"
            "MemFree:         1024000 kB\n"
            "MemAvailable:    8192000 kB\n"
        )
        real_open = open

        def fake_open(path, *args, **kwargs):
            if str(path) == "/proc/meminfo":
                return io.StringIO(meminfo)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)

        # kB -> MB: 16384000/1024 = 16000; MemAvailable wins over MemFree.
        assert _detect_ram() == (16000, 8000)

    def test_detect_ram_meminfo_uses_memfree_when_no_available(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "psutil", None)
        meminfo = "MemTotal:       4096000 kB\nMemFree:        2048000 kB\n"
        real_open = open

        def fake_open(path, *args, **kwargs):
            if str(path) == "/proc/meminfo":
                return io.StringIO(meminfo)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)

        assert _detect_ram() == (4000, 2000)

    def test_detect_ram_meminfo_malformed_returns_none(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "psutil", None)
        meminfo = "MemTotal: garbage\n"
        real_open = open

        def fake_open(path, *args, **kwargs):
            if str(path) == "/proc/meminfo":
                return io.StringIO(meminfo)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)

        assert _detect_ram() == (None, None)

    def test_detect_disk_free_gb_reports_real_drive(self, tmp_path):
        free = detect_disk_free_gb(str(tmp_path))
        assert isinstance(free, float)
        assert free >= 0.0

    def test_detect_disk_free_gb_oserror_returns_none(self, monkeypatch):
        def raise_oserror(path):
            raise OSError("drive not ready")

        monkeypatch.setattr("shutil.disk_usage", raise_oserror)
        assert detect_disk_free_gb("/definitely/missing") is None


# ============================================
# Public entry point
# ============================================


class TestDetectHardware:
    def test_nvidia_gpu_sets_cuda_flag(self, monkeypatch):
        gpus = [GPUInfo(vendor="nvidia", name="RTX 4090")]
        monkeypatch.setattr(f"{MODULE}.detect_gpus", lambda url=None: gpus)

        info = detect_hardware()

        assert info.gpus == gpus
        assert info.cuda_available is True
        assert info.rocm_available is False
        assert info.accelerator == "cuda"

    def test_amd_gpu_sets_rocm_flag(self, monkeypatch):
        gpus = [GPUInfo(vendor="amd", name="RX 7900 XTX")]
        monkeypatch.setattr(f"{MODULE}.detect_gpus", lambda url=None: gpus)

        info = detect_hardware()

        assert info.cuda_available is False
        assert info.rocm_available is True
        # detect_hardware sets rocm_available for AMD GPUs, so the property
        # reports "rocm" rather than plain "amd".
        assert info.accelerator == "rocm"

    def test_no_gpu_reports_cpu(self, monkeypatch):
        monkeypatch.setattr(f"{MODULE}.detect_gpus", lambda url=None: [])

        info = detect_hardware()

        assert info.gpus == []
        assert info.cuda_available is False
        assert info.rocm_available is False
        assert info.accelerator == "cpu"

