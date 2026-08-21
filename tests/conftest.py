"""Shared pytest fixtures and collection rules."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# These files predate the test suite: they are standalone diagnostic scripts
# that print at import time and, in several cases, require a live Ollama server.
# They are kept (they are useful to run by hand: `python tests/test_endpoints.py`)
# but excluded from automated collection, where they would hang or error.
collect_ignore = [
    "debug_chat_route.py",
    "test_chat_error.py",
    "test_endpoints.py",
    "test_ollama_detailed.py",
    "test_ollama_endpoints.py",
    "test_routes_debug.py",
    "test_routes_detailed.py",
]


@pytest.fixture
def tmp_settings(tmp_path):
    """Point every writable storage root at a temp dir for the duration of a test.

    ``Settings`` is a frozen dataclass and every module holds a reference to the
    same singleton, so the fields are patched in place via ``object.__setattr__``
    and restored afterwards rather than swapped for a copy.
    """
    from app.core import config

    replacements = {
        "lora_data_dir": str(tmp_path / "loras"),
        "generated_dir": str(tmp_path / "generated"),
        "comfyui_workflow_dir": str(tmp_path / "workflows"),
        "jobs_file": str(tmp_path / "jobs.json"),
        "comfyui_lora_dir": "",
        "ai_toolkit_path": "",
        "ai_toolkit_python": "",
    }
    originals = {name: getattr(config.settings, name) for name in replacements}

    for name, value in replacements.items():
        object.__setattr__(config.settings, name, value)
    for key in ("lora_data_dir", "generated_dir", "comfyui_workflow_dir"):
        Path(replacements[key]).mkdir(parents=True, exist_ok=True)

    yield config.settings

    for name, value in originals.items():
        object.__setattr__(config.settings, name, value)


@pytest.fixture
def set_setting(tmp_settings):
    """Override one more setting inside a test, restored at teardown."""
    from app.core import config

    changed: dict[str, object] = {}

    def _set(name: str, value: object) -> None:
        changed.setdefault(name, getattr(config.settings, name))
        object.__setattr__(config.settings, name, value)

    yield _set

    for name, value in changed.items():
        object.__setattr__(config.settings, name, value)


@pytest.fixture
def project_service(tmp_settings):
    from app.services.lora_dataset_service import LoRAProjectService

    return LoRAProjectService(data_dir=tmp_settings.lora_data_dir)


def png_bytes(size: int = 64) -> bytes:
    """A minimal but genuinely valid 1x1 PNG, padded to ``size`` bytes."""
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000a49444154789c6360000002000100ffff0300000600055773d8b400"
        "00000049454e44ae426082"
    )
    return png + b"\x00" * max(0, size - len(png))
