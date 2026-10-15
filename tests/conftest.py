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


def _install_shared_embedding_model() -> None:
    """Make repeated sentence-transformers loads share one model instance.

    ``EmbeddingClient`` falls back to ``SentenceTransformer("all-MiniLM-L6-v2")``
    once per instance (app/services/rag/embeddings.py), and sentence-transformers
    6.x keeps no class-level cache, so every RAGService construction re-reads the
    model from disk (~3-6 s each; ~14 s cold including the torch import). The
    tests never assert on vector values or on how often a model is loaded — only
    that an embedding backend exists and works — so sharing one real instance
    preserves behaviour while removing the suite's dominant cost. Ollama itself
    stays genuinely unreachable: every client still attempts (and fails) its HTTP
    probe before falling back.
    """
    try:
        import sentence_transformers as st
    except ImportError:  # pragma: no cover - ST is required for the fallback path
        return

    cls = st.SentenceTransformer
    if getattr(cls, "_apollo_shared_model", False):
        return

    original_init = cls.__init__
    shared: dict[str, object] = {}

    def cached_init(self, model_name_or_path=None, *args, **kwargs) -> None:
        key = (
            model_name_or_path
            if isinstance(model_name_or_path, str) and not args and not kwargs
            else None
        )
        if key is not None and key in shared:
            self.__dict__.update(shared[key].__dict__)
            return
        original_init(self, model_name_or_path, *args, **kwargs)
        if key is not None:
            shared[key] = self

    cls._apollo_shared_model = True  # type: ignore[attr-defined]
    cls.__init__ = cached_init  # type: ignore[method-assign]


_install_shared_embedding_model()


@pytest.fixture(autouse=True)
def fast_external_timeouts():
    """Fail fast against the absent Ollama/ComfyUI during tests.

    On this machine a connect to a closed localhost port is refused only after
    ~2-4 s (delayed RST), so every real-client probe in the suite pays that delay
    on top of its work. The tests assert on outcomes — unreachable services
    surface as errors, 503s or fallbacks — not on how long an attempt took, and
    none of them read these settings directly (the one test that needs specific
    values sets them itself via ``set_setting``). Originals are restored after
    each test.
    """
    from app.core import config

    overrides = {
        "request_timeout": 0.2,
        "vision_timeout": 0.2,
        "comfyui_timeout": 0.2,
        "comfyui_generation_timeout": 0.5,
        "comfyui_poll_interval": 0.05,
    }
    originals = {name: getattr(config.settings, name) for name in overrides}

    for name, value in overrides.items():
        object.__setattr__(config.settings, name, value)
    yield
    for name, value in originals.items():
        object.__setattr__(config.settings, name, value)


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
        "data_dir": str(tmp_path / "data"),
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


@pytest.fixture
def project(project_service):
    """A ready-made LoRA project for dataset/caption/training tests."""
    return project_service.create_project(
        "Test Character", description="demo", arch="sdxl", trigger_word="p3r5on"
    )


def png_bytes(size: int = 64) -> bytes:
    """A minimal but genuinely valid 1x1 PNG, padded to ``size`` bytes."""
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000a49444154789c6360000002000100ffff0300000600055773d8b400"
        "00000049454e44ae426082"
    )
    return png + b"\x00" * max(0, size - len(png))
