import os
from dataclasses import dataclass
from pathlib import Path

# Project root = two levels up from app/core/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load .env before any getenv call below. The README has always documented a
# .env file, but nothing actually read it; this makes that real. Real
# environment variables still win over the file.
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:  # python-dotenv is optional; env vars still work without it
    pass


def _resolve(value: str) -> str:
    """Resolve a possibly-relative configured path against the project root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    default_model: str = os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2")
    embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    embedding_fallback_model: str = os.getenv(
        "EMBEDDING_FALLBACK_MODEL", "all-MiniLM-L6-v2"
    )
    request_timeout: float = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "30"))
    rag_chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    rag_chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "4"))

    # ---- Ollama vision (LoRA caption generation) ----
    vision_model: str = os.getenv("OLLAMA_VISION_MODEL", "llava")
    vision_timeout: float = float(os.getenv("OLLAMA_VISION_TIMEOUT", "180"))

    # ---- ComfyUI ----
    comfyui_base_url: str = os.getenv("COMFYUI_BASE_URL", "http://127.0.0.1:8188")
    comfyui_timeout: float = float(os.getenv("COMFYUI_TIMEOUT", "30"))
    comfyui_poll_interval: float = float(os.getenv("COMFYUI_POLL_INTERVAL", "1.0"))
    comfyui_generation_timeout: float = float(
        os.getenv("COMFYUI_GENERATION_TIMEOUT", "600")
    )
    comfyui_workflow_dir: str = _resolve(os.getenv("COMFYUI_WORKFLOW_DIR", "./workflows"))
    # ComfyUI's own models/loras folder, so manually added LoRAs are discoverable.
    comfyui_lora_dir: str = os.getenv("COMFYUI_LORA_DIR", "")

    # ---- Ostris AI Toolkit (external local process) ----
    ai_toolkit_path: str = os.getenv("AI_TOOLKIT_PATH", "")
    ai_toolkit_python: str = os.getenv("AI_TOOLKIT_PYTHON", "")

    # ---- Local storage roots ----
    lora_data_dir: str = _resolve(os.getenv("LORA_DATA_DIR", "./data/loras"))
    generated_dir: str = _resolve(os.getenv("GENERATED_DIR", "./data/generated"))
    jobs_file: str = _resolve(os.getenv("JOBS_FILE", "./data/jobs.json"))

    # ---- Upload limits ----
    max_image_upload_mb: float = float(os.getenv("MAX_IMAGE_UPLOAD_MB", "25"))

    @property
    def max_image_upload_bytes(self) -> int:
        return int(self.max_image_upload_mb * 1024 * 1024)

    @property
    def ai_toolkit_configured(self) -> bool:
        """True only when both the toolkit path and its interpreter really exist."""
        return bool(
            self.ai_toolkit_path
            and self.ai_toolkit_python
            and Path(self.ai_toolkit_path, "run.py").is_file()
            and Path(self.ai_toolkit_python).is_file()
        )


settings = Settings()
