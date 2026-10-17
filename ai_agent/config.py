import os
from pathlib import Path

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")

    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[CONFIG ERROR] Failed to load .env: {e}")

    return values


def _normalize_base_url(url: str) -> str:
    url = url.strip()

    if not url:
        return DEFAULT_OLLAMA_BASE_URL

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    return url.rstrip("/")


def get_ollama_base_url() -> str:
    env_value = os.environ.get("OLLAMA_BASE_URL")

    if env_value:
        return _normalize_base_url(env_value)

    dotenv_values = _load_dotenv(Path(__file__).resolve().parent / ".env")

    if "OLLAMA_BASE_URL" in dotenv_values:
        return _normalize_base_url(dotenv_values["OLLAMA_BASE_URL"])

    return DEFAULT_OLLAMA_BASE_URL


OLLAMA_BASE_URL = get_ollama_base_url()
