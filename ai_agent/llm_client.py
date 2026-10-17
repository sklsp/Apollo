from __future__ import annotations

from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - exercised in minimal environments
    requests = None  # type: ignore[assignment]

from .config import OLLAMA_BASE_URL

DEFAULT_MODEL = "llama3.2:3b"


def _format_url(base_url: str, endpoint: str) -> str:
    return f"{base_url.rstrip('/')}{endpoint}"


def _build_payload(prompt: str, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "stream": False
    }


def _parse_response(resp: Any) -> str:
    try:
        data = resp.json()
    except Exception:
        return resp.text

    # Ollama /api/generate response
    if isinstance(data, dict):
        if "response" in data:
            return data["response"]

        if "message" in data and isinstance(data["message"], dict):
            return data["message"].get("content", "")

    return str(data)


def ask_llm(
    prompt: str,
    model: str = DEFAULT_MODEL,
    base_url: str | None = None
) -> str:

    if requests is None:
        return "❌ requests package is not installed"

    base_url = base_url or OLLAMA_BASE_URL
    url = _format_url(base_url, "/api/generate")

    try:
        resp = requests.post(
            url,
            json=_build_payload(prompt, model),
            timeout=60
        )

        resp.raise_for_status()
        return _parse_response(resp)

    except requests.exceptions.ConnectionError:
        return f"❌ Cannot connect to Ollama at {base_url}"

    except requests.exceptions.HTTPError as e:
        return f"❌ HTTP error from Ollama: {e} | Response: {getattr(e.response, 'text', '')}"

    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"


def list_models(base_url: str | None = None) -> dict:
    base_url = base_url or OLLAMA_BASE_URL

    endpoints = ["/models", "/api/models", "/models/list", "/v1/models"]
    results = {}
    for endpoint in endpoints:
        url = _format_url(base_url, endpoint)
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            try:
                results[url] = resp.json()
            except Exception:
                results[url] = resp.text
        except Exception as exc:
            results[url] = str(exc)
    return results
