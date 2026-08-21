"""HTTP client for a locally running ComfyUI instance.

Mirrors the shape of :mod:`app.clients.ollama_client`: all transport concerns,
timeouts and error translation live here so services never touch ``requests``.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import requests

from app.core.config import settings
from app.core.exceptions import ComfyUIServiceError

logger = logging.getLogger(__name__)


class ComfyUIClient:
    """Talks to ComfyUI's HTTP API (``/prompt``, ``/history``, ``/view``, ...)."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None) -> None:
        self.base_url = (base_url or settings.comfyui_base_url).rstrip("/")
        self.timeout = timeout or settings.comfyui_timeout
        # Stable per-process id so ComfyUI groups our submissions together.
        self.client_id = uuid.uuid4().hex

    # ---------- transport ----------

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        params: dict | None = None,
        files: dict | None = None,
        data: dict | None = None,
        raw: bool = False,
        timeout: float | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = requests.request(
                method,
                url,
                json=json,
                params=params,
                files=files,
                data=data,
                timeout=timeout or self.timeout,
            )
        except requests.Timeout as exc:
            raise ComfyUIServiceError(
                f"ComfyUI timed out after {timeout or self.timeout:.0f}s",
                status_code=504,
                detail=str(exc),
            ) from exc
        except requests.RequestException as exc:
            raise ComfyUIServiceError(
                f"ComfyUI is not reachable at {self.base_url}",
                status_code=503,
                detail=str(exc),
            ) from exc

        if response.status_code >= 400:
            # ComfyUI reports workflow validation problems as 400 with a JSON body.
            detail = _extract_error(response)
            raise ComfyUIServiceError(
                f"ComfyUI rejected the request ({response.status_code})",
                status_code=502,
                detail=detail,
            )

        if raw:
            return response.content
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise ComfyUIServiceError(
                "ComfyUI returned a malformed response",
                status_code=502,
                detail=str(exc),
            ) from exc

    # ---------- api ----------

    def ping(self) -> dict:
        """Return ComfyUI's system stats. Raises if unreachable."""
        stats = self._request("GET", "/system_stats", timeout=min(self.timeout, 5))
        if not isinstance(stats, dict):
            raise ComfyUIServiceError("ComfyUI returned unexpected system stats")
        return stats

    def object_info(self, node_class: str | None = None) -> dict:
        """Node schemas. Used to discover installed models and validate workflows."""
        path = f"/object_info/{node_class}" if node_class else "/object_info"
        result = self._request("GET", path)
        return result if isinstance(result, dict) else {}

    def queue_prompt(self, workflow: dict) -> str:
        """Submit an API-format workflow; returns ComfyUI's prompt_id."""
        payload = self._request(
            "POST", "/prompt", json={"prompt": workflow, "client_id": self.client_id}
        )
        prompt_id = payload.get("prompt_id") if isinstance(payload, dict) else None
        if not prompt_id:
            node_errors = (payload or {}).get("node_errors") or {}
            raise ComfyUIServiceError(
                "ComfyUI did not accept the workflow",
                status_code=502,
                detail=str(node_errors) or str(payload),
            )
        logger.info("[COMFYUI] Queued prompt %s", prompt_id)
        return str(prompt_id)

    def history(self, prompt_id: str) -> dict:
        """Execution record for one prompt; empty dict while still queued."""
        result = self._request("GET", f"/history/{prompt_id}")
        if isinstance(result, dict):
            return result.get(prompt_id, {}) or {}
        return {}

    def queue_state(self) -> dict:
        """Current running/pending queue. Read-only — never mutate the queue."""
        result = self._request("GET", "/queue")
        return result if isinstance(result, dict) else {}

    def view_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Download one generated image's bytes."""
        return self._request(
            "GET",
            "/view",
            params={"filename": filename, "subfolder": subfolder, "type": folder_type},
            raw=True,
        )

    def upload_image(self, filename: str, content: bytes, overwrite: bool = True) -> dict:
        """Upload an image into ComfyUI's input folder (for img2img workflows)."""
        result = self._request(
            "POST",
            "/upload/image",
            files={"image": (filename, content)},
            data={"overwrite": str(overwrite).lower()},
        )
        return result if isinstance(result, dict) else {}

    # ---------- model discovery ----------

    def _combo_options(self, node_class: str, input_name: str) -> list[str]:
        """Pull a node's dropdown options out of /object_info."""
        info = self.object_info(node_class).get(node_class, {})
        required = info.get("input", {}).get("required", {})
        spec = required.get(input_name)
        if isinstance(spec, list) and spec and isinstance(spec[0], list):
            return [str(option) for option in spec[0]]
        return []

    def list_loras(self) -> list[str]:
        return self._combo_options("LoraLoader", "lora_name")

    def list_checkpoints(self) -> list[str]:
        return self._combo_options("CheckpointLoaderSimple", "ckpt_name")


def _extract_error(response: requests.Response) -> str:
    """Best-effort human-readable reason out of a ComfyUI error response."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
        if body.get("node_errors"):
            return str(body["node_errors"])[:500]
    return str(body)[:500]
