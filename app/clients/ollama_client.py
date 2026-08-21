from __future__ import annotations

import requests

from app.core.config import settings
from app.core.exceptions import OllamaServiceError


class OllamaClient:
    """Client for talking to Ollama HTTP API using OpenAI-compatible endpoints."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout or settings.request_timeout

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        timeout: float | None = None,
    ) -> dict:
        url = f"{self.base_url}{path}"
        try:
            response = requests.request(
                method=method, url=url, json=json, timeout=timeout or self.timeout
            )
        except requests.RequestException as exc:
            raise OllamaServiceError("Ollama is unreachable", status_code=503, detail=str(exc)) from exc

        # Convert all client/server errors from Ollama to 502 Bad Gateway
        # Never return 4xx status codes as that confuses clients about endpoint availability
        if response.status_code == 404:
            raise OllamaServiceError(
                f"Ollama endpoint not found: {path}",
                status_code=502,
                detail=f"The Ollama API endpoint {path} does not exist. Check Ollama version compatibility."
            )
        if response.status_code >= 500:
            raise OllamaServiceError("Ollama returned a server error", status_code=502)
        if response.status_code >= 400:
            raise OllamaServiceError(f"Ollama request failed with status {response.status_code}", status_code=502)

        try:
            return response.json()
        except ValueError as exc:
            raise OllamaServiceError("Ollama returned an invalid response", status_code=502) from exc

    def chat_completion(self, *, model: str, messages: list[dict]) -> dict:
        """Call OpenAI-compatible chat completion endpoint (Ollama v0.31+)."""
        return self._request("POST", "/v1/chat/completions", json={
            "model": model,
            "messages": messages,
            "stream": False
        })

    def chat(self, *, model: str, messages: list[dict]) -> dict:
        """Alias for chat_completion for compatibility."""
        return self.chat_completion(model=model, messages=messages)

    def generate(self, *, model: str, prompt: str) -> dict:
        """Legacy generate endpoint - converts to chat completion format."""
        # Use chat completion with system role for simple prompt completion
        messages = [{"role": "user", "content": prompt}]
        response = self.chat_completion(model=model, messages=messages)
        
        # Convert OpenAI response format to Ollama generate format for compatibility
        if "choices" in response and response["choices"]:
            return {
                "response": response["choices"][0]["message"]["content"],
                "model": model,
                "done": True
            }
        raise OllamaServiceError("Invalid response from chat completion", status_code=502)

    def list_models(self) -> dict:
        """List available models using OpenAI-compatible endpoint."""
        return self._request("GET", "/v1/models")

    def generate_with_images(
        self,
        *,
        model: str,
        prompt: str,
        images: list[str],
        timeout: float | None = None,
    ) -> str:
        """Prompt a vision model with base64 images via Ollama's /api/generate.

        Used for LoRA dataset captioning. Kept on the native endpoint because
        it takes bare base64, with no data-URI wrapping.
        """
        payload = self._request(
            "POST",
            "/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": images,
                "stream": False,
            },
            timeout=timeout,
        )
        response = payload.get("response")
        if not isinstance(response, str):
            raise OllamaServiceError(
                "Ollama returned an invalid vision response", status_code=502
            )
        return response.strip()

    def embed(self, *, model: str, inputs: list[str]) -> list[list[float]]:
        """Generate embeddings via Ollama's /api/embed endpoint."""
        payload = self._request(
            "POST",
            "/api/embed",
            json={"model": model, "input": inputs},
        )

        embeddings = payload.get("embeddings")
        if isinstance(embeddings, list) and len(embeddings) == len(inputs):
            return embeddings

        # Legacy fallback for older Ollama versions (single prompt per request).
        if len(inputs) == 1:
            legacy = self._request(
                "POST",
                "/api/embeddings",
                json={"model": model, "prompt": inputs[0]},
            )
            vector = legacy.get("embedding")
            if isinstance(vector, list):
                return [vector]

        raise OllamaServiceError("Ollama returned an invalid embedding response", status_code=502)
