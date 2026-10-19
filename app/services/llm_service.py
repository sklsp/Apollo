from __future__ import annotations

from app.clients.ollama_client import OllamaClient
from app.core.config import settings
from app.core.exceptions import OllamaServiceError


class LLMService:
    """Service layer for Ollama-backed chat and model listing."""

    def __init__(self, client: OllamaClient | None = None) -> None:
        self.client = client or OllamaClient()

    def health_check(self) -> dict:
        try:
            models = self.list_models()
        except OllamaServiceError:
            raise
        return {
            "status": "ok",
            "ollama_reachable": True,
            "model": settings.default_model,
            "available_models": models,
        }

    def chat(self, prompt: str, model: str | None = None) -> str:
        """Send a prompt to Ollama and get a response.

        Uses the OpenAI-compatible /v1/chat/completions endpoint
        to get a text completion for the provided prompt.
        """
        selected_model = model or settings.default_model

        # Format prompt as a chat message for the /v1/chat/completions endpoint
        messages = [{"role": "user", "content": prompt}]

        response = self.client.chat_completion(model=selected_model, messages=messages)

        # Extract content from OpenAI-compatible response format
        if isinstance(response.get("choices"), list) and response["choices"]:
            content = response["choices"][0].get("message", {}).get("content")
            if isinstance(content, str):
                return content

        raise OllamaServiceError("Ollama returned an invalid chat response", status_code=502)

    def list_models(self) -> list[str]:
        """List available models from Ollama."""
        payload = self.client.list_models()

        # Handle OpenAI-compatible response format: {"data": [{"id": "model:tag"}, ...]}
        if "data" in payload and isinstance(payload["data"], list):
            return [model.get("id", "") for model in payload["data"] if isinstance(model, dict) and model.get("id")]

        # Fallback for legacy /api/tags format: {"models": [{"name": "model:tag"}, ...]}
        if "models" in payload and isinstance(payload["models"], list):
            return [model.get("name", "") for model in payload["models"] if isinstance(model, dict) and model.get("name")]

        raise OllamaServiceError("Ollama returned an invalid model list", status_code=502)
