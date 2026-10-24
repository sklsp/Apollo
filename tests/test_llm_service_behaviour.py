"""Unit tests for ``app.services.llm_service`` (service layer over Ollama).

The service is exercised with a fake ``OllamaClient`` injected through the
constructor, so no Ollama server or network access is needed. Covers client
injection, every branch of ``health_check``, ``chat`` and ``list_models``
including their failure paths.
"""

from __future__ import annotations

import pytest

from app.core.exceptions import OllamaServiceError
from app.services.llm_service import LLMService


class FakeLLMClient:
    """Minimal stand-in for ``OllamaClient`` with canned responses."""

    def __init__(self, chat_response=None, models_payload=None):
        self.chat_calls = []
        self.models_calls = 0
        self._chat_response = chat_response if chat_response is not None else {}
        self._models_payload = models_payload if models_payload is not None else {}

    def chat_completion(self, model, messages):
        self.chat_calls.append({"model": model, "messages": messages})
        return self._chat_response

    def list_models(self):
        self.models_calls += 1
        return self._models_payload


def make_service(monkeypatch, client=None, default_model="test-model"):
    """Build an ``LLMService`` with a stubbed settings object."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        "app.services.llm_service.settings", SimpleNamespace(default_model=default_model)
    )
    return LLMService(client=client)


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #


def test_init_uses_provided_client():
    client = FakeLLMClient()

    service = LLMService(client=client)

    assert service.client is client


def test_init_creates_default_client_when_none(monkeypatch):
    created = []

    class SentinelClient:
        def __init__(self, *args, **kwargs):
            created.append((args, kwargs))

    monkeypatch.setattr("app.services.llm_service.OllamaClient", SentinelClient)

    service = LLMService()

    assert isinstance(service.client, SentinelClient)
    assert len(created) == 1


# --------------------------------------------------------------------------- #
# health_check
# --------------------------------------------------------------------------- #


def test_health_check_reports_ok_with_models(monkeypatch):
    client = FakeLLMClient(models_payload={"data": [{"id": "llama3.2"}, {"id": "llava"}]})

    service = make_service(monkeypatch, client)
    result = service.health_check()

    assert result == {
        "status": "ok",
        "ollama_reachable": True,
        "model": "test-model",
        "available_models": ["llama3.2", "llava"],
    }
    assert client.models_calls == 1


def test_health_check_propagates_service_error(monkeypatch):
    class UnreachableClient(FakeLLMClient):
        def list_models(self):
            raise OllamaServiceError("Ollama is unreachable", status_code=503)

    service = make_service(monkeypatch, UnreachableClient())

    with pytest.raises(OllamaServiceError, match="Ollama is unreachable"):
        service.health_check()


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #


def test_chat_returns_content_with_default_model(monkeypatch):
    client = FakeLLMClient(
        chat_response={"choices": [{"message": {"content": "hello back"}}]}
    )

    service = make_service(monkeypatch, client)
    reply = service.chat("hi there")

    assert reply == "hello back"
    (call,) = client.chat_calls
    assert call["model"] == "test-model"
    assert call["messages"] == [{"role": "user", "content": "hi there"}]


def test_chat_uses_explicit_model(monkeypatch):
    client = FakeLLMClient(
        chat_response={"choices": [{"message": {"content": "ok"}}]}
    )

    service = make_service(monkeypatch, client)
    reply = service.chat("hi", model="llava")

    assert reply == "ok"
    (call,) = client.chat_calls
    assert call["model"] == "llava"


def test_chat_missing_choices_raises(monkeypatch):
    client = FakeLLMClient(chat_response={"foo": "bar"})

    service = make_service(monkeypatch, client)

    with pytest.raises(OllamaServiceError, match="Ollama returned an invalid chat response"):
        service.chat("hi")


def test_chat_empty_choices_raises(monkeypatch):
    client = FakeLLMClient(chat_response={"choices": []})

    service = make_service(monkeypatch, client)

    with pytest.raises(OllamaServiceError, match="Ollama returned an invalid chat response"):
        service.chat("hi")


def test_chat_non_string_content_raises(monkeypatch):
    client = FakeLLMClient(chat_response={"choices": [{"message": {"content": None}}]})

    service = make_service(monkeypatch, client)

    with pytest.raises(OllamaServiceError, match="Ollama returned an invalid chat response"):
        service.chat("hi")


def test_chat_missing_message_key_raises(monkeypatch):
    client = FakeLLMClient(chat_response={"choices": [{"index": 0}]})

    service = make_service(monkeypatch, client)

    with pytest.raises(OllamaServiceError, match="Ollama returned an invalid chat response"):
        service.chat("hi")


# --------------------------------------------------------------------------- #
# list_models
# --------------------------------------------------------------------------- #


def test_list_models_openai_format_filters_entries(monkeypatch):
    payload = {
        "data": [
            {"id": "llama3.2"},
            "not-a-dict",
            {"no_id": True},
            {"id": "llava:7b"},
        ]
    }
    client = FakeLLMClient(models_payload=payload)

    service = make_service(monkeypatch, client)
    models = service.list_models()

    assert models == ["llama3.2", "llava:7b"]


def test_list_models_legacy_format_filters_entries(monkeypatch):
    payload = {
        "models": [
            {"name": "mistral"},
            42,
            {"other": True},
            {"name": "nomic-embed-text"},
        ]
    }
    client = FakeLLMClient(models_payload=payload)

    service = make_service(monkeypatch, client)
    models = service.list_models()

    assert models == ["mistral", "nomic-embed-text"]


def test_list_models_invalid_payload_raises(monkeypatch):
    client = FakeLLMClient(models_payload={"unexpected": True})

    service = make_service(monkeypatch, client)

    with pytest.raises(OllamaServiceError, match="Ollama returned an invalid model list"):
        service.list_models()


def test_list_models_non_list_data_falls_through_to_error(monkeypatch):
    client = FakeLLMClient(models_payload={"data": "not-a-list"})

    service = make_service(monkeypatch, client)

    with pytest.raises(OllamaServiceError, match="Ollama returned an invalid model list"):
        service.list_models()
