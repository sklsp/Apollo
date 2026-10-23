"""Unit tests for ``app.clients.ollama_client`` (HTTP layer).

The client is exercised with a fake ``requests.request`` injected via
monkeypatch, so no Ollama server or network access is needed. Covers the
request plumbing (URL building, timeout), every error branch of
``_request``, each public method including its failure paths, and the
module-level ``_extract_ollama_error`` helper.
"""

from __future__ import annotations

import json

import pytest

from app.clients.ollama_client import OllamaClient, _extract_ollama_error
from app.core.exceptions import OllamaServiceError


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self.text = text
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


def make_client(monkeypatch, captured, response=None, base_url="http://localhost:11434", timeout=30.0):
    """Build an ``OllamaClient`` whose HTTP calls land in ``captured``."""

    def fake_request(*args, **kwargs):
        captured.append((args, kwargs))
        return response if response is not None else FakeResponse()

    monkeypatch.setattr("app.clients.ollama_client.requests.request", fake_request)
    return OllamaClient(base_url=base_url, timeout=timeout)


# --------------------------------------------------------------------------- #
# _request plumbing and error branches
# --------------------------------------------------------------------------- #


def test_request_success_builds_url_and_default_timeout(monkeypatch):
    captured = []
    client = make_client(
        monkeypatch, captured, FakeResponse(200, payload={"data": [{"id": "llama3.2"}]})
    )

    result = client._request("GET", "/v1/models")

    assert result == {"data": [{"id": "llama3.2"}]}
    (args, kwargs), = captured
    assert args == ()
    assert kwargs["method"] == "GET"
    assert kwargs["url"] == "http://localhost:11434/v1/models"
    assert kwargs["json"] is None
    assert kwargs["timeout"] == 30.0


def test_request_explicit_timeout_overrides_default(monkeypatch):
    captured = []
    client = make_client(monkeypatch, captured)

    client._request("POST", "/api/generate", json={"model": "llava"}, timeout=120)

    (args, kwargs), = captured
    assert kwargs["timeout"] == 120


def test_request_unreachable_raises_503(monkeypatch):
    import requests

    def boom(*args, **kwargs):
        raise requests.exceptions.ConnectionError("connection refused")

    monkeypatch.setattr("app.clients.ollama_client.requests.request", boom)
    client = OllamaClient()

    with pytest.raises(OllamaServiceError, match="Ollama is unreachable") as excinfo:
        client._request("GET", "/v1/models")
    assert excinfo.value.status_code == 503
    assert "connection refused" in excinfo.value.detail


def test_request_timeout_exception_is_unreachable(monkeypatch):
    import requests

    def boom(*args, **kwargs):
        raise requests.exceptions.Timeout("timed out")

    monkeypatch.setattr("app.clients.ollama_client.requests.request", boom)
    client = OllamaClient()

    with pytest.raises(OllamaServiceError, match="Ollama is unreachable"):
        client._request("GET", "/v1/models")


def test_request_404_model_not_found(monkeypatch):
    monkeypatch.setattr(
        "app.clients.ollama_client.requests.request",
        lambda *a, **kw: FakeResponse(
            404, payload={"error": "model 'llava' not found, try pulling it with 'ollama pull llava'"}
        ),
    )
    client = OllamaClient()

    with pytest.raises(OllamaServiceError, match="Ollama model not available: /v1/models") as excinfo:
        client._request("GET", "/v1/models")
    assert excinfo.value.status_code == 502


def test_request_404_unknown_endpoint(monkeypatch):
    monkeypatch.setattr(
        "app.clients.ollama_client.requests.request",
        lambda *a, **kw: FakeResponse(404, payload={"error": "not found"}),
    )
    client = OllamaClient()

    with pytest.raises(OllamaServiceError, match="Ollama endpoint not found: /v1/models") as excinfo:
        client._request("GET", "/v1/models")
    assert excinfo.value.status_code == 502


def test_request_server_error_raises_502(monkeypatch):
    monkeypatch.setattr(
        "app.clients.ollama_client.requests.request", lambda *a, **kw: FakeResponse(500)
    )
    client = OllamaClient()

    with pytest.raises(OllamaServiceError, match="Ollama returned a server error") as excinfo:
        client._request("GET", "/v1/models")
    assert excinfo.value.status_code == 502


def test_request_client_error_includes_status(monkeypatch):
    monkeypatch.setattr(
        "app.clients.ollama_client.requests.request", lambda *a, **kw: FakeResponse(413)
    )
    client = OllamaClient()

    with pytest.raises(OllamaServiceError, match="Ollama request failed with status 413"):
        client._request("POST", "/api/generate")


def test_request_invalid_json_raises(monkeypatch):
    class BadResponse:
        status_code = 200

        def json(self):
            raise json.JSONDecodeError("bad", "", 0)

    monkeypatch.setattr(
        "app.clients.ollama_client.requests.request", lambda *a, **kw: BadResponse()
    )
    client = OllamaClient()

    with pytest.raises(OllamaServiceError, match="Ollama returned an invalid response"):
        client._request("GET", "/v1/models")


# --------------------------------------------------------------------------- #
# Public methods
# --------------------------------------------------------------------------- #


def test_chat_completion_posts_openai_payload(monkeypatch):
    captured = []
    client = make_client(
        monkeypatch,
        captured,
        FakeResponse(200, payload={"choices": [{"message": {"content": "hi"}}]}),
    )

    result = client.chat_completion(model="llama3.2", messages=[{"role": "user", "content": "hello"}])

    assert result["choices"][0]["message"]["content"] == "hi"
    (args, kwargs), = captured
    assert kwargs["url"].endswith("/v1/chat/completions")
    assert kwargs["json"] == {
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }


def test_chat_is_alias_for_chat_completion(monkeypatch):
    captured = []
    client = make_client(
        monkeypatch,
        captured,
        FakeResponse(200, payload={"choices": [{"message": {"content": "hi"}}]}),
    )

    result = client.chat(model="llama3.2", messages=[{"role": "user", "content": "hello"}])

    assert result["choices"][0]["message"]["content"] == "hi"
    (args, kwargs), = captured
    assert kwargs["url"].endswith("/v1/chat/completions")


def test_generate_converts_chat_response(monkeypatch):
    captured = []
    client = make_client(
        monkeypatch,
        captured,
        FakeResponse(200, payload={"choices": [{"message": {"content": "answer"}}]}),
    )

    result = client.generate(model="llama3.2", prompt="say something")

    assert result == {"response": "answer", "model": "llama3.2", "done": True}
    (args, kwargs), = captured
    assert kwargs["json"]["messages"] == [{"role": "user", "content": "say something"}]


def test_generate_without_choices_raises(monkeypatch):
    monkeypatch.setattr(
        "app.clients.ollama_client.requests.request", lambda *a, **kw: FakeResponse(200, payload={})
    )
    client = OllamaClient()

    with pytest.raises(OllamaServiceError, match="Invalid response from chat completion"):
        client.generate(model="llama3.2", prompt="say something")


def test_list_models_uses_get(monkeypatch):
    captured = []
    client = make_client(
        monkeypatch, captured, FakeResponse(200, payload={"data": [{"id": "llama3.2"}]})
    )

    result = client.list_models()

    assert result["data"] == [{"id": "llama3.2"}]
    (args, kwargs), = captured
    assert kwargs["method"] == "GET"
    assert kwargs["url"].endswith("/v1/models")


def test_generate_with_images_returns_stripped_text(monkeypatch):
    captured = []
    client = make_client(
        monkeypatch,
        captured,
        FakeResponse(200, payload={"response": "  a cat on a keyboard \n"}),
    )

    caption = client.generate_with_images(
        model="llava", prompt="describe this image", images=["aGVsbG8="], timeout=120
    )

    assert caption == "a cat on a keyboard"
    (args, kwargs), = captured
    assert kwargs["url"].endswith("/api/generate")
    assert kwargs["json"] == {
        "model": "llava",
        "prompt": "describe this image",
        "images": ["aGVsbG8="],
        "stream": False,
    }
    assert kwargs["timeout"] == 120


def test_generate_with_images_invalid_response_raises(monkeypatch):
    monkeypatch.setattr(
        "app.clients.ollama_client.requests.request",
        lambda *a, **kw: FakeResponse(200, payload={"response": 42}),
    )
    client = OllamaClient()

    with pytest.raises(OllamaServiceError, match="Ollama returned an invalid vision response"):
        client.generate_with_images(model="llava", prompt="describe", images=["aGVsbG8="])


def test_embed_returns_embeddings_list(monkeypatch):
    captured = []
    client = make_client(
        monkeypatch,
        captured,
        FakeResponse(200, payload={"embeddings": [[0.1, 0.2], [0.3, 0.4]]}),
    )

    vectors = client.embed(model="nomic-embed-text", inputs=["alpha", "beta"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    (args, kwargs), = captured
    assert kwargs["url"].endswith("/api/embed")
    assert kwargs["json"] == {"model": "nomic-embed-text", "input": ["alpha", "beta"]}


def test_embed_legacy_fallback_for_single_input(monkeypatch):
    captured = []
    responses = iter(
        [
            FakeResponse(200, payload={"embeddings": []}),
            FakeResponse(200, payload={"embedding": [0.5, 0.6]}),
        ]
    )

    def fake_request(*args, **kwargs):
        captured.append((args, kwargs))
        return next(responses)

    monkeypatch.setattr("app.clients.ollama_client.requests.request", fake_request)
    client = OllamaClient(base_url="http://localhost:11434", timeout=30.0)

    vectors = client.embed(model="nomic-embed-text", inputs=["only one"])

    assert vectors == [[0.5, 0.6]]
    assert len(captured) == 2
    first_kwargs = captured[0][1]
    second_kwargs = captured[1][1]
    assert first_kwargs["url"].endswith("/api/embed")
    assert second_kwargs["url"].endswith("/api/embeddings")
    assert second_kwargs["json"] == {"model": "nomic-embed-text", "prompt": "only one"}


def test_embed_invalid_response_raises(monkeypatch):
    monkeypatch.setattr(
        "app.clients.ollama_client.requests.request", lambda *a, **kw: FakeResponse(200, payload={})
    )
    client = OllamaClient()

    with pytest.raises(OllamaServiceError, match="Ollama returned an invalid embedding response"):
        client.embed(model="nomic-embed-text", inputs=["only one"])


# --------------------------------------------------------------------------- #
# _extract_ollama_error helper
# --------------------------------------------------------------------------- #


def test_extract_ollama_error_prefers_error_field():
    response = FakeResponse(404, payload={"error": "model 'x' not found"}, text="raw body")
    assert _extract_ollama_error(response) == "model 'x' not found"


def test_extract_ollama_error_truncates_long_errors():
    response = FakeResponse(404, payload={"error": "x" * 500}, text="raw body")
    assert _extract_ollama_error(response) == "x" * 300


def test_extract_ollama_error_falls_back_to_text_on_bad_json():
    class BadResponse:
        status_code = 404
        text = "plain text error"

        def json(self):
            raise json.JSONDecodeError("bad", "", 0)

    assert _extract_ollama_error(BadResponse()) == "plain text error"


def test_extract_ollama_error_no_error_key_returns_text():
    response = FakeResponse(404, payload={"foo": "bar"}, text="raw body")
    assert _extract_ollama_error(response) == "raw body"


