from fastapi.testclient import TestClient

from app.main import create_app
from app.core.exceptions import OllamaServiceError


class StubService:
    def health_check(self) -> dict:
        return {"status": "ok", "ollama_reachable": True, "model": "llama3.2"}

    def chat(self, prompt: str, model: str | None = None) -> str:
        return f"echo:{prompt}"

    def list_models(self) -> list[str]:
        return ["llama3.2", "mistral"]


def test_health_endpoint_reports_status() -> None:
    app = create_app(service=StubService())
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_endpoint_returns_response() -> None:
    app = create_app(service=StubService())
    client = TestClient(app)

    response = client.post("/chat", json={"prompt": "hello"})

    assert response.status_code == 200
    # /chat assembles the system prompt, RAG context and history around the
    # user's message, so the stub echoes the full assembled prompt.
    assert "USER QUESTION:\nhello" in response.json()["response"]


def test_models_endpoint_lists_models() -> None:
    app = create_app(service=StubService())
    client = TestClient(app)

    response = client.get("/models")

    assert response.status_code == 200
    assert response.json()["models"] == ["llama3.2", "mistral"]


def test_service_errors_are_mapped_to_http_errors() -> None:
    class FailingService:
        def health_check(self) -> dict:
            raise OllamaServiceError("Ollama is unreachable", status_code=503)

        def chat(self, prompt: str, model: str | None = None) -> str:
            raise OllamaServiceError("model not found", status_code=404)

        def list_models(self) -> list[str]:
            raise OllamaServiceError("Ollama is unreachable", status_code=503)

    app = create_app(service=FailingService())
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 503

    response = client.post("/chat", json={"prompt": "hello"})
    assert response.status_code == 404
