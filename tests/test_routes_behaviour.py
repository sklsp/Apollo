"""Behaviour tests for the core API routes (app/api/routes.py).

Complements test_api_app.py (full-app smoke) and the per-service behaviour
files: these drive the router itself with a minimal app that wires only the
five route dependencies, so every handler runs against recording fakes. The
tests pin how each endpoint assembles responses, degrades when Ollama is down
(503 guidance card on /health, degraded payload on /ready), rolls back a
document whose RAG indexing failed, and maps service errors onto HTTP status
codes - without touching disk or the network.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.api.routes import _build_system_prompt, router, upload_document
from app.core.config import settings
from app.core.exceptions import OllamaServiceError


# ============================================
# FAKES (recording stand-ins for the five services)
# ============================================


class FakeLLM:
    """Stand-in for ``LLMService``; optional per-method errors, records chats."""

    def __init__(
        self,
        *,
        health_error: Exception | None = None,
        models_error: Exception | None = None,
        chat_error: Exception | None = None,
        base_url: str | None = "http://127.0.0.1:11435",
    ) -> None:
        self.health_error = health_error
        self.models_error = models_error
        self.chat_error = chat_error
        # ``None`` mirrors a service whose client was never built, so the
        # routes fall back to settings.ollama_base_url.
        self.client = SimpleNamespace(base_url=base_url) if base_url is not None else None
        self.chat_calls: list[tuple[str, str | None]] = []

    def health_check(self) -> dict:
        if self.health_error:
            raise self.health_error
        return {"status": "ok", "ollama_reachable": True, "model": "llama3.2"}

    def chat(self, prompt: str, model: str | None = None) -> str:
        if self.chat_error:
            raise self.chat_error
        self.chat_calls.append((prompt, model))
        return "REPLY"

    def list_models(self) -> list[str]:
        if self.models_error:
            raise self.models_error
        return ["llama3.2", "mistral"]


class FakeMemory:
    """Stand-in for ``MemoryService``; records every mutation."""

    def __init__(self, history_text: str = "HISTORY") -> None:
        self.history_text = history_text
        self.use_documents_calls: list[tuple[str, bool]] = []
        self.added: list[tuple[str, str, str]] = []
        self.cleared: list[str] = []

    def set_use_documents(self, session_id: str, flag: bool) -> None:
        self.use_documents_calls.append((session_id, flag))

    def format_history_for_prompt(self, session_id: str) -> str:
        return self.history_text

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self.added.append((session_id, role, content))

    def get_history(self, session_id: str) -> list[dict]:
        return [
            {"role": "user", "content": "hi", "timestamp": "2026-01-01T00:00:00"},
            {"role": "assistant", "content": "hello!", "timestamp": "2026-01-01T00:00:05"},
        ]

    def clear_history(self, session_id: str) -> None:
        self.cleared.append(session_id)


class FakeDocs:
    """Stand-in for ``DocumentService``; ids are sequential doc-N."""

    def __init__(self, docs=(), delete_result: bool = True) -> None:
        self.docs = list(docs)
        self.delete_result = delete_result
        self.stored: list[tuple[str, str]] = []
        self.deleted: list[str] = []
        self.cleared = 0

    def store_document(self, text: str, source: str) -> str:
        doc_id = f"doc-{len(self.stored) + 1}"
        self.stored.append((text, source))
        return doc_id

    def get_all_documents(self) -> list[dict]:
        return list(self.docs)

    def delete_document(self, doc_id: str) -> bool:
        self.deleted.append(doc_id)
        return self.delete_result

    def clear_all(self) -> None:
        self.cleared += 1


class FakePrompts:
    """Stand-in for ``PromptService``; in-memory store with fixed timestamps."""

    def __init__(
        self,
        prompts=(),
        default: dict | None = None,
        update_result: bool = True,
        delete_result: bool = True,
        get_after_create: bool = True,
    ) -> None:
        self.store = {p["id"]: dict(p) for p in prompts}
        self.default = default
        self.update_result = update_result
        self.delete_result = delete_result
        self.get_after_create = get_after_create
        self.created: list[tuple[str, str]] = []
        self.updated: list[tuple[str, str | None, str | None]] = []
        self.deleted: list[str] = []

    def create_prompt(self, name: str, content: str) -> str:
        prompt_id = f"p{len(self.created) + 1}"
        self.created.append((name, content))
        if self.get_after_create:
            self.store[prompt_id] = {
                "id": prompt_id,
                "name": name,
                "content": content,
                "created_at": "2026-01-02T03:04:05",
                "updated_at": "2026-01-02T03:04:05",
            }
        return prompt_id

    def get_prompt(self, prompt_id: str) -> dict | None:
        prompt = self.store.get(prompt_id)
        return dict(prompt) if prompt else None

    def get_prompt_by_name(self, name: str) -> dict | None:
        if self.default and name == "default":
            return dict(self.default)
        return None

    def apply_template(self, prompt_id: str, message: str) -> str:
        return f"[tpl:{prompt_id}] {message}"

    def update_prompt(
        self, prompt_id: str, name: str | None = None, content: str | None = None
    ) -> bool:
        self.updated.append((prompt_id, name, content))
        if prompt_id in self.store:
            if name is not None:
                self.store[prompt_id]["name"] = name
            if content is not None:
                self.store[prompt_id]["content"] = content
        return self.update_result

    def delete_prompt(self, prompt_id: str) -> bool:
        self.deleted.append(prompt_id)
        if prompt_id not in self.store:
            return False
        del self.store[prompt_id]
        return self.delete_result

    def list_prompts(self) -> list[dict]:
        return [dict(p) for p in self.store.values()]


class FakeRAG:
    """Stand-in for ``RAGService``; fixed chunks/context, records calls."""

    def __init__(
        self,
        chunks=("chunk-a", "chunk-b"),
        context: str = "CTX",
        add_error: Exception | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.context = context
        self.add_error = add_error
        self.query_calls: list[str] = []
        self.debug_calls: list[tuple[str, int]] = []
        self.added: list[tuple[str, dict]] = []
        self.removed: list[str] = []
        self.cleared = 0

    def status(self) -> dict:
        return {"model": "fake-embed", "documents": 2}

    def query(self, text: str) -> list[dict]:
        self.query_calls.append(text)
        return list(self.chunks)

    def format_context(self, chunks: list[dict]) -> str:
        return self.context if chunks else ""

    def add_document(self, text: str, meta: dict) -> int:
        if self.add_error:
            raise self.add_error
        self.added.append((text, dict(meta)))
        return 3

    def remove_document(self, doc_id: str) -> None:
        self.removed.append(doc_id)

    def clear(self) -> None:
        self.cleared += 1

    def query_debug(self, q: str, top_k: int = 4) -> dict:
        self.debug_calls.append((q, top_k))
        return {"query": q}


# ============================================
# FIXTURES & HELPERS
# ============================================


@pytest.fixture
def routes_env():
    """Minimal FastAPI app with only the core router and fake services."""
    llm = FakeLLM()
    memory = FakeMemory()
    docs = FakeDocs()
    prompts = FakePrompts()
    rag = FakeRAG()

    app = FastAPI()
    app.include_router(router)
    app.state.llm_service = llm
    app.state.memory_service = memory
    app.state.document_service = docs
    app.state.prompt_service = prompts
    app.state.rag_service = rag
    # /dashboard reads four more services off the app state.
    app.state.lora_project_service = SimpleNamespace(kind="lora_projects")
    app.state.lora_training_service = SimpleNamespace(kind="training")
    app.state.comfyui_service = SimpleNamespace(kind="comfyui")
    app.state.job_store = SimpleNamespace(kind="jobs")

    return SimpleNamespace(
        client=TestClient(app),
        llm=llm,
        memory=memory,
        docs=docs,
        prompts=prompts,
        rag=rag,
    )


def _chat(client: TestClient, **overrides) -> dict:
    payload = {"prompt": "hello", "session_id": "s1"}
    payload.update(overrides)
    response = client.post("/chat", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


DEFAULT_PROMPT = {
    "id": "default",
    "name": "default",
    "content": "Be terse.",
    "created_at": "2026-01-01T00:00:00",
    "updated_at": "2026-01-01T00:00:00",
}


# ============================================
# HEALTH & STATUS ENDPOINTS
# ============================================


def test_live_is_dependency_free(routes_env) -> None:
    response = routes_env.client.get("/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_ready_ok_when_ollama_answers(routes_env) -> None:
    response = routes_env.client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_degraded_reports_url_and_detail(routes_env) -> None:
    routes_env.llm.models_error = OllamaServiceError(
        "Ollama is unreachable", status_code=503, detail="connect ECONNREFUSED"
    )

    response = routes_env.client.get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "status": "degraded",
        "reason": "Ollama is unreachable",
        "url": "http://127.0.0.1:11435",
        "detail": "connect ECONNREFUSED",
    }


def test_ready_falls_back_to_settings_url_without_client(routes_env) -> None:
    routes_env.llm.client = None
    routes_env.llm.models_error = OllamaServiceError(
        "Ollama is unreachable", status_code=503, detail="down"
    )

    response = routes_env.client.get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"]["url"] == settings.ollama_base_url


def test_health_ok_returns_payload(routes_env) -> None:
    response = routes_env.client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "ollama_reachable": True,
        "model": "llama3.2",
        "available_models": None,
    }


def test_health_unreachable_returns_setup_card(routes_env) -> None:
    routes_env.llm.health_error = OllamaServiceError(
        "Ollama is unreachable", status_code=503, detail="raw probe failure"
    )

    response = routes_env.client.get("/health")

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["message"] == "Ollama is not reachable."
    assert detail["url_checked"] == "http://127.0.0.1:11435"
    assert detail["expected_model"] == settings.default_model
    assert detail["how_to_fix"][1] == (
        "Start it with: ollama serve (listening on http://127.0.0.1:11435)"
    )
    assert detail["unavailable_features"] == [
        "Chat",
        "RAG document Q&A",
        "AI dataset captioning",
    ]
    assert detail["working_features"] == [
        "ComfyUI image generation",
        "Dataset management",
        "Dataset validation",
        "LoRA training",
    ]
    assert detail["raw_detail"] == "raw probe failure"


def test_models_ok(routes_env) -> None:
    response = routes_env.client.get("/models")

    assert response.status_code == 200
    assert response.json() == {"models": ["llama3.2", "mistral"]}


def test_models_maps_service_error_status_and_detail(routes_env) -> None:
    routes_env.llm.models_error = OllamaServiceError(
        "Ollama is unreachable", status_code=502, detail="bad gateway"
    )

    response = routes_env.client.get("/models")

    assert response.status_code == 502
    assert response.json()["detail"] == "bad gateway"


def test_rag_status_passthrough(routes_env) -> None:
    response = routes_env.client.get("/rag/status")

    assert response.status_code == 200
    assert response.json() == {"model": "fake-embed", "documents": 2}


@pytest.mark.parametrize("q", ["", "   "])
def test_debug_query_rejects_blank_q(routes_env, q: str) -> None:
    response = routes_env.client.get("/rag/debug-query", params={"q": q})

    assert response.status_code == 400
    assert response.json()["detail"] == "Query 'q' is required"


def test_debug_query_strips_q_and_clamps_top_k(routes_env) -> None:
    client = routes_env.client

    response = client.get("/rag/debug-query", params={"q": "  hi  ", "top_k": 99})
    assert response.status_code == 200
    assert response.json() == {"query": "hi"}

    client.get("/rag/debug-query", params={"q": "hi", "top_k": 0})

    assert routes_env.rag.debug_calls == [("hi", 20), ("hi", 1)]


def test_dashboard_delegates_to_build_dashboard(routes_env, monkeypatch) -> None:
    calls: list[dict] = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr("app.services.dashboard.build_dashboard", fake_build)

    response = routes_env.client.get("/dashboard")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert len(calls) == 1
    # The route passes the live service objects straight through.
    kwargs = calls[0]
    assert kwargs["document_service"] is routes_env.docs
    assert kwargs["rag_service"] is routes_env.rag
    assert kwargs["project_service"].kind == "lora_projects"
    assert kwargs["training_service"].kind == "training"
    assert kwargs["comfyui_service"].kind == "comfyui"
    assert kwargs["job_store"].kind == "jobs"


# ============================================
# CHAT ENDPOINTS
# ============================================


def test_chat_plain_assembles_prompt_and_records_turns(routes_env) -> None:
    routes_env.prompts.default = DEFAULT_PROMPT

    body = _chat(routes_env.client, use_documents=False)

    assert body == {"response": "REPLY", "session_id": "s1"}
    (full_prompt, model), = routes_env.llm.chat_calls
    assert model is None
    assert full_prompt == (
        "You are an AI assistant.\nBe terse.\n\n"
        "CHAT HISTORY:\nHISTORY\n\n"
        "USER QUESTION:\nhello"
    )
    assert routes_env.memory.use_documents_calls == [("s1", False)]
    assert routes_env.memory.added == [
        ("s1", "user", "hello"),
        ("s1", "assistant", "REPLY"),
    ]


def test_chat_with_documents_injects_context(routes_env) -> None:
    routes_env.prompts.default = DEFAULT_PROMPT

    _chat(routes_env.client, use_documents=True)

    assert routes_env.rag.query_calls == ["hello"]
    (full_prompt, model), = routes_env.llm.chat_calls
    assert full_prompt == (
        "You are an AI assistant.\nBe terse.\n\n"
        "Use the context below if relevant.\n\nCONTEXT:\nCTX\n\n"
        "CHAT HISTORY:\nHISTORY\n\n"
        "USER QUESTION:\nhello"
    )


def test_chat_applies_template_but_stores_original_message(routes_env) -> None:
    routes_env.prompts.default = DEFAULT_PROMPT
    routes_env.prompts.store["p9"] = {
        "id": "p9",
        "name": "tpl",
        "content": "Template body.",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
    }

    _chat(routes_env.client, prompt_id="p9", use_documents=False)

    (full_prompt, model), = routes_env.llm.chat_calls
    assert full_prompt.endswith("USER QUESTION:\n[tpl:p9] hello")
    # The session memory keeps the user's original words, not the templated form.
    assert routes_env.memory.added[0] == ("s1", "user", "hello")


def test_chat_unknown_prompt_id_404_without_side_effects(routes_env) -> None:
    response = routes_env.client.post(
        "/chat", json={"prompt": "hello", "session_id": "s1", "prompt_id": "nope"}
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Prompt nope not found"
    assert routes_env.llm.chat_calls == []
    assert routes_env.memory.added == []


def test_chat_falls_back_to_builtin_system_prompt_without_default(routes_env) -> None:
    routes_env.memory.history_text = ""

    _chat(routes_env.client, use_documents=False)

    (full_prompt, model), = routes_env.llm.chat_calls
    # Empty doc context and history drop those sections entirely.
    assert full_prompt == (
        "You are an AI assistant.\nYou are a helpful AI assistant.\n\n"
        "USER QUESTION:\nhello"
    )


def test_chat_maps_ollama_error_to_http(routes_env) -> None:
    routes_env.llm.chat_error = OllamaServiceError(
        "model not found", status_code=404, detail="no such model"
    )

    response = routes_env.client.post("/chat", json={"prompt": "hello"})

    assert response.status_code == 404
    assert response.json()["detail"] == "no such model"


# ============================================
# SESSION ENDPOINTS
# ============================================


def test_session_history_shape(routes_env) -> None:
    response = routes_env.client.get("/sessions/s1/history")

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "s1"
    assert body["message_count"] == 2
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
    assert body["messages"][0]["content"] == "hi"


def test_clear_session_history(routes_env) -> None:
    response = routes_env.client.delete("/sessions/s1/history")

    assert response.status_code == 200
    assert response.json() == {"message": "History cleared for session s1"}
    assert routes_env.memory.cleared == ["s1"]


# ============================================
# DOCUMENT ENDPOINTS
# ============================================


def test_documents_list_projects_id_and_source(routes_env) -> None:
    routes_env.docs.docs = [
        {
            "id": "doc-1",
            "source": "notes.txt",
            "char_count": 42,
            "created_at": "2026-01-01T00:00:00",
        }
    ]

    response = routes_env.client.get("/documents")

    assert response.status_code == 200
    body = response.json()
    # Only id + source are exposed; internal fields stay server-side.
    assert body["count"] == 1
    assert body["documents"] == [{"id": "doc-1", "source": "notes.txt"}]


def test_document_upload_stores_and_indexes(routes_env) -> None:
    response = routes_env.client.post(
        "/documents/upload", files={"file": ("notes.txt", b"hello docs", "text/plain")}
    )

    assert response.status_code == 200
    assert response.json() == {
        "id": "doc-1",
        "source": "notes.txt",
        "message": "Document stored successfully",
    }
    assert routes_env.docs.stored == [("hello docs", "notes.txt")]
    # RAG indexing is keyed by doc_id so chunks can be removed with the document.
    (indexed_text, meta), = routes_env.rag.added
    assert indexed_text == "hello docs"
    assert meta == {"doc_id": "doc-1", "filename": "notes.txt"}


def test_upload_document_rejects_falsy_filename(routes_env) -> None:
    # Starlette validates UploadFile at the framework level, so an empty
    # filename arrives as a 422 before the handler body runs. Call the
    # handler directly to pin the guard's own contract.

    async def _never_read() -> bytes:
        raise AssertionError("file.read must not run for an empty filename")

    fake_file = SimpleNamespace(filename="", read=_never_read)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(upload_document(fake_file, routes_env.docs, routes_env.rag))

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "Filename is required"


def test_document_upload_rejects_unsupported_extension(routes_env) -> None:
    response = routes_env.client.post(
        "/documents/upload", files={"file": ("virus.exe", b"MZ", "application/octet-stream")}
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "Unsupported file type. Supported: .docx, .pdf, .txt"
    )
    assert routes_env.docs.stored == []


def test_document_upload_rejects_invalid_utf8(routes_env) -> None:
    response = routes_env.client.post(
        "/documents/upload", files={"file": ("notes.txt", b"\xff\xfe\xfa", "text/plain")}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "File must be valid UTF-8 text"


def test_document_upload_rejects_blank_text(routes_env) -> None:
    response = routes_env.client.post(
        "/documents/upload", files={"file": ("notes.txt", b"   \n", "text/plain")}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "No extractable text found in 'notes.txt'"


def test_document_upload_rolls_back_when_rag_fails(routes_env) -> None:
    routes_env.rag.add_error = OllamaServiceError(
        "down", status_code=503, detail="nope"
    )

    response = routes_env.client.post(
        "/documents/upload", files={"file": ("notes.txt", b"hello docs", "text/plain")}
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "nope"
    # The stored document is deleted again so the store and the index stay in sync.
    assert routes_env.docs.stored == [("hello docs", "notes.txt")]
    assert routes_env.docs.deleted == ["doc-1"]


def test_document_upload_500s_on_generic_failure_without_rollback(routes_env) -> None:
    routes_env.rag.add_error = RuntimeError("indexer exploded")

    response = routes_env.client.post(
        "/documents/upload", files={"file": ("notes.txt", b"hello docs", "text/plain")}
    )

    assert response.status_code == 500
    assert (
        response.json()["detail"] == "Error uploading document: indexer exploded"
    )
    # Only OllamaServiceError triggers the rollback path.
    assert routes_env.docs.deleted == []


def test_document_delete_found_and_missing(routes_env) -> None:
    client = routes_env.client

    response = client.delete("/documents/doc-7")
    assert response.status_code == 200
    assert response.json() == {"message": "Document doc-7 deleted"}
    # A successful delete also drops the document's chunks from RAG.
    assert routes_env.rag.removed == ["doc-7"]

    routes_env.docs.delete_result = False
    response = client.delete("/documents/doc-8")
    assert response.status_code == 404
    assert response.json()["detail"] == "Document doc-8 not found"

    assert routes_env.docs.deleted == ["doc-7", "doc-8"]


def test_documents_clear_all(routes_env) -> None:
    response = routes_env.client.delete("/documents")

    assert response.status_code == 200
    assert response.json() == {"message": "All documents cleared"}
    assert routes_env.docs.cleared == 1
    assert routes_env.rag.cleared == 1


# ============================================
# PROMPT ENDPOINTS
# ============================================


def test_prompts_list_includes_count(routes_env) -> None:
    routes_env.prompts.store["p1"] = {
        "id": "p1",
        "name": "one",
        "content": "c1",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
    }

    response = routes_env.client.get("/prompts")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["prompts"][0]["name"] == "one"


def test_prompt_create_returns_fresh_record(routes_env) -> None:
    response = routes_env.client.post(
        "/prompts", json={"name": "one", "content": "c1"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "p1"
    assert body["created_at"] == "2026-01-02T03:04:05"
    assert routes_env.prompts.created == [("one", "c1")]


def test_prompt_create_500_when_record_missing(routes_env) -> None:
    routes_env.prompts.get_after_create = False

    response = routes_env.client.post(
        "/prompts", json={"name": "ghost", "content": "c"}
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to create prompt"


def test_prompt_get_found_and_missing(routes_env) -> None:
    routes_env.prompts.store["p1"] = {
        "id": "p1",
        "name": "one",
        "content": "c1",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
    }
    client = routes_env.client

    response = client.get("/prompts/p1")
    assert response.status_code == 200
    # The record is returned flat, not wrapped in a "prompt" key.
    assert response.json() == routes_env.prompts.store["p1"]

    response = client.get("/prompts/missing")
    assert response.status_code == 404
    assert response.json()["detail"] == "Prompt missing not found"


def test_prompt_update_partial_and_missing(routes_env) -> None:
    routes_env.prompts.store["p1"] = {
        "id": "p1",
        "name": "one",
        "content": "c1",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
    }
    client = routes_env.client

    response = client.put("/prompts/p1", json={"name": "renamed"})
    assert response.status_code == 200
    body = response.json()
    # Partial update: only the provided field changes, content is untouched.
    assert body["id"] == "p1"
    assert body["name"] == "renamed"
    assert body["content"] == "c1"
    assert routes_env.prompts.updated == [("p1", "renamed", None)]

    routes_env.prompts.update_result = False
    response = client.put("/prompts/p2", json={"content": "x"})
    assert response.status_code == 404
    assert response.json()["detail"] == "Prompt p2 not found"


def test_prompt_delete_found_and_missing(routes_env) -> None:
    routes_env.prompts.store["p1"] = {
        "id": "p1",
        "name": "one",
        "content": "c1",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
    }
    client = routes_env.client

    response = client.delete("/prompts/p1")
    assert response.status_code == 200
    assert response.json() == {"message": "Prompt p1 deleted"}
    assert "p1" not in routes_env.prompts.store

    response = client.delete("/prompts/missing")
    assert response.status_code == 404
    assert response.json()["detail"] == "Prompt missing not found"

    assert routes_env.prompts.deleted == ["p1", "missing"]


# ============================================
# _build_system_prompt HELPER
# ============================================


def test_build_system_prompt_includes_all_sections() -> None:
    full = _build_system_prompt("Be terse.", "CTX", "HISTORY", "hello")

    assert full == (
        "You are an AI assistant.\nBe terse.\n\n"
        "Use the context below if relevant.\n\nCONTEXT:\nCTX\n\n"
        "CHAT HISTORY:\nHISTORY\n\n"
        "USER QUESTION:\nhello"
    )


def test_build_system_prompt_omits_empty_sections() -> None:
    # Empty doc context and history drop those sections entirely.
    assert _build_system_prompt("Be terse.", "", "", "hello") == (
        "You are an AI assistant.\nBe terse.\n\nUSER QUESTION:\nhello"
    )
