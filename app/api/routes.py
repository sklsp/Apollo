"""API routes for Apollo."""

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.core.config import settings
from app.core.exceptions import OllamaServiceError
from app.models.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    DocumentResponse,
    DocumentsListResponse,
    HealthResponse,
    ModelsResponse,
    PromptCreate,
    PromptListResponse,
    PromptResponse,
    PromptUpdate,
    SessionHistoryResponse,
)
from app.services.document_service import DocumentService
from app.services.llm_service import LLMService
from app.services.memory_service import MemoryService
from app.services.prompt_service import PromptService
from app.services.rag.ingestion import SUPPORTED_EXTENSIONS, extract_text
from app.services.rag.service import RAGService

router = APIRouter()


# ============================================
# DEPENDENCY INJECTION
# ============================================


def get_llm_service(request: Request) -> LLMService:
    """Get LLM service from app state."""
    return request.app.state.llm_service


def get_memory_service(request: Request) -> MemoryService:
    """Get memory service from app state."""
    return request.app.state.memory_service


def get_document_service(request: Request) -> DocumentService:
    """Get document service from app state."""
    return request.app.state.document_service


def get_prompt_service(request: Request) -> PromptService:
    """Get prompt service from app state."""
    return request.app.state.prompt_service


def get_rag_service(request: Request) -> RAGService:
    """Get RAG service from app state."""
    return request.app.state.rag_service


# ============================================
# HEALTH & STATUS ENDPOINTS
# ============================================


@router.get("/health", response_model=HealthResponse)
def health(service: LLMService = Depends(get_llm_service)) -> HealthResponse:
    """Check service health and Ollama connectivity."""
    try:
        payload = service.health_check()
    except OllamaServiceError as exc:
        # A structured 503 with setup guidance, not a bare error — the UI
        # renders this as an actionable card rather than a dead endpoint.
        base_url = getattr(getattr(service, "client", None), "base_url",
                           settings.ollama_base_url)
        raise HTTPException(
            status_code=exc.status_code,
            detail={
                "message": "Ollama is not reachable.",
                "url_checked": base_url,
                "expected_model": settings.default_model,
                "how_to_fix": [
                    "Install Ollama from https://ollama.com if it is not installed.",
                    f"Start it with: ollama serve (listening on {base_url})",
                    "Pull a chat model with: ollama pull llama3.2",
                    "Pull an embedding model with: ollama pull nomic-embed-text",
                ],
                "unavailable_features": [
                    "Chat", "RAG document Q&A", "AI dataset captioning",
                ],
                "working_features": [
                    "ComfyUI image generation", "Dataset management",
                    "Dataset validation", "LoRA training",
                ],
                "raw_detail": exc.detail,
            },
        ) from exc
    return HealthResponse(**payload)


@router.get("/models", response_model=ModelsResponse)
def list_models(service: LLMService = Depends(get_llm_service)) -> ModelsResponse:
    """List available Ollama models."""
    try:
        model_names = service.list_models()
    except OllamaServiceError as exc:
        raise exc.to_http_exception() from exc
    return ModelsResponse(models=model_names)


@router.get("/rag/status")
def rag_status(
    rag_service: RAGService = Depends(get_rag_service),
) -> dict:
    """Developer view of the vector index: what is indexed, with which model.

    Exposes retrieval metadata (content hashes, chunk counts, embedding model)
    but never raw document text.
    """
    return rag_service.status()


@router.get("/rag/debug-query")
def rag_debug_query(
    q: str,
    top_k: int = 4,
    rag_service: RAGService = Depends(get_rag_service),
) -> dict:
    """Developer view of one retrieval: query → chunks → scores → context.

    Shows exactly what the model would see for this question. Not used by the
    normal chat flow — it exists to explain why an answer came out the way it
    did.
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query 'q' is required")
    return rag_service.query_debug(q.strip(), top_k=max(1, min(top_k, 20)))


@router.get("/dashboard")
def dashboard(request: Request) -> dict:
    """Unified workspace overview: counts, recent activity, system health."""
    from app.services.dashboard import build_dashboard

    state = request.app.state
    return build_dashboard(
        document_service=state.document_service,
        rag_service=state.rag_service,
        project_service=state.lora_project_service,
        training_service=state.lora_training_service,
        comfyui_service=state.comfyui_service,
        job_store=state.job_store,
    )


# ============================================
# CHAT ENDPOINTS
# ============================================


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    llm_service: LLMService = Depends(get_llm_service),
    memory_service: MemoryService = Depends(get_memory_service),
    rag_service: RAGService = Depends(get_rag_service),
    prompt_service: PromptService = Depends(get_prompt_service),
) -> ChatResponse:
    """
    Enhanced chat endpoint with:
    - Session-based conversation history
    - RAG document context retrieval
    - Prompt template integration
    """
    try:
        session_id = request.session_id
        user_message = request.prompt
        model = request.model

        memory_service.set_use_documents(session_id, request.use_documents)
        use_documents = request.use_documents

        # 1. Apply optional prompt template from library
        llm_user_prompt = user_message
        if request.prompt_id:
            template = prompt_service.get_prompt(request.prompt_id)
            if template is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Prompt {request.prompt_id} not found",
                )
            llm_user_prompt = prompt_service.apply_template(request.prompt_id, user_message)

        # 2. Retrieve chat history
        history_text = memory_service.format_history_for_prompt(session_id)

        # 3. Optionally retrieve document context via RAG
        doc_context = ""
        if use_documents:
            retrieved_chunks = rag_service.query(llm_user_prompt)
            doc_context = rag_service.format_context(retrieved_chunks)

        # 4. System instructions
        default_prompt = prompt_service.get_prompt_by_name("default")
        system_prompt = (
            default_prompt["content"]
            if default_prompt
            else "You are a helpful AI assistant."
        )

        # 5. Build the full prompt
        full_prompt = _build_system_prompt(
            system_prompt=system_prompt,
            doc_context=doc_context,
            history=history_text,
            user_prompt=llm_user_prompt,
        )

        # 6. Store the original user message in session memory
        memory_service.add_message(session_id, "user", user_message)

        # 7. Get response from LLM
        response_text = llm_service.chat(full_prompt, model)

        # 8. Add assistant response to history
        memory_service.add_message(session_id, "assistant", response_text)

        return ChatResponse(response=response_text, session_id=session_id)

    except OllamaServiceError as exc:
        raise exc.to_http_exception() from exc


@router.get("/sessions/{session_id}/history", response_model=SessionHistoryResponse)
def get_session_history(
    session_id: str,
    memory_service: MemoryService = Depends(get_memory_service),
) -> SessionHistoryResponse:
    """Get conversation history for a specific session."""
    history = memory_service.get_history(session_id)
    messages = [ChatMessage(**msg) for msg in history]
    return SessionHistoryResponse(
        session_id=session_id,
        messages=messages,
        message_count=len(messages),
    )


@router.delete("/sessions/{session_id}/history")
def clear_session_history(
    session_id: str,
    memory_service: MemoryService = Depends(get_memory_service),
) -> dict:
    """Clear conversation history for a specific session."""
    memory_service.clear_history(session_id)
    return {"message": f"History cleared for session {session_id}"}


# ============================================
# DOCUMENT ENDPOINTS
# ============================================


@router.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
    rag_service: RAGService = Depends(get_rag_service),
) -> DocumentResponse:
    """Upload and index a document (.txt, .pdf, .docx)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    extension = Path(file.filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Supported: {supported}")

    temp_path: str | None = None
    try:
        raw_content = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_file.write(raw_content)
            temp_path = temp_file.name

        print("[UPLOAD] filename:", file.filename)
        print("[UPLOAD] file type detected:", extension)

        text_content = extract_text(temp_path, file.filename)

        print("[UPLOAD] extracted chars:", len(text_content))

        doc_id = document_service.store_document(text_content, source=file.filename)
        try:
            chunk_count = rag_service.add_document(
                text_content,
                {"doc_id": doc_id, "filename": file.filename},
            )
            print("[UPLOAD] rag chunks indexed:", chunk_count)
        except OllamaServiceError:
            document_service.delete_document(doc_id)
            raise

        return DocumentResponse(id=doc_id, source=file.filename)

    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be valid UTF-8 text")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except OllamaServiceError as exc:
        raise exc.to_http_exception() from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@router.get("/documents", response_model=DocumentsListResponse)
def list_documents(
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentsListResponse:
    """List all stored documents."""
    docs = document_service.get_all_documents()
    doc_list = [{"id": doc["id"], "source": doc["source"]} for doc in docs]
    return DocumentsListResponse(documents=doc_list, count=len(doc_list))


@router.delete("/documents/{doc_id}")
def delete_document(
    doc_id: str,
    document_service: DocumentService = Depends(get_document_service),
    rag_service: RAGService = Depends(get_rag_service),
) -> dict:
    """Delete a stored document."""
    if document_service.delete_document(doc_id):
        rag_service.remove_document(doc_id)
        return {"message": f"Document {doc_id} deleted"}
    raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")


@router.delete("/documents")
def clear_all_documents(
    document_service: DocumentService = Depends(get_document_service),
    rag_service: RAGService = Depends(get_rag_service),
) -> dict:
    """Clear all stored documents."""
    document_service.clear_all()
    rag_service.clear()
    return {"message": "All documents cleared"}


# ============================================
# PROMPT ENDPOINTS
# ============================================


@router.post("/prompts", response_model=PromptResponse)
def create_prompt(
    prompt_create: PromptCreate,
    prompt_service: PromptService = Depends(get_prompt_service),
) -> PromptResponse:
    """Create a new prompt template."""
    prompt_id = prompt_service.create_prompt(prompt_create.name, prompt_create.content)
    prompt = prompt_service.get_prompt(prompt_id)
    if prompt:
        return PromptResponse(
            id=prompt["id"],
            name=prompt["name"],
            content=prompt["content"],
            created_at=prompt["created_at"],
            updated_at=prompt["updated_at"],
        )
    raise HTTPException(status_code=500, detail="Failed to create prompt")


@router.get("/prompts", response_model=PromptListResponse)
def list_prompts(
    prompt_service: PromptService = Depends(get_prompt_service),
) -> PromptListResponse:
    """List all prompt templates."""
    prompts = prompt_service.list_prompts()
    prompt_responses = [
        PromptResponse(
            id=prompt["id"],
            name=prompt["name"],
            content=prompt["content"],
            created_at=prompt["created_at"],
            updated_at=prompt["updated_at"],
        )
        for prompt in prompts
    ]
    return PromptListResponse(prompts=prompt_responses, count=len(prompt_responses))


@router.get("/prompts/{prompt_id}", response_model=PromptResponse)
def get_prompt(
    prompt_id: str,
    prompt_service: PromptService = Depends(get_prompt_service),
) -> PromptResponse:
    """Get a specific prompt template."""
    prompt = prompt_service.get_prompt(prompt_id)
    if prompt:
        return PromptResponse(
            id=prompt["id"],
            name=prompt["name"],
            content=prompt["content"],
            created_at=prompt["created_at"],
            updated_at=prompt["updated_at"],
        )
    raise HTTPException(status_code=404, detail=f"Prompt {prompt_id} not found")


@router.put("/prompts/{prompt_id}", response_model=PromptResponse)
def update_prompt(
    prompt_id: str,
    prompt_update: PromptUpdate,
    prompt_service: PromptService = Depends(get_prompt_service),
) -> PromptResponse:
    """Update an existing prompt template."""
    if prompt_service.update_prompt(
        prompt_id,
        name=prompt_update.name,
        content=prompt_update.content,
    ):
        prompt = prompt_service.get_prompt(prompt_id)
        if prompt:
            return PromptResponse(
                id=prompt["id"],
                name=prompt["name"],
                content=prompt["content"],
                created_at=prompt["created_at"],
                updated_at=prompt["updated_at"],
            )
    raise HTTPException(status_code=404, detail=f"Prompt {prompt_id} not found")


@router.delete("/prompts/{prompt_id}")
def delete_prompt(
    prompt_id: str,
    prompt_service: PromptService = Depends(get_prompt_service),
) -> dict:
    """Delete a prompt template."""
    if prompt_service.delete_prompt(prompt_id):
        return {"message": f"Prompt {prompt_id} deleted"}
    raise HTTPException(status_code=404, detail=f"Prompt {prompt_id} not found")


# ============================================
# HELPERS
# ============================================


def _build_system_prompt(
    system_prompt: str,
    doc_context: str,
    history: str,
    user_prompt: str,
) -> str:
    """Build the complete prompt with RAG context, history, and user question."""
    parts = [f"You are an AI assistant.\n{system_prompt}"]

    if doc_context:
        parts.append(
            "Use the context below if relevant.\n\n"
            f"CONTEXT:\n{doc_context}"
        )

    if history:
        parts.append(f"CHAT HISTORY:\n{history}")

    parts.append(f"USER QUESTION:\n{user_prompt}")
    return "\n\n".join(parts)
