"""Pydantic models and schemas for API requests and responses."""

from datetime import datetime

from pydantic import BaseModel, Field

# ============================================
# CHAT SCHEMAS
# ============================================


class ChatRequest(BaseModel):
    """Schema for enhanced chat completion request."""

    prompt: str = Field(..., description="The prompt to send to the model")
    session_id: str = Field(default="default", description="Session ID for conversation history")
    model: str | None = Field(None, description="Optional model name override")
    use_documents: bool = Field(
        default=True,
        description="When true, retrieve and inject document context via RAG",
    )
    prompt_id: str | None = Field(
        default=None,
        description="Optional prompt template ID from the prompt library",
    )


class ChatMessage(BaseModel):
    """Schema for a chat message."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(..., description="ISO format timestamp")


class ChatResponse(BaseModel):
    """Schema for chat completion response."""

    response: str = Field(..., description="The model's response")
    session_id: str = Field(..., description="Session ID")


# ============================================
# DOCUMENT SCHEMAS
# ============================================


class DocumentResponse(BaseModel):
    """Schema for document storage response."""

    id: str = Field(..., description="Document ID")
    source: str = Field(..., description="Document source")
    message: str = Field(default="Document stored successfully")


class DocumentsListResponse(BaseModel):
    """Schema for listing documents."""

    documents: list[dict] = Field(..., description="List of documents with id and source")
    count: int = Field(..., description="Total number of documents")


# ============================================
# PROMPT SCHEMAS
# ============================================


class PromptCreate(BaseModel):
    """Schema for creating a prompt template."""

    name: str = Field(..., description="Prompt name/identifier", min_length=1)
    content: str = Field(..., description="Prompt content/template", min_length=1)


class PromptUpdate(BaseModel):
    """Schema for updating a prompt template."""

    name: str | None = Field(None, description="New prompt name")
    content: str | None = Field(None, description="New prompt content")


class PromptResponse(BaseModel):
    """Schema for prompt response."""

    id: str = Field(..., description="Prompt ID")
    name: str = Field(..., description="Prompt name")
    content: str = Field(..., description="Prompt content")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class PromptListResponse(BaseModel):
    """Schema for listing prompts."""

    prompts: list[PromptResponse] = Field(..., description="List of prompts")
    count: int = Field(..., description="Total number of prompts")


# ============================================
# HEALTH & STATUS SCHEMAS
# ============================================


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str = Field(..., description="Health status of the service")
    ollama_reachable: bool | None = Field(None, description="Whether Ollama is reachable")
    model: str | None = Field(None, description="Default model")
    available_models: list[str] | None = Field(None, description="List of available models")


class ModelsResponse(BaseModel):
    """Schema for available models response."""

    models: list[str] = Field(..., description="List of available model names")


# ============================================
# SESSION & MEMORY SCHEMAS
# ============================================


class SessionHistoryResponse(BaseModel):
    """Schema for session chat history response."""

    session_id: str = Field(..., description="Session ID")
    messages: list[ChatMessage] = Field(..., description="List of messages in chronological order")
    message_count: int = Field(..., description="Total number of messages")

