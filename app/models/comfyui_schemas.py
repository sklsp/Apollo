"""Pydantic schemas for the ComfyUI generation API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ComfyUIStatusResponse(BaseModel):
    connected: bool = Field(..., description="Whether ComfyUI answered")
    base_url: str = Field(..., description="Configured ComfyUI URL")
    version: str | None = Field(None, description="ComfyUI version")
    python_version: str | None = None
    devices: list[dict] = Field(default_factory=list, description="GPU devices and VRAM")
    checkpoints: list[str] = Field(default_factory=list, description="Installed checkpoints")
    loras: list[str] = Field(default_factory=list, description="LoRAs ComfyUI can load")
    error: str | None = Field(None, description="Why the connection failed, if it did")


class WorkflowSummary(BaseModel):
    id: str
    name: str
    description: str = ""
    arch: str = "generic"
    inputs: dict[str, dict[str, str]] = Field(default_factory=dict)
    supported_inputs: list[str] = Field(default_factory=list)
    node_count: int = 0


class WorkflowListResponse(BaseModel):
    workflows: list[WorkflowSummary]
    count: int


class WorkflowImportRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Display name for the workflow")
    workflow: dict[str, Any] = Field(..., description="ComfyUI API-format workflow JSON")
    description: str = Field("", description="What this workflow does")
    arch: str = Field("generic", description="Model family, e.g. sdxl / flux / krea2")
    inputs: dict[str, dict[str, str]] | None = Field(
        None,
        description="Optional node/field mapping. Auto-detected when omitted.",
    )


class GenerationRequest(BaseModel):
    workflow_id: str = Field(..., description="Workflow to run")
    prompt: str | None = Field(None, description="Positive prompt")
    negative_prompt: str | None = Field(None, description="Negative prompt")
    seed: int | None = Field(None, description="Seed; omit or use -1 to randomize")
    steps: int | None = Field(None, ge=1, le=200)
    cfg: float | None = Field(None, ge=0, le=30)
    width: int | None = Field(None, ge=64, le=4096)
    height: int | None = Field(None, ge=64, le=4096)
    checkpoint: str | None = Field(None, description="Checkpoint filename override")
    lora_name: str | None = Field(None, description="LoRA filename as ComfyUI sees it")
    lora_strength_model: float | None = Field(None, ge=-10, le=10)
    lora_strength_clip: float | None = Field(None, ge=-10, le=10)

    def to_params(self) -> dict[str, Any]:
        """Only the workflow-injectable fields, in mapping key form."""
        return self.model_dump(exclude={"workflow_id"}, exclude_none=True)


class JobResponse(BaseModel):
    id: str
    type: str
    status: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    progress: float | None = None
    message: str | None = None
    error: str | None = None
    outputs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
    count: int
