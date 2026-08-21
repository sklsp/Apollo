"""Pydantic schemas for LoRA projects, datasets, and training."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LoRAProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, description="Project name")
    description: str = Field("", description="What this LoRA is for")
    arch: str = Field("sdxl", description="Model family: sdxl / flux / qwen_image / krea2")
    base_model: str | None = Field(None, description="Base model path or HF id")
    trigger_word: str = Field("", description="Token that activates the LoRA")


class LoRAProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    arch: str | None = None
    base_model: str | None = None
    trigger_word: str | None = None


class LoRAProjectResponse(BaseModel):
    id: str
    name: str
    description: str = ""
    created_at: str
    base_model: str
    arch: str
    trigger_word: str = ""
    training_status: str = "not_started"
    trained_lora_path: str | None = None
    training_job_id: str | None = None
    image_count: int = 0
    captioned_count: int = 0


class LoRAProjectListResponse(BaseModel):
    projects: list[LoRAProjectResponse]
    count: int


class DatasetImageResponse(BaseModel):
    id: str
    filename: str
    caption: str = ""
    caption_edited: bool = False
    size_bytes: int = 0
    url: str = Field("", description="Backend-served image URL")


class DatasetImageListResponse(BaseModel):
    images: list[DatasetImageResponse]
    count: int


class DatasetValidationResponse(BaseModel):
    """Quality-control report for one dataset."""

    total_images: int
    valid_images: int
    invalid_images: int
    captioned: int
    missing_captions: int
    duplicates: int
    near_duplicates: int
    extreme_resolutions: int
    average_resolution: list[int] | None = None
    issue_count: int
    score: int = Field(..., ge=0, le=100, description="Quality score out of 100")
    findings: list[dict[str, Any]] = Field(default_factory=list)


class CaptionUpdate(BaseModel):
    caption: str = Field(..., description="Caption text written to the matching .txt")


class CaptionGenerateRequest(BaseModel):
    model: str | None = Field(None, description="Ollama vision model; defaults to config")
    overwrite: bool = Field(
        False, description="Also replace captions the user has edited by hand"
    )


class CaptionGenerateResponse(BaseModel):
    results: list[dict[str, Any]]
    generated: int
    skipped: int
    model: str


class TrainingConfigRequest(BaseModel):
    """User-facing training knobs. Converted to AI Toolkit YAML server-side."""

    arch: str | None = Field(None, description="Model family override")
    base_model: str | None = Field(None, description="Base model path or HF id")
    trigger_word: str | None = None
    steps: int = Field(2000, ge=1, le=100_000)
    learning_rate: float = Field(1e-4, gt=0, le=1)
    batch_size: int = Field(1, ge=1, le=64)
    resolution: list[int] | None = Field(None, description="e.g. [768, 1024]")
    lora_rank: int = Field(16, ge=1, le=512)
    lora_alpha: int | None = Field(None, ge=1, le=512)
    save_every: int = Field(250, ge=1)
    sample_every: int = Field(0, ge=0, description="0 disables mid-training samples")


class TrainingStatusResponse(BaseModel):
    project_id: str
    status: str
    job_id: str | None = None
    running: bool = False
    progress: float | None = Field(None, description="Null when it cannot be determined")
    current_step: int = 0
    total_steps: int = 0
    loss: float | None = None
    message: str = ""
    error: str | None = None
    trained_lora_path: str | None = None


class AIToolkitStatusResponse(BaseModel):
    configured: bool
    path: str | None = None
    python: str | None = None
    supported_archs: list[str] = Field(default_factory=list)
    error: str | None = None


class TrainingPreset(BaseModel):
    """A reusable training starting point with plain-language explanation."""

    id: str
    label: str
    description: str = Field(..., description="What this preset is for")
    values: dict[str, Any] = Field(..., description="TrainingConfigRequest overrides")


class TrainingPresetListResponse(BaseModel):
    presets: list[TrainingPreset]
    count: int


class LoRALibraryEntry(BaseModel):
    name: str
    filename: str
    path: str
    source: str = Field(..., description="'project' or 'comfyui'")
    project_id: str | None = None
    project_name: str | None = None
    base_model: str | None = None
    arch: str | None = None
    trigger_word: str | None = None
    training_status: str = "external"
    size_bytes: int = 0
    created_at: str


class LoRALibraryResponse(BaseModel):
    loras: list[LoRALibraryEntry]
    count: int
