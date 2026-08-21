"""LoRA project, dataset, caption, training, and library endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import Response

from app.core.config import settings
from app.core.exceptions import ServiceError
from app.core.paths import UnsafePathError
from app.models.comfyui_schemas import JobResponse
from app.models.lora_schemas import (
    AIToolkitStatusResponse,
    CaptionGenerateRequest,
    CaptionGenerateResponse,
    CaptionUpdate,
    DatasetImageListResponse,
    DatasetImageResponse,
    LoRALibraryResponse,
    LoRAProjectCreate,
    LoRAProjectListResponse,
    LoRAProjectResponse,
    LoRAProjectUpdate,
    TrainingConfigRequest,
    TrainingStatusResponse,
)
from app.services.lora_dataset_service import LoRAProject, LoRAProjectService
from app.services.lora_training_service import LoRATrainingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/loras", tags=["lora"])


def get_project_service(request: Request) -> LoRAProjectService:
    return request.app.state.lora_project_service


def get_training_service(request: Request) -> LoRATrainingService:
    return request.app.state.lora_training_service


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ServiceError):
        return exc.to_http_exception()
    if isinstance(exc, UnsafePathError):
        logger.warning("[LORA] Rejected unsafe path: %s", exc)
        return HTTPException(status_code=400, detail="Invalid path")
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail="Unexpected error")


def _project_response(
    project: LoRAProject, service: LoRAProjectService
) -> LoRAProjectResponse:
    images = service.list_images(project.id)
    return LoRAProjectResponse(
        **project.to_dict()
        | {
            "image_count": len(images),
            "captioned_count": sum(1 for image in images if image.caption),
        }
    )


# ============================================
# LIBRARY & TOOLKIT STATUS
# ============================================


@router.get("", response_model=LoRALibraryResponse)
def list_loras(
    service: LoRAProjectService = Depends(get_project_service),
) -> LoRALibraryResponse:
    """Every discovered LoRA: trained here plus ComfyUI's own folder."""
    entries = service.list_library()
    return LoRALibraryResponse(loras=entries, count=len(entries))


@router.get("/files", response_model=LoRALibraryResponse)
def list_lora_files(
    service: LoRAProjectService = Depends(get_project_service),
) -> LoRALibraryResponse:
    """Alias of ``GET /loras`` kept for symmetry with the documented API."""
    return list_loras(service)


@router.get("/toolkit/status", response_model=AIToolkitStatusResponse)
def toolkit_status(
    service: LoRATrainingService = Depends(get_training_service),
) -> AIToolkitStatusResponse:
    """Whether Ostris AI Toolkit is configured and usable."""
    return AIToolkitStatusResponse(**service.status())


# ============================================
# PROJECTS
# ============================================


@router.post("/projects", response_model=LoRAProjectResponse)
def create_project(
    payload: LoRAProjectCreate,
    service: LoRAProjectService = Depends(get_project_service),
) -> LoRAProjectResponse:
    try:
        project = service.create_project(
            name=payload.name,
            description=payload.description,
            base_model=payload.base_model,
            arch=payload.arch,
            trigger_word=payload.trigger_word,
        )
    except (ServiceError, UnsafePathError, ValueError) as exc:
        raise _http_error(exc) from exc
    return _project_response(project, service)


@router.get("/projects", response_model=LoRAProjectListResponse)
def list_projects(
    service: LoRAProjectService = Depends(get_project_service),
) -> LoRAProjectListResponse:
    projects = [_project_response(project, service) for project in service.list_projects()]
    return LoRAProjectListResponse(projects=projects, count=len(projects))


@router.get("/projects/{project_id}", response_model=LoRAProjectResponse)
def get_project(
    project_id: str,
    service: LoRAProjectService = Depends(get_project_service),
) -> LoRAProjectResponse:
    try:
        project = service.get_project(project_id)
    except (ServiceError, UnsafePathError) as exc:
        raise _http_error(exc) from exc
    return _project_response(project, service)


@router.put("/projects/{project_id}", response_model=LoRAProjectResponse)
def update_project(
    project_id: str,
    payload: LoRAProjectUpdate,
    service: LoRAProjectService = Depends(get_project_service),
) -> LoRAProjectResponse:
    try:
        project = service.update_project(
            project_id, **payload.model_dump(exclude_none=True)
        )
    except (ServiceError, UnsafePathError) as exc:
        raise _http_error(exc) from exc
    return _project_response(project, service)


@router.delete("/projects/{project_id}")
def delete_project(
    project_id: str,
    service: LoRAProjectService = Depends(get_project_service),
) -> dict:
    try:
        deleted = service.delete_project(project_id)
    except UnsafePathError as exc:
        raise _http_error(exc) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return {"message": f"Project {project_id} deleted"}


# ============================================
# DATASET
# ============================================


@router.post("/projects/{project_id}/images", response_model=DatasetImageListResponse)
async def upload_images(
    project_id: str,
    files: list[UploadFile] = File(...),
    service: LoRAProjectService = Depends(get_project_service),
) -> DatasetImageListResponse:
    """Upload one or more training images (PNG/JPG/JPEG only, size-capped)."""
    added: list[DatasetImageResponse] = []
    errors: list[str] = []

    for upload in files:
        content = await upload.read()
        try:
            image = service.add_image(project_id, upload.filename or "image", content)
        except (ServiceError, UnsafePathError, ValueError) as exc:
            # One bad file must not discard a whole multi-file drop.
            errors.append(f"{upload.filename}: {exc}")
            continue
        added.append(
            DatasetImageResponse(
                **vars(image),
                url=f"/loras/projects/{project_id}/images/{image.id}/file",
            )
        )

    if not added and errors:
        raise HTTPException(status_code=400, detail="; ".join(errors[:5]))
    if errors:
        logger.warning("[DATASET] %s: rejected %d file(s): %s", project_id, len(errors), errors)
    return DatasetImageListResponse(images=added, count=len(added))


@router.get("/projects/{project_id}/images", response_model=DatasetImageListResponse)
def list_images(
    project_id: str,
    service: LoRAProjectService = Depends(get_project_service),
) -> DatasetImageListResponse:
    try:
        images = service.list_images(project_id)
    except (ServiceError, UnsafePathError) as exc:
        raise _http_error(exc) from exc
    payload = [
        DatasetImageResponse(
            **vars(image),
            url=f"/loras/projects/{project_id}/images/{image.id}/file",
        )
        for image in images
    ]
    return DatasetImageListResponse(images=payload, count=len(payload))


@router.get("/projects/{project_id}/images/{image_id}/file")
def get_image_file(
    project_id: str,
    image_id: str,
    service: LoRAProjectService = Depends(get_project_service),
) -> Response:
    """Serve a dataset image for preview. Paths are validated, not echoed."""
    try:
        content, media_type = service.read_image(project_id, image_id)
    except (ServiceError, UnsafePathError) as exc:
        raise _http_error(exc) from exc
    return Response(content=content, media_type=media_type)


@router.delete("/projects/{project_id}/images/{image_id}")
def delete_image(
    project_id: str,
    image_id: str,
    service: LoRAProjectService = Depends(get_project_service),
) -> dict:
    try:
        deleted = service.delete_image(project_id, image_id)
    except (ServiceError, UnsafePathError) as exc:
        raise _http_error(exc) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
    return {"message": f"Image {image_id} deleted"}


# ============================================
# CAPTIONS
# ============================================


@router.put("/projects/{project_id}/captions/{image_id}", response_model=DatasetImageResponse)
def update_caption(
    project_id: str,
    image_id: str,
    payload: CaptionUpdate,
    service: LoRAProjectService = Depends(get_project_service),
) -> DatasetImageResponse:
    """Write a caption to the matching .txt and mark it human-edited."""
    try:
        image = service.set_caption(project_id, image_id, payload.caption)
    except (ServiceError, UnsafePathError) as exc:
        raise _http_error(exc) from exc
    return DatasetImageResponse(
        **vars(image), url=f"/loras/projects/{project_id}/images/{image_id}/file"
    )


@router.post(
    "/projects/{project_id}/generate-captions", response_model=CaptionGenerateResponse
)
def generate_captions(
    project_id: str,
    payload: CaptionGenerateRequest | None = None,
    service: LoRAProjectService = Depends(get_project_service),
) -> CaptionGenerateResponse:
    """Caption the dataset with a local Ollama vision model.

    Runs synchronously: the caller wants the captions back to review. Manually
    edited captions are preserved unless ``overwrite`` is set.
    """
    payload = payload or CaptionGenerateRequest()
    try:
        results = service.generate_captions(
            project_id, model=payload.model, overwrite=payload.overwrite
        )
    except (ServiceError, UnsafePathError) as exc:
        raise _http_error(exc) from exc

    skipped = sum(1 for result in results if result.get("skipped"))
    return CaptionGenerateResponse(
        results=results,
        generated=len(results) - skipped,
        skipped=skipped,
        model=payload.model or settings.vision_model,
    )


# ============================================
# TRAINING
# ============================================


@router.post("/projects/{project_id}/train", response_model=JobResponse)
def start_training(
    project_id: str,
    payload: TrainingConfigRequest | None = None,
    service: LoRATrainingService = Depends(get_training_service),
) -> JobResponse:
    """Generate the AI Toolkit config and launch training as a subprocess."""
    payload = payload or TrainingConfigRequest()
    try:
        job = service.start_training(project_id, payload.model_dump(exclude_none=True))
    except (ServiceError, UnsafePathError) as exc:
        raise _http_error(exc) from exc
    return JobResponse(**job.to_dict())


@router.post("/projects/{project_id}/stop")
def stop_training(
    project_id: str,
    service: LoRATrainingService = Depends(get_training_service),
) -> dict:
    """Terminate this project's training process — and only that one."""
    try:
        stopped = service.stop_training(project_id)
    except (ServiceError, UnsafePathError) as exc:
        raise _http_error(exc) from exc
    if not stopped:
        raise HTTPException(
            status_code=409, detail="No training process is running for this project"
        )
    return {"message": f"Training stopped for {project_id}"}


@router.get("/projects/{project_id}/training", response_model=TrainingStatusResponse)
def get_training_status(
    project_id: str,
    service: LoRATrainingService = Depends(get_training_service),
) -> TrainingStatusResponse:
    try:
        status = service.training_status(project_id)
    except (ServiceError, UnsafePathError) as exc:
        raise _http_error(exc) from exc
    return TrainingStatusResponse(**status)


@router.get("/projects/{project_id}/training/log")
def get_training_log(
    project_id: str,
    lines: int = 200,
    service: LoRATrainingService = Depends(get_training_service),
) -> dict:
    """Tail of the training log for this project."""
    try:
        log = service.read_log(project_id, tail_lines=max(1, min(lines, 2000)))
    except (ServiceError, UnsafePathError) as exc:
        raise _http_error(exc) from exc
    return {"project_id": project_id, "log": log}


@router.post("/projects/{project_id}/config")
def write_training_config(
    project_id: str,
    payload: TrainingConfigRequest,
    service: LoRATrainingService = Depends(get_training_service),
) -> dict:
    """Generate training.yml without starting a run — lets the user preview it."""
    try:
        path = service.write_config(project_id, payload.model_dump(exclude_none=True))
    except (ServiceError, UnsafePathError) as exc:
        raise _http_error(exc) from exc
    return {"path": path.name, "config": path.read_text(encoding="utf-8")}
