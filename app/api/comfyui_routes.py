"""ComfyUI generation, workflow library, and job endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from app.core.exceptions import ServiceError
from app.core.jobs import Job, JobStore
from app.core.paths import UnsafePathError
from app.models.comfyui_schemas import (
    ComfyUIStatusResponse,
    GenerationRequest,
    JobListResponse,
    JobResponse,
    WorkflowImportRequest,
    WorkflowListResponse,
    WorkflowSummary,
)
from app.services.comfyui_service import ComfyUIService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/comfyui", tags=["comfyui"])
jobs_router = APIRouter(prefix="/jobs", tags=["jobs"])


def get_comfyui_service(request: Request) -> ComfyUIService:
    return request.app.state.comfyui_service


def get_job_store(request: Request) -> JobStore:
    return request.app.state.job_store


def _job_response(job: Job) -> JobResponse:
    return JobResponse(**job.to_dict())


# ============================================
# CONNECTION
# ============================================


@router.get("/status", response_model=ComfyUIStatusResponse)
def comfyui_status(
    service: ComfyUIService = Depends(get_comfyui_service),
) -> ComfyUIStatusResponse:
    """Connection state plus the models the running instance actually has.

    Never raises: an unreachable ComfyUI is reported as ``connected: false``
    so the dashboard can render an offline card instead of an error page.
    """
    return ComfyUIStatusResponse(**service.status())


@router.post("/test", response_model=ComfyUIStatusResponse)
def comfyui_test(
    service: ComfyUIService = Depends(get_comfyui_service),
) -> ComfyUIStatusResponse:
    """Explicit 'Test connection' action for the UI button."""
    return ComfyUIStatusResponse(**service.status())


# ============================================
# WORKFLOWS
# ============================================


@router.get("/workflows", response_model=WorkflowListResponse)
def list_workflows(
    service: ComfyUIService = Depends(get_comfyui_service),
) -> WorkflowListResponse:
    """List saved workflows and the inputs each one exposes."""
    workflows = [WorkflowSummary(**info.to_dict()) for info in service.list_workflows()]
    return WorkflowListResponse(workflows=workflows, count=len(workflows))


@router.post("/workflows", response_model=WorkflowSummary)
def import_workflow(
    payload: WorkflowImportRequest,
    service: ComfyUIService = Depends(get_comfyui_service),
) -> WorkflowSummary:
    """Import a ComfyUI API-format workflow, auto-mapping its inputs."""
    try:
        info = service.save_workflow(
            name=payload.name,
            graph=payload.workflow,
            description=payload.description,
            arch=payload.arch,
            inputs=payload.inputs,
        )
    except ServiceError as exc:
        raise exc.to_http_exception() from exc
    return WorkflowSummary(**info.to_dict())


@router.get("/workflows/{workflow_id}", response_model=dict)
def get_workflow(
    workflow_id: str,
    service: ComfyUIService = Depends(get_comfyui_service),
) -> dict:
    """Full graph plus mapping — used by the workflow input mapper UI."""
    try:
        graph, mapping = service.load_workflow(workflow_id)
    except (ServiceError, UnsafePathError) as exc:
        raise _http_error(exc) from exc
    return {
        "id": workflow_id,
        "workflow": graph,
        "inputs": mapping,
        "nodes": [
            {"id": node_id, "class_type": node.get("class_type")}
            for node_id, node in graph.items()
        ],
    }


@router.delete("/workflows/{workflow_id}")
def delete_workflow(
    workflow_id: str,
    service: ComfyUIService = Depends(get_comfyui_service),
) -> dict:
    try:
        deleted = service.delete_workflow(workflow_id)
    except UnsafePathError as exc:
        raise _http_error(exc) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    return {"message": f"Workflow {workflow_id} deleted"}


# ============================================
# GENERATION
# ============================================


@router.post("/generate", response_model=JobResponse)
def generate(
    payload: GenerationRequest,
    service: ComfyUIService = Depends(get_comfyui_service),
) -> JobResponse:
    """Queue a generation. Returns immediately with a job to poll."""
    try:
        job = service.submit_generation(payload.workflow_id, payload.to_params())
    except (ServiceError, UnsafePathError) as exc:
        raise _http_error(exc) from exc
    return _job_response(job)


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_generation_job(
    job_id: str,
    store: JobStore = Depends(get_job_store),
) -> JobResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _job_response(job)


@router.get("/images/{filename}")
def get_generated_image(
    filename: str,
    service: ComfyUIService = Depends(get_comfyui_service),
) -> Response:
    """Serve a generated image from our managed output folder only."""
    try:
        content = service.read_generated(filename)
    except (ServiceError, UnsafePathError) as exc:
        raise _http_error(exc) from exc
    media_type = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
    return Response(content=content, media_type=media_type)


# ============================================
# JOBS (shared by generation and training)
# ============================================


@jobs_router.get("", response_model=JobListResponse)
def list_jobs(
    job_type: str | None = None,
    limit: int = 50,
    store: JobStore = Depends(get_job_store),
) -> JobListResponse:
    jobs = [_job_response(job) for job in store.list(job_type=job_type, limit=limit)]
    return JobListResponse(jobs=jobs, count=len(jobs))


@jobs_router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, store: JobStore = Depends(get_job_store)) -> JobResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _job_response(job)


@jobs_router.delete("/{job_id}", response_model=JobResponse)
def cancel_job(job_id: str, store: JobStore = Depends(get_job_store)) -> JobResponse:
    """Request cancellation. Workers stop at their next checkpoint."""
    if not store.cancel(job_id):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        raise HTTPException(status_code=409, detail=f"Job {job_id} has already finished")
    job = store.get(job_id)
    assert job is not None
    return _job_response(job)


def _http_error(exc: Exception) -> HTTPException:
    """Map service/path errors onto HTTP, never leaking a traceback."""
    if isinstance(exc, ServiceError):
        return exc.to_http_exception()
    if isinstance(exc, UnsafePathError):
        logger.warning("[COMFYUI] Rejected unsafe path: %s", exc)
        return HTTPException(status_code=400, detail="Invalid path")
    return HTTPException(status_code=500, detail="Unexpected error")
