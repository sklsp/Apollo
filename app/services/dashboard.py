"""Project dashboard: one view of everything in the workspace.

Aggregates counts and recent activity across the document, dataset, training
and generation subsystems. Read-only over existing services — no duplicated
state, no new storage.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


def build_dashboard(
    *,
    document_service: Any,
    rag_service: Any,
    project_service: Any,
    training_service: Any,
    comfyui_service: Any,
    job_store: Any,
) -> dict[str, Any]:
    """Assemble the unified dashboard payload.

    Every section degrades independently: a subsystem being down (ComfyUI
    offline, toolkit unconfigured) must never blank the whole dashboard.
    """
    dashboard: dict[str, Any] = {"generated_at": _now()}

    # ---- documents ----
    try:
        docs = document_service.get_all_documents()
        dashboard["documents"] = {
            "count": len(docs),
            "indexed_chunks": rag_service.chunk_count,
            "embedding_backend": rag_service.embedding_client.backend,
            "embedding_model": rag_service.embedding_client.model,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("[DASHBOARD] documents section failed: %s", exc)
        dashboard["documents"] = {"error": str(exc)[:120]}

    # ---- datasets / LoRA projects ----
    try:
        projects = []
        for project in project_service.list_projects():
            images = project_service.list_images(project.id)
            captioned = sum(1 for image in images if image.caption.strip())
            runs = training_service.list_runs(project.id)
            projects.append({
                "id": project.id,
                "name": project.name,
                "arch": project.arch,
                "trigger_word": project.trigger_word,
                "training_status": project.training_status,
                "image_count": len(images),
                "captioned_count": captioned,
                "run_count": len(runs),
                "latest_run": runs[0] if runs else None,
                "trained_lora_path": project.trained_lora_path,
            })
        total_images = sum(p["image_count"] for p in projects)
        total_loras = sum(
            1 for p in projects if p["trained_lora_path"]
        )
        dashboard["datasets"] = {
            "project_count": len(projects),
            "image_count": total_images,
            "lora_count": total_loras,
            "projects": projects,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("[DASHBOARD] datasets section failed: %s", exc)
        dashboard["datasets"] = {"error": str(exc)[:120]}

    # ---- generated images ----
    try:
        generated = comfyui_service.list_generated(limit=12)
        dashboard["generations"] = {
            "recent_count": len(generated),
            "recent": generated,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("[DASHBOARD] generations section failed: %s", exc)
        dashboard["generations"] = {"error": str(exc)[:120]}

    # ---- jobs ----
    try:
        jobs = job_store.list(limit=10)
        active = [j for j in jobs if j.status.value in ("queued", "running")]
        dashboard["jobs"] = {
            "active_count": len(active),
            "active": [_job_summary(job) for job in active],
            "recent": [_job_summary(job) for job in jobs[:6]],
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("[DASHBOARD] jobs section failed: %s", exc)
        dashboard["jobs"] = {"error": str(exc)[:120]}

    # ---- system health (never raises; each dependency reports its own state) --
    dashboard["health"] = _health_summary()

    return dashboard


def _job_summary(job: Any) -> dict[str, Any]:
    data = job.to_dict()
    return {
        "id": data["id"],
        "type": data["type"],
        "status": data["status"],
        "progress": data.get("progress"),
        "message": data.get("message"),
        "elapsed_seconds": data.get("elapsed_seconds"),
        "created_at": data["created_at"],
    }


def _health_summary() -> dict[str, Any]:
    """Lightweight dependency probe with per-dependency status."""
    import requests

    health: dict[str, Any] = {}

    health["ollama"] = _probe(f"{settings.ollama_base_url}/api/version")
    health["comfyui"] = _probe(f"{settings.comfyui_base_url}/system_stats")

    configured = settings.ai_toolkit_configured
    health["ai_toolkit"] = {
        "configured": configured,
        "detail": settings.ai_toolkit_path or "not configured",
    }
    return health


def _probe(url: str, timeout: float = 2.0) -> dict[str, Any]:
    try:
        response = requests.get(url, timeout=timeout)
        return {"online": response.status_code < 500, "url": url}
    except requests.RequestException:
        return {"online": False, "url": url}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
