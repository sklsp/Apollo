"""ComfyUI orchestration: workflow library, input injection, generation jobs.

Workflows live on disk as two files so they can be swapped without touching
Python:

``workflows/<id>.json``      raw ComfyUI **API-format** export (Save (API Format))
``workflows/<id>.map.json``  which node/field each logical input writes to

Node ids are never hardcoded here — everything goes through the mapping.
"""

from __future__ import annotations

import copy
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.clients.comfyui_client import ComfyUIClient
from app.core.config import settings
from app.core.exceptions import ComfyUIServiceError, WorkflowError
from app.core.jobs import Job, JobStatus, JobStore
from app.core.paths import read_json, safe_join, sanitize_filename, write_json_atomic

logger = logging.getLogger(__name__)

# Logical inputs a workflow may expose. The UI renders controls for whichever
# ones a given workflow's mapping declares.
KNOWN_INPUTS = (
    "prompt",
    "negative_prompt",
    "seed",
    "steps",
    "cfg",
    "width",
    "height",
    "checkpoint",
    "image",
    "lora_name",
    "lora_strength_model",
    "lora_strength_clip",
)

# Node classes we can auto-map on import, and the field each logical input uses.
_SAMPLER_CLASSES = {"KSampler", "KSamplerAdvanced"}
_LATENT_CLASSES = {"EmptyLatentImage", "EmptySD3LatentImage", "EmptyLatentImagePresets"}
_LORA_CLASSES = {"LoraLoader", "LoraLoaderModelOnly"}


@dataclass
class WorkflowInfo:
    """Everything the UI needs to render a workflow without loading the graph."""

    id: str
    name: str
    description: str = ""
    arch: str = "generic"
    inputs: dict[str, dict[str, str]] = field(default_factory=dict)
    node_count: int = 0

    @property
    def supported_inputs(self) -> list[str]:
        return sorted(self.inputs.keys())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "arch": self.arch,
            "inputs": self.inputs,
            "supported_inputs": self.supported_inputs,
            "node_count": self.node_count,
        }


def validate_graph(graph: Any) -> dict:
    """Check that ``graph`` is a ComfyUI API-format workflow.

    The UI-format export (``{"nodes": [...], "links": [...]}``) is a common
    mistake, so it gets its own message rather than a generic rejection.
    """
    if not isinstance(graph, dict) or not graph:
        raise WorkflowError("Workflow must be a non-empty JSON object")

    if "nodes" in graph and isinstance(graph.get("nodes"), list):
        raise WorkflowError(
            "This looks like a ComfyUI UI-format workflow. Re-export it with "
            "'Workflow > Export (API)' and import that file instead."
        )

    for node_id, node in graph.items():
        if not isinstance(node, dict):
            raise WorkflowError(f"Node '{node_id}' is not an object")
        if not isinstance(node.get("class_type"), str):
            raise WorkflowError(f"Node '{node_id}' is missing a string 'class_type'")
        if not isinstance(node.get("inputs"), dict):
            raise WorkflowError(f"Node '{node_id}' is missing an 'inputs' object")
    return graph


def infer_mapping(graph: dict) -> dict[str, dict[str, str]]:
    """Best-effort auto-mapping so an imported workflow is usable immediately.

    Positive/negative prompts are resolved by following the sampler's own
    ``positive``/``negative`` links, so it is a lookup rather than a guess.
    """
    mapping: dict[str, dict[str, str]] = {}

    def put(name: str, node_id: str, field_name: str) -> None:
        mapping.setdefault(name, {"node": str(node_id), "field": field_name})

    for node_id, node in graph.items():
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})

        if class_type in _SAMPLER_CLASSES:
            for logical, field_name in (
                ("seed", "seed"),
                ("seed", "noise_seed"),
                ("steps", "steps"),
                ("cfg", "cfg"),
            ):
                if field_name in inputs:
                    put(logical, node_id, field_name)
            # Follow the sampler's conditioning links to the actual text nodes.
            for logical, link_name in (("prompt", "positive"), ("negative_prompt", "negative")):
                link = inputs.get(link_name)
                if isinstance(link, list) and link:
                    target = graph.get(str(link[0]), {})
                    if "text" in target.get("inputs", {}):
                        put(logical, str(link[0]), "text")

        elif class_type in _LATENT_CLASSES:
            if "width" in inputs:
                put("width", node_id, "width")
            if "height" in inputs:
                put("height", node_id, "height")

        elif class_type in _LORA_CLASSES:
            put("lora_name", node_id, "lora_name")
            if "strength_model" in inputs:
                put("lora_strength_model", node_id, "strength_model")
            if "strength_clip" in inputs:
                put("lora_strength_clip", node_id, "strength_clip")

        elif class_type == "CheckpointLoaderSimple":
            put("checkpoint", node_id, "ckpt_name")

        elif class_type == "LoadImage" and "image" in inputs:
            put("image", node_id, "image")

    return mapping


def inject_inputs(graph: dict, mapping: dict, params: dict[str, Any]) -> dict:
    """Return a copy of ``graph`` with mapped inputs replaced by ``params``.

    Unmapped or ``None`` parameters are left alone, so a workflow that has no
    CFG control simply keeps whatever its author saved.
    """
    result = copy.deepcopy(graph)

    for name, value in params.items():
        if value is None or name not in mapping:
            continue
        target = mapping[name]
        node_id, field_name = str(target.get("node")), target.get("field")
        node = result.get(node_id)
        if node is None:
            raise WorkflowError(
                f"Workflow mapping points at node '{node_id}' for '{name}', "
                "but that node is not in the workflow"
            )
        if not field_name:
            raise WorkflowError(f"Mapping for '{name}' is missing a 'field'")
        node.setdefault("inputs", {})[field_name] = value

    return result


class ComfyUIService:
    """Workflow library plus generation orchestration."""

    def __init__(
        self,
        client: ComfyUIClient | None = None,
        workflow_dir: str | None = None,
        jobs: JobStore | None = None,
        output_dir: str | None = None,
    ) -> None:
        self.client = client or ComfyUIClient()
        self.workflow_dir = Path(workflow_dir or settings.comfyui_workflow_dir)
        self.output_dir = Path(output_dir or settings.generated_dir)
        self.jobs = jobs or JobStore(persist_path=settings.jobs_file)
        self.workflow_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- status ----------

    def status(self) -> dict[str, Any]:
        """Connection state plus what the running instance actually has installed."""
        info: dict[str, Any] = {"base_url": self.client.base_url, "connected": False}
        try:
            stats = self.client.ping()
        except ComfyUIServiceError as exc:
            info["error"] = exc.detail
            return info

        system = stats.get("system", {})
        info.update(
            connected=True,
            version=system.get("comfyui_version"),
            python_version=system.get("python_version"),
            devices=[
                {
                    "name": device.get("name"),
                    "vram_total": device.get("vram_total"),
                    "vram_free": device.get("vram_free"),
                }
                for device in stats.get("devices", [])
            ],
        )
        # Model lists are a separate call and must not break the status card.
        try:
            info["checkpoints"] = self.client.list_checkpoints()
            info["loras"] = self.client.list_loras()
        except ComfyUIServiceError as exc:
            logger.warning("[COMFYUI] Could not list models: %s", exc.detail)
            info["checkpoints"] = []
            info["loras"] = []
        return info

    # ---------- workflow library ----------

    def _graph_path(self, workflow_id: str) -> Path:
        return safe_join(self.workflow_dir, f"{workflow_id}.json")

    def _map_path(self, workflow_id: str) -> Path:
        return safe_join(self.workflow_dir, f"{workflow_id}.map.json")

    def list_workflows(self) -> list[WorkflowInfo]:
        workflows: list[WorkflowInfo] = []
        for path in sorted(self.workflow_dir.glob("*.json")):
            if path.name.endswith(".map.json"):
                continue
            try:
                workflows.append(self._load_info(path.stem))
            except WorkflowError as exc:
                logger.warning("[COMFYUI] Skipping workflow %s: %s", path.stem, exc.detail)
        return workflows

    def _load_info(self, workflow_id: str) -> WorkflowInfo:
        graph = read_json(self._graph_path(workflow_id))
        if not isinstance(graph, dict):
            raise WorkflowError(f"Workflow '{workflow_id}' could not be read")
        meta = read_json(self._map_path(workflow_id), default={}) or {}
        return WorkflowInfo(
            id=workflow_id,
            name=meta.get("name", workflow_id.replace("_", " ").title()),
            description=meta.get("description", ""),
            arch=meta.get("arch", "generic"),
            inputs=meta.get("inputs", {}),
            node_count=len(graph),
        )

    def load_workflow(self, workflow_id: str) -> tuple[dict, dict]:
        """Return ``(graph, mapping)`` for a stored workflow."""
        graph_path = self._graph_path(workflow_id)
        if not graph_path.is_file():
            raise WorkflowError(f"Workflow '{workflow_id}' not found", status_code=404)

        graph = read_json(graph_path)
        if not isinstance(graph, dict):
            raise WorkflowError(f"Workflow '{workflow_id}' is not valid JSON")
        validate_graph(graph)

        meta = read_json(self._map_path(workflow_id), default={}) or {}
        return graph, meta.get("inputs", {})

    def save_workflow(
        self,
        name: str,
        graph: dict,
        description: str = "",
        arch: str = "generic",
        inputs: dict | None = None,
    ) -> WorkflowInfo:
        """Validate and store an imported workflow, auto-mapping its inputs."""
        validate_graph(graph)

        workflow_id = sanitize_filename(name, default="workflow").removesuffix(".json")
        workflow_id = workflow_id.replace(".", "_").lower() or "workflow"

        mapping = inputs if inputs else infer_mapping(graph)
        unknown = set(mapping) - set(KNOWN_INPUTS)
        if unknown:
            raise WorkflowError(f"Unknown mapped inputs: {', '.join(sorted(unknown))}")

        write_json_atomic(self._graph_path(workflow_id), graph)
        write_json_atomic(
            self._map_path(workflow_id),
            {"name": name, "description": description, "arch": arch, "inputs": mapping},
        )
        logger.info("[COMFYUI] Saved workflow '%s' (%d nodes)", workflow_id, len(graph))
        return self._load_info(workflow_id)

    def delete_workflow(self, workflow_id: str) -> bool:
        graph_path = self._graph_path(workflow_id)
        if not graph_path.is_file():
            return False
        graph_path.unlink()
        self._map_path(workflow_id).unlink(missing_ok=True)
        logger.info("[COMFYUI] Deleted workflow '%s'", workflow_id)
        return True

    # ---------- generation ----------

    def build_graph(self, workflow_id: str, params: dict[str, Any]) -> tuple[dict, int]:
        """Prepare a submittable graph. Returns ``(graph, resolved_seed)``."""
        graph, mapping = self.load_workflow(workflow_id)

        params = dict(params)
        seed = params.get("seed")
        if seed is None or int(seed) < 0:
            seed = random.randint(0, 2**32 - 1)
        params["seed"] = int(seed)

        # ponytail: a workflow with a LoRA node but no LoRA selected is neutralised
        # by zeroing its strengths rather than rewiring the graph around the node.
        # Swap for real bypass only if a workflow appears where 0.0 is not a no-op.
        if "lora_name" in mapping and not params.get("lora_name"):
            params.pop("lora_name", None)
            params["lora_strength_model"] = 0.0
            params["lora_strength_clip"] = 0.0

        return inject_inputs(graph, mapping, params), int(seed)

    def submit_generation(self, workflow_id: str, params: dict[str, Any]) -> Job:
        """Queue a generation and return immediately with a job handle."""
        graph, seed = self.build_graph(workflow_id, params)

        job = self.jobs.submit(
            "comfyui_generation",
            lambda job: self._run_generation(job, graph),
            metadata={
                "workflow_id": workflow_id,
                "seed": seed,
                "prompt": params.get("prompt", ""),
                "lora_name": params.get("lora_name"),
            },
        )
        logger.info("[COMFYUI JOB] %s queued for workflow '%s'", job.id, workflow_id)
        return job

    def _run_generation(self, job: Job, graph: dict) -> None:
        prompt_id = self.client.queue_prompt(graph)
        self.jobs.update(
            job.id,
            message="Queued in ComfyUI",
            metadata={**job.metadata, "prompt_id": prompt_id},
        )

        deadline = time.monotonic() + settings.comfyui_generation_timeout
        while time.monotonic() < deadline:
            if self.jobs.is_cancelled(job.id):
                logger.info("[COMFYUI JOB] %s cancelled by user", job.id)
                return

            record = self.client.history(prompt_id)
            if record:
                status = record.get("status", {})
                if status.get("status_str") == "error" or status.get("completed") is False:
                    raise ComfyUIServiceError(
                        "ComfyUI reported an execution error",
                        detail=_first_error(status),
                    )
                if record.get("outputs"):
                    saved = self._save_outputs(job.id, record["outputs"])
                    self.jobs.update(
                        job.id,
                        outputs=saved,
                        message=f"Generated {len(saved)} image(s)",
                    )
                    logger.info("[COMFYUI JOB] %s produced %d image(s)", job.id, len(saved))
                    return
            else:
                self.jobs.update(job.id, status=JobStatus.RUNNING, message="Waiting for ComfyUI")

            time.sleep(settings.comfyui_poll_interval)

        raise ComfyUIServiceError(
            f"Generation timed out after {settings.comfyui_generation_timeout:.0f}s",
            status_code=504,
        )

    def _save_outputs(self, job_id: str, outputs: dict) -> list[str]:
        """Copy ComfyUI's images into our own managed folder.

        The browser is never handed a ComfyUI filesystem path — it gets a
        filename that only resolves inside ``generated_dir``.
        """
        saved: list[str] = []
        for node_output in outputs.values():
            for index, image in enumerate(node_output.get("images", [])):
                filename = image.get("filename")
                if not filename:
                    continue
                try:
                    content = self.client.view_image(
                        filename,
                        image.get("subfolder", ""),
                        image.get("type", "output"),
                    )
                except ComfyUIServiceError as exc:
                    logger.warning("[COMFYUI JOB] Could not fetch %s: %s", filename, exc.detail)
                    continue

                suffix = Path(sanitize_filename(filename)).suffix or ".png"
                local_name = f"{job_id}_{len(saved)}{suffix}"
                (self.output_dir / local_name).write_bytes(content)
                saved.append(local_name)
        return saved

    def read_generated(self, filename: str) -> bytes:
        """Read one previously generated image by name (traversal-checked)."""
        path = safe_join(self.output_dir, sanitize_filename(filename))
        if not path.is_file():
            raise WorkflowError(f"Image '{filename}' not found", status_code=404)
        return path.read_bytes()

    # ---------- LoRA test integration ----------

    def prepare_lora_test(
        self,
        lora_filename: str,
        workflow_id: str | None = None,
        prompt: str | None = None,
        trigger_word: str | None = None,
    ) -> dict[str, Any]:
        """Build everything needed to test a trained LoRA in one click.

        Picks (or validates) a workflow with a LoRA node, checks that ComfyUI
        can actually see the file, and returns a prefilled
        :class:`GenerationRequest`-shaped payload for ``POST /comfyui/generate``.
        No generation is queued here — the user still presses Generate.
        """
        workflows = self.list_workflows()
        candidates = [w for w in workflows if "lora_name" in w.supported_inputs]
        if not candidates:
            raise WorkflowError(
                "No saved workflow has a LoRA node. Import one with a "
                "'LoraLoader' node to test LoRAs.",
                status_code=404,
            )

        chosen: WorkflowInfo | None = None
        if workflow_id:
            chosen = next((w for w in candidates if w.id == workflow_id), None)
            if chosen is None:
                raise WorkflowError(
                    f"Workflow '{workflow_id}' either does not exist or has no "
                    "LoRA node",
                    status_code=404,
                )
        else:
            chosen = candidates[0]

        connected = False
        visible_loras: list[str] = []
        try:
            status = self.status()
            connected = bool(status.get("connected"))
            visible_loras = status.get("loras") or []
        except ComfyUIServiceError:
            pass

        warnings: list[str] = []
        if not connected:
            warnings.append(
                "ComfyUI is not reachable; start it before generating."
            )
        elif lora_filename not in visible_loras:
            warnings.append(
                f"'{lora_filename}' is not in ComfyUI's loras folder yet. Copy "
                "it there (or set COMFYUI_LORA_DIR) and restart ComfyUI."
            )

        full_prompt = prompt or (
            f"{trigger_word}, portrait photo, soft lighting" if trigger_word
            else "portrait photo, soft lighting"
        )

        return {
            "workflow": chosen.to_dict(),
            "generation_request": {
                "workflow_id": chosen.id,
                "prompt": full_prompt,
                "lora_name": lora_filename,
                "lora_strength_model": 0.8,
                "lora_strength_clip": 0.8,
            },
            "connected": connected,
            "warnings": warnings,
        }


def _first_error(status: dict) -> str:
    for message in status.get("messages", []):
        if isinstance(message, list) and len(message) > 1 and "error" in str(message[0]):
            return str(message[1])[:500]
    return str(status)[:500]
