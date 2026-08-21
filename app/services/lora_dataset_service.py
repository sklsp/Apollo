"""LoRA projects: dataset assembly, captions, and the trained-LoRA library.

On-disk layout (everything under ``LORA_DATA_DIR``)::

    data/loras/<project_id>/
        project.json
        dataset/   image001.png + image001.txt  (AI Toolkit's expected pairing)
        config/    training.yml
        output/    <name>.safetensors
"""

from __future__ import annotations

import base64
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.clients.ollama_client import OllamaClient
from app.core.config import settings
from app.core.exceptions import LoRAProjectError, OllamaServiceError
from app.core.paths import (
    IMAGE_EXTENSIONS,
    is_safetensors,
    read_json,
    safe_join,
    validate_image_upload,
    write_json_atomic,
)

logger = logging.getLogger(__name__)

CAPTION_PROMPT = (
    "Write a single-line image caption for training a LoRA model. Describe the "
    "subject, clothing, pose, setting, lighting and camera framing as a comma-"
    "separated list of short phrases. Do not use full sentences, do not add "
    "commentary, and do not start with 'a photo of'. Reply with the caption only."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LoRAProject:
    """Metadata for one LoRA training project."""

    id: str
    name: str
    description: str = ""
    created_at: str = field(default_factory=_now)
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    arch: str = "sdxl"
    trigger_word: str = ""
    training_status: str = "not_started"
    trained_lora_path: str | None = None
    training_job_id: str | None = None
    training_config: dict[str, Any] = field(default_factory=dict)
    edited_captions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetImage:
    id: str
    filename: str
    caption: str
    caption_edited: bool
    size_bytes: int


class LoRAProjectService:
    """Create projects, manage their datasets, and discover trained LoRAs."""

    def __init__(
        self,
        data_dir: str | None = None,
        ollama_client: OllamaClient | None = None,
    ) -> None:
        self.data_dir = Path(data_dir or settings.lora_data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.ollama_client = ollama_client or OllamaClient()

    # ---------- paths ----------

    def project_dir(self, project_id: str) -> Path:
        path = safe_join(self.data_dir, project_id)
        if not path.is_dir():
            raise LoRAProjectError(f"LoRA project '{project_id}' not found", status_code=404)
        return path

    def dataset_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "dataset"

    def output_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "output"

    def config_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "config"

    # ---------- projects ----------

    def create_project(
        self,
        name: str,
        description: str = "",
        base_model: str | None = None,
        arch: str = "sdxl",
        trigger_word: str = "",
    ) -> LoRAProject:
        if not name.strip():
            raise LoRAProjectError("Project name is required")

        slug = re.sub(r"[^a-z0-9_-]", "_", name.strip().lower()).strip("_")[:40] or "lora"
        project_id = f"{slug}_{uuid.uuid4().hex[:6]}"

        project = LoRAProject(
            id=project_id,
            name=name.strip(),
            description=description,
            arch=arch,
            trigger_word=trigger_word.strip(),
            **({"base_model": base_model} if base_model else {}),
        )

        root = safe_join(self.data_dir, project_id)
        for sub in ("dataset", "config", "output"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        self._save_project(project)

        logger.info("[LORA] Created project '%s' (%s)", project.name, project_id)
        return project

    def _save_project(self, project: LoRAProject) -> None:
        write_json_atomic(
            safe_join(self.data_dir, project.id, "project.json"), project.to_dict()
        )

    def get_project(self, project_id: str) -> LoRAProject:
        path = self.project_dir(project_id) / "project.json"
        data = read_json(path)
        if not isinstance(data, dict):
            raise LoRAProjectError(f"Project '{project_id}' metadata is unreadable", status_code=404)
        known = {f for f in LoRAProject.__dataclass_fields__}
        return LoRAProject(**{k: v for k, v in data.items() if k in known})

    def list_projects(self) -> list[LoRAProject]:
        projects: list[LoRAProject] = []
        for entry in sorted(self.data_dir.iterdir()) if self.data_dir.is_dir() else []:
            if not (entry / "project.json").is_file():
                continue
            try:
                projects.append(self.get_project(entry.name))
            except LoRAProjectError:
                logger.warning("[LORA] Skipping unreadable project %s", entry.name)
        projects.sort(key=lambda p: p.created_at, reverse=True)
        return projects

    def update_project(self, project_id: str, **fields: Any) -> LoRAProject:
        project = self.get_project(project_id)
        for key, value in fields.items():
            if value is not None and hasattr(project, key):
                setattr(project, key, value)
        self._save_project(project)
        return project

    def delete_project(self, project_id: str) -> bool:
        import shutil

        root = safe_join(self.data_dir, project_id)
        if not root.is_dir():
            return False
        shutil.rmtree(root)
        logger.info("[LORA] Deleted project %s", project_id)
        return True

    # ---------- dataset ----------

    def add_image(self, project_id: str, filename: str, content: bytes) -> DatasetImage:
        """Store one training image plus an empty caption file beside it."""
        dataset = self.dataset_dir(project_id)
        safe_name = validate_image_upload(
            filename, len(content), settings.max_image_upload_bytes
        )

        target = safe_join(dataset, safe_name)
        # Never silently clobber an existing image.
        stem, suffix = target.stem, target.suffix
        counter = 1
        while target.exists():
            target = safe_join(dataset, f"{stem}_{counter}{suffix}")
            counter += 1

        target.write_bytes(content)
        caption_path = target.with_suffix(".txt")
        if not caption_path.exists():
            caption_path.write_text("", encoding="utf-8")

        logger.info("[DATASET] %s: added %s (%d bytes)", project_id, target.name, len(content))
        return DatasetImage(
            id=target.stem,
            filename=target.name,
            caption="",
            caption_edited=False,
            size_bytes=len(content),
        )

    def _image_path(self, project_id: str, image_id: str) -> Path:
        dataset = self.dataset_dir(project_id)
        for extension in IMAGE_EXTENSIONS:
            candidate = safe_join(dataset, f"{image_id}{extension}")
            if candidate.is_file():
                return candidate
        raise LoRAProjectError(f"Image '{image_id}' not found", status_code=404)

    def list_images(self, project_id: str) -> list[DatasetImage]:
        dataset = self.dataset_dir(project_id)
        project = self.get_project(project_id)
        edited = set(project.edited_captions)

        images: list[DatasetImage] = []
        for path in sorted(dataset.iterdir()) if dataset.is_dir() else []:
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            caption_path = path.with_suffix(".txt")
            caption = (
                caption_path.read_text(encoding="utf-8").strip()
                if caption_path.is_file()
                else ""
            )
            images.append(
                DatasetImage(
                    id=path.stem,
                    filename=path.name,
                    caption=caption,
                    caption_edited=path.stem in edited,
                    size_bytes=path.stat().st_size,
                )
            )
        return images

    def read_image(self, project_id: str, image_id: str) -> tuple[bytes, str]:
        path = self._image_path(project_id, image_id)
        media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return path.read_bytes(), media_type

    def delete_image(self, project_id: str, image_id: str) -> bool:
        try:
            path = self._image_path(project_id, image_id)
        except LoRAProjectError:
            return False
        path.unlink()
        path.with_suffix(".txt").unlink(missing_ok=True)

        project = self.get_project(project_id)
        if image_id in project.edited_captions:
            project.edited_captions.remove(image_id)
            self._save_project(project)
        logger.info("[DATASET] %s: deleted %s", project_id, image_id)
        return True

    def set_caption(self, project_id: str, image_id: str, caption: str) -> DatasetImage:
        """Write a caption and mark it as human-edited so AI runs won't clobber it."""
        path = self._image_path(project_id, image_id)
        path.with_suffix(".txt").write_text(caption.strip(), encoding="utf-8")

        project = self.get_project(project_id)
        if image_id not in project.edited_captions:
            project.edited_captions.append(image_id)
            self._save_project(project)

        return DatasetImage(
            id=image_id,
            filename=path.name,
            caption=caption.strip(),
            caption_edited=True,
            size_bytes=path.stat().st_size,
        )

    def _write_caption_unmarked(self, project_id: str, image_id: str, caption: str) -> None:
        """Write a caption without marking it edited (used by AI generation)."""
        self._image_path(project_id, image_id).with_suffix(".txt").write_text(
            caption.strip(), encoding="utf-8"
        )

    # ---------- dataset quality ----------

    def validate_dataset(self, project_id: str) -> dict[str, Any]:
        """Run quality control over this project's dataset.

        Returns the report as a plain dict (see
        :mod:`app.services.dataset_validation` for the shape).
        """
        from app.services.dataset_validation import validate_dataset

        self.get_project(project_id)  # 404 for unknown projects
        images = [vars(image) for image in self.list_images(project_id)]
        report = validate_dataset(images, lambda image_id: self.read_image(project_id, image_id)[0])
        return report.to_dict()

    # ---------- AI captions ----------

    def generate_captions(
        self,
        project_id: str,
        model: str | None = None,
        overwrite: bool = False,
    ) -> list[dict[str, Any]]:
        """Caption dataset images with a local Ollama vision model.

        Manually edited captions are skipped unless ``overwrite`` is set, so a
        bulk run can never silently destroy the user's own wording.
        """
        project = self.get_project(project_id)
        vision_model = model or settings.vision_model
        images = self.list_images(project_id)
        if not images:
            raise LoRAProjectError("This project has no images to caption")

        prompt = CAPTION_PROMPT
        if project.trigger_word:
            prompt += (
                f" The subject's identifier is '{project.trigger_word}'; do not "
                "include it in the caption, it is added automatically."
            )

        results: list[dict[str, Any]] = []
        for image in images:
            if image.caption_edited and not overwrite:
                results.append(
                    {
                        "image_id": image.id,
                        "caption": image.caption,
                        "skipped": True,
                        "reason": "manually edited",
                    }
                )
                continue

            content, _ = self.read_image(project_id, image.id)
            encoded = base64.b64encode(content).decode("ascii")
            try:
                caption = self.ollama_client.generate_with_images(
                    model=vision_model,
                    prompt=prompt,
                    images=[encoded],
                    timeout=settings.vision_timeout,
                )
            except OllamaServiceError as exc:
                logger.warning("[LORA] Caption failed for %s: %s", image.id, exc.detail)
                results.append(
                    {"image_id": image.id, "caption": image.caption,
                     "skipped": True, "reason": exc.detail}
                )
                continue

            caption = _clean_caption(caption)
            if project.trigger_word and project.trigger_word.lower() not in caption.lower():
                caption = f"{project.trigger_word}, {caption}"

            # Saved but NOT marked edited — the user still reviews and confirms.
            self._write_caption_unmarked(project_id, image.id, caption)
            results.append({"image_id": image.id, "caption": caption, "skipped": False})

        logger.info(
            "[LORA] %s: captioned %d/%d images with %s",
            project_id,
            sum(1 for r in results if not r["skipped"]),
            len(results),
            vision_model,
        )
        return results

    def generate_captions_batch(
        self,
        project_id: str,
        progress_cb: Any = None,
        cancel_check: Any = None,
        model: str | None = None,
        overwrite: bool = False,
    ) -> list[dict[str, Any]]:
        """Job-friendly variant of :meth:`generate_captions`.

        Same behaviour, plus optional ``progress_cb(done, total)`` after each
        image and cooperative cancellation via ``cancel_check() -> bool`` so
        long runs can be watched and stopped from the job monitor instead of
        blocking one HTTP request for minutes.
        """
        project = self.get_project(project_id)
        vision_model = model or settings.vision_model
        images = self.list_images(project_id)
        if not images:
            raise LoRAProjectError("This project has no images to caption")

        prompt = CAPTION_PROMPT
        if project.trigger_word:
            prompt += (
                f" The subject's identifier is '{project.trigger_word}'; do not "
                "include it in the caption, it is added automatically."
            )

        results: list[dict[str, Any]] = []
        for index, image in enumerate(images):
            if cancel_check and cancel_check():
                logger.info("[LORA] %s: captioning cancelled at %d/%d",
                            project_id, index, len(images))
                break

            if image.caption_edited and not overwrite:
                results.append(
                    {
                        "image_id": image.id,
                        "caption": image.caption,
                        "skipped": True,
                        "reason": "manually edited",
                    }
                )
                continue

            content, _ = self.read_image(project_id, image.id)
            encoded = base64.b64encode(content).decode("ascii")
            try:
                caption = self.ollama_client.generate_with_images(
                    model=vision_model,
                    prompt=prompt,
                    images=[encoded],
                    timeout=settings.vision_timeout,
                )
            except OllamaServiceError as exc:
                logger.warning("[LORA] Caption failed for %s: %s", image.id, exc.detail)
                results.append(
                    {"image_id": image.id, "caption": image.caption,
                     "skipped": True, "reason": exc.detail}
                )
                continue

            caption = _clean_caption(caption)
            if project.trigger_word and project.trigger_word.lower() not in caption.lower():
                caption = f"{project.trigger_word}, {caption}"

            self._write_caption_unmarked(project_id, image.id, caption)
            results.append({"image_id": image.id, "caption": caption, "skipped": False})

            if progress_cb:
                progress_cb(index + 1, len(images))

        return results

    # ---------- trained LoRA library ----------

    def list_library(self) -> list[dict[str, Any]]:
        """All discovered LoRA files: project outputs plus ComfyUI's own folder.

        Files are verified by reading the safetensors header, so a renamed
        ``.safetensors`` that is not one does not enter the library.
        """
        entries: list[dict[str, Any]] = []
        seen: set[str] = set()

        for project in self.list_projects():
            output = safe_join(self.data_dir, project.id, "output")
            for path in sorted(output.rglob("*.safetensors")) if output.is_dir() else []:
                if not is_safetensors(path):
                    logger.warning("[LORA] %s is not a valid safetensors file", path.name)
                    continue
                seen.add(path.name)
                entries.append(_library_entry(path, project))

        comfy_dir = settings.comfyui_lora_dir
        if comfy_dir and Path(comfy_dir).is_dir():
            for path in sorted(Path(comfy_dir).rglob("*.safetensors")):
                if path.name in seen or not is_safetensors(path):
                    continue
                entries.append(_library_entry(path, None, source="comfyui"))

        return entries


def _library_entry(
    path: Path, project: LoRAProject | None, source: str = "project"
) -> dict[str, Any]:
    return {
        "name": path.stem,
        "filename": path.name,
        "path": str(path),
        "source": source,
        "project_id": project.id if project else None,
        "project_name": project.name if project else None,
        "base_model": project.base_model if project else None,
        "arch": project.arch if project else None,
        "trigger_word": project.trigger_word if project else None,
        "training_status": project.training_status if project else "external",
        "size_bytes": path.stat().st_size,
        "created_at": datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
    }


def _clean_caption(text: str) -> str:
    """Strip the boilerplate vision models like to wrap captions in."""
    caption = " ".join(text.strip().split())
    caption = re.sub(r'^(here is |this is |the image shows |sure[,!] )', "", caption, flags=re.I)
    return caption.strip().strip('"').strip()
