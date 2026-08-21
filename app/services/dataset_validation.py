"""Dataset quality control: validation, duplicate detection, quality score.

Runs entirely on bytes already on disk — no network calls — so it is safe to
run from a background job or synchronously for small datasets.

Checks performed per image:

* decodable by Pillow (catches truncated / corrupted files)
* extreme resolutions (too small to train on, or absurdly large)
* inconsistent dimensions (informational — AI Toolkit buckets resolutions)
* exact duplicates (SHA-256 of decoded pixels, so a re-encoded copy matches)
* near-duplicates (perceptual hash, Hamming distance <= threshold)
* missing / empty / suspicious captions

The quality score is a simple weighted penalty model, chosen so the numbers
behave intuitively: a clean dataset scores 100, each problem class subtracts a
bounded amount, and the score never goes below zero.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Resolutions outside these bounds are flagged. 256px is the practical floor
# for a LoRA crop; 8192 catches accidental scans or phone panoramas.
MIN_DIMENSION = 256
MAX_DIMENSION = 8192

# Perceptual-hash Hamming distance at or below which two images are
# "near-duplicates". 0 = identical-looking, 10+ = clearly different.
NEAR_DUPLICATE_DISTANCE = 5

# Score weights (points subtracted per finding, capped per class).
WEIGHTS = {
    "invalid": 15,
    "duplicate": 8,
    "near_duplicate": 4,
    "extreme_resolution": 6,
    "missing_caption": 5,
    "empty_caption": 5,
    "suspicious_caption": 3,
    "dimension_inconsistency": 2,
}


@dataclass
class ImageFinding:
    """One problem found on one image."""

    image_id: str
    filename: str
    kind: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "image_id": self.image_id,
            "filename": self.filename,
            "kind": self.kind,
            "detail": self.detail,
        }


@dataclass
class DatasetReport:
    """Aggregated quality report for one dataset."""

    total_images: int = 0
    valid_images: int = 0
    captioned: int = 0
    findings: list[ImageFinding] = field(default_factory=list)
    average_resolution: tuple[int, int] | None = None
    score: int = 100

    @property
    def issue_count(self) -> int:
        return len(self.findings)

    def to_dict(self) -> dict[str, Any]:
        avg = self.average_resolution
        return {
            "total_images": self.total_images,
            "valid_images": self.valid_images,
            "invalid_images": self.total_images - self.valid_images,
            "captioned": self.captioned,
            "missing_captions": sum(
                1 for f in self.findings if f.kind == "missing_caption"
            ),
            "duplicates": sum(1 for f in self.findings if f.kind == "duplicate"),
            "near_duplicates": sum(
                1 for f in self.findings if f.kind == "near_duplicate"
            ),
            "extreme_resolutions": sum(
                1 for f in self.findings if f.kind == "extreme_resolution"
            ),
            "average_resolution": list(avg) if avg else None,
            "issue_count": self.issue_count,
            "score": self.score,
            "findings": [f.to_dict() for f in self.findings],
        }


def _perceptual_hash(image: Any) -> str:
    """8x8 average-hash (aHash). Returns a 64-bit hex string.

    Deliberately dependency-light: Pillow resize + mean threshold. Not as
    accurate as pHash but catches re-encodes, crops-with-border and resizes,
    which is what duplicate training images actually look like.
    """
    gray = image.convert("L").resize((8, 8))
    pixels = list(gray.getdata())
    mean = sum(pixels) / len(pixels)
    bits = 0
    for pixel in pixels:
        bits = (bits << 1) | (1 if pixel > mean else 0)
    return f"{bits:016x}"


def _hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _pixel_sha256(image: Any) -> str:
    """Hash of decoded pixels, so a PNG and its JPEG re-encode match."""
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def validate_dataset(
    images: list[dict[str, Any]],
    read_bytes,
) -> DatasetReport:
    """Validate a dataset.

    Args:
        images: ``DatasetImage``-shaped dicts with at least
            ``id``, ``filename``, ``caption``.
        read_bytes: Callable ``(image_id) -> bytes`` used to load image data.
            Kept injectable so tests can feed synthetic images without a
            project on disk.

    Returns:
        A :class:`DatasetReport` with per-image findings and a 0-100 score.
    """
    from io import BytesIO

    from PIL import Image

    report = DatasetReport(total_images=len(images))
    widths: list[int] = []
    heights: list[int] = []
    exact: dict[str, str] = {}  # pixel hash -> image_id of first occurrence
    phashes: list[tuple[str, str]] = []  # (image_id, ahash)

    for entry in images:
        image_id = entry["id"]
        filename = entry.get("filename", image_id)
        caption = (entry.get("caption") or "").strip()

        # ---- captions ----
        if not caption:
            report.findings.append(
                ImageFinding(image_id, filename, "missing_caption")
            )
        elif len(caption) < 3:
            report.findings.append(
                ImageFinding(image_id, filename, "empty_caption", caption)
            )
        elif _looks_suspicious(caption):
            report.findings.append(
                ImageFinding(image_id, filename, "suspicious_caption",
                             caption[:80])
            )
        else:
            report.captioned += 1

        # ---- image decoding ----
        try:
            content = read_bytes(image_id)
            with Image.open(BytesIO(content)) as image:
                image.load()
                width, height = image.size
        except Exception as exc:  # noqa: BLE001 - any decode failure is "invalid"
            logger.info("[DATASET] %s failed to decode: %s", filename, exc)
            report.findings.append(
                ImageFinding(image_id, filename, "invalid", str(exc)[:120])
            )
            continue

        report.valid_images += 1
        widths.append(width)
        heights.append(height)

        if width < MIN_DIMENSION or height < MIN_DIMENSION:
            report.findings.append(
                ImageFinding(image_id, filename, "extreme_resolution",
                             f"{width}x{height} below {MIN_DIMENSION}px minimum")
            )
        elif width > MAX_DIMENSION or height > MAX_DIMENSION:
            report.findings.append(
                ImageFinding(image_id, filename, "extreme_resolution",
                             f"{width}x{height} above {MAX_DIMENSION}px maximum")
            )

        # ---- duplicates ----
        pixel_hash = _pixel_sha256(image)
        if pixel_hash in exact:
            report.findings.append(
                ImageFinding(image_id, filename, "duplicate",
                             f"identical to {exact[pixel_hash]}")
            )
        else:
            exact[pixel_hash] = image_id

        phashes.append((image_id, _perceptual_hash(image)))

    # ---- near-duplicates (O(n^2) on hashes only — cheap) ----
    seen_pairs: set[tuple[str, str]] = set()
    for i, (id_a, hash_a) in enumerate(phashes):
        for id_b, hash_b in phashes[i + 1:]:
            if _hamming(hash_a, hash_b) <= NEAR_DUPLICATE_DISTANCE:
                pair = tuple(sorted((id_a, id_b)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                # Only flag the second one, so deleting flagged images fixes
                # the report rather than cascading.
                later = id_b if pair[1] == id_b else id_a
                report.findings.append(
                    ImageFinding(later, _filename_of(images, later),
                                 "near_duplicate", f"very similar to {pair[0]}")
                )

    # ---- dimension consistency (informational) ----
    if widths and len(set(zip(widths, heights))) > 1:
        distinct = len(set(zip(widths, heights)))
        report.findings.append(
            ImageFinding("*", "(dataset)", "dimension_inconsistency",
                         f"{distinct} distinct dimensions; AI Toolkit will "
                         "bucket them")
        )

    if widths:
        report.average_resolution = (
            round(sum(widths) / len(widths)),
            round(sum(heights) / len(heights)),
        )

    report.score = _score(report)
    return report


def _filename_of(images: list[dict[str, Any]], image_id: str) -> str:
    for entry in images:
        if entry["id"] == image_id:
            return entry.get("filename", image_id)
    return image_id


def _looks_suspicious(caption: str) -> bool:
    """Captions that suggest the vision model failed or leaked instructions."""
    lowered = caption.lower()
    markers = (
        "as an ai",
        "i cannot",
        "i can't",
        "sorry",
        "lorem ipsum",
        "undefined",
        "null,",
        "[error",
    )
    return any(marker in lowered for marker in markers)


def _score(report: DatasetReport) -> int:
    """Weighted penalty score, floored at 0."""
    counts: dict[str, int] = {}
    for finding in report.findings:
        counts[finding.kind] = counts.get(finding.kind, 0) + 1

    penalty = 0
    for kind, count in counts.items():
        weight = WEIGHTS.get(kind, 2)
        # Cap each class so one systematic problem cannot zero the score.
        penalty += min(count, 10) * weight

    return max(0, 100 - penalty)
