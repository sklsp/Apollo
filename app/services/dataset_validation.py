"""Dataset quality control: validation, duplicate detection, quality score.

Runs entirely on bytes already on disk — no network calls — so it is safe to
run from a background job or synchronously for small datasets.

Checks performed per image:

* decodable by Pillow (catches truncated / corrupted files)
* extreme resolutions (too small to train on, or absurdly large)
* inconsistent dimensions (informational — AI Toolkit buckets resolutions)
* exact duplicates (SHA-256 of decoded pixels, so a re-encoded copy matches)
* near-duplicates (multi-signal, see below)
* missing / empty / suspicious captions

Near-duplicate classification uses several signals combined, because a single
perceptual hash produces false positives on solid-color images and other
low-entropy content:

* **exact**       — identical decoded pixels (SHA-256)
* **high**        — small perceptual distance AND similar dimensions AND
                    (for low-entropy images) matching color statistics
* **possible**    — small perceptual distance but differing structure

The classifier returns ``exact_duplicate``, ``near_duplicate_high``,
``near_duplicate_possible`` or nothing (unique). Solid-color and other flat
images are only ever flagged when their mean colors also match closely.

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

# Perceptual-hash Hamming distance bands (out of 64 bits).
NEAR_HASH_EXACT = 0        # 0      -> visually identical
NEAR_HASH_CLOSE = 6        # 1..6   -> candidate near-duplicate
# Anything above NEAR_HASH_CLOSE is treated as unique by the hash signal.

# Aspect-ratio tolerance for "same shape" (resized copies keep the ratio).
ASPECT_RATIO_TOLERANCE = 0.05

# Mean-color distance (0-255 scale) below which two flat/low-entropy images
# count as the same picture rather than merely hash-similar.
FLAT_COLOR_DISTANCE = 12

# An image is "low entropy" (flat, gradient, simple) when this fraction of its
# 8x8 aHash bits sit on one side of the mean. Such images carry almost no
# structure, so their hashes match trivially and need corroborating signals.
LOW_ENTROPY_BITS = 4  # out of 64: nearly all bits identical

# Score weights (points subtracted per finding, capped per class).
WEIGHTS = {
    "invalid": 15,
    "duplicate": 8,
    "near_duplicate": 5,
    "near_duplicate_possible": 2,
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
            "possible_duplicates": sum(
                1 for f in self.findings if f.kind == "near_duplicate_possible"
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


def _mean_color(image: Any) -> tuple[float, float, float]:
    """Average RGB color, used to corroborate matches on flat images."""
    small = image.convert("RGB").resize((1, 1))
    return small.getpixel((0, 0))


def _color_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def _is_low_entropy(hash_hex: str) -> bool:
    """True when an aHash is nearly all-0 or all-1 (flat/gradient images).

    Such hashes carry no structure, so hash similarity alone is meaningless
    for them — two unrelated solid-color photos would "match".
    """
    bits = int(hash_hex, 16)
    ones = bin(bits).count("1")
    return ones <= LOW_ENTROPY_BITS or ones >= 64 - LOW_ENTROPY_BITS


def _same_aspect_ratio(w_a: int, h_a: int, w_b: int, h_b: int) -> bool:
    ratio_a = w_a / max(h_a, 1)
    ratio_b = w_b / max(h_b, 1)
    if ratio_a == 0 or ratio_b == 0:
        return False
    return abs(ratio_a - ratio_b) / max(ratio_a, ratio_b) <= ASPECT_RATIO_TOLERANCE


def classify_pair(
    hash_a: str,
    hash_b: str,
    *,
    dims_a: tuple[int, int] | None = None,
    dims_b: tuple[int, int] | None = None,
    color_a: tuple[float, float, float] | None = None,
    color_b: tuple[float, float, float] | None = None,
) -> str:
    """Classify how similar two images are, using every available signal.

    Returns one of:

    * ``"near_duplicate_high"``      — hash distance <= NEAR_HASH_CLOSE with
      corroborating signals (same aspect ratio; flat images also need the
      same mean color)
    * ``"near_duplicate_possible"``  — hash-similar but signals disagree
      (e.g. a crop or a reframe); surfaced for human review, never auto-merged
    * ``"unique"``

    Pixel-identical copies are caught earlier by SHA-256 and reported as
    plain ``duplicate``.
    """
    distance = _hamming(hash_a, hash_b)

    if distance == 0:
        # Identical structure. For flat images the hash is trivially equal for
        # any same-brightness pair, so require the mean color to agree too.
        if _is_low_entropy(hash_a) and color_a is not None and color_b is not None:
            if _color_distance(color_a, color_b) > FLAT_COLOR_DISTANCE:
                return "unique"
        return "near_duplicate_high"

    if distance > NEAR_HASH_CLOSE:
        return "unique"

    # Hash says "close". Corroborate with structure signals when available.
    corroboration = 0
    required = 0

    if dims_a and dims_b:
        required += 1
        if _same_aspect_ratio(*dims_a, *dims_b):
            corroboration += 1

    if _is_low_entropy(hash_a) or _is_low_entropy(hash_b):
        # Flat images: hash distance is noise, color must match.
        if color_a is not None and color_b is not None:
            required += 1
            if _color_distance(color_a, color_b) <= FLAT_COLOR_DISTANCE:
                corroboration += 1

    if required and corroboration < required:
        return "near_duplicate_possible"
    return "near_duplicate_high"


def _mean_color(image: Any) -> tuple[float, float, float]:
    """Average RGB color, used to corroborate matches on flat images."""
    small = image.convert("RGB").resize((1, 1))
    return small.getpixel((0, 0))


def _color_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def _is_low_entropy(hash_hex: str) -> bool:
    """True when an aHash is nearly all-0 or all-1 (flat/gradient images).

    Such hashes carry no structure, so hash similarity alone is meaningless
    for them — two unrelated solid-color photos would "match".
    """
    bits = int(hash_hex, 16)
    ones = bin(bits).count("1")
    return ones <= LOW_ENTROPY_BITS or ones >= 64 - LOW_ENTROPY_BITS


def _same_aspect_ratio(w_a: int, h_a: int, w_b: int, h_b: int) -> bool:
    ratio_a = w_a / max(h_a, 1)
    ratio_b = w_b / max(h_b, 1)
    if ratio_a == 0 or ratio_b == 0:
        return False
    return abs(ratio_a - ratio_b) / max(ratio_a, ratio_b) <= ASPECT_RATIO_TOLERANCE


def classify_pair(
    hash_a: str,
    hash_b: str,
    *,
    dims_a: tuple[int, int] | None = None,
    dims_b: tuple[int, int] | None = None,
    color_a: tuple[float, float, float] | None = None,
    color_b: tuple[float, float, float] | None = None,
) -> str:
    """Classify how similar two images are, using every available signal.

    Returns one of:

    * ``"exact_duplicate"``          — identical perceptual hash AND identical
      mean color (pixel-identical copies are caught earlier by SHA-256)
    * ``"near_duplicate_high"``      — hash distance <= NEAR_HASH_CLOSE with
      corroborating signals (same aspect ratio; flat images also need the
      same mean color)
    * ``"near_duplicate_possible"``  — hash-similar but signals disagree
      (e.g. a crop or a reframe); surfaced for human review, never auto-merged
    * ``"unique"``
    """
    distance = _hamming(hash_a, hash_b)

    if distance == 0:
        # Identical structure. For flat images the hash is trivially equal for
        # any same-brightness pair, so require the mean color to agree too.
        if _is_low_entropy(hash_a) and color_a is not None and color_b is not None:
            if _color_distance(color_a, color_b) > FLAT_COLOR_DISTANCE:
                return "unique"
        return "near_duplicate_high"

    if distance > NEAR_HASH_CLOSE:
        return "unique"

    # Hash says "close". Corroborate with structure signals when available.
    corroboration = 0
    required = 0

    if dims_a and dims_b:
        required += 1
        if _same_aspect_ratio(*dims_a, *dims_b):
            corroboration += 1

    if _is_low_entropy(hash_a) or _is_low_entropy(hash_b):
        # Flat images: hash distance is noise, color must match.
        if color_a is not None and color_b is not None:
            required += 1
            if _color_distance(color_a, color_b) <= FLAT_COLOR_DISTANCE:
                corroboration += 1

    if required and corroboration < required:
        return "near_duplicate_possible"
    return "near_duplicate_high"


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
    # Per-image fingerprint for near-duplicate classification.
    fingerprints: list[dict[str, Any]] = []

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

        fingerprints.append({
            "id": image_id,
            "hash": _perceptual_hash(image),
            "dims": (width, height),
            "color": _mean_color(image),
        })

    # ---- near-duplicates: multi-signal classification, O(n^2) on
    # fingerprints only — cheap for realistic dataset sizes. ----
    seen_pairs: set[tuple[str, str]] = set()
    for i, fp_a in enumerate(fingerprints):
        for fp_b in fingerprints[i + 1:]:
            verdict = classify_pair(
                fp_a["hash"], fp_b["hash"],
                dims_a=fp_a["dims"], dims_b=fp_b["dims"],
                color_a=fp_a["color"], color_b=fp_b["color"],
            )
            if verdict == "unique":
                continue

            pair = tuple(sorted((fp_a["id"], fp_b["id"])))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            # Only flag the second one, so deleting flagged images fixes
            # the report rather than cascading.
            later_id = pair[1]
            kind = ("near_duplicate" if verdict == "near_duplicate_high"
                    else "near_duplicate_possible")
            report.findings.append(
                ImageFinding(later_id, _filename_of(images, later_id), kind,
                             f"{verdict.replace('_', ' ')} vs {pair[0]}")
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
