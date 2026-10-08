"""Training preflight: validate a config against this machine before launching.

The advisor classifies configurations into four honest buckets:

* ``ok``        — likely safe
* ``heavy``     — potentially heavy; may work depending on the model
* ``risky``     — high risk of OOM or extreme runtime
* ``unsupported`` — cannot run here at all (e.g. no accelerator configured)

It never modifies the user's configuration and never guarantees a fit — VRAM
use depends on the base model, sequence caching and AI Toolkit internals that
no external estimator can see. The raw error from a failed run is always kept
alongside the translated explanation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.services.hardware import HardwareInfo, detect_hardware, detect_disk_free_gb

logger = logging.getLogger(__name__)

# Rough VRAM pressure model for LoRA fine-tuning (rank <= 64, adamw8bit).
# These are heuristics calibrated against AI Toolkit's documented examples,
# not guarantees — their purpose is ranking configs, not predicting exact use.
VRAM_BASE_MB = {          # per resolution bucket, batch 1
    "low":    6 * 1024,   # <= 512px
    "medium": 9 * 1024,   # <= 768px
    "high":   14 * 1024,  # <= 1024px
    "extreme": 22 * 1024, # > 1024px
}
VRAM_PER_BATCH_MB = 1200  # each extra batch slot costs roughly this much


@dataclass
class PreflightCheck:
    """One line of the preflight report."""

    name: str
    passed: bool
    detail: str = ""
    severity: str = "info"  # info | warning | error

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "severity": self.severity,
        }


@dataclass
class PreflightReport:
    """Structured result of checking one training configuration."""

    checks: list[PreflightCheck] = field(default_factory=list)
    verdict: str = "ok"           # ok | heavy | risky | unsupported | blocked
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)
    estimated_vram_mb: int | None = None
    hardware: dict[str, Any] | None = None

    @property
    def can_proceed(self) -> bool:
        """False only when starting would certainly fail (no GPU, no disk)."""
        return self.verdict not in ("blocked", "unsupported")

    def to_dict(self) -> dict[str, Any]:
        return {
            "checks": [check.to_dict() for check in self.checks],
            "verdict": self.verdict,
            "summary": self.summary,
            "recommendations": self.recommendations,
            "estimated_vram_mb": self.estimated_vram_mb,
            "hardware": self.hardware,
            "can_proceed": self.can_proceed,
        }


def _resolution_bucket(resolution: list[int] | None) -> tuple[str, int]:
    """Map a resolution list to (bucket, max_px)."""
    if not resolution:
        return "high", 1024
    largest = max(resolution)
    if largest <= 512:
        return "low", largest
    if largest <= 768:
        return "medium", largest
    if largest <= 1024:
        return "high", largest
    return "extreme", largest


def estimate_vram_mb(options: dict[str, Any]) -> int:
    """Rough VRAM estimate for a training config. Heuristic, not a promise."""
    bucket, _ = _resolution_bucket(options.get("resolution"))
    estimate = VRAM_BASE_MB[bucket]
    batch = int(options.get("batch_size", 1))
    rank = int(options.get("lora_rank", 16))
    estimate += max(0, batch - 1) * VRAM_PER_BATCH_MB
    # Higher rank grows optimizer state; sublinear in practice with 8-bit Adam.
    estimate += int((max(rank, 16) - 16) / 16 * 150)
    return estimate


def advise(
    options: dict[str, Any],
    hardware: HardwareInfo | None = None,
) -> PreflightReport:
    """Evaluate a training configuration against detected hardware."""
    hardware = hardware or detect_hardware()
    report = PreflightReport(hardware=hardware.to_dict())

    gpu = hardware.primary_gpu
    vram_total = hardware.vram_total_mb

    # ---- accelerator ----
    if hardware.accelerator == "cpu" or (gpu is None):
        report.checks.append(PreflightCheck(
            name="Accelerator",
            passed=False,
            detail="No GPU detected. Training on CPU is impractically slow "
                   "for SDXL/Flux-class models.",
            severity="error",
        ))
        report.verdict = "unsupported"
        report.summary = "No usable GPU detected: training is not practical on this machine."
        report.recommendations.append(
            "Install/enable GPU drivers, or run training on a different machine."
        )
        return report

    report.checks.append(PreflightCheck(
        name="Accelerator",
        passed=True,
        detail=f"{gpu.name} ({gpu.vendor}) via {gpu.source}",
    ))

    # ---- VRAM pressure ----
    estimate = estimate_vram_mb(options)
    report.estimated_vram_mb = estimate

    if vram_total:
        ratio = estimate / vram_total
        if ratio <= 0.75:
            report.checks.append(PreflightCheck(
                name="VRAM",
                passed=True,
                detail=f"~{estimate // 1024} GB estimated vs {vram_total // 1024} GB available",
            ))
        elif ratio <= 1.0:
            report.checks.append(PreflightCheck(
                name="VRAM",
                passed=True,
                detail=f"~{estimate // 1024} GB estimated vs {vram_total // 1024} GB available, tight",
                severity="warning",
            ))
            report.verdict = "heavy"
            report.recommendations.append(
                "Reduce batch size to 1, lower resolution, or enable gradient "
                "accumulation to trade speed for memory."
            )
        else:
            report.checks.append(PreflightCheck(
                name="VRAM",
                passed=False,
                detail=f"~{estimate // 1024} GB estimated vs {vram_total // 1024} GB available",
                severity="warning",
            ))
            report.verdict = "risky"
            report.recommendations.append(
                f"Estimated usage exceeds the GPU's {vram_total // 1024} GB. "
                "Cut resolution first (biggest lever), then batch size."
            )
    else:
        report.checks.append(PreflightCheck(
            name="VRAM",
            passed=True,
            detail=f"GPU found but VRAM size unknown; ~{estimate // 1024} GB will be needed",
            severity="warning",
        ))

    # ---- system RAM ----
    ram = hardware.ram_total_mb
    if ram and ram < 16 * 1024:
        report.checks.append(PreflightCheck(
            name="System RAM",
            passed=True,
            detail=f"{ram // 1024} GB total, below the 16 GB comfort zone for SDXL",
            severity="warning",
        ))
        if report.verdict == "ok":
            report.verdict = "heavy"
    elif ram:
        report.checks.append(PreflightCheck(
            name="System RAM", passed=True, detail=f"{ram // 1024} GB total",
        ))

    # ---- disk ----
    free_gb = detect_disk_free_gb()
    if free_gb is not None:
        if free_gb < 20:
            report.checks.append(PreflightCheck(
                name="Disk space",
                passed=False,
                detail=f"{free_gb} GB free; checkpoints need 5-15 GB",
                severity="error",
            ))
            report.verdict = "blocked"
            report.summary = "Not enough disk space for training checkpoints."
            return report
        report.checks.append(PreflightCheck(
            name="Disk space", passed=True, detail=f"{free_gb} GB free",
        ))

    # ---- parameter sanity ----
    steps = int(options.get("steps", 2000))
    lr = float(options.get("learning_rate", 1e-4))
    if lr > 5e-4:
        report.checks.append(PreflightCheck(
            name="Learning rate",
            passed=True,
            detail=f"{lr:g} is aggressive for LoRA; expect instability",
            severity="warning",
        ))
        if report.verdict == "ok":
            report.verdict = "heavy"
    elif lr < 1e-5:
        report.checks.append(PreflightCheck(
            name="Learning rate",
            passed=True,
            detail=f"{lr:g} is very low; training may barely move",
            severity="warning",
        ))
    else:
        report.checks.append(PreflightCheck(
            name="Learning rate", passed=True, detail=f"{lr:g}",
        ))

    if steps > 10000:
        report.checks.append(PreflightCheck(
            name="Steps",
            passed=True,
            detail=f"{steps} steps is a long run; consider checkpoints + early review",
            severity="warning",
        ))
    else:
        report.checks.append(PreflightCheck(
            name="Steps", passed=True, detail=f"{steps}",
        ))

    # ---- verdict summary ----
    if report.verdict == "ok":
        report.summary = "Configuration looks safe for this machine."
    elif report.verdict == "heavy":
        report.summary = "Training may be tight on resources but should fit."
    elif report.verdict == "risky":
        report.summary = "High risk of out-of-memory: reduce load before starting."
    return report


# ============================================
# Failure diagnostics
# ============================================

# Raw trainer errors -> human explanation. First matching pattern wins.
_FAILURE_PATTERNS: list[tuple[str, str, str]] = [
    # (regex, explanation, recommendation)
    (
        r"cuda(out of memory|_error)?[:\s]*out of memory|out of memory",
        "Training ran out of GPU memory.",
        "Reduce batch size to 1, lower the resolution, or shorten the max "
        "sequence length. The last successful checkpoint is still valid.",
    ),
    (
        r"no space left on device|disk full",
        "The disk holding the training output ran out of space.",
        "Free up space or change the output directory, then resume.",
    ),
    (
        r"modulenotfounderror|importerror",
        "A dependency is missing from the AI Toolkit environment.",
        "Install the missing package inside AI_TOOLKIT_PYTHON's environment.",
    ),
    (
        r"filenotfounderror|no such file",
        "A file the training config references was not found.",
        "Check the dataset folder and base model path exist.",
    ),
    (
        r"connectionerror|max retries|timeout",
        "A download (usually the base model) failed mid-run.",
        "Check your internet connection, or pre-download the model.",
    ),
    (
        r"outofmemoryerror|cannot allocate",
        "The system ran out of RAM.",
        "Close other applications, or reduce dataloader workers/batch size.",
    ),
]


def diagnose_failure(raw_error: str) -> dict[str, str]:
    """Translate a raw training error into an explanation.

    Returns ``{"explanation": ..., "recommendation": ..., "raw": ...}``.
    The raw text is always preserved for advanced users.
    """
    lowered = raw_error.lower()
    for pattern, explanation, recommendation in _FAILURE_PATTERNS:
        if re.search(pattern, lowered):
            return {
                "explanation": explanation,
                "recommendation": recommendation,
                "raw": raw_error[:2000],
            }
    return {
        "explanation": "Training failed. The log tail has the details.",
        "recommendation": "Check training.log; common causes are OOM, disk "
                          "space, or a bad model path.",
        "raw": raw_error[:2000],
    }
