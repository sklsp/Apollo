"""Filesystem safety helpers for user-supplied names and paths.

Every path built from browser input must go through :func:`safe_join`, and every
uploaded filename through :func:`sanitize_filename`. Nothing else in the app is
allowed to concatenate a user string onto a directory.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
LORA_EXTENSIONS = {".safetensors"}

# Anything outside this set is replaced with "_".
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
# Windows reserves these device names regardless of extension.
_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


class UnsafePathError(ValueError):
    """Raised when user input would escape its designated directory."""


def sanitize_filename(filename: str, *, default: str = "file") -> str:
    """Reduce an arbitrary upload name to a safe single path component.

    Strips any directory portion (POSIX *and* Windows separators), removes
    characters outside ``[A-Za-z0-9._-]``, and refuses Windows device names.
    """
    if not filename:
        return default

    # Take the last component under either separator convention, so
    # "../../etc/passwd" and "..\\..\\windows\\system32\\x" both collapse.
    name = re.split(r"[\\/]", filename)[-1]
    name = _UNSAFE.sub("_", name).strip("._ ")

    if not name:
        return default
    if Path(name).stem.upper() in _RESERVED:
        name = f"_{name}"
    # Leave room for suffixes callers may append.
    return name[:120]


def safe_join(root: str | Path, *parts: str) -> Path:
    """Join ``parts`` under ``root``, refusing anything that escapes it.

    Rejects absolute paths, drive letters, UNC prefixes, backslashes and ``..`` traversal —
    checked after resolution, so symlinks and ``a/../../b`` are caught too.
    """
    base = Path(root).resolve()

    for part in parts:
        if part is None or part == "":
            raise UnsafePathError("Empty path component")
        if os.path.isabs(part) or re.match(r"^[A-Za-z]:", part) or part.startswith("\\\\"):
            raise UnsafePathError(f"Absolute paths are not allowed: {part!r}")
        if "\\" in part:
            # a separator on Windows only; refused everywhere so a data folder stays safe on either OS
            raise UnsafePathError(f"Backslashes are not allowed: {part!r}")

    candidate = base.joinpath(*parts)
    # strict=False: the target may not exist yet (we are often creating it).
    resolved = Path(os.path.normpath(candidate)).resolve()

    if resolved != base and base not in resolved.parents:
        raise UnsafePathError(f"Path escapes its root directory: {'/'.join(parts)!r}")
    return resolved


def validate_image_upload(filename: str, size_bytes: int, max_bytes: int) -> str:
    """Validate an image upload and return its sanitized filename.

    Raises ``ValueError`` with a user-safe message on rejection.
    """
    safe = sanitize_filename(filename)
    extension = Path(safe).suffix.lower()

    if extension not in IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise ValueError(f"Unsupported image type '{extension or safe}'. Allowed: {allowed}")
    if size_bytes <= 0:
        raise ValueError("Uploaded file is empty")
    if size_bytes > max_bytes:
        raise ValueError(
            f"File is {size_bytes / 1024 / 1024:.1f} MB; the limit is "
            f"{max_bytes / 1024 / 1024:.0f} MB"
        )
    return safe


def is_safetensors(path: Path) -> bool:
    """Check the safetensors header instead of trusting the file extension.

    Format: 8-byte little-endian header length, then that many bytes of JSON.
    """
    try:
        with open(path, "rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) < 8:
                return False
            length = int.from_bytes(raw_length, "little")
            # A sane header is small; anything huge means this is not safetensors.
            if not 0 < length <= 100_000_000 or length > path.stat().st_size:
                return False
            header = json.loads(handle.read(length).decode("utf-8"))
        return isinstance(header, dict)
    except (OSError, ValueError, UnicodeDecodeError):
        return False


def write_json_atomic(path: str | Path, payload: object) -> None:
    """Write JSON via a temp file + replace, so a crash cannot truncate state.

    On Windows ``os.replace`` can fail with PermissionError when another thread
    replaces the same target at the same instant; retry briefly before giving up.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False, suffix=".tmp"
    )
    try:
        with handle:
            json.dump(payload, handle, indent=2, default=str)
        _replace_with_retry(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _replace_with_retry(source: str, target: Path, attempts: int = 10) -> None:
    """``os.replace`` with a short backoff for Windows replace races."""
    delay = 0.005
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(delay)
            delay *= 2


def read_json(path: str | Path, default: object = None) -> object:
    """Read JSON, returning ``default`` when the file is missing or corrupt.

    On Windows, opening a file that another thread is atomically replacing can
    transiently fail with Access Denied; retry briefly before giving up.
    """
    delay = 0.005
    for attempt in range(10):
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            return default
        except OSError:
            if attempt + 1 >= 10:
                return default
            time.sleep(delay)
            delay *= 2
        except json.JSONDecodeError:
            return default
    return default
