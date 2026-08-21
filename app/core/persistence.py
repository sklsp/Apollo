"""JSON-file persistence shared by the document and memory services.

Same philosophy as :mod:`app.core.jobs`: a local single-user app does not need
a database server. Atomic writes + load-on-start give restart survival without
a new moving part to run.
"""

from __future__ import annotations

import threading

from app.core.config import settings
from app.core.paths import read_json, write_json_atomic


class JsonPersist:
    """Tiny thread-safe JSON document store with debounced saves.

    ``save()`` is cheap enough to call after every mutation; writes are atomic
    so a crash mid-write cannot corrupt the file.
    """

    def __init__(self, filename: str, data_dir: str | None = None) -> None:
        from pathlib import Path

        self.path = Path(data_dir or settings.data_dir) / filename
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self, default: object) -> object:
        return read_json(self.path, default=default)

    def save(self, payload: object) -> None:
        with self._lock:
            try:
                write_json_atomic(self.path, payload)
            except OSError as exc:
                # Persistence must never take the app down with it.
                import logging

                logging.getLogger(__name__).warning(
                    "[PERSIST] Could not write %s: %s", self.path.name, exc
                )
