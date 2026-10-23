"""Behaviour tests for app/core/persistence.py (JsonPersist)."""

from __future__ import annotations

import json
import logging
from pathlib import Path


# ---------- construction ------------------------------------------------------


class TestInit:
    def test_explicit_data_dir_builds_path_and_creates_parent(self, tmp_path):
        from app.core.persistence import JsonPersist

        store = JsonPersist("mem.json", data_dir=str(tmp_path / "nested" / "data"))

        assert str(store.path) == str(tmp_path / "nested" / "data" / "mem.json")
        assert (tmp_path / "nested" / "data").is_dir()

    def test_without_data_dir_uses_settings(self, tmp_settings):
        from app.core.persistence import JsonPersist

        store = JsonPersist("mem.json")

        assert str(store.path) == str(Path(tmp_settings.data_dir) / "mem.json")


# ---------- load --------------------------------------------------------------


class TestLoad:
    def test_missing_file_returns_default(self, tmp_path):
        from app.core.persistence import JsonPersist

        store = JsonPersist("mem.json", data_dir=str(tmp_path))

        assert store.load({"fallback": True}) == {"fallback": True}

    def test_corrupt_file_returns_default(self, tmp_path):
        from app.core.persistence import JsonPersist

        store = JsonPersist("mem.json", data_dir=str(tmp_path))
        store.path.write_text("{not json", encoding="utf-8")

        assert store.load(None) is None


# ---------- save --------------------------------------------------------------


class TestSave:
    def test_save_then_load_roundtrip(self, tmp_path):
        from app.core.persistence import JsonPersist

        store = JsonPersist("mem.json", data_dir=str(tmp_path))
        payload = {"entries": [{"id": 1}], "count": 1}
        store.save(payload)

        assert store.load(None) == payload

    def test_save_writes_valid_json_and_leaves_no_temp_files(self, tmp_path):
        from app.core.persistence import JsonPersist

        store = JsonPersist("mem.json", data_dir=str(tmp_path))
        store.save({"a": [1, 2]})

        on_disk = json.loads(store.path.read_text(encoding="utf-8"))
        assert on_disk == {"a": [1, 2]}
        assert list(store.path.parent.glob("*.tmp")) == []

    def test_save_swallows_oserror_and_logs_warning(self, tmp_path, monkeypatch, caplog):
        from app.core import persistence
        from app.core.persistence import JsonPersist

        store = JsonPersist("mem.json", data_dir=str(tmp_path))

        def _boom(path, payload):
            raise OSError("disk full")

        monkeypatch.setattr(persistence, "write_json_atomic", _boom)

        with caplog.at_level(logging.WARNING, logger="app.core.persistence"):
            store.save({"a": 1})  # must not raise

        assert store.path.exists() is False
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "Could not write mem.json" in message
        assert "disk full" in message
