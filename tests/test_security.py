"""Path traversal, filename sanitisation, and upload validation."""

from __future__ import annotations

import pytest

from app.core.paths import (
    UnsafePathError,
    is_safetensors,
    safe_join,
    sanitize_filename,
    validate_image_upload,
    write_json_atomic,
)


class TestSafeJoin:
    @pytest.mark.parametrize(
        "attack",
        [
            "../secret.txt",
            "../../etc/passwd",
            "a/../../b",
            "..\\..\\windows\\system32",
            "sub/../../outside",
        ],
    )
    def test_traversal_is_rejected(self, tmp_path, attack):
        with pytest.raises(UnsafePathError):
            safe_join(tmp_path, attack)

    @pytest.mark.parametrize(
        "attack",
        ["C:\\Windows\\System32", "/etc/passwd", "\\\\server\\share", "D:/data"],
    )
    def test_absolute_and_unc_paths_are_rejected(self, tmp_path, attack):
        with pytest.raises(UnsafePathError):
            safe_join(tmp_path, attack)

    def test_empty_component_is_rejected(self, tmp_path):
        with pytest.raises(UnsafePathError):
            safe_join(tmp_path, "")

    def test_legitimate_paths_are_allowed(self, tmp_path):
        result = safe_join(tmp_path, "project", "dataset", "img.png")
        assert result == (tmp_path / "project" / "dataset" / "img.png").resolve()

    def test_root_itself_is_allowed(self, tmp_path):
        assert safe_join(tmp_path, ".") == tmp_path.resolve()


class TestSanitizeFilename:
    @pytest.mark.parametrize(
        "raw,forbidden",
        [
            ("../../etc/passwd", "/"),
            ("..\\..\\evil.png", "\\"),
            ("sub/dir/file.png", "/"),
        ],
    )
    def test_directory_parts_are_stripped(self, raw, forbidden):
        result = sanitize_filename(raw)
        assert forbidden not in result
        assert ".." not in result

    def test_shell_metacharacters_are_removed(self):
        result = sanitize_filename("img$(rm -rf ~);&|`.png")
        assert not set("$();&|`") & set(result)
        assert result.endswith(".png")

    def test_windows_reserved_names_are_defused(self):
        assert sanitize_filename("CON.png") != "CON.png"
        assert sanitize_filename("nul.jpg").lower() != "nul.jpg"

    def test_empty_name_falls_back(self):
        assert sanitize_filename("") == "file"
        assert sanitize_filename("...") == "file"

    def test_normal_names_survive(self):
        assert sanitize_filename("photo_01.png") == "photo_01.png"

    def test_length_is_capped(self):
        assert len(sanitize_filename("a" * 500 + ".png")) <= 120


class TestUploadValidation:
    @pytest.mark.parametrize(
        "filename",
        ["evil.exe", "script.py", "doc.pdf", "archive.zip", "shell.php", "noext"],
    )
    def test_disallowed_extensions_are_rejected(self, filename):
        with pytest.raises(ValueError, match="Unsupported image type"):
            validate_image_upload(filename, 1000, 10_000)

    @pytest.mark.parametrize("filename", ["a.png", "b.jpg", "c.jpeg", "D.PNG"])
    def test_allowed_extensions_pass(self, filename):
        assert validate_image_upload(filename, 1000, 10_000)

    def test_oversized_upload_is_rejected(self):
        with pytest.raises(ValueError, match="limit"):
            validate_image_upload("big.png", 50_000_000, 25 * 1024 * 1024)

    def test_empty_upload_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            validate_image_upload("x.png", 0, 10_000)

    def test_double_extension_is_judged_on_the_last_one(self):
        with pytest.raises(ValueError):
            validate_image_upload("payload.png.exe", 100, 10_000)


class TestSafetensorsDetection:
    def test_renamed_file_is_not_accepted(self, tmp_path):
        fake = tmp_path / "not_really.safetensors"
        fake.write_bytes(b"this is plain text, not a tensor file at all")
        assert is_safetensors(fake) is False

    def test_valid_header_is_accepted(self, tmp_path):
        import json

        header = json.dumps({"__metadata__": {"format": "pt"}}).encode()
        real = tmp_path / "real.safetensors"
        real.write_bytes(len(header).to_bytes(8, "little") + header + b"\x00" * 32)
        assert is_safetensors(real) is True

    def test_truncated_file_is_rejected(self, tmp_path):
        stub = tmp_path / "short.safetensors"
        stub.write_bytes(b"\x01\x02")
        assert is_safetensors(stub) is False

    def test_absurd_header_length_is_rejected(self, tmp_path):
        bomb = tmp_path / "bomb.safetensors"
        bomb.write_bytes((2**60).to_bytes(8, "little") + b"x" * 16)
        assert is_safetensors(bomb) is False


class TestAtomicWrite:
    def test_existing_file_survives_a_failed_write(self, tmp_path):
        target = tmp_path / "state.json"
        write_json_atomic(target, {"keep": True})

        # A circular reference is one of the few things json cannot encode even
        # with the ``default=str`` fallback.
        circular: dict = {}
        circular["self"] = circular

        with pytest.raises(ValueError):
            write_json_atomic(target, circular)

        import json

        assert json.loads(target.read_text()) == {"keep": True}
        assert not list(tmp_path.glob("*.tmp")), "temp file was left behind"
