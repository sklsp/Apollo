"""Unit tests for ``app.services.rag.ingestion`` (text extraction).

Real files are written to a tmp dir for the .txt paths; the PDF and DOCX
parsers are exercised with fakes injected into ``pypdf.PdfReader`` and
``docx.Document`` so no binary fixtures or network access are needed.
"""

from __future__ import annotations

import pytest

from app.services.rag.ingestion import (
    SUPPORTED_EXTENSIONS,
    _extract_from_bytes,
    extract_text,
)


class FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


def make_pdf_reader(pages):
    """Return a ``PdfReader`` stand-in whose pages yield the given texts."""

    class FakeReader:
        last_stream = None

        def __init__(self, stream):
            type(self).last_stream = stream
            self.pages = [FakePage(text) for text in pages]

    return FakeReader


class FakeParagraph:
    def __init__(self, text):
        self.text = text


def make_docx_document(paragraphs):
    """Return a ``Document`` stand-in with the given paragraph texts."""

    class FakeDoc:
        last_stream = None

        def __init__(self, stream):
            type(self).last_stream = stream
            self.paragraphs = [FakeParagraph(text) for text in paragraphs]

    return FakeDoc


# --------------------------------------------------------------------------- #
# extract_text: .txt and extension handling
# --------------------------------------------------------------------------- #


def test_extract_txt_returns_stripped_text(tmp_path):
    file = tmp_path / "notes.txt"
    file.write_bytes(b"\n  hello world \n\n")

    assert extract_text(file) == "hello world"


def test_extract_txt_invalid_utf8_raises(tmp_path):
    file = tmp_path / "bad.txt"
    file.write_bytes(b"\xff\xfe not utf-8")

    with pytest.raises(UnicodeDecodeError):
        extract_text(file)


def test_extract_empty_txt_raises_no_text(tmp_path):
    file = tmp_path / "empty.txt"
    file.write_bytes(b"   \n\t  ")

    with pytest.raises(ValueError, match="No extractable text found in 'empty.txt'"):
        extract_text(file)


def test_unsupported_extension_raises_with_supported_list(tmp_path):
    file = tmp_path / "virus.exe"
    file.write_bytes(b"MZ")

    with pytest.raises(ValueError, match="Unsupported file type '\\.exe'") as excinfo:
        extract_text(file)
    assert ".docx, .pdf, .txt" in str(excinfo.value)


def test_missing_file_raises_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_text(tmp_path / "missing.txt")


def test_filename_param_overrides_and_lowercases_extension(tmp_path):
    file = tmp_path / "data"
    file.write_bytes(b"payload text")

    assert extract_text(file, filename="Report.TXT") == "payload text"


def test_supported_extensions_constant():
    assert SUPPORTED_EXTENSIONS == {".txt", ".pdf", ".docx"}


# --------------------------------------------------------------------------- #
# pdf extraction (pypdf faked)
# --------------------------------------------------------------------------- #


def test_pdf_extracts_pages_and_joins_non_empty(tmp_path, monkeypatch):
    reader_cls = make_pdf_reader(["Hello", None, "   ", "World"])
    monkeypatch.setattr("pypdf.PdfReader", reader_cls)
    file = tmp_path / "doc.pdf"
    content = b"%PDF-1.4 fake bytes"
    file.write_bytes(content)

    assert extract_text(file) == "Hello\n\nWorld"
    assert reader_cls.last_stream.getvalue() == content


def test_pdf_all_empty_pages_raises_no_text(tmp_path, monkeypatch):
    monkeypatch.setattr("pypdf.PdfReader", make_pdf_reader([None, "  "]))
    file = tmp_path / "blank.pdf"
    file.write_bytes(b"%PDF-1.4")

    with pytest.raises(ValueError, match="No extractable text found in 'blank.pdf'"):
        extract_text(file)


# --------------------------------------------------------------------------- #
# docx extraction (python-docx faked)
# --------------------------------------------------------------------------- #


def test_docx_extracts_paragraphs_and_joins_non_empty(tmp_path, monkeypatch):
    doc_cls = make_docx_document(["  Alpha ", "", "Beta"])
    monkeypatch.setattr("docx.Document", doc_cls)
    file = tmp_path / "brief.docx"
    content = b"PK fake docx bytes"
    file.write_bytes(content)

    assert extract_text(file) == "Alpha\n\nBeta"
    assert doc_cls.last_stream.getvalue() == content


def test_docx_all_empty_paragraphs_raises_no_text(tmp_path, monkeypatch):
    monkeypatch.setattr("docx.Document", make_docx_document(["", "   "]))
    file = tmp_path / "empty.docx"
    file.write_bytes(b"PK")

    with pytest.raises(ValueError, match="No extractable text found in 'empty.docx'"):
        extract_text(file)


# --------------------------------------------------------------------------- #
# _extract_from_bytes fallback branch
# --------------------------------------------------------------------------- #


def test_extract_from_bytes_unsupported_extension_raises():
    with pytest.raises(ValueError, match="Unsupported file type '\\.xyz'"):
        _extract_from_bytes("file.xyz", ".xyz", b"data")

