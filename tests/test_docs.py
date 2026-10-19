"""Keep the docs honest (item 41).

Reads ``README.md`` and every ``docs/*.md`` that exists, then checks three
things per file: no em or en dash characters, backticked repo paths exist
relative to the repo root, and every mermaid block is closed. The test keeps
working while those files do not exist yet.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PATH_RE = re.compile(r"`((?:app|tests|eval|scripts|workflows|docs)/[^`\s]*)`")


def _doc_files() -> list[Path]:
    files = [PROJECT_ROOT / "README.md"]
    docs_dir = PROJECT_ROOT / "docs"
    if docs_dir.is_dir():
        files.extend(sorted(docs_dir.glob("*.md")))
    return [f for f in files if f.is_file()]


def _strip_suffix(path: str) -> str:
    """Cut a ``:<line>`` or ``::<name>`` suffix off a documented path."""
    if "::" in path:
        return path.split("::", 1)[0]
    if ":" in path:
        return path.split(":", 1)[0]
    return path


def _problems(doc: Path) -> list[str]:
    text = doc.read_text(encoding="utf-8")
    lines = text.splitlines()
    problems: list[str] = []

    for lineno, line in enumerate(lines, 1):
        if "\u2014" in line:
            problems.append(f"line {lineno}: contains an em dash")
        if "\u2013" in line:
            problems.append(f"line {lineno}: contains an en dash")

    for lineno, line in enumerate(lines, 1):
        for match in PATH_RE.finditer(line):
            raw = match.group(1)
            if any(ch in raw for ch in "<*{"):
                continue
            path = _strip_suffix(raw)
            if not (PROJECT_ROOT / path).exists():
                problems.append(f"line {lineno}: `{raw}` does not exist")

    for lineno, line in enumerate(lines, 1):
        if "```mermaid" in line:
            closed = any("```" in later for later in lines[lineno:])
            if not closed:
                problems.append(f"line {lineno}: mermaid block is never closed")

    return problems


def test_docs_are_honest() -> None:
    problems = [
        f"{doc.relative_to(PROJECT_ROOT).as_posix()}: {problem}"
        for doc in _doc_files()
        for problem in _problems(doc)
    ]
    assert problems == [], "\n".join(problems)
