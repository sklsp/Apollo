"""Loading of the RAG evaluation dataset (portfolio item 18).

The dataset is plain data on disk: ground-truth documents under ``corpus/``
and a question set in ``questions.json``. This module turns that into typed,
validated records so every later stage (index building, retrieval, metrics)
can rely on the same shapes and on referential integrity between the two.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVALS_DIR / "corpus"
QUESTIONS_FILE = EVALS_DIR / "questions.json"


@dataclass(frozen=True)
class CorpusDoc:
    """One ground-truth document from the corpus."""

    doc_id: str  # file stem, e.g. "sdxl"
    title: str
    text: str


@dataclass(frozen=True)
class Question:
    """One evaluation question with its ground truth."""

    id: str
    question: str
    source_docs: tuple[str, ...]
    key_facts: tuple[str, ...]


@dataclass(frozen=True)
class EvalDataset:
    """The full dataset: corpus documents plus the validated question set."""

    docs: tuple[CorpusDoc, ...]
    questions: tuple[Question, ...]

    @property
    def doc_ids(self) -> frozenset[str]:
        return frozenset(doc.doc_id for doc in self.docs)


class DatasetError(ValueError):
    """Raised when the on-disk dataset is malformed or inconsistent."""


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[CorpusDoc]:
    """Load every ``*.md`` document under *corpus_dir*, sorted by file name."""
    if not corpus_dir.is_dir():
        raise DatasetError(f"corpus directory not found: {corpus_dir}")
    docs: list[CorpusDoc] = []
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise DatasetError(f"empty corpus document: {path.name}")
        title = _first_heading(text) or path.stem
        docs.append(CorpusDoc(doc_id=path.stem, title=title, text=text))
    if not docs:
        raise DatasetError(f"no markdown documents in {corpus_dir}")
    return docs


def load_questions(path: Path = QUESTIONS_FILE) -> list[Question]:
    """Load and validate the question set from *path*."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise DatasetError(f"questions file must contain a JSON array: {path}")
    questions: list[Question] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(raw):
        question = _parse_question(entry, index)
        if question.id in seen_ids:
            raise DatasetError(f"duplicate question id: {question.id!r}")
        seen_ids.add(question.id)
        questions.append(question)
    return questions


def load_dataset(
    corpus_dir: Path = CORPUS_DIR,
    questions_file: Path = QUESTIONS_FILE,
) -> EvalDataset:
    """Load the corpus and question set, then validate that they agree.

    Every ``source_docs`` entry of every question must name a document that
    exists in the corpus; this catches typos and documents renamed or deleted
    after the questions were written.
    """
    docs = load_corpus(corpus_dir)
    questions = load_questions(questions_file)
    doc_ids = {doc.doc_id for doc in docs}
    for question in questions:
        for ref in question.source_docs:
            if ref not in doc_ids:
                raise DatasetError(
                    f"question {question.id!r} references unknown document {ref!r}"
                )
    return EvalDataset(docs=tuple(docs), questions=tuple(questions))


def _parse_question(entry: object, index: int) -> Question:
    if not isinstance(entry, dict):
        raise DatasetError(f"question #{index} is not a JSON object")
    try:
        qid = entry["id"]
        question_text = entry["question"]
        source_docs = entry["source_docs"]
        key_facts = entry["key_facts"]
    except KeyError as exc:
        raise DatasetError(f"question #{index} missing field {exc.args[0]!r}") from None
    if not isinstance(qid, str) or not qid.strip():
        raise DatasetError(f"question #{index}: 'id' must be a non-empty string")
    if not isinstance(question_text, str) or not question_text.strip():
        raise DatasetError(f"question {qid!r}: 'question' must be a non-empty string")
    return Question(
        id=qid.strip(),
        question=question_text.strip(),
        source_docs=_str_tuple(source_docs, qid, "source_docs"),
        key_facts=_str_tuple(key_facts, qid, "key_facts"),
    )


def _str_tuple(value: object, qid: str, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise DatasetError(f"question {qid!r}: '{field}' must be a list of strings")
    items = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if len(items) != len(value):
        raise DatasetError(
            f"question {qid!r}: '{field}' entries must all be non-empty strings"
        )
    if not items:
        raise DatasetError(f"question {qid!r}: '{field}' must not be empty")
    return tuple(items)
