"""Loading of the RAG evaluation dataset (portfolio item 18).

The first class pins the committed dataset itself (size, ids, referential
integrity) so a silent change to the ground truth cannot invalidate earlier
eval numbers. The remaining classes cover the loader's error paths with
synthetic data in tmp_path.
"""

from __future__ import annotations

import json

import pytest

from evals.dataset import DatasetError, load_corpus, load_dataset, load_questions


class TestCommittedDataset:
    """The dataset committed by item 17 must stay exactly as pinned here."""

    def test_corpus_has_twelve_documents(self):
        docs = load_corpus()
        assert len(docs) == 12
        assert all(doc.text.strip() for doc in docs)
        assert all(doc.title for doc in docs)

    def test_questions_are_pinned_q01_to_q48(self):
        questions = load_questions()
        ids = [question.id for question in questions]
        assert ids == [f"q{i:02d}" for i in range(1, 49)]

    def test_every_source_doc_resolves(self):
        dataset = load_dataset()
        for question in dataset.questions:
            assert set(question.source_docs) <= dataset.doc_ids

    def test_cross_document_questions_exist(self):
        cross = [q for q in load_questions() if len(q.source_docs) > 1]
        assert len(cross) == 4


class TestCorpusLoading:
    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(DatasetError, match="corpus directory not found"):
            load_corpus(tmp_path / "nope")

    def test_empty_directory_raises(self, tmp_path):
        (tmp_path / "corpus").mkdir()
        with pytest.raises(DatasetError, match="no markdown documents"):
            load_corpus(tmp_path / "corpus")

    def test_blank_file_raises(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "blank.md").write_text("   \n", encoding="utf-8")
        with pytest.raises(DatasetError, match="empty corpus document"):
            load_corpus(corpus)

    def test_title_falls_back_to_stem_when_no_heading(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "no-heading.md").write_text("just text\n", encoding="utf-8")
        docs = load_corpus(corpus)
        assert docs[0].title == "no-heading"

    def test_heading_becomes_title(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "doc.md").write_text("# The Title\nBody.\n", encoding="utf-8")
        docs = load_corpus(corpus)
        assert docs[0].title == "The Title"


class TestQuestionLoading:
    def _write(self, tmp_path, payload):
        path = tmp_path / "questions.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @staticmethod
    def _entry(**overrides):
        entry = {"id": "q01", "question": "Q?", "source_docs": ["a"], "key_facts": ["f"]}
        entry.update(overrides)
        return entry

    def test_non_array_raises(self, tmp_path):
        with pytest.raises(DatasetError, match="JSON array"):
            load_questions(self._write(tmp_path, {"id": "q01"}))

    def test_missing_field_raises(self, tmp_path):
        entry = self._entry()
        del entry["key_facts"]
        with pytest.raises(DatasetError, match="missing field 'key_facts'"):
            load_questions(self._write(tmp_path, [entry]))

    def test_duplicate_id_raises(self, tmp_path):
        with pytest.raises(DatasetError, match="duplicate question id"):
            load_questions(self._write(tmp_path, [self._entry(), dict(self._entry())]))

    def test_empty_key_facts_raises(self, tmp_path):
        with pytest.raises(DatasetError, match="'key_facts' must not be empty"):
            load_questions(self._write(tmp_path, [self._entry(key_facts=[])]))

    def test_non_string_fact_raises(self, tmp_path):
        with pytest.raises(DatasetError, match="non-empty strings"):
            load_questions(self._write(tmp_path, [self._entry(key_facts=[42])]))


class TestCrossValidation:
    @pytest.fixture
    def corpus_dir(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "alpha.md").write_text("# Alpha\nText.\n", encoding="utf-8")
        return corpus

    def _questions_file(self, tmp_path, source_docs):
        path = tmp_path / "questions.json"
        entry = {"id": "q01", "question": "Q?", "source_docs": source_docs, "key_facts": ["f"]}
        path.write_text(json.dumps([entry]), encoding="utf-8")
        return path

    def test_unknown_document_reference_raises(self, tmp_path, corpus_dir):
        questions = self._questions_file(tmp_path, ["beta"])
        with pytest.raises(DatasetError, match="unknown document 'beta'"):
            load_dataset(corpus_dir=corpus_dir, questions_file=questions)

    def test_valid_pair_loads(self, tmp_path, corpus_dir):
        questions = self._questions_file(tmp_path, ["alpha"])
        dataset = load_dataset(corpus_dir=corpus_dir, questions_file=questions)
        assert len(dataset.docs) == 1
        assert len(dataset.questions) == 1
