"""Behavioural tests for app/services/document_service.py."""

import json
from pathlib import Path

from app.services.document_service import DocumentService


def _seed_documents(tmp_settings, payload) -> None:
    """Write a raw documents.json into the tmp data dir before construction."""
    data_dir = Path(tmp_settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "documents.json").write_text(json.dumps(payload), encoding="utf-8")


def _read_documents_file(tmp_settings):
    return json.loads((Path(tmp_settings.data_dir) / "documents.json").read_text(encoding="utf-8"))


class TestStoreAndRetrieve:
    def test_store_returns_sequential_ids_and_persists_across_restart(self, tmp_settings):
        svc = DocumentService()
        first_id = svc.store_document("first body", source="a.txt")
        second_id = svc.store_document("second body")

        assert (first_id, second_id) == ("doc_0", "doc_1")
        stored = _read_documents_file(tmp_settings)
        assert [d["id"] for d in stored] == ["doc_0", "doc_1"]
        assert stored[1]["source"] == "upload"  # default source

        fresh = DocumentService()
        restored = {d["id"]: d for d in fresh.get_all_documents()}
        assert set(restored) == {"doc_0", "doc_1"}
        assert restored["doc_0"]["content"] == "first body"
        assert restored["doc_0"]["source"] == "a.txt"

    def test_store_documents_batch_returns_ids_in_order(self, tmp_settings):
        svc = DocumentService()
        ids = svc.store_documents([("c1", "s1"), ("c2", "s2")])

        assert ids == ["doc_0", "doc_1"]
        assert svc.get_document("doc_0")["source"] == "s1"
        assert svc.get_document("doc_1")["content"] == "c2"

    def test_get_document_missing_returns_none(self, tmp_settings):
        assert DocumentService().get_document("nope") is None

    def test_delete_existing_and_missing(self, tmp_settings):
        svc = DocumentService()
        svc.store_documents([("one", "s"), ("two", "s")])

        assert svc.delete_document("doc_0") is True
        assert svc.get_document("doc_0") is None
        assert [d["id"] for d in _read_documents_file(tmp_settings)] == ["doc_1"]

        assert svc.delete_document("doc_99") is False

    def test_clear_all_resets_storage_and_counter(self, tmp_settings):
        svc = DocumentService()
        svc.store_documents([("one", "s"), ("two", "s")])

        svc.clear_all()

        assert svc.get_all_documents() == []
        assert _read_documents_file(tmp_settings) == []
        assert svc.store_document("again") == "doc_0"


class TestLoad:
    def test_load_skips_malformed_entries_and_syncs_counter(self, tmp_settings):
        _seed_documents(
            tmp_settings,
            [
                {"id": "doc_9", "content": "kept"},
                "junk",
                42,
                {"no_id": True},
                {"id": 7},
                {"id": "legacy_doc"},
            ],
        )

        svc = DocumentService()
        docs = {d["id"]: d for d in svc.get_all_documents()}

        # Only non-dict entries and dicts without an id are skipped; "legacy_doc" is a valid doc.
        assert set(docs) == {"doc_9", "7", "legacy_doc"}
        assert docs["doc_9"]["content"] == "kept"
        assert docs["7"]["content"] == ""  # missing content defaults to empty string
        assert docs["7"]["source"] == "upload"  # missing source defaults to upload

        # Counter sits ahead of doc_9; the non-numeric suffixes never raised.
        assert svc.store_document("next") == "doc_10"

    def test_load_ignores_non_list_payload(self, tmp_settings):
        _seed_documents(tmp_settings, {"id": "doc_0", "content": "not a list"})

        svc = DocumentService()
        assert svc.get_all_documents() == []


class TestRelevantContext:
    def test_empty_storage_returns_empty_string(self, tmp_settings):
        assert DocumentService().get_relevant_context("anything") == ""

    def test_ranks_by_keyword_overlap_and_drops_zero_score_docs(self, tmp_settings):
        svc = DocumentService()
        svc.store_document("the quick brown fox", source="a")
        svc.store_document("quick silver jumps", source="b")
        svc.store_document("zebra stripes only", source="c")

        ctx = svc.get_relevant_context("quick brown", top_k=3)

        assert "[Source: a]" in ctx and "[Source: b]" in ctx
        assert ctx.index("[Source: a]") < ctx.index("[Source: b]")  # score 2 beats 1
        assert "[Source: c]" not in ctx

    def test_respects_top_k(self, tmp_settings):
        svc = DocumentService()
        for i in range(4):
            svc.store_document(f"alpha filler {i}", source=f"s{i}")

        ctx = svc.get_relevant_context("alpha", top_k=2)
        assert ctx.count("[Source:") == 2

    def test_no_overlap_returns_empty_string(self, tmp_settings):
        svc = DocumentService()
        svc.store_document("completely unrelated words")

        assert svc.get_relevant_context("zzz qqq") == ""

    def test_truncates_content_to_500_chars(self, tmp_settings):
        svc = DocumentService()
        content = "word " * 200  # 1000 chars of one repeated word
        svc.store_document(content, source="s")

        ctx = svc.get_relevant_context("word")
        assert ctx == "[Source: s]\n" + content[:500]

    def test_matching_ignores_case_and_punctuation(self, tmp_settings):
        svc = DocumentService()
        svc.store_document("Hello, World!")

        ctx = svc.get_relevant_context("hello world")
        assert "[Source: upload]" in ctx


class TestNormalizeText:
    def test_lowercases_and_strips_special_characters(self):
        assert DocumentService._normalize_text("MiXeD 123, case!") == "mixed 123 case"
