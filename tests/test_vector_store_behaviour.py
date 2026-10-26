"""Behaviour tests for app/services/rag/vector_store.py."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.services.rag.vector_store import (
    INDEX_VERSION,
    IndexIncompatibleError,
    IndexMeta,
    RetrievedChunk,
    StoredChunk,
    VectorStore,
)


def _vec(i: int, dim: int = 4) -> np.ndarray:
    """A deterministic unit vector; basis vectors keep inner products exact."""
    row = np.zeros(dim, dtype="float32")
    row[i % dim] = 1.0
    return row


def _chunk(doc_id: str, i: int = 0, text: str | None = None) -> StoredChunk:
    return StoredChunk(
        doc_id=doc_id,
        filename=f"{doc_id}.txt",
        chunk_index=i,
        text=text if text is not None else f"text of {doc_id} part {i}",
    )


def _save_store(persist_dir, dimension: int = 4, model: str = "model-a") -> VectorStore:
    """Build a store with two docs (doc_a x2 chunks, doc_b x1), save it."""
    store = VectorStore(dimension, persist_dir=persist_dir, embedding_model=model)
    embeddings = np.vstack([_vec(0), _vec(1), _vec(2)])
    store.add(embeddings, [_chunk("doc_a", 0), _chunk("doc_a", 1), _chunk("doc_b", 0)])
    store.save()
    return store


def _read_meta(persist_dir) -> dict:
    return json.loads((persist_dir / "meta.json").read_text(encoding="utf-8"))


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class _FakeIndex:
    """Test double for faiss.Index that returns one negative (invalid) slot."""

    ntotal = 2

    def search(self, vector, k):
        return np.array([[0.9, -1.0]], dtype="float32"), np.array([[0, -1]])


def _quarantined_names(persist_dir) -> list[str]:
    q = persist_dir / "quarantined"
    return sorted(p.name for p in q.iterdir()) if q.is_dir() else []


# ---------- IndexMeta ---------------------------------------------------------


class TestIndexMeta:
    def test_to_dict_round_trip(self):
        meta = IndexMeta(embedding_model="m", dimension=768, vector_count=12)
        restored = IndexMeta.from_dict(meta.to_dict())
        assert restored == meta

    def test_from_dict_defaults_and_casts(self):
        raw = {"embedding_model": 42, "dimension": "768", "vector_count": "3"}
        meta = IndexMeta.from_dict(raw)
        assert meta.version == 0  # missing key -> 0, not INDEX_VERSION
        assert meta.embedding_model == "42"
        assert meta.dimension == 768
        assert meta.vector_count == 3
        assert meta.updated_at == 0.0

    def test_from_dict_casts_every_field(self):
        raw = {
            "version": str(INDEX_VERSION),
            "embedding_model": "m",
            "dimension": "8",
            "vector_count": "2",
            "updated_at": "1761290435.5",
        }
        meta = IndexMeta.from_dict(raw)
        assert (meta.version, meta.embedding_model, meta.dimension, meta.vector_count) == (
            INDEX_VERSION, "m", 8, 2,
        )
        assert meta.updated_at == pytest.approx(1761290435.5)


# ---------- add / search ------------------------------------------------------


class TestAddAndSearch:
    def test_empty_store(self):
        store = VectorStore(4)
        assert store.size == 0
        assert store.indexed_doc_ids == set()
        assert store.search(_vec(0)) == []

    def test_add_empty_is_noop(self, tmp_path):
        persist = tmp_path / "rag"
        store = VectorStore(4, persist_dir=persist)
        store.add(np.zeros((0, 4), dtype="float32"), [])
        assert store.size == 0
        store.save()  # never dirtied -> nothing written
        assert not (persist / "meta.json").exists()

    def test_add_count_mismatch_raises(self):
        store = VectorStore(4)
        with pytest.raises(ValueError, match="Embedding count must match chunk count"):
            store.add(np.vstack([_vec(0), _vec(1)]), [_chunk("doc_a")])

    def test_add_dimension_mismatch_raises(self):
        store = VectorStore(4)
        bad = np.zeros((1, 8), dtype="float32")
        with pytest.raises(ValueError, match=r"Expected dimension 4, got 8"):
            store.add(bad, [_chunk("doc_a")])

    def test_add_grows_store_and_doc_ids(self):
        store = VectorStore(4)
        store.add(np.vstack([_vec(0), _vec(1)]), [_chunk("doc_a"), _chunk("doc_b")])
        assert store.size == 2
        assert store.indexed_doc_ids == {"doc_a", "doc_b"}

    def test_search_ranks_by_inner_product(self):
        store = VectorStore(4)
        store.add(np.vstack([_vec(0), _vec(1)]), [_chunk("doc_a"), _chunk("doc_b")])
        results = store.search(_vec(0), top_k=2)
        assert [r.doc_id for r in results] == ["doc_a", "doc_b"]
        assert all(isinstance(r, RetrievedChunk) for r in results)
        assert results[0].score == pytest.approx(1.0)
        assert results[1].score == pytest.approx(0.0)
        assert (results[0].filename, results[0].chunk_index, results[0].text) == (
            "doc_a.txt", 0, "text of doc_a part 0",
        )

    def test_search_clamps_top_k_to_size(self):
        store = VectorStore(4)
        store.add(np.vstack([_vec(0), _vec(1)]), [_chunk("doc_a"), _chunk("doc_b")])
        assert len(store.search(_vec(0), top_k=50)) == 2

    def test_search_skips_negative_indices(self, monkeypatch):
        store = VectorStore(4)
        store.add(np.vstack([_vec(0), _vec(1)]), [_chunk("doc_a"), _chunk("doc_b")])
        # FAISS reports an invalid slot as index -1; search() must skip it.
        monkeypatch.setattr(store, "_index", _FakeIndex())

        results = store.search(_vec(0), top_k=2)

        assert [r.doc_id for r in results] == ["doc_a"]
        assert results[0].score == pytest.approx(0.9)


# ---------- persistence -------------------------------------------------------


class TestPersistence:
    def test_save_without_persist_dir_is_noop(self):
        store = VectorStore(4)
        store.add(np.vstack([_vec(0)]), [_chunk("doc_a")])
        store.save()  # must not raise

    def test_save_when_clean_writes_nothing(self, tmp_path):
        persist = tmp_path / "rag"
        VectorStore(4, persist_dir=persist).save()
        assert list(persist.iterdir()) == [] if persist.exists() else True

    def test_save_writes_all_three_files(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)
        for name in ("index.faiss", "chunks.json", "meta.json"):
            assert (persist / name).is_file(), name
        meta = _read_meta(persist)
        assert meta["version"] == INDEX_VERSION
        assert meta["embedding_model"] == "model-a"
        assert meta["dimension"] == 4
        assert meta["vector_count"] == 3
        chunks = json.loads((persist / "chunks.json").read_text(encoding="utf-8"))
        assert len(chunks) == 3
        assert chunks[0] == {
            "doc_id": "doc_a",
            "filename": "doc_a.txt",
            "chunk_index": 0,
            "text": "text of doc_a part 0",
            "content_hash": "",
        }

    def test_restart_restores_state(self, tmp_path):
        persist = tmp_path / "rag"
        first = _save_store(persist)
        before = first.search(_vec(1), top_k=3)

        second = VectorStore(4, persist_dir=persist, embedding_model="model-a")
        assert second.size == 3
        assert second.indexed_doc_ids == {"doc_a", "doc_b"}
        after = second.search(_vec(1), top_k=3)
        assert [r.doc_id for r in after] == [r.doc_id for r in before]
        assert [r.score for r in after] == pytest.approx([r.score for r in before])

    def test_load_missing_meta_starts_fresh(self, tmp_path):
        persist = tmp_path / "rag"
        persist.mkdir()
        store = VectorStore(4, persist_dir=persist)
        assert store.size == 0

    def test_load_non_dict_meta_starts_fresh(self, tmp_path):
        persist = tmp_path / "rag"
        persist.mkdir()
        _write_json(persist / "meta.json", [1, 2, 3])
        store = VectorStore(4, persist_dir=persist)
        assert store.size == 0


# ---------- load failure: quarantine ------------------------------------------


class TestLoadQuarantine:
    def test_version_mismatch_quarantines(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)
        meta = _read_meta(persist)
        meta["version"] = 999
        _write_json(persist / "meta.json", meta)

        store = VectorStore(4, persist_dir=persist, embedding_model="model-a")
        assert store.size == 0
        for name in ("index.faiss", "chunks.json", "meta.json"):
            assert not (persist / name).exists(), name
        assert len(_quarantined_names(persist)) == 3

    def test_embedding_model_mismatch_raises(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)
        with pytest.raises(IndexIncompatibleError) as excinfo:
            VectorStore(4, persist_dir=persist, embedding_model="model-b")
        message = str(excinfo.value)
        assert "model-a" in message and "model-b" in message
        # Incompatibility is reported, not quarantined: files stay put.
        assert (persist / "meta.json").is_file()

    def test_dimension_mismatch_in_meta_raises(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)
        meta = _read_meta(persist)
        meta["dimension"] = 8
        _write_json(persist / "meta.json", meta)

        with pytest.raises(IndexIncompatibleError, match=r"dimension 8"):
            VectorStore(4, persist_dir=persist, embedding_model="model-a")

    def test_missing_chunks_file_quarantines(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)
        (persist / "chunks.json").unlink()

        store = VectorStore(4, persist_dir=persist, embedding_model="model-a")
        assert store.size == 0
        assert len(_quarantined_names(persist)) >= 1

    def test_chunk_count_mismatch_quarantines(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)
        chunks = json.loads((persist / "chunks.json").read_text(encoding="utf-8"))
        _write_json(persist / "chunks.json", chunks[:1])  # meta still says 3

        store = VectorStore(4, persist_dir=persist, embedding_model="model-a")
        assert store.size == 0
        assert len(_quarantined_names(persist)) == 3

    def test_corrupt_index_file_quarantines(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)
        (persist / "index.faiss").write_bytes(b"this is not a faiss index")

        store = VectorStore(4, persist_dir=persist, embedding_model="model-a")
        assert store.size == 0
        assert len(_quarantined_names(persist)) == 3

    def test_index_ntotal_mismatch_quarantines(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)
        meta = _read_meta(persist)
        meta["vector_count"] = 5
        _write_json(persist / "meta.json", meta)
        chunks = json.loads((persist / "chunks.json").read_text(encoding="utf-8"))
        while len(chunks) < 5:
            chunks.append({"doc_id": f"pad{len(chunks)}"})
        _write_json(persist / "chunks.json", chunks)

        store = VectorStore(4, persist_dir=persist, embedding_model="model-a")
        assert store.size == 0
        assert len(_quarantined_names(persist)) == 3

    def test_index_dimension_mismatch_quarantines(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)  # real index is dim 4
        meta = _read_meta(persist)
        meta["dimension"] = 8  # lie so the meta check passes...
        _write_json(persist / "meta.json", meta)

        store = VectorStore(8, persist_dir=persist, embedding_model="model-a")
        assert store.size == 0
        assert len(_quarantined_names(persist)) == 3

    def test_incomplete_chunk_records_quarantines(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)
        chunks = json.loads((persist / "chunks.json").read_text(encoding="utf-8"))
        chunks[1] = "not a dict"  # same length, but one record is unusable
        _write_json(persist / "chunks.json", chunks)

        store = VectorStore(4, persist_dir=persist, embedding_model="model-a")
        assert store.size == 0
        assert len(_quarantined_names(persist)) == 3

    def test_quarantine_survives_move_failure(self, tmp_path, monkeypatch):
        persist = tmp_path / "rag"
        _save_store(persist)
        meta = _read_meta(persist)
        meta["version"] = 999
        _write_json(persist / "meta.json", meta)

        def boom(src, dst):
            raise OSError("disk on fire")

        monkeypatch.setattr("app.services.rag.vector_store.shutil.move", boom)

        store = VectorStore(4, persist_dir=persist, embedding_model="model-a")

        assert store.size == 0
        # The move failed, so the originals must still be where they were.
        for name in ("index.faiss", "chunks.json", "meta.json"):
            assert (persist / name).is_file(), name

    def test_chunk_record_defaults_on_load(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)
        # Keep 3 records (matching vector_count) but strip the optional fields
        # from one: it must load with default filename/text/chunk_index.
        chunks = json.loads((persist / "chunks.json").read_text(encoding="utf-8"))
        chunks[1] = {"doc_id": "doc_a"}
        _write_json(persist / "chunks.json", chunks)

        store = VectorStore(4, persist_dir=persist, embedding_model="model-a")
        assert store.size == 3
        top = store.search(_vec(1), top_k=1)[0]
        assert (top.doc_id, top.filename, top.chunk_index, top.text) == ("doc_a", "", 0, "")


# ---------- remove / clear ----------------------------------------------------


class TestRemoveAndClear:
    def test_remove_by_doc_id_removes_all_chunks_for_doc(self):
        store = VectorStore(4)
        store.add(np.vstack([_vec(0), _vec(1), _vec(2)]),
                  [_chunk("doc_a", 0), _chunk("doc_a", 1), _chunk("doc_b")])
        removed = store.remove_by_doc_id("doc_a")
        assert removed == 2
        assert store.size == 1
        results = store.search(_vec(2), top_k=5)
        assert [r.doc_id for r in results] == ["doc_b"]

    def test_remove_unknown_doc_is_noop(self):
        store = VectorStore(4)
        store.add(np.vstack([_vec(0)]), [_chunk("doc_a")])
        assert store.remove_by_doc_id("ghost") == 0
        assert store.size == 1

    def test_remove_after_restart(self, tmp_path):
        persist = tmp_path / "rag"
        _save_store(persist)
        reloaded = VectorStore(4, persist_dir=persist, embedding_model="model-a")
        assert reloaded.size == 3
        removed = reloaded.remove_by_doc_id("doc_a")
        assert removed == 2

    def test_clear_empties_in_memory_store(self):
        store = VectorStore(4)
        store.add(np.vstack([_vec(0), _vec(1)]), [_chunk("doc_a"), _chunk("doc_b")])
        store.clear()
        assert store.size == 0
        assert store.indexed_doc_ids == set()
        store.add(np.vstack([_vec(2)]), [_chunk("doc_c")])
        assert store.size == 1

    def test_clear_deletes_persisted_files(self, tmp_path):
        persist = tmp_path / "rag"
        store = _save_store(persist)
        store.clear()
        for name in ("index.faiss", "chunks.json", "meta.json"):
            assert not (persist / name).exists(), name
        # The store is usable again and re-persists cleanly.
        store.add(np.vstack([_vec(0)]), [_chunk("doc_c")])
        store.save()
        meta = _read_meta(persist)
        assert meta["vector_count"] == 1
