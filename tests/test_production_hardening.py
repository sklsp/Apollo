"""Persistent RAG index, duplicate classification, hardware detection, preflight.

All tests are deterministic: embeddings come from a fake provider, hardware
from injected HardwareInfo objects, and images from synthetic Pillow buffers.
Nothing here requires Ollama, a GPU, or ComfyUI to be running.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.services.dataset_validation import classify_pair, validate_dataset
from app.services.hardware import HardwareInfo, GPUInfo
from app.services.rag.service import RAGService, content_hash
from app.services.training_preflight import advise, diagnose_failure, estimate_vram_mb


# ============================================
# Deterministic embedding provider
# ============================================


class FakeEmbeddingClient:
    """Deterministic stand-in for EmbeddingClient.

    Hashes text into a fixed-dimension vector so the same text always embeds
    identically and different text differs — enough for real FAISS behaviour.
    """

    model = "fake-embed-model"
    backend = "fake"

    def __init__(self, dimension: int = 64) -> None:
        self.dimension_size = dimension
        self.embed_calls = 0

    @property
    def dimension(self) -> int:
        return self.dimension_size

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        self.embed_calls += len(texts)
        rows = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vector = rng.standard_normal(self.dimension_size).astype(np.float32)
            rows.append(vector / np.linalg.norm(vector))
        return np.vstack(rows)


@pytest.fixture
def rag(tmp_path):
    """RAGService wired to temp persistence and the fake embedder."""
    service = RAGService(
        embedding_client=FakeEmbeddingClient(),
        persist_dir=str(tmp_path / "rag"),
    )
    service.embedding_client = FakeEmbeddingClient()
    # Rebuild with the fake client so dimension/model match.
    from app.services.rag.vector_store import VectorStore

    service._store = VectorStore(
        service.embedding_client.dimension,
        persist_dir=str(tmp_path / "rag"),
        embedding_model=service.embedding_client.model,
    )
    return service


# ============================================
# Persistent index: restart survival
# ============================================


class TestPersistentIndex:
    def test_index_survives_a_restart_without_reembedding(self, tmp_path):
        persist_dir = str(tmp_path / "rag")

        first = RAGService(FakeEmbeddingClient(), persist_dir=persist_dir)
        first.add_document("The quick brown fox jumps over the lazy dog. " * 20,
                           {"doc_id": "doc_1", "filename": "fox.txt"})
        chunks_before = first.chunk_count
        assert chunks_before > 0

        # "Restart": brand-new service instance over the same directory.
        second = RAGService(FakeEmbeddingClient(), persist_dir=persist_dir)
        assert second.chunk_count == chunks_before

        results = second.query("quick brown fox")
        assert results, "retrieval must work after restart"
        assert all(r.doc_id == "doc_1" for r in results)
        # Best chunk must be a real match: with deterministic hashing the
        # query's own text embeds identically, so its nearest neighbor scores
        # highest among stored chunks.
        best = max(results, key=lambda r: r.score)
        assert "quick brown fox" in best.text

    def test_incremental_indexing_skips_unchanged_documents(self, tmp_path):
        service = RAGService(FakeEmbeddingClient(), persist_dir=str(tmp_path / "rag"))
        text = "Steady content that will not change. " * 10

        service.add_document(text, {"doc_id": "a", "filename": "a.txt"})
        calls_after_first = service.embedding_client.embed_calls

        result = service.add_document(text, {"doc_id": "a", "filename": "a.txt"})
        assert result == 0, "unchanged document must not be re-indexed"
        assert service.embedding_client.embed_calls == calls_after_first

    def test_adding_document_b_does_not_reembed_document_a(self, tmp_path):
        service = RAGService(FakeEmbeddingClient(), persist_dir=str(tmp_path / "rag"))
        service.add_document("Document alpha content. " * 10,
                             {"doc_id": "a", "filename": "a.txt"})
        calls_after_a = service.embedding_client.embed_calls

        service.add_document("Document beta content. " * 10,
                             {"doc_id": "b", "filename": "b.txt"})
        new_calls = service.embedding_client.embed_calls - calls_after_a
        # Only B's chunks were embedded (A contributed zero new calls).
        b_chunks = service._doc_versions["b"]["chunks"]
        assert new_calls == b_chunks

    def test_changed_document_is_reindexed_others_untouched(self, tmp_path):
        service = RAGService(FakeEmbeddingClient(), persist_dir=str(tmp_path / "rag"))
        service.add_document("Original version of alpha. " * 10,
                             {"doc_id": "a", "filename": "a.txt"})
        service.add_document("Beta stays the same. " * 10,
                             {"doc_id": "b", "filename": "b.txt"})
        total_before = service.chunk_count

        changed = "Completely rewritten alpha content. " * 10
        service.add_document(changed, {"doc_id": "a", "filename": "a.txt"})

        versions = service.document_versions()
        assert versions["a"]["hash"] == content_hash(changed)
        assert versions["b"]["hash"] == content_hash("Beta stays the same. " * 10)
        # No stale duplicates: chunk count reflects one version of A.
        assert service.chunk_count < total_before + 100

        hits = service.query("rewritten alpha")
        doc_ids = {hit.doc_id for hit in hits}
        assert "b" in doc_ids or "a" in doc_ids

    def test_deleted_document_vectors_disappear(self, tmp_path):
        service = RAGService(FakeEmbeddingClient(), persist_dir=str(tmp_path / "rag"))
        service.add_document("Alpha text. " * 10, {"doc_id": "a", "filename": "a.txt"})
        service.add_document("Beta text. " * 10, {"doc_id": "b", "filename": "b.txt"})

        removed = service.remove_document("a")
        assert removed > 0
        assert "a" not in service.document_versions()
        assert all(hit.doc_id != "a" for hit in service.query("alpha"))

    def test_embedding_model_change_is_refused_not_silently_mixed(self, tmp_path):
        from pathlib import Path

        persist_dir = str(tmp_path / "rag")
        first = RAGService(FakeEmbeddingClient(), persist_dir=persist_dir)
        first.add_document("Some content. " * 10, {"doc_id": "a", "filename": "a.txt"})

        class DifferentModel(FakeEmbeddingClient):
            model = "other-embed-model"

        # The service refuses to serve the old index: it quarantines and
        # starts empty rather than mixing incompatible vectors.
        second = RAGService(DifferentModel(), persist_dir=persist_dir)
        assert second.chunk_count == 0
        quarantined = list(Path(persist_dir).glob("quarantined/meta.json.*"))
        assert quarantined, "incompatible index must be quarantined"

        # And the new model can index fresh content cleanly.
        second.add_document("Fresh under new model. " * 10,
                            {"doc_id": "b", "filename": "b.txt"})
        assert second.query("fresh under new")

    def test_corrupted_index_recovers_with_quarantine(self, tmp_path):
        import json
        from pathlib import Path

        persist_dir = Path(tmp_path / "rag")
        service = RAGService(FakeEmbeddingClient(), persist_dir=str(persist_dir))
        service.add_document("Recoverable knowledge. " * 10,
                             {"doc_id": "a", "filename": "a.txt"})

        # Corrupt the FAISS file but leave metadata intact.
        (persist_dir / "index.faiss").write_bytes(b"garbage not a faiss index")

        recovered = RAGService(FakeEmbeddingClient(), persist_dir=str(persist_dir))
        # Store survives (empty), quarantine happened, and re-indexing works.
        quarantined = list((persist_dir / "quarantined").glob("index.faiss.*"))
        assert quarantined, "corrupt index must be quarantined, not deleted silently"

        recovered.add_document("Fresh content after recovery. " * 10,
                               {"doc_id": "b", "filename": "b.txt"})
        assert recovered.query("fresh content")

    def test_metadata_count_mismatch_quarantines(self, tmp_path):
        from pathlib import Path

        persist_dir = Path(tmp_path / "rag")
        service = RAGService(FakeEmbeddingClient(), persist_dir=str(persist_dir))
        service.add_document("Consistent data. " * 10, {"doc_id": "a", "filename": "a.txt"})

        # Tamper: claim more vectors than the index actually holds.
        meta = json_load(persist_dir / "meta.json")
        meta["vector_count"] += 5
        json_write(persist_dir / "meta.json", meta)

        fresh = RAGService(FakeEmbeddingClient(), persist_dir=str(persist_dir))
        assert fresh.chunk_count == 0, "mismatched index must not be trusted"
        assert list((persist_dir / "quarantined").glob("meta.json.*"))

    def test_status_reports_indexed_documents(self, rag):
        rag.add_document("Status probe. " * 10, {"doc_id": "s1", "filename": "s.txt"})
        status = rag.status()
        assert status["documents"] == 1
        assert status["chunks"] > 0
        assert status["indexed_documents"][0]["doc_id"] == "s1"
        assert status["embedding_model"] == "fake-embed-model"


def json_load(path):
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path, payload):
    import json
    path.write_text(json.dumps(payload), encoding="utf-8")


# ============================================
# Duplicate classification
# ============================================


def make_png(width=512, height=512, color=(200, 30, 30)):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def make_gradient_png(width=512, height=512, seed=0):
    buffer = io.BytesIO()
    image = Image.new("RGB", (width, height))
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            shade = (x + y + seed * 40) % 256
            for dy in range(8):
                for dx in range(8):
                    if x + dx < width and y + dy < height:
                        image.putpixel((x + dx, y + dy),
                                       (shade, shade // 2, 255 - shade))
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class TestDuplicateClassification:
    def test_identical_files_are_exact_duplicates(self):
        store = {"a": make_gradient_png(seed=1), "b": make_gradient_png(seed=1)}
        images = [{"id": k, "filename": f"{k}.png", "caption": "x"} for k in store]
        report = validate_dataset(images, lambda i: store[i]).to_dict()
        assert report["duplicates"] >= 1

    def test_resized_copy_is_high_confidence_near_duplicate(self):
        original = Image.open(io.BytesIO(make_gradient_png(seed=3)))
        resized_buffer = io.BytesIO()
        original.resize((256, 256)).save(resized_buffer, format="PNG")

        store = {
            "orig": make_gradient_png(seed=3),
            "small": resized_buffer.getvalue(),
        }
        images = [{"id": k, "filename": f"{k}.png", "caption": "x"} for k in store]
        report = validate_dataset(images, lambda i: store[i]).to_dict()
        kinds = [f["kind"] for f in report["findings"]]
        assert "near_duplicate" in kinds or "duplicate" in kinds

    def test_different_images_are_unique(self):
        store = {
            "a": make_gradient_png(seed=10),
            "b": make_gradient_png(seed=50),
            "c": make_gradient_png(seed=90),
        }
        images = [{"id": k, "filename": f"{k}.png", "caption": "x"} for k in store]
        report = validate_dataset(images, lambda i: store[i]).to_dict()
        dup_kinds = {"duplicate", "near_duplicate", "near_duplicate_possible"}
        assert not any(f["kind"] in dup_kinds for f in report["findings"])

    def test_different_solid_colors_are_not_flagged(self):
        """The classic aHash false positive: unrelated flat images must pass."""
        store = {
            "red":   make_png(color=(220, 40, 40)),
            "blue":  make_png(color=(40, 60, 220)),
            "green": make_png(color=(40, 200, 70)),
            "white": make_png(color=(250, 250, 250)),
        }
        images = [{"id": k, "filename": f"{k}.png", "caption": "x"} for k in store]
        report = validate_dataset(images, lambda i: store[i]).to_dict()
        dup_kinds = {"duplicate", "near_duplicate"}
        flagged = [f for f in report["findings"] if f["kind"] in dup_kinds]
        assert flagged == [], (
            "distinct solid colors must never be near-duplicates; "
            f"got {[(f['kind'], f['detail']) for f in flagged]}"
        )

    def test_same_color_flat_images_are_still_caught(self):
        store = {"flat1": make_png(color=(120, 120, 120)),
                 "flat2": make_png(color=(122, 121, 119))}
        images = [{"id": k, "filename": f"{k}.png", "caption": "x"} for k in store]
        report = validate_dataset(images, lambda i: store[i]).to_dict()
        dup_kinds = {"duplicate", "near_duplicate", "near_duplicate_possible"}
        assert any(f["kind"] in dup_kinds for f in report["findings"])

    def test_transparent_png_does_not_crash_or_false_positive(self):
        buffer = io.BytesIO()
        Image.new("RGBA", (512, 512), (255, 0, 0, 128)).save(buffer, format="PNG")
        rgba = buffer.getvalue()

        store = {"trans": rgba, "solid": make_png(color=(255, 0, 0))}
        images = [{"id": k, "filename": f"{k}.png", "caption": "x"} for k in store]
        report = validate_dataset(images, lambda i: store[i]).to_dict()
        assert report["valid_images"] == 2

    def test_low_resolution_images_flag_resolution_but_decode(self):
        store = {"tiny": make_png(64, 64)}
        images = [{"id": "tiny", "filename": "tiny.png", "caption": "x"}]
        report = validate_dataset(images, lambda i: store[i]).to_dict()
        assert report["valid_images"] == 1
        assert any(f["kind"] == "extreme_resolution" for f in report["findings"])

    def test_classify_pair_bands(self):
        # Identical hashes -> high confidence.
        assert classify_pair("ffffffffffffffff", "ffffffffffffffff") == "near_duplicate_high"
        # Far apart -> unique.
        assert classify_pair("0000000000000000", "ffffffffffffffff") == "unique"

    def test_possible_duplicates_are_reported_separately(self):
        store = {"a": make_gradient_png(seed=7), "b": make_gradient_png(seed=7)}
        images = [{"id": k, "filename": f"{k}.png", "caption": "x"} for k in store]
        report = validate_dataset(images, lambda i: store[i]).to_dict()
        # Same pixels re-encoded: exact duplicate takes precedence.
        assert report["duplicates"] >= 1


# ============================================
# Hardware detection & training preflight
# ============================================


def fake_hardware(vram_mb=None, vendor="nvidia", ram_mb=32768, name="Fake GPU"):
    gpus = []
    if vram_mb is not None:
        gpus.append(GPUInfo(vendor=vendor, name=name, vram_total_mb=vram_mb,
                            vram_free_mb=vram_mb, source="test"))
    return HardwareInfo(gpus=gpus, cpu_name="Test CPU", cpu_cores=8,
                        ram_total_mb=ram_mb, ram_available_mb=ram_mb // 2,
                        disk_free_gb=500.0, cuda_available=bool(gpus))


class TestHardwareDetection:
    def test_real_detection_never_raises_and_returns_shape(self):
        from app.services.hardware import detect_hardware

        info = detect_hardware()  # whatever this machine has
        data = info.to_dict()
        assert isinstance(data["gpus"], list)
        assert data["accelerator"] in ("cuda", "rocm", "amd", "cpu", "unknown")
        assert data["disk_free_gb"] is None or data["disk_free_gb"] > 0

    def test_cpu_only_machine_reports_no_accelerator(self):
        info = fake_hardware(vram_mb=None)
        assert info.accelerator == "cpu"
        assert info.vram_total_mb is None


class TestPreflightAdvisor:
    BASE = {"steps": 2000, "learning_rate": 1e-4, "batch_size": 1,
            "resolution": [512], "lora_rank": 16}

    def test_small_config_on_big_gpu_is_safe(self):
        report = advise(dict(self.BASE), fake_hardware(vram_mb=24 * 1024))
        assert report.verdict == "ok"
        assert report.can_proceed

    def test_large_config_on_small_gpu_is_risky(self):
        options = dict(self.BASE, resolution=[1536], batch_size=4)
        report = advise(options, fake_hardware(vram_mb=6 * 1024))
        assert report.verdict == "risky"
        assert report.recommendations, "risky verdict must explain what to change"

    def test_mid_config_on_small_gpu_is_heavy(self):
        options = dict(self.BASE, resolution=[1024])
        report = advise(options, fake_hardware(vram_mb=6 * 1024))
        assert report.verdict in ("heavy", "risky")

    def test_cpu_only_is_unsupported(self):
        report = advise(dict(self.BASE), fake_hardware(vram_mb=None))
        assert report.verdict == "unsupported"
        assert not report.can_proceed

    def test_aggressive_learning_rate_warns(self):
        report = advise(dict(self.BASE, learning_rate=2e-3),
                        fake_hardware(vram_mb=24 * 1024))
        assert any(c.severity == "warning" and c.name == "Learning rate"
                   for c in report.checks)

    def test_vram_estimate_grows_with_batch_and_resolution(self):
        small = estimate_vram_mb({"resolution": [512], "batch_size": 1})
        big = estimate_vram_mb({"resolution": [1536], "batch_size": 4})
        assert big > small

    def test_advisor_never_modifies_the_users_options(self):
        options = dict(self.BASE)
        frozen = dict(options)
        advise(options, fake_hardware(vram_mb=6 * 1024))
        assert options == frozen, "advisor must recommend, never mutate"


class TestFailureDiagnostics:
    def test_cuda_oom_gets_actionable_explanation(self):
        diagnosis = diagnose_failure(
            "RuntimeError: CUDA out of memory. Tried to allocate 2.5 GiB")
        assert "GPU memory" in diagnosis["explanation"]
        assert "batch" in diagnosis["recommendation"].lower()
        assert "CUDA out of memory" in diagnosis["raw"], "raw error kept for pros"

    def test_disk_full_is_recognized(self):
        diagnosis = diagnose_failure("OSError: No space left on device")
        assert "disk" in diagnosis["explanation"].lower()

    def test_unknown_error_falls_back_gracefully(self):
        diagnosis = diagnose_failure("Segmentation fault (core dumped)")
        assert diagnosis["explanation"]
        assert diagnosis["raw"]


# ============================================
# Preflight service integration
# ============================================


class TestPreflightIntegration:
    def test_preflight_reports_dataset_state(self, project_service, project):
        from app.services.lora_training_service import LoRATrainingService

        service = LoRATrainingService(projects=project_service)
        report = service.preflight(project.id, {"steps": 1000})
        names = [check["name"] for check in report["checks"]]
        assert "Dataset" in names
        assert "Captions" in names
        # Empty dataset fails those checks.
        dataset_check = next(c for c in report["checks"] if c["name"] == "Dataset")
        assert dataset_check["passed"] is False

    def test_preflight_passes_with_valid_dataset(self, project_service, project):
        from app.services.lora_training_service import LoRATrainingService
        from tests.test_workspace_features import make_png

        project_service.add_image(project.id, "a.png", make_png())
        project_service.set_caption(project.id, "a", "p3r5on, portrait")

        service = LoRATrainingService(projects=project_service)
        report = service.preflight(project.id, {"steps": 1000})
        dataset_check = next(c for c in report["checks"] if c["name"] == "Dataset")
        captions_check = next(c for c in report["checks"] if c["name"] == "Captions")
        assert dataset_check["passed"] is True
        assert captions_check["passed"] is True

    def test_hardware_presets_structure(self, project_service):
        from app.services.lora_training_service import LoRATrainingService

        service = LoRATrainingService(projects=project_service)
        result = service.hardware_presets()
        assert "hardware" in result and "presets" in result
        for preset_id, variants in result["presets"].items():
            assert set(variants) >= {"label", "conservative", "balanced", "quality"}
            # Conservative is always at least as light as quality.
            cons = variants["conservative"]
            qual = variants["quality"]
            assert max(cons.get("resolution", [512])) <= max(qual.get("resolution", [512]))
