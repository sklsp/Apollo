"""Stress test: 120-image dataset validation + large document indexing.

Measures wall time and peak behavior; prints a summary. Not part of the
default suite (marked slow) — run explicitly with:

    pytest tests/test_stress.py -m slow -v
"""

from __future__ import annotations

import io
import json
import random
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.slow


def make_photo_png(width=640, height=480, seed=0) -> bytes:
    """A noisy 'photo-like' image (random texture, not flat)."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, (height // 4, width // 4, 3), dtype=np.uint8)
    image = Image.fromarray(array).resize((width, height))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class TestLargeDataset:
    def test_120_image_dataset_validation_under_30s(self, project_service, project):
        from app.services.dataset_validation import validate_dataset

        rng = random.Random(42)
        images_meta = []
        store: dict[str, bytes] = {}

        t0 = time.perf_counter()
        for index in range(120):
            # Every 10th image is an exact copy of another (12 duplicates).
            if index % 10 == 9 and index >= 10:
                content = store[f"img{index - 9}"]
            else:
                content = make_photo_png(seed=index)
            key = f"img{index}"
            store[key] = content
            caption = f"photo {index}, studio" if index % 7 else ""  # ~17 missing
            images_meta.append(
                {"id": key, "filename": f"{key}.png", "caption": caption})

        build_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        report = validate_dataset(images_meta, lambda i: store[i]).to_dict()
        validate_seconds = time.perf_counter() - t1

        print(f"\n[STRESS] built 120 images in {build_seconds:.1f}s")
        print(f"[STRESS] validated 120 images in {validate_seconds:.1f}s")
        print(f"[STRESS] score={report['score']} duplicates={report['duplicates']} "
              f"missing_captions={report['missing_captions']}")

        assert validate_seconds < 30, "validation must stay interactive"
        assert report["total_images"] == 120
        assert report["duplicates"] >= 11, "planted copies must be found"
        assert report["missing_captions"] > 0

    def test_upload_and_list_60_images_via_service(self, project_service, project):
        t0 = time.perf_counter()
        for index in range(60):
            project_service.add_image(project.id, f"s{index}.png",
                                      make_photo_png(seed=1000 + index))
        upload_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        images = project_service.list_images(project.id)
        list_seconds = time.perf_counter() - t1

        print(f"\n[STRESS] uploaded 60 images in {upload_seconds:.1f}s "
              f"({60 / max(upload_seconds, 0.01):.0f}/s)")
        print(f"[STRESS] listed {len(images)} images in {list_seconds:.2f}s")

        assert len(images) == 60
        assert upload_seconds < 60
        assert list_seconds < 5, "listing must not degrade with dataset size"


class TestLargeDocument:
    def test_large_document_chunk_index_persist_retrieve(self, tmp_path):
        from tests.test_production_hardening import FakeEmbeddingClient
        from app.services.rag.service import RAGService

        # ~200 KB of text -> roughly 250+ chunks at default settings.
        paragraph = (
            "The integration layer coordinates document ingestion, embedding "
            "and retrieval across subsystems while preserving provenance. "
        )
        big_text = " ".join(paragraph for _ in range(1400))

        service = RAGService(FakeEmbeddingClient(), persist_dir=str(tmp_path / "rag"))

        t0 = time.perf_counter()
        chunks = service.add_document(big_text, {"doc_id": "big", "filename": "big.txt"})
        index_seconds = time.perf_counter() - t0

        print(f"\n[STRESS] indexed {len(big_text) // 1024} KB into "
              f"{chunks} chunks in {index_seconds:.1f}s")

        assert chunks > 100
        assert index_seconds < 60

        # Restart survival with a large index.
        fresh = RAGService(FakeEmbeddingClient(), persist_dir=str(tmp_path / "rag"))
        assert fresh.chunk_count == chunks

        t1 = time.perf_counter()
        hits = fresh.query("integration layer coordinates ingestion")
        query_seconds = time.perf_counter() - t1
        print(f"[STRESS] queried large index in {query_seconds:.3f}s")
        assert hits and query_seconds < 2

        # Incremental skip on the unchanged large doc.
        assert service.add_document(big_text, {"doc_id": "big"}) == 0
