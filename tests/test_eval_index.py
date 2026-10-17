"""Index building for the RAG evaluation (portfolio item 20).

``build_index`` reuses the production chunker and vector store; these tests
run it against a deterministic fake embedder so no Ollama or model download is
needed. One class pins the build against the committed corpus itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.config import settings
from app.services.rag.chunking import chunk_text
from evals.dataset import EvalDataset, load_dataset
from evals.index import build_index
from tests.fakes import FakeEmbedder


class TestBuildIndex:
    def test_indexes_every_chunk_of_every_document(self, synthetic_dataset):
        result = build_index(synthetic_dataset, FakeEmbedder())

        expected_counts = {doc.doc_id: len(chunk_text(doc.text)) for doc in synthetic_dataset.docs}
        assert result.chunks_per_doc == expected_counts
        assert result.num_chunks == sum(expected_counts.values())
        assert result.store.size == result.num_chunks
        assert result.store.indexed_doc_ids == {"alpha", "beta"}

    def test_long_document_splits_into_multiple_chunks(self, synthetic_dataset):
        result = build_index(synthetic_dataset, FakeEmbedder())
        assert result.chunks_per_doc["alpha"] == 1
        assert result.chunks_per_doc["beta"] >= 2

    def test_stored_chunks_carry_metadata_in_order(self, synthetic_dataset):
        result = build_index(synthetic_dataset, FakeEmbedder())

        expected_texts: list[str] = []
        for doc in synthetic_dataset.docs:
            expected_texts.extend(chunk_text(doc.text))
        stored = result.store._chunks  # same access pattern as RAGService
        assert [chunk.text for chunk in stored] == expected_texts

        for doc in synthetic_dataset.docs:
            indices = [c.chunk_index for c in stored if c.doc_id == doc.doc_id]
            assert indices == list(range(len(chunk_text(doc.text))))

    def test_single_embed_batch_call_in_document_order(self, synthetic_dataset):
        embedder = FakeEmbedder()
        build_index(synthetic_dataset, embedder)

        expected: list[str] = []
        for doc in synthetic_dataset.docs:
            expected.extend(chunk_text(doc.text))
        assert len(embedder.calls) == 1
        assert embedder.calls[0] == expected

    def test_dimension_inferred_from_embedder(self, synthetic_dataset):
        result = build_index(synthetic_dataset, FakeEmbedder(dim=32))
        assert result.store.dimension == 32


class TestSearchIntegration:
    def test_querying_with_chunk_vector_ranks_its_doc_first(self, synthetic_dataset):
        embedder = FakeEmbedder()
        result = build_index(synthetic_dataset, embedder)

        target_text = chunk_text(synthetic_dataset.docs[0].text)[0]
        query = embedder.embed_batch([target_text])[0]
        hits = result.store.search(query, top_k=3)

        assert len(hits) == 3
        assert hits[0].doc_id == "alpha"
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)


class TestErrors:
    def test_empty_dataset_raises(self):
        with pytest.raises(ValueError, match="no documents"):
            build_index(EvalDataset(docs=(), questions=()), FakeEmbedder())

    def test_embedder_wrong_row_count_raises(self, synthetic_dataset):
        class Broken(FakeEmbedder):
            def embed_batch(self, texts: list[str]) -> np.ndarray:
                return np.zeros((len(texts) - 1, self.dim), dtype=np.float32)

        with pytest.raises(ValueError, match="shape"):
            build_index(synthetic_dataset, Broken())


class TestCommittedCorpus:
    """The committed corpus must index cleanly end to end."""

    def test_full_corpus_builds(self):
        dataset = load_dataset()
        result = build_index(dataset, FakeEmbedder(dim=32))

        assert set(result.chunks_per_doc) == dataset.doc_ids
        assert all(count >= 1 for count in result.chunks_per_doc.values())
        assert result.num_chunks == sum(result.chunks_per_doc.values())
        assert result.store.size == result.num_chunks


class TestChunkingDefaults:
    def test_defaults_follow_production_settings(self, synthetic_dataset):
        default = build_index(synthetic_dataset, FakeEmbedder())
        explicit = build_index(
            synthetic_dataset,
            FakeEmbedder(),
            chunk_size=settings.rag_chunk_size,
            overlap=settings.rag_chunk_overlap,
        )
        assert default.num_chunks == explicit.num_chunks

    def test_explicit_parameters_are_honored(self, synthetic_dataset):
        # Tiny chunks force many splits; huge chunks collapse each doc to one.
        small = build_index(synthetic_dataset, FakeEmbedder(), chunk_size=100, overlap=20)
        large = build_index(synthetic_dataset, FakeEmbedder(), chunk_size=10_000, overlap=0)
        assert all(count >= 1 for count in small.chunks_per_doc.values())
        assert sum(large.chunks_per_doc.values()) == len(synthetic_dataset.docs)

