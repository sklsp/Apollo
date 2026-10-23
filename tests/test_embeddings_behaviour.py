"""Unit tests for ``app.services.rag.embeddings`` (EmbeddingClient).

The client is exercised with a fake ``OllamaClient`` injected through the
constructor, so no Ollama server or network access is needed. The sentence-
transformers fallback is covered two ways: its lazy construction path uses a
patched ``SentenceTransformer`` class (no model download), and its import-failure
path poisons ``sys.modules`` to force an ``ImportError``. Covers construction,
both backends, the empty-batch short circuit, normalization (including the
zero-vector guard) and the lazy ``dimension``/``backend`` probes.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.exceptions import OllamaServiceError
from app.services.rag.embeddings import EmbeddingClient


class FakeOllamaClient:
    """Minimal stand-in for ``OllamaClient`` with canned or failing embed calls."""

    def __init__(self, vectors=None, error=None):
        self.embed_calls = []
        self._vectors = vectors if vectors is not None else [[1.0, 0.0]]
        self._error = error

    def embed(self, *, model, inputs):
        self.embed_calls.append({"model": model, "inputs": list(inputs)})
        if self._error is not None:
            raise self._error
        return [list(row) for row in self._vectors]


class FakeSentenceTransformer:
    """Records construction and encode calls; returns canned vectors."""

    instances = []

    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path
        self.encode_calls = []
        FakeSentenceTransformer.instances.append(self)

    def encode(self, texts, convert_to_numpy=False, show_progress_bar=True):
        self.encode_calls.append(
            {
                "texts": list(texts),
                "convert_to_numpy": convert_to_numpy,
                "show_progress_bar": show_progress_bar,
            }
        )
        return np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)


def make_client(monkeypatch, ollama=None, model="test-embed-model", fallback_model="test-fallback-model"):
    """Build an ``EmbeddingClient`` with a stubbed settings object."""
    monkeypatch.setattr(
        "app.services.rag.embeddings.settings",
        SimpleNamespace(embedding_model=model, embedding_fallback_model=fallback_model),
    )
    return EmbeddingClient(ollama_client=ollama)


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #


def test_init_uses_provided_client_and_models(monkeypatch):
    ollama = FakeOllamaClient()

    client = make_client(monkeypatch, ollama=ollama, model="m1", fallback_model="f1")

    assert client.ollama_client is ollama
    assert client.model == "m1"
    assert client.fallback_model == "f1"
    assert client._backend is None
    assert client._dimension is None


def test_init_defaults_from_settings_and_creates_ollama_client(monkeypatch):
    created = []

    class RecorderOllama:
        def __init__(self):
            created.append(True)

    monkeypatch.setattr("app.services.rag.embeddings.OllamaClient", RecorderOllama)
    client = make_client(monkeypatch, model="settings-model", fallback_model="settings-fallback")

    assert len(created) == 1
    assert client.model == "settings-model"
    assert client.fallback_model == "settings-fallback"


# --------------------------------------------------------------------------- #
# embed / embed_batch via the ollama backend
# --------------------------------------------------------------------------- #


def test_embed_batch_empty_returns_zero_shape_without_calling_backend(monkeypatch):
    ollama = FakeOllamaClient()
    client = make_client(monkeypatch, ollama=ollama)

    result = client.embed_batch([])

    assert result.shape == (0, 0)
    assert result.dtype == np.float32
    assert ollama.embed_calls == []


def test_embed_single_returns_normalized_first_row(monkeypatch):
    ollama = FakeOllamaClient(vectors=[[3.0, 4.0]])
    client = make_client(monkeypatch, ollama=ollama)

    vector = client.embed("hello")

    assert vector.tolist() == pytest.approx([0.6, 0.8])
    assert vector.dtype == np.float32
    assert ollama.embed_calls == [{"model": "test-embed-model", "inputs": ["hello"]}]


def test_embed_batch_normalizes_rows_and_sets_backend(monkeypatch):
    ollama = FakeOllamaClient(vectors=[[3.0, 4.0], [0.0, 2.0]])
    client = make_client(monkeypatch, ollama=ollama)

    matrix = client.embed_batch(["a", "b"])

    assert np.allclose(matrix, [[0.6, 0.8], [0.0, 1.0]])
    assert matrix.dtype == np.float32
    assert client.backend == "ollama"
    assert client.dimension == 2


def test_embed_batch_casts_integer_vectors_to_float32(monkeypatch):
    ollama = FakeOllamaClient(vectors=[[4, 0]])
    client = make_client(monkeypatch, ollama=ollama)

    matrix = client.embed_batch(["a"])

    assert matrix.dtype == np.float32
    assert np.allclose(matrix, [[1.0, 0.0]])


def test_zero_vector_row_stays_finite(monkeypatch):
    """The zero-norm guard in _normalize must not produce NaN or inf."""
    ollama = FakeOllamaClient(vectors=[[0.0, 0.0], [2.0, 0.0]])
    client = make_client(monkeypatch, ollama=ollama)

    matrix = client.embed_batch(["zero", "nonzero"])

    assert np.isfinite(matrix).all()
    assert matrix[0].tolist() == [0.0, 0.0]
    assert matrix[1].tolist() == pytest.approx([1.0, 0.0])


# --------------------------------------------------------------------------- #
# lazy dimension / backend probes
# --------------------------------------------------------------------------- #


def test_backend_property_probes_lazily(monkeypatch):
    ollama = FakeOllamaClient(vectors=[[1.0, 2.0]])
    client = make_client(monkeypatch, ollama=ollama)

    assert client.backend == "ollama"

    assert ollama.embed_calls == [{"model": "test-embed-model", "inputs": ["backend probe"]}]


def test_dimension_property_probes_once_and_caches(monkeypatch):
    ollama = FakeOllamaClient(vectors=[[1.0, 2.0, 3.0]])
    client = make_client(monkeypatch, ollama=ollama)

    first = client.dimension
    second = client.dimension

    assert first == 3
    assert second == 3
    assert len(ollama.embed_calls) == 1
    assert ollama.embed_calls[0]["inputs"] == ["dimension probe"]


# --------------------------------------------------------------------------- #
# sentence-transformers fallback
# --------------------------------------------------------------------------- #


def test_ollama_failure_falls_back_to_sentence_transformers(monkeypatch, caplog):
    FakeSentenceTransformer.instances.clear()
    failing = FakeOllamaClient(error=OllamaServiceError("Ollama is down", status_code=503))
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeSentenceTransformer)
    client = make_client(monkeypatch, ollama=failing)

    with caplog.at_level("WARNING", logger="app.services.rag.embeddings"):
        matrix = client.embed_batch(["a", "b"])

    assert client.backend == "sentence-transformers"
    assert np.allclose(matrix, [[1.0, 0.0], [0.0, 1.0]])
    assert len(FakeSentenceTransformer.instances) == 1
    model = FakeSentenceTransformer.instances[0]
    assert model.model_name_or_path == "test-fallback-model"
    assert model.encode_calls == [
        {
            "texts": ["a", "b"],
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
    ]
    warning = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warning) == 1
    assert warning[0].message == "Ollama embeddings unavailable (Ollama is down); using fallback."


def test_fallback_model_is_reused_across_calls(monkeypatch):
    FakeSentenceTransformer.instances.clear()
    failing = FakeOllamaClient(error=OllamaServiceError("down", status_code=503))
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeSentenceTransformer)
    client = make_client(monkeypatch, ollama=failing)

    client.embed_batch(["a"])
    client.embed_batch(["b"])

    assert len(FakeSentenceTransformer.instances) == 1
    assert client._st_model is FakeSentenceTransformer.instances[0]


def test_fallback_import_error_raises_ollama_service_error(monkeypatch):
    failing = FakeOllamaClient(error=OllamaServiceError("down", status_code=503))
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    client = make_client(monkeypatch, ollama=failing)

    with pytest.raises(OllamaServiceError, match="Embedding fallback unavailable: install sentence-transformers") as excinfo:
        client.embed_batch(["a"])

    assert excinfo.value.status_code == 503
    # The failed import must not have left a half-built model behind.
    assert client._st_model is None
