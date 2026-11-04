from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.embeddings.embeddings import EmbeddingService
from backend.embeddings.vector_store import ChromaVectorStore


@pytest.mark.parametrize(
    "vectors",
    [
        [[1.0, 0.0]],
        [[1.0, 0.0], [0.0, 0.0]],
        [[1.0, 0.0], [1.0]],
        [[1.0, 0.0], [float("nan"), 1]],
        [[1.0, 0.0], [True, 1]],
        [[1.0, 0.0], ["1", 1]],
    ],
)
def test_bad_provider_batches_are_rejected_before_caching(vectors):
    service = EmbeddingService(model_name="test-model")
    service._model_loaded = True
    service._model = SimpleNamespace(embed=lambda **kwargs: SimpleNamespace(embeddings=vectors))
    with pytest.raises((ValueError, RuntimeError)):
        service.embed_chunks(["first passage", "second passage"])
    assert len(service.cache) == 0


def test_missing_model_never_generates_hash_vectors_by_default():
    service = EmbeddingService(model_name="test-model")
    service._model_loaded = True
    with pytest.raises(RuntimeError, match="unavailable"):
        service.embed_chunks(["policy text"])


def test_required_persistence_cannot_fall_back_to_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.embeddings.vector_store.get_shared_chroma_client", MagicMock(side_effect=OSError("disk unavailable"))
    )
    with pytest.raises(RuntimeError, match="Persistent"):
        ChromaVectorStore(persist_dir=tmp_path, require_persistence=True)


def test_collection_rejects_another_embedding_model_without_reindexing(tmp_path):
    store = ChromaVectorStore(persist_dir=tmp_path)
    store.bind_embedding_model("model-a", "")
    with pytest.raises(RuntimeError, match="model"):
        store.bind_embedding_model("model-b", "")
    restarted = ChromaVectorStore(persist_dir=tmp_path)
    with pytest.raises(RuntimeError, match="model"):
        restarted.bind_embedding_model("model-b", "")


def test_distinct_hf_models_do_not_share_an_unkeyed_singleton(monkeypatch):
    from backend.embeddings import embeddings

    model_a, model_b = object(), object()
    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
    monkeypatch.setattr(
        embeddings, "_shared_embedding_models", {("test/model-a", "cpu"): model_a, ("test/model-b", "cpu"): model_b}
    )
    monkeypatch.setattr(embeddings, "_shared_embedding_model", model_a)
    monkeypatch.setattr(embeddings, "_shared_embedding_model_loaded", True)
    service = EmbeddingService(model_name="test/model-b")
    service._init_model()
    assert service._model is model_b
