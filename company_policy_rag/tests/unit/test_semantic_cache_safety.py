from __future__ import annotations

from unittest.mock import MagicMock

from backend.embeddings.vector_store import ChromaVectorStore
from backend.models.chunk import Chunk, ChunkMetadata
from backend.models.rag import Citation
from backend.rag.semantic_cache import SemanticCacheManager


def _citation() -> Citation:
    return Citation(
        source_index=1,
        chunk_id="chunk_source",
        document_id="doc_source",
        source_file="policy.pdf",
        snippet="Authoritative policy evidence.",
    )


def _constant_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_text.return_value = [1.0, 0.0, 0.0]
    return embedder


def test_semantic_cache_rejects_conflicting_numeric_identifiers(tmp_path):
    cache = SemanticCacheManager(
        collection_name="numeric_entity_guard",
        persist_dir=tmp_path / "chroma",
        embedding_service=_constant_embedder(),
    )
    cache._collection = None
    assert cache.put("What is Article 368?", "Amendment procedure.", [_citation()])

    assert cache.get("What is Article 370?", threshold=0.5) is None


def test_semantic_cache_rejects_conflicting_audiences(tmp_path):
    cache = SemanticCacheManager(
        collection_name="audience_guard",
        persist_dir=tmp_path / "chroma",
        embedding_service=_constant_embedder(),
    )
    cache._collection = None
    assert cache.put("Can employees work remotely?", "Employees may work remotely.", [_citation()])

    assert cache.get("Can contractors work remotely?", threshold=0.5) is None


def test_semantic_cache_isolated_by_prompt_and_retrieval_context(tmp_path):
    cache = SemanticCacheManager(
        collection_name="context_guard",
        persist_dir=tmp_path / "chroma",
        embedding_service=_constant_embedder(),
    )
    cache._collection = None
    assert cache.put(
        "What is the leave policy?",
        "Leave answer.",
        [_citation()],
        prompt_version="v1",
        cache_context='{"scope":"global"}',
    )

    assert cache.get(
        "What is the leave policy?",
        prompt_version="v2",
        cache_context='{"scope":"global"}',
    ) is None
    assert cache.get(
        "What is the leave policy?",
        prompt_version="v1",
        cache_context='{"scope":"current_document"}',
    ) is None


def test_corpus_mutation_invalidates_semantic_answer_cache(tmp_path):
    vector_store = ChromaVectorStore(
        collection_name="corpus_version_source",
        persist_dir=str(tmp_path / "chroma"),
    )
    vector_store.clear()
    first = Chunk(
        id="chunk_one",
        text="Initial policy text.",
        metadata=ChunkMetadata(document_id="doc_one", source_file="one.txt"),
        embedding=[1.0, 0.0, 0.0],
    )
    vector_store.add_chunks([first])

    cache = SemanticCacheManager(
        collection_name="corpus_version_cache",
        persist_dir=tmp_path / "chroma",
        vector_store=vector_store,
        embedding_service=_constant_embedder(),
    )
    cache.clear()
    assert cache.put("What is the policy?", "Initial answer.", [_citation()])
    assert cache.get("What is the policy?") is not None

    second = Chunk(
        id="chunk_two",
        text="New policy text that changes the knowledge base.",
        metadata=ChunkMetadata(document_id="doc_two", source_file="two.txt"),
        embedding=[1.0, 0.0, 0.0],
    )
    vector_store.add_chunks([second])

    assert cache.get("What is the policy?") is None
