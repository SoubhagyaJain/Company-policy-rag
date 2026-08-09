from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from backend.models.rag import Citation
from backend.rag.semantic_cache import (
    CachedResponse,
    SemanticCache,
    SemanticCacheManager,
)
from src.config import settings


def test_config_defaults():
    assert settings.semantic_cache_enabled is True
    assert settings.semantic_cache_threshold == 0.95
    assert settings.semantic_cache_collection_name == "semantic_cache"
    assert settings.SEMANTIC_CACHE_ENABLED is True
    assert settings.SEMANTIC_CACHE_THRESHOLD == 0.95
    assert settings.SEMANTIC_CACHE_COLLECTION_NAME == "semantic_cache"


def test_cached_response_model():
    cit = Citation(
        source_index=1,
        chunk_id="chunk_1",
        document_id="doc_1",
        source_file="policy.pdf",
        snippet="Sample policy text",
    )
    resp = CachedResponse(
        answer="Sample answer",
        citations=[cit],
        similarity_score=0.97,
        distance=0.03,
        lookup_latency_ms=12.5,
        timestamp=100000.0,
        kb_version="v1.0",
    )
    assert resp.answer == "Sample answer"
    assert len(resp.citations) == 1
    assert resp.citations[0].source_file == "policy.pdf"
    assert resp.similarity_score == 0.97
    assert resp.distance == 0.03
    assert resp.lookup_latency_ms == 12.5
    assert resp.timestamp == 100000.0
    assert resp.kb_version == "v1.0"


def test_initialization(tmp_path):
    cache = SemanticCacheManager(
        collection_name="test_init_cache",
        persist_dir=tmp_path / "chroma",
    )
    assert cache.collection_name == "test_init_cache"
    assert cache.persist_dir == tmp_path / "chroma"
    assert SemanticCache is SemanticCacheManager


def test_put_and_get_exact_hit(tmp_path):
    cache = SemanticCacheManager(
        collection_name="test_exact_hit",
        persist_dir=tmp_path / "chroma",
    )
    cache.clear()

    query = "What is the remote work policy?"
    answer = "Employees may work remotely up to 2 days per week."
    citations = [
        Citation(
            source_index=1,
            chunk_id="c1",
            document_id="d1",
            source_file="remote_work.pdf",
            snippet="Remote work allowed 2 days/week",
        )
    ]

    success = cache.put(query=query, answer=answer, citations=citations)
    assert success is True

    result = cache.get(query=query, threshold=0.90)
    assert result is not None
    assert result.answer == answer
    assert len(result.citations) == 1
    assert result.citations[0].source_file == "remote_work.pdf"
    assert result.similarity_score >= 0.90


def test_similarity_hit_above_threshold(tmp_path):
    mock_embedder = MagicMock()
    mock_embedder.embed_text.side_effect = (
        lambda q: [0.9, 0.1, 0.0] if "remote" in q.lower() else [0.89, 0.11, 0.0]
    )

    cache = SemanticCacheManager(
        collection_name="test_sim_hit",
        persist_dir=tmp_path / "chroma",
        embedding_service=mock_embedder,
    )
    cache.clear()

    citations = [
        Citation(
            source_index=1,
            chunk_id="c1",
            document_id="d1",
            source_file="remote_work.pdf",
            snippet="Remote work guidelines",
        )
    ]
    cache.put(query="remote work policy", answer="2 days WFH", citations=citations)

    result = cache.get(query="remote work rules", threshold=0.95)
    assert result is not None
    assert result.answer == "2 days WFH"


def test_get_below_threshold_miss(tmp_path):
    mock_embedder = MagicMock()
    mock_embedder.embed_text.side_effect = (
        lambda q: [1.0, 0.0, 0.0] if "vacation" in q.lower() else [0.0, 1.0, 0.0]
    )

    cache = SemanticCacheManager(
        collection_name="test_cache_miss",
        persist_dir=tmp_path / "chroma",
        embedding_service=mock_embedder,
    )
    cache.clear()

    citations = [
        Citation(
            source_index=1,
            chunk_id="c1",
            document_id="d1",
            source_file="vacation.pdf",
            snippet="Vacation info",
        )
    ]
    cache.put(query="vacation policy", answer="20 days annual leave", citations=citations)

    result = cache.get(query="sick leave policy", threshold=0.95)
    assert result is None


def test_put_rejected_on_empty_answer(tmp_path):
    cache = SemanticCacheManager(
        collection_name="test_put_invalid", persist_dir=tmp_path / "chroma"
    )
    citations = [
        Citation(
            source_index=1,
            chunk_id="c1",
            document_id="d1",
            source_file="s.pdf",
            snippet="snip",
        )
    ]
    assert cache.put(query="q", answer="", citations=citations) is False
    assert cache.put(query="q", answer="   ", citations=citations) is False


def test_put_rejected_on_empty_citations(tmp_path):
    cache = SemanticCacheManager(
        collection_name="test_put_nocit", persist_dir=tmp_path / "chroma"
    )
    assert cache.put(query="q", answer="Valid answer", citations=[]) is False


def test_put_accepts_dict_citations(tmp_path):
    cache = SemanticCacheManager(
        collection_name="test_put_dict_cit", persist_dir=tmp_path / "chroma"
    )
    cache.clear()
    cit_dict = {
        "source_index": 1,
        "chunk_id": "c1",
        "document_id": "d1",
        "source_file": "s.pdf",
        "snippet": "snip",
    }
    assert cache.put(query="dict query", answer="Valid answer", citations=[cit_dict]) is True
    res = cache.get("dict query")
    assert res is not None
    assert res.citations[0].source_file == "s.pdf"


def test_put_exception_safety(monkeypatch, tmp_path):
    cache = SemanticCacheManager(
        collection_name="test_safe_put", persist_dir=tmp_path / "chroma"
    )
    citations = [
        Citation(
            source_index=1,
            chunk_id="c1",
            document_id="d1",
            source_file="s.pdf",
            snippet="snip",
        )
    ]

    monkeypatch.setattr(
        cache.embedding_service,
        "embed_text",
        MagicMock(side_effect=RuntimeError("Embedding model crashed")),
    )
    assert cache.put(query="q", answer="Ans", citations=citations) is False


def test_version_matching_and_mismatching(tmp_path):
    cache = SemanticCacheManager(
        collection_name="test_kb_ver", persist_dir=tmp_path / "chroma"
    )
    cache.clear()
    citations = [
        Citation(
            source_index=1,
            chunk_id="c1",
            document_id="d1",
            source_file="s.pdf",
            snippet="snip",
        )
    ]

    cache.put(query="policy q", answer="Policy A", citations=citations, kb_version="v1.0")

    assert cache.get("policy q", kb_version="v1.0") is not None
    assert cache.get("policy q", kb_version="v2.0") is None


def test_clear_cache(tmp_path):
    cache = SemanticCacheManager(
        collection_name="test_clear", persist_dir=tmp_path / "chroma"
    )
    citations = [
        Citation(
            source_index=1,
            chunk_id="c1",
            document_id="d1",
            source_file="s.pdf",
            snippet="snip",
        )
    ]

    cache.put(query="q1", answer="a1", citations=citations)
    cache.clear()
    assert cache.get("q1") is None


def test_kb_version_empty_string_matching(tmp_path):
    """
    Verifies that kb_version="" stores and retrieves correctly without false mismatch,
    while mismatching requests (e.g. kb_version="v2.0") return None (MISS).
    """
    cache = SemanticCacheManager(
        collection_name="test_kb_ver_empty", persist_dir=tmp_path / "chroma"
    )
    cache.clear()
    citations = [
        Citation(
            source_index=1,
            chunk_id="c1",
            document_id="d1",
            source_file="s.pdf",
            snippet="snip",
        )
    ]

    cache.put(query="policy empty", answer="Policy Text", citations=citations, kb_version="")

    # Matching kb_version="" should succeed as HIT
    hit = cache.get("policy empty", kb_version="")
    assert hit is not None
    assert hit.answer == "Policy Text"
    assert hit.kb_version == ""

    # Mismatching kb_version="v2.0" should fail as MISS
    miss = cache.get("policy empty", kb_version="v2.0")
    assert miss is None


def test_multithreaded_concurrent_access(tmp_path):
    """
    Verifies that multi-threaded concurrent reads, writes, and clears
    do not raise RuntimeError or race conditions on memory cache.
    """
    import threading
    import time

    cache = SemanticCacheManager(
        collection_name="test_concurrent", persist_dir=tmp_path / "chroma"
    )
    cache.clear()
    citations = [
        Citation(
            source_index=1,
            chunk_id="c1",
            document_id="d1",
            source_file="s.pdf",
            snippet="snip",
        )
    ]

    errors: list[Exception] = []
    stop_event = threading.Event()

    def writer(worker_id: int):
        i = 0
        while not stop_event.is_set():
            try:
                cache.put(
                    query=f"concurrent query {worker_id}_{i}",
                    answer=f"answer_{worker_id}_{i}",
                    citations=citations,
                    kb_version="v1",
                )
                i += 1
                if i >= 50:
                    break
            except Exception as exc:
                errors.append(exc)
                break

    def reader():
        while not stop_event.is_set():
            try:
                cache.get("concurrent query 0_0", threshold=0.5)
            except Exception as exc:
                errors.append(exc)
                break

    def remover():
        while not stop_event.is_set():
            try:
                cache.clear()
            except Exception as exc:
                errors.append(exc)
                break

    threads = [
        threading.Thread(target=writer, args=(1,)),
        threading.Thread(target=writer, args=(2,)),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
        threading.Thread(target=remover),
    ]

    for t in threads:
        t.start()

    time.sleep(0.5)
    stop_event.set()

    for t in threads:
        t.join(timeout=2.0)

    assert len(errors) == 0, f"Concurrent thread errors: {errors}"
