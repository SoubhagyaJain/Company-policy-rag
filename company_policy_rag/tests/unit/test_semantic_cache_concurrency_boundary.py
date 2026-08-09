from __future__ import annotations

import concurrent.futures
import time
from unittest.mock import MagicMock
import pytest

from backend.models.rag import Citation
from backend.rag.semantic_cache import (
    CachedResponse,
    SemanticCacheManager,
)


@pytest.fixture
def dummy_citation():
    return Citation(
        source_index=1,
        chunk_id="chunk_1",
        document_id="doc_1",
        source_file="policy.pdf",
        snippet="Sample policy text",
    )


@pytest.fixture
def mock_embedder():
    import hashlib
    embedder = MagicMock()
    def embed_func(text: str):
        digest = hashlib.md5(text.encode('utf-8')).digest()
        vec = [float(b) - 128.0 for b in digest]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]
    embedder.embed_text.side_effect = embed_func
    return embedder


# ============================================================================
# 1. MULTIPLE RAPID PUTS AND GETS (CONCURRENCY & STRESS)
# ============================================================================

def test_rapid_sequential_puts_and_gets(tmp_path, dummy_citation):
    """Test rapid sequential put and get operations for multiple distinct queries."""
    cache = SemanticCacheManager(
        collection_name="test_rapid_seq",
        persist_dir=tmp_path / "chroma",
    )
    cache.clear()

    for i in range(50):
        query = f"What is policy item number {i}?"
        answer = f"Answer for item {i}"
        put_ok = cache.put(query=query, answer=answer, citations=[dummy_citation])
        assert put_ok is True

        res = cache.get(query=query, threshold=0.95)
        assert res is not None
        assert res.answer == answer


def test_concurrent_multi_threaded_puts_and_gets(tmp_path, dummy_citation, mock_embedder):
    """
    Stress test multi-threaded concurrent put, get, and clear operations.
    Exposes thread-safety issues like dictionary mutation during iteration or Chroma DB lock contention.
    """
    cache = SemanticCacheManager(
        collection_name="test_concurrent_puts_gets",
        persist_dir=tmp_path / "chroma",
        embedding_service=mock_embedder,
    )
    cache.clear()

    errors = []
    num_threads = 8
    ops_per_thread = 15

    def worker(thread_id: int):
        for i in range(ops_per_thread):
            try:
                q = f"query_{thread_id}_{i}"
                ans = f"answer_{thread_id}_{i}"
                
                cache.put(query=q, answer=ans, citations=[dummy_citation])
                
                res = cache.get(query=q, threshold=0.90)
                if res is not None:
                    assert res.answer == ans
                
                other_q = f"query_{(thread_id + 1) % num_threads}_{i}"
                _ = cache.get(query=other_q, threshold=0.90)
            except Exception as e:
                errors.append((thread_id, i, e))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, tid) for tid in range(num_threads)]
        concurrent.futures.wait(futures)

    assert len(errors) == 0, f"Concurrent execution errors encountered ({len(errors)} errors): {errors[:5]}"


def test_concurrent_memory_cache_dictionary_mutation(tmp_path, dummy_citation, mock_embedder):
    """
    Specifically test in-memory fallback lookup concurrency where dictionary modification
    during iteration causes RuntimeError: dictionary changed size during iteration.
    """
    cache = SemanticCacheManager(
        collection_name="test_mem_mutation",
        persist_dir=tmp_path / "chroma",
        embedding_service=mock_embedder,
    )
    # Force collection to None to force in-memory fallback execution path
    cache._collection = None

    errors = []

    def writer():
        for i in range(50):
            try:
                cache.put(query=f"write_q_{i}", answer=f"ans_{i}", citations=[dummy_citation])
                time.sleep(0.001)
            except Exception as e:
                errors.append(("writer", i, e))

    def reader():
        for i in range(50):
            try:
                _ = cache.get(query=f"read_q_{i}", threshold=0.90)
                time.sleep(0.001)
            except Exception as e:
                errors.append(("reader", i, e))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        f1 = executor.submit(writer)
        f2 = executor.submit(reader)
        f3 = executor.submit(writer)
        f4 = executor.submit(reader)
        concurrent.futures.wait([f1, f2, f3, f4])

    assert len(errors) == 0, f"Memory cache mutation during iteration errors: {errors}"


# ============================================================================
# 2. CACHE CLEAR FOLLOWED BY GET
# ============================================================================

def test_cache_clear_followed_by_get(tmp_path, dummy_citation):
    """Verify that clear() empties both Chroma collection and memory cache, making subsequent get() return None."""
    cache = SemanticCacheManager(
        collection_name="test_clear_get",
        persist_dir=tmp_path / "chroma",
    )

    queries = ["What is dental policy?", "What is vacation policy?", "What is 401k policy?"]
    for q in queries:
        assert cache.put(query=q, answer=f"Ans for {q}", citations=[dummy_citation]) is True

    for q in queries:
        assert cache.get(query=q) is not None

    cache.clear()

    for q in queries:
        res = cache.get(query=q)
        assert res is None, f"Expected cache MISS after clear() for query '{q}', got {res}"


def test_clear_followed_by_reput_and_get(tmp_path, dummy_citation):
    """Verify cache works properly after clear followed by fresh put."""
    cache = SemanticCacheManager(
        collection_name="test_clear_reput",
        persist_dir=tmp_path / "chroma",
    )
    cache.clear()

    q1 = "Original query before clear"
    cache.put(query=q1, answer="Original answer", citations=[dummy_citation])
    cache.clear()

    assert cache.get(query=q1) is None

    q2 = "New query after clear"
    cache.put(query=q2, answer="New answer", citations=[dummy_citation])

    assert cache.get(query=q1) is None
    res2 = cache.get(query=q2)
    assert res2 is not None
    assert res2.answer == "New answer"


# ============================================================================
# 3. KB_VERSION EDGE CASES (None VS STRING VERSION VS EMPTY STRING)
# ============================================================================

def test_kb_version_none_vs_string_version(tmp_path, dummy_citation):
    """
    Test matrix of kb_version behavior:
    1. put(kb_version=None), get(kb_version=None) -> HIT
    2. put(kb_version="v1.0"), get(kb_version="v1.0") -> HIT
    3. put(kb_version="v1.0"), get(kb_version="v2.0") -> MISS
    4. put(kb_version=None), get(kb_version="v1.0") -> MISS
    """
    cache = SemanticCacheManager(
        collection_name="test_kb_matrix",
        persist_dir=tmp_path / "chroma",
    )
    cache.clear()

    # 1. Unversioned entry
    cache.put(query="unversioned query", answer="unversioned ans", citations=[dummy_citation], kb_version=None)
    assert cache.get(query="unversioned query", kb_version=None) is not None

    # 2 & 3. Versioned entry v1.0
    cache.put(query="versioned query v1", answer="v1 ans", citations=[dummy_citation], kb_version="v1.0")
    assert cache.get(query="versioned query v1", kb_version="v1.0") is not None
    assert cache.get(query="versioned query v1", kb_version="v2.0") is None

    # 4. Requesting specific version when entry was stored unversioned (None)
    assert cache.get(query="unversioned query", kb_version="v1.0") is None


def test_kb_version_versioned_entry_get_with_none(tmp_path, dummy_citation):
    """
    Test querying a versioned entry (kb_version="v1.0") with get(kb_version=None).
    Check whether get(kb_version=None) matches versioned entries or bypasses filtering.
    """
    cache = SemanticCacheManager(
        collection_name="test_kb_ver_get_none",
        persist_dir=tmp_path / "chroma",
    )
    cache.clear()

    cache.put(query="versioned item", answer="v1 ans", citations=[dummy_citation], kb_version="v1.0")
    res = cache.get(query="versioned item", kb_version=None)
    # Observe behavior: if kb_version=None in get(), does it match v1.0?
    assert res is not None, "get(kb_version=None) returns hit for versioned entry because filter is skipped when kb_version is None."


def test_kb_version_empty_string_edge_case(tmp_path, dummy_citation):
    """
    Test empty string kb_version="" vs None vs "v1.0".
    Exposes ChromaDB metadata conversion flaw: `cached_kb = metadata.get("kb_version") or None`
    which converts "" to None, causing mismatch when get(kb_version="") is called!
    """
    cache = SemanticCacheManager(
        collection_name="test_kb_empty_str",
        persist_dir=tmp_path / "chroma",
    )
    cache.clear()

    cache.put(query="empty str query", answer="empty str ans", citations=[dummy_citation], kb_version="")

    res_empty = cache.get(query="empty str query", kb_version="")
    assert res_empty is not None, "BUG: get(kb_version='') returned None due to 'or None' converting empty string to None in ChromaDB lookup!"
