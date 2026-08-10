from __future__ import annotations

import concurrent.futures
import hashlib
import os
import shutil
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend.models.rag import Citation
from backend.rag.semantic_cache import CachedResponse, SemanticCacheManager


class FastDummyEmbedder:
    """Fast deterministic dummy embedder for stress testing."""

    def embed_text(self, text: str) -> list[float]:
        if not text or not isinstance(text, str):
            return []
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [(float(b) / 255.0) for b in h[:32]] * 12  # 384 floats
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec] if norm > 0 else vec


def make_citation(i: int = 1) -> Citation:
    return Citation(
        source_index=i,
        chunk_id=f"chunk_{i}",
        document_id=f"doc_{i}",
        source_file=f"policy_{i}.pdf",
        snippet=f"Sample policy snippet text for citation {i}.",
    )


# ============================================================================
# 1. HIGH-CONCURRENCY READ/WRITE/CLEAR RACE CONDITIONS
# ============================================================================

def test_high_concurrency_race_conditions(tmp_path):
    """
    Stress test high-concurrency read/write/clear operations.
    Spawns 20 parallel worker threads executing 500 total operations.
    Verifies no deadlocks, race conditions, or unhandled exceptions occur.
    """
    cache_dir = tmp_path / "chroma_race"
    cache = SemanticCacheManager(
        collection_name="race_test",
        persist_dir=cache_dir,
        embedding_service=FastDummyEmbedder(),
    )

    errors = []
    success_puts = 0
    success_gets = 0
    lock = threading.Lock()

    def worker_put(thread_id: int):
        nonlocal success_puts
        for i in range(25):
            try:
                q = f"concurrency_query_{thread_id}_{i}"
                ans = f"concurrency_ans_{thread_id}_{i}"
                cit = make_citation(i)
                ok = cache.put(query=q, answer=ans, citations=[cit], model_name="m1", kb_version="v1")
                if ok:
                    with lock:
                        success_puts += 1
            except Exception as e:
                with lock:
                    errors.append(("put", thread_id, i, e))

    def worker_get(thread_id: int):
        nonlocal success_gets
        for i in range(25):
            try:
                q = f"concurrency_query_{thread_id}_{i}"
                res = cache.get(query=q, model_name="m1", kb_version="v1")
                if res is not None:
                    with lock:
                        success_gets += 1
            except Exception as e:
                with lock:
                    errors.append(("get", thread_id, i, e))

    def worker_clear():
        for _ in range(5):
            try:
                time.sleep(0.01)
                cache.clear()
            except Exception as e:
                with lock:
                    errors.append(("clear", 0, 0, e))

    threads = []
    for t_id in range(10):
        threads.append(threading.Thread(target=worker_put, args=(t_id,)))
        threads.append(threading.Thread(target=worker_get, args=(t_id,)))
    threads.append(threading.Thread(target=worker_clear))

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert len(errors) == 0, f"Encountered race condition errors: {errors}"
    assert success_puts > 0, "Expected at least some puts to succeed during race test"


# ============================================================================
# 2. SIMULATED DATABASE CORRUPTION & CHROMADB EXCEPTION INJECTION
# ============================================================================

def test_corrupted_database_file_initialization_fallback(tmp_path):
    """
    Simulate database corruption by writing garbage data into chroma.sqlite3 file before init.
    Verify SemanticCacheManager degrades gracefully to in-memory mode without raising an exception.
    """
    corrupt_dir = tmp_path / "corrupt_chroma"
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    
    # Write garbage content to fake sqlite database
    db_file = corrupt_dir / "chroma.sqlite3"
    db_file.write_bytes(b"INVALID_CORRUPTED_SQLITE_HEADER_GARBAGE_BYTES_1234567890")

    cache = SemanticCacheManager(
        collection_name="corrupt_test",
        persist_dir=corrupt_dir,
        embedding_service=FastDummyEmbedder(),
    )

    # Collection should be None due to initialization failure log fallback
    assert cache._collection is None

    # Operations must continue working seamlessly via in-memory fallback
    q = "Does corrupt storage fail the app?"
    ans = "No, gracefully falls back to memory."
    cits = [make_citation(1)]

    put_ok = cache.put(query=q, answer=ans, citations=cits)
    assert put_ok is True

    res = cache.get(query=q)
    assert res is not None
    assert res.answer == ans


def test_exception_injection_during_chroma_query(tmp_path):
    """
    Inject arbitrary exceptions (e.g. RuntimeError, IOError, DatabaseError) into Chroma collection.query.
    Verify get() catches exceptions gracefully and returns None without propagating errors.
    """
    cache_dir = tmp_path / "chroma_exc_query"
    cache = SemanticCacheManager(
        collection_name="exc_query_test",
        persist_dir=cache_dir,
        embedding_service=FastDummyEmbedder(),
    )
    cache.put(query="test query", answer="ans", citations=[make_citation()])

    mock_coll = MagicMock()
    mock_coll.count.return_value = 5
    mock_coll.query.side_effect = RuntimeError("Simulated ChromaDB internal crash")
    cache._collection = mock_coll

    # get() must not crash, should return None safely
    res = cache.get(query="test query")
    assert res is None


def test_exception_injection_during_chroma_upsert(tmp_path):
    """
    Inject arbitrary exceptions into Chroma collection.upsert during put().
    Verify put() catches exception, returns False, and does not crash the caller.
    """
    cache_dir = tmp_path / "chroma_exc_upsert"
    cache = SemanticCacheManager(
        collection_name="exc_upsert_test",
        persist_dir=cache_dir,
        embedding_service=FastDummyEmbedder(),
    )

    mock_coll = MagicMock()
    mock_coll.upsert.side_effect = OSError("Simulated Disk I/O Failure")
    cache._collection = mock_coll

    put_ok = cache.put(query="test query", answer="ans", citations=[make_citation()])
    assert put_ok is False


# ============================================================================
# 3. MANAGER RE-INSTANTIATION ACROSS RESTARTS
# ============================================================================

def test_manager_reinstantiation_disk_persistence(tmp_path):
    """
    Test local persistence integrity across application restarts.
    Instantiates Manager #1, populates items, deletes Manager #1.
    Instantiates Manager #2 pointing to the same persist_dir.
    Verifies Manager #2 successfully retrieves all items with exact citations and metadata.
    """
    persist_dir = tmp_path / "persistent_chroma"
    embedder = FastDummyEmbedder()

    # Step 1: Initialize manager 1 and store data
    mgr1 = SemanticCacheManager(
        collection_name="reinstantiate_test",
        persist_dir=persist_dir,
        embedding_service=embedder,
    )
    mgr1.clear()

    test_data = [
        ("What is the remote work policy?", "Employees may work remotely 2 days per week.", "v1.0", "qwen2.5:7b"),
        ("How many vacation days do I get?", "Standard PTO is 20 days annually.", "v1.0", "llama3.1:8b"),
        ("What is the travel reimbursement process?", "Submit receipts via concur within 30 days.", "v2.0", "qwen2.5:7b"),
    ]

    for q, ans, kb, mod in test_data:
        cit = make_citation(1)
        ok = mgr1.put(query=q, answer=ans, citations=[cit], kb_version=kb, model_name=mod)
        assert ok is True

    # Step 2: Destroy manager 1
    del mgr1

    # Step 3: Re-instantiate manager 2 on same persist_dir
    mgr2 = SemanticCacheManager(
        collection_name="reinstantiate_test",
        persist_dir=persist_dir,
        embedding_service=embedder,
    )

    # Step 4: Verify all queries return exact hits with preserved attributes
    for q, expected_ans, expected_kb, expected_mod in test_data:
        res = mgr2.get(query=q, kb_version=expected_kb, model_name=expected_mod)
        assert res is not None, f"Failed to retrieve query '{q}' after re-instantiation"
        assert res.answer == expected_ans
        assert res.kb_version == expected_kb
        assert len(res.citations) == 1
        assert res.citations[0].source_file == "policy_1.pdf"

    # Step 5: Verify rephrased query semantic similarity match across restart
    rephrased_q = "What is the policy for remote work?"
    res_rephrased = mgr2.get(query=rephrased_q, threshold=0.90, kb_version="v1.0", model_name="qwen2.5:7b")
    assert res_rephrased is not None
    assert res_rephrased.answer == "Employees may work remotely 2 days per week."


def test_reinstantiation_after_clear(tmp_path):
    """
    Verify that if a collection is cleared before restart, re-instantiated manager starts clean.
    """
    persist_dir = tmp_path / "persistent_chroma_clear"
    embedder = FastDummyEmbedder()

    mgr1 = SemanticCacheManager(
        collection_name="clear_restart_test",
        persist_dir=persist_dir,
        embedding_service=embedder,
    )
    mgr1.put(query="temporary item", answer="temp ans", citations=[make_citation()])
    mgr1.clear()
    del mgr1

    mgr2 = SemanticCacheManager(
        collection_name="clear_restart_test",
        persist_dir=persist_dir,
        embedding_service=embedder,
    )
    assert mgr2.get(query="temporary item") is None
