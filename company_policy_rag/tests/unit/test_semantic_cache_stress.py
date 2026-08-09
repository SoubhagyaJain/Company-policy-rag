from __future__ import annotations

import copy
import hashlib
import threading
import time
from unittest.mock import MagicMock
import pytest

from backend.models.rag import Citation
from backend.rag.semantic_cache import (
    CachedResponse,
    SemanticCacheManager,
)


class FastDummyEmbedder:
    """Fast deterministic dummy embedder to avoid loading transformer models during stress testing."""

    def embed_text(self, text: str) -> list[float]:
        if not text or not isinstance(text, str):
            return []
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Produce a normalized 384-dim vector float representation
        vec = [(float(b) / 255.0) for b in h[:32]] * 12  # 384 floats
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec] if norm > 0 else vec


def make_valid_citation() -> Citation:
    return Citation(
        source_index=1,
        chunk_id="chunk_100",
        document_id="doc_100",
        source_file="policy_document.pdf",
        snippet="Employees get 20 days of paid leave per year.",
    )


# ============================================================================
# 1. Stress Test: Invalid Input Structures
# ============================================================================


@pytest.mark.parametrize(
    "invalid_query",
    [
        None,
        123,
        12.34,
        True,
        False,
        ["query_in_list"],
        {"query_key": "val"},
        b"bytes_query",
        (1, 2, 3),
    ],
)
def test_invalid_input_structures_query_get_and_put(tmp_path, invalid_query):
    cache = SemanticCacheManager(
        collection_name="test_invalid_query",
        persist_dir=tmp_path / "chroma",
        embedding_service=FastDummyEmbedder(),
    )

    # get() with invalid query should safely return None without raising an exception
    res = cache.get(query=invalid_query)
    assert res is None

    # put() with invalid query should safely return False without crashing
    cits = [make_valid_citation()]
    put_res = cache.put(query=invalid_query, answer="Valid answer", citations=cits)
    assert put_res is False


@pytest.mark.parametrize(
    "invalid_answer",
    [
        None,
        123,
        12.34,
        True,
        False,
        ["answer_in_list"],
        {"ans_key": "val"},
        b"bytes_answer",
        (1, 2, 3),
    ],
)
def test_invalid_input_structures_answer_put(tmp_path, invalid_answer):
    cache = SemanticCacheManager(
        collection_name="test_invalid_ans",
        persist_dir=tmp_path / "chroma",
        embedding_service=FastDummyEmbedder(),
    )
    cits = [make_valid_citation()]
    put_res = cache.put(query="valid query", answer=invalid_answer, citations=cits)
    assert put_res is False


@pytest.mark.parametrize(
    "invalid_citations",
    [
        None,
        "not_a_list",
        123,
        12.34,
        True,
        {"source_file": "doc.pdf"},
        (make_valid_citation(),),
        [],
    ],
)
def test_invalid_input_structures_citations_put(tmp_path, invalid_citations):
    cache = SemanticCacheManager(
        collection_name="test_invalid_cits",
        persist_dir=tmp_path / "chroma",
        embedding_service=FastDummyEmbedder(),
    )
    put_res = cache.put(query="valid query", answer="valid answer", citations=invalid_citations)
    assert put_res is False


@pytest.mark.parametrize(
    "invalid_element_list",
    [
        [123],
        ["not_a_citation"],
        [None],
        [{"invalid_key_for_citation": 999}],
        [{"source_index": "not_an_int", "chunk_id": "c1", "document_id": "d1", "source_file": "f.pdf", "snippet": "s"}],
    ],
)
def test_invalid_citation_elements_in_list(tmp_path, invalid_element_list):
    cache = SemanticCacheManager(
        collection_name="test_invalid_cit_elems",
        persist_dir=tmp_path / "chroma",
        embedding_service=FastDummyEmbedder(),
    )
    put_res = cache.put(query="valid query", answer="valid answer", citations=invalid_element_list)
    assert put_res is False


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        "0.95",
        [0.95],
        {"threshold": 0.95},
        object(),
    ],
)
def test_invalid_threshold_types_in_get(tmp_path, invalid_threshold):
    cache = SemanticCacheManager(
        collection_name="test_invalid_thresh",
        persist_dir=tmp_path / "chroma",
        embedding_service=FastDummyEmbedder(),
    )
    cits = [make_valid_citation()]
    cache.put(query="test threshold query", answer="answer text", citations=cits)

    # Calling get() with invalid threshold type should be caught by exception handler and return None
    res = cache.get(query="test threshold query", threshold=invalid_threshold)
    assert res is None


@pytest.mark.parametrize(
    "invalid_metadata",
    [
        "not_a_dict",
        123,
        [1, 2, 3],
        True,
    ],
)
def test_invalid_metadata_structure_in_put(tmp_path, invalid_metadata):
    cache = SemanticCacheManager(
        collection_name="test_invalid_meta",
        persist_dir=tmp_path / "chroma",
        embedding_service=FastDummyEmbedder(),
    )
    cits = [make_valid_citation()]
    # Passing non-dict metadata should be safely caught and return False
    put_res = cache.put(
        query="valid query for meta",
        answer="valid answer",
        citations=cits,
        metadata=invalid_metadata,
    )
    assert put_res is False


def test_metadata_with_complex_non_primitive_values(tmp_path):
    cache = SemanticCacheManager(
        collection_name="test_complex_meta",
        persist_dir=tmp_path / "chroma",
        embedding_service=FastDummyEmbedder(),
    )
    cits = [make_valid_citation()]
    complex_meta = {
        "str_val": "ok",
        "int_val": 42,
        "float_val": 3.14,
        "bool_val": True,
        "list_val": [1, 2, 3],  # non-primitive, should be skipped
        "dict_val": {"nested": "value"},  # non-primitive, should be skipped
        "obj_val": object(),  # non-primitive, should be skipped
    }
    # put() should succeed by serializing allowed primitive metadata and skipping non-primitives
    put_res = cache.put(
        query="query with complex metadata",
        answer="answer text",
        citations=cits,
        metadata=complex_meta,
    )
    assert put_res is True

    res = cache.get(query="query with complex metadata")
    assert res is not None
    assert res.answer == "answer text"


# ============================================================================
# 2. Stress Test: Non-String Keys and Edge-Case Types
# ============================================================================


def test_non_string_keys_in_metadata_dict(tmp_path):
    cache = SemanticCacheManager(
        collection_name="test_non_string_meta_keys",
        persist_dir=tmp_path / "chroma",
        embedding_service=FastDummyEmbedder(),
    )
    cits = [make_valid_citation()]
    bad_keys_meta = {
        123: "integer_key",
        (1, 2): "tuple_key",
        "valid_key": "string_key",
    }
    # Should safely handle non-string keys without raising unhandled exception
    put_res = cache.put(
        query="query for bad keys meta",
        answer="answer text",
        citations=cits,
        metadata=bad_keys_meta,
    )
    # Even if bad keys cause iteration error or Chroma error, put() must return bool (False/True) without crashing
    assert isinstance(put_res, bool)


def test_edge_case_query_strings(tmp_path):
    cache = SemanticCacheManager(
        collection_name="test_edge_query_strings",
        persist_dir=tmp_path / "chroma",
        embedding_service=FastDummyEmbedder(),
    )
    cits = [make_valid_citation()]

    # Unicode, zero-width space, null bytes
    unicode_query = "What is the policy for 🌍 & 🚀 employees? \u200b \x00"
    assert cache.put(query=unicode_query, answer="Global policy", citations=cits) is True
    res = cache.get(query=unicode_query)
    assert res is not None
    assert res.answer == "Global policy"

    # Extremely long string (1 MB)
    huge_query = "policy " + "x" * (1024 * 1024)
    put_huge = cache.put(query=huge_query, answer="Huge answer", citations=cits)
    assert isinstance(put_huge, bool)


# ============================================================================
# 3. Stress Test: Missing Attributes & Malformed Chroma Outputs
# ============================================================================


def test_malformed_chroma_metadata_missing_fields(tmp_path):
    mock_collection = MagicMock()
    # Mock Chroma returning metadata with None or missing values
    mock_collection.count.return_value = 1
    mock_collection.query.return_value = {
        "ids": [["entry_1"]],
        "metadatas": [[{"answer": None, "citations_json": "invalid_json_text{", "timestamp": "invalid_ts"}]],
        "distances": [[0.01]],
    }

    cache = SemanticCacheManager(
        collection_name="test_malformed_chroma",
        persist_dir=tmp_path / "chroma",
        embedding_service=FastDummyEmbedder(),
    )
    cache._collection = mock_collection

    # get() must handle malformed metadata gracefully without throwing unhandled exceptions
    res = cache.get(query="query against malformed chroma")
    # Pydantic or json error should cause get() to catch exception or handle missing citations gracefully
    assert res is None or isinstance(res, CachedResponse)


def test_chroma_query_missing_distances_key(tmp_path):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 1
    # Mock missing 'distances' key entirely from results dict
    mock_collection.query.return_value = {
        "ids": [["entry_1"]],
        "metadatas": [[{"answer": "ans", "citations_json": "[]"}]],
    }

    cache = SemanticCacheManager(
        collection_name="test_missing_distances",
        persist_dir=tmp_path / "chroma",
        embedding_service=FastDummyEmbedder(),
    )
    cache._collection = mock_collection

    # get() should catch KeyError/TypeError safely and return None
    res = cache.get(query="query with missing distances")
    assert res is None


# ============================================================================
# 4. Stress Test: Rapid Sequential Cache Mutation
# ============================================================================


def test_rapid_sequential_overwrites_same_key(tmp_path):
    cache = SemanticCacheManager(
        collection_name="test_rapid_overwrites",
        persist_dir=tmp_path / "chroma",
        embedding_service=FastDummyEmbedder(),
    )
    cits = [make_valid_citation()]
    query = "What is the parental leave policy?"

    # Perform 200 rapid sequential put operations on the same key
    for i in range(200):
        answer_text = f"Parental leave version {i}"
        ok = cache.put(query=query, answer=answer_text, citations=cits)
        assert ok is True

    # Final get should reflect the latest state
    res = cache.get(query=query)
    assert res is not None
    assert res.answer == "Parental leave version 199"


def test_rapid_sequential_put_clear_cycles(tmp_path):
    cache = SemanticCacheManager(
        collection_name="test_rapid_put_clear",
        persist_dir=tmp_path / "chroma",
        embedding_service=FastDummyEmbedder(),
    )
    cits = [make_valid_citation()]

    for i in range(50):
        query = f"query_{i}"
        answer = f"answer_{i}"
        assert cache.put(query=query, answer=answer, citations=cits) is True
        res = cache.get(query=query)
        assert res is not None
        assert res.answer == answer
        cache.clear()
        assert cache.get(query=query) is None


def test_external_citation_mutation_isolation(tmp_path):
    """
    Verifies that modifying a citation object after calling put()
    does not corrupt or mutate the cached citation data.
    """
    cache = SemanticCacheManager(
        collection_name="test_cit_mutation_isolation",
        persist_dir=tmp_path / "chroma",
        embedding_service=FastDummyEmbedder(),
    )
    cit = make_valid_citation()
    cits = [cit]

    cache.put(query="isolation test query", answer="isolation answer", citations=cits)

    # Mutate original citation object externally
    cit.snippet = "MUTATED EXTERNAL SNIPPET"
    cit.source_file = "MUTATED_FILE.pdf"

    # Fetch from cache
    res = cache.get(query="isolation test query")
    assert res is not None
    assert res.citations[0].snippet == "Employees get 20 days of paid leave per year."
    assert res.citations[0].source_file == "policy_document.pdf"


# ============================================================================
# 5. Stress Test: High Concurrency Multi-Threaded Mutation
# ============================================================================


def test_high_concurrency_rapid_mutation(tmp_path):
    cache = SemanticCacheManager(
        collection_name="test_high_concurrency_stress",
        persist_dir=tmp_path / "chroma",
        embedding_service=FastDummyEmbedder(),
    )
    cits = [make_valid_citation()]

    errors: list[Exception] = []
    stop_event = threading.Event()

    def worker_put(thread_idx: int):
        count = 0
        while not stop_event.is_set() and count < 100:
            try:
                cache.put(
                    query=f"stress query {thread_idx}_{count}",
                    answer=f"stress answer {thread_idx}_{count}",
                    citations=cits,
                    kb_version=f"v{thread_idx}",
                )
                count += 1
            except Exception as e:
                errors.append(e)

    def worker_get(thread_idx: int):
        count = 0
        while not stop_event.is_set() and count < 100:
            try:
                cache.get(
                    query=f"stress query {thread_idx}_{count}",
                    threshold=0.8,
                )
                count += 1
            except Exception as e:
                errors.append(e)

    def worker_clear():
        while not stop_event.is_set():
            try:
                cache.clear()
                time.sleep(0.01)
            except Exception as e:
                errors.append(e)

    threads = []
    for i in range(4):
        threads.append(threading.Thread(target=worker_put, args=(i,)))
        threads.append(threading.Thread(target=worker_get, args=(i,)))
    threads.append(threading.Thread(target=worker_clear))

    for t in threads:
        t.start()

    time.sleep(0.5)
    stop_event.set()

    for t in threads:
        t.join(timeout=3.0)

    assert len(errors) == 0, f"Encountered thread errors during high concurrency stress: {errors}"
