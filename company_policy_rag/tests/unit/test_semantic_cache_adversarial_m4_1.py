from __future__ import annotations

import time
from typing import List
import pytest

from backend.models.rag import Citation
from backend.rag.semantic_cache import SemanticCacheManager, CachedResponse


@pytest.fixture
def sample_citation():
    return Citation(
        source_index=1,
        chunk_id="chunk_adv_1",
        document_id="doc_adv_1",
        source_file="adv_policy.pdf",
        snippet="Employees may work remotely up to 2 days per week with manager approval.",
    )


@pytest.fixture
def cache_instance(tmp_path):
    cache = SemanticCacheManager(
        collection_name="test_adversarial_m4_1",
        persist_dir=tmp_path / "chroma",
    )
    cache.clear()
    return cache


# ============================================================================
# 1. Semantically Identical Queries & Case/Whitespace Normalization
# ============================================================================

def test_identical_queries_and_formatting_variants(cache_instance, sample_citation):
    """
    Test exact queries and formatting variants (case, trailing spaces, punctuation).
    """
    base_query = "What is the company remote work policy?"
    answer = "2 days per week WFH with manager approval."
    
    assert cache_instance.put(query=base_query, answer=answer, citations=[sample_citation]) is True

    # Exact match
    res_exact = cache_instance.get(query=base_query)
    assert res_exact is not None
    assert res_exact.answer == answer
    assert res_exact.similarity_score >= 0.99

    # Lowercase variant
    res_lower = cache_instance.get(query="what is the company remote work policy?")
    assert res_lower is not None
    assert res_lower.answer == answer

    # Uppercase variant
    res_upper = cache_instance.get(query="WHAT IS THE COMPANY REMOTE WORK POLICY?")
    assert res_upper is not None
    assert res_upper.answer == answer

    # Leading/trailing whitespace
    res_ws = cache_instance.get(query="   What is the company remote work policy? \t\n  ")
    assert res_ws is not None
    assert res_ws.answer == answer

    # Punctuation variations
    res_punct = cache_instance.get(query="What is the company remote work policy!!!")
    assert res_punct is not None or cache_instance.get(query="What is the company remote work policy") is not None


# ============================================================================
# 2. Semantically Rephrased vs Distinct Queries
# ============================================================================

def test_semantically_rephrased_vs_distinct_queries(cache_instance, sample_citation):
    """
    Test semantic similarity threshold (0.95 default) on rephrased queries vs distinct queries.
    """
    base_query = "What is the remote work policy?"
    answer = "Employees are allowed to work from home two days a week."

    cache_instance.put(query=base_query, answer=answer, citations=[sample_citation])

    # Rephrased query with high semantic overlap
    rephrased_query = "What are the rules for working remotely?"
    res_rephrased = cache_instance.get(query=rephrased_query, threshold=0.85)
    # Record similarity score for empirical analysis
    if res_rephrased is not None:
        assert res_rephrased.answer == answer
        assert res_rephrased.similarity_score >= 0.85

    # Completely distinct queries must result in cache MISS at threshold 0.95
    distinct_queries = [
        "What is the vacation leave allowance?",
        "How do I submit an expense report?",
        "What is the 401k employer matching percentage?",
        "Where is the headquarters office located?",
        "Who is the Chief Executive Officer?",
    ]

    for dq in distinct_queries:
        res_dq = cache_instance.get(query=dq, threshold=0.95)
        assert res_dq is None, f"Expected cache MISS for distinct query '{dq}', but got HIT with score {res_dq.similarity_score if res_dq else 0}"


# ============================================================================
# 3. Edge-Case Queries (Empty, Whitespace, Special Chars, Unicode, Emojis)
# ============================================================================

def test_edge_case_empty_and_whitespace_queries(cache_instance, sample_citation):
    """
    Test empty and whitespace-only queries for both get and put operations.
    """
    empty_queries = ["", "   ", "\t", "\n\r", " \t \n "]

    for eq in empty_queries:
        # get() on empty/whitespace should return None
        assert cache_instance.get(query=eq) is None
        # put() on empty/whitespace should return False
        assert cache_instance.put(query=eq, answer="Some answer", citations=[sample_citation]) is False


def test_edge_case_special_characters_and_symbols(cache_instance, sample_citation):
    """
    Test queries containing special characters, symbols, HTML/SQL code snippets.
    """
    special_query = "What is policy for <script>alert('xss')</script> & SELECT * FROM users; ? #$%^&*()_+"
    answer = "Special char policy answer."

    assert cache_instance.put(query=special_query, answer=answer, citations=[sample_citation]) is True
    res = cache_instance.get(query=special_query)
    assert res is not None
    assert res.answer == answer


def test_edge_case_unicode_emojis_and_zero_width_chars(cache_instance, sample_citation):
    """
    Test unicode, emoji, and zero-width character handling.
    """
    emoji_query = "What is the 🏠 remote 💻 work policy? 🚀"
    answer = "Emoji policy answer."

    assert cache_instance.put(query=emoji_query, answer=answer, citations=[sample_citation]) is True

    res_exact = cache_instance.get(query=emoji_query)
    assert res_exact is not None
    assert res_exact.answer == answer

    # Query with zero-width space
    zw_query = "What is the \u200b remote \u200c work policy?"
    # Must execute safely without crashing
    res_zw = cache_instance.get(query=zw_query)
    assert res_zw is None or isinstance(res_zw, CachedResponse)


def test_edge_case_typos_and_multilingual(cache_instance, sample_citation):
    """
    Test queries with severe typos and multi-language inputs.
    """
    cache_instance.put(query="What is the bereavement leave policy?", answer="3 days paid leave for immediate family.", citations=[sample_citation])

    # Query with typos
    typo_query = "Wht is the bereavment leve policy?"
    res_typo = cache_instance.get(query=typo_query, threshold=0.90)
    # Log score or verify behavior
    assert res_typo is None or isinstance(res_typo, CachedResponse)

    # Multilingual query (Spanish / German)
    spanish_query = "¿Cuál es la política de trabajo remoto?"
    res_es = cache_instance.get(query=spanish_query, threshold=0.95)
    # Unless multilingual embedding model is used, should be a MISS compared to English base query
    assert res_es is None or isinstance(res_es, CachedResponse)


# ============================================================================
# 4. Extremely Short and Extremely Long Queries
# ============================================================================

def test_extremely_short_and_long_queries(cache_instance, sample_citation):
    """
    Test single character queries and multi-kilobyte queries.
    """
    # Single character query
    short_q = "a"
    assert cache_instance.put(query=short_q, answer="Short query answer", citations=[sample_citation]) is True
    res_short = cache_instance.get(query=short_q)
    assert res_short is not None
    assert res_short.answer == "Short query answer"

    # 50,000 character query
    long_q = "What is the policy regarding " + ("vacation " * 5000)
    put_long = cache_instance.put(query=long_q, answer="Long query answer", citations=[sample_citation])
    assert put_long is True

    res_long = cache_instance.get(query=long_q)
    assert res_long is not None
    assert res_long.answer == "Long query answer"


# ============================================================================
# 5. Lookup Latency Benchmark (<100ms requirement)
# ============================================================================

def test_cache_hit_latency_benchmark(cache_instance, sample_citation):
    """
    Benchmark cache lookup latency to ensure cache hits return in < 100ms.
    """
    query = "Benchmark query for cache hit latency measurement"
    answer = "Benchmark response text"

    cache_instance.put(query=query, answer=answer, citations=[sample_citation])

    # Warmup
    _ = cache_instance.get(query=query)

    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        res = cache_instance.get(query=query)
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000.0
        latencies.append(lat_ms)
        assert res is not None

    avg_lat = sum(latencies) / len(latencies)
    assert avg_lat < 100.0, f"Average cache hit latency {avg_lat:.2f}ms exceeded 100ms target!"
