"""Parallel retrieval (H6).

Dense + BM25 run concurrently inside HybridRetriever, and sub-queries run in a
thread pool via RAGPipeline._gather_hybrid_candidates. Both must be
behavior-equivalent to the previous sequential code: same fused output, same
merged candidate pool (max score per chunk, sub-query order preserved), same
degraded-fallback semantics. All tests use fakes — no models or Ollama.
"""

from __future__ import annotations

import threading

from backend.models.chunk import Chunk, ChunkMetadata
from backend.models.rag import ScoredChunk
from backend.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from backend.rag.pipeline import RAGPipeline


def _sc(cid: str, score: float, *, dense: float | None = None, sparse: float | None = None) -> ScoredChunk:
    meta = ChunkMetadata(document_id="d1", source_file="f.pdf", chunk_index=0)
    return ScoredChunk(chunk=Chunk(id=cid, text=f"text {cid}", metadata=meta), score=score,
                       dense_score=dense, sparse_score=sparse)


class _FakeDense:
    def __init__(self, hits):
        self.hits = hits
        self.called = threading.Event()

    def retrieve(self, query, top_k=25, filters=None):
        self.called.set()
        return list(self.hits)


class _FakeBM25:
    def __init__(self, hits):
        self.hits = hits
        self.called = threading.Event()

    def search(self, query, top_k=25, filters=None):
        self.called.set()
        return list(self.hits)


# ── HybridRetriever: dense + BM25 concurrent, fusion unchanged ───────────────

def test_hybrid_runs_both_and_fuses_like_rrf() -> None:
    dense = _FakeDense([_sc("a", 0.9, dense=0.9), _sc("b", 0.8, dense=0.8)])
    bm25 = _FakeBM25([_sc("b", 5.0, sparse=5.0), _sc("c", 4.0, sparse=4.0)])
    hr = HybridRetriever(dense_retriever=dense, bm25_index=bm25, reranker=object(), rrf_k=60)

    result = hr.retrieve("q", dense_top_k=10, bm25_top_k=10)

    assert dense.called.is_set() and bm25.called.is_set()
    expected = reciprocal_rank_fusion([dense.hits, bm25.hits], rrf_k=60)
    assert [r.chunk.id for r in result] == [e.chunk.id for e in expected]
    assert {r.chunk.id for r in result} == {"a", "b", "c"}


def test_hybrid_dense_only_when_bm25_empty() -> None:
    dense = _FakeDense([_sc("a", 0.9)])
    bm25 = _FakeBM25([])
    hr = HybridRetriever(dense_retriever=dense, bm25_index=bm25, reranker=object())
    result = hr.retrieve("q")
    assert [r.chunk.id for r in result] == ["a"]


def test_hybrid_bm25_only_when_dense_empty() -> None:
    dense = _FakeDense([])
    bm25 = _FakeBM25([_sc("c", 4.0)])
    hr = HybridRetriever(dense_retriever=dense, bm25_index=bm25, reranker=object())
    result = hr.retrieve("q")
    assert [r.chunk.id for r in result] == ["c"]


def test_hybrid_dense_exception_propagates() -> None:
    class _Boom:
        def retrieve(self, *a, **k):
            raise RuntimeError("chroma down")
    hr = HybridRetriever(dense_retriever=_Boom(), bm25_index=_FakeBM25([_sc("c", 1.0)]), reranker=object())
    try:
        hr.retrieve("q")
        assert False, "expected the dense error to propagate"
    except RuntimeError:
        pass


# ── _gather_hybrid_candidates: parallel sub-queries, merge semantics ─────────

class _Strategy:
    dense_top_k = 10
    bm25_top_k = 10
    rrf_k = 60


def _pipeline_stub(retrieve_impl, bm25_search_impl=None):
    """A RAGPipeline with only the attributes _gather_hybrid_candidates needs."""
    pipe = RAGPipeline.__new__(RAGPipeline)

    class _HR:
        class _BM25:
            def search(self, sq, top_k=25, filters=None):
                return bm25_search_impl(sq) if bm25_search_impl else []
        bm25_index = _BM25()

        def retrieve(self, query, dense_top_k=25, bm25_top_k=25, filters=None, rrf_k=60):
            return retrieve_impl(query)

    pipe.hybrid_retriever = _HR()
    return pipe


def test_gather_merges_all_subqueries_keeping_max_score() -> None:
    # "b" appears in two sub-queries with different scores -> keep the higher.
    per_query = {
        "q1": [_sc("a", 0.5), _sc("b", 0.4)],
        "q2": [_sc("b", 0.9), _sc("c", 0.3)],
    }
    pipe = _pipeline_stub(lambda q: per_query[q])
    cands, degraded = pipe._gather_hybrid_candidates(["q1", "q2"], None, _Strategy())
    by_id = {c.chunk.id: c.score for c in cands}
    assert set(by_id) == {"a", "b", "c"}
    assert by_id["b"] == 0.9  # max score kept across sub-queries
    assert degraded is False


def test_gather_parallel_matches_sequential_result() -> None:
    per_query = {f"q{i}": [_sc(f"c{i}", 1.0 / (i + 1)), _sc("shared", i)] for i in range(6)}
    pipe = _pipeline_stub(lambda q: per_query[q])
    subs = list(per_query)
    cands, _ = pipe._gather_hybrid_candidates(subs, None, _Strategy())
    # shared chunk keeps the highest score seen (i=5).
    assert {c.chunk.id for c in cands} == {"shared", *(f"c{i}" for i in range(6))}
    assert next(c.score for c in cands if c.chunk.id == "shared") == 5


def test_gather_dense_error_falls_back_to_bm25_and_flags_degraded() -> None:
    def _boom(q):
        raise RuntimeError("dense down")
    pipe = _pipeline_stub(_boom, bm25_search_impl=lambda q: [_sc("bm", 2.0)])
    cands, degraded = pipe._gather_hybrid_candidates(["q1"], None, _Strategy())
    assert [c.chunk.id for c in cands] == ["bm"]
    assert degraded is True


def test_gather_no_bm25_fallback_when_disabled() -> None:
    def _boom(q):
        raise RuntimeError("dense down")
    pipe = _pipeline_stub(_boom, bm25_search_impl=lambda q: [_sc("bm", 2.0)])
    cands, degraded = pipe._gather_hybrid_candidates(
        ["q1"], None, _Strategy(), bm25_fallback_on_error=False
    )
    assert cands == []
    assert degraded is False
