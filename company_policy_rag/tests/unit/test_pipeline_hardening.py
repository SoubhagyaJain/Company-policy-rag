"""Pipeline boundary and resource regressions; no live models or services."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.models.chunk import Chunk, ChunkMetadata
from backend.models.rag import RAGResponse, RAGTrace, ScoredChunk
from backend.rag.pipeline import RAGPipeline, _extract_requested_numbered_list, _format_history_for_prompt
from backend.rag.query_context import QueryContext
from backend.rag.scope_resolver import DocumentRetrievalScope
from src.config import settings


def _sc(cid, text="Evidence", document_id="doc"):
    return ScoredChunk(
        chunk=Chunk(id=cid, text=text, metadata=ChunkMetadata(document_id=document_id, source_file="guide.pdf")),
        score=1.0,
    )


@pytest.mark.parametrize("max_turns,max_chars", [(0, 100), (-1, 100), (2, 0), (2, -1)])
def test_disabled_history_budget_is_empty(max_turns, max_chars):
    assert _format_history_for_prompt([{"role": "user", "content": "Hello"}], max_turns, max_chars) == ""


def test_history_truncation_respects_content_budget():
    result = _format_history_for_prompt([{"role": "user", "content": "x" * 100}], max_chars=10)
    assert len(result.removeprefix("Recent Conversation History:\n").removesuffix("\n\n")) <= 10


def test_enumeration_does_not_combine_different_documents():
    chunks = [_sc("a", "1. Alpha", "doc-a"), _sc("b", "2. Beta", "doc-b")]
    assert _extract_requested_numbered_list("List the two methods", chunks) is None


def test_enumeration_continuation_in_same_document():
    chunks = [_sc("a", "1. Alpha"), _sc("b", "2. Beta")]
    assert _extract_requested_numbered_list("List the two methods", chunks) == ["Alpha", "Beta"]


@pytest.mark.parametrize("parts,top_n,expected", [([], 2, ["0", "1"]), (["q"], 0, []), (["q"], -1, [])])
def test_reranking_empty_parts_and_nonpositive_budget(parts, top_n, expected):
    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe.reranker = MagicMock()
    chunks = [_sc(str(i)) for i in range(3)]
    assert [s.chunk.id for s in pipe._rerank_for_parts(parts, chunks, top_n, 0.5)] == expected
    pipe.reranker.rerank.assert_not_called()


def test_reranking_fills_odd_budget_and_deduplicates_parts():
    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe.reranker = MagicMock()
    chunks = [_sc(str(i)) for i in range(6)]
    pipe.reranker.rerank.side_effect = lambda q, c, top_n, **kw: (chunks[:3] if q == "a" else chunks[3:])[:top_n]
    result = pipe._rerank_for_parts(["a", "b", "a"], chunks, 5, 0.5)
    assert [s.chunk.id for s in result] == ["0", "3", "1", "4", "2"]
    assert pipe.reranker.rerank.call_count == 2


def test_duplicate_subqueries_run_once(monkeypatch):
    monkeypatch.setattr(settings, "retrieval_max_workers", 1)
    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe.hybrid_retriever = SimpleNamespace(retrieve=MagicMock(return_value=[_sc("a")]))
    strategy = SimpleNamespace(dense_top_k=2, bm25_top_k=2, rrf_k=60)
    result, degraded = pipe._gather_hybrid_candidates(["q", "q"], None, strategy)
    assert len(result) == 1 and not degraded
    pipe.hybrid_retriever.retrieve.assert_called_once()


def test_warmup_failure_still_uses_keyword_fallback():
    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe.hybrid_retriever = SimpleNamespace(
        _warm_dense_model=MagicMock(side_effect=RuntimeError("model unavailable")),
        retrieve=MagicMock(side_effect=RuntimeError("dense unavailable")),
        bm25_index=SimpleNamespace(search=MagicMock(return_value=[_sc("keyword")])),
    )
    result, degraded = pipe._gather_hybrid_candidates(
        ["q"], None, SimpleNamespace(dense_top_k=2, bm25_top_k=2, rrf_k=60)
    )
    assert [s.chunk.id for s in result] == ["keyword"]
    assert degraded


@pytest.mark.parametrize("soft_filter,degraded", [(False, False), (True, False), (False, True)])
def test_filter_relaxation_and_cache_boundaries(monkeypatch, soft_filter, degraded):
    from backend.rag import pipeline as module

    cache = MagicMock()
    cache.get.return_value = None
    monkeypatch.setattr(module, "get_retrieval_cache", lambda: cache)
    monkeypatch.setattr(settings, "retrieval_cache_enabled", True)
    monkeypatch.setattr(settings, "enable_filter_fallback_relaxation", True)
    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe.hybrid_retriever = SimpleNamespace()
    pipe._gather_hybrid_candidates = MagicMock(side_effect=[([_sc("a")] if degraded else [], degraded), ([_sc("a")], False)])
    filters = {"document_id": "doc"}
    if soft_filter:
        filters["category"] = "policy"
    ctx = QueryContext(
        user_query="q", applied_filters=filters,
        current_strategy=SimpleNamespace(dense_top_k=2, bm25_top_k=2, rrf_k=60, rerank_top_n=2, min_score_ratio=0.5),
        scope_decision=SimpleNamespace(scope=DocumentRetrievalScope.GLOBAL, is_structural_query=False),
        rewrite_res=SimpleNamespace(rewritten_query="q"), is_fast_path=True,
        thinking_sm=MagicMock(),
    )
    pipe._stage_retrieve(ctx, "")
    assert pipe._gather_hybrid_candidates.call_count == (2 if soft_filter else 1)
    cache.set.assert_not_called()
    assert ctx.filter_relaxed == soft_filter


def test_closing_document_stream_cancels_worker_without_external_token():
    pipe = RAGPipeline.__new__(RAGPipeline)
    stopped = threading.Event()
    release = threading.Event()

    def query(**kwargs):
        cancel = kwargs["cancel_event"]
        try:
            while not release.is_set() and not (cancel is not None and cancel.is_set()):
                kwargs["stream_callback"]("token ")
            return RAGResponse(query="q", answer="token", trace=RAGTrace(query="q"))
        finally:
            stopped.set()

    pipe.query = query
    stream = pipe._stream_query_internal("q", thinking_detail_level="off")
    try:
        assert next(stream)["type"] == "retrieval_done"
        stream.close()
        assert stopped.wait(1), "Closing the consumer must release the producer"
    finally:
        release.set()


def test_token_queue_applies_backpressure(monkeypatch):
    from backend.rag import pipeline as module

    queues = []
    queue_type = module.queue.Queue

    def make_queue(*args, **kwargs):
        q = queue_type(*args, **kwargs)
        queues.append(q)
        return q

    monkeypatch.setattr(module.queue, "Queue", make_queue)
    pipe = RAGPipeline.__new__(RAGPipeline)
    stopped = threading.Event()
    release = threading.Event()

    def query(**kwargs):
        try:
            while not release.is_set() and not kwargs["cancel_event"].is_set():
                kwargs["stream_callback"]("token")
            return RAGResponse(query="q", answer="token", trace=RAGTrace(query="q"))
        finally:
            stopped.set()

    pipe.query = query
    stream = pipe._stream_query_internal("q", thinking_detail_level="off")
    try:
        next(stream)
        assert queues[0].maxsize == 64
        assert queues[0].qsize() <= 64
    finally:
        stream.close()
        release.set()
        assert stopped.wait(1)


def test_already_cancelled_general_stream_skips_llm():
    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe._get_effective_llm = MagicMock(side_effect=AssertionError("must not select LLM"))
    cancel = threading.Event()
    cancel.set()
    assert list(pipe._stream_query_internal("q", filters={"chat_mode": "general"}, cancel_token=cancel)) == []


@pytest.mark.asyncio
async def test_closing_async_stream_cancels_and_closes_iterator():
    pipe = RAGPipeline.__new__(RAGPipeline)
    closed = threading.Event()
    observed = []

    def internal(**kwargs):
        observed.append(kwargs["cancel_token"])
        try:
            yield {"type": "token", "content": "first"}
            yield {"type": "token", "content": "second"}
        finally:
            closed.set()

    pipe._stream_query_internal = internal
    stream = pipe.stream_query("q")
    assert (await anext(stream))["content"] == "first"
    await stream.aclose()
    assert observed[0].is_set()
    assert closed.is_set()


@pytest.mark.asyncio
async def test_successful_stream_does_not_cancel_callers_event():
    pipe = RAGPipeline.__new__(RAGPipeline)
    cancel = threading.Event()
    pipe.query = lambda **kw: RAGResponse(query="q", answer="answer", trace=RAGTrace(query="q"))
    events = [event async for event in pipe.stream_query("q", cancel_token=cancel, thinking_detail_level="off")]
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"] == "answer"
    assert not cancel.is_set()
