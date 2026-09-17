"""A disconnected client must stop generation instead of letting it run to completion."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from backend.models.chunk import Chunk, ChunkMetadata
from backend.models.rag import ScoredChunk
from backend.rag.pipeline import RAGPipeline
from src.config import settings


class _Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def retrieve(self, query, dense_top_k=8, bm25_top_k=8, filters=None, rrf_k=60):
        return [ScoredChunk(chunk=c, score=0.9) for c in self.chunks]


class _SlowStreamingLLM:
    model = "stub"

    def __init__(self) -> None:
        self.tokens_generated = 0
        self.stream_closed = threading.Event()

    def complete(self, prompt, **kwargs):
        return "unused"

    def stream_complete(self, prompt, **kwargs):
        def gen():
            try:
                for _ in range(500):
                    self.tokens_generated += 1
                    yield "token "
            finally:
                self.stream_closed.set()

        return gen()


@pytest.fixture
def quiet(monkeypatch):
    for key, value in {
        "retrieval_cache_enabled": False,
        "vision_enabled": False,
        "enable_lazy_vision_fallback": False,
        "enable_llm_multi_query": False,
        "enable_reranker": False,
        "enable_conversation_interpreter": False,
    }.items():
        monkeypatch.setattr(settings, key, value)


def test_cancel_during_streaming_stops_generation_and_skips_follow_up_work(quiet) -> None:
    chunk = Chunk(
        id="memory",
        text="Agent memory stores context between turns so the agent can recall earlier steps.",
        metadata=ChunkMetadata(document_id="doc_aaaaaaaaaaaa", source_file="guide.pdf", page_number=3),
    )
    llm = _SlowStreamingLLM()
    verifier = MagicMock()
    semantic_cache = MagicMock()
    semantic_cache.get.return_value = None
    pipeline = RAGPipeline(
        hybrid_retriever=_Retriever([chunk]),
        docstore={chunk.id: chunk},
        llm=llm,
        verifier=verifier,
        semantic_cache=semantic_cache,
    )

    cancel = threading.Event()
    received: list[str] = []

    def on_token(delta: str) -> None:
        received.append(delta)
        if len(received) == 3:
            cancel.set()

    response = pipeline.query(
        "Explain how agent memory works.",
        stream_callback=on_token,
        cancel_event=cancel,
    )

    assert len(received) == 3
    assert llm.tokens_generated < 10
    assert llm.stream_closed.is_set()
    assert response.trace.fallback_reason == "cancelled"
    verifier.verify.assert_not_called()
    semantic_cache.put.assert_not_called()


def test_already_cancelled_request_never_calls_the_llm(quiet) -> None:
    chunk = Chunk(
        id="tools",
        text="Tools let an agent call external APIs and read their results.",
        metadata=ChunkMetadata(document_id="doc_aaaaaaaaaaaa", source_file="guide.pdf", page_number=4),
    )
    llm = _SlowStreamingLLM()
    llm.complete = MagicMock(return_value="should not run")
    cancel = threading.Event()
    cancel.set()
    pipeline = RAGPipeline(hybrid_retriever=_Retriever([chunk]), docstore={chunk.id: chunk}, llm=llm)

    response = pipeline.query("Explain agent tools in detail.", cancel_event=cancel)

    assert llm.tokens_generated == 0
    llm.complete.assert_not_called()
    assert response.trace.fallback_reason == "cancelled"
