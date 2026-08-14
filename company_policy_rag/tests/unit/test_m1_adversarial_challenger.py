"""
Adversarial and Empirical Verification Test Suite for Milestone 1:
Dynamic Retrieval Parameter Execution and SSE Stream Trace Contracts.

Authoritative Reference:
- ORIGINAL_REQUEST.md § R1 (Query Router & Strategy Selector)
- PROJECT.md § Architecture & Interface Contracts (Query Router & SSE Trace)
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
import pytest_asyncio

from backend.api.dependencies import get_chat_service, reset_dependencies
from backend.api.main import create_app
from backend.models.api_dto import ChatRequest
from backend.models.chunk import Chunk, ChunkMetadata
from backend.models.rag import (
    Citation,
    QueryCategory,
    QueryClassification,
    RetrievalStrategy,
    ScoredChunk,
)
from backend.rag.pipeline import RAGPipeline
from backend.rag.query_router import QueryRouter
from backend.services.chat_service import ChatService
from backend.services.telemetry_service import TelemetryService
from tests.e2e.helpers.sse_client import SSEDecoder


# ============================================================================
# TEST HELPERS & SPY MOCKS
# ============================================================================

def make_test_chunk(chunk_id: str, text: str, doc_id: str = "doc_1") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            document_id=doc_id,
            source_file="Company_Policy.pdf",
            file_path=f"/policies/{doc_id}.pdf",
            file_hash="hash123",
            document_type="company_policy",
            chunk_strategy="recursive",
        ),
    )


class SpyRetriever:
    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._chunks = chunks or [make_test_chunk(f"c_{i}", f"Policy text snippet {i}") for i in range(15)]

    def retrieve(
        self,
        query: str,
        dense_top_k: int = 25,
        bm25_top_k: int = 25,
        filters: dict[str, Any] | None = None,
        rrf_k: int | None = None,
    ) -> list[ScoredChunk]:
        self.calls.append({
            "query": query,
            "dense_top_k": dense_top_k,
            "bm25_top_k": bm25_top_k,
            "filters": filters,
            "rrf_k": rrf_k,
        })
        limit = min(len(self._chunks), max(dense_top_k, bm25_top_k))
        return [
            ScoredChunk(chunk=c, score=1.0 - (idx * 0.05), dense_score=0.9, sparse_score=0.8)
            for idx, c in enumerate(self._chunks[:limit])
        ]


class SpyReranker:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_n: int = 6,
        min_ratio: float = 0.40,
    ) -> list[ScoredChunk]:
        self.calls.append({
            "query": query,
            "candidate_count": len(chunks),
            "top_n": top_n,
            "min_ratio": min_ratio,
        })
        return chunks[:top_n]


class SpyCompressor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def expand_to_parents(
        self,
        chunks: list[ScoredChunk],
        docstore: dict[str, Chunk],
        enable_expansion: bool = True,
    ) -> list[ScoredChunk]:
        self.calls.append({
            "chunk_count": len(chunks),
            "enable_expansion": enable_expansion,
        })
        return chunks

    def format_context_for_prompt(self, chunks: list[ScoredChunk]) -> str:
        return "\n".join(f"[Source {i+1}] {c.chunk.text}" for i, c in enumerate(chunks))


class SpyMultiQueryGen:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate_subqueries(self, query: str) -> list[str]:
        self.calls.append(query)
        return [f"{query} part 1", f"{query} part 2"]


class MockFastLLM:
    def __init__(self, answer: str = "Based on policy, employees accrue 15 days of PTO. [Source 1]"):
        self.answer = answer
        self.model = "qwen2.5:7b"
        self.temperature = 0.1

    def complete(self, prompt: str, **kwargs: Any) -> Any:
        class Res:
            def __init__(self, text: str):
                self.text = text
            def __str__(self):
                return self.text
        return Res(self.answer)

    def stream_complete(self, prompt: str, **kwargs: Any) -> Any:
        class Delta:
            def __init__(self, d: str):
                self.delta = d
        words = self.answer.split(" ")
        for i, w in enumerate(words):
            token = w + (" " if i < len(words) - 1 else "")
            yield Delta(token)


# ============================================================================
# 1. DYNAMIC RETRIEVAL STRATEGY PARAMETER SPECIFICATION TESTS
# ============================================================================

class TestStrategyParameterSpecifications:
    """Empirically test exact parameter specifications for all query categories."""

    def test_factual_strategy_parameters(self):
        router = QueryRouter()
        strat = router.get_strategy_for_category(QueryCategory.FACTUAL)
        assert strat.name == "factual_precision"
        assert strat.dense_top_k == 10
        assert strat.bm25_top_k == 10
        assert strat.rrf_k == 60
        assert strat.rerank_top_n == 4
        assert strat.min_score_ratio == 0.45
        assert strat.enable_multi_query is False
        assert strat.enable_parent_expansion is False

    def test_comparison_strategy_parameters(self):
        router = QueryRouter()
        strat = router.get_strategy_for_category(QueryCategory.COMPARISON)
        assert strat.name == "comparison_broad"
        assert strat.dense_top_k == 25
        assert strat.bm25_top_k == 25
        assert strat.rrf_k == 60
        assert strat.rerank_top_n == 10
        assert strat.min_score_ratio == 0.30
        assert strat.enable_multi_query is True
        assert strat.enable_parent_expansion is True

    def test_enumeration_strategy_parameters(self):
        router = QueryRouter()
        strat = router.get_strategy_for_category(QueryCategory.ENUMERATION)
        assert strat.name == "enumeration_exhaustive"
        assert strat.dense_top_k == 30
        assert strat.bm25_top_k == 30
        assert strat.rrf_k == 60
        assert strat.rerank_top_n == 12
        assert strat.min_score_ratio == 0.25
        assert strat.enable_multi_query is True
        assert strat.enable_parent_expansion is False

    def test_procedural_strategy_parameters(self):
        router = QueryRouter()
        strat = router.get_strategy_for_category(QueryCategory.PROCEDURAL)
        assert strat.name == "procedural_workflow"
        assert strat.dense_top_k == 15
        assert strat.bm25_top_k == 15
        assert strat.rrf_k == 60
        assert strat.rerank_top_n == 6
        assert strat.min_score_ratio == 0.35
        assert strat.enable_multi_query is False
        assert strat.enable_parent_expansion is True

    def test_conversational_strategy_parameters(self):
        router = QueryRouter()
        strat = router.get_strategy_for_category(QueryCategory.CONVERSATIONAL)
        assert strat.name == "conversational_bypass"
        assert strat.dense_top_k == 0
        assert strat.bm25_top_k == 0
        assert strat.rerank_top_n == 0
        assert strat.enable_multi_query is False
        assert strat.enable_parent_expansion is False


# ============================================================================
# 2. PIPELINE EXECUTION & PARAMETER PROPAGATION VERIFICATION
# ============================================================================

class TestPipelineDynamicParameterExecution:
    """Empirically verify that RAGPipeline actually executes the selected parameters."""

    def _setup_pipeline(self):
        retriever = SpyRetriever()
        reranker = SpyReranker()
        compressor = SpyCompressor()
        multi_query = SpyMultiQueryGen()
        llm = MockFastLLM()
        router = QueryRouter()

        pipeline = RAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            compressor=compressor,
            multi_query_gen=multi_query,
            query_router=router,
            llm=llm,
        )
        return pipeline, retriever, reranker, compressor, multi_query

    def test_factual_pipeline_execution(self):
        pipeline, retriever, reranker, compressor, multi_query = self._setup_pipeline()
        query = "What is the standard mileage reimbursement rate?"

        resp = pipeline.query(user_query=query)

        # 1. Multi-query should NOT be triggered
        assert len(multi_query.calls) == 0

        # 2. Retriever should be called with dense_top_k=10, bm25_top_k=10, rrf_k=60
        assert len(retriever.calls) == 1
        assert retriever.calls[0]["dense_top_k"] == 10
        assert retriever.calls[0]["bm25_top_k"] == 10
        assert retriever.calls[0]["rrf_k"] == 60

        # 3. Reranker should be called with top_n=4, min_ratio=0.45
        assert len(reranker.calls) == 1
        assert reranker.calls[0]["top_n"] == 4
        assert reranker.calls[0]["min_ratio"] == 0.45

        # 4. Context compressor should be called with enable_expansion=False
        assert len(compressor.calls) == 1
        assert compressor.calls[0]["enable_expansion"] is False

        # 5. Trace telemetry
        assert resp.trace.query_type == "factual"
        assert resp.trace.retrieval_strategy == "factual_precision"
        assert resp.trace.routing_confidence >= 0.70

    def test_comparison_pipeline_execution(self):
        pipeline, retriever, reranker, compressor, multi_query = self._setup_pipeline()
        query = "Compare the maternity leave policy versus the paternity leave policy."

        resp = pipeline.query(user_query=query)

        # 1. Multi-query SHOULD be triggered
        assert len(multi_query.calls) == 1

        # 2. Retriever should be called for each generated subquery with dense_top_k=25, bm25_top_k=25, rrf_k=60
        assert len(retriever.calls) == 2  # 2 subqueries generated
        for call in retriever.calls:
            assert call["dense_top_k"] == 25
            assert call["bm25_top_k"] == 25
            assert call["rrf_k"] == 60

        # 3. Reranker should be called with top_n=10, min_ratio=0.30
        assert len(reranker.calls) == 1
        assert reranker.calls[0]["top_n"] == 10
        assert reranker.calls[0]["min_ratio"] == 0.30

        # 4. Context compressor should be called with enable_expansion=True
        assert len(compressor.calls) == 1
        assert compressor.calls[0]["enable_expansion"] is True

        # 5. Trace telemetry
        assert resp.trace.query_type == "comparison"
        assert resp.trace.retrieval_strategy == "comparison_broad"
        assert resp.trace.routing_confidence >= 0.70

    def test_enumeration_pipeline_execution(self):
        pipeline, retriever, reranker, compressor, multi_query = self._setup_pipeline()
        query = "List all eligible expenses covered under the wellness stipend."

        resp = pipeline.query(user_query=query)

        # 1. Multi-query SHOULD be triggered
        assert len(multi_query.calls) == 1

        # 2. Retriever should be called with dense_top_k=30, bm25_top_k=30, rrf_k=60
        assert len(retriever.calls) == 2
        for call in retriever.calls:
            assert call["dense_top_k"] == 30
            assert call["bm25_top_k"] == 30
            assert call["rrf_k"] == 60

        # 3. Reranker should be called with top_n=12, min_ratio=0.25
        assert len(reranker.calls) == 1
        assert reranker.calls[0]["top_n"] == 12
        assert reranker.calls[0]["min_ratio"] == 0.25

        # 4. Context compressor should be called with enable_expansion=False
        assert len(compressor.calls) == 1
        assert compressor.calls[0]["enable_expansion"] is False

        # 5. Trace telemetry
        assert resp.trace.query_type == "enumeration"
        assert resp.trace.retrieval_strategy == "enumeration_exhaustive"
        assert resp.trace.routing_confidence >= 0.70

    def test_procedural_pipeline_execution(self):
        pipeline, retriever, reranker, compressor, multi_query = self._setup_pipeline()
        query = "How do I submit an international travel approval request step by step?"

        resp = pipeline.query(user_query=query)

        # 1. Multi-query should NOT be triggered
        assert len(multi_query.calls) == 0

        # 2. Retriever should be called with dense_top_k=15, bm25_top_k=15, rrf_k=60
        assert len(retriever.calls) == 1
        assert retriever.calls[0]["dense_top_k"] == 15
        assert retriever.calls[0]["bm25_top_k"] == 15
        assert retriever.calls[0]["rrf_k"] == 60

        # 3. Reranker should be called with top_n=6, min_ratio=0.35
        assert len(reranker.calls) == 1
        assert reranker.calls[0]["top_n"] == 6
        assert reranker.calls[0]["min_ratio"] == 0.35

        # 4. Context compressor should be called with enable_expansion=True
        assert len(compressor.calls) == 1
        assert compressor.calls[0]["enable_expansion"] is True

        # 5. Trace telemetry
        assert resp.trace.query_type == "procedural"
        assert resp.trace.retrieval_strategy == "procedural_workflow"
        assert resp.trace.routing_confidence >= 0.70

    def test_conversational_bypass_pipeline_execution(self):
        pipeline, retriever, reranker, compressor, multi_query = self._setup_pipeline()
        query = "Hello! How can you help me?"

        resp = pipeline.query(user_query=query)

        # Vector search, reranking, multi-query, compression should NEVER be called
        assert len(retriever.calls) == 0
        assert len(reranker.calls) == 0
        assert len(compressor.calls) == 0
        assert len(multi_query.calls) == 0

        # Immediate greeting answer
        assert "Hello!" in resp.answer
        assert resp.citations == []
        assert resp.trace.query_type == "conversational"
        assert resp.trace.retrieval_strategy == "conversational_bypass"
        assert resp.trace.routing_confidence >= 0.90
        assert resp.trace.fallback_reason == "conversational_greeting"
        assert resp.trace.retrieved_candidate_count == 0


# ============================================================================
# 3. SSE STREAM TRACE & DONE EVENT INTEGRATION TESTS
# ============================================================================

class TestSSEStreamTraceAndDoneContracts:
    """Empirically test that SSE events emit proper query_type, routing_confidence, and retrieval_strategy."""

    @pytest.fixture(autouse=True)
    def clean_deps(self):
        reset_dependencies()
        yield
        reset_dependencies()

    @pytest_asyncio.fixture
    async def stream_client(self):
        retriever = SpyRetriever()
        reranker = SpyReranker()
        compressor = SpyCompressor()
        multi_query = SpyMultiQueryGen()
        llm = MockFastLLM()
        router = QueryRouter()

        pipeline = RAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            compressor=compressor,
            multi_query_gen=multi_query,
            query_router=router,
            llm=llm,
        )
        telemetry = TelemetryService()
        chat_service = ChatService(rag_pipeline=pipeline, telemetry_service=telemetry)

        app = create_app()
        app.dependency_overrides[get_chat_service] = lambda: chat_service

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
            headers={"Content-Type": "application/json"},
        ) as client:
            yield client

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query,expected_type,expected_strategy",
        [
            ("What is the annual leave rollover limit?", "factual", "factual_precision"),
            ("Compare the difference between HSA and FSA accounts.", "comparison", "comparison_broad"),
            ("List all company paid holidays for 2026.", "enumeration", "enumeration_exhaustive"),
            ("How do I request parental leave step by step?", "procedural", "procedural_workflow"),
            ("Hello! How can you help me?", "conversational", "conversational_bypass"),
        ],
    )
    async def test_sse_stream_events_contain_routing_metadata(
        self, stream_client: httpx.AsyncClient, query: str, expected_type: str, expected_strategy: str
    ):
        payload = {"message": query, "session_id": f"sess_sse_{expected_type}"}
        resp = await stream_client.post("/api/chat/stream", json=payload)
        assert resp.status_code == 200

        events = await SSEDecoder.collect_all(resp)
        event_types = [e["event"] for e in events]

        assert "start" in event_types
        assert "trace" in event_types
        assert "done" in event_types

        # Inspect 'trace' event
        trace_evt = next(e for e in events if e["event"] == "trace")
        trace_data = trace_evt["data"]["trace"]

        assert trace_data["query_type"] == expected_type
        assert trace_data["retrieval_strategy"] == expected_strategy
        assert isinstance(trace_data["routing_confidence"], float)
        assert 0.0 <= trace_data["routing_confidence"] <= 1.0

        # Inspect 'done' event
        done_evt = next(e for e in events if e["event"] == "done")
        done_data = done_evt["data"]

        assert done_data["status"] == "completed"
        assert "retrieval_trace" in done_data
        r_trace = done_data["retrieval_trace"]
        assert r_trace is not None
        assert r_trace["query_type"] == expected_type
        assert r_trace["retrieval_strategy"] == expected_strategy
        assert isinstance(r_trace["routing_confidence"], float)
        assert 0.0 <= r_trace["routing_confidence"] <= 1.0


# ============================================================================
# 4. ADVERSARIAL STRESS & EDGE CASE CHALLENGES
# ============================================================================

class TestAdversarialRoutingStress:
    """Stress tests on query router classification under boundary conditions."""

    def test_empty_and_whitespace_queries(self):
        router = QueryRouter()
        for empty_q in ["", "   ", "\t\n", "   \n\t  "]:
            classification = router.classify(empty_q)
            assert classification.category == QueryCategory.FACTUAL
            assert classification.confidence == 0.50
            # Because confidence is below default 0.70 threshold, strategy falls back to balanced_hybrid default
            assert classification.strategy.name == "balanced_hybrid"

    def test_mixed_intent_query_precedence(self):
        router = QueryRouter()
        # Query with comparison AND list
        query = "Compare the full list of dental benefits vs vision benefits."
        classification = router.classify(query)
        # Comparison takes precedence due to differential focus
        assert classification.category == QueryCategory.COMPARISON
        assert classification.strategy.enable_multi_query is True
        assert classification.strategy.enable_parent_expansion is True

    def test_long_adversarial_query(self):
        router = QueryRouter()
        long_query = "What is the policy " + "and ".join(["regarding employee benefits"] * 200) + "?"
        classification = router.classify(long_query)
        assert classification.category in [QueryCategory.FACTUAL, QueryCategory.ENUMERATION]
        assert classification.confidence > 0.0
        assert classification.strategy is not None

    def test_unicode_and_special_characters(self):
        router = QueryRouter()
        query = "💰 What is the 401(k) match % & bonus structure (2026/2027)???"
        classification = router.classify(query)
        assert classification.category == QueryCategory.FACTUAL
        assert classification.confidence > 0.70

    def test_conversational_boundaries(self):
        router = QueryRouter()
        # Short standard greetings -> conversational
        assert router.classify("Hello").category == QueryCategory.CONVERSATIONAL
        assert router.classify("Hi there").category == QueryCategory.CONVERSATIONAL
        assert router.classify("Good morning").category == QueryCategory.CONVERSATIONAL
        assert router.classify("Hello! How can you help me?").category == QueryCategory.CONVERSATIONAL
        assert router.classify("Who are you?").category == QueryCategory.CONVERSATIONAL

        # Question asking about policy starting with 'Hello' -> NOT conversational
        query_mixed = "Hello, what is the bereavement leave policy for full-time staff?"
        classification = router.classify(query_mixed)
        assert classification.category != QueryCategory.CONVERSATIONAL
        assert classification.category == QueryCategory.FACTUAL
