"""
Adversarial Stress Test Suite for RetryEngine and RAGPipeline Retry Loop.

Empirical verification suite challenging:
1. Dynamic Parameter Adjustments (Faithfulness, Completeness, Citations, Coherence)
2. Hard Cap Enforcement (attempt >= 2 strictly stops, raises on prepare_retry)
3. Retry Exhaustion Fallback (low_confidence=True, fallback_reason="retry_exhausted_fallback")
4. Attempt Counting & Telemetry Trace Accuracy (0, 1, 2 retries in sync and streaming)
5. Best Candidate Tracking under Degrading Retries
6. Empty Retrieval, Conversational Bypass, and Concurrency Isolation
"""
from __future__ import annotations

import asyncio
import json
import pytest
from typing import Any, List, Optional
from unittest.mock import MagicMock

from backend.models.chunk import Chunk, ChunkMetadata, ChunkRole, ContentType
from backend.models.rag import (
    Citation,
    QueryCategory,
    QueryClassification,
    RAGResponse,
    RAGTrace,
    RetrievalStrategy,
    ScoredChunk,
    VerificationReport,
)
from backend.rag.citations import CitationEngine
from backend.rag.context_compression import ContextCompressor
from backend.rag.multi_query import MultiQueryGenerator
from backend.rag.pipeline import RAGPipeline
from backend.rag.query_rewrite import QueryRewriter
from backend.rag.query_router import QueryRouter
from backend.rag.retry_engine import RetryEngine
from backend.rag.verifier import SelfReflectionVerifier
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.reranker import CrossEncoderReranker
from backend.services.chat_service import ChatService
from backend.services.telemetry_service import TelemetryService
from backend.models.api_dto import ChatRequest


def _make_sample_chunk(cid: str = "chunk_01", text: str = "Policy details regarding PTO.") -> Chunk:
    meta = ChunkMetadata(
        document_id="doc_01",
        source_file="handbook.pdf",
        file_path="data/handbook.pdf",
        file_hash="hash01",
        document_type="policy",
        chunk_strategy="recursive",
    )
    return Chunk(id=cid, text=text, metadata=meta)


# ============================================================================
# 1. RETRY ENGINE PARAMETER ADJUSTMENT ADVERSARIAL TESTS
# ============================================================================

class TestRetryEngineParameterAdjustments:
    """Stress-test parameter adjustment dynamics under various failure modes."""

    def test_faithfulness_failure_adjustments(self):
        engine = RetryEngine(max_retries=2)
        initial_strategy = RetrievalStrategy(
            name="initial",
            min_score_ratio=0.40,
            temperature=0.10,
            dense_top_k=25,
            bm25_top_k=25,
            rerank_top_n=6,
        )

        # Report with low faithfulness and unsupported claims
        report = VerificationReport(
            faithfulness=0.40,
            completeness=0.90,
            citation_coverage=0.90,
            coherence=0.90,
            composite_score=0.65,
            passed=False,
            unsupported_claims=["$5,000 allowance", "executive desk"],
        )

        adj_strategy, prompt_ref = engine.prepare_retry(
            attempt=0, report=report, strategy=initial_strategy, query="What is the allowance?"
        )

        # min_score_ratio should increase by 0.10
        assert adj_strategy.min_score_ratio == 0.50
        # temperature should decrease by 0.05
        assert adj_strategy.temperature == 0.05
        # recall parameters should remain unchanged
        assert adj_strategy.dense_top_k == 25
        assert adj_strategy.bm25_top_k == 25
        assert adj_strategy.rerank_top_n == 6
        # prompt refinement should contain strict grounding instructions and list claims
        assert "Strictly adhere to the retrieved facts" in prompt_ref
        assert "$5,000 allowance" in prompt_ref
        assert "executive desk" in prompt_ref

    def test_faithfulness_clamping_boundaries(self):
        engine = RetryEngine(max_retries=2)
        # Strategy already at high min_score_ratio and low temp
        initial_strategy = RetrievalStrategy(
            min_score_ratio=0.55,
            temperature=0.02,
        )

        report = VerificationReport(
            faithfulness=0.30,
            passed=False,
            unsupported_claims=["hallucinated item"],
        )

        adj_strategy, _ = engine.prepare_retry(
            attempt=0, report=report, strategy=initial_strategy
        )

        # min_score_ratio must be capped at 0.60
        assert adj_strategy.min_score_ratio == 0.60
        # temperature must be floored at 0.0
        assert adj_strategy.temperature == 0.0

    def test_completeness_failure_adjustments(self):
        engine = RetryEngine(max_retries=2)
        initial_strategy = RetrievalStrategy(
            dense_top_k=20,
            bm25_top_k=20,
            rerank_top_n=5,
            enable_multi_query=False,
            enable_parent_expansion=False,
            min_score_ratio=0.40,
            temperature=0.10,
        )

        report = VerificationReport(
            faithfulness=0.95,
            completeness=0.40,
            citation_coverage=0.90,
            coherence=0.90,
            composite_score=0.68,
            passed=False,
            missing_aspects=["application submission deadline", "manager approval process"],
        )

        adj_strategy, prompt_ref = engine.prepare_retry(
            attempt=0, report=report, strategy=initial_strategy, query="Explain PTO application"
        )

        # dense_top_k and bm25_top_k increased by +10
        assert adj_strategy.dense_top_k == 30
        assert adj_strategy.bm25_top_k == 30
        # rerank_top_n increased by +3
        assert adj_strategy.rerank_top_n == 8
        # multi-query and parent expansion forced True
        assert adj_strategy.enable_multi_query is True
        assert adj_strategy.enable_parent_expansion is True
        # precision parameters unchanged
        assert adj_strategy.min_score_ratio == 0.40
        assert adj_strategy.temperature == 0.10
        # prompt refinement should list missing aspects
        assert "application submission deadline" in prompt_ref
        assert "manager approval process" in prompt_ref

    def test_completeness_rerank_top_n_clamping(self):
        engine = RetryEngine(max_retries=2)
        initial_strategy = RetrievalStrategy(
            rerank_top_n=14,
        )
        report = VerificationReport(
            completeness=0.30,
            passed=False,
            missing_aspects=["full list"],
        )
        adj_strategy, _ = engine.prepare_retry(
            attempt=0, report=report, strategy=initial_strategy
        )
        # rerank_top_n capped at 15
        assert adj_strategy.rerank_top_n == 15

    def test_citation_and_coherence_failure_guidance(self):
        engine = RetryEngine(max_retries=2)
        initial_strategy = RetrievalStrategy()

        # Citation failure
        cit_report = VerificationReport(
            faithfulness=0.85,
            completeness=0.85,
            citation_coverage=0.30,
            coherence=0.85,
            composite_score=0.60,
            passed=False,
        )
        _, cit_prompt = engine.prepare_retry(attempt=0, report=cit_report, strategy=initial_strategy)
        assert "[Source N]" in cit_prompt

        # Coherence failure
        coh_report = VerificationReport(
            faithfulness=0.85,
            completeness=0.85,
            citation_coverage=0.85,
            coherence=0.50,
            composite_score=0.60,
            passed=False,
        )
        _, coh_prompt = engine.prepare_retry(attempt=0, report=coh_report, strategy=initial_strategy)
        assert "clear logical flow" in coh_prompt

    def test_combined_multi_dimensional_failure(self):
        engine = RetryEngine(max_retries=2)
        initial_strategy = RetrievalStrategy(
            dense_top_k=20,
            bm25_top_k=20,
            rerank_top_n=5,
            min_score_ratio=0.40,
            temperature=0.10,
            enable_multi_query=False,
            enable_parent_expansion=False,
        )

        report = VerificationReport(
            faithfulness=0.50,
            completeness=0.50,
            citation_coverage=0.30,
            coherence=0.50,
            composite_score=0.45,
            passed=False,
            missing_aspects=["aspect A"],
            unsupported_claims=["claim B"],
        )

        adj_strategy, prompt_ref = engine.prepare_retry(
            attempt=0, report=report, strategy=initial_strategy
        )

        # Both faithfulness and completeness adjustments applied
        assert adj_strategy.min_score_ratio == 0.50
        assert adj_strategy.temperature == 0.05
        assert adj_strategy.dense_top_k == 30
        assert adj_strategy.bm25_top_k == 30
        assert adj_strategy.rerank_top_n == 8
        assert adj_strategy.enable_multi_query is True
        assert adj_strategy.enable_parent_expansion is True
        # All prompt directives present
        assert "claim B" in prompt_ref
        assert "aspect A" in prompt_ref
        assert "[Source N]" in prompt_ref
        assert "clear logical flow" in prompt_ref


# ============================================================================
# 2. HARD CAP AND RETRY EXHAUSTION ENFORCEMENT
# ============================================================================

class TestHardCapEnforcement:
    """Stress-test retry boundary gates and hard cap enforcement."""

    def test_should_retry_exact_attempt_boundaries(self):
        engine = RetryEngine(max_retries=2)
        failed_report = VerificationReport(composite_score=0.5, passed=False)
        passed_report = VerificationReport(composite_score=0.9, passed=True)

        # Attempt 0: initial attempt
        assert engine.should_retry(0, failed_report) is True
        assert engine.should_retry(0, passed_report) is False

        # Attempt 1: 1st retry
        assert engine.should_retry(1, failed_report) is True
        assert engine.should_retry(1, passed_report) is False

        # Attempt 2: 2nd retry completed -> hard cap reached, must NOT retry
        assert engine.should_retry(2, failed_report) is False
        assert engine.should_retry(2, passed_report) is False

        # Attempt 3+: adversarial out-of-bounds attempts
        assert engine.should_retry(3, failed_report) is False
        assert engine.should_retry(99, failed_report) is False

    def test_prepare_retry_hard_cap_exception(self):
        engine = RetryEngine(max_retries=2)
        failed_report = VerificationReport(composite_score=0.5, passed=False)
        strategy = RetrievalStrategy()

        # Attempt 0 & 1 succeed
        s1, _ = engine.prepare_retry(0, failed_report, strategy)
        assert s1 is not None
        s2, _ = engine.prepare_retry(1, failed_report, strategy)
        assert s2 is not None

        # Attempt 2 must raise ValueError immediately
        with pytest.raises(ValueError, match="Max retries"):
            engine.prepare_retry(2, failed_report, strategy)

        # Attempt 3 must also raise
        with pytest.raises(ValueError, match="Max retries"):
            engine.prepare_retry(3, failed_report, strategy)


# ============================================================================
# 3. RAG PIPELINE INTEGRATION & ATTEMPT COUNTING (MOCKED LLM/RETRIEVER)
# ============================================================================

class ControlledMockLLM:
    """Mock LLM allowing precise control over per-attempt response generation."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0
        self.prompts: list[str] = []
        self.temperatures: list[float] = []
        self.model = "qwen2.5:7b"

    def complete(self, prompt: str, temperature: float = 0.1, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        self.temperatures.append(temperature)
        idx = min(self.call_count, len(self.responses) - 1)
        resp = self.responses[idx]
        self.call_count += 1
        return resp

    def stream_complete(self, prompt: str, **kwargs: Any):
        resp = self.complete(prompt, **kwargs)
        for word in resp.split(" "):
            yield word + " "


def _build_test_pipeline(
    mock_llm: ControlledMockLLM,
    custom_verifier_fn=None,
    max_retries: int = 2,
) -> RAGPipeline:
    chunk1 = _make_sample_chunk("c1", "PTO accrual is 15 days annually for all full-time employees.")
    chunk2 = _make_sample_chunk("c2", "Part-time employees accrue PTO on a pro-rata basis.")

    scored_chunks = [
        ScoredChunk(chunk=chunk1, score=0.95),
        ScoredChunk(chunk=chunk2, score=0.85),
    ]

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = scored_chunks

    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = lambda query, chunks, top_n=6, min_ratio=0.4: chunks[:top_n]

    mock_query_rewriter = QueryRewriter(enable_llm_rewrite=False)
    mock_query_router = QueryRouter()

    verifier = SelfReflectionVerifier(threshold=0.70)
    if custom_verifier_fn:
        verifier.verify = custom_verifier_fn

    retry_engine = RetryEngine(max_retries=max_retries)

    pipeline = RAGPipeline(
        hybrid_retriever=mock_retriever,
        reranker=mock_reranker,
        query_rewriter=mock_query_rewriter,
        query_router=mock_query_router,
        llm=mock_llm,
        verifier=verifier,
        retry_engine=retry_engine,
    )
    return pipeline


class TestPipelineRetryIntegration:
    """Stress-test full pipeline retry loop with 0, 1, 2 retries and exhaustion."""

    def test_pipeline_zero_retries_immediate_pass(self):
        """Scenario A: Verification passes on attempt 0 -> exactly 1 LLM call, retry_count=0."""
        good_answer = "Full-time employees accrue 15 days of PTO annually [Source 1]."
        mock_llm = ControlledMockLLM([good_answer])

        pipeline = _build_test_pipeline(mock_llm)
        response = pipeline.query("What is PTO accrual?")

        assert mock_llm.call_count == 1
        assert response.trace.retry_count == 0
        assert response.trace.fallback_reason == "none"
        assert response.trace.faithfulness_passed is True
        assert len(response.trace.retry_reasons) == 0
        assert "15 days" in response.answer

    def test_pipeline_one_retry_pass_on_attempt_1(self):
        """Scenario B: Attempt 0 fails (hallucinated $5000), Attempt 1 passes."""
        bad_answer = "Employees receive $5,000 equipment reimbursement [Source 1]."
        good_answer = "Full-time employees accrue 15 days of PTO annually [Source 1]."
        mock_llm = ControlledMockLLM([bad_answer, good_answer])

        pipeline = _build_test_pipeline(mock_llm)
        response = pipeline.query("What is PTO accrual?")

        # Total 2 LLM complete calls
        assert mock_llm.call_count == 2
        assert response.trace.retry_count == 1
        assert response.trace.fallback_reason == "none"
        assert response.trace.faithfulness_passed is True
        assert len(response.trace.retry_reasons) == 1
        # Verification score should be populated
        assert response.trace.verification_score is not None
        assert response.trace.verification_score >= 0.70
        # Prompt on attempt 1 should include refinement instructions
        assert "Strictly adhere to the retrieved facts" in mock_llm.prompts[1] or "Refinement Instructions" in mock_llm.prompts[1]

    def test_pipeline_two_retries_pass_on_attempt_2(self):
        """Scenario C: Attempt 0 & 1 fail, Attempt 2 passes."""
        call_tracker = {"count": 0}

        def mock_verify(query, answer, context_chunks, citations, llm=None):
            cnt = call_tracker["count"]
            call_tracker["count"] += 1
            if cnt == 0:
                return VerificationReport(
                    faithfulness=0.4, completeness=0.5, composite_score=0.45, passed=False, critique="Attempt 0 failed"
                )
            elif cnt == 1:
                return VerificationReport(
                    faithfulness=0.5, completeness=0.6, composite_score=0.55, passed=False, critique="Attempt 1 failed"
                )
            else:
                return VerificationReport(
                    faithfulness=0.9, completeness=0.9, composite_score=0.90, passed=True, critique=None
                )

        mock_llm = ControlledMockLLM(["ans0", "ans1", "ans2 [Source 1]"])
        pipeline = _build_test_pipeline(mock_llm, custom_verifier_fn=mock_verify, max_retries=2)
        response = pipeline.query("What is PTO accrual?")

        # Total 3 LLM calls (attempts 0, 1, 2)
        assert mock_llm.call_count == 3
        assert response.trace.retry_count == 2
        assert response.trace.fallback_reason == "none"
        assert response.trace.faithfulness_passed is True
        assert len(response.trace.retry_reasons) == 2
        assert response.trace.retry_reasons[0] == "Attempt 0 failed"
        assert response.trace.retry_reasons[1] == "Attempt 1 failed"
        assert response.answer == "ans2 [Source 1]"

    def test_pipeline_retry_exhaustion_fallback(self):
        """Scenario D: All 3 attempts fail -> returns best candidate, low_confidence=True, retry_exhausted_fallback."""
        reports = [
            VerificationReport(faithfulness=0.6, completeness=0.6, composite_score=0.60, passed=False, critique="Fail 0"),
            VerificationReport(faithfulness=0.3, completeness=0.3, composite_score=0.30, passed=False, critique="Fail 1"),
            VerificationReport(faithfulness=0.5, completeness=0.5, composite_score=0.50, passed=False, critique="Fail 2"),
        ]
        report_idx = [0]

        def mock_verify_exhaustion(query, answer, context_chunks, citations, llm=None):
            idx = min(report_idx[0], len(reports) - 1)
            report_idx[0] += 1
            return reports[idx]

        mock_llm = ControlledMockLLM(["Answer Attempt 0", "Answer Attempt 1", "Answer Attempt 2"])
        pipeline = _build_test_pipeline(mock_llm, custom_verifier_fn=mock_verify_exhaustion, max_retries=2)
        
        telemetry = TelemetryService()
        chat_service = ChatService(rag_pipeline=pipeline, telemetry_service=telemetry)

        chat_req = ChatRequest(message="What is PTO accrual?", stream=False)
        chat_resp = chat_service.execute_query(chat_req)

        # Total 3 LLM calls executed (attempts 0, 1, 2)
        assert mock_llm.call_count == 3
        # Must select best candidate (Attempt 0 had highest composite_score 0.60)
        assert chat_resp.answer == "Answer Attempt 0"
        # Hard cap stopped execution at 2 retries
        assert chat_resp.trace.retry_count == 2
        assert chat_resp.trace.fallback_reason == "retry_exhausted_fallback"
        assert chat_resp.trace.faithfulness_passed is False
        # ChatResponse low_confidence flag strictly True
        assert chat_resp.low_confidence is True

    @pytest.mark.asyncio
    async def test_streaming_pipeline_retry_exhaustion(self):
        """Scenario E: Streaming pipeline retry exhaustion yields low_confidence=True and retry_exhausted_fallback."""
        reports = [
            VerificationReport(faithfulness=0.5, completeness=0.5, composite_score=0.50, passed=False, critique="Stream Fail 0"),
            VerificationReport(faithfulness=0.6, completeness=0.6, composite_score=0.65, passed=False, critique="Stream Fail 1"),
            VerificationReport(faithfulness=0.4, completeness=0.4, composite_score=0.40, passed=False, critique="Stream Fail 2"),
        ]
        report_idx = [0]

        def mock_verify_stream(query, answer, context_chunks, citations, llm=None):
            idx = min(report_idx[0], len(reports) - 1)
            report_idx[0] += 1
            return reports[idx]

        mock_llm = ControlledMockLLM(["Stream Ans 0", "Stream Best Ans 1", "Stream Ans 2"])
        pipeline = _build_test_pipeline(mock_llm, custom_verifier_fn=mock_verify_stream, max_retries=2)
        
        telemetry = TelemetryService()
        chat_service = ChatService(rag_pipeline=pipeline, telemetry_service=telemetry)

        chat_req = ChatRequest(message="Stream test query", stream=True)
        events = []
        async for sse_raw in chat_service.stream_query(chat_req):
            events.append(sse_raw)

        # Find done event
        done_raw = next((e for e in events if "event: done" in e), None)
        assert done_raw is not None
        data_str = done_raw.split("data: ")[1].strip()
        done_json = json.loads(data_str)

        assert done_json["low_confidence"] is True
        assert done_json["answer"] == "Stream Best Ans 1"
        assert done_json["retrieval_trace"]["faithfulness_passed"] is False
        assert done_json["retrieval_trace"]["retry_count"] == 2


# ============================================================================
# 4. EDGE CASE & CONCURRENCY ADVERSARIAL TESTS
# ============================================================================

class TestEdgeCasesAndConcurrency:
    """Stress-test edge cases, bypasses, and concurrent query execution."""

    def test_zero_retrieved_candidates_no_wasteful_retries(self):
        """Zero candidates from retriever returns unanswerable without retrying."""
        mock_llm = ControlledMockLLM(["Should not be called"])
        pipeline = _build_test_pipeline(mock_llm)
        pipeline.hybrid_retriever.retrieve.return_value = []

        response = pipeline.query("Completely unindexed obscure term")
        assert mock_llm.call_count == 0
        assert response.trace.retry_count == 0
        assert "unable to answer" in response.answer.lower()
        assert response.trace.faithfulness_passed is True

    def test_conversational_bypass_no_retries(self):
        """Conversational greeting bypasses retrieval and retry loops entirely."""
        mock_llm = ControlledMockLLM(["Should not be called"])
        pipeline = _build_test_pipeline(mock_llm)

        response = pipeline.query("Hello there!")
        assert mock_llm.call_count == 0
        assert response.trace.retry_count == 0
        assert response.trace.fallback_reason == "conversational_greeting"
        assert response.trace.faithfulness_passed is True

    def test_concurrent_retry_execution_thread_safety(self):
        """Multiple parallel queries with varying retry requirements execute safely."""
        mock_llm = ControlledMockLLM([
            "Full-time employees receive 15 days of PTO [Source 1].",
            "Unsupported claim $5000",
            "Fixed answer with 15 days PTO [Source 1].",
        ])
        pipeline = _build_test_pipeline(mock_llm)

        import concurrent.futures

        queries = [
            "What is PTO accrual rate?",
            "Hello there!",
            "What is PTO accrual rate?",
            "What is PTO accrual rate?",
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(pipeline.query, q) for q in queries]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 4
        for r in results:
            assert r.answer is not None
            assert r.trace is not None
            assert r.trace.retry_count <= 2
