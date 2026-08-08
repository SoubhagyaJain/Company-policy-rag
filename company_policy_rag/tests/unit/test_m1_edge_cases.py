import sys
from unittest.mock import MagicMock
import pytest

from backend.models.api_dto import ChatRequest
from backend.models.chunk import Chunk, ChunkMetadata
from backend.models.rag import ScoredChunk, RAGResponse, QueryRewriteResult
from backend.services.chat_service import ChatService
from backend.rag.pipeline import RAGPipeline, _format_history_for_prompt
from backend.rag.query_rewrite import QueryRewriter, _format_history_for_rewrite


class MockRetriever:
    def retrieve(self, query, dense_top_k=25, bm25_top_k=25, filters=None):
        metadata = ChunkMetadata(
            document_id="doc1",
            source_file="policy.pdf",
            file_path="data/policy.pdf",
            file_hash="hash_doc1",
            document_type="pdf",
            chunk_strategy="recursive",
        )
        chunk = Chunk(
            id="c1",
            text="Health insurance eligibility starts after a waiting period of 30 days.",
            metadata=metadata,
        )
        return [ScoredChunk(chunk=chunk, score=0.9, dense_score=0.9, bm25_score=0.9)]


class MockReranker:
    def rerank(self, query, chunks):
        return chunks


class MockCompressor:
    def expand_to_parents(self, chunks, docstore):
        return chunks

    def format_context_for_prompt(self, chunks):
        return "[Source 1] Health insurance eligibility starts after a waiting period of 30 days."


class MockCitationEngine:
    def select_citations(self, answer_text, generation_chunks, user_query):
        return []


class MockTelemetryService:
    def record_from_rag_response(self, rag_response, ttft_ms=0.0):
        mock_trace = MagicMock()
        mock_trace.model_dump.return_value = {"trace_id": "test_trace"}
        return mock_trace


def test_empty_history_handling():
    """Verify pipeline and chat_service work cleanly with None/empty history."""
    retriever = MockRetriever()
    reranker = MockReranker()
    compressor = MockCompressor()
    citation_engine = MockCitationEngine()
    telemetry = MockTelemetryService()
    query_rewriter = QueryRewriter(enable_llm_rewrite=False)
    pipeline = RAGPipeline(
        hybrid_retriever=retriever,
        reranker=reranker,
        query_rewriter=query_rewriter,
        compressor=compressor,
        citation_engine=citation_engine,
        llm=None
    )
    chat_service = ChatService(
        rag_pipeline=pipeline,
        telemetry_service=telemetry
    )

    # 1. Pipeline query with history=None and history=[]
    res_none = pipeline.query(user_query="What is the health insurance policy?", history=None)
    assert res_none.answer is not None
    assert "Health insurance" in res_none.answer

    res_empty = pipeline.query(user_query="What is the health insurance policy?", history=[])
    assert res_empty.answer is not None

    # 2. QueryRewriter with empty history
    qr_res = query_rewriter.rewrite("What is the health insurance policy?", history=None)
    assert qr_res.rewritten_query is not None

    # 3. ChatService execute_query with fresh session (no history stored)
    req = ChatRequest(message="What is the health insurance policy?", session_id="sess_empty_1")
    resp = chat_service.execute_query(req)
    assert resp.session_id == "sess_empty_1"

    # Check stored session history length (user msg + assistant msg)
    assert len(chat_service._sessions["sess_empty_1"]) == 2


def test_long_history_truncation_and_malformed():
    """Verify long multi-turn history truncation to max turns and resilience to malformed items."""
    long_history = []
    for i in range(50):
        long_history.append({"role": "user", "content": f"User question {i}"})
        long_history.append({"role": "assistant", "content": f"Assistant answer {i}"})

    # Test _format_history_for_prompt
    prompt_hist = _format_history_for_prompt(long_history, max_turns=6)
    lines = prompt_hist.strip().split("\n")
    assert len(lines) == 13
    assert "User question 49" in prompt_hist
    assert "User question 0" not in prompt_hist

    # Test _format_history_for_rewrite
    rewrite_hist = _format_history_for_rewrite(long_history, max_turns=4)
    rewrite_lines = rewrite_hist.strip().split("\n")
    assert len(rewrite_lines) == 8

    # Malformed items (missing fields, None content, non-string values)
    malformed_history = [
        {"role": "user"},  # missing content
        {"role": None, "content": None},  # Nones
        {"role": "assistant", "content": 12345},  # integer content
        {"random_key": "val"}  # missing role & content
    ]
    prompt_malformed = _format_history_for_prompt(malformed_history)
    assert isinstance(prompt_malformed, str)

    rewrite_malformed = _format_history_for_rewrite(malformed_history)
    assert isinstance(rewrite_malformed, str)


def test_missing_model_parameter():
    """Verify system defaults to 'qwen2.5:7b' when model is missing/None/empty string."""
    retriever = MockRetriever()
    reranker = MockReranker()
    compressor = MockCompressor()
    citation_engine = MockCitationEngine()
    telemetry = MockTelemetryService()
    query_rewriter = QueryRewriter(enable_llm_rewrite=False)
    pipeline = RAGPipeline(
        hybrid_retriever=retriever,
        reranker=reranker,
        query_rewriter=query_rewriter,
        compressor=compressor,
        citation_engine=citation_engine,
        llm=None
    )
    chat_service = ChatService(
        rag_pipeline=pipeline,
        telemetry_service=telemetry
    )

    req_none = ChatRequest(message="Tell me about health insurance", model=None)
    res_none = chat_service.execute_query(req_none)
    assert res_none.model == "qwen2.5:7b"

    req_empty = ChatRequest(message="Tell me about health insurance", model="")
    res_empty = chat_service.execute_query(req_empty)
    assert res_empty.model == "qwen2.5:7b"

    res_pipe = pipeline.query(user_query="Tell me about health insurance", model=None)
    assert res_pipe.model == "qwen2.5:7b"


def test_unpulled_model_and_llm_exception_fallback():
    """Verify that LLM exceptions (e.g. unpulled model, offline server) fall back gracefully."""
    retriever = MockRetriever()
    reranker = MockReranker()
    compressor = MockCompressor()
    citation_engine = MockCitationEngine()
    query_rewriter = QueryRewriter(enable_llm_rewrite=False)

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = RuntimeError("model 'unpulled_model:latest' not found, try pulling it first")
    mock_llm.stream_complete.side_effect = RuntimeError("model 'unpulled_model:latest' not found")
    mock_llm.model = "unpulled_model:latest"

    pipeline_with_llm = RAGPipeline(
        hybrid_retriever=retriever,
        reranker=reranker,
        query_rewriter=query_rewriter,
        compressor=compressor,
        citation_engine=citation_engine,
        llm=mock_llm
    )

    # 1. Sync query execution
    res = pipeline_with_llm.query(user_query="What is the waiting period for health insurance?", model="unpulled_model:latest")
    assert res.answer is not None

    # 2. Stream query execution
    events = list(pipeline_with_llm.stream_query(user_query="What is the waiting period for health insurance?", model="unpulled_model:latest"))
    event_types = [e["type"] for e in events]
    assert "retrieval_done" in event_types
    assert "token" in event_types
    assert "done" in event_types

    # 3. Query rewriter fallback when LLM fails
    llm_rewriter = QueryRewriter(enable_llm_rewrite=True, llm=mock_llm)
    rewrite_res = llm_rewriter.rewrite("What is the waiting period?")
    assert rewrite_res.rewritten_query is not None


def test_query_rewriter_coreference_behavior():
    """Verify coreference resolution with pronouns vs independent queries."""
    query_rewriter = QueryRewriter(enable_llm_rewrite=False)
    history = [
        {"role": "user", "content": "What is the waiting period for health insurance eligibility?"},
        {"role": "assistant", "content": "The waiting period for health insurance is 30 days."}
    ]

    # Case A: Follow-up query with pronoun ("what about it?")
    res_pronoun = query_rewriter.rewrite("what about it?", history=history)
    assert "health insurance" in res_pronoun.rewritten_query.lower() or "waiting period" in res_pronoun.rewritten_query.lower()

    # Case B: Follow-up query with pronoun ("how long is that?")
    res_pronoun2 = query_rewriter.rewrite("how long is that?", history=history)
    assert "health insurance" in res_pronoun2.rewritten_query.lower()

    # Case C: Independent query ("What is the dress code policy?")
    res_independent = query_rewriter.rewrite("What is the dress code policy?", history=history)
    assert "health insurance eligibility" not in res_independent.rewritten_query

    # Case D: Mocked LLM coreference rewrite
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Standalone Query: What is the health insurance waiting period duration?"
    llm_rewriter = QueryRewriter(enable_llm_rewrite=True, llm=mock_llm)
    res_llm = llm_rewriter.rewrite("how long is that?", history=history)
    assert res_llm.rewritten_query == "Standalone Query: What is the health insurance waiting period duration?"
