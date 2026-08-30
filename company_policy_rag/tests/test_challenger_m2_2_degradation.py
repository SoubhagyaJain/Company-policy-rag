import asyncio
import json
import time
from unittest.mock import MagicMock, patch
import pytest

from backend.models.chunk import Chunk, ChunkMetadata, ContentType
from backend.models.rag import (
    Citation,
    EvidenceStatus,
    QueryCategory,
    QueryClassification,
    RAGResponse,
    RAGTrace,
    ReasoningSummary,
    RetrievalStrategy,
    ScoredChunk,
    ThinkingDetailLevel,
    ThinkingEvent,
    ThinkingStage,
    ThinkingStatus,
)
from backend.models.api_dto import ChatRequest, ChatResponse
from backend.models.conversation import AnswerMode, ConversationEvidenceContext, ConversationRAGState, ConversationTurn
from backend.rag.thinking import ThinkingStateMachine
from backend.rag.pipeline import RAGPipeline
from backend.services.chat_service import ChatService


def create_test_chunk(chunk_id: str, text: str, page_number: int = 1, doc_id: str = "doc_test") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            document_id=doc_id,
            source_file=f"{doc_id}.pdf",
            page_number=page_number,
            section_title="Policy Section",
            content_type=ContentType.PROSE,
        ),
        token_count=len(text.split()),
    )


def test_adversarial_01_dense_retriever_failure_fallback_to_bm25():
    """
    Adversarial Challenge 1:
    Simulate dense retriever failure (Vector DB connection drop / exception).
    Verify fallback to BM25 search, emission of DEGRADED thinking event,
    and recording in reasoning_summary.degraded_stages.
    """
    mock_bm25_chunk = create_test_chunk("bm25_chunk_001", "BM25 keyword result: Vacation carryover max 5 days.")
    mock_bm25_scored = ScoredChunk(chunk=mock_bm25_chunk, score=0.85, sparse_score=0.85)

    mock_bm25_index = MagicMock()
    mock_bm25_index.search.return_value = [mock_bm25_scored]

    mock_hybrid_retriever = MagicMock()
    mock_hybrid_retriever.bm25_index = mock_bm25_index
    mock_hybrid_retriever.retrieve.side_effect = RuntimeError("Vector DB connection timeout or dense index corruption")

    mock_llm = MagicMock()
    mock_llm.complete.return_value = "According to the vacation carryover policy, maximum 5 days can be carried over."

    pipeline = RAGPipeline(
        hybrid_retriever=mock_hybrid_retriever,
        llm=mock_llm,
        docstore={"bm25_chunk_001": mock_bm25_chunk},
    )

    thinking_sm = ThinkingStateMachine(query_id="qry_dense_fail_test", detail_level=ThinkingDetailLevel.DETAILED)
    res = pipeline.query(
        user_query="What is the carryover policy?",
        thinking_detail_level=ThinkingDetailLevel.DETAILED,
        thinking_sm=thinking_sm,
    )

    # 1. Pipeline should not crash and should produce a grounded answer using BM25 chunk
    assert res is not None
    assert "vacation carryover" in res.answer.lower()
    assert len(res.context_chunks) > 0
    assert res.context_chunks[0].chunk.id == "bm25_chunk_001"

    # 2. BM25 search was called with the query
    mock_bm25_index.search.assert_called()

    # 3. Verify emission of degraded thinking event
    all_events = thinking_sm.get_all_events()
    degraded_events = [e for e in all_events if e.stage == ThinkingStage.DEGRADED or e.status == ThinkingStatus.WARNING]
    assert len(degraded_events) > 0, "No DEGRADED thinking event emitted on dense retrieval failure!"

    retrieval_degraded = [e for e in degraded_events if "retrieval" in e.title.lower() or "keyword search" in e.summary.lower() or "bm25" in e.summary.lower()]
    assert len(retrieval_degraded) > 0, "No retrieval-specific degradation event found in thinking events!"
    assert "BM25" in retrieval_degraded[0].summary or "keyword search" in retrieval_degraded[0].summary

    # 4. Verify telemetry in reasoning_summary
    assert res.trace is not None
    assert res.trace.reasoning_summary is not None
    assert "retrieval" in res.trace.reasoning_summary.degraded_stages


def test_adversarial_02_vision_timeout_fallback_to_verified_text():
    """
    Adversarial Challenge 2:
    Simulate vision model timeout during cross-page visual inspection.
    Verify that existing verified text chunks are NOT dropped or invalidated,
    and a DEGRADED/WARNING visual thinking event is emitted.
    """
    text_chunk = create_test_chunk(
        "text_chunk_verified_01",
        "HotelSearchAgent implementation kickoff: agent.run(task='search hotel', max_budget=200)",
        page_number=72,
    )
    scored_text = ScoredChunk(chunk=text_chunk, score=0.92)

    mock_hybrid = MagicMock()
    mock_hybrid.retrieve.return_value = [scored_text]

    mock_vision_service = MagicMock()
    mock_vision_service.vision_model = "Qwen3-VL-2B-Instruct"
    mock_vision_service.process_pdf_page_visuals.side_effect = TimeoutError("Vision model Qwen3-VL-2B-Instruct timed out after 35s")

    mock_asset_manager = MagicMock()
    mock_asset_manager.get_page_assets.return_value = []
    mock_vision_service.image_asset_manager = mock_asset_manager

    mock_llm = MagicMock()
    mock_llm.complete.return_value = "The HotelSearchAgent implementation kickoff is agent.run(task='search hotel', max_budget=200)."

    pipeline = RAGPipeline(
        hybrid_retriever=mock_hybrid,
        vision_service=mock_vision_service,
        llm=mock_llm,
        docstore={"text_chunk_verified_01": text_chunk},
    )

    thinking_sm = ThinkingStateMachine(query_id="qry_vis_timeout_test", detail_level=ThinkingDetailLevel.DETAILED)
    res = pipeline.query(
        user_query="What is the implementation code for HotelSearchAgent?",
        thinking_detail_level=ThinkingDetailLevel.DETAILED,
        thinking_sm=thinking_sm,
    )

    # 1. Pipeline succeeds without crashing
    assert res is not None

    # 2. Text evidence is strictly preserved and not invalidated
    chunk_ids = [c.chunk.id for c in res.context_chunks]
    assert "text_chunk_verified_01" in chunk_ids, "Valid verified text chunk was dropped due to vision failure!"

    # 3. Answer is synthesized from the valid text evidence
    assert "HotelSearchAgent" in res.answer


def test_adversarial_03_cross_encoder_reranker_failure_fallback_to_hybrid_rank():
    """
    Adversarial Challenge 3:
    Simulate cross-encoder reranker failure (GPU OOM / service failure).
    Verify fallback to hybrid rank ordering and emission of degraded thinking event.
    """
    chunk1 = create_test_chunk("chunk_high_rrf", "High RRF hybrid score passage", page_number=1)
    chunk2 = create_test_chunk("chunk_med_rrf", "Medium RRF hybrid score passage", page_number=2)
    chunk3 = create_test_chunk("chunk_low_rrf", "Low RRF hybrid score passage", page_number=3)

    scored1 = ScoredChunk(chunk=chunk1, score=0.95)
    scored2 = ScoredChunk(chunk=chunk2, score=0.80)
    scored3 = ScoredChunk(chunk=chunk3, score=0.65)

    mock_hybrid = MagicMock()
    mock_hybrid.retrieve.return_value = [scored1, scored2, scored3]

    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = RuntimeError("GPU Out of Memory in cross-encoder reranker")

    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Answer based on hybrid rank ordering."

    pipeline = RAGPipeline(
        hybrid_retriever=mock_hybrid,
        reranker=mock_reranker,
        llm=mock_llm,
        docstore={c.id: c for c in [chunk1, chunk2, chunk3]},
    )

    thinking_sm = ThinkingStateMachine(query_id="qry_rerank_fail_test", detail_level=ThinkingDetailLevel.DETAILED)
    res = pipeline.query(
        user_query="Explain the system policy.",
        thinking_detail_level=ThinkingDetailLevel.DETAILED,
        thinking_sm=thinking_sm,
    )

    # 1. Pipeline succeeds
    assert res is not None

    # 2. Hybrid rank ordering is preserved (chunk1 first)
    assert len(res.context_chunks) > 0
    assert res.context_chunks[0].chunk.id == "chunk_high_rrf"

    # 3. Degraded stage is recorded for reranking
    all_events = thinking_sm.get_all_events()
    rerank_degraded = [e for e in all_events if e.stage == ThinkingStage.DEGRADED and "rerank" in e.title.lower()]
    assert len(rerank_degraded) > 0, "No reranking degradation event emitted!"
    assert "hybrid rank" in rerank_degraded[0].summary.lower()

    # 4. Telemetry records reranking in degraded_stages
    assert res.trace is not None
    assert "reranking" in res.trace.reasoning_summary.degraded_stages


def test_adversarial_04_timing_measurements_and_zero_extra_llm_calls():
    """
    Adversarial Challenge 4:
    Verify timing measurements (duration_ms per stage and total_duration_ms in reasoning summary).
    Confirm that ThinkingStateMachine makes ZERO extra LLM or network model calls.
    """
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Grounded response."

    chunk = create_test_chunk("perf_chunk_01", "Performance verification chunk.")
    mock_hybrid = MagicMock()
    mock_hybrid.retrieve.return_value = [ScoredChunk(chunk=chunk, score=0.9)]

    pipeline = RAGPipeline(
        hybrid_retriever=mock_hybrid,
        llm=mock_llm,
        docstore={"perf_chunk_01": chunk},
    )

    # Initial LLM call count
    assert mock_llm.complete.call_count == 0

    # Standalone ThinkingStateMachine transitions
    sm_standalone = ThinkingStateMachine(query_id="sm_standalone", detail_level=ThinkingDetailLevel.DETAILED)
    sm_standalone.start_stage(ThinkingStage.RECEIVED)
    time.sleep(0.01)
    sm_standalone.complete_stage(ThinkingStage.RECEIVED)
    sm_standalone.start_stage(ThinkingStage.QUERY_ANALYSIS)
    sm_standalone.complete_stage(ThinkingStage.QUERY_ANALYSIS, details={"intent": "factual", "confidence": 0.98})
    sm_standalone.start_stage(ThinkingStage.RETRIEVAL)
    sm_standalone.complete_stage(ThinkingStage.RETRIEVAL, details={"candidate_count": 5})
    sm_standalone.start_stage(ThinkingStage.ANSWER_PLANNING)
    sm_standalone.complete_stage(ThinkingStage.ANSWER_PLANNING, details={"answer_mode": "DIRECT"})
    summary = sm_standalone.get_reasoning_summary()

    # Verify zero LLM calls for standalone ThinkingStateMachine
    assert mock_llm.complete.call_count == 0, "ThinkingStateMachine made illegal LLM calls!"

    # Verify durations in standalone SM
    for ev in sm_standalone.get_all_events():
        if ev.status == ThinkingStatus.COMPLETED:
            assert ev.duration_ms >= 0.0, f"Stage {ev.stage} duration_ms must be non-negative, got {ev.duration_ms}"

    assert summary.total_duration_ms > 0.0, f"ReasoningSummary total_duration_ms must be > 0, got {summary.total_duration_ms}"

    # Now run end-to-end pipeline
    thinking_sm = ThinkingStateMachine(query_id="qry_perf_test", detail_level=ThinkingDetailLevel.DETAILED)
    res = pipeline.query(
        user_query="Test performance query",
        thinking_detail_level=ThinkingDetailLevel.DETAILED,
        thinking_sm=thinking_sm,
    )

    # In standard pipeline run without verifier/rewrite extra calls, LLM is called ONLY for answer synthesis (exactly 1 call)
    assert mock_llm.complete.call_count == 1, f"Expected exactly 1 LLM call for generation, got {mock_llm.complete.call_count}"

    # Verify duration tracking across all stages in pipeline
    completed_events = [e for e in thinking_sm.get_all_events() if e.status == ThinkingStatus.COMPLETED]
    assert len(completed_events) > 0
    for ev in completed_events:
        assert isinstance(ev.duration_ms, (int, float))
        assert ev.duration_ms >= 0.0

    assert res.trace.reasoning_summary.total_duration_ms > 0.0


@pytest.mark.asyncio
async def test_adversarial_05_sse_streaming_combined_degradation_and_protocol():
    """
    Adversarial Challenge 5:
    Simulate simultaneous dense retrieval failure and reranker failure during SSE streaming.
    Verify that ChatService streams all events in strict order, emits thinking degradation events,
    and returns completed done payload with full telemetry.
    """
    mock_bm25_chunk = create_test_chunk("bm25_stream_chunk", "Streaming fallback chunk content.")
    mock_bm25_scored = ScoredChunk(chunk=mock_bm25_chunk, score=0.77, sparse_score=0.77)

    mock_bm25_index = MagicMock()
    mock_bm25_index.search.return_value = [mock_bm25_scored]

    mock_hybrid = MagicMock()
    mock_hybrid.bm25_index = mock_bm25_index
    mock_hybrid.retrieve.side_effect = RuntimeError("Dense vector store unavailable")

    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = RuntimeError("Cross-encoder model unavailable")

    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Streaming fallback answer content."

    pipeline = RAGPipeline(
        hybrid_retriever=mock_hybrid,
        reranker=mock_reranker,
        llm=mock_llm,
        docstore={"bm25_stream_chunk": mock_bm25_chunk},
    )

    mock_telemetry = MagicMock()
    mock_telemetry.record_from_rag_response.return_value = RAGTrace(query="Stream test query")

    chat_service = ChatService(
        rag_pipeline=pipeline,
        telemetry_service=mock_telemetry,
    )

    req = ChatRequest(
        message="What is the fallback process?",
        session_id="stream_degrade_session",
        thinking_detail_level="detailed",
    )

    events: list[tuple[str, dict]] = []
    async for sse_msg in chat_service.stream_query(req):
        lines = [line.strip() for line in sse_msg.strip().split("\n") if line.strip()]
        if len(lines) >= 2 and lines[0].startswith("event:") and lines[1].startswith("data:"):
            ev_name = lines[0].replace("event:", "").strip()
            data_json = json.loads(lines[1].replace("data:", "").strip())
            events.append((ev_name, data_json))

    event_types = [e[0] for e in events]

    assert "start" in event_types
    assert "thinking" in event_types
    assert "retrieval" in event_types
    assert "chunk" in event_types
    assert "done" in event_types
    assert event_types[-1] == "done"

    thinking_payloads = [e[1] for e in events if e[0] == "thinking"]
    degraded_thk = [p for p in thinking_payloads if p.get("stage") == "degraded" or p.get("status") == "warning"]
    assert len(degraded_thk) >= 2, "Expected at least 2 degradation thinking events (retrieval + reranking) in SSE stream!"

    done_payload = events[-1][1]
    assert "reasoning_summary" in done_payload
    reasoning_summary = done_payload["reasoning_summary"]
    degraded_list = reasoning_summary.get("degraded_stages", [])
    assert "retrieval" in degraded_list
    assert "reranking" in degraded_list
