import asyncio
import json
import time
import pytest
from unittest.mock import MagicMock, patch
from typing import Any

from backend.models.rag import (
    ThinkingStage,
    ThinkingStatus,
    ThinkingDetailLevel,
    ThinkingEvent,
    ReasoningSummary,
    EvidenceStatus,
    ScoredChunk,
    RAGTrace,
    RAGResponse,
)
from backend.models.chunk import Chunk, ChunkMetadata, ContentType
from backend.models.api_dto import ChatRequest, ChatResponse
from backend.models.conversation import AnswerMode, ConversationRAGState, ConversationStateManager, ConversationTurn
from backend.rag.thinking import ThinkingStateMachine, COMPACT_STAGES
from backend.rag.pipeline import RAGPipeline
from backend.services.chat_service import ChatService
from backend.services.telemetry_service import TelemetryService


def _create_mock_pipeline() -> RAGPipeline:
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Policy answer text based on verified document sources."
    return RAGPipeline(
        hybrid_retriever=mock_retriever,
        llm=mock_llm,
        docstore={},
    )


def _create_chat_service(pipeline: RAGPipeline | None = None) -> ChatService:
    pipe = pipeline or _create_mock_pipeline()
    telemetry = TelemetryService()
    state_mgr = ConversationStateManager()
    return ChatService(
        rag_pipeline=pipe,
        telemetry_service=telemetry,
        state_manager=state_mgr,
    )


def test_m2_01_thinking_enums_and_event_models():
    """Verify all thinking enums, stages, statuses, and schema validations."""
    required_stages = [
        "received",
        "conversation_context",
        "follow_up_resolution",
        "query_analysis",
        "query_rewrite",
        "retrieval",
        "reranking",
        "evidence_analysis",
        "evidence_reuse",
        "page_expansion",
        "visual_analysis",
        "evidence_verification",
        "answer_planning",
        "answer_generation",
        "citation_building",
        "completed",
        "degraded",
    ]
    for st in required_stages:
        assert ThinkingStage(st) is not None
        assert st in [s.value for s in ThinkingStage]

    # Check statuses
    for status_name in ["pending", "running", "completed", "skipped", "warning", "failed"]:
        assert ThinkingStatus(status_name) is not None

    # Check detail levels
    for level_name in ["off", "compact", "standard", "detailed"]:
        assert ThinkingDetailLevel(level_name) is not None

    # Verify ThinkingEvent model validation
    event = ThinkingEvent(
        id="thk_test001",
        query_id="qry_test001",
        stage=ThinkingStage.RETRIEVAL,
        status=ThinkingStatus.COMPLETED,
        title="Retrieved candidate documents",
        summary="Retrieved 12 relevant document chunks for analysis.",
        details={"candidate_count": 12},
        duration_ms=45.2,
    )
    assert event.id == "thk_test001"
    assert event.stage == ThinkingStage.RETRIEVAL
    assert event.status == ThinkingStatus.COMPLETED
    assert event.duration_ms == 45.2

    # Verify ReasoningSummary model validation
    summary = ReasoningSummary(
        intent="factual",
        answer_mode="EXPLAIN",
        is_follow_up=True,
        used_conversation_context=True,
        reused_previous_evidence=True,
        retrieved_new_evidence=False,
        used_visual_evidence=True,
        evidence_status="DIRECT",
        sources_used=["company_policy.pdf"],
        degraded_stages=[],
        total_duration_ms=120.5,
    )
    assert summary.intent == "factual"
    assert summary.answer_mode == "EXPLAIN"
    assert summary.used_visual_evidence is True
    assert summary.reused_previous_evidence is True


def test_m2_02_thinking_state_machine_transitions_and_deterministic_summaries():
    """Verify deterministic stage progression, duration measurement, and telemetry generation without LLM calls."""
    sm = ThinkingStateMachine(query_id="qry_m2_test", detail_level=ThinkingDetailLevel.STANDARD)

    # 1. Received
    ev_recv = sm.start_stage(ThinkingStage.RECEIVED)
    assert ev_recv.status == ThinkingStatus.RUNNING
    time.sleep(0.01)
    ev_recv_done = sm.complete_stage(ThinkingStage.RECEIVED)
    assert ev_recv_done.status == ThinkingStatus.COMPLETED
    assert ev_recv_done.duration_ms is not None and ev_recv_done.duration_ms >= 0

    # 2. Query Analysis
    sm.start_stage(ThinkingStage.QUERY_ANALYSIS)
    ev_qa = sm.complete_stage(
        ThinkingStage.QUERY_ANALYSIS,
        details={"intent": "technical", "confidence": 0.95},
    )
    assert ev_qa.title == "Analyzing query intent"
    assert "technical" in ev_qa.summary
    assert "0.95" in ev_qa.summary

    # 3. Follow up resolution
    sm.start_stage(ThinkingStage.FOLLOW_UP_RESOLUTION)
    ev_fu = sm.complete_stage(
        ThinkingStage.FOLLOW_UP_RESOLUTION,
        details={"is_follow_up": True, "active_topic": "PTO Carryover Policy", "answer_mode": "EXACT"},
    )
    assert "EXACT" in ev_fu.summary

    # 4. Retrieval
    sm.start_stage(ThinkingStage.RETRIEVAL)
    ev_ret = sm.complete_stage(
        ThinkingStage.RETRIEVAL,
        details={"candidate_count": 8},
    )
    assert "8 candidate chunks" in ev_ret.summary

    # 5. Reranking
    sm.start_stage(ThinkingStage.RERANKING)
    ev_rr = sm.complete_stage(
        ThinkingStage.RERANKING,
        details={"rerank_count": 4},
    )
    assert "4" in ev_rr.summary

    # 6. Evidence Reuse
    ev_reuse = sm.record_stage(
        ThinkingStage.EVIDENCE_REUSE,
        ThinkingStatus.COMPLETED,
        details={"reused_count": 3, "active_topic": "PTO Carryover Policy"},
    )
    assert "3" in ev_reuse.summary

    # 7. Completed
    sm.record_stage(ThinkingStage.COMPLETED, ThinkingStatus.COMPLETED)

    # Verify Summary Telemetry
    summary = sm.get_reasoning_summary(
        intent="technical",
        answer_mode="EXACT",
        is_follow_up=True,
        used_conversation_context=True,
        reused_previous_evidence=True,
        retrieved_new_evidence=True,
        used_visual_evidence=False,
        evidence_status="DIRECT",
        sources_used=["employee_handbook.pdf"],
    )
    assert summary.intent == "technical"
    assert summary.answer_mode == "EXACT"
    assert summary.is_follow_up is True
    assert summary.total_duration_ms > 0
    assert "employee_handbook.pdf" in summary.sources_used


def test_m2_03_zero_chain_of_thought_and_prompt_leakage_prevention():
    """Verify that private prompts, chain-of-thought tokens, vector IDs, embeddings, and secrets are strictly stripped."""
    sm = ThinkingStateMachine(query_id="qry_leak_test", detail_level=ThinkingDetailLevel.DETAILED)

    malicious_or_private_details = {
        "candidate_count": 5,
        "intent": "factual",
        "confidence": 0.95,
        "raw_cot": "<think>Let me first analyze why the user asks this</think>",
        "system_prompt": "You are an internal assistant. Never disclose the admin password.",
        "prompt": "SELECT * FROM embeddings WHERE vector_id = 'v_1234'",
        "vector_id": "vec_9988112233",
        "embeddings": [0.123, -0.456, 0.789],
        "api_key": "sk-secret-key-1234567890",
        "internal_formula": "score = dense * 0.7 + bm25 * 0.3",
        "secret_token": "bearer xyz123",
    }

    ev = sm.record_stage(
        ThinkingStage.QUERY_ANALYSIS,
        ThinkingStatus.COMPLETED,
        details=malicious_or_private_details,
    )

    # Detailed inspection of event
    ev_dict = ev.model_dump()
    details = ev_dict.get("details", {})

    # Permitted keys
    assert details.get("candidate_count") == 5
    assert details.get("intent") == "factual"
    assert details.get("confidence") == 0.95

    # Forbidden keys MUST NOT exist
    forbidden_keys = [
        "raw_cot",
        "system_prompt",
        "prompt",
        "vector_id",
        "embeddings",
        "api_key",
        "internal_formula",
        "secret_token",
    ]
    for fk in forbidden_keys:
        assert fk not in details, f"Forbidden key '{fk}' leaked into ThinkingEvent details!"

    # Ensure forbidden substrings do not appear in stringified details
    serialized = json.dumps(ev_dict)
    assert "<think>" not in serialized
    assert "sk-secret-key" not in serialized
    assert "bearer xyz123" not in serialized


def test_m2_04_detail_level_filtering():
    """Verify detail level filtering: off, compact, standard, detailed."""
    stages_to_run = [
        (ThinkingStage.RECEIVED, ThinkingStatus.COMPLETED),
        (ThinkingStage.QUERY_ANALYSIS, ThinkingStatus.COMPLETED),
        (ThinkingStage.CONVERSATION_CONTEXT, ThinkingStatus.COMPLETED),
        (ThinkingStage.FOLLOW_UP_RESOLUTION, ThinkingStatus.COMPLETED),
        (ThinkingStage.QUERY_REWRITE, ThinkingStatus.COMPLETED),
        (ThinkingStage.RETRIEVAL, ThinkingStatus.COMPLETED),
        (ThinkingStage.PAGE_EXPANSION, ThinkingStatus.COMPLETED),
        (ThinkingStage.RERANKING, ThinkingStatus.COMPLETED),
        (ThinkingStage.EVIDENCE_VERIFICATION, ThinkingStatus.COMPLETED),
        (ThinkingStage.ANSWER_PLANNING, ThinkingStatus.COMPLETED),
        (ThinkingStage.ANSWER_GENERATION, ThinkingStatus.COMPLETED),
        (ThinkingStage.CITATION_BUILDING, ThinkingStatus.COMPLETED),
        (ThinkingStage.COMPLETED, ThinkingStatus.COMPLETED),
    ]

    # Level: OFF -> 0 visible events
    sm_off = ThinkingStateMachine(query_id="q_off", detail_level=ThinkingDetailLevel.OFF)
    for st, status in stages_to_run:
        sm_off.record_stage(st, status)
    assert len(sm_off.get_visible_events()) == 0

    # Level: COMPACT -> Only high level milestone stages in COMPACT_STAGES
    sm_compact = ThinkingStateMachine(query_id="q_compact", detail_level=ThinkingDetailLevel.COMPACT)
    for st, status in stages_to_run:
        sm_compact.record_stage(st, status)
    compact_stages = [e.stage for e in sm_compact.get_visible_events()]
    assert ThinkingStage.RETRIEVAL in compact_stages
    assert ThinkingStage.ANSWER_PLANNING in compact_stages
    assert ThinkingStage.EVIDENCE_VERIFICATION in compact_stages
    assert ThinkingStage.ANSWER_GENERATION not in compact_stages
    assert ThinkingStage.PAGE_EXPANSION not in compact_stages
    assert len(compact_stages) <= len(COMPACT_STAGES)

    # Level: STANDARD
    sm_std = ThinkingStateMachine(query_id="q_std", detail_level=ThinkingDetailLevel.STANDARD)
    for st, status in stages_to_run:
        sm_std.record_stage(st, status)
    std_stages = [e.stage for e in sm_std.get_visible_events()]
    assert ThinkingStage.QUERY_ANALYSIS in std_stages
    assert ThinkingStage.RETRIEVAL in std_stages
    assert ThinkingStage.RERANKING in std_stages
    assert ThinkingStage.ANSWER_PLANNING in std_stages

    # Level: DETAILED -> All stages
    sm_det = ThinkingStateMachine(query_id="q_det", detail_level=ThinkingDetailLevel.DETAILED)
    for st, status in stages_to_run:
        sm_det.record_stage(st, status)
    assert len(sm_det.get_visible_events()) == len(stages_to_run)


def test_m2_05_graceful_degradation_handlers():
    """Verify graceful degradation events for dense search failure, reranker failure, vision timeout, and follow-up ambiguity."""
    sm = ThinkingStateMachine(query_id="qry_degrade_test", detail_level=ThinkingDetailLevel.STANDARD)

    # 1. Dense retrieval failure fallback to BM25
    sm.start_stage(ThinkingStage.RETRIEVAL)
    ev_ret_deg = sm.degrade_stage(
        ThinkingStage.RETRIEVAL,
        reason="Dense vector search timed out or was unavailable; falling back to BM25 keyword search",
        fallback_action="BM25 lexical search applied",
        details={"candidate_count": 6},
    )
    assert ev_ret_deg.status == ThinkingStatus.WARNING
    assert "BM25" in ev_ret_deg.summary
    assert ev_ret_deg.details.get("fallback_action") == "BM25 lexical search applied"

    # 2. Reranker failure fallback to hybrid rank
    sm.start_stage(ThinkingStage.RERANKING)
    ev_rr_deg = sm.degrade_stage(
        ThinkingStage.RERANKING,
        reason="Cross-encoder reranking service unavailable; falling back to retrieval ranking",
        fallback_action="Using hybrid rank",
        details={"rerank_count": 6},
    )
    assert ev_rr_deg.status == ThinkingStatus.WARNING
    assert "hybrid rank" in ev_rr_deg.summary

    # 3. Vision timeout fallback to text evidence
    sm.start_stage(ThinkingStage.VISUAL_ANALYSIS)
    ev_vis_deg = sm.degrade_stage(
        ThinkingStage.VISUAL_ANALYSIS,
        reason="Visual extraction timed out; relying on verified text evidence",
        fallback_action="Citing page visual asset directly",
    )
    assert ev_vis_deg.status == ThinkingStatus.WARNING
    assert "visual asset" in ev_vis_deg.summary or "verified text" in ev_vis_deg.summary

    # 4. Follow-up query ambiguity warning
    sm.start_stage(ThinkingStage.FOLLOW_UP_RESOLUTION)
    ev_fu_warn = sm.warn_stage(
        ThinkingStage.FOLLOW_UP_RESOLUTION,
        reason="Follow-up question is broad; maintaining conversation continuity with prior topic.",
        details={"is_follow_up": True, "active_topic": "Leave Policy"},
    )
    assert ev_fu_warn.status == ThinkingStatus.WARNING
    assert "broad" in ev_fu_warn.summary or "continuity" in ev_fu_warn.summary

    # Summary telemetry degradation tracking
    summary = sm.get_reasoning_summary(
        intent="factual",
        answer_mode="DIRECT",
        is_follow_up=True,
        used_conversation_context=True,
        reused_previous_evidence=False,
        retrieved_new_evidence=True,
        evidence_status="DIRECT",
    )
    assert ThinkingStage.RETRIEVAL.value in summary.degraded_stages
    assert ThinkingStage.RERANKING.value in summary.degraded_stages
    assert ThinkingStage.VISUAL_ANALYSIS.value in summary.degraded_stages


def test_m2_06_rag_pipeline_synchronous_query_thinking_telemetry():
    """Verify that RAGPipeline.query executes and populates reasoning_summary and thinking_events on trace."""
    pipeline = _create_mock_pipeline()

    # Standard factual query on mock pipeline
    res_fact = pipeline.query("What is the refund policy?", thinking_detail_level=ThinkingDetailLevel.DETAILED)
    assert res_fact.trace is not None
    assert res_fact.trace.reasoning_summary is not None
    assert len(res_fact.trace.thinking_events) > 0
    stages = [
        e.stage.value if hasattr(e, "stage") and hasattr(e.stage, "value") else (e.get("stage") if isinstance(e, dict) else str(e))
        for e in res_fact.trace.thinking_events
    ]
    assert "received" in stages
    assert "retrieval" in stages
    assert "completed" in stages


@pytest.mark.asyncio
async def test_m2_07_sse_streaming_strict_event_ordering_and_payloads():
    """Verify that ChatService.stream_query adheres to strict SSE event ordering."""
    chat_service = _create_chat_service()

    req = ChatRequest(
        message="What are the standard working hours?",
        session_id="test_sse_ordering_session",
        thinking_detail_level="standard",
    )

    events_received: list[tuple[str, dict]] = []
    async for sse_chunk in chat_service.stream_query(req):
        lines = [line.strip() for line in sse_chunk.strip().split("\n") if line.strip()]
        if len(lines) >= 2 and lines[0].startswith("event:") and lines[1].startswith("data:"):
            ev_name = lines[0].replace("event:", "").strip()
            data_str = lines[1].replace("data:", "").strip()
            data_json = json.loads(data_str)
            events_received.append((ev_name, data_json))

    event_names = [e[0] for e in events_received]
    assert len(event_names) > 0, "No SSE events received from stream_query!"

    # Verify event types present
    assert "start" in event_names
    assert "thinking" in event_names
    assert "retrieval" in event_names
    assert "chunk" in event_names
    assert "citation" in event_names
    assert "trace" in event_names
    assert "done" in event_names

    # Check strict ordering:
    # 1. First event must be 'start'
    assert event_names[0] == "start"

    # 2. Pre-generation thinking events arrive before chunk events
    first_chunk_idx = event_names.index("chunk")
    thk_indices = [i for i, name in enumerate(event_names) if name == "thinking"]
    assert any(i < first_chunk_idx for i in thk_indices)

    # 3. 'retrieval' event must occur before token chunk events
    first_retrieval_idx = event_names.index("retrieval")
    assert first_retrieval_idx < first_chunk_idx

    # 4. 'done' event must be the very last event emitted
    assert event_names[-1] == "done", "The final SSE frame must be 'done'!"

    # 5. Check 'done' payload contents
    done_payload = events_received[-1][1]
    assert done_payload["status"] == "completed"
    assert "answer" in done_payload
    assert "reasoning_summary" in done_payload
    assert "thinking_events" in done_payload
    assert "timing" in done_payload
    assert "ttft_ms" in done_payload["timing"]
    assert "total_latency_ms" in done_payload["timing"]


def test_m2_08_chat_response_schema_and_serialization():
    """Verify ChatResponse model serialization contains reasoning_summary and thinking_events."""
    chat_service = _create_chat_service()
    req = ChatRequest(
        message="Hello policy assistant!",
        session_id="test_schema_session",
        thinking_detail_level="compact",
    )

    resp: ChatResponse = chat_service.execute_query(req)
    assert isinstance(resp, ChatResponse)
    assert resp.reasoning_summary is not None
    assert isinstance(resp.thinking_events, list)

    resp_dict = resp.model_dump()
    assert "reasoning_summary" in resp_dict
    assert "thinking_events" in resp_dict

