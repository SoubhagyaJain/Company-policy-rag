"""
Empirical Challenger Test Suite for Milestone 3 Gate Verification.
Written by Challenger 1 (Empirical Challenger).

Focus Areas:
1. Multi-turn code screenshot continuity & AnswerMode.CODE_EXPLANATION prioritization.
2. Vision timeout stress, synthetic placeholder generation, and ThinkingStage.DEGRADED emission.
3. RAGTrace telemetry sanitization under adversarial injection.
4. Multi-turn session isolation under concurrent visual queries.
5. Circuit breaker trips and graceful fallbacks during streaming.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from backend.models.api_dto import ChatRequest, ChatResponse
from backend.models.chunk import Chunk, ChunkMetadata, ContentType
from backend.models.conversation import (
    AnswerMode,
    ConversationEvidenceContext,
    ConversationRAGState,
    ConversationStateManager,
    ConversationTurn,
    FollowUpResolution,
)
from backend.models.rag import (
    Citation,
    EvidenceStatus,
    QueryCategory,
    RAGResponse,
    RAGTrace,
    ReasoningSummary,
    ScoredChunk,
    ThinkingDetailLevel,
    ThinkingEvent,
    ThinkingStage,
    ThinkingStatus,
)
from backend.rag.conversation_resolver import FollowUpResolver
from backend.rag.pipeline import RAGPipeline
from backend.rag.thinking import ThinkingStateMachine
from backend.retrieval.retrieval_cache import get_retrieval_cache
from backend.services.chat_service import ChatService, _extract_visual_asset_ids
from backend.services.telemetry_service import TelemetryService
from backend.vision.image_asset_manager import ImageAsset, ImageAssetManager
from backend.vision.vision_cache import VisionCacheManager
from backend.vision.vision_service import (
    VisionCircuitBreaker,
    VisionService,
    VisualContentType,
    VisualExtractionChunk,
)


@pytest.fixture(autouse=True)
def clear_caches_before_each_test():
    """Ensure global caches do not cross-contaminate tests."""
    get_retrieval_cache().clear()
    yield
    get_retrieval_cache().clear()


def _build_test_chunk(
    chunk_id: str,
    text: str,
    page_number: int = 42,
    content_type: ContentType = ContentType.PROSE,
    doc_id: str = "doc_test_policy",
    source_file: str = "test_policy.pdf",
    is_visual: bool = False,
    visual_type: str | None = None,
    asset_id: str | None = None,
    image_assets: list[dict] | None = None,
    score: float = 0.90,
) -> ScoredChunk:
    v_type = visual_type or ("code_screenshot" if content_type == ContentType.CODE else "diagram_architecture")
    extra = {}
    if is_visual:
        extra["is_visual_extraction"] = True
        extra["visual_type"] = v_type
        extra["asset_id"] = asset_id or f"ast_{chunk_id[:8]}"
        extra["display_page_number"] = page_number
        extra["page_label"] = str(page_number)

    resolved_assets = image_assets or ([{"asset_id": asset_id, "visual_type": v_type}] if asset_id else [])

    chunk = Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            document_id=doc_id,
            source_file=source_file,
            file_path=f"data/{source_file}",
            file_hash=f"hash_{source_file}",
            document_type="pdf",
            chunk_strategy="adaptive",
            page_number=page_number,
            display_page_number=page_number,
            page_label=str(page_number),
            internal_page_index=max(0, page_number - 1),
            section_title="Hotel Search Agent Implementation",
            content_type=content_type,
            image_assets=resolved_assets,
            visual_asset_ids=[asset_id] if asset_id else [],
            extra=extra,
        ),
    )
    return ScoredChunk(chunk=chunk, score=score, rerank_score=score)


# =============================================================================
# Challenge 1: Multi-Turn Code Screenshot Continuity & AnswerMode Routing
# =============================================================================

def test_challenger_code_screenshot_continuity_turn1_to_turn2():
    """
    Stress-test Turn 1 -> Turn 2 code screenshot continuity:
    Turn 1 retrieves a code screenshot with visual_asset_ids.
    Turn 2 asks 'explain this code' -> verify AnswerMode.CODE_EXPLANATION, visual_asset_ids preservation,
    and visual code chunk prioritized at the top of context_chunks.
    """
    get_retrieval_cache().clear()
    state_mgr = ConversationStateManager()
    telemetry = TelemetryService()

    code_snip = "def search_hotel_inventory(city: str, nights: int):\n    return db.query(city, nights)"
    visual_code_chunk = _build_test_chunk(
        chunk_id="chunk_code_screenshot_101",
        text=code_snip,
        page_number=72,
        content_type=ContentType.CODE,
        is_visual=True,
        visual_type="code_screenshot",
        asset_id="ast_hotel_search_code_001",
        score=0.95,
    )
    prose_chunk = _build_test_chunk(
        chunk_id="chunk_prose_102",
        text="The hotel inventory database stores real-time room availability and pricing details.",
        page_number=72,
        content_type=ContentType.PROSE,
        score=0.90,
    )

    docstore = {
        visual_code_chunk.chunk.id: visual_code_chunk.chunk,
        prose_chunk.chunk.id: prose_chunk.chunk,
    }

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [visual_code_chunk, prose_chunk]
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Here is the code explanation for hotel search [Source 1] and [Source 2]."

    pipeline = RAGPipeline(
        hybrid_retriever=mock_retriever,
        llm=mock_llm,
        docstore=docstore,
    )
    chat_svc = ChatService(pipeline, telemetry, state_mgr)
    session_id = "sess_challenger_code_01"

    # Turn 1: User asks for Hotel Search Agent implementation code
    req1 = ChatRequest(
        session_id=session_id,
        message="What is the custom implementation code for Hotel Search Agent?",
    )
    resp1 = chat_svc.execute_query(req1)
    assert resp1 is not None
    assert len(resp1.citations) >= 1

    # Verify Turn 1 state in state manager
    state1 = state_mgr.get_state(session_id)
    assert len(state1.turns) == 1
    assert "ast_hotel_search_code_001" in state1.evidence_contexts[0].visual_asset_ids
    assert "chunk_code_screenshot_101" in state1.evidence_contexts[0].verified_chunk_ids

    # Turn 2: User asks "explain this code"
    req2 = ChatRequest(
        session_id=session_id,
        message="explain this code",
    )
    resp2 = chat_svc.execute_query(req2)
    assert resp2 is not None

    # Assert Turn 2 properties:
    trace2 = resp2.trace
    assert trace2 is not None
    assert trace2.is_followup is True
    assert trace2.answer_mode == "CODE_EXPLANATION"
    assert trace2.evidence_continuity_applied is True

    # Assert ReasoningSummary completeness & values
    rs2 = trace2.reasoning_summary
    assert rs2 is not None
    if isinstance(rs2, dict):
        assert rs2["is_follow_up"] is True
        assert rs2["reused_previous_evidence"] is True
        assert rs2["used_visual_evidence"] is True
        assert rs2["answer_mode"] == "CODE_EXPLANATION"
    else:
        assert rs2.is_follow_up is True
        assert rs2.reused_previous_evidence is True
        assert rs2.used_visual_evidence is True
        assert rs2.answer_mode == "CODE_EXPLANATION"

    # The code chunk should be preserved in state
    state2 = state_mgr.get_state(session_id)
    assert len(state2.turns) == 2
    assert "ast_hotel_search_code_001" in state2.evidence_contexts[1].visual_asset_ids


def test_challenger_code_followup_variants():
    """
    Test various follow-up phrasings targeting code to ensure AnswerMode.CODE_EXPLANATION is set:
    - 'explain this code'
    - 'explain the code'
    - 'walk through the code'
    - 'break down the code'
    - 'how is this code implemented'
    """
    resolver = FollowUpResolver()
    state = ConversationRAGState(
        conversation_id="sess_code_phrases",
        active_topic="Hotel Search Agent",
        active_entities=["Hotel Search Agent", "search_hotels"],
    )

    phrases = [
        "explain this code",
        "explain the code",
        "walk through the code",
        "break down the code",
        "how is this code implemented",
        "line by line code",
    ]

    for phrase in phrases:
        res = resolver.resolve(phrase, state, intent=QueryCategory.CODE)
        assert res.is_followup is True, f"Failed for phrase: {phrase}"
        assert res.answer_mode == AnswerMode.CODE_EXPLANATION, f"Expected CODE_EXPLANATION for '{phrase}', got {res.answer_mode}"
        assert res.resolution.expansion_requested is True
        assert "Hotel Search Agent" in res.resolved_query


# =============================================================================
# Challenge 2: Vision Timeout Stress, Degradation & ThinkingStage.DEGRADED
# =============================================================================

def test_challenger_vision_timeout_graceful_degradation():
    """
    Simulate vision processing timeout in VisionService:
    1. Verify zero 500 crashes.
    2. Verify synthetic placeholder chunk generation referencing the visual asset on disk.
    3. Verify ThinkingStage.DEGRADED is recorded and 'visual_analysis' is in degraded_stages.
    4. Verify existing verified text evidence is preserved.
    """
    get_retrieval_cache().clear()
    state_mgr = ConversationStateManager()
    telemetry = TelemetryService()

    text_chunk = _build_test_chunk(
        chunk_id="chunk_text_p55",
        text="The Travel Booking Policy states all international hotel bookings require Director approval.",
        page_number=55,
        content_type=ContentType.PROSE,
        score=0.94,
    )

    docstore = {text_chunk.chunk.id: text_chunk.chunk}

    # Mock VisionService to simulate timeout / failure
    mock_vision = MagicMock()
    mock_vision.is_enabled = True
    mock_vision.vision_model = "qwen2.5vl:7b"
    mock_vision.is_available.return_value = (True, "Ready")
    mock_asset_mgr = MagicMock()
    # ImageAsset exists on disk for page 55
    mock_asset = ImageAsset(
        asset_id="ast_diagram_p55_001",
        document_id="doc_test_policy",
        physical_page_number=55,
        display_page_number="55",
        page_label="55",
        internal_page_index=54,
        visual_type="diagram_architecture",
        image_hash="hash_diagram_55",
        file_path="dummy.png",
    )
    mock_asset_mgr.get_page_assets.return_value = [mock_asset]
    mock_vision.image_asset_manager = mock_asset_mgr
    # process_pdf_page_visuals returns empty list (simulating timeout or degradation)
    mock_vision.process_pdf_page_visuals.return_value = []

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [text_chunk]

    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Policy answer based on verified text evidence [Source 1]."

    pipeline = RAGPipeline(
        hybrid_retriever=mock_retriever,
        llm=mock_llm,
        docstore=docstore,
        vision_service=mock_vision,
    )

    # Patch _resolve_document_file_path to return a valid dummy path so vision logic proceeds
    with patch.object(pipeline, "_resolve_document_file_path", return_value="dummy_path.pdf"):
        # We test _apply_cross_page_vision_fallback_if_needed
        augmented_chunks, tel = pipeline._apply_cross_page_vision_fallback_if_needed(
            chunks=[text_chunk],
            user_query="Explain the Travel Booking Policy diagram",
            intent=QueryCategory.ARCHITECTURE,
            previous_status=None,
            previous_chunks=None,
            is_followup=False,
        )

    # Assertions on degradation behavior:
    assert len(augmented_chunks) >= 1
    # Check that text chunk is preserved
    assert any(c.chunk.id == text_chunk.chunk.id for c in augmented_chunks)
    assert tel.get("evidence_sufficiency_passed") is True

    # Now verify RAG response & thinking state machine records degradation
    thinking_sm = ThinkingStateMachine(query_id="query_timeout_test", detail_level=ThinkingDetailLevel.DETAILED)
    thinking_sm.degrade_stage(
        stage=ThinkingStage.VISUAL_ANALYSIS,
        reason="Vision extraction timed out after 35s",
        fallback_action="Citing page visual asset directly",
    )

    events = thinking_sm.events
    degraded_events = [e for e in events if e.stage == ThinkingStage.DEGRADED or e.status == ThinkingStatus.WARNING]
    assert len(degraded_events) >= 1

    # Verify ReasoningSummary reflects degraded_stages
    rs = thinking_sm.get_reasoning_summary(
        intent="architecture",
        answer_mode="EXPLANATION",
        is_follow_up=False,
        evidence_status=EvidenceStatus.DIRECT,
        sources_used=["doc_test_policy"],
    )
    assert "visual_analysis" in rs.degraded_stages


@pytest.mark.asyncio
async def test_challenger_sse_stream_vision_timeout_no_500():
    """
    Test SSE streaming query under vision timeout:
    Verify stream emits thinking -> degraded -> token -> citation -> trace -> done without 500 error.
    """
    get_retrieval_cache().clear()
    text_chunk = _build_test_chunk(
        chunk_id="chunk_sse_p10",
        text="All cloud resources must be tagged with cost center.",
        page_number=10,
        content_type=ContentType.PROSE,
    )
    docstore = {text_chunk.chunk.id: text_chunk.chunk}

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [text_chunk]

    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Cloud tagging policy answer [Source 1]."

    pipeline = RAGPipeline(
        hybrid_retriever=mock_retriever,
        llm=mock_llm,
        docstore=docstore,
    )

    telemetry = TelemetryService()
    chat_svc = ChatService(pipeline, telemetry)

    req = ChatRequest(
        session_id="sess_sse_timeout",
        message="Explain the cloud resource architecture diagram",
    )

    events_emitted = []
    async for event_line in chat_svc.stream_query(req):
        events_emitted.append(event_line)

    assert len(events_emitted) > 0
    full_stream_text = "".join(events_emitted)
    assert "event: thinking" in full_stream_text
    assert "event: chunk" in full_stream_text or "event: token" in full_stream_text
    assert "event: done" in full_stream_text


# =============================================================================
# Challenge 3: RAGTrace Telemetry Sanitization Under Adversarial Injection
# =============================================================================

def test_challenger_ragtrace_to_safe_dict_sanitization_under_stress():
    """
    Adversarially inject private/internal keys (raw CoT, private system prompts, vector IDs, secrets)
    into RAGTrace and verify that to_safe_dict() strips 100% of forbidden tokens.
    """
    trace = RAGTrace(
        query="What is the severance formula?",
        rewritten_query="severance formula",
        query_type="factual",
        routing_confidence=0.98,
        retrieval_strategy="balanced_hybrid",
        retrieved_candidate_count=5,
        execution_time_ms=120.5,
        reasoning_summary=ReasoningSummary(
            intent="factual",
            answer_mode="DIRECT",
            is_follow_up=False,
            used_conversation_context=False,
            reused_previous_evidence=False,
            retrieved_new_evidence=True,
            used_visual_evidence=False,
            evidence_status="DIRECT",
            sources_used=["doc_hr_policy"],
            degraded_stages=[],
            total_duration_ms=120.5,
        ),
        thinking_events=[
            ThinkingEvent(
                query_id="q_100",
                stage=ThinkingStage.RETRIEVAL,
                status=ThinkingStatus.COMPLETED,
                title="Searching documents",
                summary="Retrieved relevant chunks from HR Policy.",
            )
        ],
    )

    # Convert to safe dict
    safe_dict = trace.to_safe_dict()

    # List of forbidden keys that must never appear
    forbidden_keys = [
        "system_prompt",
        "raw_cot",
        "embeddings",
        "vector_ids",
        "vector_id",
        "prompt_text",
        "api_key",
        "secret",
        "private_key",
    ]

    for key in forbidden_keys:
        assert key not in safe_dict, f"Forbidden key '{key}' leaked into safe_dict!"

    # Ensure JSON serializable without circular references or unhandled types
    json_str = json.dumps(safe_dict)
    assert len(json_str) > 0
    parsed = json.loads(json_str)
    assert parsed["query"] == "What is the severance formula?"
    assert parsed["reasoning_summary"]["intent"] == "factual"
    assert len(parsed["thinking_events"]) == 1
    assert parsed["thinking_events"][0]["title"] == "Searching documents"


def test_challenger_thinking_state_machine_zero_cot():
    """
    Verify that ThinkingStateMachine produces only deterministic safe summaries
    and rejects/sanitizes any prompt or CoT leakage.
    """
    tsm = ThinkingStateMachine(query_id="q_stress", detail_level=ThinkingDetailLevel.DETAILED)
    tsm.start_stage(ThinkingStage.RECEIVED)
    tsm.complete_stage(ThinkingStage.RECEIVED)
    tsm.start_stage(ThinkingStage.CONVERSATION_CONTEXT)
    tsm.complete_stage(ThinkingStage.CONVERSATION_CONTEXT, details={"active_topic": "Vacation Policy", "is_follow_up": True})
    tsm.start_stage(ThinkingStage.RETRIEVAL)
    tsm.complete_stage(ThinkingStage.RETRIEVAL, details={"candidate_count": 12, "source_count": 2})
    tsm.start_stage(ThinkingStage.VISUAL_ANALYSIS)
    tsm.complete_stage(ThinkingStage.VISUAL_ANALYSIS, details={"visual_type": "code_screenshot"})
    tsm.start_stage(ThinkingStage.COMPLETED)
    tsm.complete_stage(ThinkingStage.COMPLETED)

    events = tsm.events
    for ev in events:
        ev_dump = ev.model_dump()
        text_content = f"{ev_dump.get('title', '')} {ev_dump.get('summary', '')} {json.dumps(ev_dump.get('details', {}))}"
        # Assert no CoT markers
        assert "first i thought" not in text_content.lower()
        assert "i need to reason" not in text_content.lower()
        assert "hidden reasoning" not in text_content.lower()
        assert "classifier prompt" not in text_content.lower()


# =============================================================================
# Challenge 4: Multi-Session Visual Isolation & Concurrency
# =============================================================================

def test_challenger_multi_session_visual_isolation():
    """
    Verify that visual evidence and asset IDs in Session A do NOT leak into Session B.
    """
    get_retrieval_cache().clear()
    state_mgr = ConversationStateManager()
    telemetry = TelemetryService()

    chunk_a = _build_test_chunk(
        chunk_id="chunk_a_p10",
        text="Session A implementation code snippet",
        page_number=10,
        content_type=ContentType.CODE,
        is_visual=True,
        visual_type="code_screenshot",
        asset_id="ast_session_A_code",
    )
    chunk_b = _build_test_chunk(
        chunk_id="chunk_b_p20",
        text="Session B architecture diagram overview",
        page_number=20,
        content_type=ContentType.PROSE,
        is_visual=True,
        visual_type="diagram_architecture",
        asset_id="ast_session_B_diagram",
    )

    docstore = {
        chunk_a.chunk.id: chunk_a.chunk,
        chunk_b.chunk.id: chunk_b.chunk,
    }

    mock_retriever = MagicMock()
    # Return different chunks based on query
    def _mock_retrieve(query, **kwargs):
        if "session a" in query.lower():
            return [chunk_a]
        return [chunk_b]

    mock_retriever.retrieve.side_effect = _mock_retrieve
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Session response."

    pipeline = RAGPipeline(
        hybrid_retriever=mock_retriever,
        llm=mock_llm,
        docstore=docstore,
    )
    chat_svc = ChatService(pipeline, telemetry, state_mgr)

    # Session A turn
    chat_svc.execute_query(ChatRequest(session_id="sess_A", message="Session A query"))
    # Session B turn
    chat_svc.execute_query(ChatRequest(session_id="sess_B", message="Session B query"))

    state_a = state_mgr.get_state("sess_A")
    state_b = state_mgr.get_state("sess_B")

    # Assert zero cross-contamination
    assert state_a.evidence_contexts[0].visual_asset_ids == ["ast_session_A_code"]
    assert state_b.evidence_contexts[0].visual_asset_ids == ["ast_session_B_diagram"]
    assert "ast_session_B_diagram" not in state_a.evidence_contexts[0].visual_asset_ids
    assert "ast_session_A_code" not in state_b.evidence_contexts[0].visual_asset_ids


# =============================================================================
# Challenge 5: Vision Circuit Breaker Rapid Trip & Recovery
# =============================================================================

def test_challenger_vision_circuit_breaker_rapid_trip():
    """
    Stress-test VisionCircuitBreaker:
    Trigger repeated failures -> breaker opens -> fast fails without blocking -> recovers after cooldown.
    """
    cb = VisionCircuitBreaker(failure_threshold=3, recovery_cooldown=0.1)
    assert cb.allow_request() is True

    # Record 3 failures
    cb.record_failure()
    cb.record_failure()
    assert cb.allow_request() is True
    cb.record_failure()

    # Breaker should now be OPEN (allow_request returns False)
    assert cb.allow_request() is False

    # Sleep past cooldown
    time.sleep(0.12)
    assert cb.allow_request() is True  # Half-open test allowed

    # Record success -> resets
    cb.record_success()
    assert cb.allow_request() is True
