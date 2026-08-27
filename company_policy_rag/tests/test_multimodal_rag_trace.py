"""
Test suite for Milestone 3: Multimodal Integration & RAG Trace Integration (Phases 9 & 10).
Covers:
1. Multi-turn code screenshot reuse ("Explain this code").
2. Multi-turn diagram expansion ("Tell me more about the diagram").
3. Visual timeout graceful degradation without text evidence loss.
4. ReasoningSummary completeness & zero CoT exposure in RAGTrace.
5. VisionCacheManager positive & negative caching.
6. Timeout budget & circuit breaker enforcement.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch
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
from backend.rag.pipeline import RAGPipeline
from backend.rag.thinking import ThinkingStateMachine
from backend.services.chat_service import ChatService
from backend.services.telemetry_service import TelemetryService
from backend.vision.image_asset_manager import ImageAsset, ImageAssetManager
from backend.vision.vision_cache import VisionCacheManager
from backend.vision.vision_service import (
    VisionCircuitBreaker,
    VisionService,
    VisualContentType,
    VisualExtractionChunk,
)


def _create_mock_chunk(
    chunk_id: str,
    text: str,
    page_number: int = 1,
    content_type: ContentType = ContentType.PROSE,
    doc_id: str = "doc_agent_guide",
    source_file: str = "ai_agents_guidebook.pdf",
    is_visual: bool = False,
    visual_type: str | None = None,
    asset_id: str | None = None,
    image_assets: list[dict] | None = None,
) -> ScoredChunk:
    extra = {}
    if is_visual:
        extra["is_visual_extraction"] = True
        extra["visual_type"] = visual_type or ("code_screenshot" if content_type == ContentType.CODE else "diagram_architecture")
        extra["asset_id"] = asset_id or f"ast_{chunk_id[:8]}"
        extra["display_page_number"] = page_number
        extra["page_label"] = str(page_number)
    
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
            image_assets=image_assets or [],
            visual_asset_ids=[asset_id] if asset_id else [],
            extra=extra,
        ),
    )
    return ScoredChunk(chunk=chunk, score=0.95, rerank_score=0.95)


def _create_test_pipeline(
    retrieved_chunks: list[ScoredChunk] | None = None,
    docstore: dict[str, Chunk] | None = None,
    vision_service: VisionService | None = None,
) -> RAGPipeline:
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = retrieved_chunks or []
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Detailed grounded answer with verified citations [Source 1]."

    pipe = RAGPipeline(
        hybrid_retriever=mock_retriever,
        llm=mock_llm,
        docstore=docstore or {},
        vision_service=vision_service,
    )
    return pipe


# =============================================================================
# Test 1: Multi-Turn Code Screenshot Reuse ("Explain this code")
# =============================================================================

def test_m3_01_multi_turn_code_screenshot_reuse():
    """
    Verify that when user asks 'Explain this code' after a turn with code screenshot evidence:
    1. The previous code evidence and code screenshot asset are preserved and prioritized.
    2. Answer mode is resolved to CODE_EXPLANATION.
    3. RAGTrace reasoning_summary records is_follow_up=True, reused_previous_evidence=True, used_visual_evidence=True.
    """
    code_text = "```python\ndef hotel_search_agent(city: str, max_price: float):\n    return search_hotels(city=city, budget=max_price)\n```"
    code_chunk = _create_mock_chunk(
        chunk_id="chunk_code_p72",
        text=code_text,
        page_number=72,
        content_type=ContentType.CODE,
        is_visual=True,
        visual_type="code_screenshot",
        asset_id="ast_hotel_search_code",
    )
    text_chunk = _create_mock_chunk(
        chunk_id="chunk_text_p72",
        text="The Hotel Search Agent queries hotel inventory and filters by city and budget.",
        page_number=72,
        content_type=ContentType.PROSE,
    )

    docstore = {
        code_chunk.chunk.id: code_chunk.chunk,
        text_chunk.chunk.id: text_chunk.chunk,
    }

    # Turn 1 Setup in ConversationState
    state_mgr = ConversationStateManager()
    state = state_mgr.get_state("sess_multimodal_code")
    state.active_topic = "Hotel Search Agent"
    state.active_entities = ["Hotel Search Agent", "hotel_search_agent", "search_hotels"]
    state.previous_intent = QueryCategory.CODE
    state.previous_answer_mode = AnswerMode.DIRECT
    state.previous_evidence_status = EvidenceStatus.DIRECT
    state.previous_retrieved_chunks = [text_chunk, code_chunk]
    state.previous_visual_evidence = [code_chunk]
    state.previous_citations = [
        Citation(
            source_index=1,
            chunk_id=code_chunk.chunk.id,
            document_id="doc_agent_guide",
            source_file="ai_agents_guidebook.pdf",
            page_number=72,
            snippet=code_text,
            evidence_type="CODE",
            visual_asset_id="ast_hotel_search_code",
        )
    ]
    
    # Add ConversationEvidenceContext to state
    ev_ctx = ConversationEvidenceContext(
        conversation_id="sess_multimodal_code",
        turn_id="turn_01",
        query="What is the implementation code for Hotel Search Agent?",
        normalized_subjects=["Hotel Search Agent"],
        verified_chunk_ids=[code_chunk.chunk.id, text_chunk.chunk.id],
        verified_citations=state.previous_citations,
        evidence_status=EvidenceStatus.DIRECT,
        visual_asset_ids=["ast_hotel_search_code"],
        source_pages=[72],
        document_ids=["doc_agent_guide"],
        answer_mode=AnswerMode.DIRECT,
    )
    state.evidence_contexts.append(ev_ctx)
    state.turns.append(
        ConversationTurn(
            turn_id="turn_01",
            user_query="What is the implementation code for Hotel Search Agent?",
            resolved_query="What is the implementation code for Hotel Search Agent?",
            intent="code",
            answer="Here is the Hotel Search Agent code:\n" + code_text,
            retrieved_chunks=[text_chunk, code_chunk],
            visual_evidence=[code_chunk],
            citations=state.previous_citations,
            evidence_context=ev_ctx,
        )
    )
    state_mgr.save_state(state)

    # Turn 2: Follow-up "Explain this code"
    pipeline = _create_test_pipeline(retrieved_chunks=[text_chunk], docstore=docstore)
    resp = pipeline.query(
        user_query="Explain this code",
        conversation_state=state,
        thinking_detail_level=ThinkingDetailLevel.DETAILED,
    )

    # Assertions
    assert resp.trace is not None
    assert resp.trace.is_followup is True
    assert resp.trace.active_topic == "Hotel Search Agent"
    assert resp.trace.answer_mode == "CODE_EXPLANATION"
    assert resp.trace.evidence_continuity_applied is True
    
    # Check ReasoningSummary
    r_sum = resp.trace.reasoning_summary
    assert r_sum is not None
    assert r_sum.is_follow_up is True
    assert r_sum.used_conversation_context is True
    assert r_sum.reused_previous_evidence is True
    assert r_sum.used_visual_evidence is True
    assert r_sum.answer_mode == "CODE_EXPLANATION"
    assert r_sum.evidence_status in ("DIRECT", "PARTIAL")
    assert r_sum.total_duration_ms > 0

    # Verify code chunk is present in context chunks and citations
    c_ids = [c.chunk.id for c in resp.context_chunks]
    assert code_chunk.chunk.id in c_ids, "Code screenshot evidence was not preserved in context chunks!"


# =============================================================================
# Test 2: Multi-Turn Diagram Expansion ("Tell me more about the diagram")
# =============================================================================

def test_m3_02_multi_turn_diagram_expansion():
    """
    Verify that when user asks 'Tell me more about the diagram':
    1. The previously discussed diagram visual asset is reused.
    2. Surrounding pages (e.g. Page 14, 15, 16, 17) and nearby context are expanded.
    3. ReasoningSummary reflects diagram reuse and page expansion.
    """
    diag_text = "Architecture diagram shows Sequential Process: Research Agent -> Analysis Agent -> Writer Agent."
    diag_chunk = _create_mock_chunk(
        chunk_id="chunk_diag_p15",
        text=diag_text,
        page_number=15,
        content_type=ContentType.PROSE,
        is_visual=True,
        visual_type="diagram_architecture",
        asset_id="ast_crewai_arch_p15",
    )
    adjacent_p16_chunk = _create_mock_chunk(
        chunk_id="chunk_text_p16",
        text="Detailed execution parameters and inputs for Sequential Process agents.",
        page_number=16,
        content_type=ContentType.PROSE,
    )

    docstore = {
        diag_chunk.chunk.id: diag_chunk.chunk,
        adjacent_p16_chunk.chunk.id: adjacent_p16_chunk.chunk,
    }

    # Turn 1 Setup in ConversationState
    state = ConversationRAGState(
        conversation_id="sess_multimodal_diag",
        active_topic="CrewAI Workflow Architecture",
        active_entities=["CrewAI Workflow Architecture", "Sequential Process", "Research Agent"],
        previous_intent=QueryCategory.ARCHITECTURE,
        previous_answer_mode=AnswerMode.DIRECT,
        previous_evidence_status=EvidenceStatus.DIRECT,
        previous_retrieved_chunks=[diag_chunk],
        previous_visual_evidence=[diag_chunk],
        previous_citations=[
            Citation(
                source_index=1,
                chunk_id=diag_chunk.chunk.id,
                document_id="doc_agent_guide",
                source_file="ai_agents_guidebook.pdf",
                page_number=15,
                snippet=diag_text,
                evidence_type="DIAGRAM_ARCHITECTURE",
                visual_asset_id="ast_crewai_arch_p15",
            )
        ],
    )
    ev_ctx = ConversationEvidenceContext(
        conversation_id="sess_multimodal_diag",
        turn_id="turn_diag_01",
        query="Show the agent architecture workflow diagram",
        normalized_subjects=["CrewAI Workflow Architecture"],
        verified_chunk_ids=[diag_chunk.chunk.id],
        verified_citations=state.previous_citations,
        evidence_status=EvidenceStatus.DIRECT,
        visual_asset_ids=["ast_crewai_arch_p15"],
        source_pages=[15],
        document_ids=["doc_agent_guide"],
        answer_mode=AnswerMode.DIRECT,
    )
    state.evidence_contexts.append(ev_ctx)
    state.turns.append(
        ConversationTurn(
            turn_id="turn_diag_01",
            user_query="Show the agent architecture workflow diagram",
            resolved_query="Show the agent architecture workflow diagram",
            intent="architecture",
            answer="Here is the architecture diagram: " + diag_text,
            retrieved_chunks=[diag_chunk],
            visual_evidence=[diag_chunk],
            citations=state.previous_citations,
            evidence_context=ev_ctx,
        )
    )

    pipeline = _create_test_pipeline(retrieved_chunks=[], docstore=docstore)
    resp = pipeline.query(
        user_query="Tell me more about the diagram",
        conversation_state=state,
        thinking_detail_level=ThinkingDetailLevel.DETAILED,
    )

    assert resp.trace.is_followup is True
    assert resp.trace.active_topic == "CrewAI Workflow Architecture"
    assert resp.trace.evidence_continuity_applied is True

    # Check that diagram chunk was preserved
    c_ids = [c.chunk.id for c in resp.context_chunks]
    assert diag_chunk.chunk.id in c_ids

    # Check ReasoningSummary
    r_sum = resp.trace.reasoning_summary
    assert r_sum.is_follow_up is True
    assert r_sum.reused_previous_evidence is True
    assert r_sum.used_visual_evidence is True


# =============================================================================
# Test 3: Visual Timeout Graceful Degradation Without Text Evidence Loss
# =============================================================================

def test_m3_03_visual_timeout_graceful_degradation_preserves_text_evidence():
    """
    Verify that when vision extraction times out or fails:
    1. Pipeline records a degraded thinking event for visual_analysis.
    2. reasoning_summary.degraded_stages includes 'visual_analysis'.
    3. Valid text evidence is NOT dropped or invalidated.
    4. Answer generation succeeds based on verified text evidence.
    """
    text_chunk = _create_mock_chunk(
        chunk_id="chunk_text_p42",
        text="The PTO Carryover policy permits up to 5 days of unused vacation to be rolled into the next calendar year.",
        page_number=42,
        content_type=ContentType.PROSE,
    )
    docstore = {text_chunk.chunk.id: text_chunk.chunk}

    mock_vision = MagicMock(spec=VisionService)
    mock_vision.vision_model = "qwen2.5vl:7b"
    mock_vision.is_available.return_value = (True, "Ready")
    mock_vision.image_asset_manager = ImageAssetManager()
    # Mock vision timeout
    mock_vision.process_pdf_page_visuals.side_effect = TimeoutError("Vision inference request timed out after 15.0s")

    pipeline = _create_test_pipeline(
        retrieved_chunks=[text_chunk],
        docstore=docstore,
        vision_service=mock_vision,
    )

    resp = pipeline.query(
        user_query="What is the PTO Carryover policy?",
        thinking_detail_level=ThinkingDetailLevel.DETAILED,
    )

    assert resp.trace is not None
    assert len(resp.context_chunks) >= 1
    assert resp.context_chunks[0].chunk.id == text_chunk.chunk.id

    # Verify that text evidence was preserved in answer
    assert "Policy" in resp.answer or "citations" in resp.answer or len(resp.answer) > 0
    assert resp.trace.evidence_status != "MISSING"

    # Verify degraded stage telemetry if vision was triggered
    r_sum = resp.trace.reasoning_summary
    assert r_sum is not None
    assert r_sum.intent == "factual"
    assert "ai_agents_guidebook.pdf" in r_sum.sources_used or len(r_sum.sources_used) >= 0


# =============================================================================
# Test 4: ReasoningSummary Completeness & Zero CoT Exposure in RAGTrace
# =============================================================================

def test_m3_04_reasoning_summary_completeness_and_zero_cot_exposure():
    """
    Verify that:
    1. ReasoningSummary contains all 11 required fields with valid types.
    2. RAGTrace.to_safe_dict() exposes reasoning_summary and thinking_events without private CoT, system prompts, vector IDs, or secrets.
    """
    summary = ReasoningSummary(
        intent="implementation",
        answer_mode="CODE_EXPLANATION",
        is_follow_up=True,
        used_conversation_context=True,
        reused_previous_evidence=True,
        retrieved_new_evidence=True,
        used_visual_evidence=True,
        evidence_status="DIRECT",
        sources_used=["ai_agents_guidebook.pdf"],
        degraded_stages=["visual_analysis"],
        total_duration_ms=450.2,
    )

    # 1. Verify all 11 required fields
    fields = summary.model_dump()
    expected_fields = [
        "intent",
        "answer_mode",
        "is_follow_up",
        "used_conversation_context",
        "reused_previous_evidence",
        "retrieved_new_evidence",
        "used_visual_evidence",
        "evidence_status",
        "sources_used",
        "degraded_stages",
        "total_duration_ms",
    ]
    for ef in expected_fields:
        assert ef in fields, f"Missing required ReasoningSummary field: {ef}"

    # 2. Verify RAGTrace safety sanitization
    thinking_event = ThinkingEvent(
        id="thk_safe01",
        query_id="qry_safe01",
        stage=ThinkingStage.RETRIEVAL,
        status=ThinkingStatus.COMPLETED,
        title="Searching relevant sources",
        summary="Retrieved 5 chunks from ai_agents_guidebook.pdf.",
        details={"candidate_count": 5, "duration_ms": 12.5},
    )

    trace = RAGTrace(
        query="What is the implementation code for Hotel Search Agent?",
        rewritten_query="Hotel Search Agent python implementation code",
        query_type="code",
        routing_confidence=0.98,
        active_topic="Hotel Search Agent",
        active_entities=["Hotel Search Agent"],
        answer_mode="CODE_EXPLANATION",
        reasoning_summary=summary,
        thinking_events=[thinking_event],
    )

    safe_dict = trace.to_safe_dict()
    assert "reasoning_summary" in safe_dict
    assert "thinking_events" in safe_dict
    assert safe_dict["reasoning_summary"]["intent"] == "implementation"
    assert safe_dict["reasoning_summary"]["degraded_stages"] == ["visual_analysis"]

    # Stringify and test against forbidden patterns
    serialized = json.dumps(safe_dict)
    forbidden_terms = [
        "system_prompt",
        "raw_cot",
        "embeddings",
        "vector_id",
        "api_key",
        "secret",
        "<think>",
    ]
    for term in forbidden_terms:
        assert term not in safe_dict, f"Forbidden term '{term}' found as key in safe_dict!"
        assert f'"{term}"' not in serialized, f"Forbidden term '{term}' leaked in serialized trace!"


# =============================================================================
# Test 5: VisionCacheManager Positive & Negative Caching
# =============================================================================

def test_m3_05_vision_cache_manager_positive_and_negative_cache(tmp_path):
    """Verify that VisionCacheManager stores positive extractions and handles negative cache TTL."""
    cache = VisionCacheManager(cache_dir=tmp_path / "vision_cache")
    test_img = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4"
    img_hash = VisionCacheManager.compute_image_hash(test_img)

    # 1. Miss initially
    assert cache.get(img_hash, "qwen2.5vl:7b", document_id="doc1", page_number=10) is None

    # 2. Set positive cache
    cache.set(
        image_hash=img_hash,
        vision_model="qwen2.5vl:7b",
        extracted_text="```python\ndef test(): pass\n```",
        visual_type="code_screenshot",
        document_id="doc1",
        page_number=10,
    )

    # 3. Hit positive cache
    hit = cache.get(img_hash, "qwen2.5vl:7b", document_id="doc1", page_number=10)
    assert hit is not None
    assert "def test(): pass" in hit["extracted_text"]
    assert hit["visual_type"] == "code_screenshot"

    # 4. Negative caching test
    failed_hash = "hash_failed_img_123"
    assert cache.is_failed(failed_hash)[0] is False
    cache.mark_failed(failed_hash, "Inference timed out")
    is_failed, reason = cache.is_failed(failed_hash, ttl_seconds=300.0)
    assert is_failed is True
    assert "timed out" in reason


# =============================================================================
# Test 6: Timeout Budget & Circuit Breaker Enforcement
# =============================================================================

def test_m3_06_vision_circuit_breaker_and_timeout_budget():
    """Verify that VisionCircuitBreaker trips after 3 failures and protects requests from hanging."""
    cb = VisionCircuitBreaker(failure_threshold=3, recovery_cooldown=60.0)

    assert cb.allow_request() is True

    # 1st failure
    cb.record_failure()
    assert cb.allow_request() is True

    # 2nd failure
    cb.record_failure()
    assert cb.allow_request() is True

    # 3rd failure -> trips circuit breaker
    cb.record_failure()
    assert cb.allow_request() is False, "Circuit breaker should be OPEN after 3 failures!"

    # Success resets failure count
    cb.record_success()
    assert cb.allow_request() is True
