"""
Unit and Integration Tests for Milestone 1:
Backend Conversation & Evidence Continuity (Phases 2, 3, 4, 5, 12).
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock
import pytest

from backend.models.api_dto import ChatRequest
from backend.models.chunk import Chunk, ChunkMetadata, ContentType
from backend.models.conversation import (
    AnswerMode,
    ConversationEvidenceContext,
    ConversationRAGState,
    ConversationStateManager,
    ConversationTurn,
    ExpansionPlan,
    FollowUpResolution,
)
from backend.models.rag import (
    Citation,
    EvidenceStatus,
    QueryCategory,
    RAGResponse,
    RAGTrace,
    ScoredChunk,
)
from backend.rag.consistency_guard import ConversationConsistencyGuard
from backend.rag.conversation_resolver import FollowUpResolver, ConversationResolver
from backend.rag.evidence_gate import EvidenceSufficiencyGate, compute_monotonic_evidence_status
from backend.rag.pipeline import GROUNDED_SYSTEM_PROMPT
from backend.services.chat_service import ChatService


def _create_mock_scored_chunk(
    chunk_id: str,
    text: str,
    page_number: int = 1,
    content_type: ContentType = ContentType.PROSE,
    doc_id: str = "doc_test",
    source_file: str = "guide.pdf",
    image_assets: list[dict] | None = None,
) -> ScoredChunk:
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
            section_title="Test Section",
            content_type=content_type,
            image_assets=image_assets or [],
        ),
    )
    return ScoredChunk(chunk=chunk, score=0.95)


def test_m1_01_models_and_enums():
    """Verify all required M1 models and enums exist with exact signatures."""
    # 1. AnswerMode enum
    assert AnswerMode.DIRECT == "DIRECT"
    assert AnswerMode.SUMMARY == "SUMMARY"
    assert AnswerMode.EXPLANATION == "EXPLANATION"
    assert AnswerMode.DETAILED == "DETAILED"
    assert AnswerMode.STEP_BY_STEP == "STEP_BY_STEP"
    assert AnswerMode.CODE_EXPLANATION == "CODE_EXPLANATION"
    assert AnswerMode.COMPARISON == "COMPARISON"
    assert AnswerMode.CONTINUATION == "CONTINUATION"
    assert AnswerMode.EXPAND == "EXPAND"

    # 2. ExpansionPlan model
    plan = ExpansionPlan(
        restate_subject="minimal",
        preserve_prior_facts=True,
        retrieve_additional_context=True,
        inspect_adjacent_evidence=True,
        explain_components=True,
        explain_execution_flow=True,
        explain_code_line_by_line=True,
        target_detail_level="detailed",
    )
    assert plan.restate_subject == "minimal"
    assert plan.preserve_prior_facts is True
    assert plan.explain_code_line_by_line is True

    # 3. FollowUpResolution model
    resolution = FollowUpResolution(
        is_follow_up=True,
        confidence=0.95,
        resolved_query="Provide a detailed explanation of Hotel Search Agent implementation",
        primary_subject="Hotel Search Agent",
        referenced_answer_id="turn_123",
        answer_mode=AnswerMode.DETAILED,
        expansion_requested=True,
        requested_detail_level="detailed",
        preserve_previous_evidence=True,
        evidence_continuity_ids=["c1", "c2"],
        ambiguity_detected=False,
        rationale="Follow up detected",
    )
    assert resolution.is_follow_up is True
    assert resolution.confidence == 0.95
    assert resolution.primary_subject == "Hotel Search Agent"
    assert resolution.referenced_answer_id == "turn_123"
    assert resolution.answer_mode == AnswerMode.DETAILED
    assert resolution.expansion_requested is True
    assert resolution.preserve_previous_evidence is True
    assert resolution.evidence_continuity_ids == ["c1", "c2"]
    assert resolution.ambiguity_detected is False

    # 4. ConversationEvidenceContext model
    cit = Citation(
        source_index=1,
        chunk_id="c1",
        document_id="doc_1",
        source_file="handbook.pdf",
        snippet="Hotel search code",
    )
    evidence_ctx = ConversationEvidenceContext(
        conversation_id="sess_abc",
        turn_id="turn_1",
        query="What is Hotel Search Agent?",
        normalized_subjects=["Hotel Search Agent"],
        verified_chunk_ids=["c1"],
        verified_citations=[cit],
        evidence_status=EvidenceStatus.DIRECT,
        visual_asset_ids=["ast_1"],
        source_pages=[5],
        document_ids=["doc_1"],
        answer_mode=AnswerMode.DIRECT,
    )
    assert evidence_ctx.conversation_id == "sess_abc"
    assert evidence_ctx.session_id == "sess_abc"
    assert evidence_ctx.verified_chunk_ids == ["c1"]
    assert evidence_ctx.source_pages == [5]
    assert evidence_ctx.evidence_status == EvidenceStatus.DIRECT


def test_m1_02_layered_follow_up_resolver_tell_me_about_it_in_detail():
    """Verify Layer 1-4 resolution for 'tell me about it in detail' on code implementation."""
    resolver = FollowUpResolver()
    state = ConversationRAGState(
        conversation_id="conv_m1_02",
        active_topic="Hotel Search Agent",
        active_entities=["Hotel Search Agent", "convert_currency", "CrewAI"],
        previous_intent=QueryCategory.IMPLEMENTATION,
        turns=[
            ConversationTurn(
                turn_id="turn_01",
                user_query="What is the implementation code for Hotel Search Agent?",
                resolved_query="What is the implementation code for Hotel Search Agent?",
                intent="implementation",
                answer="Here is the Hotel Search Agent implementation code: def hotel_search(): ...",
            )
        ],
    )

    query = "tell me about it in detail"
    result = resolver.resolve(query, state, intent=QueryCategory.IMPLEMENTATION)

    assert result.is_followup is True
    assert result.topic_shift is False
    assert result.active_topic == "Hotel Search Agent"
    assert result.answer_mode in (AnswerMode.EXPAND, AnswerMode.DETAILED)
    assert result.resolution is not None
    assert result.resolution.is_follow_up is True
    assert result.resolution.primary_subject == "Hotel Search Agent"
    assert result.resolution.referenced_answer_id == "turn_01"
    assert result.resolution.expansion_requested is True
    assert result.resolution.preserve_previous_evidence is True

    # Check that resolved query generically expands without hardcoded document names
    assert "Hotel Search Agent" in result.resolved_query
    assert "it" not in result.resolved_query.lower().split()
    assert any(k in result.resolved_query.lower() for k in ("implementation", "code", "detail", "explanation"))

    # Check expansion plan
    assert result.expansion_plan is not None
    assert result.expansion_plan.preserve_prior_facts is True
    assert result.expansion_plan.retrieve_additional_context is True
    assert result.expansion_plan.target_detail_level == "detailed"


def test_m1_03_explain_this_code_resolution():
    """Verify Layer 1-4 resolution for 'explain this code'."""
    resolver = FollowUpResolver()
    state = ConversationRAGState(
        conversation_id="conv_m1_03",
        active_topic="Content Creator Agent",
        active_entities=["Content Creator Agent", "Agent", "Task"],
        previous_intent=QueryCategory.CODE,
        turns=[
            ConversationTurn(
                turn_id="turn_code_01",
                user_query="Show me the content creator agent",
                resolved_query="Show me the content creator agent",
                intent="code",
                answer="```python\ncontent_agent = Agent(role='Creator')\n```",
            )
        ],
    )

    query = "explain this code"
    result = resolver.resolve(query, state, intent=QueryCategory.CODE)

    assert result.is_followup is True
    assert result.answer_mode == AnswerMode.CODE_EXPLANATION
    assert result.resolution.primary_subject == "Content Creator Agent"
    assert result.resolution.referenced_answer_id == "turn_code_01"
    assert "Content Creator Agent" in result.resolved_query
    assert "code" in result.resolved_query.lower()
    assert result.expansion_plan.explain_code_line_by_line is True


def test_m1_04_diagram_explanation_resolution():
    """Verify Layer 1-4 resolution for 'tell me more about the diagram'."""
    resolver = FollowUpResolver()
    state = ConversationRAGState(
        conversation_id="conv_m1_04",
        active_topic="CrewAI Workflow Architecture",
        active_entities=["CrewAI Workflow Architecture", "Sequential Process"],
        previous_intent=QueryCategory.ARCHITECTURE,
    )

    query = "tell me more about the diagram"
    result = resolver.resolve(query, state, intent=QueryCategory.ARCHITECTURE)

    assert result.is_followup is True
    assert "CrewAI Workflow Architecture" in result.resolved_query
    assert "diagram" in result.resolved_query.lower() or "architecture" in result.resolved_query.lower()


def test_m1_05_conversation_consistency_guard_downgrade_protection():
    """Verify ConversationConsistencyGuard prevents downgrading to MISSING when prior evidence exists."""
    guard = ConversationConsistencyGuard()

    c1 = _create_mock_scored_chunk("c1", "Hotel Search Agent code definition: def search_hotel(): pass", page_number=72)
    cit1 = Citation(
        source_index=1,
        chunk_id="c1",
        document_id="doc_test",
        source_file="guide.pdf",
        snippet="def search_hotel(): pass",
    )

    # Scenario: Turn 1 was DIRECT, Turn 2 search returned 0 chunks (MISSING)
    eff_st, merged_chunks, preserved_cits, continuity = guard.enforce_downgrade_protection(
        previous_status=EvidenceStatus.DIRECT,
        previous_chunks=[c1],
        previous_citations=[cit1],
        current_status=EvidenceStatus.MISSING,
        current_chunks=[],
        is_followup=True,
    )

    assert eff_st == EvidenceStatus.DIRECT  # Protected!
    assert len(merged_chunks) == 1
    assert merged_chunks[0].chunk.id == "c1"
    assert len(preserved_cits) == 1
    assert continuity is True

    # Directive generation
    directive = guard.format_monotonic_prompt_directive(
        is_followup=True,
        retained_prior_evidence=True,
        additional_evidence_found=False,
        previous_topic="Hotel Search Agent",
    )
    assert "CONVERSATION CONTINUITY DIRECTIVE" in directive
    assert "DO NOT state that the information cannot be found" in directive
    assert "Hotel Search Agent" in directive


def test_m1_06_grounded_system_prompt_rules_a_to_f():
    """Verify GROUNDED_SYSTEM_PROMPT includes all Rules A-F."""
    assert "RULE A — Conversation Continuity" in GROUNDED_SYSTEM_PROMPT
    assert "RULE B — Evidence Continuity" in GROUNDED_SYSTEM_PROMPT
    assert "RULE C — Expansion" in GROUNDED_SYSTEM_PROMPT
    assert "RULE D — No False Absence" in GROUNDED_SYSTEM_PROMPT
    assert "RULE E — Evidence Distinction" in GROUNDED_SYSTEM_PROMPT
    assert "RULE F — Detailed Code Explanations" in GROUNDED_SYSTEM_PROMPT


def test_m1_07_chat_service_evidence_context_persistence():
    """Verify ChatService records ConversationEvidenceContext across turns."""
    mock_pipeline = MagicMock()
    mock_pipeline.get_active_model.return_value = "qwen2.5:7b"

    chunk1 = _create_mock_scored_chunk("chunk_101", "Hotel Search Agent details", page_number=72)
    cit1 = Citation(
        source_index=1,
        chunk_id="chunk_101",
        document_id="doc_test",
        source_file="guide.pdf",
        snippet="Hotel Search Agent details",
    )

    rag_response = RAGResponse(
        id="resp_01",
        query="What is Hotel Search Agent?",
        answer="Hotel Search Agent searches hotel inventory.",
        citations=[cit1],
        context_chunks=[chunk1],
        trace=RAGTrace(
            query="What is Hotel Search Agent?",
            active_topic="Hotel Search Agent",
            active_entities=["Hotel Search Agent"],
            query_type="factual",
            evidence_status="DIRECT",
            answer_mode="DIRECT",
        ),
        model="qwen2.5:7b",
    )
    mock_pipeline.query.return_value = rag_response

    mock_telemetry = MagicMock()
    mock_telemetry.record_from_rag_response.return_value = MagicMock(
        model_dump=lambda: {"execution_time_ms": 25.0},
        verification=None,
    )

    state_mgr = ConversationStateManager()
    chat_service = ChatService(
        rag_pipeline=mock_pipeline,
        telemetry_service=mock_telemetry,
        state_manager=state_mgr,
    )

    req = ChatRequest(message="What is Hotel Search Agent?", session_id="sess_m1_test")
    resp = chat_service.execute_query(req)

    assert resp.answer == "Hotel Search Agent searches hotel inventory."
    saved_state = state_mgr.get_state("sess_m1_test")
    assert len(saved_state.turns) == 1
    assert len(saved_state.evidence_contexts) == 1
    ev_ctx = saved_state.evidence_contexts[0]
    assert ev_ctx.conversation_id == "sess_m1_test"
    assert ev_ctx.verified_chunk_ids == ["chunk_101"]
    assert ev_ctx.source_pages == [72]
    assert ev_ctx.evidence_status == EvidenceStatus.DIRECT
