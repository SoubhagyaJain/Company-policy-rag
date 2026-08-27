"""
Adversarial Verification Suite for Milestone 1:
Generic Follow-Up Resolution, Non-Shrinking Expansion Policy, and Session Isolation.

Tests:
1. Adversarial Follow-Up Variations & Intent/Mode Mapping
2. Pronoun Referential Resolution Matrix
3. Topic Shifts & Independent Query Handling
4. Non-Shrinking Expansion Plan Guarantees
5. Monotonic Evidence Consistency & Downgrade Protection Matrix
6. Session Isolation & Multi-Thread Concurrency (ChatService & ConversationStateManager)
7. Multi-Turn Evidence Accumulation & Streaming Context Persistence
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
from unittest.mock import AsyncMock, MagicMock
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
from backend.rag.conversation_resolver import FollowUpResolver
from backend.services.chat_service import ChatService


def _make_scored_chunk(
    chunk_id: str,
    text: str,
    page_number: int = 1,
    doc_id: str = "doc_hotel",
    source_file: str = "hotel_agent_guide.pdf",
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
            section_title="Hotel Search Agent Architecture",
            content_type=ContentType.CODE if "def " in text or "class " in text else ContentType.PROSE,
            image_assets=image_assets or [],
        ),
    )
    return ScoredChunk(chunk=chunk, score=0.92)


# =============================================================================
# 1. Adversarial Follow-Up Variations & Mode Mapping
# =============================================================================

@pytest.mark.parametrize(
    "followup_query,expected_mode_family,should_have_code_explanation",
    [
        ("tell me more", [AnswerMode.EXPAND, AnswerMode.DETAILED], False),
        ("tell me about it in detail", [AnswerMode.EXPAND, AnswerMode.DETAILED], False),
        ("explain more", [AnswerMode.EXPAND, AnswerMode.DETAILED], False),
        ("explain further", [AnswerMode.EXPAND, AnswerMode.DETAILED], False),
        ("go deeper", [AnswerMode.EXPAND, AnswerMode.DETAILED], False),
        ("elaborate on that", [AnswerMode.EXPAND, AnswerMode.DETAILED], False),
        ("can you expand on that?", [AnswerMode.EXPAND, AnswerMode.DETAILED], False),
        ("explain the code in detail", [AnswerMode.CODE_EXPLANATION, AnswerMode.EXPAND], True),
        ("explain this code", [AnswerMode.CODE_EXPLANATION], True),
        ("show the code", [AnswerMode.CODE_EXPLANATION], True),
        ("break down the code", [AnswerMode.CODE_EXPLANATION], True),
        ("break it down step by step", [AnswerMode.STEP_BY_STEP], False),
        ("walk me through it step by step", [AnswerMode.STEP_BY_STEP], False),
        ("show diagram flow", [AnswerMode.EXPLANATION, AnswerMode.DIRECT], False),
        ("explain the diagram", [AnswerMode.EXPLANATION], False),
        ("tell me more about the diagram", [AnswerMode.EXPLANATION, AnswerMode.EXPAND], False),
        ("compare them", [AnswerMode.COMPARISON], False),
        ("compare these", [AnswerMode.COMPARISON], False),
        ("continue", [AnswerMode.CONTINUE, AnswerMode.CONTINUATION], False),
        ("and then?", [AnswerMode.CONTINUE, AnswerMode.CONTINUATION], False),
        ("how does it work?", [AnswerMode.EXPLANATION, AnswerMode.DIRECT], False),
        ("what does that mean?", [AnswerMode.EXPLANATION, AnswerMode.DIRECT], False),
        ("and for contractors?", [AnswerMode.DIRECT, AnswerMode.EXPAND], False),
        ("how to enroll?", [AnswerMode.DIRECT, AnswerMode.STEP_BY_STEP], False),
    ],
)
def test_adversarial_followup_variations_matrix(
    followup_query: str,
    expected_mode_family: list[AnswerMode],
    should_have_code_explanation: bool,
):
    resolver = FollowUpResolver()
    state = ConversationRAGState(
        conversation_id="conv_adv_01",
        active_topic="Hotel Search Agent",
        active_entities=["Hotel Search Agent", "CrewAI", "convert_currency"],
        previous_intent=QueryCategory.IMPLEMENTATION,
        turns=[
            ConversationTurn(
                turn_id="turn_h01",
                user_query="What is the implementation code for Hotel Search Agent?",
                resolved_query="What is the implementation code for Hotel Search Agent?",
                intent="implementation",
                answer="Here is the Hotel Search Agent: def search_hotels(city): pass",
            )
        ],
    )

    result = resolver.resolve(followup_query, state, intent=QueryCategory.IMPLEMENTATION)

    assert result.is_followup is True, f"Failed follow-up detection for '{followup_query}'"
    assert result.topic_shift is False
    assert result.active_topic == "Hotel Search Agent"
    assert result.resolution is not None
    assert result.resolution.is_follow_up is True
    assert result.resolution.primary_subject == "Hotel Search Agent"
    assert result.resolution.referenced_answer_id == "turn_h01"
    assert result.resolution.preserve_previous_evidence is True

    # Check mode alignment
    assert result.answer_mode in expected_mode_family or result.resolution.answer_mode in expected_mode_family, (
        f"Query '{followup_query}' resolved to mode '{result.answer_mode}', expected one of {expected_mode_family}"
    )

    # Check non-shrinking expansion plan
    assert result.expansion_plan is not None
    assert result.expansion_plan.preserve_prior_facts is True
    if result.resolution.expansion_requested:
        assert result.expansion_plan.target_detail_level == "detailed"
        assert result.expansion_plan.retrieve_additional_context is True
        assert result.expansion_plan.inspect_adjacent_evidence is True

    if should_have_code_explanation:
        assert result.expansion_plan.explain_code_line_by_line is True

    # Verify query rewrite does not retain isolated ambiguous pronouns
    assert "Hotel Search Agent" in result.resolved_query


# =============================================================================
# 2. Pronoun Referential Resolution Matrix
# =============================================================================

@pytest.mark.parametrize(
    "pronoun_query,expected_topic_present",
    [
        ("How is it configured?", True),
        ("What are the parameters for this?", True),
        ("Show me the inputs for that", True),
        ("Explain these in detail", True),
        ("How do they interact?", True),
        ("What about the above?", True),
    ],
)
def test_pronoun_referential_resolution(pronoun_query: str, expected_topic_present: bool):
    resolver = FollowUpResolver()
    state = ConversationRAGState(
        conversation_id="conv_pronouns",
        active_topic="Vacation Leave Accrual Policy",
        active_entities=["Vacation Leave Accrual Policy", "PTO"],
        previous_intent=QueryCategory.POLICY,
    )

    result = resolver.resolve(pronoun_query, state, intent=QueryCategory.POLICY)
    assert result.is_followup is True
    if expected_topic_present:
        assert "Vacation Leave Accrual Policy" in result.resolved_query


# =============================================================================
# 3. Topic Shift and Independence
# =============================================================================

@pytest.mark.parametrize(
    "topic_shift_query,expected_new_topic",
    [
        ("What is the company Maternity Leave policy?", "Maternity Leave policy"),
        ("Explain the 401k Retirement Plan options", "401k Retirement Plan options"),
        ("Define Code of Conduct guidelines", "Code of Conduct guidelines"),
        ("List all dental insurance providers", "dental insurance providers"),
    ],
)
def test_topic_shift_detection(topic_shift_query: str, expected_new_topic: str):
    resolver = FollowUpResolver()
    state = ConversationRAGState(
        conversation_id="conv_topic_shift",
        active_topic="Hotel Search Agent",
        active_entities=["Hotel Search Agent", "CrewAI"],
        previous_intent=QueryCategory.IMPLEMENTATION,
    )

    result = resolver.resolve(topic_shift_query, state, intent=QueryCategory.POLICY)
    assert result.is_followup is False
    assert result.topic_shift is True
    assert result.resolution.preserve_previous_evidence is False
    assert expected_new_topic.lower() in result.active_topic.lower()


# =============================================================================
# 4. Non-Shrinking Expansion Plan Guarantees
# =============================================================================

def test_expansion_plan_all_modes():
    resolver = FollowUpResolver()

    # Expanded modes
    for mode in [
        AnswerMode.EXPAND,
        AnswerMode.DETAILED,
        AnswerMode.CODE_EXPLANATION,
        AnswerMode.STEP_BY_STEP,
        AnswerMode.EXPLANATION,
    ]:
        plan = resolver.create_expansion_plan(mode, has_code=True)
        assert plan.preserve_prior_facts is True
        assert plan.retrieve_additional_context is True
        assert plan.inspect_adjacent_evidence is True
        assert plan.explain_components is True
        assert plan.explain_execution_flow is True
        assert plan.target_detail_level == "detailed"
        assert plan.restate_subject == "minimal"

    # Direct mode
    plan_direct = resolver.create_expansion_plan(AnswerMode.DIRECT, has_code=False)
    assert plan_direct.preserve_prior_facts is True
    assert plan_direct.retrieve_additional_context is False
    assert plan_direct.target_detail_level == "standard"


# =============================================================================
# 5. Monotonic Evidence Consistency & Downgrade Protection Matrix
# =============================================================================

def test_consistency_guard_full_state_transition_matrix():
    guard = ConversationConsistencyGuard()

    c1 = _make_scored_chunk("c1", "Hotel Search Agent code definition", page_number=72)
    c2 = _make_scored_chunk("c2", "Hotel Search Agent tools and parameters", page_number=73)
    c3 = _make_scored_chunk("c3", "Hotel Search Agent kickoff workflow", page_number=74)

    cit1 = Citation(source_index=1, chunk_id="c1", document_id="doc_hotel", source_file="hotel.pdf", snippet="def search()")
    cit2 = Citation(source_index=2, chunk_id="c2", document_id="doc_hotel", source_file="hotel.pdf", snippet="tools = []")

    # 1. DIRECT + MISSING -> DIRECT (Preserve previous evidence)
    eff_st, merged, cits, cont = guard.enforce_downgrade_protection(
        previous_status=EvidenceStatus.DIRECT,
        previous_chunks=[c1, c2],
        previous_citations=[cit1, cit2],
        current_status=EvidenceStatus.MISSING,
        current_chunks=[],
        is_followup=True,
    )
    assert eff_st == EvidenceStatus.DIRECT
    assert len(merged) == 2
    assert {c.chunk.id for c in merged} == {"c1", "c2"}
    assert len(cits) == 2
    assert cont is True

    # 2. DIRECT + PARTIAL -> DIRECT
    eff_st, merged, cits, cont = guard.enforce_downgrade_protection(
        previous_status=EvidenceStatus.DIRECT,
        previous_chunks=[c1],
        previous_citations=[cit1],
        current_status=EvidenceStatus.PARTIAL,
        current_chunks=[c3],
        is_followup=True,
    )
    assert eff_st == EvidenceStatus.DIRECT
    assert len(merged) == 2
    assert {c.chunk.id for c in merged} == {"c1", "c3"}

    # 3. PARTIAL + MISSING -> PARTIAL
    eff_st, merged, cits, cont = guard.enforce_downgrade_protection(
        previous_status=EvidenceStatus.PARTIAL,
        previous_chunks=[c1],
        previous_citations=[cit1],
        current_status=EvidenceStatus.MISSING,
        current_chunks=[],
        is_followup=True,
    )
    assert eff_st == EvidenceStatus.PARTIAL
    assert len(merged) == 1

    # 4. PARTIAL + DIRECT -> DIRECT (Upgrade!)
    eff_st, merged, cits, cont = guard.enforce_downgrade_protection(
        previous_status=EvidenceStatus.PARTIAL,
        previous_chunks=[c1],
        previous_citations=[cit1],
        current_status=EvidenceStatus.DIRECT,
        current_chunks=[c2],
        is_followup=True,
    )
    assert eff_st == EvidenceStatus.DIRECT
    assert len(merged) == 2

    # 5. RELATED + MISSING -> RELATED
    eff_st, merged, cits, cont = guard.enforce_downgrade_protection(
        previous_status=EvidenceStatus.RELATED,
        previous_chunks=[c1],
        previous_citations=[cit1],
        current_status=EvidenceStatus.MISSING,
        current_chunks=[],
        is_followup=True,
    )
    assert eff_st == EvidenceStatus.RELATED
    assert len(merged) == 1

    # 6. Deduplication check: duplicate chunk IDs between turns
    eff_st, merged, cits, cont = guard.enforce_downgrade_protection(
        previous_status=EvidenceStatus.DIRECT,
        previous_chunks=[c1, c2],
        previous_citations=[cit1, cit2],
        current_status=EvidenceStatus.DIRECT,
        current_chunks=[c2, c3],
        is_followup=True,
    )
    assert eff_st == EvidenceStatus.DIRECT
    assert len(merged) == 3
    assert {c.chunk.id for c in merged} == {"c1", "c2", "c3"}

    # 7. Directive verification
    directive_no_new = guard.format_monotonic_prompt_directive(
        is_followup=True,
        retained_prior_evidence=True,
        additional_evidence_found=False,
        previous_topic="Hotel Search Agent",
    )
    assert "DO NOT state that the information cannot be found" in directive_no_new

    directive_with_new = guard.format_monotonic_prompt_directive(
        is_followup=True,
        retained_prior_evidence=True,
        additional_evidence_found=True,
        previous_topic="Hotel Search Agent",
    )
    assert "Integrate previously verified facts with newly retrieved evidence" in directive_with_new


# =============================================================================
# 6. Session Isolation & Concurrency (ConversationStateManager & ChatService)
# =============================================================================

def test_session_isolation_between_distinct_conversations():
    """Verify zero cross-talk or evidence leakage between distinct conversation sessions."""
    state_mgr = ConversationStateManager()
    mock_pipeline = MagicMock()
    mock_pipeline.get_active_model.return_value = "qwen2.5:7b"
    mock_telemetry = MagicMock()
    mock_telemetry.record_from_rag_response.return_value = MagicMock(model_dump=lambda: {})

    chat_service = ChatService(
        rag_pipeline=mock_pipeline,
        telemetry_service=mock_telemetry,
        state_manager=state_mgr,
    )

    # Session Alpha: Hotel Search Agent
    c_alpha = _make_scored_chunk("chunk_alpha", "Hotel Search Agent code", page_number=72)
    cit_alpha = Citation(source_index=1, chunk_id="chunk_alpha", document_id="doc_alpha", source_file="hotel.pdf", snippet="code")
    res_alpha = RAGResponse(
        id="resp_alpha_1",
        query="What is the Hotel Search Agent?",
        answer="Hotel Search Agent searches hotels.",
        citations=[cit_alpha],
        context_chunks=[c_alpha],
        trace=RAGTrace(
            query="What is the Hotel Search Agent?",
            active_topic="Hotel Search Agent",
            active_entities=["Hotel Search Agent"],
            evidence_status="DIRECT",
            answer_mode="DIRECT",
        ),
    )

    # Session Beta: Maternity Leave Policy
    c_beta = _make_scored_chunk("chunk_beta", "Maternity Leave is 16 weeks paid", page_number=12, doc_id="doc_beta", source_file="hr.pdf")
    cit_beta = Citation(source_index=1, chunk_id="chunk_beta", document_id="doc_beta", source_file="hr.pdf", snippet="16 weeks")
    res_beta = RAGResponse(
        id="resp_beta_1",
        query="What is the Maternity Leave policy?",
        answer="Maternity Leave is 16 weeks.",
        citations=[cit_beta],
        context_chunks=[c_beta],
        trace=RAGTrace(
            query="What is the Maternity Leave policy?",
            active_topic="Maternity Leave policy",
            active_entities=["Maternity Leave policy"],
            evidence_status="DIRECT",
            answer_mode="DIRECT",
        ),
    )

    # Execute Turn 1 for both sessions
    mock_pipeline.query.return_value = res_alpha
    chat_service.execute_query(ChatRequest(message="What is the Hotel Search Agent?", session_id="sess_user_alpha"))

    mock_pipeline.query.return_value = res_beta
    chat_service.execute_query(ChatRequest(message="What is the Maternity Leave policy?", session_id="sess_user_beta"))

    state_alpha = state_mgr.get_state("sess_user_alpha")
    state_beta = state_mgr.get_state("sess_user_beta")

    assert state_alpha.active_topic == "Hotel Search Agent"
    assert state_beta.active_topic == "Maternity Leave policy"
    assert len(state_alpha.turns) == 1
    assert len(state_beta.turns) == 1
    assert state_alpha.evidence_contexts[0].verified_chunk_ids == ["chunk_alpha"]
    assert state_beta.evidence_contexts[0].verified_chunk_ids == ["chunk_beta"]

    # Now verify follow-up resolution isolation
    resolver = FollowUpResolver()
    res_followup_alpha = resolver.resolve("tell me about it in detail", state_alpha)
    res_followup_beta = resolver.resolve("tell me about it in detail", state_beta)

    assert res_followup_alpha.active_topic == "Hotel Search Agent"
    assert "Hotel Search Agent" in res_followup_alpha.resolved_query
    assert "Maternity" not in res_followup_alpha.resolved_query

    assert res_followup_beta.active_topic == "Maternity Leave policy"
    assert "Maternity" in res_followup_beta.resolved_query
    assert "Hotel" not in res_followup_beta.resolved_query

    # Test session deletion isolation
    chat_service.delete_session("sess_user_alpha")
    assert not state_mgr.exists("sess_user_alpha")
    assert state_mgr.exists("sess_user_beta")
    assert state_mgr.get_state("sess_user_beta").active_topic == "Maternity Leave policy"


def test_concurrent_state_manager_access():
    """Verify thread-safety and zero data corruption under concurrent access."""
    state_mgr = ConversationStateManager()

    def worker(worker_id: int):
        sess_id = f"sess_worker_{worker_id}"
        for step in range(10):
            st = state_mgr.get_state(sess_id)
            st.active_topic = f"Topic_{worker_id}_{step}"
            st.turns.append(
                ConversationTurn(
                    turn_id=f"turn_{worker_id}_{step}",
                    user_query=f"Query {step}",
                    resolved_query=f"Query {step}",
                )
            )
            state_mgr.save_state(st)
            time.sleep(0.001)
        final_st = state_mgr.get_state(sess_id)
        assert len(final_st.turns) == 10
        assert final_st.active_topic == f"Topic_{worker_id}_9"

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(8)]
        for f in concurrent.futures.as_completed(futures):
            f.result()


# =============================================================================
# 7. Multi-Turn Evidence Accumulation & Streaming Context Persistence
# =============================================================================

@pytest.mark.asyncio
async def test_streaming_multi_turn_evidence_persistence():
    """Verify stream_query records turns and evidence contexts consistently."""
    state_mgr = ConversationStateManager()
    mock_pipeline = MagicMock()
    mock_pipeline.get_active_model.return_value = "qwen2.5:7b"
    mock_telemetry = MagicMock()
    mock_telemetry.record_from_rag_response.return_value = MagicMock(model_dump=lambda: {}, verification=None)

    chunk_1 = _make_scored_chunk("chunk_s1", "Search agent definition", page_number=72)
    cit_1 = Citation(source_index=1, chunk_id="chunk_s1", document_id="doc_hotel", source_file="hotel.pdf", snippet="code")

    async def mock_stream_turn_1(*args, **kwargs):
        yield {"type": "retrieval_done", "candidate_count": 1, "context_count": 1}
        yield {"type": "token", "content": "Hotel Search "}
        yield {"type": "token", "content": "Agent code."}
        yield {
            "type": "done",
            "citations": [cit_1],
            "context_chunks": [chunk_1],
            "trace": RAGTrace(
                query="What is Hotel Search Agent?",
                active_topic="Hotel Search Agent",
                active_entities=["Hotel Search Agent"],
                evidence_status="DIRECT",
                answer_mode="DIRECT",
            ),
        }

    mock_pipeline.stream_query = mock_stream_turn_1

    chat_service = ChatService(
        rag_pipeline=mock_pipeline,
        telemetry_service=mock_telemetry,
        state_manager=state_mgr,
    )

    # Run Stream Turn 1
    events_t1 = []
    async for ev in chat_service.stream_query(ChatRequest(message="What is Hotel Search Agent?", session_id="sess_stream_adv")):
        events_t1.append(ev)

    saved_state = state_mgr.get_state("sess_stream_adv")
    assert len(saved_state.turns) == 1
    assert len(saved_state.evidence_contexts) == 1
    assert saved_state.active_topic == "Hotel Search Agent"
    assert saved_state.evidence_contexts[0].verified_chunk_ids == ["chunk_s1"]

    # Stream Turn 2: Follow-up
    chunk_2 = _make_scored_chunk("chunk_s2", "Search agent execution flow and parameters", page_number=73)
    cit_2 = Citation(source_index=2, chunk_id="chunk_s2", document_id="doc_hotel", source_file="hotel.pdf", snippet="params")

    async def mock_stream_turn_2(*args, **kwargs):
        yield {"type": "retrieval_done", "candidate_count": 2, "context_count": 2}
        yield {"type": "token", "content": "Detailed breakdown "}
        yield {"type": "token", "content": "of the code."}
        yield {
            "type": "done",
            "citations": [cit_1, cit_2],
            "context_chunks": [chunk_1, chunk_2],
            "trace": RAGTrace(
                query="tell me about it in detail",
                rewritten_query="Provide a detailed explanation of Hotel Search Agent implementation",
                is_followup=True,
                active_topic="Hotel Search Agent",
                active_entities=["Hotel Search Agent"],
                evidence_status="DIRECT",
                answer_mode="DETAILED",
            ),
        }

    mock_pipeline.stream_query = mock_stream_turn_2

    events_t2 = []
    async for ev in chat_service.stream_query(ChatRequest(message="tell me about it in detail", session_id="sess_stream_adv")):
        events_t2.append(ev)

    saved_state_t2 = state_mgr.get_state("sess_stream_adv")
    assert len(saved_state_t2.turns) == 2
    assert len(saved_state_t2.evidence_contexts) == 2
    assert saved_state_t2.evidence_contexts[1].verified_chunk_ids == ["chunk_s1", "chunk_s2"]
    assert saved_state_t2.evidence_contexts[1].source_pages == [72, 73]
    assert saved_state_t2.last_user_query == "tell me about it in detail"
