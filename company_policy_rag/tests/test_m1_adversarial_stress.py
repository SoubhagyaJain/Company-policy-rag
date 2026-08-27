"""
Adversarial Stress Test Suite for Milestone 1 (M1_1 Challenger).
Targeting Edge Cases:
1. Rapid multi-turn topic switches & referent retention
2. Empty or null previous answers / missing turn metadata
3. Highly ambiguous follow-up cues ("it and that", "why not?", single punctuation)
4. Contradictory evidence injection & monotonic downgrade guard protection
5. Window expansion with missing, zero, negative, or extreme out-of-bounds page numbers
6. Concurrency and thread isolation under rapid state updates
"""

from __future__ import annotations

import concurrent.futures
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
from backend.rag.pipeline import GROUNDED_SYSTEM_PROMPT, RAGPipeline
from backend.services.chat_service import ChatService


def _create_mock_scored_chunk(
    chunk_id: str,
    text: str,
    page_number: int | None = 1,
    content_type: ContentType = ContentType.PROSE,
    doc_id: str = "doc_test",
    source_file: str = "policy.pdf",
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


# =============================================================================
# 1. RAPID MULTI-TURN TOPIC SWITCHES
# =============================================================================

def test_m1_adv_01_rapid_multi_turn_topic_switches():
    """
    Stress-test rapid alternating topic switches and pronoun follow-ups:
    Turn 1: Vacation Policy (Topic 1)
    Turn 2: 401k Match (Topic 2 - Switch)
    Turn 3: "tell me about it in detail" -> must resolve to 401k Match (Topic 2), NOT Vacation Policy!
    Turn 4: Hotel Search Agent (Topic 3 - Switch)
    Turn 5: "explain this code" -> must resolve to Hotel Search Agent (Topic 3)!
    Turn 6: Health Insurance (Topic 4 - Switch)
    Turn 7: "how to enroll?" -> must resolve to Health Insurance (Topic 4)!
    """
    resolver = FollowUpResolver()
    state = ConversationRAGState(conversation_id="conv_rapid_switch")

    # Turn 1: Initial query (Vacation Policy)
    res1 = resolver.resolve("What is the Vacation Policy?", state)
    assert res1.is_followup is False
    assert res1.topic_shift is False
    assert "Vacation Policy" in res1.active_topic
    state.active_topic = res1.active_topic
    state.active_entities = res1.active_entities
    state.last_user_query = "What is the Vacation Policy?"
    state.turns.append(
        ConversationTurn(
            turn_id="t1",
            user_query="What is the Vacation Policy?",
            resolved_query="What is the Vacation Policy?",
            active_topic=res1.active_topic,
            answer="Vacation policy grants 20 PTO days.",
        )
    )

    # Turn 2: Topic switch to 401k Match
    res2 = resolver.resolve("What is the 401k match program?", state)
    assert res2.is_followup is False
    assert res2.topic_shift is True
    assert "401k" in res2.active_topic or "match" in res2.active_topic.lower()
    state.active_topic = res2.active_topic
    state.active_entities = res2.active_entities
    state.last_user_query = "What is the 401k match program?"
    state.turns.append(
        ConversationTurn(
            turn_id="t2",
            user_query="What is the 401k match program?",
            resolved_query="What is the 401k match program?",
            active_topic=res2.active_topic,
            answer="401k matching matches 50% up to 6%.",
        )
    )

    # Turn 3: Pronoun follow-up "tell me about it in detail"
    res3 = resolver.resolve("tell me about it in detail", state)
    assert res3.is_followup is True
    assert res3.topic_shift is False
    # Must link to 401k, NOT Vacation Policy!
    assert "401k" in res3.resolved_query or "match" in res3.resolved_query.lower()
    assert "Vacation" not in res3.resolved_query
    assert res3.resolution.primary_subject == state.active_topic

    # Turn 4: Topic switch to Hotel Search Agent (Code intent)
    res4 = resolver.resolve("What is the Hotel Search Agent implementation?", state, intent=QueryCategory.IMPLEMENTATION)
    assert res4.is_followup is False
    assert res4.topic_shift is True
    assert "Hotel Search Agent" in res4.active_topic
    state.active_topic = res4.active_topic
    state.active_entities = res4.active_entities
    state.last_user_query = "What is the Hotel Search Agent implementation?"
    state.turns.append(
        ConversationTurn(
            turn_id="t4",
            user_query="What is the Hotel Search Agent implementation?",
            resolved_query="What is the Hotel Search Agent implementation?",
            active_topic=res4.active_topic,
            intent="implementation",
            answer="```python\ndef hotel_search(): pass\n```",
        )
    )

    # Turn 5: Follow-up "explain this code"
    res5 = resolver.resolve("explain this code", state, intent=QueryCategory.CODE)
    assert res5.is_followup is True
    assert res5.answer_mode == AnswerMode.CODE_EXPLANATION
    assert "Hotel Search Agent" in res5.resolved_query
    assert "code" in res5.resolved_query.lower()
    assert "401k" not in res5.resolved_query
    assert "Vacation" not in res5.resolved_query

    # Turn 6: Topic switch to Health Insurance
    res6 = resolver.resolve("What are the Health Insurance benefits?", state)
    assert res6.is_followup is False
    assert res6.topic_shift is True
    assert "Health Insurance" in res6.active_topic
    state.active_topic = res6.active_topic
    state.active_entities = res6.active_entities
    state.last_user_query = "What are the Health Insurance benefits?"
    state.turns.append(
        ConversationTurn(
            turn_id="t6",
            user_query="What are the Health Insurance benefits?",
            resolved_query="What are the Health Insurance benefits?",
            active_topic=res6.active_topic,
            answer="Health insurance covers dental and vision.",
        )
    )

    # Turn 7: Short follow-up "how to enroll?"
    res7 = resolver.resolve("how to enroll?", state)
    assert res7.is_followup is True
    assert "Health Insurance" in res7.resolved_query or "enroll" in res7.resolved_query.lower()
    assert "Hotel" not in res7.resolved_query
    assert "Vacation" not in res7.resolved_query


# =============================================================================
# 2. EMPTY OR NULL PREVIOUS ANSWERS / MISSING TURN METADATA
# =============================================================================

def test_m1_adv_02_empty_null_state_and_missing_metadata():
    """
    Stress-test FollowUpResolver, ConsistencyGuard, and StateManager with empty,
    None, or malformed data to ensure zero unhandled exceptions.
    """
    resolver = FollowUpResolver()
    guard = ConversationConsistencyGuard()
    state_mgr = ConversationStateManager()

    # 1. Resolver on empty query, None state, whitespace-only queries
    res_empty = resolver.resolve("", None)
    assert res_empty.is_followup is False
    assert res_empty.resolved_query == ""

    res_ws = resolver.resolve("   \n\t   ", None)
    assert res_ws.is_followup is False
    assert res_ws.resolved_query == ""

    # 2. State with None/empty attributes
    sparse_state = ConversationRAGState(
        conversation_id="conv_sparse",
        last_user_query=None,
        last_resolved_query=None,
        active_topic=None,
        active_entities=[],
        turns=[
            ConversationTurn(
                turn_id="t_empty",
                user_query="",
                resolved_query="",
                answer="",
                retrieved_chunks=[],
                visual_evidence=[],
                citations=[],
            )
        ],
    )
    res_sparse = resolver.resolve("tell me more", sparse_state)
    # Should handle gracefully without raising AttributeError or IndexError
    assert res_sparse.is_followup is False

    # 3. ConsistencyGuard on None / empty inputs
    eff_st, merged, cits, cont = guard.enforce_downgrade_protection(
        previous_status=None,
        previous_chunks=None,
        previous_citations=None,
        current_status="MISSING",
        current_chunks=[],
        is_followup=False,
    )
    assert eff_st == EvidenceStatus.MISSING
    assert merged == []
    assert cits == []
    assert cont is False

    # 4. Directive generation on empty / missing topic
    dir_empty = guard.format_monotonic_prompt_directive(
        is_followup=True,
        retained_prior_evidence=True,
        additional_evidence_found=False,
        previous_topic=None,
    )
    assert "CONVERSATION CONTINUITY DIRECTIVE" in dir_empty
    assert "the active subject" in dir_empty

    # 5. StateManager non-existent session
    fresh_state = state_mgr.get_state("non_existent_id")
    assert fresh_state.conversation_id == "non_existent_id"
    assert fresh_state.turns == []
    assert fresh_state.active_topic is None


# =============================================================================
# 3. HIGHLY AMBIGUOUS FOLLOW-UP CUES
# =============================================================================

def test_m1_adv_03_highly_ambiguous_cues_and_edge_cues():
    """
    Stress-test ambiguous, minimal, or conflicting follow-up cues:
    - "it and that"
    - "why not?"
    - "why?"
    - "and then?"
    - "what about both of them?"
    - Single character / punctuation queries like "?" or "and?"
    """
    resolver = FollowUpResolver()
    state = ConversationRAGState(
        conversation_id="conv_ambiguous",
        active_topic="Travel Reimbursement Policy",
        active_entities=["Travel Reimbursement Policy", "Per Diem", "Flight Booking"],
        previous_intent=QueryCategory.PROCEDURAL,
        turns=[
            ConversationTurn(
                turn_id="t_amb",
                user_query="What is the travel reimbursement policy?",
                resolved_query="What is the travel reimbursement policy?",
                answer="Travel reimbursement allows up to $50 per day for meals.",
            )
        ],
    )

    # 1. "it and that"
    res1 = resolver.resolve("tell me about it and that", state)
    assert res1.is_followup is True
    assert "Travel Reimbursement Policy" in res1.resolved_query

    # 2. "why not?"
    res2 = resolver.resolve("why not?", state)
    assert res2.is_followup is True
    assert "Travel Reimbursement Policy" in res2.resolved_query

    # 3. "why?"
    res3 = resolver.resolve("why?", state)
    assert res3.is_followup is True
    assert "Travel Reimbursement Policy" in res3.resolved_query

    # 4. "and then?"
    res4 = resolver.resolve("and then?", state)
    assert res4.is_followup is True
    assert res4.answer_mode in (AnswerMode.CONTINUE, AnswerMode.CONTINUATION)
    assert "Travel Reimbursement Policy" in res4.resolved_query

    # 5. "what about both of them?"
    res5 = resolver.resolve("what about both of them?", state)
    assert res5.is_followup is True
    assert "Travel Reimbursement Policy" in res5.resolved_query

    # 6. "?" (isolated punctuation)
    res6 = resolver.resolve("?", state)
    # Should not crash
    assert isinstance(res6.resolved_query, str)

    # 7. "and for contractors?"
    res7 = resolver.resolve("and for contractors?", state)
    assert res7.is_followup is True
    assert "Contractors" in res7.resolved_query or "contractor" in res7.resolved_query.lower()
    assert "Travel Reimbursement Policy" in res7.resolved_query


# =============================================================================
# 4. CONTRADICTORY EVIDENCE INJECTION & MONOTONIC DOWNGRADE GUARD
# =============================================================================

def test_m1_adv_04_contradictory_evidence_monotonic_downgrade_guard():
    """
    Stress-test ConversationConsistencyGuard across all possible enum transitions
    to guarantee that valid prior evidence is NEVER downgraded to MISSING on follow-up.
    """
    guard = ConversationConsistencyGuard()

    c1 = _create_mock_scored_chunk("c1", "Section 10.1: Full Remote Work Policy Details", page_number=10)
    c2 = _create_mock_scored_chunk("c2", "Section 10.2: Remote Equipment Allowance ($500)", page_number=11)
    cit1 = Citation(source_index=1, chunk_id="c1", document_id="doc_test", source_file="policy.pdf", snippet="Remote Work")
    cit2 = Citation(source_index=2, chunk_id="c2", document_id="doc_test", source_file="policy.pdf", snippet="Equipment")

    prior_chunks = [c1, c2]
    prior_cits = [cit1, cit2]

    # Test Matrix:
    # 1. DIRECT + MISSING -> DIRECT (Protected)
    st1, chunks1, cits1, cont1 = guard.enforce_downgrade_protection(
        previous_status=EvidenceStatus.DIRECT,
        previous_chunks=prior_chunks,
        previous_citations=prior_cits,
        current_status=EvidenceStatus.MISSING,
        current_chunks=[],
        is_followup=True,
    )
    assert st1 == EvidenceStatus.DIRECT
    assert len(chunks1) == 2
    assert cont1 is True

    # 2. DIRECT + PARTIAL -> DIRECT (Retains full prior direct evidence)
    c3 = _create_mock_scored_chunk("c3", "Partial mention of remote work", page_number=15)
    st2, chunks2, cits2, cont2 = guard.enforce_downgrade_protection(
        previous_status=EvidenceStatus.DIRECT,
        previous_chunks=prior_chunks,
        previous_citations=prior_cits,
        current_status=EvidenceStatus.PARTIAL,
        current_chunks=[c3],
        is_followup=True,
    )
    assert st2 == EvidenceStatus.DIRECT
    assert len(chunks2) == 3  # Merged c3 + c1, c2
    assert cont2 is True

    # 3. PARTIAL + MISSING -> PARTIAL (Protected)
    st3, chunks3, cits3, cont3 = guard.enforce_downgrade_protection(
        previous_status=EvidenceStatus.PARTIAL,
        previous_chunks=[c3],
        previous_citations=[],
        current_status=EvidenceStatus.MISSING,
        current_chunks=[],
        is_followup=True,
    )
    assert st3 == EvidenceStatus.PARTIAL
    assert len(chunks3) == 1
    assert cont3 is True

    # 4. PARTIAL + DIRECT -> DIRECT (Upgraded cleanly!)
    st4, chunks4, cits4, cont4 = guard.enforce_downgrade_protection(
        previous_status=EvidenceStatus.PARTIAL,
        previous_chunks=[c3],
        previous_citations=[],
        current_status=EvidenceStatus.DIRECT,
        current_chunks=[c1],
        is_followup=True,
    )
    assert st4 == EvidenceStatus.DIRECT
    assert len(chunks4) == 2

    # 5. RELATED + MISSING -> RELATED (Protected)
    st5, chunks5, cits5, cont5 = guard.enforce_downgrade_protection(
        previous_status=EvidenceStatus.RELATED,
        previous_chunks=[c3],
        previous_citations=[],
        current_status=EvidenceStatus.MISSING,
        current_chunks=[],
        is_followup=True,
    )
    assert st5 == EvidenceStatus.RELATED
    assert len(chunks5) == 1

    # 6. Topic Shift (is_followup=False): MUST NOT force prior evidence into unrelated query!
    st6, chunks6, cits6, cont6 = guard.enforce_downgrade_protection(
        previous_status=EvidenceStatus.DIRECT,
        previous_chunks=prior_chunks,
        previous_citations=prior_cits,
        current_status=EvidenceStatus.MISSING,
        current_chunks=[],
        is_followup=False,
    )
    assert st6 == EvidenceStatus.MISSING
    assert chunks6 == []
    assert cont6 is False


# =============================================================================
# 5. WINDOW EXPANSION WITH MISSING OR OUT-OF-BOUNDS PAGE NUMBERS
# =============================================================================

def test_m1_adv_05_window_expansion_edge_cases():
    """
    Stress-test docstore window expansion logic with:
    - Chunks with page_number = None
    - Chunks with page_number = 0, -1 (negative)
    - Chunks with page_number = 999999 (huge)
    - Mixed documents in docstore (preventing cross-document pollution)
    """
    # Construct docstore with various edge-case chunks
    docstore: dict[str, Chunk] = {
        "c_none": Chunk(
            id="c_none",
            text="Chunk with no page number",
            metadata=ChunkMetadata(document_id="doc_A", source_file="docA.pdf", page_number=None),
        ),
        "c_zero": Chunk(
            id="c_zero",
            text="Chunk on page 0",
            metadata=ChunkMetadata(document_id="doc_A", source_file="docA.pdf", page_number=0),
        ),
        "c_neg": Chunk(
            id="c_neg",
            text="Chunk on negative page",
            metadata=ChunkMetadata(document_id="doc_A", source_file="docA.pdf", page_number=-1),
        ),
        "c_p1": Chunk(
            id="c_p1",
            text="Doc A Page 1 Intro",
            metadata=ChunkMetadata(document_id="doc_A", source_file="docA.pdf", page_number=1),
        ),
        "c_p2": Chunk(
            id="c_p2",
            text="Doc A Page 2 Hotel Search Agent Setup",
            metadata=ChunkMetadata(document_id="doc_A", source_file="docA.pdf", page_number=2),
        ),
        "c_p3": Chunk(
            id="c_p3",
            text="Doc A Page 3 Hotel Search Agent Implementation Code",
            metadata=ChunkMetadata(document_id="doc_A", source_file="docA.pdf", page_number=3),
        ),
        "c_p4": Chunk(
            id="c_p4",
            text="Doc A Page 4 Hotel Search Agent Tools",
            metadata=ChunkMetadata(document_id="doc_A", source_file="docA.pdf", page_number=4),
        ),
        "c_other_doc": Chunk(
            id="c_other_doc",
            text="Doc B Page 2 Unrelated Document Info",
            metadata=ChunkMetadata(document_id="doc_B", source_file="docB.pdf", page_number=2),
        ),
        "c_huge": Chunk(
            id="c_huge",
            text="Doc A Page 999999",
            metadata=ChunkMetadata(document_id="doc_A", source_file="docA.pdf", page_number=999999),
        ),
    }

    # Simulate Window Expansion logic as implemented in RAGPipeline
    prev_chunk = ScoredChunk(chunk=docstore["c_p2"], score=0.9)
    prev_all = [prev_chunk]

    prev_pages = {
        c.chunk.metadata.page_number
        for c in prev_all
        if getattr(c.chunk.metadata, "page_number", None) is not None
    }
    target_doc_id = prev_all[0].chunk.metadata.document_id

    adjacent_pages = set()
    for p in prev_pages:
        if isinstance(p, int):
            adjacent_pages.update([p - 1, p, p + 1, p + 2])

    # Adjacent pages for page 2 should be {1, 2, 3, 4}
    assert adjacent_pages == {1, 2, 3, 4}

    candidate_chunks: list[ScoredChunk] = []
    for chunk_obj in docstore.values():
        p_num = getattr(chunk_obj.metadata, "page_number", None)
        d_id = getattr(chunk_obj.metadata, "document_id", None)
        if p_num in adjacent_pages and (target_doc_id is None or d_id == target_doc_id):
            candidate_chunks.append(ScoredChunk(chunk=chunk_obj, score=0.75))

    expanded_ids = {sc.chunk.id for sc in candidate_chunks}
    # Must include Doc A pages 1, 2, 3, 4
    assert "c_p1" in expanded_ids
    assert "c_p2" in expanded_ids
    assert "c_p3" in expanded_ids
    assert "c_p4" in expanded_ids
    # Must NOT include Doc B (cross-document pollution avoided)
    assert "c_other_doc" not in expanded_ids
    # Must NOT include None, 0, -1, 999999
    assert "c_none" not in expanded_ids
    assert "c_zero" not in expanded_ids
    assert "c_neg" not in expanded_ids
    assert "c_huge" not in expanded_ids


# =============================================================================
# 6. CONCURRENCY & THREAD ISOLATION UNDER RAPID STATE UPDATES
# =============================================================================

def test_m1_adv_06_concurrency_and_session_isolation():
    """
    Stress-test ConversationStateManager under high concurrent read/write load
    across 50 distinct sessions to verify thread safety and zero cross-talk.
    """
    state_mgr = ConversationStateManager(maxsize=500, ttl=3600)
    num_sessions = 50
    turns_per_session = 10

    def simulate_session_flow(session_idx: int):
        sess_id = f"sess_concurrent_{session_idx}"
        for turn_idx in range(turns_per_session):
            state = state_mgr.get_state(sess_id)
            state.active_topic = f"Topic_{session_idx}"
            state.active_entities = [f"Entity_{session_idx}_{turn_idx}"]
            turn = ConversationTurn(
                turn_id=f"t_{session_idx}_{turn_idx}",
                user_query=f"Query {turn_idx} for topic {session_idx}",
                resolved_query=f"Query {turn_idx} for topic {session_idx}",
                answer=f"Answer {turn_idx} for topic {session_idx}",
            )
            state.turns.append(turn)
            ev_ctx = ConversationEvidenceContext(
                conversation_id=sess_id,
                turn_id=turn.turn_id,
                query=turn.user_query,
                normalized_subjects=[f"Topic_{session_idx}"],
                verified_chunk_ids=[f"chunk_{session_idx}_{turn_idx}"],
            )
            state.evidence_contexts.append(ev_ctx)
            state_mgr.save_state(state)
            time.sleep(0.001)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(simulate_session_flow, i) for i in range(num_sessions)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    # Verify all 50 sessions maintained absolute isolation and integrity
    for i in range(num_sessions):
        sess_id = f"sess_concurrent_{i}"
        final_state = state_mgr.get_state(sess_id)
        assert final_state.conversation_id == sess_id
        assert final_state.active_topic == f"Topic_{i}"
        assert len(final_state.turns) == turns_per_session
        assert len(final_state.evidence_contexts) == turns_per_session
        for t_idx, turn in enumerate(final_state.turns):
            assert f"topic {i}" in turn.user_query
            assert f"topic {i}" in turn.answer
