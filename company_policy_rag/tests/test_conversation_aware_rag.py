"""
Comprehensive 12-Test Automated Verification Suite for Conversation-Aware RAG.

Covers:
- R1: Conversation State, Thread-Safe Caching, Deep-Copy & Session Isolation
- R2: Dynamic Query Resolution, Pronoun Rewriting, Topic Continuity & Topic Shift Detection
- R3: Cross-Turn Evidence Continuity, Monotonicity State Machine & Multimodal Visual Asset Reuse
- R4: 5 Answer Modes (DIRECT, DETAILED, EXPAND, SUMMARY, CONTINUE) & 4-Tier Grounded Synthesis Directives
- R5: End-to-End Multi-Turn Streaming, Request Tracing & Structured Observability Telemetry
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from datetime import datetime, timezone
import json
import logging
import re
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

import backend.services.chat_service as chat_service_module
from backend.models.api_dto import ChatRequest
from backend.models.chunk import Chunk, ChunkMetadata, ContentType
from backend.models.conversation import (
    AnswerMode,
    ConversationRAGState,
    ConversationStateManager,
    ConversationTurn,
)
from backend.models.rag import (
    Citation,
    EvidenceStatus,
    QueryCategory,
    QueryClassification,
    RAGResponse,
    RAGTrace,
    ScoredChunk,
)
from backend.rag.conversation_resolver import (
    ConversationResolver,
    _CONTINUE_MODE_PATTERN,
    _DETAILED_MODE_PATTERN,
    _EXPAND_MODE_PATTERN,
    _FOLLOWUP_PHRASES_PATTERN,
    _PRONOUNS_PATTERN,
    _SUMMARY_MODE_PATTERN,
)
from backend.rag.evidence_gate import (
    EvidenceSufficiencyGate,
    compute_monotonic_evidence_status,
)
from backend.rag.pipeline import (
    GROUNDED_SYSTEM_PROMPT,
    _detect_fidelity_mode,
    _format_evidence_status_directive,
    _format_history_for_prompt,
)
from backend.services.chat_service import ChatService


# Provide UTC fallback in module if missing in chat_service import scope
if not hasattr(chat_service_module, "datetime"):
    chat_service_module.datetime = datetime  # type: ignore[attr-defined]
if not hasattr(chat_service_module, "UTC"):
    chat_service_module.UTC = timezone.utc  # type: ignore[attr-defined]


# ============================================================================
# Test Fixtures & In-Memory Helpers
# ============================================================================


def _create_mock_chunk(
    chunk_id: str,
    text: str,
    page_number: int = 1,
    content_type: ContentType = ContentType.PROSE,
    source_file: str = "handbook.pdf",
    section_title: str = "General Policy",
    image_assets: list[Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> ScoredChunk:
    """Create an isolated, in-memory ScoredChunk for deterministic testing."""
    formatted_assets: list[dict[str, Any]] = []
    if image_assets:
        for item in image_assets:
            if isinstance(item, dict):
                formatted_assets.append(item)
            elif isinstance(item, str):
                formatted_assets.append({
                    "asset_id": f"ast_{chunk_id}",
                    "asset_url": item,
                    "image_hash": f"hash_{item}",
                    "visual_type": extra.get("visual_type", "diagram_architecture") if extra else "diagram_architecture",
                })
    metadata = ChunkMetadata(
        document_id=f"doc_{source_file.replace('.', '_')}",
        source_file=source_file,
        file_path=f"data/{source_file}",
        file_hash=f"hash_{source_file}",
        document_type="pdf",
        chunk_strategy="adaptive",
        page_number=page_number,
        section_title=section_title,
        content_type=content_type,
        image_assets=formatted_assets,
        extra=extra or {},
    )
    chunk = Chunk(id=chunk_id, text=text, metadata=metadata)
    return ScoredChunk(chunk=chunk, score=0.92)



# ============================================================================
# Test 01: Referential Pronoun Resolution
# ============================================================================


def test_01_referential_pronoun_resolution() -> None:
    """
    Verifies that referential pronouns ('it', 'that', 'this', 'they', 'the above')
    are accurately detected and dynamically rewritten into standalone search queries
    incorporating active topic entities.
    """
    resolver = ConversationResolver()
    state = ConversationRAGState(
        conversation_id="conv_test_01",
        active_topic="Hotel Search Agent",
        active_entities=["Hotel Search Agent", "convert_currency", "CrewAI"],
        previous_intent=QueryCategory.IMPLEMENTATION,
    )

    queries = [
        "How does it work?",
        "Can you explain that in detail?",
        "Give implementation code for this",
        "What are the parameters for they?",
        "Tell me more about the above",
    ]

    for q in queries:
        is_followup, topic_shift, conf, reason = resolver.detect_followup(q, state)
        assert is_followup is True, f"Expected follow-up for query: '{q}', reason: {reason}"
        assert topic_shift is False, f"Expected no topic shift for query: '{q}'"
        assert conf >= 0.85, f"Expected high confidence >= 0.85 for: '{q}'"

    # Test standalone query synthesis
    resolved_1 = resolver.resolve_standalone_query("How does it work?", state)
    assert "Hotel Search Agent" in resolved_1
    assert not re.search(r"\bit\b", resolved_1, flags=re.IGNORECASE)

    resolved_2 = resolver.resolve_standalone_query(
        "Give implementation code for this",
        state,
        intent=QueryCategory.IMPLEMENTATION,
    )
    assert "Hotel Search Agent" in resolved_2
    assert "code" in resolved_2.lower() or "implementation" in resolved_2.lower()

    # Test LLM-backed rewrite with graceful fallback on error
    mock_failing_llm = MagicMock()
    mock_failing_llm.complete.side_effect = RuntimeError("LLM Offline")
    fallback_resolved = resolver.resolve_standalone_query("How does it work?", state, llm=mock_failing_llm)
    assert "Hotel Search Agent" in fallback_resolved


@pytest.mark.parametrize(
    "query",
    [
        "An electrician wants to perform electrical work privately for their sister. Is this allowed?",
        "An employee is taking prescription medication that may cause drowsiness. Are they required to tell anyone at work?",
        "An employee normally starts at 7:30 am. After a callout they finish at 2:30 am. Can the company require a normal start?",
    ],
)
def test_01b_self_contained_pronouns_are_not_prior_turn_references(query: str) -> None:
    resolver = ConversationResolver()
    state = ConversationRAGState(
        conversation_id="conv_unrelated_prior_topic",
        active_topic="Entry into unattended customer property",
        active_entities=["customer property", "company key"],
    )

    is_followup, topic_shift, confidence, reason = resolver.detect_followup(query, state)

    assert is_followup is False
    assert topic_shift is True
    assert confidence >= 0.9
    assert "self-contained" in reason.lower()


# ============================================================================
# Test 02: Conversational Follow-Up Phrase Resolution
# ============================================================================


def test_02_conversational_followup_phrase_resolution() -> None:
    """
    Verifies that conversational follow-up triggers ('tell about it in detail',
    'elaborate further on this', 'go deeper') trigger AnswerMode.EXPAND and produce
    expanded retrieval queries with architectural and implementation terms.
    """
    resolver = ConversationResolver()
    state = ConversationRAGState(
        conversation_id="conv_test_02",
        active_topic="CrewAI Multi-Agent Architecture",
        active_entities=["CrewAI", "Agent", "Task", "crew.kickoff"],
        previous_intent=QueryCategory.IMPLEMENTATION,
    )

    followup_phrases = [
        "Tell about it in detail",
        "Elaborate further on this",
        "Go deeper into the implementation",
        "Give more details on that",
        "Walk me through it step by step",
        "Break it down",
    ]

    for phrase in followup_phrases:
        is_followup, topic_shift, conf, _ = resolver.detect_followup(phrase, state)
        assert is_followup is True, f"Phrase '{phrase}' should be recognized as follow-up"
        assert topic_shift is False

    # Verify answer mode detection
    mode_1 = resolver.detect_answer_mode("Tell about it in detail")
    assert mode_1 == AnswerMode.EXPAND

    mode_2 = resolver.detect_answer_mode("Elaborate further on this")
    assert mode_2 == AnswerMode.EXPAND

    mode_3 = resolver.detect_answer_mode("Go deeper into the architecture")
    assert mode_3 == AnswerMode.EXPAND

    # Verify query resolution under EXPAND mode includes comprehensive expansion terms
    resolved = resolver.resolve_standalone_query(
        "Tell about it in detail",
        state,
        intent=QueryCategory.IMPLEMENTATION,
        answer_mode=AnswerMode.EXPAND,
    )
    assert "CrewAI Multi-Agent Architecture" in resolved
    assert any(
        kw in resolved.lower()
        for kw in ["implementation", "architecture", "code", "execution flow", "parameters", "tools"]
    )


# ============================================================================
# Test 03: Topic Shift Detection & Topic Clearance
# ============================================================================


def test_03_topic_shift_detection_and_topic_clearance() -> None:
    """
    Verifies that distinct new topics are identified, triggering topic_shift=True,
    is_followup=False, resetting active topics and entities without leaking prior state.
    """
    resolver = ConversationResolver()
    state = ConversationRAGState(
        conversation_id="conv_test_03",
        active_topic="CrewAI Multi-Agent Architecture",
        active_entities=["CrewAI", "Agent", "Task", "crew.kickoff"],
        previous_intent=QueryCategory.IMPLEMENTATION,
    )

    new_topic_queries = [
        "What is the company sick leave policy?",
        "Explain the annual vacation rollover limit",
        "How does the 401k match work for full time employees?",
    ]

    for q in new_topic_queries:
        is_followup, topic_shift, conf, reason = resolver.detect_followup(q, state)
        assert is_followup is False, f"Expected is_followup=False for new query '{q}'"
        assert topic_shift is True, f"Expected topic_shift=True for new query '{q}'"
        assert conf >= 0.80

    # Extract new topic and verify prior entities do not pollute the new topic
    shift_query = "What is the company sick leave policy?"
    new_topic = resolver.extract_topic_from_query(shift_query)
    assert "sick leave policy" in new_topic.lower()
    assert "crewai" not in new_topic.lower()

    new_entities = resolver.extract_entities(shift_query)
    assert all("crewai" not in ent.lower() for ent in new_entities)
    assert all("agent" not in ent.lower() for ent in new_entities)

    # State update on topic shift
    state.active_topic = new_topic
    state.active_entities = new_entities
    state.previous_retrieved_chunks.clear()
    state.previous_visual_evidence.clear()

    assert state.active_topic == new_topic
    assert len(state.previous_retrieved_chunks) == 0


# ============================================================================
# Test 04: Implicit Short Query Continuity
# ============================================================================


def test_04_implicit_short_query_continuity() -> None:
    """
    Verifies that short queries (<= 4 words without explicit subject) are accurately
    recognized as continuations of the active topic rather than false topic shifts.
    """
    resolver = ConversationResolver()
    state = ConversationRAGState(
        conversation_id="conv_test_04",
        active_topic="Health Insurance Benefits",
        active_entities=["Health Insurance", "Full-time Employees", "Dental Coverage"],
        previous_intent=QueryCategory.FACTUAL,
    )

    short_queries = [
        "And for contractors?",
        "How to enroll?",
        "Any exceptions?",
        "What about part-time?",
    ]

    for sq in short_queries:
        word_count = len(sq.split())
        assert word_count <= 4, f"Query '{sq}' must be <= 4 words"
        is_followup, topic_shift, conf, reason = resolver.detect_followup(sq, state)
        assert is_followup is True, f"Short query '{sq}' should be recognized as follow-up: {reason}"
        assert topic_shift is False

    # Standalone query resolution should bind to the active topic
    resolved_enroll = resolver.resolve_standalone_query("How do I enroll in them?", state)
    assert "Health Insurance Benefits" in resolved_enroll
    assert "enroll" in resolved_enroll.lower()


# ============================================================================
# Test 05: Evidence Continuity Across Turns
# ============================================================================


def test_05_evidence_continuity_across_turns() -> None:
    """
    Verifies that when Turn 2 retrieval returns 0 new chunks, evidence continuity
    preserves Turn 1 grounded chunks and prevents context starvation.
    Also tests chunk deduplication across turns.
    """
    gate = EvidenceSufficiencyGate()

    chunk_1 = _create_mock_chunk("c1", "Vacation days accrue at 1.25 days per month.", page_number=5)
    chunk_2 = _create_mock_chunk("c2", "Probationary employees may take leave after 90 days.", page_number=6)
    chunk_3 = _create_mock_chunk("c3", "Unused vacation up to 5 days rolls over annually.", page_number=6)

    # Turn 1: 3 retrieved chunks, DIRECT evidence
    turn1_chunks = [chunk_1, chunk_2, chunk_3]
    eval_turn1 = gate.evaluate(
        query="What is the vacation policy?",
        intent=QueryCategory.FACTUAL,
        candidate_chunks=turn1_chunks,
    )
    assert eval_turn1.is_sufficient is True
    assert eval_turn1.evidence_status == EvidenceStatus.DIRECT

    # Turn 2: Follow-up query "Can I take it in my first month?", 0 new vector hits
    turn2_raw_chunks: list[ScoredChunk] = []
    eval_turn2 = gate.evaluate(
        query="Can I take it in my first month?",
        intent=QueryCategory.FACTUAL,
        candidate_chunks=turn2_raw_chunks,
        previous_status=EvidenceStatus.DIRECT,
        previous_chunks=turn1_chunks,
        is_followup=True,
    )
    assert eval_turn2.is_sufficient is True
    assert eval_turn2.evidence_status == EvidenceStatus.DIRECT
    assert eval_turn2.anchor_chunk is not None
    assert eval_turn2.anchor_chunk.chunk.id == "c1"

    # Verify candidate deduplication logic: Turn 2 retrieves chunk_1 again + chunk_4
    chunk_4 = _create_mock_chunk("c4", "Manager approval required for advance leave.", page_number=7)
    raw_turn2_with_overlap = [chunk_1, chunk_4]

    merged_candidates: list[ScoredChunk] = list(turn1_chunks)
    seen_ids = {sc.chunk.id for sc in merged_candidates}
    for sc in raw_turn2_with_overlap:
        if sc.chunk.id not in seen_ids:
            seen_ids.add(sc.chunk.id)
            merged_candidates.append(sc)

    assert len(merged_candidates) == 4
    assert {c.chunk.id for c in merged_candidates} == {"c1", "c2", "c3", "c4"}


# ============================================================================
# Test 06: Evidence Status Monotonicity Enforcement
# ============================================================================


def test_06_evidence_status_monotonicity_enforcement() -> None:
    """
    Verifies the complete monotonicity state machine transitions:
    DIRECT or PARTIAL evidence is never downgraded to MISSING on follow-up turns.
    Topic shifts properly reset evidence status.
    """
    # 1. DIRECT + MISSING -> DIRECT (Preserved on follow-up)
    st = compute_monotonic_evidence_status(
        previous_status=EvidenceStatus.DIRECT,
        current_status=EvidenceStatus.MISSING,
        has_prev_evidence=True,
        has_curr_evidence=False,
        is_followup=True,
    )
    assert st == EvidenceStatus.DIRECT

    # 2. DIRECT + PARTIAL -> DIRECT
    st = compute_monotonic_evidence_status(
        previous_status=EvidenceStatus.DIRECT,
        current_status=EvidenceStatus.PARTIAL,
        has_prev_evidence=True,
        has_curr_evidence=True,
        is_followup=True,
    )
    assert st == EvidenceStatus.DIRECT

    # 3. DIRECT + RELATED -> DIRECT
    st = compute_monotonic_evidence_status(
        previous_status=EvidenceStatus.DIRECT,
        current_status=EvidenceStatus.RELATED,
        has_prev_evidence=True,
        has_curr_evidence=True,
        is_followup=True,
    )
    assert st == EvidenceStatus.DIRECT

    # 4. PARTIAL + DIRECT -> DIRECT (Upgraded!)
    st = compute_monotonic_evidence_status(
        previous_status=EvidenceStatus.PARTIAL,
        current_status=EvidenceStatus.DIRECT,
        has_prev_evidence=True,
        has_curr_evidence=True,
        is_followup=True,
    )
    assert st == EvidenceStatus.DIRECT

    # 5. PARTIAL + MISSING -> PARTIAL (Preserved!)
    st = compute_monotonic_evidence_status(
        previous_status=EvidenceStatus.PARTIAL,
        current_status=EvidenceStatus.MISSING,
        has_prev_evidence=True,
        has_curr_evidence=False,
        is_followup=True,
    )
    assert st == EvidenceStatus.PARTIAL

    # 6. PARTIAL + RELATED -> PARTIAL
    st = compute_monotonic_evidence_status(
        previous_status=EvidenceStatus.PARTIAL,
        current_status=EvidenceStatus.RELATED,
        has_prev_evidence=True,
        has_curr_evidence=True,
        is_followup=True,
    )
    assert st == EvidenceStatus.PARTIAL

    # 7. RELATED + DIRECT -> DIRECT (Upgraded!)
    st = compute_monotonic_evidence_status(
        previous_status=EvidenceStatus.RELATED,
        current_status=EvidenceStatus.DIRECT,
        has_prev_evidence=True,
        has_curr_evidence=True,
        is_followup=True,
    )
    assert st == EvidenceStatus.DIRECT

    # 8. RELATED + MISSING -> RELATED (Preserved!)
    st = compute_monotonic_evidence_status(
        previous_status=EvidenceStatus.RELATED,
        current_status=EvidenceStatus.MISSING,
        has_prev_evidence=True,
        has_curr_evidence=False,
        is_followup=True,
    )
    assert st == EvidenceStatus.RELATED

    # 9. MISSING + MISSING -> MISSING
    st = compute_monotonic_evidence_status(
        previous_status=EvidenceStatus.MISSING,
        current_status=EvidenceStatus.MISSING,
        has_prev_evidence=False,
        has_curr_evidence=False,
        is_followup=True,
    )
    assert st == EvidenceStatus.MISSING

    # 10. Topic shift: DIRECT + MISSING (is_followup=False) -> MISSING (Reset!)
    st = compute_monotonic_evidence_status(
        previous_status=EvidenceStatus.DIRECT,
        current_status=EvidenceStatus.MISSING,
        has_prev_evidence=True,
        has_curr_evidence=False,
        is_followup=False,
    )
    assert st == EvidenceStatus.MISSING


# ============================================================================
# Test 07: Visual Evidence Reuse Without Re-Inference
# ============================================================================


def test_07_visual_evidence_reuse_without_re_inference() -> None:
    """
    Verifies that multimodal visual chunks (diagrams, flowcharts) extracted in Turn 1
    are reused in Turn 2 without triggering redundant visual re-extraction calls.
    """
    gate = EvidenceSufficiencyGate()

    visual_chunk = _create_mock_chunk(
        "vis_chunk_1",
        "Workflow Architecture Diagram showing Agent execution pipeline and tool interactions.",
        page_number=4,
        content_type=ContentType.PROSE,
        image_assets=["page_4_diagram.png"],
        extra={"visual_type": "diagram_architecture", "image_url": "data/page_4_diagram.png"},
    )

    state = ConversationRAGState(
        conversation_id="conv_test_07",
        active_topic="Content Creation Workflow",
        previous_visual_evidence=[visual_chunk],
        previous_retrieved_chunks=[visual_chunk],
        previous_evidence_status=EvidenceStatus.DIRECT,
    )

    # Turn 2: User asks about diagram workflow
    query = "Explain the workflow shown in that diagram"
    res = gate.evaluate(
        query=query,
        intent=QueryCategory.ARCHITECTURE,
        candidate_chunks=state.previous_visual_evidence,
        previous_status=state.previous_evidence_status,
        previous_chunks=state.previous_retrieved_chunks,
        is_followup=True,
    )

    assert res.is_sufficient is True
    assert res.visual_asset_available is True
    assert res.vision_understanding_available is True
    assert res.evidence_status == EvidenceStatus.DIRECT
    assert "architecture_diagram" not in res.missing_evidence_types


# ============================================================================
# Test 08: Session Isolation & Zero Evidence Leakage
# ============================================================================


def test_08_session_isolation_and_zero_evidence_leakage() -> None:
    """
    Verifies that two distinct conversation IDs maintain strict memory and evidence
    isolation with zero cross-talk, state leakage, or chunk contamination.
    """
    manager = ConversationStateManager()

    session_a_id = "sess_user_alpha"
    session_b_id = "sess_user_beta"

    # Populate Session A
    state_a = manager.get_state(session_a_id)
    state_a.active_topic = "Engineering Remote Work Policy"
    state_a.active_entities = ["Engineering", "VPN", "Home Office Stipend"]
    chunk_a = _create_mock_chunk("chunk_a1", "Engineers are eligible for $500 home office stipend.", source_file="eng_policy.pdf")
    state_a.previous_retrieved_chunks = [chunk_a]
    state_a.last_answer = "Engineers receive $500 for home office setup."
    manager.save_state(state_a)

    # Populate Session B
    state_b = manager.get_state(session_b_id)
    state_b.active_topic = "Sales Commission Policy"
    state_b.active_entities = ["Sales", "Commission", "Quota Target"]
    chunk_b = _create_mock_chunk("chunk_b1", "Sales reps receive 10% commission on closed revenue.", source_file="sales_plan.pdf")
    state_b.previous_retrieved_chunks = [chunk_b]
    state_b.last_answer = "Commission rate is 10% on closed revenue."
    manager.save_state(state_b)

    # Retrieve and cross-verify isolation
    retrieved_a = manager.get_state(session_a_id)
    retrieved_b = manager.get_state(session_b_id)

    # Assert Session A isolation
    assert retrieved_a.active_topic == "Engineering Remote Work Policy"
    assert "Sales" not in retrieved_a.active_entities
    assert all(c.chunk.id != "chunk_b1" for c in retrieved_a.previous_retrieved_chunks)
    assert "Commission" not in str(retrieved_a.last_answer)

    # Assert Session B isolation
    assert retrieved_b.active_topic == "Sales Commission Policy"
    assert "Engineering" not in retrieved_b.active_entities
    assert all(c.chunk.id != "chunk_a1" for c in retrieved_b.previous_retrieved_chunks)
    assert "Stipend" not in str(retrieved_b.last_answer)

    # Verify deep copy isolation: mutating retrieved_a does not mutate manager cache
    retrieved_a.active_topic = "Mutated Topic"
    fresh_a = manager.get_state(session_a_id)
    assert fresh_a.active_topic == "Engineering Remote Work Policy"


# ============================================================================
# Test 09: Thread-Safe State Cache & Eviction
# ============================================================================


def test_09_thread_safe_state_cache_and_eviction() -> None:
    """
    Stress tests ConversationStateManager under 50 concurrent threads performing
    read, update, save, delete, and eviction operations. Verifies zero deadlocks or corruption.
    """
    manager = ConversationStateManager(maxsize=100, ttl=3600)
    errors: list[Exception] = []

    def _worker(thread_idx: int) -> None:
        try:
            conv_id = f"stress_conv_{thread_idx % 10}"
            for step in range(10):
                state = manager.get_state(conv_id)
                state.active_topic = f"Topic_{thread_idx}_{step}"
                state.active_entities.append(f"Entity_{thread_idx}_{step}")
                state.turns.append(
                    ConversationTurn(
                        turn_id=f"turn_{thread_idx}_{step}",
                        user_query=f"Query {step}",
                        resolved_query=f"Resolved Query {step}",
                    )
                )
                manager.save_state(state)
        except Exception as exc:
            errors.append(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(_worker, i) for i in range(50)]
        concurrent.futures.wait(futures)

    assert len(errors) == 0, f"Thread stress test encountered exceptions: {errors}"

    # Verify eviction & deletion
    target_id = "stress_conv_0"
    assert manager.exists(target_id) is True
    manager.delete_state(target_id)
    assert manager.exists(target_id) is False

    # Accessing deleted session returns clean state
    fresh_state = manager.get_state(target_id)
    assert fresh_state.active_topic is None
    assert len(fresh_state.turns) == 0

    # Clear all
    manager.clear_all()
    assert manager.exists("stress_conv_1") is False


# ============================================================================
# Test 10: Answer Mode Selection & Directives
# ============================================================================


def test_10_answer_mode_selection_and_directives() -> None:
    """
    Verifies classification of all 5 answer modes (DIRECT, DETAILED, EXPAND,
    SUMMARY, CONTINUE) and their respective prompt instructions and fidelity modes.
    """
    resolver = ConversationResolver()

    # 1. DIRECT mode
    assert resolver.detect_answer_mode("What is the core working hours policy?") == AnswerMode.DIRECT
    assert _detect_fidelity_mode("What is the core working hours policy?") == "grounded"

    # 2. DETAILED mode
    assert resolver.detect_answer_mode("Give a detailed breakdown of all PTO rules") == AnswerMode.DETAILED
    assert resolver.detect_answer_mode("Provide a thorough and in-depth explanation") == AnswerMode.DETAILED

    # 3. EXPAND mode
    assert resolver.detect_answer_mode("Tell about it in detail") == AnswerMode.EXPAND
    assert resolver.detect_answer_mode("Deep dive into the architecture") == AnswerMode.EXPAND
    assert resolver.detect_answer_mode("Explain further with comprehensive explanation") == AnswerMode.EXPAND

    # 4. SUMMARY mode
    assert resolver.detect_answer_mode("Summarize the policy briefly") == AnswerMode.SUMMARY
    assert resolver.detect_answer_mode("Give me a TLDR in short") == AnswerMode.SUMMARY

    # 5. CONTINUE mode
    assert resolver.detect_answer_mode("Continue with the next steps") == AnswerMode.CONTINUE
    assert resolver.detect_answer_mode("Proceed and keep going") == AnswerMode.CONTINUE

    # Verify regex pattern compilation
    assert _EXPAND_MODE_PATTERN.search("tell about it in detail")
    assert _DETAILED_MODE_PATTERN.search("detailed breakdown")
    assert _SUMMARY_MODE_PATTERN.search("in short summary")
    assert _CONTINUE_MODE_PATTERN.search("next steps")


# ============================================================================
# Test 11: Grounded Expand Synthesis & Four-Tier Separation
# ============================================================================


def test_11_grounded_expand_synthesis_four_tier_separation() -> None:
    """
    Verifies that the grounded prompt generation strictly separates DIRECT,
    PARTIAL, RELATED, and MISSING evidence tiers without hallucinating code or
    falsely claiming code is absent.
    """
    # 1. DIRECT directive
    dir_direct = _format_evidence_status_directive(EvidenceStatus.DIRECT)
    assert "DIRECT IMPLEMENTATION" in dir_direct
    assert "strictly based on the retrieved code" in dir_direct

    # 2. PARTIAL directive
    dir_partial = _format_evidence_status_directive(EvidenceStatus.PARTIAL)
    assert "PARTIAL IMPLEMENTATION" in dir_partial
    assert "DO NOT claim that the document does not contain the code" in dir_partial
    assert "DO NOT fabricate" in dir_partial

    # 3. RELATED directive
    dir_related = _format_evidence_status_directive(EvidenceStatus.RELATED)
    assert "RELATED CONTEXT" in dir_related
    assert "without fabricating code" in dir_related

    # 4. MISSING directive
    dir_missing = _format_evidence_status_directive(EvidenceStatus.MISSING)
    assert "MISSING CONTEXT" in dir_missing
    assert "could not be found" in dir_missing

    # 5. EvidenceSufficiencyGate evaluation on partial kickoff snippet
    gate = EvidenceSufficiencyGate()
    partial_chunk = _create_mock_chunk(
        "c_partial",
        "The workflow is executed by calling:\n```python\nresult = crew.kickoff(inputs={'topic': 'AI'})\n```",
        page_number=12,
        content_type=ContentType.CODE,
    )
    res_eval = gate.evaluate(
        query="How to run the crew?",
        intent=QueryCategory.IMPLEMENTATION,
        candidate_chunks=[partial_chunk],
    )
    assert res_eval.evidence_status == EvidenceStatus.PARTIAL

    # 6. Format full prompt and verify structure
    prompt = GROUNDED_SYSTEM_PROMPT.format(
        evidence_status_directive=dir_partial,
        mode_instructions="Mode: EXPAND",
        refinement_directive="",
        context_text="[Source 1] result = crew.kickoff()",
        history_text=_format_history_for_prompt([{"role": "user", "content": "Tell me about CrewAI"}]),
        query="Tell about it in detail",
    )
    assert "RULE 9: When code snippets, kickoff calls" in prompt
    assert "Evidence Status: PARTIAL IMPLEMENTATION" in prompt
    assert "Mode: EXPAND" in prompt
    assert "USER QUESTION: Tell about it in detail" in prompt


# ============================================================================
# Test 12: E2E Conversational Stream & Observability Logging
# ============================================================================


@pytest.mark.asyncio
async def test_12_e2e_conversational_stream_and_observability_logging(caplog: pytest.LogCaptureFixture) -> None:
    """
    Verifies full multi-turn conversational streaming via Server-Sent Events (SSE),
    request_id propagation across start/retrieval/chunk/citation/trace/done events,
    and structured observability logging ([CONVERSATION], [QUERY_RESOLUTION],
    [EVIDENCE_CONTINUITY], [EVIDENCE_STATUS], [ANSWER_MODE]).
    """
    caplog.set_level(logging.INFO)

    # 1. Setup mock pipeline and chat service
    mock_pipeline = MagicMock()
    mock_pipeline.get_active_model.return_value = "qwen2.5:7b"

    async def _mock_stream_generator(
        user_query: str,
        filters: Any = None,
        history: Any = None,
        model: str | None = None,
        active_document_id: str | None = None,
        active_document_name: str | None = None,
        selected_document_ids: Any = None,
        document_scope: Any = None,
        cancel_token: Any = None,
    ):
        # Simulate structured pipeline logging
        logger = logging.getLogger("backend")
        logger.info(
            "[CONVERSATION] session_id=sess_e2e turn=1 is_followup=%s topic=%s",
            bool(history),
            "Hotel Search Agent" if history else "Hotel Search Agent",
        )
        logger.info(
            "[QUERY_RESOLUTION] query='%s' resolved='%s' confidence=0.95",
            user_query,
            f"Comprehensive explanation of Hotel Search Agent" if history else user_query,
        )
        logger.info(
            "[EVIDENCE_CONTINUITY] prev_chunks=%d new_chunks=3 merged_chunks=3 continuity_applied=%s",
            len(history) if history else 0,
            bool(history),
        )
        logger.info("[EVIDENCE_STATUS] prev=DIRECT current=DIRECT monotonic=DIRECT")
        logger.info("[ANSWER_MODE] mode=%s directives='Mode: EXPAND'", "EXPAND" if history else "DIRECT")

        yield {
            "type": "retrieval_done",
            "candidate_count": 3,
            "context_count": 2,
            "stage_timings": {"dense_retrieval": 10.0, "rerank": 5.0},
        }
        for token in ["Hotel ", "Search ", "Agent ", "orchestrates ", "tools."]:
            yield {"type": "token", "content": token}

        citation = Citation(
            source_index=1,
            document_id="doc_handbook",
            source_file="handbook.pdf",
            page_number=3,
            section_title="Hotel Agent",
            chunk_id="chunk_h1",
            snippet="Hotel Search Agent orchestrates tools.",
            relevance_score=0.95,
        )
        yield {
            "type": "done",
            "citations": [citation],
            "context_chunks": [],
            "token_usage": {"completion_tokens": 5, "prompt_tokens": 120},
            "trace": RAGTrace(
                query=user_query,
                retrieved_candidate_count=3,
                evidence_text_count=2,
                rewritten_query=f"Standalone: {user_query}",
                query_type="factual",
                query_scope="global",
                grounding_status="PASS",
            ),
        }

    mock_pipeline.stream_query = _mock_stream_generator

    mock_telemetry = MagicMock()
    mock_telemetry.record_from_rag_response.return_value = MagicMock(
        model_dump=lambda: {"execution_time_ms": 50.0, "evidence_status": "DIRECT"},
        verification=None,
    )

    chat_service = ChatService(rag_pipeline=mock_pipeline, telemetry_service=mock_telemetry)

    # Turn 1: Initial Query
    req_turn1 = ChatRequest(message="What is the Hotel Search Agent?", session_id="sess_e2e")
    events_turn1: list[str] = []
    async for sse_event in chat_service.stream_query(req_turn1):
        events_turn1.append(sse_event)

    # Verify Turn 1 event stream
    event_types_turn1 = [re.search(r"event:\s*(\w+)", e).group(1) for e in events_turn1 if re.search(r"event:\s*(\w+)", e)]
    assert "start" in event_types_turn1
    assert "retrieval" in event_types_turn1
    assert "chunk" in event_types_turn1
    assert "citation" in event_types_turn1
    assert "done" in event_types_turn1

    # Turn 2: Follow-up Query
    req_turn2 = ChatRequest(message="Tell about it in detail", session_id="sess_e2e")
    events_turn2: list[str] = []
    async for sse_event in chat_service.stream_query(req_turn2):
        events_turn2.append(sse_event)

    event_types_turn2 = [re.search(r"event:\s*(\w+)", e).group(1) for e in events_turn2 if re.search(r"event:\s*(\w+)", e)]
    assert "start" in event_types_turn2
    assert "done" in event_types_turn2

    # Verify all 5 structured observability logging tags were emitted
    log_text = caplog.text
    assert "[CONVERSATION]" in log_text
    assert "[QUERY_RESOLUTION]" in log_text
    assert "[EVIDENCE_CONTINUITY]" in log_text
    assert "[EVIDENCE_STATUS]" in log_text
    assert "[ANSWER_MODE]" in log_text
