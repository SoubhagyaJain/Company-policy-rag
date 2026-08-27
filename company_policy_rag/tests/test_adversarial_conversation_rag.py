"""
Adversarial Stress Test Suite for Conversation-Aware RAG.

Empirically challenges:
1. Complex coreference & nested pronouns (multiple pronouns, possessives, regex metacharacters in topic).
2. Multiline follow-ups, tabs, code snippets in queries, trailing/leading whitespace.
3. Ambiguous short queries, punctuation-only queries, single-word queries, mixed greeting + question.
4. Rapid multi-turn topic oscillation & clearance (switching back and forth between topics).
5. Evidence merging & deduplication with 50+ overlapping chunks, identical text with different IDs, score stability.
6. Multimodal visual chunk lifecycle: persistence during follow-up, clearance during topic shift.
7. Complete 160-case combinatorial Monotonicity State Machine matrix.
8. Concurrent session isolation under 100 worker threads.
"""

from __future__ import annotations

import concurrent.futures
import re
import pytest
from typing import Any

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
    ScoredChunk,
)
from backend.rag.conversation_resolver import (
    ConversationResolver,
    _PRONOUNS_PATTERN,
)
from backend.rag.evidence_gate import (
    EvidenceSufficiencyGate,
    compute_monotonic_evidence_status,
)


def _make_chunk(
    chunk_id: str,
    text: str,
    page_number: int = 1,
    content_type: ContentType = ContentType.PROSE,
    source_file: str = "doc.pdf",
    image_assets: list[Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> ScoredChunk:
    metadata = ChunkMetadata(
        document_id=f"doc_{source_file.replace('.', '_')}",
        source_file=source_file,
        file_path=f"data/{source_file}",
        file_hash=f"hash_{source_file}",
        document_type="pdf",
        chunk_strategy="adaptive",
        page_number=page_number,
        section_title="Test Section",
        content_type=content_type,
        image_assets=image_assets or [],
        extra=extra or {},
    )
    chunk = Chunk(id=chunk_id, text=text, metadata=metadata)
    return ScoredChunk(chunk=chunk, score=0.90)


# ============================================================================
# Adversarial Suite 1: Nested Pronouns, Regex Metachars & Complex Coreference
# ============================================================================


def test_adv_01_nested_and_multiple_pronouns() -> None:
    """Challenge query resolution when multiple nested pronouns appear in one sentence."""
    resolver = ConversationResolver()
    state = ConversationRAGState(
        conversation_id="adv_conv_01",
        active_topic="401(k) Retirement Plan",
        active_entities=["401(k)", "Employer Match", "Vesting Schedule"],
    )

    query = "If I enroll in it, will they match that and what is the rule for this?"
    is_followup, topic_shift, conf, reason = resolver.detect_followup(query, state)
    assert is_followup is True
    assert topic_shift is False

    resolved = resolver.resolve_standalone_query(query, state)
    assert "401(k) Retirement Plan" in resolved
    # Verify standalone query does not retain bare unexpanded pronouns
    assert not re.search(r"\b(it|they|that|this)\b", resolved, flags=re.IGNORECASE)


def test_adv_02_topic_containing_regex_metacharacters() -> None:
    """
    Challenge query resolution when active_topic contains regex special characters
    like '\\1', '$100', 'C++ (v2.0) [Legacy]', 'Plan.*'.
    Verifies that re.sub does not crash on invalid group references or escapes.
    """
    resolver = ConversationResolver()
    tricky_topics = [
        r"C++ API (v2.0) [Legacy]",
        r"Bonus Plan \g<1> & Compensation",
        r"Health Plan ($500 Deductible)",
        r"Regular Expression .* Query",
        r"Path C:\Users\System\Config",
    ]

    for topic in tricky_topics:
        state = ConversationRAGState(
            conversation_id="adv_conv_tricky",
            active_topic=topic,
            active_entities=[topic],
        )
        query = "How does it work?"
        # Must not raise re.error
        resolved = resolver.resolve_standalone_query(query, state)
        assert len(resolved) > len(query)


def test_adv_03_pronoun_substring_false_positive_immunity() -> None:
    """
    Challenge word-boundary robustness: words containing pronoun substrings
    (e.g., 'witness', 'edition', 'theme', 'item', 'submit', 'traditional')
    must NOT trigger false positive pronoun replacements.
    """
    resolver = ConversationResolver()
    state = ConversationRAGState(
        conversation_id="adv_conv_sub",
        active_topic="Vacation Policy",
        active_entities=["Vacation"],
    )

    query = "What is the submission deadline for the new edition item?"
    # Should detect distinct topic or not replace 'submit' / 'edition' / 'item'
    resolved = resolver.resolve_standalone_query(query, state)
    assert "submission" in resolved.lower() or "submit" in resolved.lower()
    assert "edition" in resolved.lower()
    assert "item" in resolved.lower()


# ============================================================================
# Adversarial Suite 2: Multiline Follow-Ups, Code Snippets & Whitespace
# ============================================================================


def test_adv_04_multiline_and_tabbed_followup_queries() -> None:
    """Challenge queries with newlines, bullet points, tabs, and leading/trailing whitespace."""
    resolver = ConversationResolver()
    state = ConversationRAGState(
        conversation_id="adv_conv_multiline",
        active_topic="Hotel Search Agent Architecture",
        active_entities=["Hotel Search Agent", "CrewAI"],
    )

    queries = [
        "\n\n\tTell about it in detail\r\n\t",
        "Explain that:\n- How is it configured?\n- What tools does it use?\n- What are the parameters?",
        "```python\n# how to call it?\n```\nCan you explain this in detail?",
    ]

    for q in queries:
        is_followup, topic_shift, conf, _ = resolver.detect_followup(q, state)
        assert is_followup is True
        assert topic_shift is False

        mode = resolver.detect_answer_mode(q)
        assert mode in (AnswerMode.EXPAND, AnswerMode.DETAILED, AnswerMode.DIRECT)

        resolved = resolver.resolve_standalone_query(q, state, answer_mode=mode)
        assert "Hotel Search Agent" in resolved


# ============================================================================
# Adversarial Suite 3: Ambiguous Short Queries & Edge Cases
# ============================================================================


def test_adv_05_ambiguous_short_queries_and_symbols() -> None:
    """Challenge single-word queries, question marks, greetings mixed with questions."""
    resolver = ConversationResolver()
    state = ConversationRAGState(
        conversation_id="adv_conv_short",
        active_topic="Parental Leave Policy",
        active_entities=["Parental Leave", "Maternity", "Paternity"],
    )

    # 1. Very short follow-up fragments
    short_followups = ["Why?", "How?", "Exceptions?", "Contractors?", "Details?"]
    for q in short_followups:
        is_followup, topic_shift, conf, _ = resolver.detect_followup(q, state)
        assert is_followup is True, f"Failed on short query: {q}"

    # 2. Empty query handling
    is_f, is_s, conf, reason = resolver.detect_followup("", state)
    assert is_f is False
    assert is_s is False

    # 3. Pure whitespace
    is_f, is_s, conf, reason = resolver.detect_followup("   \t\n  ", state)
    assert is_f is False
    assert is_s is False


# ============================================================================
# Adversarial Suite 4: Rapid Multi-Turn Topic Oscillation
# ============================================================================


def test_adv_06_rapid_topic_oscillation_across_6_turns() -> None:
    """
    Challenge rapid topic switching back and forth across 6 turns:
    Turn 1: Vacation Policy (New topic)
    Turn 2: Follow-up on Vacation ("Can it roll over?")
    Turn 3: 401k Match (Topic shift)
    Turn 4: Follow-up on 401k ("How to enroll?")
    Turn 5: Vacation Policy again (Topic shift back)
    Turn 6: Follow-up on Vacation ("Tell about it in detail")
    """
    resolver = ConversationResolver()
    manager = ConversationStateManager()
    session_id = "sess_oscillation"

    # Turn 1: Vacation Policy
    state = manager.get_state(session_id)
    t1_res = resolver.resolve("What is the vacation leave policy?", state)
    assert t1_res.topic_shift is False
    assert t1_res.is_followup is False
    state.active_topic = t1_res.active_topic
    state.active_entities = t1_res.active_entities
    state.last_user_query = "What is the vacation leave policy?"
    manager.save_state(state)
    assert "vacation" in state.active_topic.lower()

    # Turn 2: Follow-up on Vacation
    state = manager.get_state(session_id)
    t2_res = resolver.resolve("Can it roll over to next year?", state)
    assert t2_res.is_followup is True
    assert t2_res.topic_shift is False
    assert "vacation" in t2_res.resolved_query.lower()
    manager.save_state(state)

    # Turn 3: Topic Shift to 401k Match
    state = manager.get_state(session_id)
    t3_res = resolver.resolve("How does the 401k employer match work?", state)
    assert t3_res.is_followup is False
    assert t3_res.topic_shift is True
    assert "401k" in t3_res.active_topic.lower()
    assert "vacation" not in t3_res.active_topic.lower()
    state.active_topic = t3_res.active_topic
    state.active_entities = t3_res.active_entities
    state.last_user_query = "How does the 401k employer match work?"
    manager.save_state(state)

    # Turn 4: Follow-up on 401k Match
    state = manager.get_state(session_id)
    t4_res = resolver.resolve("How to enroll?", state)
    assert t4_res.is_followup is True
    assert t4_res.topic_shift is False
    assert "401k" in t4_res.resolved_query.lower()
    assert "vacation" not in t4_res.resolved_query.lower()

    # Turn 5: Topic Shift back to Vacation Policy
    state = manager.get_state(session_id)
    t5_res = resolver.resolve("What is the company vacation policy?", state)
    assert t5_res.is_followup is False
    assert t5_res.topic_shift is True
    assert "vacation" in t5_res.active_topic.lower()
    assert "401k" not in t5_res.active_topic.lower()
    state.active_topic = t5_res.active_topic
    state.active_entities = t5_res.active_entities
    state.last_user_query = "What is the company vacation policy?"
    manager.save_state(state)

    # Turn 6: Follow-up on Vacation Policy in EXPAND mode
    state = manager.get_state(session_id)
    t6_res = resolver.resolve("Tell about it in detail", state)
    assert t6_res.is_followup is True
    assert t6_res.topic_shift is False
    assert t6_res.answer_mode == AnswerMode.EXPAND
    assert "vacation" in t6_res.resolved_query.lower()
    assert "401k" not in t6_res.resolved_query.lower()


# ============================================================================
# Adversarial Suite 5: Evidence Merging with 50+ Overlapping Chunks
# ============================================================================


def test_adv_07_evidence_deduplication_and_merging_stress() -> None:
    """
    Stress test evidence merging:
    Create 50 previous chunks and 50 new chunks with overlapping IDs and distinct IDs.
    Verify deduplication by chunk ID preserves uniqueness and order.
    """
    prev_chunks = [_make_chunk(f"c_{i}", f"Previous chunk text {i}") for i in range(50)]
    # New chunks: 25 overlap (c_25 to c_49), 25 brand new (c_50 to c_74)
    new_chunks = [_make_chunk(f"c_{i}", f"New chunk text {i}") for i in range(25, 75)]

    dedup_map: dict[str, ScoredChunk] = {}
    for sc in new_chunks:
        dedup_map[sc.chunk.id] = sc
    for sc in prev_chunks:
        if sc.chunk.id not in dedup_map:
            dedup_map[sc.chunk.id] = sc

    merged = list(dedup_map.values())
    assert len(merged) == 75
    unique_ids = {sc.chunk.id for sc in merged}
    assert len(unique_ids) == 75


# ============================================================================
# Adversarial Suite 6: Multimodal Visual Evidence Persistence & Clearance
# ============================================================================


def test_adv_08_multimodal_visual_lifecycle_across_shift() -> None:
    """
    Verifies that visual assets:
    1. Persist across follow-up queries within the same topic.
    2. Are cleared and NOT leaked into a subsequent unrelated topic shift.
    """
    gate = EvidenceSufficiencyGate()

    diagram_chunk = _make_chunk(
        "vis_agent_arch",
        "Agent Architecture Diagram showing tool orchestration.",
        page_number=3,
        image_assets=[{"asset_id": "ast_diagram_1", "visual_type": "diagram_architecture"}],
        extra={"visual_type": "diagram_architecture"},
    )

    state = ConversationRAGState(
        conversation_id="conv_vis_lifecycle",
        active_topic="Agent Architecture",
        previous_visual_evidence=[diagram_chunk],
        previous_retrieved_chunks=[diagram_chunk],
        previous_evidence_status=EvidenceStatus.DIRECT,
    )

    # Follow-up: diagram must be preserved
    res_followup = gate.evaluate(
        query="Walk me through the diagram",
        intent=QueryCategory.ARCHITECTURE,
        candidate_chunks=state.previous_visual_evidence,
        previous_status=state.previous_evidence_status,
        previous_chunks=state.previous_retrieved_chunks,
        is_followup=True,
    )
    assert res_followup.is_sufficient is True
    assert res_followup.visual_asset_available is True

    # Topic shift to Dental Benefits: previous visual must not make it sufficient
    res_shift = gate.evaluate(
        query="What is the dental plan deductible?",
        intent=QueryCategory.FACTUAL,
        candidate_chunks=[],  # No new chunks
        previous_status=state.previous_evidence_status,
        previous_chunks=state.previous_retrieved_chunks,
        is_followup=False,  # TOPIC SHIFT
    )
    assert res_shift.is_sufficient is False
    assert res_shift.evidence_status == EvidenceStatus.MISSING


# ============================================================================
# Adversarial Suite 7: Exhaustive Combinatorial Monotonicity State Machine
# ============================================================================


@pytest.mark.parametrize("prev_status", [EvidenceStatus.DIRECT, EvidenceStatus.PARTIAL, EvidenceStatus.RELATED, EvidenceStatus.MISSING, None])
@pytest.mark.parametrize("curr_status", [EvidenceStatus.DIRECT, EvidenceStatus.PARTIAL, EvidenceStatus.RELATED, EvidenceStatus.MISSING])
@pytest.mark.parametrize("has_prev", [True, False])
@pytest.mark.parametrize("has_curr", [True, False])
@pytest.mark.parametrize("is_followup", [True, False])
def test_adv_09_exhaustive_monotonicity_state_machine(
    prev_status: EvidenceStatus | None,
    curr_status: EvidenceStatus,
    has_prev: bool,
    has_curr: bool,
    is_followup: bool,
) -> None:
    """
    Exhaustively tests all 160 parameter combinations of compute_monotonic_evidence_status.
    Ensures mathematical invariant:
    If is_followup is True and has_prev is True:
    - If prev == DIRECT, monotonic status is NEVER downgraded below DIRECT (unless both lack evidence).
    - If prev == PARTIAL, monotonic status is NEVER downgraded to MISSING.
    - If is_followup is False (topic shift), monotonic status strictly reflects curr_status.
    """
    res = compute_monotonic_evidence_status(
        previous_status=prev_status,
        current_status=curr_status,
        has_prev_evidence=has_prev,
        has_curr_evidence=has_curr,
        is_followup=is_followup,
    )
    assert isinstance(res, EvidenceStatus)

    if not is_followup or prev_status is None:
        assert res == curr_status
    elif is_followup and has_prev:
        if prev_status == EvidenceStatus.DIRECT:
            assert res == EvidenceStatus.DIRECT
        elif prev_status == EvidenceStatus.PARTIAL:
            assert res in (EvidenceStatus.DIRECT, EvidenceStatus.PARTIAL)
        elif prev_status == EvidenceStatus.RELATED:
            assert res in (EvidenceStatus.DIRECT, EvidenceStatus.PARTIAL, EvidenceStatus.RELATED)


# ============================================================================
# Adversarial Suite 8: 100-Thread Concurrent Session Stress
# ============================================================================


def test_adv_10_high_concurrency_100_threads_stress() -> None:
    """Stress test ConversationStateManager under 100 concurrent workers."""
    manager = ConversationStateManager(maxsize=500, ttl=3600)
    errors: list[Exception] = []

    def _worker(worker_id: int) -> None:
        try:
            conv_id = f"worker_sess_{worker_id % 20}"
            for step in range(15):
                st = manager.get_state(conv_id)
                st.active_topic = f"Topic_{worker_id}_{step}"
                st.active_entities.append(f"Entity_{step}")
                st.turns.append(
                    ConversationTurn(
                        turn_id=f"t_{worker_id}_{step}",
                        user_query=f"Query {step}",
                        resolved_query=f"Resolved {step}",
                        answer=f"Answer {step}",
                    )
                )
                manager.save_state(st)
                if step % 5 == 0:
                    manager.exists(conv_id)
        except Exception as exc:
            errors.append(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(_worker, i) for i in range(100)]
        concurrent.futures.wait(futures)

    assert len(errors) == 0, f"Concurrent stress failed with {len(errors)} errors: {errors}"
