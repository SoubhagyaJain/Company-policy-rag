"""
Regression tests for multi-turn conversation quality.

These cover the four failure modes that made late turns worse than early ones:
history pollution, topic-shift blindness, weak standalone rewriting, and
multi-part questions losing every part after the first.
"""
from __future__ import annotations

import pytest

from backend.models.conversation import ConversationRAGState, ConversationTurn
from backend.rag.conversation_resolver import (
    _MAX_ACTIVE_ENTITIES,
    FollowUpResolver,
)
from backend.rag.multi_query import MultiQueryGenerator, decompose_multi_part
from backend.rag.query_rewrite import QueryRewriter


def _state(topic: str, entities: list[str], last_query: str = "") -> ConversationRAGState:
    return ConversationRAGState(
        conversation_id="sess_test",
        active_topic=topic,
        active_entities=list(entities),
        last_user_query=last_query,
    )


def _history(pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for user_msg, assistant_msg in pairs:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    return messages


# ---------------------------------------------------------------------------
# History pollution
# ---------------------------------------------------------------------------


def test_fallback_rewrite_does_not_drag_in_the_first_question_of_the_session():
    """The opening topic must not be welded onto every later follow-up."""
    rewriter = QueryRewriter(enable_llm_rewrite=False)
    history = _history(
        [
            ("What is the parental leave policy?", "Twelve weeks..."),
            ("How do I submit an expense report?", "Use the portal..."),
            ("What is the travel per diem?", "Sixty dollars..."),
        ]
    )

    rewritten = rewriter._fallback_rewrite("what about it for contractors", "what about it for contractors", history)

    assert "parental leave" not in rewritten.lower(), (
        f"turn-1 topic leaked into a later turn: {rewritten!r}"
    )
    assert "per diem" in rewritten.lower(), "the immediately preceding turn should bind the follow-up"


def test_long_conversation_does_not_accumulate_unbounded_entities():
    """Entity memory stays bounded so late turns are not forced into follow-ups."""
    resolver = FollowUpResolver()
    state = _state("vacation policy", ["Vacation Policy"])

    for turn in range(25):
        result = resolver.resolve(f"Tell me more about it, detail number {turn}", state)
        state = _state(
            result.active_topic or "",
            result.active_entities,
            last_query=f"Tell me more about it, detail number {turn}",
        )

    assert len(state.active_entities) <= _MAX_ACTIVE_ENTITIES, (
        f"active_entities grew to {len(state.active_entities)}; "
        "an unbounded list makes every later question look like a follow-up"
    )


# ---------------------------------------------------------------------------
# Topic-shift detection
# ---------------------------------------------------------------------------


def test_new_question_after_twenty_turns_is_not_forced_into_a_followup():
    """A fresh question must survive a long, entity-rich conversation."""
    resolver = FollowUpResolver()
    # A conversation that has accumulated many entities, as turn 20 would have.
    state = _state(
        "expense reimbursement policy",
        [
            "Expense Report", "Travel Policy", "Per Diem", "Manager Approval",
            "Corporate Card", "Receipt Threshold", "Mileage Rate", "Airfare Booking",
            "Hotel Allowance", "Meal Limit", "Approval Workflow", "Finance Team",
        ],
    )

    result = resolver.resolve("What is the parental leave policy?", state)

    assert result.topic_shift is True, f"topic shift missed: {result.reason}"
    assert result.is_followup is False
    assert result.resolved_query == "What is the parental leave policy?", (
        "a topic-shifting question must reach retrieval unmodified"
    )


def test_single_shared_word_does_not_make_a_new_question_a_followup():
    """One shared token ('policy') is too weak to claim continuity."""
    resolver = FollowUpResolver()
    state = _state("remote work policy", ["Remote Work Policy"])

    result = resolver.resolve("What is the dress code policy?", state)

    assert result.topic_shift is True
    assert "remote work" not in result.resolved_query.lower()


def test_genuine_followup_is_still_detected():
    """The fix must not break real follow-ups."""
    resolver = FollowUpResolver()
    state = _state("vacation carryover", ["Vacation Carryover"])

    result = resolver.resolve("tell me more about it", state)

    assert result.is_followup is True
    assert result.topic_shift is False
    assert "vacation carryover" in result.resolved_query.lower(), (
        "a referential follow-up must be resolved against the active topic"
    )


# ---------------------------------------------------------------------------
# Standalone rewriting
# ---------------------------------------------------------------------------


def test_short_llm_rewrite_is_accepted():
    """A correct rewrite is often shorter than the question it replaces."""

    class ShortRewriteLLM:
        def complete(self, prompt: str) -> str:
            return "vacation carryover limit"

    resolver = FollowUpResolver(llm=ShortRewriteLLM())
    state = _state("vacation carryover", ["Vacation Carryover"])

    resolved = resolver.resolve_standalone_query(
        "could you please tell me a lot more about that whole thing", state
    )

    assert resolved == "vacation carryover limit"


def test_refusal_from_the_rewriter_is_rejected():
    class RefusingLLM:
        def complete(self, prompt: str) -> str:
            return "I cannot rewrite that question."

    resolver = FollowUpResolver(llm=RefusingLLM())
    state = _state("vacation carryover", ["Vacation Carryover"])

    resolved = resolver.resolve_standalone_query("tell me more about it", state)

    assert "cannot" not in resolved.lower()
    assert "vacation carryover" in resolved.lower()


def test_pronoun_binding_does_not_repeat_the_topic():
    """Only the first pronoun binds; repeating the topic makes a spammy query."""
    resolver = FollowUpResolver()
    state = _state("health insurance", ["Health Insurance"])

    resolved = resolver.resolve_standalone_query("is that the same as this one", state)

    assert resolved.lower().count("health insurance") == 1, (
        f"topic repeated in search query: {resolved!r}"
    )


# ---------------------------------------------------------------------------
# Multi-part decomposition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected_parts",
    [
        ("What is the leave policy and how do I expense travel?", 2),
        ("What is the dress code? How many sick days do I get? Who approves overtime?", 3),
        ("Tell me about health insurance and also explain the dental plan", 2),
        ("1) What is the notice period 2) How do I resign", 2),
    ],
)
def test_multi_part_questions_are_decomposed(query: str, expected_parts: int):
    parts = decompose_multi_part(query)
    assert len(parts) == expected_parts, f"{query!r} -> {parts}"


@pytest.mark.parametrize(
    "query",
    [
        "What are the terms and conditions?",
        "Explain the health and safety policy",
        "What is the vacation policy?",
        "How do I submit an expense report and receipt?",
    ],
)
def test_single_questions_are_not_split(query: str):
    """Conjunctions joining nouns must not be mistaken for separate questions."""
    assert decompose_multi_part(query) == []


def test_generator_emits_a_subquery_for_every_part():
    """Each part must reach retrieval, or its evidence is never in the pool."""
    generator = MultiQueryGenerator()
    query = "What is the leave policy and how do I expense travel?"

    subqueries = generator.generate_subqueries(query)
    joined = " || ".join(q.lower() for q in subqueries)

    assert "leave policy" in joined
    assert "expense travel" in joined


def test_decomposition_is_stable_for_empty_and_trivial_input():
    assert decompose_multi_part("") == []
    assert decompose_multi_part("   ") == []
    assert decompose_multi_part("hi") == []
