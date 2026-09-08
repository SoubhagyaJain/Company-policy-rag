from __future__ import annotations

import json

from backend.models.chunk import Chunk, ChunkMetadata
from backend.models.conversation import (
    ConversationRAGState,
    ConversationStateManager,
    ConversationTurn,
)
from backend.models.rag import (
    QueryCategory,
    QueryClassification,
    QueryRewriteResult,
    RetrievalStrategy,
    ScoredChunk,
)
from backend.rag.conversation_interpreter import (
    ConversationInterpreter,
    RetrievalDecision,
)
from backend.rag.conversation_resolver import ConversationResolutionResult
from backend.rag.pipeline import RAGPipeline
from backend.rag.query_context import QueryContext
from backend.rag.response_modes import get_response_mode_config
from backend.rag.thinking import ThinkingStateMachine


def _chunk(chunk_id: str = "chunk-policy") -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=chunk_id,
            text="Verified policy evidence.",
            metadata=ChunkMetadata(
                document_id="doc-policy",
                source_file="policy.pdf",
                section_title="Eligibility",
                page_number=3,
            ),
        ),
        score=0.95,
    )


def _turn(
    turn_id: str,
    query: str,
    topic: str,
    *,
    evidence: bool = False,
    answer: str = "",
    entities: list[str] | None = None,
) -> ConversationTurn:
    return ConversationTurn(
        turn_id=turn_id,
        user_query=query,
        resolved_query=query,
        active_topic=topic,
        active_entities=entities or [topic],
        retrieved_chunks=[_chunk(f"chunk-{turn_id}")] if evidence else [],
        evidence_status="DIRECT",
        answer=answer,
    )


def _state(topic: str, turn: ConversationTurn, session: str = "session-a") -> ConversationRAGState:
    return ConversationRAGState(
        conversation_id=session,
        active_topic=topic,
        active_entities=list(turn.active_entities),
        turns=[turn],
    )


def test_resolves_pronoun_followup_against_maternity_leave() -> None:
    state = _state(
        "maternity leave",
        _turn("leave-1", "What is maternity leave?", "maternity leave"),
    )

    result = ConversationInterpreter(llm=None).interpret(
        "Does it apply during probation?", state
    )

    assert result.is_followup is True
    assert result.topic_shift is False
    assert result.retrieval_decision == RetrievalDecision.RETRIEVE
    assert "maternity leave" in result.standalone_query.lower()
    assert "probation" in result.standalone_query.lower()


def test_resolves_incomplete_international_travel_followup() -> None:
    state = _state(
        "travel limit",
        _turn("travel-1", "What is the travel limit?", "travel limit"),
    )

    result = ConversationInterpreter(llm=None).interpret("International?", state)

    assert result.is_followup is True
    assert result.retrieval_decision == RetrievalDecision.RETRIEVE
    assert "international" in result.standalone_query.lower()
    assert "travel limit" in result.standalone_query.lower()


def test_keeps_remote_work_topic_for_eligibility_and_contractors() -> None:
    state = _state(
        "remote work",
        _turn("remote-1", "Explain remote work.", "remote work"),
    )
    interpreter = ConversationInterpreter(llm=None)

    eligibility = interpreter.interpret("Who qualifies?", state)
    contractors = interpreter.interpret("And contractors?", state)

    assert eligibility.active_topic == "remote work"
    assert "who qualifies for remote work" in eligibility.standalone_query.lower()
    assert contractors.active_topic == "remote work"
    assert "contractor" in contractors.standalone_query.lower()
    assert "remote work" in contractors.standalone_query.lower()


def test_topic_switch_then_explicit_return_uses_matching_older_topic() -> None:
    state = ConversationRAGState(
        conversation_id="session-return",
        active_topic="reset VPN",
        active_entities=["reset VPN"],
        turns=[
            _turn("leave-1", "Tell me about leave.", "leave"),
            _turn("vpn-1", "How do I reset VPN?", "reset VPN"),
        ],
    )

    result = ConversationInterpreter(llm=None).interpret(
        "Going back to leave, what about probation?", state
    )

    assert result.is_followup is True
    assert result.topic_shift is True
    assert result.returned_to_topic is True
    assert result.active_topic == "leave"
    assert "leave" in result.standalone_query.lower()
    assert "probation" in result.standalone_query.lower()


def test_simplification_reuses_only_verified_prior_evidence() -> None:
    trusted = _turn(
        "remote-1",
        "Explain remote work.",
        "remote work",
        evidence=True,
        answer=(
            "1. Employees need manager approval. [Source 1]\n"
            "2. Contractors require a written exception. [Source 1]"
        ),
    )
    state = _state("remote work", trusted)

    result = ConversationInterpreter(llm=None).interpret(
        "Explain point 2 more simply.", state
    )

    assert result.retrieval_decision == RetrievalDecision.REUSE_PREVIOUS
    assert result.reuse_turn_id == "remote-1"
    assert "contractors require a written exception" in result.standalone_query.lower()


def test_show_source_reuses_evidence_but_missing_evidence_forces_retrieval() -> None:
    interpreter = ConversationInterpreter(llm=None)
    trusted = _state(
        "remote work",
        _turn("trusted", "Explain remote work.", "remote work", evidence=True),
    )
    untrusted = _state(
        "remote work",
        _turn("empty", "Explain remote work.", "remote work", evidence=False),
    )

    assert interpreter.interpret("Show the source", trusted).retrieval_decision == (
        RetrievalDecision.REUSE_PREVIOUS
    )
    assert interpreter.interpret("Show the source", untrusted).retrieval_decision == (
        RetrievalDecision.RETRIEVE
    )
    assert interpreter.interpret(
        "Show the source in the contractor handbook", trusted
    ).retrieval_decision == RetrievalDecision.RETRIEVE


def test_ambiguous_plural_reference_asks_for_clarification() -> None:
    state = _state(
        "maternity leave",
        _turn("leave-1", "What is maternity leave?", "maternity leave"),
    )

    result = ConversationInterpreter(llm=None).interpret("Does it apply to them?", state)

    assert result.retrieval_decision == RetrievalDecision.ASK_CLARIFICATION
    assert result.ambiguous is True
    assert "which" in (result.clarification_question or "").lower()


def test_plural_reference_resolves_when_recent_user_named_one_group() -> None:
    state = ConversationRAGState(
        conversation_id="session-contractors",
        active_topic="remote work",
        active_entities=["remote work", "contractors"],
        turns=[
            _turn("remote-1", "Explain remote work.", "remote work"),
            _turn(
                "remote-2",
                "And contractors?",
                "remote work",
                entities=["remote work", "contractors"],
            ),
        ],
    )

    result = ConversationInterpreter(llm=None).interpret("Does it apply to them?", state)

    assert result.retrieval_decision == RetrievalDecision.RETRIEVE
    assert "remote work" in result.standalone_query.lower()
    assert "contractor" in result.standalone_query.lower()


def test_how_much_and_why_fragments_become_standalone_queries() -> None:
    state = _state(
        "travel reimbursement",
        _turn("travel-1", "Explain travel reimbursement.", "travel reimbursement"),
    )
    interpreter = ConversationInterpreter(llm=None)

    amount = interpreter.interpret("How much?", state)
    reason = interpreter.interpret("Why?", state)

    assert "travel reimbursement" in amount.standalone_query.lower()
    assert "how much" in amount.standalone_query.lower()
    assert "travel reimbursement" in reason.standalone_query.lower()
    assert "why" in reason.standalone_query.lower()


def test_acknowledgement_needs_no_document_retrieval() -> None:
    state = _state("leave", _turn("leave-1", "Tell me about leave.", "leave"))

    result = ConversationInterpreter(llm=None).interpret("Thanks for explaining!", state)

    assert result.intent == QueryCategory.CONVERSATIONAL
    assert result.retrieval_decision == RetrievalDecision.NO_RETRIEVAL


def test_multi_part_question_is_decomposed_without_losing_parts() -> None:
    state = _state("leave", _turn("leave-1", "Tell me about leave.", "leave"))

    result = ConversationInterpreter(llm=None).interpret(
        "What is the travel limit and how do I request approval?", state
    )

    assert result.retrieval_decision == RetrievalDecision.DECOMPOSE
    assert len(result.sub_queries) == 2
    assert any("travel limit" in part.lower() for part in result.sub_queries)
    assert any("request approval" in part.lower() for part in result.sub_queries)


class _RecordingLLM:
    def __init__(self, response: dict[str, object] | str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response if isinstance(self.response, str) else json.dumps(self.response)


def test_interpreter_uses_one_structured_call_without_assistant_answer() -> None:
    state = _state(
        "remote work",
        _turn(
            "remote-1",
            "Explain remote work.",
            "remote work",
            evidence=True,
            answer="UNTRUSTED_ASSISTANT_CLAIM_9274",
        ),
    )
    llm = _RecordingLLM(
        {
            "intent": "factual",
            "answer_mode": "DIRECT",
            "is_followup": True,
            "topic_shift": False,
            "returned_to_topic": False,
            "active_topic": "remote work",
            "active_entities": ["remote work", "contractors"],
            "standalone_query": "What does the remote work policy specify for contractors?",
            "retrieval_decision": "retrieve",
            "reuse_turn_id": None,
            "resolved_references": [],
            "sub_queries": [],
            "ambiguous": False,
            "clarification_question": None,
            "confidence": 0.97,
            "rationale": "Resolved follow-up.",
        }
    )

    result = ConversationInterpreter(llm=llm).interpret("And contractors?", state)

    assert len(llm.prompts) == 1
    assert "UNTRUSTED_ASSISTANT_CLAIM_9274" not in llm.prompts[0]
    assert result.standalone_query == "What does the remote work policy specify for contractors?"


def test_invalid_model_output_falls_back_deterministically() -> None:
    state = _state(
        "travel limit",
        _turn("travel-1", "What is the travel limit?", "travel limit"),
    )
    llm = _RecordingLLM("not json")

    result = ConversationInterpreter(llm=llm).interpret("International?", state)

    assert len(llm.prompts) == 1
    assert result.retrieval_decision == RetrievalDecision.RETRIEVE
    assert "international" in result.standalone_query.lower()
    assert "travel limit" in result.standalone_query.lower()


def test_model_cannot_reuse_old_evidence_for_a_new_fact_request() -> None:
    state = _state(
        "remote work",
        _turn("remote-1", "Explain remote work.", "remote work", evidence=True),
    )
    llm = _RecordingLLM(
        {
            "intent": "factual",
            "answer_mode": "DIRECT",
            "is_followup": True,
            "topic_shift": False,
            "returned_to_topic": False,
            "active_topic": "remote work",
            "active_entities": ["remote work", "contractors"],
            "standalone_query": "What does remote work specify for contractors?",
            "retrieval_decision": "reuse_previous",
            "reuse_turn_id": "remote-1",
            "resolved_references": [],
            "sub_queries": [],
            "ambiguous": False,
            "clarification_question": None,
            "confidence": 0.99,
            "rationale": "Reuse should be enough.",
        }
    )

    result = ConversationInterpreter(llm=llm).interpret("And contractors?", state)

    assert result.retrieval_decision == RetrievalDecision.RETRIEVE
    assert result.reuse_turn_id is None


def test_model_cannot_drag_old_topic_into_clear_topic_switch() -> None:
    state = _state("leave", _turn("leave-1", "Tell me about leave.", "leave"))
    llm = _RecordingLLM(
        {
            "intent": "procedural",
            "answer_mode": "DIRECT",
            "is_followup": True,
            "topic_shift": False,
            "returned_to_topic": False,
            "active_topic": "leave",
            "active_entities": ["leave"],
            "standalone_query": "How do I reset VPN under the leave policy?",
            "retrieval_decision": "retrieve",
            "reuse_turn_id": None,
            "resolved_references": [],
            "sub_queries": [],
            "ambiguous": False,
            "clarification_question": None,
            "confidence": 0.99,
            "rationale": "Continue old topic.",
        }
    )

    result = ConversationInterpreter(llm=llm).interpret("How do I reset VPN?", state)

    assert result.topic_shift is True
    assert result.is_followup is False
    assert "leave" not in result.standalone_query.lower()
    assert "vpn" in (result.active_topic or "").lower()


def test_conversation_state_manager_keeps_sessions_deeply_isolated() -> None:
    manager = ConversationStateManager()
    state_a = manager.get_state("a")
    state_a.active_topic = "leave"
    state_a.turns.append(_turn("a-1", "Tell me about leave.", "leave", evidence=True))
    manager.save_state(state_a)

    state_b = manager.get_state("b")
    state_b.active_topic = "VPN"
    manager.save_state(state_b)
    state_b.turns.append(_turn("b-local", "Local mutation", "mutation", evidence=True))

    stored_a = manager.get_state("a")
    stored_b = manager.get_state("b")
    assert stored_a.active_topic == "leave"
    assert [turn.turn_id for turn in stored_a.turns] == ["a-1"]
    assert stored_b.active_topic == "VPN"
    assert stored_b.turns == []


class _NeverRetrieve:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, *args: object, **kwargs: object) -> list[ScoredChunk]:
        self.calls += 1
        raise AssertionError("retrieval must not run for verified evidence reuse")


class _NoReranker:
    def rerank(self, *args: object, **kwargs: object) -> list[ScoredChunk]:
        raise AssertionError("reranking must not run for verified evidence reuse")


def test_pipeline_reuse_policy_skips_retriever_and_loads_exact_turn_chunks() -> None:
    retriever = _NeverRetrieve()
    trusted_turn = _turn(
        "trusted",
        "Explain remote work.",
        "remote work",
        evidence=True,
    )
    state = _state("remote work", trusted_turn)
    # A later clarification has no evidence; selecting the older trusted turn
    # must retain that turn's DIRECT status instead of inheriting MISSING.
    state.turns.append(
        ConversationTurn(
            turn_id="clarification",
            user_query="Which group?",
            resolved_query="Which group?",
            active_topic="remote work",
            evidence_status="MISSING",
            clarification_required=True,
        )
    )
    state.previous_evidence_status = "MISSING"
    pipeline = RAGPipeline(hybrid_retriever=retriever, reranker=_NoReranker())
    strategy = RetrievalStrategy(rerank_top_n=4, enable_parent_expansion=True)
    ctx = QueryContext(
        user_query="Explain that simply",
        effective_search_query="Explain remote work simply",
        conversation_state=state,
        retrieval_decision=RetrievalDecision.REUSE_PREVIOUS.value,
        reuse_turn_id="trusted",
        thinking_sm=ThinkingStateMachine(query_id="reuse-test"),
        current_strategy=strategy,
        response_mode_config=get_response_mode_config("standard"),
        rewrite_res=QueryRewriteResult(
            original_query="Explain that simply",
            rewritten_query="Explain remote work simply",
        ),
        classification=QueryClassification(
            category=QueryCategory.EXPLANATION,
            strategy=strategy,
        ),
        conv_res=ConversationResolutionResult(
            resolved_query="Explain remote work simply",
            is_followup=True,
            active_topic="remote work",
        ),
    )

    pipeline._stage_retrieve(ctx, "")
    pipeline._stage_rerank_and_context(ctx, "")

    assert retriever.calls == 0
    assert ctx.continuity_applied is True
    assert ctx.raw_new_chunk_count == 0
    assert [item.chunk.id for item in ctx.candidate_chunks] == ["chunk-trusted"]
    assert [item.chunk.id for item in ctx.expanded_chunks] == ["chunk-trusted"]
    assert ctx.telemetry_extra["evidence_reused"] is True
    assert ctx.telemetry_extra["evidence_status"] == "DIRECT"


def test_pipeline_clarification_returns_before_retrieval() -> None:
    retriever = _NeverRetrieve()
    state = _state(
        "maternity leave",
        _turn("leave-1", "What is maternity leave?", "maternity leave"),
    )
    pipeline = RAGPipeline(hybrid_retriever=retriever, reranker=_NoReranker())

    response = pipeline.query(
        "Does it apply to them?",
        conversation_state=state,
    )

    assert retriever.calls == 0
    assert response.citations == []
    assert response.context_chunks == []
    assert response.trace.clarification_required is True
    assert response.trace.retrieval_decision == RetrievalDecision.ASK_CLARIFICATION.value
    assert "which" in response.answer.lower()


class _AnswerLLM:
    model = "test-model"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        return "Remote work eligibility is defined in the policy. [Source 1]"


def test_grounded_generation_excludes_assistant_history_and_keeps_reused_citation() -> None:
    answer_llm = _AnswerLLM()
    pipeline = RAGPipeline(
        hybrid_retriever=_NeverRetrieve(),
        reranker=_NoReranker(),
        llm=answer_llm,
    )
    chunk = _chunk("trusted-evidence")
    strategy = RetrievalStrategy(rerank_top_n=4)
    conv_res = ConversationResolutionResult(
        resolved_query="Explain remote work simply",
        is_followup=True,
        active_topic="remote work",
    )
    ctx = QueryContext(
        user_query="Explain that simply",
        effective_search_query="Explain remote work simply",
        history=[
            {"role": "user", "content": "Explain remote work."},
            {"role": "assistant", "content": "UNTRUSTED_ASSISTANT_FACT_6621"},
        ],
        is_history_followup=True,
        conv_res=conv_res,
        retrieval_decision=RetrievalDecision.REUSE_PREVIOUS.value,
        response_mode_config=get_response_mode_config("standard"),
        current_strategy=strategy,
        rewrite_res=QueryRewriteResult(
            original_query="Explain that simply",
            rewritten_query="Explain remote work simply",
        ),
        classification=QueryClassification(
            category=QueryCategory.EXPLANATION,
            strategy=strategy,
        ),
        expanded_chunks=[chunk],
        policy_selection=pipeline.governing_clause_selector.select(
            "Explain remote work simply", [chunk]
        ),
        telemetry_extra={"evidence_status": "DIRECT"},
        req_llm=answer_llm,
        thinking_sm=ThinkingStateMachine(query_id="generation-test"),
        enable_verification=False,
    )

    pipeline._stage_generate(ctx, "")
    pipeline._stage_verify(ctx, "", attempt=0)

    assert len(answer_llm.prompts) == 1
    assert "STANDALONE QUESTION: Explain remote work simply" in answer_llm.prompts[0]
    assert "Recent Conversation History:" not in answer_llm.prompts[0]
    assert "UNTRUSTED_ASSISTANT_FACT_6621" not in answer_llm.prompts[0]
    assert [citation.chunk_id for citation in ctx.citations] == ["trusted-evidence"]
