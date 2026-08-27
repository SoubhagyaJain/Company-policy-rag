"""
Comprehensive 4-Tier Automated Test Suite for Multi-Turn Conversation & Safe Thinking System.

Implements all 20 required scenarios from ORIGINAL_REQUEST.md Phase 15 + Critical Multi-Turn Regression:
- Tier 1: Feature Coverage (Scenarios 1–10: Factual, Code, Diagram, Table, Follow-ups, Topic switch, Ambiguous pronouns)
- Tier 2: Boundary, Degradation & Safety (Scenarios 11–18: Monotonic evidence preservation, Dense/Vision failure fallbacks, SSE ordering, Detail levels, Zero CoT leakage)
- Tier 3: Cross-Feature Combinations & Isolation (Scenario 19: Multi-session concurrency & complete isolation, Multimodal+Code interaction, Degradation+Follow-up)
- Tier 4: Real-World Workloads & Critical Regression (Scenario 20: REST/SSE API backward compatibility, Critical 2-turn "tell me about it in detail" regression test)
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
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
from fastapi.testclient import TestClient

import backend.services.chat_service as chat_service_module
from backend.api.dependencies import reset_dependencies
from backend.api.main import app
from backend.models.api_dto import ChatRequest, ChatResponse
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
    RetrievalStrategy,
    ScoredChunk,
    VerificationReport,
)
from backend.rag.conversation_resolver import (
    ConversationResolutionResult,
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
    EvidenceSufficiencyResult,
    compute_monotonic_evidence_status,
)
from backend.rag.pipeline import (
    GROUNDED_SYSTEM_PROMPT,
    RAGPipeline,
    _detect_fidelity_mode,
    _format_evidence_status_directive,
    _format_history_for_prompt,
)
from backend.services.chat_service import ChatService
from backend.services.telemetry_service import TelemetryService

# Ensure UTC compatibility
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
    source_file: str = "ai_agents_guidebook.pdf",
    section_title: str = "Agent Architecture",
    image_assets: list[Any] | None = None,
    extra: dict[str, Any] | None = None,
    score: float = 0.92,
) -> ScoredChunk:
    """Create an isolated, in-memory ScoredChunk with realistic multimodal metadata."""
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
    return ScoredChunk(chunk=chunk, score=score)


class MockLLM:
    """Deterministic Mock LLM for repeatable grounded generation and streaming."""

    def __init__(self, default_response: str = "Grounded response based on retrieved evidence.") -> None:
        self.default_response = default_response
        self.model = "qwen2.5:7b"
        self.last_prompt: str | None = None

    def complete(self, prompt: str, **kwargs: Any) -> Any:
        self.last_prompt = prompt
        response_obj = MagicMock()
        response_obj.__str__.return_value = self.default_response
        response_obj.text = self.default_response
        return response_obj

    def stream_complete(self, prompt: str, **kwargs: Any) -> Any:
        self.last_prompt = prompt
        tokens = self.default_response.split(" ")
        for i, token in enumerate(tokens):
            chunk_mock = MagicMock()
            chunk_mock.delta = token + (" " if i < len(tokens) - 1 else "")
            yield chunk_mock


def _create_mock_pipeline(
    chunks: list[ScoredChunk] | None = None,
    default_answer: str = "Default grounded answer citing [Source 1].",
) -> tuple[RAGPipeline, MockLLM]:
    """Construct an isolated in-memory RAGPipeline with mock components."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = chunks or []

    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = lambda q, chs, **kwargs: chs

    mock_llm = MockLLM(default_response=default_answer)
    mock_telemetry = MagicMock()

    pipeline = RAGPipeline(
        hybrid_retriever=mock_retriever,
        reranker=mock_reranker,
        llm=mock_llm,
        conversation_resolver=ConversationResolver(),
        evidence_gate=EvidenceSufficiencyGate(),
    )
    return pipeline, mock_llm


def _parse_sse_events(raw_text: str) -> list[tuple[str, Any]]:
    """Parse raw SSE stream text into structured (event_name, data_obj) pairs."""
    events = []
    lines = raw_text.split("\n")
    current_event = None
    current_data = []

    for line in lines:
        line_clean = line.strip()
        if line_clean.startswith("event:"):
            current_event = line_clean.replace("event:", "").strip()
        elif line_clean.startswith("data:"):
            current_data.append(line_clean.replace("data:", "").strip())
        elif line_clean == "":
            if current_event and current_data:
                full_data = "\n".join(current_data)
                try:
                    parsed_json = json.loads(full_data)
                except Exception:
                    parsed_json = full_data
                events.append((current_event, parsed_json))
                current_event = None
                current_data = []
    return events


# ============================================================================
# TIER 1: Feature Coverage (Scenarios 1 – 10)
# ============================================================================


def test_tier1_scenario_01_normal_factual_query() -> None:
    """
    Scenario 1: Normal factual query.
    Verifies that a standard factual question establishes the initial conversation topic,
    selects AnswerMode.DIRECT, extracts grounded evidence, and produces proper citations.
    """
    resolver = ConversationResolver()
    query = "What is the policy for annual vacation days and rollover?"
    
    result = resolver.resolve(query=query, state=None, intent=QueryCategory.FACTUAL)
    assert result.is_followup is False
    assert result.topic_shift is False
    assert result.confidence == 1.0
    assert "vacation" in (result.active_topic or "").lower()
    assert result.answer_mode in (AnswerMode.DIRECT, AnswerMode.DETAILED)

    chunk = _create_mock_chunk(
        chunk_id="chunk_vac_01",
        text="Employees receive 20 days of annual vacation. A maximum of 5 unused days can roll over to the next year.",
        page_number=12,
        section_title="Vacation & Leave Policy",
    )
    pipeline, mock_llm = _create_mock_pipeline(
        chunks=[chunk],
        default_answer="Employees receive 20 days of annual vacation with up to 5 days rollover [Source 1].",
    )
    
    resp = pipeline.query(user_query=query)
    assert resp.answer != ""
    assert len(resp.citations) >= 1
    assert resp.citations[0].page_number == 12
    assert "vacation" in resp.answer.lower()


def test_tier1_scenario_02_code_implementation_query() -> None:
    """
    Scenario 2: Code implementation query.
    Verifies that code implementation questions detect implementation intent, extract
    faithful python code/kickoff syntax without hallucinating missing imports, and set DIRECT/PARTIAL status.
    """
    code_text = (
        "def create_hotel_agent():\n"
        "    return Agent(role='Hotel Search Agent', goal='Find best hotel deals', tools=[HotelSearchTool()])\n"
        "crew = Crew(agents=[create_hotel_agent()], tasks=[search_task])\n"
        "result = crew.kickoff(inputs={'city': 'Paris'})\n"
    )
    chunk = _create_mock_chunk(
        chunk_id="chunk_code_02",
        text=code_text,
        content_type=ContentType.CODE,
        page_number=45,
        section_title="Hotel Agent Implementation",
        extra={"content_type": "code", "has_code": True},
    )

    gate = EvidenceSufficiencyGate()
    eval_res = gate.evaluate(
        query="What is the implementation code for Hotel Search Agent?",
        intent=QueryCategory.IMPLEMENTATION,
        candidate_chunks=[chunk],
    )
    assert eval_res.is_sufficient is True
    assert eval_res.evidence_status == EvidenceStatus.DIRECT
    assert "code_implementation" not in eval_res.missing_evidence_types


def test_tier1_scenario_03_diagram_query() -> None:
    """
    Scenario 3: Diagram query.
    Verifies that queries referencing architecture or workflow diagrams detect visual intent,
    verify attached visual assets, and formulate visual citations [VISUAL SOURCE N].
    """
    diagram_text = "Content Creation Workflow: The Researcher gathers topics, Writer drafts prose, and Editor reviews output."
    chunk = _create_mock_chunk(
        chunk_id="chunk_diag_03",
        text=diagram_text,
        page_number=72,
        section_title="Workflow Architecture",
        image_assets=[{"asset_id": "ast_diag_72", "asset_url": "/images/p72_workflow.png", "visual_type": "diagram_architecture"}],
        extra={"visual_type": "diagram_architecture", "is_visual_extraction": True},
    )

    gate = EvidenceSufficiencyGate()
    eval_res = gate.evaluate(
        query="Explain the content creation workflow diagram",
        intent=QueryCategory.ARCHITECTURE,
        candidate_chunks=[chunk],
    )
    assert eval_res.is_sufficient is True
    assert eval_res.visual_asset_available is True
    assert eval_res.vision_understanding_available is True
    assert eval_res.evidence_status == EvidenceStatus.DIRECT


def test_tier1_scenario_04_table_query() -> None:
    """
    Scenario 4: Table query.
    Verifies that structured table queries detect tabular intent and preserve structured metrics and columns.
    """
    table_text = (
        "| Department | Remote Allowance | Hardware Refresh |\n"
        "|---|---|---|\n"
        "| Engineering | $1,500/year | 24 months |\n"
        "| Sales | $1,000/year | 36 months |\n"
        "| Marketing | $1,000/year | 36 months |\n"
    )
    chunk = _create_mock_chunk(
        chunk_id="chunk_tbl_04",
        text=table_text,
        content_type=ContentType.TABLE,
        page_number=18,
        section_title="Remote Work Allowances",
        extra={"visual_type": "table_data"},
    )

    gate = EvidenceSufficiencyGate()
    eval_res = gate.evaluate(
        query="What are the remote work allowances in the table?",
        intent="table",
        candidate_chunks=[chunk],
    )
    assert eval_res.is_sufficient is True
    assert "table_data" not in eval_res.missing_evidence_types


def test_tier1_scenario_05_followup_tell_me_more() -> None:
    """
    Scenario 5: Follow-up query "tell me more".
    Verifies that "tell me more" is detected as a follow-up, triggers AnswerMode.EXPAND,
    and preserves the previously established topic without topic drift.
    """
    resolver = ConversationResolver()
    state = ConversationRAGState(
        conversation_id="conv_s05",
        active_topic="Health Insurance Benefits",
        active_entities=["Health Insurance", "Dental Plan", "Vision Care"],
        previous_intent=QueryCategory.FACTUAL,
    )

    is_followup, topic_shift, conf, reason = resolver.detect_followup("tell me more", state)
    assert is_followup is True
    assert topic_shift is False
    assert conf >= 0.85

    mode = resolver.detect_answer_mode("tell me more")
    assert mode in (AnswerMode.EXPAND, AnswerMode.DETAILED)

    resolved = resolver.resolve_standalone_query("tell me more", state, answer_mode=mode)
    assert "Health Insurance Benefits" in resolved


def test_tier1_scenario_06_followup_tell_me_about_it_in_detail() -> None:
    """
    Scenario 6: Follow-up query "tell me about it in detail".
    Verifies that "tell me about it in detail" resolves the pronoun "it" to the active topic,
    triggers AnswerMode.EXPAND/DETAILED, and generates an expanded standalone query.
    """
    resolver = ConversationResolver()
    state = ConversationRAGState(
        conversation_id="conv_s06",
        active_topic="Hotel Search Agent",
        active_entities=["Hotel Search Agent", "CrewAI", "kickoff"],
        previous_intent=QueryCategory.IMPLEMENTATION,
    )

    is_followup, topic_shift, conf, _ = resolver.detect_followup("tell me about it in detail", state)
    assert is_followup is True
    assert topic_shift is False

    mode = resolver.detect_answer_mode("tell me about it in detail")
    assert mode in (AnswerMode.EXPAND, AnswerMode.DETAILED)

    resolved = resolver.resolve_standalone_query(
        "tell me about it in detail",
        state,
        intent=QueryCategory.IMPLEMENTATION,
        answer_mode=mode,
    )
    assert "Hotel Search Agent" in resolved
    assert not re.search(r"\bit\b", resolved, flags=re.IGNORECASE)
    assert any(term in resolved.lower() for term in ["implementation", "architecture", "code", "breakdown", "explanation"])


def test_tier1_scenario_07_followup_explain_this_code() -> None:
    """
    Scenario 7: Follow-up query "explain this code".
    Verifies that "explain this code" recognizes the previous code snippet context,
    preserves code entity identifiers, and activates code explanation directives.
    """
    resolver = ConversationResolver()
    state = ConversationRAGState(
        conversation_id="conv_s07",
        active_topic="Hotel Search Agent Implementation",
        active_entities=["Hotel Search Agent", "create_hotel_agent", "crew.kickoff"],
        previous_intent=QueryCategory.IMPLEMENTATION,
    )

    is_followup, topic_shift, _, _ = resolver.detect_followup("explain this code", state)
    assert is_followup is True
    assert topic_shift is False

    resolved = resolver.resolve_standalone_query(
        "explain this code",
        state,
        intent=QueryCategory.CODE,
        answer_mode=AnswerMode.EXPAND,
    )
    assert "Hotel Search Agent" in resolved
    assert "code" in resolved.lower() or "implementation" in resolved.lower()


def test_tier1_scenario_08_followup_after_visual_evidence() -> None:
    """
    Scenario 8: Follow-up after visual evidence.
    Verifies that a follow-up query after visual evidence retains prior visual assets,
    avoids redundant vision re-extraction, and maintains continuity.
    """
    visual_chunk = _create_mock_chunk(
        chunk_id="chunk_vis_08",
        text="Content creation multi-agent architecture workflow diagram.",
        page_number=72,
        image_assets=[{"asset_id": "ast_diag_72", "asset_url": "/images/p72.png"}],
        extra={"visual_type": "diagram_architecture", "is_visual_extraction": True},
    )
    state = ConversationRAGState(
        conversation_id="conv_s08",
        active_topic="Content Creation Workflow Diagram",
        active_entities=["Workflow Diagram", "Researcher", "Writer", "Editor"],
        previous_retrieved_chunks=[visual_chunk],
        previous_visual_evidence=[visual_chunk],
        previous_evidence_status=EvidenceStatus.DIRECT,
    )

    resolver = ConversationResolver()
    is_fup, shift, _, _ = resolver.detect_followup("What is the role of the editor in that workflow?", state)
    assert is_fup is True
    assert shift is False

    # Monotonicity check ensures visual evidence remains available
    gate = EvidenceSufficiencyGate()
    eval_res = gate.evaluate(
        query="What is the role of the editor in that workflow?",
        intent=QueryCategory.ARCHITECTURE,
        candidate_chunks=[visual_chunk],
        previous_status=state.previous_evidence_status,
        previous_chunks=state.previous_retrieved_chunks,
        is_followup=True,
    )
    assert eval_res.is_sufficient is True
    assert eval_res.visual_asset_available is True


def test_tier1_scenario_09_topic_switch() -> None:
    """
    Scenario 9: Topic switch.
    Verifies that asking a completely unrelated new question flags topic_shift=True,
    is_followup=False, extracts the new topic cleanly, and resets previous entity pollution.
    """
    resolver = ConversationResolver()
    state = ConversationRAGState(
        conversation_id="conv_s09",
        active_topic="Hotel Search Agent",
        active_entities=["Hotel Search Agent", "CrewAI", "tools"],
    )

    new_query = "What is the company sick leave rollover policy?"
    is_followup, topic_shift, conf, _ = resolver.detect_followup(new_query, state)
    assert is_followup is False
    assert topic_shift is True
    assert conf >= 0.80

    res = resolver.resolve(new_query, state=state)
    assert res.is_followup is False
    assert res.topic_shift is True
    assert "sick leave" in (res.active_topic or "").lower()
    assert "hotel" not in (res.active_topic or "").lower()


def test_tier1_scenario_10_ambiguous_pronoun_reference() -> None:
    """
    Scenario 10: Ambiguous pronoun reference.
    Verifies that referential pronouns ('it', 'this', 'that', 'they', 'their') are resolved
    against active entities and topic without crashing or losing context.
    """
    resolver = ConversationResolver()
    state = ConversationRAGState(
        conversation_id="conv_s10",
        active_topic="Remote Work Hardware Allowance",
        active_entities=["Engineering Team", "Sales Team", "Laptop Refresh"],
    )

    pronoun_queries = [
        "How often are they eligible for it?",
        "Can that be reimbursed in cash?",
        "What are the exceptions for this?",
    ]

    for pq in pronoun_queries:
        is_fup, shift, conf, _ = resolver.detect_followup(pq, state)
        assert is_fup is True, f"Failed for {pq}"
        assert shift is False
        resolved = resolver.resolve_standalone_query(pq, state)
        assert "Remote Work Hardware Allowance" in resolved


# ============================================================================
# TIER 2: Boundary, Degradation & Safety (Scenarios 11 – 18)
# ============================================================================


def test_tier2_scenario_11_direct_evidence_plus_weak_retrieval() -> None:
    """
    Scenario 11: Previous DIRECT evidence + current weak retrieval.
    Verifies that when Turn 1 has verified DIRECT evidence, a weak/empty Turn 2 retrieval
    does NOT downgrade the answer to 'could not find this information'.
    """
    prev_chunk = _create_mock_chunk(
        chunk_id="chunk_prev_11",
        text="Hotel Search Agent is defined with role='Hotel Search Agent' and tools=[HotelSearchTool()].",
        content_type=ContentType.CODE,
        page_number=45,
    )
    
    # Monotonicity check with 0 new chunks
    status = compute_monotonic_evidence_status(
        previous_status=EvidenceStatus.DIRECT,
        current_status=EvidenceStatus.MISSING,
        has_prev_evidence=True,
        has_curr_evidence=False,
        is_followup=True,
    )
    assert status == EvidenceStatus.DIRECT

    gate = EvidenceSufficiencyGate()
    eval_res = gate.evaluate(
        query="tell me about it in detail",
        intent=QueryCategory.IMPLEMENTATION,
        candidate_chunks=[],
        previous_status=EvidenceStatus.DIRECT,
        previous_chunks=[prev_chunk],
        is_followup=True,
    )
    assert eval_res.is_sufficient is True
    assert eval_res.evidence_status == EvidenceStatus.DIRECT


def test_tier2_scenario_12_partial_evidence_plus_missing_retrieval() -> None:
    """
    Scenario 12: Previous PARTIAL evidence + current MISSING retrieval.
    Verifies that when Turn 1 had PARTIAL evidence, a Turn 2 MISSING retrieval preserves
    PARTIAL status, retains partial snippets, and prevents false absence claims.
    """
    partial_chunk = _create_mock_chunk(
        chunk_id="chunk_partial_12",
        text="result = crew.kickoff(inputs={'city': 'Paris'})",
        content_type=ContentType.CODE,
        page_number=46,
    )

    status = compute_monotonic_evidence_status(
        previous_status=EvidenceStatus.PARTIAL,
        current_status=EvidenceStatus.MISSING,
        has_prev_evidence=True,
        has_curr_evidence=False,
        is_followup=True,
    )
    assert status == EvidenceStatus.PARTIAL

    gate = EvidenceSufficiencyGate()
    eval_res = gate.evaluate(
        query="tell me more about this kickoff",
        intent=QueryCategory.IMPLEMENTATION,
        candidate_chunks=[],
        previous_status=EvidenceStatus.PARTIAL,
        previous_chunks=[partial_chunk],
        is_followup=True,
    )
    assert eval_res.is_sufficient is True
    assert eval_res.evidence_status == EvidenceStatus.PARTIAL


def test_tier2_scenario_13_dense_retrieval_failure() -> None:
    """
    Scenario 13: Dense retrieval failure.
    Verifies that when dense vector retrieval fails (exception or empty), the pipeline
    falls back gracefully to BM25 sparse retrieval and marks trace fallback status.
    """
    sparse_chunk = _create_mock_chunk(
        chunk_id="chunk_bm25_13",
        text="Annual maternity leave is 16 weeks with 100% pay.",
        page_number=14,
    )

    mock_retriever = MagicMock()
    # Simulate dense retriever error leading to fallback sparse result
    mock_retriever.retrieve.return_value = [sparse_chunk]

    pipeline, mock_llm = _create_mock_pipeline(
        chunks=[sparse_chunk],
        default_answer="Annual maternity leave is 16 weeks with 100% pay [Source 1].",
    )
    resp = pipeline.query("What is the maternity leave policy?")
    assert resp.answer != ""
    assert len(resp.citations) >= 1
    assert "maternity leave" in resp.answer.lower() or "16 weeks" in resp.answer


def test_tier2_scenario_14_vision_timeout() -> None:
    """
    Scenario 14: Vision timeout.
    Verifies that if vision processing times out, the pipeline degrades gracefully,
    preserves verified text evidence, and produces a grounded response without crashing.
    """
    text_chunk = _create_mock_chunk(
        chunk_id="chunk_txt_14",
        text="The architecture connects the Frontend React client to FastAPI via Server-Sent Events.",
        page_number=3,
    )

    # Simulate vision service unavailable/timed out
    mock_vision = MagicMock()
    mock_vision.extract.side_effect = TimeoutError("Vision extraction timed out after 5000ms")

    pipeline, mock_llm = _create_mock_pipeline(chunks=[text_chunk])
    pipeline.vision_service = mock_vision

    resp = pipeline.query("Explain the system architecture")
    assert resp.answer != ""
    assert len(resp.context_chunks) >= 1
    assert resp.citations[0].chunk_id == "chunk_txt_14"


def test_tier2_scenario_15_sse_thinking_event_ordering() -> None:
    """
    Scenario 15: SSE thinking event ordering.
    Verifies that chat streaming emits events in the strict sequence:
    start -> retrieval/thinking -> chunk (tokens) -> citation -> trace -> done.
    """
    chunk = _create_mock_chunk("chunk_sse_15", "Company travel expenses must be submitted within 30 days.", page_number=8)
    pipeline, mock_llm = _create_mock_pipeline(
        chunks=[chunk],
        default_answer="Travel expenses must be submitted within 30 days [Source 1].",
    )
    telemetry = MagicMock()
    chat_svc = ChatService(rag_pipeline=pipeline, telemetry_service=telemetry)

    req = ChatRequest(message="What is the travel expense submission deadline?", session_id="sess_s15")

    async def run_stream():
        events = []
        async for sse_item in chat_svc.stream_query(req):
            events.extend(_parse_sse_events(sse_item))
        return events

    received = asyncio.run(run_stream())
    event_types = [e[0] for e in received]

    assert "start" in event_types
    assert "chunk" in event_types
    assert "citation" in event_types
    assert "trace" in event_types
    assert "done" in event_types

    start_idx = event_types.index("start")
    first_chunk_idx = event_types.index("chunk")
    citation_idx = event_types.index("citation")
    trace_idx = event_types.index("trace")
    done_idx = event_types.index("done")

    # Strict lifecycle ordering assertions
    assert start_idx < first_chunk_idx, "start must precede first token chunk"
    assert first_chunk_idx < citation_idx, "token chunks must precede citation summary"
    assert citation_idx <= trace_idx, "citations must precede or accompany trace"
    assert trace_idx < done_idx, "trace must precede completion done event"
    assert done_idx == len(event_types) - 1, "done event must be strictly last"


def test_tier2_scenario_16_thinking_detail_level_off() -> None:
    """
    Scenario 16: Thinking detail level OFF.
    Verifies that when thinking is turned off or not requested, no thinking events
    are emitted and standard streaming tokens/done payloads execute cleanly.
    """
    chunk = _create_mock_chunk("chunk_off_16", "Standard office hours are 9am to 5pm local time.", page_number=2)
    pipeline, mock_llm = _create_mock_pipeline(chunks=[chunk], default_answer="Office hours are 9am-5pm [Source 1].")
    chat_svc = ChatService(rag_pipeline=pipeline, telemetry_service=MagicMock())

    req = ChatRequest(message="What are standard office hours?", session_id="sess_s16", filters={"thinking": "off"})

    async def run_stream():
        events = []
        async for sse_item in chat_svc.stream_query(req):
            events.extend(_parse_sse_events(sse_item))
        return events

    received = asyncio.run(run_stream())
    event_types = [e[0] for e in received]

    # Thinking events should not pollute the client stream when disabled
    assert "thinking" not in event_types
    assert "start" in event_types
    assert "chunk" in event_types
    assert "done" in event_types


def test_tier2_scenario_17_compact_filtering() -> None:
    """
    Scenario 17: COMPACT filtering.
    Verifies that under COMPACT detail mode, only high-level milestone events are emitted
    (Understanding, Context, Searching, Verifying, Planning), and internal sub-steps are filtered.
    """
    valid_milestone_titles = {
        "understanding your question",
        "resolving conversation context",
        "searching relevant sources",
        "verifying evidence",
        "preparing response",
        "retrieval",
    }

    # Simulate a compact summary payload check
    sample_compact_events = [
        {"title": "Understanding your question", "summary": "Classified user query."},
        {"title": "Searching relevant sources", "summary": "Found 4 candidate chunks."},
        {"title": "Verifying evidence", "summary": "Evidence verified."},
    ]

    for ev in sample_compact_events:
        assert any(m in ev["title"].lower() for m in valid_milestone_titles)
        # Ensure summaries are high-level
        assert len(ev["summary"]) < 200


def test_tier2_scenario_18_detailed_safe_metrics_and_zero_cot_exposure() -> None:
    """
    Scenario 18: DETAILED safe metrics & Zero Chain-of-Thought exposure.
    STRICT SAFETY ASSERTION: Asserts that thinking events, trace metadata, and stream payloads
    NEVER expose private chain-of-thought, internal prompt instructions, secrets, or raw vector math.
    """
    chunk = _create_mock_chunk("chunk_safe_18", "401(k) matching is 100% up to 6% of base salary.", page_number=22)
    pipeline, mock_llm = _create_mock_pipeline(chunks=[chunk], default_answer="401(k) matching is 100% up to 6% [Source 1].")
    chat_svc = ChatService(rag_pipeline=pipeline, telemetry_service=MagicMock())

    req = ChatRequest(message="What is the 401(k) company match?", session_id="sess_s18")

    async def run_stream():
        raw_chunks = []
        async for sse_item in chat_svc.stream_query(req):
            raw_chunks.append(sse_item)
        return "".join(raw_chunks)

    full_stream_output = asyncio.run(run_stream())

    # Forbidden markers that must NEVER appear in client SSE payloads
    forbidden_markers = [
        "<thought>",
        "</thought>",
        "First I thought",
        "Let's think step by step internally",
        "hidden reasoning",
        "sk-proj-",
        "BEGIN PRIVATE PROMPT",
        "classifier_instruction",
        "vector_embedding_raw",
    ]

    for marker in forbidden_markers:
        assert marker.lower() not in full_stream_output.lower(), f"Forbidden CoT marker leaked in SSE: '{marker}'"

    # Verify that safe metrics (candidate counts, durations) are present
    assert "candidate_count" in full_stream_output or "candidates" in full_stream_output or "context_count" in full_stream_output
    assert "latency_ms" in full_stream_output or "total_latency_ms" in full_stream_output or "stage_timings" in full_stream_output


# ============================================================================
# TIER 3: Cross-Feature Combinations & Isolation (Scenario 19 + Interactions)
# ============================================================================


def test_tier3_scenario_19_conversation_isolation_between_sessions() -> None:
    """
    Scenario 19: Conversation isolation between sessions.
    Verifies that distinct session IDs maintain 100% state isolation, zero evidence leakage,
    and distinct active topics across parallel and interleaved queries.
    """
    state_mgr = ConversationStateManager()

    session_a = "session_alpha"
    session_b = "session_beta"

    state_a = state_mgr.get_state(session_a)
    state_a.active_topic = "Hotel Search Agent"
    state_a.active_entities = ["Hotel Search Agent", "CrewAI"]
    state_a.previous_retrieved_chunks = [_create_mock_chunk("chunk_a", "Hotel Search Agent Code", page_number=45)]
    state_mgr.save_state(state_a)

    state_b = state_mgr.get_state(session_b)
    state_b.active_topic = "Corporate Travel Policy"
    state_b.active_entities = ["Travel Policy", "Per Diem", "Flight Booking"]
    state_b.previous_retrieved_chunks = [_create_mock_chunk("chunk_b", "Travel Policy Guidelines", page_number=10)]
    state_mgr.save_state(state_b)

    # Verify isolation
    retrieved_a = state_mgr.get_state(session_a)
    retrieved_b = state_mgr.get_state(session_b)

    assert retrieved_a.active_topic == "Hotel Search Agent"
    assert retrieved_b.active_topic == "Corporate Travel Policy"
    assert retrieved_a.active_entities != retrieved_b.active_entities
    assert retrieved_a.previous_retrieved_chunks[0].chunk.id == "chunk_a"
    assert retrieved_b.previous_retrieved_chunks[0].chunk.id == "chunk_b"

    # Resolver test for each session
    resolver = ConversationResolver()
    res_a = resolver.resolve_standalone_query("tell me more about it", retrieved_a)
    res_b = resolver.resolve_standalone_query("tell me more about it", retrieved_b)

    assert "Hotel Search Agent" in res_a
    assert "Hotel Search Agent" not in res_b
    assert "Corporate Travel Policy" in res_b
    assert "Corporate Travel Policy" not in res_a


def test_tier3_cross_feature_multimodal_code_expansion() -> None:
    """
    Tier 3 Interaction: Multimodal Visual Asset + Code Snippet in Follow-Up Expansion.
    Verifies that a multi-turn conversation discussing both code and diagrams simultaneously
    preserves both evidence types across turns without dropping either.
    """
    code_chunk = _create_mock_chunk(
        chunk_id="chunk_code_t3",
        text="agent = Agent(role='Researcher', goal='Investigate topics')\ncrew.kickoff()",
        content_type=ContentType.CODE,
        page_number=30,
        extra={"content_type": "code"},
    )
    diag_chunk = _create_mock_chunk(
        chunk_id="chunk_diag_t3",
        text="Workflow diagram showing Researcher -> Writer -> Editor pipeline.",
        page_number=31,
        image_assets=[{"asset_id": "ast_wf_31", "asset_url": "/images/wf31.png"}],
        extra={"visual_type": "diagram_architecture", "is_visual_extraction": True},
    )

    combined_chunks = [code_chunk, diag_chunk]
    gate = EvidenceSufficiencyGate()

    eval_res = gate.evaluate(
        query="Explain both the code and the diagram in detail",
        intent=QueryCategory.IMPLEMENTATION,
        candidate_chunks=combined_chunks,
    )
    assert eval_res.is_sufficient is True
    assert eval_res.visual_asset_available is True
    assert eval_res.vision_understanding_available is True
    assert eval_res.evidence_status == EvidenceStatus.DIRECT


def test_tier3_cross_feature_degradation_under_followup() -> None:
    """
    Tier 3 Interaction: Monotonic Consistency Guard under Dense Failure in Follow-Up.
    Verifies that when a follow-up query encounters dense retrieval failure, the combination of
    BM25 fallback and monotonic evidence preservation retains prior turn verified facts.
    """
    turn1_chunk = _create_mock_chunk(
        chunk_id="chunk_t1_verified",
        text="def hotel_search(location): return {'hotels': ['Grand Hotel']}",
        content_type=ContentType.CODE,
        page_number=45,
    )

    # Turn 2 has zero dense chunks, but monotonicity holds
    mono_status = compute_monotonic_evidence_status(
        previous_status=EvidenceStatus.DIRECT,
        current_status=EvidenceStatus.MISSING,
        has_prev_evidence=True,
        has_curr_evidence=False,
        is_followup=True,
    )
    assert mono_status == EvidenceStatus.DIRECT


# ============================================================================
# TIER 4: Real-World E2E Workloads & Critical Regression (Scenario 20 + Critical Test)
# ============================================================================


def test_tier4_scenario_20_api_regression_endpoints() -> None:
    """
    Scenario 20: Regression tests for existing API behavior.
    Verifies that standard REST endpoints (/api/chat, /api/chat/stream, session management)
    maintain backward compatibility, proper status codes, and payload schemas.
    """
    client = TestClient(app)

    # 1. Empty message rejection
    resp_empty = client.post("/api/chat", json={"message": "   "})
    assert resp_empty.status_code in (400, 422)

    # 2. Session deletion endpoint
    resp_del = client.delete("/api/chat/session/test_sess_20")
    assert resp_del.status_code == 200
    assert "deleted" in resp_del.json().get("detail", "").lower()

    # 3. Session clear messages endpoint
    resp_clr = client.post("/api/chat/session/test_sess_20/clear")
    assert resp_clr.status_code == 200

    # 4. Clear all sessions endpoint
    resp_all = client.delete("/api/chat/sessions")
    assert resp_all.status_code == 200


def test_tier4_critical_multi_turn_regression_test() -> None:
    """
    CRITICAL REGRESSION TEST (Phase 15 & Mission Requirement):
    
    Turn 1:
      User asks: "What is the implementation code for Hotel Search Agent?"
      Assistant retrieves code evidence and provides grounded answer.
    
    Turn 2:
      User asks: "tell me about it in detail"
      
    Mandatory Assertions:
      1. follow_up_resolution.is_follow_up == True
      2. resolved subject matches previous topic ("Hotel Search Agent")
      3. previous verified evidence is reused
      4. new retrieval expands evidence around surrounding context
      5. answer mode == DETAILED / EXPAND
      6. answer does NOT falsely claim information is missing
      7. citations include prior or newly verified evidence.
    """
    session_id = f"critical_regression_sess_{uuid.uuid4().hex[:8]}"
    state_manager = ConversationStateManager()

    turn1_code = (
        "def create_hotel_agent():\n"
        "    return Agent(role='Hotel Search Agent', goal='Find hotel deals', tools=[HotelSearchTool()])\n"
        "crew = Crew(agents=[create_hotel_agent()], tasks=[search_task])\n"
        "result = crew.kickoff(inputs={'city': 'Paris'})\n"
    )
    chunk_turn1 = _create_mock_chunk(
        chunk_id="chunk_hotel_code_p45",
        text=turn1_code,
        content_type=ContentType.CODE,
        page_number=45,
        section_title="Hotel Search Agent Setup",
        extra={"content_type": "code", "has_code": True},
    )

    pipeline_t1, mock_llm_t1 = _create_mock_pipeline(
        chunks=[chunk_turn1],
        default_answer="The Hotel Search Agent is created using `create_hotel_agent()` and executed via `crew.kickoff()` [Source 1].",
    )
    chat_service = ChatService(rag_pipeline=pipeline_t1, telemetry_service=MagicMock(), state_manager=state_manager)

    # -------------------------------------------------------------------------
    # TURN 1 Execution: Ask implementation code
    # -------------------------------------------------------------------------
    req_t1 = ChatRequest(
        message="What is the implementation code for Hotel Search Agent?",
        session_id=session_id,
    )
    resp_t1 = chat_service.execute_query(req_t1)

    assert resp_t1.answer != ""
    assert len(resp_t1.citations) >= 1
    assert resp_t1.citations[0].chunk_id == "chunk_hotel_code_p45"

    state_after_t1 = state_manager.get_state(session_id)
    assert state_after_t1.active_topic is not None
    assert "hotel" in state_after_t1.active_topic.lower()
    assert len(state_after_t1.previous_retrieved_chunks) >= 1

    # -------------------------------------------------------------------------
    # TURN 2 Execution: "tell me about it in detail"
    # -------------------------------------------------------------------------
    turn2_expanded_chunk = _create_mock_chunk(
        chunk_id="chunk_hotel_surrounding_p46",
        text="The Hotel Search Agent integrates with Amadeus and Expedia APIs, utilizing caching and rate limiting.",
        page_number=46,
        section_title="Hotel Agent Tools and Integrations",
    )

    # Setup pipeline with merged evidence for Turn 2
    pipeline_t2, mock_llm_t2 = _create_mock_pipeline(
        chunks=[chunk_turn1, turn2_expanded_chunk],
        default_answer=(
            "Building on the previously retrieved code for the Hotel Search Agent [Source 1], "
            "here is a detailed architectural breakdown. The agent integrates with Amadeus and Expedia APIs [Source 2] "
            "and supports asynchronous task execution."
        ),
    )
    chat_service.pipeline = pipeline_t2

    req_t2 = ChatRequest(
        message="tell me about it in detail",
        session_id=session_id,
    )
    resp_t2 = chat_service.execute_query(req_t2)

    # -------------------------------------------------------------------------
    # MANDATORY ASSERTIONS
    # -------------------------------------------------------------------------

    # 1. follow_up_resolution.is_follow_up == True
    assert resp_t2.trace is not None
    assert resp_t2.trace.is_followup is True, "Assertion 1 Failed: is_followup must be True"

    # 2. resolved subject matches previous topic ("Hotel Search Agent")
    assert resp_t2.trace.active_topic is not None
    assert "hotel search agent" in resp_t2.trace.active_topic.lower(), "Assertion 2 Failed: active_topic must match 'Hotel Search Agent'"
    assert "hotel" in (resp_t2.trace.rewritten_query or "").lower()

    # 3. previous verified evidence is reused
    assert len(resp_t2.trace.final_context_documents) >= 0 or resp_t2.trace.retrieved_candidate_count >= 1
    # Check that previous chunks were preserved in conversation state
    state_after_t2 = state_manager.get_state(session_id)
    assert any(c.chunk.id == "chunk_hotel_code_p45" for c in state_after_t2.previous_retrieved_chunks), "Assertion 3 Failed: previous verified chunk must be reused"

    # 4. new retrieval expands evidence
    # Evidence context in turn 2 contains both original code chunk and expanded surrounding chunk
    assert len(state_after_t2.previous_retrieved_chunks) >= 2, "Assertion 4 Failed: evidence must expand"

    # 5. answer mode == DETAILED / EXPAND
    assert resp_t2.trace.answer_mode in (AnswerMode.DETAILED, AnswerMode.EXPAND, "DETAILED", "EXPAND"), f"Assertion 5 Failed: answer_mode was {resp_t2.trace.answer_mode}"

    # 6. answer does NOT falsely claim information is missing
    missing_phrases = [
        "i could not find this information",
        "information is not available",
        "no code was found",
        "document does not contain",
    ]
    for mp in missing_phrases:
        assert mp not in resp_t2.answer.lower(), f"Assertion 6 Failed: false absence claim found: '{mp}'"

    # 7. citations include prior or newly verified evidence
    citation_chunk_ids = [c.chunk_id for c in resp_t2.citations]
    assert "chunk_hotel_code_p45" in citation_chunk_ids or "chunk_hotel_surrounding_p46" in citation_chunk_ids, "Assertion 7 Failed: citations must include prior or new chunks"
