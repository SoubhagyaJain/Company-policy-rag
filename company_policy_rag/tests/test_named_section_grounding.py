from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.models.chunk import Chunk, ChunkMetadata
from backend.models.conversation import ConversationRAGState
from backend.models.rag import QueryCategory, ScoredChunk
from backend.rag.conversation_resolver import ConversationResolver
from backend.rag.context_compression import ContextCompressor
from backend.rag.citations import CitationEngine
from backend.rag.evidence_gate import EvidenceSufficiencyGate
from backend.rag.pipeline import RAGPipeline
from backend.rag.policy_reliability import GoverningClauseSelector
from backend.rag.query_context import QueryContext
from backend.rag.response_modes import get_response_mode_config
from backend.rag.section_matching import named_section_matches, prioritize_named_sections
from backend.utils.section_tracker import parse_section_heading


def passage(cid, text, page, title="Browserbase tool"):
    return ScoredChunk(chunk=Chunk(id=cid, text=text, metadata=ChunkMetadata(
        document_id="doc_book", source_file="guide.pdf", page_number=page,
        display_page_number=page - 1, section_title=title,
    )), score=0.8)


@pytest.fixture
def evidence():
    return [
        passage("unrelated", "CrewAI supports several tools, as depicted below.", 14, "Memory"),
        passage("toc", "#4) Financial Analyst.................54", 4),
        passage("target", "#4) Financial Analyst\nBuild an AI agent that analyzes stock market trends.\n"
                "Tech stack:\n- CrewAI for multi-agent orchestration\n"
                "- Ollama to locally serve DeepSeek-R1 LLM\n- Cursor as the MCP host\n"
                "Workflow:\n- The MCP agent starts the crew.\n"
                "- The crew researches and creates a script.\n- The agent runs it to generate an analysis plot.", 55),
    ]


def test_hash_numbered_heading_and_toc():
    heading = parse_section_heading("#4) Financial Analyst")
    assert heading.section_title == "Financial Analyst"
    assert heading.section_number == "4"
    assert parse_section_heading("#4) Financial Analyst...........54") is None


def test_named_section_survives_stale_metadata_and_bad_ranking(evidence):
    query = "how can i make Financial Analyst what is the tech stack of it"
    assert [sc.chunk.id for sc in named_section_matches(query, evidence)] == ["target"]
    result = prioritize_named_sections(query, evidence[:1], evidence)
    assert result[0].chunk.id == "target"
    gate = EvidenceSufficiencyGate().evaluate(query, QueryCategory.IMPLEMENTATION, result)
    assert gate.is_sufficient
    assert gate.pages_to_inspect == []


def test_no_project_specific_names_are_required():
    chunks = [passage("target", "## Warehouse Inventory Assistant\nTech stack: Python and PostgreSQL.", 8)]
    query = "What is the tech stack of Warehouse Inventory Assistant?"
    assert named_section_matches(query, chunks) == chunks
    assert EvidenceSufficiencyGate().evaluate(query, QueryCategory.IMPLEMENTATION, chunks).is_sufficient


def test_exact_code_request_still_requires_real_code(evidence):
    query = "Give me the exact code and tech stack of Financial Analyst"
    result = EvidenceSufficiencyGate().evaluate(query, QueryCategory.CODE, evidence[-1:])
    assert not result.is_sufficient
    assert "code_implementation" in result.missing_evidence_types


def test_actual_visual_reference_remains_unresolved():
    chunk = passage("target", "#4) Financial Analyst\nThe tech stack is depicted below.", 55)
    result = EvidenceSufficiencyGate().evaluate(
        "What is the tech stack of Financial Analyst?", QueryCategory.IMPLEMENTATION, [chunk])
    assert not result.is_sufficient
    assert "referenced_visual_content" in result.missing_evidence_types


def test_placeholder_never_counts_as_read_visual():
    reference = passage("ref", "Five techniques are depicted below.", 55)
    placeholder = passage("placeholder", "Original visual asset is present.", 55)
    placeholder.chunk.metadata.extra = {
        "visual_status": "ASSET_AVAILABLE", "is_visual_extraction": True,
        "visual_type": "diagram_architecture",
    }
    result = EvidenceSufficiencyGate().evaluate(
        "What are the five techniques?", QueryCategory.FACTUAL, [reference, placeholder])
    assert "referenced_visual_content" in result.missing_evidence_types


@pytest.mark.parametrize("context_limit", [1, 10])
def test_pipeline_uses_readable_stack_when_reranker_selects_unrelated_visual(evidence, context_limit):
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.reranker = MagicMock()
    pipeline.reranker.rerank.return_value = evidence[:1]
    pipeline.compressor = ContextCompressor()
    pipeline.governing_clause_selector = GoverningClauseSelector()
    pipeline.evidence_gate = EvidenceSufficiencyGate()
    pipeline.vision_service = MagicMock()
    pipeline.docstore = {sc.chunk.id: sc.chunk for sc in evidence}
    query = "how can i make Financial Analyst what is the tech stack of it"
    ctx = QueryContext(user_query=query, candidate_chunks=evidence,
        thinking_sm=MagicMock(), response_mode_config=get_response_mode_config("detailed"),
        rewrite_res=SimpleNamespace(rewritten_query=query),
        classification=SimpleNamespace(category=QueryCategory.IMPLEMENTATION),
        current_strategy=SimpleNamespace(rerank_top_n=context_limit, min_score_ratio=0.5,
                                         enable_parent_expansion=False, temperature=0.1))
    pipeline._stage_rerank_and_context(ctx, "")
    assert ctx.expanded_chunks[0].chunk.id == "target"
    assert ctx.telemetry_extra["evidence_sufficiency_passed"]
    assert not ctx.telemetry_extra.get("requires_visual_abstention")
    pipeline.vision_service.process_pdf_page_visuals.assert_not_called()
    ctx.req_llm = MagicMock()
    ctx.req_llm.complete.return_value = "Use CrewAI, Ollama with DeepSeek-R1, and Cursor [Source 1]."
    pipeline._stage_generate(ctx, "")
    ctx.req_llm.complete.assert_called_once()
    assert "DeepSeek-R1" in ctx.answer_text
    assert "CrewAI for multi-agent orchestration" in ctx.req_llm.complete.call_args.args[0]
    citations = CitationEngine().select_citations(ctx.answer_text, ctx.expanded_chunks)
    assert citations[0].chunk_id == "target"
    assert citations[0].display_page == "54"


def test_prioritization_preserves_reranker_score(evidence):
    target = evidence[-1].model_copy(update={"rerank_score": 4.2, "score": 4.2})
    result = prioritize_named_sections("Financial Analyst tech stack", [target], evidence)
    assert result[0].rerank_score == 4.2


def test_named_section_prefers_substantive_heading_over_empty_page_header():
    header = passage("header", "DailyDoseofDS.com", 68, "Multi-agent Hotel Finder")
    content = passage(
        "content",
        "#6) Multi-agent Hotel Finder\nTech stack: CrewAI and Ollama.\nWorkflow: Parse and search.",
        68,
        "Multi-agent Hotel Finder",
    )
    matches = named_section_matches("how can i make Multi-agent Hotel Finder", [header, content])
    assert [sc.chunk.id for sc in matches] == ["content", "header"]


def test_followup_uses_resolved_named_section_for_grounding_and_generation():
    unrelated = passage("unrelated", "Several memory techniques are depicted below.", 14, "Memory")
    target = passage(
        "hotel",
        "#6) Multi-agent Hotel Finder\n"
        "Build an Agentic workflow that parses a travel query, fetches live flights and hotel data.\n"
        "Tech stack:\n- CrewAI for multi-agent orchestration\n"
        "- Browserbase headless browser tool\n- Ollama to locally serve DeepSeek-R1\n"
        "Workflow:\n- Parse the query and create a Kayak URL\n"
        "- Extract the top flights and summarize hotel information.",
        68,
        "Multi-agent Hotel Finder",
    )
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.reranker = MagicMock()
    pipeline.reranker.rerank.return_value = [unrelated]
    pipeline.compressor = ContextCompressor()
    pipeline.governing_clause_selector = GoverningClauseSelector()
    pipeline.evidence_gate = EvidenceSufficiencyGate()
    pipeline.vision_service = MagicMock()
    pipeline.docstore = {sc.chunk.id: sc.chunk for sc in [unrelated, target]}
    resolved = "how can i make Multi-agent Hotel Finder"
    ctx = QueryContext(
        user_query="try again",
        effective_search_query=resolved,
        candidate_chunks=[unrelated, target],
        thinking_sm=MagicMock(),
        response_mode_config=get_response_mode_config("detailed"),
        rewrite_res=SimpleNamespace(rewritten_query=resolved),
        classification=SimpleNamespace(category=QueryCategory.FACTUAL),
        current_strategy=SimpleNamespace(
            rerank_top_n=10,
            min_score_ratio=0.5,
            enable_parent_expansion=False,
            temperature=0.1,
        ),
    )

    pipeline._stage_rerank_and_context(ctx, "")

    assert ctx.expanded_chunks[0].chunk.id == "hotel"
    assert ctx.telemetry_extra["evidence_sufficiency_passed"]
    assert not ctx.telemetry_extra.get("requires_visual_abstention")
    pipeline.vision_service.process_pdf_page_visuals.assert_not_called()

    ctx.req_llm = MagicMock()
    ctx.req_llm.complete.return_value = "Use CrewAI, Browserbase, and Ollama [Source 1]."
    pipeline._stage_generate(ctx, "")
    generated_prompt = ctx.req_llm.complete.call_args.args[0]
    assert "how can i make Multi-agent Hotel Finder" in generated_prompt
    assert "Query:\ntry again" not in generated_prompt


def test_try_again_reuses_previous_standalone_question():
    resolver = ConversationResolver()
    state = ConversationRAGState(
        conversation_id="hotel",
        active_topic="Multi-agent Hotel Finder",
        last_user_query="how can i make Multi-agent Hotel Finder",
        last_resolved_query="how can i make Multi-agent Hotel Finder",
    )
    result = resolver.resolve("try again", state, intent=QueryCategory.FACTUAL)
    assert result.is_followup
    assert result.resolved_query == "how can i make Multi-agent Hotel Finder"
