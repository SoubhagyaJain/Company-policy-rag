"""The policy decision block is only spent on workplace-policy questions."""

from __future__ import annotations

import pytest

from backend.models.chunk import Chunk, ChunkMetadata
from backend.models.rag import ScoredChunk
from backend.rag.pipeline import RAGPipeline
from backend.rag.policy_reliability import (
    MAX_PROMPT_RULES,
    GoverningClauseSelector,
    bind_source_indices,
    format_policy_decision_context,
    is_policy_question,
)
from src.config import settings

POLICY_HEADER = "POLICY DECISION SUPPORT"


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        id=cid,
        text=text,
        metadata=ChunkMetadata(document_id="doc_aaaaaaaaaaaa", source_file="guide.pdf", page_number=1),
    )


class _Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def retrieve(self, query, dense_top_k=8, bm25_top_k=8, filters=None, rrf_k=60):
        return [ScoredChunk(chunk=c, score=1.0 - i * 0.1) for i, c in enumerate(self.chunks)]


@pytest.fixture
def quiet(monkeypatch):
    for key, value in {
        "retrieval_cache_enabled": False,
        "vision_enabled": False,
        "enable_lazy_vision_fallback": False,
        "enable_llm_multi_query": False,
        "enable_reranker": False,
    }.items():
        monkeypatch.setattr(settings, key, value)


def _pipeline(chunks: list[Chunk]) -> RAGPipeline:
    return RAGPipeline(
        hybrid_retriever=_Retriever(chunks),
        docstore={c.id: c for c in chunks},
        llm=None,
    )


@pytest.mark.parametrize(
    "query, expected",
    [
        ("How can I build a custom tool for an agent?", False),
        ("Explain the ReAct design pattern.", False),
        ("What does Article 12 define as the State?", False),
        ("Can employees work remotely on Fridays?", True),
        ("Must I disclose prescription medication?", True),
        ("If I finish a callout at 2:30am, when can I start work?", True),
    ],
)
def test_policy_question_classifier(query: str, expected: bool) -> None:
    assert is_policy_question(query) is expected


def test_guidebook_question_prompt_has_no_policy_block(quiet) -> None:
    chunks = [
        _chunk("tools", "Tools must be registered before the agent can call them. If a tool fails, the agent retries."),
        _chunk("memory", "Memory stores context between turns."),
    ]
    ctx = _pipeline(chunks).run_retrieval_stages("How can I build a custom tool for an agent?")

    assert POLICY_HEADER not in ctx.formatted_context
    assert ctx.retrieval_stages["policy_block_tokens"] == 0


def test_policy_question_prompt_keeps_the_block_and_counts_its_tokens(quiet) -> None:
    chunks = [
        _chunk("remote", "Employees may work remotely up to two days per week with manager approval."),
        _chunk("equipment", "Company laptops must use the VPN when employees work remotely."),
    ]
    ctx = _pipeline(chunks).run_retrieval_stages("Can employees work remotely on Fridays?")

    assert ctx.formatted_context.startswith(POLICY_HEADER)
    block_tokens = ctx.retrieval_stages["policy_block_tokens"]
    assert block_tokens > 0
    assert ctx.context_tokens >= block_tokens
    assert "QUERY FACTS" not in ctx.formatted_context


def test_prompt_rules_are_capped_with_primary_rules_first() -> None:
    sentences = " ".join(f"Employees must follow rule {i} when on shift." for i in range(15))
    chunk = ScoredChunk(chunk=_chunk("rules", sentences), score=1.0)
    selection = GoverningClauseSelector().select("Must employees follow the shift rules?", [chunk])
    bind_source_indices(selection, [chunk])
    assert len(selection.structured_rules) > MAX_PROMPT_RULES

    block = format_policy_decision_context(selection)

    rule_lines = [line for line in block.splitlines() if line.startswith("- primary_rule") or line.startswith("- supporting_rule") or line.startswith("- exception") or line.startswith("- definition")]
    assert len(rule_lines) == MAX_PROMPT_RULES
    assert rule_lines[0].startswith("- primary_rule")
