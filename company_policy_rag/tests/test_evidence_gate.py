"""
Unit Tests for Pre-Generation Evidence Sufficiency Gate.
"""

from __future__ import annotations

import pytest

from backend.models.chunk import Chunk, ChunkMetadata, ContentType
from backend.models.rag import QueryCategory, ScoredChunk
from backend.rag.evidence_gate import EvidenceSufficiencyGate


def _make_sc(
    chunk_id: str,
    text: str,
    page_number: int = 1,
    content_type: ContentType = ContentType.PROSE,
    section_title: str = "General",
) -> ScoredChunk:
    chunk = Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            document_id="doc_1",
            source_file="guide.pdf",
            file_path="data/guide.pdf",
            file_hash="hash_doc_1",
            document_type="pdf",
            chunk_strategy="adaptive",
            page_number=page_number,
            section_title=section_title,
            content_type=content_type,
        ),
    )
    return ScoredChunk(chunk=chunk, score=0.90)


def test_evidence_gate_fails_when_implementation_code_is_missing():
    gate = EvidenceSufficiencyGate()
    query = "How can I make X Writer Agent?"
    intent = QueryCategory.IMPLEMENTATION

    # Only prose text with continuation cue, no actual code
    c1 = _make_sc(
        "c1",
        "The Agent takes the output of the X Analyst agent and generates insights. Here's the code:",
        page_number=63,
        content_type=ContentType.PROSE,
        section_title="X Writer Agent",
    )

    res = gate.evaluate(query, intent, [c1])
    assert res.is_sufficient is False
    assert "code_implementation" in res.missing_evidence_types
    assert 64 in res.pages_to_inspect  # Must inspect Page 64!
    assert len(res.detected_continuation_cues) > 0


def test_evidence_gate_passes_when_implementation_code_is_present():
    gate = EvidenceSufficiencyGate()
    query = "How can I make X Writer Agent?"
    intent = QueryCategory.IMPLEMENTATION

    c1 = _make_sc(
        "c1",
        "X Writer Agent overview.",
        page_number=63,
        content_type=ContentType.PROSE,
    )
    c2 = _make_sc(
        "c2",
        "```python\nwriter_agent = Agent(role='Writer')\nwriter_task = Task(description='Write')\n```",
        page_number=64,
        content_type=ContentType.CODE,
    )

    res = gate.evaluate(query, intent, [c1, c2])
    assert res.is_sufficient is True
    assert len(res.missing_evidence_types) == 0


def test_evidence_gate_factual_query_does_not_require_code():
    gate = EvidenceSufficiencyGate()
    query = "What is the standard vacation policy for new hires?"
    intent = QueryCategory.FACTUAL

    c1 = _make_sc(
        "c1",
        "Employees accrue 15 days of vacation per year.",
        page_number=10,
        content_type=ContentType.PROSE,
    )

    res = gate.evaluate(query, intent, [c1])
    assert res.is_sufficient is True
