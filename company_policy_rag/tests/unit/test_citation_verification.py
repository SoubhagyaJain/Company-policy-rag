"""
Empirical test suite for citation verification and parsing.

Tests:
1. Citation parsing regex ([Source N] extraction and validation).
2. Chunk ID mapping, page number attribution, and snippet/quote verification.
3. Robustness against invalid, malformed, or out-of-range source markers.
"""

from __future__ import annotations

import pytest
from backend.models.chunk import Chunk, ChunkMetadata
from backend.models.rag import ScoredChunk
from backend.rag.citations import CitationEngine
from src.citations import (
    extract_cited_source_indices,
    select_citations_for_answer,
)
from llama_index.core.schema import NodeWithScore, TextNode


# ============================================================================
# Helper Fixtures & Factories
# ============================================================================

def make_scored_chunk(
    chunk_id: str = "chunk-101",
    doc_id: str = "doc-001",
    source_file: str = "employee_handbook.pdf",
    page_number: int | None = 12,
    section_title: str | None = "Vacation Policy",
    section_path: str | None = "Benefits > Vacation Policy",
    text: str = "Full-time employees receive 15 days of paid vacation annually.",
    score: float = 0.85,
    rerank_score: float | None = 0.92,
    rank: int = 1,
) -> ScoredChunk:
    metadata = ChunkMetadata(
        document_id=doc_id,
        source_file=source_file,
        file_path=f"/data/policies/{source_file}",
        file_hash="hash_abc123",
        document_type="pdf",
        chunk_strategy="recursive",
        page_number=page_number,
        section_title=section_title,
        section_path=section_path,
        chunk_index=0,
    )
    chunk = Chunk(
        id=chunk_id,
        text=text,
        metadata=metadata,
    )
    return ScoredChunk(
        chunk=chunk,
        score=score,
        rerank_score=rerank_score,
        rank=rank,
    )


def make_llama_node(
    section: str = "Benefits > Vacation Policy",
    score: float = 0.85,
    page: int | None = 12,
    file_name: str = "employee_handbook.pdf",
    text: str = "Full-time employees receive 15 days of paid vacation annually.",
) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(
            text=text,
            metadata={
                "section_path": section,
                "section_title": section.split(">")[-1].strip(),
                "page_number": page,
                "source_file": file_name,
                "document_id": "doc-001",
            },
        ),
        score=score,
    )


# ============================================================================
# 1. Citation Parsing Regex Tests
# ============================================================================

class TestCitationParsingRegex:
    """Test suite for [Source N] extraction and regex validation."""

    def test_single_source_tag_extraction(self) -> None:
        text = "Employees are entitled to 15 days of leave [Source 1]."
        tags = CitationEngine.extract_source_tags(text)
        assert tags == {1}

    def test_multiple_separate_source_tags(self) -> None:
        text = "Leave policy is stated in [Source 1], while holiday schedules are in [Source 3]."
        tags = CitationEngine.extract_source_tags(text)
        assert tags == {1, 3}

    def test_multi_index_in_single_tag(self) -> None:
        text = "For details, consult [Source 1, Source 2] and [Source 3, 4]."
        tags = CitationEngine.extract_source_tags(text)
        assert tags == {1, 2, 3, 4}

    def test_case_insensitivity(self) -> None:
        text = "Refer to [source 1], [SOURCE 2], and [SoUrCe 3]."
        tags = CitationEngine.extract_source_tags(text)
        assert tags == {1, 2, 3}

    def test_whitespace_variations(self) -> None:
        text = "See [Source 1], [Source   2], and [Source\t3]."
        tags = CitationEngine.extract_source_tags(text)
        assert tags == {1, 2, 3}

    def test_duplicate_indices_deduplicated(self) -> None:
        text = "First mention [Source 1], second mention [Source 1]."
        tags = CitationEngine.extract_source_tags(text)
        assert tags == {1}

    def test_zero_padded_number_parsing(self) -> None:
        text = "As outlined in [Source 01] and [Source 005]."
        tags = CitationEngine.extract_source_tags(text)
        assert tags == {1, 5}

    def test_non_matching_syntax_returns_empty(self) -> None:
        texts = [
            "No source tags here.",
            "Using (Source 1) instead of brackets.",
            "Using [Ref 1] tag.",
            "[Source]",
            "[Source abc]",
        ]
        for t in texts:
            tags = CitationEngine.extract_source_tags(t)
            assert tags == set(), f"Failed for text: {t}"

    def test_src_module_extract_cited_source_indices(self) -> None:
        text = "Check policy [Source 2] and [Source 5]."
        indices = extract_cited_source_indices(text)
        assert indices == {2, 5}


# ============================================================================
# 2. Chunk ID Mapping, Page Attribution, and Snippet Verification Tests
# ============================================================================

class TestChunkMappingAndAttribution:
    """Test suite for metadata mapping, page attribution, and verbatim quotes."""

    def test_exact_chunk_metadata_mapping(self) -> None:
        engine = CitationEngine()
        chunks = [
            make_scored_chunk(
                chunk_id="chk-alpha",
                doc_id="doc-100",
                source_file="handbook_v2.pdf",
                page_number=14,
                section_title="Sick Leave",
                section_path="HR > Leave > Sick Leave",
                text="Employees receive 10 paid sick days per year.",
                rerank_score=0.95,
            ),
            make_scored_chunk(
                chunk_id="chk-beta",
                doc_id="doc-101",
                source_file="remote_policy.pdf",
                page_number=3,
                section_title="Remote Work",
                section_path="IT > Remote Work",
                text="Remote work must be pre-approved by the manager.",
                rerank_score=0.80,
            ),
        ]

        answer = "Paid sick leave is 10 days [Source 1]."
        citations = engine.select_citations(answer, chunks)

        assert len(citations) == 1
        c = citations[0]
        assert c.source_index == 1
        assert c.chunk_id == "chk-alpha"
        assert c.document_id == "doc-100"
        assert c.source_file == "handbook_v2.pdf"
        assert c.page_number == 14
        assert c.section_title == "Sick Leave"
        assert c.section_path == "HR > Leave > Sick Leave"
        assert c.snippet == "Employees receive 10 paid sick days per year."
        assert c.relevance_score == 0.95
        assert c.selection_reason == "cited_in_answer"

    def test_page_number_none_attribution(self) -> None:
        engine = CitationEngine()
        chunks = [
            make_scored_chunk(
                chunk_id="chk-nopage",
                page_number=None,
                text="Policy without page number.",
            )
        ]
        answer = "Here is the policy [Source 1]."
        citations = engine.select_citations(answer, chunks)

        assert len(citations) == 1
        assert citations[0].page_number is None

    def test_verbatim_quote_snippet_truncation(self) -> None:
        engine = CitationEngine()
        long_text = "Word " * 60  # 300 characters
        chunks = [make_scored_chunk(text=long_text)]

        answer = "Statement supported by [Source 1]."
        citations = engine.select_citations(answer, chunks)

        assert len(citations) == 1
        snippet = citations[0].snippet
        assert snippet.endswith("...")
        assert len(snippet) <= 254
        assert snippet.startswith("Word Word Word")

    def test_short_snippet_not_truncated(self) -> None:
        engine = CitationEngine()
        short_text = "Short text."
        chunks = [make_scored_chunk(text=short_text)]

        answer = "Statement [Source 1]."
        citations = engine.select_citations(answer, chunks)

        assert len(citations) == 1
        assert citations[0].snippet == "Short text."
        assert not citations[0].snippet.endswith("...")

    def test_sorted_citation_output_order(self) -> None:
        engine = CitationEngine()
        chunks = [
            make_scored_chunk(chunk_id="c1", text="Chunk 1 text"),
            make_scored_chunk(chunk_id="c2", text="Chunk 2 text"),
            make_scored_chunk(chunk_id="c3", text="Chunk 3 text"),
        ]
        # LLM cites in reverse order [Source 3, Source 1]
        answer = "Details in [Source 3] and also [Source 1]."
        citations = engine.select_citations(answer, chunks)

        assert len(citations) == 2
        assert citations[0].source_index == 1
        assert citations[0].chunk_id == "c1"
        assert citations[1].source_index == 3
        assert citations[1].chunk_id == "c3"


# ============================================================================
# 3. Robustness Against Invalid & Out-of-Range Source Markers
# ============================================================================

class TestOutofRangeAndInvalidMarkers:
    """Test suite for edge cases when LLM outputs invalid source tags."""

    def test_out_of_range_high_index_fallback(self) -> None:
        engine = CitationEngine()
        chunks = [
            make_scored_chunk(chunk_id="c1", score=0.9),
            make_scored_chunk(chunk_id="c2", score=0.8),
        ]
        # LLM hallucinates [Source 99] when only 2 chunks exist
        answer = "Fact cited from non-existent chunk [Source 99]."
        citations = engine.select_citations(answer, chunks)

        # Should NOT crash with IndexError.
        # Since [Source 99] is invalid, selection falls back to score threshold fallback.
        assert len(citations) >= 1
        for c in citations:
            assert 1 <= c.source_index <= 2
            assert c.selection_reason == "score_threshold_fallback"

    def test_out_of_range_zero_or_negative_index(self) -> None:
        engine = CitationEngine()
        chunks = [make_scored_chunk(chunk_id="c1", score=0.9)]
        answer = "Fact cited from [Source 0]."
        citations = engine.select_citations(answer, chunks)

        # 0 is out of bounds (1-based index). Should fallback cleanly.
        assert len(citations) == 1
        assert citations[0].chunk_id == "c1"
        assert citations[0].selection_reason == "score_threshold_fallback"

    def test_mixed_valid_and_invalid_indices(self) -> None:
        engine = CitationEngine()
        chunks = [
            make_scored_chunk(chunk_id="c1", score=0.9),
            make_scored_chunk(chunk_id="c2", score=0.7),
        ]
        # LLM cites valid [Source 1] and invalid [Source 50]
        answer = "Valid info [Source 1] and hallucinated info [Source 50]."
        citations = engine.select_citations(answer, chunks)

        # Should retain valid Source 1 and ignore Source 50 without crashing
        assert len(citations) == 1
        assert citations[0].source_index == 1
        assert citations[0].chunk_id == "c1"
        assert citations[0].selection_reason == "cited_in_answer"

    def test_empty_chunks_list(self) -> None:
        engine = CitationEngine()
        answer = "Answer with citation [Source 1]."
        citations = engine.select_citations(answer, [])
        assert citations == []

    def test_no_tags_in_answer_score_fallback(self) -> None:
        engine = CitationEngine()
        chunks = [
            make_scored_chunk(chunk_id="c1", score=0.9, rerank_score=0.95),
            make_scored_chunk(chunk_id="c2", score=0.3, rerank_score=0.20),
        ]
        answer = "Answer with no citation tags at all."
        citations = engine.select_citations(answer, chunks)

        assert len(citations) >= 1
        assert citations[0].chunk_id == "c1"
        assert citations[0].selection_reason == "score_threshold_fallback"
        # Low scoring chunk (c2) should be filtered out by 0.45 threshold
        assert all(c.chunk_id != "c2" for c in citations)

    def test_llama_legacy_citations_out_of_range(self) -> None:
        nodes = [
            make_llama_node(section="Section A", score=9.0),
            make_llama_node(section="Section B", score=8.0),
        ]
        answer = "Answer citing [Source 100]."
        citations = select_citations_for_answer(answer, nodes)

        assert len(citations) >= 1
        assert citations[0]["selection_reason"] == "score_threshold_fallback"
