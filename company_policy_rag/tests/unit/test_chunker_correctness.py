"""Empirical verification tests for document chunking strategies, metadata preservation,
chunk overlap accuracy, boundary preservation, and adaptive chunker selection logic.

Created by Challenger M1-2.
"""

from __future__ import annotations

import pytest
from typing import List

from backend.models.document import DocumentCategory, DocumentMetadata, DocumentType, RawDocument
from backend.models.chunk import Chunk, ChunkMetadata, ChunkRole, ContentType

from backend.ingestion.chunkers.base import BaseChunker
from backend.ingestion.chunkers.recursive import RecursiveChunker
from backend.ingestion.chunkers.semantic import SemanticChunker
from backend.ingestion.chunkers.markdown_aware import MarkdownAwareChunker
from backend.ingestion.chunkers.heading_aware import HeadingAwareChunker
from backend.ingestion.chunkers.table_aware import TableAwareChunker
from backend.ingestion.chunkers.adaptive_chunker import AdaptiveChunker


# =============================================================================
# 1. METADATA PRESERVATION TESTS
# =============================================================================

class TestMetadataPreservation:
    """Verifies metadata (page numbers, section paths, table metadata, headings, extra)
    is correctly preserved across chunk boundaries.
    """

    def test_document_field_propagation_across_chunkers(self):
        """Verify core document fields (id, source_file, file_path, file_hash, doc_type, page_number, category, extra)
        propagate to all generated chunks across all 5 chunker strategies.
        """
        doc = RawDocument(
            id="doc_test_123",
            content="Line 1 text.\nLine 2 text.\nLine 3 text.\nLine 4 text.\nLine 5 text.",
            metadata=DocumentMetadata(
                source_file="handbook_2026.pdf",
                file_path="/docs/handbook_2026.pdf",
                file_hash="a1b2c3d4e5f67890",
                document_type=DocumentType.PDF,
                category="policy",
                page_number=12,
                total_pages=50,
                extra={"department": "HR", "confidential": True},
            ),
        )

        chunkers: List[BaseChunker] = [
            RecursiveChunker(chunk_size=20, chunk_overlap=5),
            SemanticChunker(chunk_size=20, chunk_overlap=5),
            HeadingAwareChunker(),
            MarkdownAwareChunker(),
            TableAwareChunker(),
        ]

        for chunker in chunkers:
            chunks = chunker.chunk([doc])
            assert len(chunks) > 0, f"Chunker {chunker.__class__.__name__} produced 0 chunks"
            for chunk in chunks:
                meta = chunk.metadata
                assert meta.document_id == "doc_test_123"
                assert meta.source_file == "handbook_2026.pdf"
                assert meta.file_path == "/docs/handbook_2026.pdf"
                assert meta.file_hash == "a1b2c3d4e5f67890"
                assert meta.document_type == "pdf"
                assert meta.category == "policy"
                assert meta.page_number == 12
                assert meta.extra.get("department") == "HR"
                assert meta.extra.get("confidential") is True

    def test_heading_aware_hierarchical_section_path_preservation(self):
        """Verify HeadingAwareChunker correctly parses hierarchical headings and
        attaches section_title, section_number, section_path, and section_level to chunks.
        """
        content = (
            "I. OVERVIEW\n"
            "This section introduces company policies.\n\n"
            "A. Scope and Applicability\n"
            "This policy applies to all global employees.\n\n"
            "1. Full-time Employees\n"
            "Full-time staff receive full benefits package.\n\n"
            "2. Part-time Employees\n"
            "Part-time staff receive prorated benefits.\n\n"
            "B. Enforcement\n"
            "Violations may result in disciplinary action."
        )

        doc = RawDocument(
            content=content,
            metadata=DocumentMetadata(
                source_file="compliance.txt",
                file_path="compliance.txt",
                file_hash="hash123456789000",
                document_type=DocumentType.TXT,
            ),
        )

        chunker = HeadingAwareChunker()
        chunks = chunker.chunk([doc])

        assert len(chunks) >= 4, f"Expected at least 4 chunks, got {len(chunks)}"

        # Check section paths
        paths = [c.metadata.section_path for c in chunks]
        titles = [c.metadata.section_title for c in chunks]

        assert "OVERVIEW" in titles
        assert "Scope and Applicability" in titles
        assert "Full-time Employees" in titles

        # Verify hierarchical path building (stack pop/push)
        full_time_chunk = next((c for c in chunks if c.metadata.section_title == "Full-time Employees"), None)
        assert full_time_chunk is not None
        assert full_time_chunk.metadata.section_path is not None
        assert "OVERVIEW" in full_time_chunk.metadata.section_path or "Scope and Applicability" in full_time_chunk.metadata.section_path

        # Verify stack pop when returning to B. Enforcement
        enforcement_chunk = next((c for c in chunks if c.metadata.section_title == "Enforcement"), None)
        assert enforcement_chunk is not None
        assert enforcement_chunk.metadata.section_path is not None
        assert "1 Full-time Employees" not in enforcement_chunk.metadata.section_path

    def test_markdown_aware_section_and_code_metadata(self):
        """Verify MarkdownAwareChunker attaches section context to prose and code chunks."""
        content = (
            "# Security Guidelines\n\n"
            "General security rules for development.\n\n"
            "## API Key Handling\n\n"
            "Store secrets safely in environment variables.\n\n"
            "```python\n"
            "import os\n"
            "api_key = os.getenv('API_KEY')\n"
            "```\n\n"
            "Never commit secrets to git repositories."
        )

        doc = RawDocument(
            content=content,
            metadata=DocumentMetadata(
                source_file="security.md",
                file_path="security.md",
                file_hash="mdhash1234567890",
                document_type=DocumentType.MARKDOWN,
                has_code=True,
            ),
        )

        chunker = MarkdownAwareChunker()
        chunks = chunker.chunk([doc])

        code_chunks = [c for c in chunks if c.metadata.content_type == ContentType.CODE]
        assert len(code_chunks) == 1
        assert code_chunks[0].metadata.is_atomic is True
        assert code_chunks[0].metadata.section_title == "API Key Handling"
        assert code_chunks[0].metadata.section_path is not None
        assert "Security Guidelines" in code_chunks[0].metadata.section_path

    def test_table_aware_metadata_and_header_row_propagation(self):
        """Verify TableAwareChunker preserves content_type=TABLE, is_atomic flag, and
        prepends table header rows when table is split across sub-chunks.
        """
        table_lines = [
            "| Employee ID | Name | Department | Salary Grade | Status |",
            "|---|---|---|---|---|",
        ]
        for i in range(1, 15):
            table_lines.append(f"| EMP-{i:03d} | Employee Name {i} | Dept-{i % 3} | Grade {i} | Active |")

        content = "Intro prose before table.\n\n" + "\n".join(table_lines) + "\n\nOutro prose after table."

        doc = RawDocument(
            content=content,
            metadata=DocumentMetadata(
                source_file="salaries.md",
                file_path="salaries.md",
                file_hash="tablehash1234567",
                document_type=DocumentType.MARKDOWN,
                has_tables=True,
            ),
        )

        # Set small chunk_size to force table split
        chunker = TableAwareChunker(chunk_size=40)
        chunks = chunker.chunk([doc])

        table_chunks = [c for c in chunks if c.metadata.content_type == ContentType.TABLE]
        assert len(table_chunks) > 1, f"Expected multiple table sub-chunks, got {len(table_chunks)}"

        header_line_1 = "| Employee ID | Name | Department | Salary Grade | Status |"
        header_line_2 = "|---|---|---|---|---|"

        for idx, tc in enumerate(table_chunks):
            assert tc.metadata.is_atomic is False
            assert header_line_1 in tc.text, f"Sub-chunk {idx} missing table header line 1"
            assert header_line_2 in tc.text, f"Sub-chunk {idx} missing table separator header line 2"


# =============================================================================
# 2. CHUNK OVERLAP & BOUNDARY PRESERVATION TESTS
# =============================================================================

class TestChunkOverlapAndBoundaries:
    """Verifies chunk overlap accuracy, sentence/token boundaries, and code block / table row atomic integrity."""

    def test_recursive_chunker_overlap_accuracy(self):
        """Verify RecursiveChunker maintains text overlap between consecutive chunks."""
        text = " ".join([f"Word{i}" for i in range(200)])
        doc = RawDocument(
            content=text,
            metadata=DocumentMetadata(
                source_file="words.txt",
                file_path="words.txt",
                file_hash="wordshash123456",
                document_type=DocumentType.TXT,
            ),
        )

        chunk_size_tokens = 30
        chunk_overlap_tokens = 10
        chunker = RecursiveChunker(chunk_size=chunk_size_tokens, chunk_overlap=chunk_overlap_tokens)
        chunks = chunker.chunk([doc])

        assert len(chunks) > 1, "Expected multiple chunks for 200 words"

        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            chunk1_words = set(chunks[i].text.split())
            chunk2_words = set(chunks[i + 1].text.split())
            common = chunk1_words.intersection(chunk2_words)
            assert len(common) > 0, f"No overlap found between chunk {i} and chunk {i+1}"

    def test_semantic_chunker_sentence_overlap(self):
        """Verify SemanticChunker preserves sentence boundaries and carries over overlap sentences."""
        sentences = [
            f"Sentence number {i} provides information about policy step {i}."
            for i in range(1, 20)
        ]
        content = " ".join(sentences)

        doc = RawDocument(
            content=content,
            metadata=DocumentMetadata(
                source_file="sentences.txt",
                file_path="sentences.txt",
                file_hash="senthash1234567",
                document_type=DocumentType.TXT,
            ),
        )

        chunker = SemanticChunker(chunk_size=40, chunk_overlap=15)
        chunks = chunker.chunk([doc])

        assert len(chunks) > 1, "Expected multiple chunks"

        for i in range(len(chunks) - 1):
            c1_text = chunks[i].text
            c2_text = chunks[i + 1].text

            # Last sentence of c1 should be present at start of c2
            c1_sents = [s.strip() for s in c1_text.split(".") if s.strip()]
            c2_sents = [s.strip() for s in c2_text.split(".") if s.strip()]

            last_sent_c1 = c1_sents[-1]
            assert last_sent_c1 in c2_sents, (
                f"Overlap assertion failed: last sentence of chunk {i} ('{last_sent_c1}') "
                f"not found in chunk {i+1} sentences ({c2_sents})"
            )

    def test_code_block_atomic_boundary_preservation(self):
        """Verify MarkdownAwareChunker preserves code blocks atomically without cutting inside fence."""
        code_block = (
            "```python\n"
            "def complex_algorithm(data):\n"
            "    # Line 1\n"
            "    result = [x * 2 for x in data]\n"
            "    # Line 2\n"
            "    return result\n"
            "```"
        )
        content = f"Before code block.\n\n{code_block}\n\nAfter code block."

        doc = RawDocument(
            content=content,
            metadata=DocumentMetadata(
                source_file="code_test.md",
                file_path="code_test.md",
                file_hash="codehash1234567",
                document_type=DocumentType.MARKDOWN,
                has_code=True,
            ),
        )

        chunker = MarkdownAwareChunker()
        chunks = chunker.chunk([doc])

        code_chunks = [c for c in chunks if c.metadata.content_type == ContentType.CODE]
        assert len(code_chunks) == 1
        assert code_chunks[0].text.startswith("```")
        assert code_chunks[0].text.endswith("```")
        assert "complex_algorithm" in code_chunks[0].text
        assert code_chunks[0].metadata.is_atomic is True

    def test_table_row_boundary_preservation(self):
        """Verify TableAwareChunker splits tables cleanly along newline row boundaries, never truncating mid-row."""
        table_content = (
            "| Col A | Col B | Col C |\n"
            "|---|---|---|\n"
            "| Row 1 Data A | Row 1 Data B | Row 1 Data C |\n"
            "| Row 2 Data A | Row 2 Data B | Row 2 Data C |\n"
            "| Row 3 Data A | Row 3 Data B | Row 3 Data C |\n"
            "| Row 4 Data A | Row 4 Data B | Row 4 Data C |"
        )

        doc = RawDocument(
            content=table_content,
            metadata=DocumentMetadata(
                source_file="table_rows.md",
                file_path="table_rows.md",
                file_hash="tablehash999999",
                document_type=DocumentType.MARKDOWN,
                has_tables=True,
            ),
        )

        chunker = TableAwareChunker(chunk_size=30)
        chunks = chunker.chunk([doc])

        for c in chunks:
            if c.metadata.content_type == ContentType.TABLE:
                lines = c.text.split("\n")
                for line in lines:
                    assert line.startswith("|") and line.endswith("|"), (
                        f"Malformed row boundary in chunk: {line}"
                    )


# =============================================================================
# 3. ADAPTIVE CHUNKER SELECTION LOGIC TESTS
# =============================================================================

class TestAdaptiveChunkerSelection:
    """Verifies AdaptiveChunker inspects document features (tables, code, sections, length)
    and selects the correct underlying chunking strategy.
    """

    def test_table_detection_routes_to_table_chunker(self):
        """Verify document with has_tables=True or markdown pipes routes to TableAwareChunker."""
        doc1 = RawDocument(
            content="Plain text with table metadata",
            metadata=DocumentMetadata(
                source_file="data.csv",
                file_path="data.csv",
                file_hash="hash111",
                document_type=DocumentType.CSV,
                has_tables=True,
            ),
        )
        doc2 = RawDocument(
            content="Header\n| A | B |\n|---|---|\n| 1 | 2 |",
            metadata=DocumentMetadata(
                source_file="report.txt",
                file_path="report.txt",
                file_hash="hash222",
                document_type=DocumentType.TXT,
                has_tables=False,
            ),
        )

        adaptive = AdaptiveChunker()
        c1 = adaptive.select_chunker(doc1)
        c2 = adaptive.select_chunker(doc2)

        assert isinstance(c1, TableAwareChunker)
        assert isinstance(c2, TableAwareChunker)

    def test_markdown_code_detection_routes_to_markdown_chunker(self):
        """Verify markdown doc type or has_code=True routes to MarkdownAwareChunker."""
        doc1 = RawDocument(
            content="# Title\nProse content",
            metadata=DocumentMetadata(
                source_file="notes.md",
                file_path="notes.md",
                file_hash="hash333",
                document_type=DocumentType.MARKDOWN,
            ),
        )
        doc2 = RawDocument(
            content="Some text\n```python\nprint(1)\n```",
            metadata=DocumentMetadata(
                source_file="script.txt",
                file_path="script.txt",
                file_hash="hash444",
                document_type=DocumentType.TXT,
                has_code=True,
            ),
        )

        adaptive = AdaptiveChunker()
        c1 = adaptive.select_chunker(doc1)
        c2 = adaptive.select_chunker(doc2)

        assert isinstance(c1, MarkdownAwareChunker)
        assert isinstance(c2, MarkdownAwareChunker)

    def test_heading_detection_routes_to_heading_chunker(self):
        """Verify section_path/section_title or PDF/DOCX doc type routes to HeadingAwareChunker."""
        doc_pdf = RawDocument(
            content="Page text from PDF",
            metadata=DocumentMetadata(
                source_file="handbook.pdf",
                file_path="handbook.pdf",
                file_hash="hash555",
                document_type=DocumentType.PDF,
            ),
        )
        doc_sec = RawDocument(
            content="I. INTRODUCTION\nWelcome to company.",
            metadata=DocumentMetadata(
                source_file="intro.txt",
                file_path="intro.txt",
                file_hash="hash666",
                document_type=DocumentType.TXT,
                section_title="INTRODUCTION",
                section_path="I. INTRODUCTION",
            ),
        )

        adaptive = AdaptiveChunker()
        c_pdf = adaptive.select_chunker(doc_pdf)
        c_sec = adaptive.select_chunker(doc_sec)

        assert isinstance(c_pdf, HeadingAwareChunker)
        assert isinstance(c_sec, HeadingAwareChunker)

    def test_length_threshold_semantic_vs_recursive(self):
        """Verify >200 words routes to SemanticChunker, <=200 words routes to RecursiveChunker."""
        short_doc = RawDocument(
            content="Word " * 50,
            metadata=DocumentMetadata(
                source_file="short.txt",
                file_path="short.txt",
                file_hash="hash777",
                document_type=DocumentType.TXT,
            ),
        )
        long_doc = RawDocument(
            content="Word " * 250,
            metadata=DocumentMetadata(
                source_file="long.txt",
                file_path="long.txt",
                file_hash="hash888",
                document_type=DocumentType.TXT,
            ),
        )

        adaptive = AdaptiveChunker()
        c_short = adaptive.select_chunker(short_doc)
        c_long = adaptive.select_chunker(long_doc)

        assert isinstance(c_short, RecursiveChunker)
        assert isinstance(c_long, SemanticChunker)

    def test_override_strategy_bypasses_autodetect(self):
        """Verify override_strategy explicitly selects chunker regardless of doc properties."""
        doc = RawDocument(
            content="| Col A | Col B |\n|---|---|\n| Val A | Val B |",
            metadata=DocumentMetadata(
                source_file="table.md",
                file_path="table.md",
                file_hash="hash999",
                document_type=DocumentType.MARKDOWN,
                has_tables=True,
            ),
        )

        adaptive_recursive = AdaptiveChunker(override_strategy="recursive")
        c_rec = adaptive_recursive.select_chunker(doc)
        assert isinstance(c_rec, RecursiveChunker)

        adaptive_semantic = AdaptiveChunker(override_strategy="semantic")
        c_sem = adaptive_semantic.select_chunker(doc)
        assert isinstance(c_sem, SemanticChunker)

    def test_batch_adaptive_chunking(self):
        """Verify batch chunking of diverse document types assigns correct chunk_strategy metadata."""
        docs = [
            RawDocument(
                content="| A | B |\n|---|---|\n| 1 | 2 |",
                metadata=DocumentMetadata(
                    source_file="tbl.md", file_path="tbl.md", file_hash="h1", document_type=DocumentType.MARKDOWN, has_tables=True
                ),
            ),
            RawDocument(
                content="```python\nx=10\n```",
                metadata=DocumentMetadata(
                    source_file="code.md", file_path="code.md", file_hash="h2", document_type=DocumentType.MARKDOWN, has_code=True
                ),
            ),
            RawDocument(
                content="Short plain text",
                metadata=DocumentMetadata(
                    source_file="short.txt", file_path="short.txt", file_hash="h3", document_type=DocumentType.TXT
                ),
            ),
        ]

        adaptive = AdaptiveChunker()
        chunks = adaptive.chunk(docs)

        strategies = [c.metadata.chunk_strategy for c in chunks]
        assert "table_aware" in strategies
        assert "markdown_aware" in strategies
        assert "recursive" in strategies


# =============================================================================
# 4. STRESS & CORNER CASE TESTS (ADVERSARIAL)
# =============================================================================

class TestChunkerEdgeCases:
    """Stress-tests edge cases: empty docs, oversized single sections, unicode text, malformed tables."""

    def test_empty_document_returns_empty_chunks(self):
        """Verify all chunkers handle empty or whitespace-only documents gracefully."""
        doc_empty = RawDocument(
            content="   \n\n  ",
            metadata=DocumentMetadata(
                source_file="empty.txt", file_path="empty.txt", file_hash="hempty", document_type=DocumentType.TXT
            ),
        )

        chunkers = [
            RecursiveChunker(),
            SemanticChunker(),
            HeadingAwareChunker(),
            MarkdownAwareChunker(),
            TableAwareChunker(),
            AdaptiveChunker(),
        ]

        for chunker in chunkers:
            chunks = chunker.chunk([doc_empty])
            assert len(chunks) == 0, f"{chunker.__class__.__name__} produced chunks for empty document"

    def test_chunk_indices_are_sequential_and_unique_ids(self):
        """Verify produced chunks have sequential 0..N indices and unique chunk IDs."""
        doc = RawDocument(
            content="Paragraph 1.\n\nParagraph 2.\n\nParagraph 3.\n\nParagraph 4.\n\nParagraph 5.",
            metadata=DocumentMetadata(
                source_file="seq.txt", file_path="seq.txt", file_hash="hseq", document_type=DocumentType.TXT
            ),
        )

        chunker = RecursiveChunker(chunk_size=10, chunk_overlap=2)
        chunks = chunker.chunk([doc])

        indices = [c.metadata.chunk_index for c in chunks]
        ids = [c.id for c in chunks]

        assert indices == list(range(len(chunks))), f"Chunk indices not sequential: {indices}"
        assert len(set(ids)) == len(ids), f"Duplicate chunk IDs found: {ids}"

    def test_oversized_single_heading_section_handling(self):
        """Test HeadingAwareChunker on a section that vastly exceeds chunk_size budget."""
        large_section_text = "I. MASSIVE SECTION\n" + ("Sentence in section. " * 300)
        doc = RawDocument(
            content=large_section_text,
            metadata=DocumentMetadata(
                source_file="massive.txt", file_path="massive.txt", file_hash="hmassive", document_type=DocumentType.TXT
            ),
        )

        chunker = HeadingAwareChunker(chunk_size=100)
        chunks = chunker.chunk([doc])

        assert len(chunks) > 0
        assert chunks[0].metadata.section_title == "MASSIVE SECTION"
