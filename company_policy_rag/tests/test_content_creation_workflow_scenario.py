import pytest
from unittest.mock import MagicMock
from backend.models.chunk import Chunk, ChunkMetadata, ContentType
from backend.models.rag import Citation, QueryCategory, ScoredChunk
from backend.rag.citations import CitationEngine
from backend.rag.context_compression import ContextCompressor
from backend.rag.pipeline import RAGPipeline


def test_content_creation_workflow_context_and_citation_generation():
    # 1. Simulate visual chunk extracted for #10 Multi-agent Content Creation System
    meta = ChunkMetadata(
        document_id="doc_guidebook_1",
        source_file="AI Agents guidebook.pdf",
        file_path="uploads/AI Agents guidebook.pdf",
        file_hash="hash_guidebook",
        document_type="pdf",
        chunk_strategy="recursive",
        internal_page_index=98,
        page_number=99,
        display_page_number=98,
        page_label="98",
        section_title="#10) Multi-agent Content Creation System",
        visual_asset_ids=["ast_workflow_98"],
        image_assets=[
            {
                "asset_id": "ast_workflow_98",
                "asset_url": "/api/documents/doc_guidebook_1/visual-assets/ast_workflow_98",
                "image_hash": "hash_workflow_98",
                "page_number": 99,
                "display_page_number": 98,
                "page_label": "98",
                "visual_type": "diagram_architecture",
            }
        ],
        extra={
            "is_visual_extraction": True,
            "visual_type": "diagram_architecture",
            "asset_id": "ast_workflow_98",
            "image_url": "/api/documents/doc_guidebook_1/visual-assets/ast_workflow_98",
        },
    )

    visual_chunk = Chunk(
        id="chk_workflow_98",
        text=(
            "The workflow consists of: 1. User submits URL. "
            "2. Scraper agent extracts web content via Firecrawl. "
            "3. Writer agent generates social media posts using local LLM."
        ),
        metadata=meta,
    )

    scored_chunk = ScoredChunk(chunk=visual_chunk, score=0.97, rerank_score=0.97)

    # 2. Context Compressor formatting
    compressor = ContextCompressor()
    formatted = compressor.format_context_for_prompt([scored_chunk])

    # Must contain [VISUAL SOURCE 1] and Page: 98 (never physical 99 alone or 0-indexed 98)
    assert "[VISUAL SOURCE 1]" in formatted
    assert "Page: 98" in formatted
    assert "Evidence Type: DIAGRAM_ARCHITECTURE" in formatted
    assert "Visual Asset ID: ast_workflow_98" in formatted

    # 3. Citation Engine citation creation
    citation_engine = CitationEngine()
    answer_text = (
        "According to the workflow diagram on Page 98, the system accepts a URL, scrapes content, "
        "and creates social posts [Visual Source 1]."
    )

    citations = citation_engine.select_citations(
        answer_text=answer_text,
        generation_chunks=[scored_chunk],
        user_query="Explain the content creation workflow.",
    )

    assert len(citations) == 1
    cit = citations[0]
    assert cit.display_page == "98"
    assert cit.display_page_number == 98
    assert cit.page_number == 99
    assert cit.internal_page_index == 98
    assert cit.evidence_type == "DIAGRAM_ARCHITECTURE"
    assert cit.visual_asset_id == "ast_workflow_98"
    assert cit.image_url == "/api/documents/doc_guidebook_1/visual-assets/ast_workflow_98"
