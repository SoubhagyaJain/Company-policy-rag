"""
Test Suite for Local Vision Model Support in RAG.

Covers:
1. Model Detection & Ollama Probing (installed vs missing pull command)
2. Visual Page Detection Heuristics (pure text skip, code screenshot, diagram, table)
3. Vision Ingestion Disk Cache (SHA256 content-addressing)
4. Complementary Chunk Packing (balanced code + prose + diagram)
5. Query-Time Lazy Visual Fallback (on visual trigger cues)
6. Exact User Scenario: "How can I make X Analyst Agent?" with visual code extraction
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.models.chunk import Chunk, ChunkMetadata, ContentType
from backend.models.document import DocumentType, RawDocument
from backend.models.rag import ScoredChunk
from backend.rag.context_compression import ContextCompressor
from backend.rag.pipeline import RAGPipeline
from backend.vision.vision_cache import VisionCacheManager
from backend.vision.vision_service import (
    CODE_EXTRACTION_PROMPT,
    DIAGRAM_EXTRACTION_PROMPT,
    VisionService,
    VisualContentType,
)
from src.config import settings
from src.ollama_client import probe_vision_model_status


# ============================================================================
# Helpers
# ============================================================================

def _make_chunk(
    chunk_id: str,
    text: str,
    document_id: str = "doc_test_101",
    source_file: str = "Test.pdf",
    page_number: int = 1,
    content_type: ContentType = ContentType.PROSE,
    section_title: str = "General",
    extra: dict | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            document_id=document_id,
            source_file=source_file,
            file_path=f"/data/docs/{source_file}",
            file_hash=f"hash_{document_id}",
            document_type="pdf",
            chunk_strategy="adaptive",
            page_number=page_number,
            section_title=section_title,
            content_type=content_type,
            extra=extra or {},
        ),
    )


# ============================================================================
# 1. Vision Model Detection & Probing Tests
# ============================================================================

def test_probe_vision_model_status_when_available():
    """Verify that when Ollama has the vision model installed, probe returns True."""
    with patch("src.ollama_client.probe_ollama_tags", return_value=(True, ["qwen2.5:7b", "Qwen3-VL-2B-Instruct", "nomic-embed-text"], None)):
        is_ready, msg = probe_vision_model_status("Qwen3-VL-2B-Instruct")
        assert is_ready is True
        assert "available locally" in msg


def test_probe_vision_model_status_when_missing():
    """Verify that when the vision model is missing, probe returns False with the pull command."""
    with patch("src.ollama_client.probe_ollama_tags", return_value=(True, ["qwen2.5:7b", "nomic-embed-text"], None)):
        is_ready, msg = probe_vision_model_status("Qwen3-VL-2B-Instruct")
        assert is_ready is False
        assert "ollama pull Qwen3-VL-2B-Instruct" in msg


# ============================================================================
# 2. Visual Page Detection Heuristics Tests
# ============================================================================

def test_visual_detection_skips_pure_text_page():
    """Pages with zero images must be skipped with zero vision calls."""
    service = VisionService()
    text = "This is standard text content with no visual figures or images."
    res = service.detect_visual_content(page_text=text, image_bytes=None, image_count=0)
    assert res.has_visual is False
    assert res.visual_type == VisualContentType.NONE


def test_visual_detection_skips_small_icons():
    """Small decorative bullets/logos (<120x80) should be skipped."""
    service = VisionService()
    fake_icon_bytes = b"fake_png_icon_bytes"
    res = service.detect_visual_content(
        page_text="Some text",
        image_bytes=fake_icon_bytes,
        image_count=1,
        image_width=40,
        image_height=40,
    )
    assert res.has_visual is False


def test_visual_detection_identifies_code_screenshot():
    """Pages with code cues and significant image are classified as CODE_SCREENSHOT."""
    service = VisionService()
    fake_img = b"fake_code_screenshot_png"
    text = "Here's how it's done: def build_agent(config):"
    res = service.detect_visual_content(
        page_text=text,
        image_bytes=fake_img,
        image_count=1,
        image_width=800,
        image_height=600,
    )
    assert res.has_visual is True
    assert res.visual_type == VisualContentType.CODE_SCREENSHOT


def test_visual_detection_identifies_architecture_diagram():
    """Pages with diagram/architecture cues are classified as DIAGRAM_ARCHITECTURE."""
    service = VisionService()
    fake_img = b"fake_diagram_png"
    text = "Figure 3.2: Multi-agent system architecture workflow and data flow pipeline."
    res = service.detect_visual_content(
        page_text=text,
        image_bytes=fake_img,
        image_count=1,
        image_width=900,
        image_height=700,
    )
    assert res.has_visual is True
    assert res.visual_type == VisualContentType.DIAGRAM_ARCHITECTURE


# ============================================================================
# 3. Vision Ingestion Disk Cache Tests
# ============================================================================

def test_vision_cache_hit_and_miss(tmp_path: Path):
    """Verify SHA256 content-addressing reuses cached extraction without calling LLM."""
    cache = VisionCacheManager(cache_dir=tmp_path)
    img_bytes = b"sample_image_binary_data"
    img_hash = VisionCacheManager.compute_image_hash(img_bytes)

    # 1. Miss on empty cache
    assert cache.get(img_hash, vision_model="Qwen3-VL-2B-Instruct") is None

    # 2. Set cache entry
    cache.set(
        image_hash=img_hash,
        vision_model="Qwen3-VL-2B-Instruct",
        extracted_text="class Agent:\n    pass",
        visual_type="code_screenshot",
        document_id="doc_1",
        page_number=3,
    )

    # 3. Hit on subsequent request
    cached = cache.get(img_hash, vision_model="Qwen3-VL-2B-Instruct", document_id="doc_1", page_number=3)
    assert cached is not None
    assert cached["extracted_text"] == "class Agent:\n    pass"
    assert cached["visual_type"] == "code_screenshot"


# ============================================================================
# 4. Complementary Chunk Packing Tests
# ============================================================================

def test_pack_complementary_chunks_balances_code_and_prose():
    """
    When user asks 'How can I make X Analyst Agent?', pack_complementary_chunks must
    retain description + code without allowing 6 redundant prose chunks to push out the code.
    """
    compressor = ContextCompressor()
    query = "How can I make X Analyst Agent?"

    chunk_prose_1 = _make_chunk("c_p1", "X Analyst Agent is an AI agent that monitors social media.", content_type=ContentType.PROSE)
    chunk_prose_2 = _make_chunk("c_p2", "X Analyst Agent analyzes sentiment across public posts.", content_type=ContentType.PROSE)
    chunk_prose_3 = _make_chunk("c_p3", "X Analyst Agent produces structured market intelligence reports.", content_type=ContentType.PROSE)
    chunk_prose_4 = _make_chunk("c_p4", "X Analyst Agent integrates with external data connectors.", content_type=ContentType.PROSE)

    chunk_code = _make_chunk(
        "c_code",
        "```python\nclass XAnalystAgent:\n    def __init__(self, key):\n        self.key = key\n```",
        content_type=ContentType.CODE,
        extra={"is_visual_extraction": True, "content_type": "code"},
    )
    chunk_table = _make_chunk("c_table", "| Parameter | Type |\n| key | str |", content_type=ContentType.TABLE)

    raw_pool = [
        ScoredChunk(chunk=chunk_prose_1, score=0.92),
        ScoredChunk(chunk=chunk_prose_2, score=0.90),
        ScoredChunk(chunk=chunk_prose_3, score=0.88),
        ScoredChunk(chunk=chunk_prose_4, score=0.86),
        ScoredChunk(chunk=chunk_code, score=0.85),
        ScoredChunk(chunk=chunk_table, score=0.80),
    ]

    packed = compressor.pack_complementary_chunks(raw_pool, query=query, max_chunks=4)

    packed_ids = [sc.chunk.id for sc in packed]
    assert "c_code" in packed_ids, "Code chunk must be preserved in complementary packing!"
    assert "c_p1" in packed_ids, "Top prose description chunk must be preserved!"
    assert len(packed) <= 4


# ============================================================================
# 5. Required End-to-End Test: "How can I make X Analyst Agent?"
# ============================================================================

def test_e2e_x_analyst_agent_code_extraction_scenario():
    """
    Exact User Requirement Test:
    - Document: Contains X Analyst Agent
    - Text: 'This agent analyzes posts scraped by Bright Data.'
    - Image: Contains implementation code (extracted via vision model)
    - Query: 'How can I make X Analyst Agent?'
    - System MUST retrieve Description + Code screenshot extraction + Agent implementation
    - Answer explains how to build the agent using actual code from image
    - Answer must NEVER say 'The document does not provide detailed instructions'
    """
    # 1. Setup chunks: text description + vision-extracted code screenshot
    chunk_desc = _make_chunk(
        chunk_id="chunk_x_desc",
        text="X Analyst Agent Overview: This agent analyzes posts scraped by Bright Data.",
        document_id="doc_x_analyst_001",
        source_file="X_Analyst_Guidebook.pdf",
        page_number=64,
        section_title="X Analyst Agent",
        content_type=ContentType.PROSE,
    )

    extracted_code_text = (
        "```python\n"
        "class XAnalystAgent:\n"
        "    def __init__(self, bright_data_api_key: str):\n"
        "        self.api_key = bright_data_api_key\n\n"
        "    def analyze_posts(self, posts: list[dict]) -> dict:\n"
        "        # Sentiment clustering and brand mention extraction\n"
        "        return {'sentiment': 'positive', 'metrics': len(posts)}\n"
        "```"
    )

    chunk_visual_code = _make_chunk(
        chunk_id="chunk_x_code_vision",
        text=extracted_code_text,
        document_id="doc_x_analyst_001",
        source_file="X_Analyst_Guidebook.pdf",
        page_number=64,
        section_title="X Analyst Agent Implementation",
        content_type=ContentType.CODE,
        extra={"is_visual_extraction": True, "visual_type": "code_screenshot", "content_type": "code"},
    )

    docstore = {
        "chunk_x_desc": chunk_desc,
        "chunk_x_code_vision": chunk_visual_code,
    }

    # 2. Mock Retriever
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        ScoredChunk(chunk=chunk_desc, score=0.94),
        ScoredChunk(chunk=chunk_visual_code, score=0.91),
    ]

    # 3. Mock LLM (Qwen 2.5 7B) receiving both description and extracted code
    mock_llm = MagicMock()
    def fake_llm_complete(prompt: str) -> str:
        # Assert LLM receives both description and vision-extracted code
        assert "Bright Data" in prompt
        assert "class XAnalystAgent" in prompt
        return (
            "To make the X Analyst Agent, use the implementation provided in the document:\n\n"
            "```python\n"
            "class XAnalystAgent:\n"
            "    def __init__(self, bright_data_api_key: str):\n"
            "        self.api_key = bright_data_api_key\n\n"
            "    def analyze_posts(self, posts: list[dict]) -> dict:\n"
            "        return {'sentiment': 'positive', 'metrics': len(posts)}\n"
            "```\n\n"
            "This agent analyzes posts scraped by Bright Data by initializing with an API key "
            "and executing sentiment clustering on the post dataset."
        )

    mock_llm.complete.side_effect = fake_llm_complete

    pipeline = RAGPipeline(
        hybrid_retriever=mock_retriever,
        docstore=docstore,
        llm=mock_llm,
    )

    # 4. Execute Query
    response = pipeline.query(
        user_query="How can I make X Analyst Agent?",
        active_document_id="doc_x_analyst_001",
        active_document_name="X_Analyst_Guidebook.pdf",
    )

    # 5. Assertions
    assert response is not None
    assert "class XAnalystAgent" in response.answer
    assert "Bright Data" in response.answer
    assert "The document does not provide detailed instructions" not in response.answer

    # Verify context chunks contain both description and vision-extracted code
    context_types = [sc.chunk.metadata.content_type for sc in response.context_chunks]
    assert ContentType.CODE in context_types
    assert ContentType.PROSE in context_types


# ============================================================================
# 6. Query-Time Lazy Fallback Tests
# ============================================================================

def test_query_time_lazy_fallback_extracts_page_on_trigger(tmp_path: Path):
    """
    If a document was ingested without visual extraction and retrieved text indicates
    'Here's how it's done' without a code chunk, the lazy fallback should extract that page.
    """
    chunk_trigger = _make_chunk(
        chunk_id="chunk_trig",
        text="Here's how it's done for the custom analysis agent.",
        document_id="doc_lazy_01",
        source_file="Guide.pdf",
        page_number=5,
        content_type=ContentType.PROSE,
    )

    # Mock VisionService
    mock_vision = MagicMock()
    mock_vision.is_available.return_value = (True, "Ready")
    mock_vision.process_pdf_page_visuals.return_value = [
        MagicMock(
            text="```python\ndef run_analysis():\n    pass\n```",
            content_type="code",
            visual_type="code_screenshot",
            image_hash="fake_hash_123",
        )
    ]

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [ScoredChunk(chunk=chunk_trigger, score=0.88)]

    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Run analysis using def run_analysis()."

    pipeline = RAGPipeline(
        hybrid_retriever=mock_retriever,
        docstore={"chunk_trig": chunk_trigger},
        llm=mock_llm,
        vision_service=mock_vision,
    )

    # Create dummy file so Path(file_path).is_file() returns True
    dummy_pdf = tmp_path / "Guide.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy")
    chunk_trigger.metadata.file_path = str(dummy_pdf)

    # Execute query
    response = pipeline.query(
        user_query="How to run the analysis agent?",
        active_document_id="doc_lazy_01",
    )

    assert response is not None
    # Vision service must have been called for page 5
    assert mock_vision.process_pdf_page_visuals.call_count >= 1
    assert any("def run_analysis" in sc.chunk.text for sc in response.context_chunks)
