from __future__ import annotations

import io
import pytest
from PIL import Image

from backend.embeddings.vector_store import unpack_chroma_metadata, ChromaVectorStore
from backend.models.chunk import Chunk, ChunkMetadata, ContentType
from backend.models.page_identity import PageIdentity
from backend.models.rag import Citation, EvidenceStatus, QueryCategory, ScoredChunk
from backend.rag.citations import CitationEngine
from backend.rag.evidence_gate import EvidenceSufficiencyGate, EvidenceSufficiencyResult
from backend.rag.pipeline import GROUNDED_SYSTEM_PROMPT, _format_evidence_status_directive
from backend.vision.image_asset_manager import ImageAssetManager
from backend.vision.vision_service import VisionService, VisualContentType


# ============================================================================
# Test A — Page identity round trip
# ============================================================================
def test_page_identity_round_trip():
    """Verify that physical sheet 74 and printed page 73 survive flattening, Chroma unpack, and citation build."""
    meta = ChunkMetadata(
        document_id="doc_audit_test",
        source_file="guidebook.pdf",
        chunk_index=0,
        page_number=74,
        internal_page_index=73,
        display_page_number=73,
        page_label="73",
        section_title="Browserbase tool",
        content_type=ContentType.CODE,
        has_code=True,
    )
    chunk = Chunk(
        id="chk_audit_1",
        text="result = crew.kickoff(inputs={'request': 'Best hotels'})",
        metadata=meta,
    )

    # 1. Flatten
    vs = ChromaVectorStore()
    flat = vs._flatten_metadata(meta)
    assert flat["display_page_number"] == 73
    assert flat["page_label"] == "73"
    assert flat["page_number"] == 74
    assert flat["internal_page_index"] == 73

    # 2. Unpack
    unpacked_meta = unpack_chroma_metadata(flat)
    assert unpacked_meta.display_page_number == 73
    assert unpacked_meta.page_label == "73"
    assert unpacked_meta.page_number == 74
    assert unpacked_meta.internal_page_index == 73

    # 3. Citation Build
    sc = ScoredChunk(
        chunk=Chunk(id="chk_audit_1", text=chunk.text, metadata=unpacked_meta),
        score=0.95,
    )
    engine = CitationEngine()
    citation = engine._build_citation_from_chunk(1, sc, "cited_in_answer")
    assert citation.display_page == "73"
    assert citation.display_page_number == 73
    assert citation.page_label == "73"
    assert citation.page_number == 74
    assert citation.evidence_type == "CODE"


# ============================================================================
# Test B — Legacy metadata compatibility
# ============================================================================
def test_legacy_metadata_compatibility():
    """Verify that legacy ChromaDB metadata with extra_* keys unpacks properly without data loss."""
    legacy_meta_dict = {
        "document_id": "doc_legacy_123",
        "source_file": "old_manual.pdf",
        "page_number": 74,
        "extra_display_page_number": 73,
        "extra_page_label": "73",
        "extra_internal_page_index": 72,
        "extra_has_code": True,
        "extra_visual_type": "code_screenshot",
        "extra_is_visual_extraction": True,
    }

    unpacked = unpack_chroma_metadata(legacy_meta_dict)
    assert unpacked.display_page_number == 73
    assert unpacked.page_label == "73"
    assert unpacked.internal_page_index == 72
    assert unpacked.page_number == 74
    assert unpacked.has_code is True

    sc = ScoredChunk(
        chunk=Chunk(
            id="chk_legacy_1",
            text="```python\nresult = crew.kickoff()\n```",
            metadata=unpacked,
        ),
        score=0.90,
    )
    engine = CitationEngine()
    citation = engine._build_citation_from_chunk(1, sc, "cited_in_answer")
    assert citation.display_page == "73"
    assert citation.evidence_type == "CODE_SCREENSHOT"


# ============================================================================
# Test C — Wide code screenshot classification
# ============================================================================
def test_wide_code_screenshot_classification():
    """Verify that a landscape (width > height) code image is classified as code_screenshot, not diagram_architecture."""
    img = Image.new("RGB", (1200, 400), color="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    code_page_text = "Here is the implementation of the hotel agent:\ndef kickoff():\n    import crew\n    return True"
    service = VisionService()
    detection = service.detect_visual_content(
        page_text=code_page_text,
        image_bytes=img_bytes,
        image_count=1,
        page_number=74,
        display_page_number=73,
        page_label="73",
        internal_page_index=73,
        image_width=1200,
        image_height=400,
    )

    assert detection.has_visual is True
    assert detection.visual_type == VisualContentType.CODE_SCREENSHOT


# ============================================================================
# Test D — Runtime reclassification
# ============================================================================
def test_runtime_reclassification():
    """Verify that if extracted text contains code, visual chunk and citation normalize to code_screenshot / CODE_SCREENSHOT."""
    sc = ScoredChunk(
        chunk=Chunk(
            id="chk_reclass_1",
            text="```python\nresult = crew.kickoff(\n    inputs={'request': 'Best hotels'}\n)\n```",
            metadata=ChunkMetadata(
                document_id="doc_rec",
                source_file="doc.pdf",
                chunk_index=0,
                page_number=74,
                display_page_number=73,
                page_label="73",
                extra={"is_visual_extraction": True, "visual_type": "diagram_architecture"},
            ),
        ),
        score=0.92,
    )

    engine = CitationEngine()
    citation = engine._build_citation_from_chunk(1, sc, "cited_in_answer")
    assert citation.evidence_type == "CODE_SCREENSHOT"


# ============================================================================
# Test E — Partial evidence classification
# ============================================================================
def test_partial_evidence_classification():
    """Verify that an invocation snippet (e.g. crew.kickoff) is classified as PARTIAL evidence, never MISSING."""
    gate = EvidenceSufficiencyGate()
    chunks = [
        ScoredChunk(
            chunk=Chunk(
                id="chk_partial_1",
                text="result = crew.kickoff(\n    inputs={\n        'request': 'Best hotels in SkyTree, Tokyo',\n        'current_year': datetime.date.today().year,\n    }\n)",
                metadata=ChunkMetadata(
                    document_id="doc_part",
                    source_file="guide.pdf",
                    chunk_index=0,
                    page_number=74,
                    display_page_number=73,
                    page_label="73",
                    has_code=True,
                ),
            ),
            score=0.88,
        )
    ]

    res = gate.evaluate(
        query="What is the implementation code for Hotel Search Agent?",
        intent=QueryCategory.CODE,
        candidate_chunks=chunks,
    )
    assert res.is_sufficient is True
    assert res.evidence_status == EvidenceStatus.PARTIAL


# ============================================================================
# Test F — Contradictory answer prevention prompt rules
# ============================================================================
def test_contradictory_answer_prompt_rules():
    """Verify that GROUNDED_SYSTEM_PROMPT and partial evidence directives forbid false absence claims."""
    assert "RULE 9:" in GROUNDED_SYSTEM_PROMPT
    assert "Never state that the document does not contain the code" in GROUNDED_SYSTEM_PROMPT

    directive = _format_evidence_status_directive(EvidenceStatus.PARTIAL)
    assert "PARTIAL IMPLEMENTATION" in directive
    assert "DO NOT claim that the document does not contain the code" in directive
    assert "DO NOT fabricate, invent, or hallucinate" in directive


# ============================================================================
# Test G — Duplicate vision prevention
# ============================================================================
def test_duplicate_vision_prevention():
    """Verify that rank_page_assets prioritizes code screenshots and handles duplicate inquiries."""
    mgr = ImageAssetManager()
    mgr.save_image_asset(
        document_id="doc_dup_test",
        internal_page_index=72,
        page_number=73,
        page_label="72",
        display_page_number=72,
        image_bytes=b"fake_image_diagram",
        visual_type="diagram_architecture",
        section_title="Hotel Agent Overview",
    )
    mgr.save_image_asset(
        document_id="doc_dup_test",
        internal_page_index=72,
        page_number=73,
        page_label="72",
        display_page_number=72,
        image_bytes=b"fake_image_code",
        visual_type="code_screenshot",
        section_title="Hotel Agent Code",
    )

    ranked = mgr.rank_page_assets(
        document_id="doc_dup_test",
        # Public ranking accepts the printed/display page number, not the PDF sheet.
        page_identifier=72,
        query="What is the code implementation for hotel agent?",
        intent="code",
    )
    assert len(ranked) == 2
    assert ranked[0].visual_type == "code_screenshot"


# ============================================================================
# Test H — Query-level vision budget and code fidelity optimization
# ============================================================================
def test_image_optimization_code_fidelity():
    """Verify that get_optimized_inference_bytes uses higher dimensions and quality for code screenshots."""
    img = Image.new("RGB", (2000, 1500), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    opt_code = ImageAssetManager.get_optimized_inference_bytes(raw_bytes, is_code=True)
    opt_diagram = ImageAssetManager.get_optimized_inference_bytes(raw_bytes, is_code=False)

    img_code = Image.open(io.BytesIO(opt_code))
    img_diag = Image.open(io.BytesIO(opt_diagram))

    # Code allows up to 1280 max dimension, generic diagram up to 1024
    assert max(img_code.size) == 1280
    assert max(img_diag.size) == 1024
