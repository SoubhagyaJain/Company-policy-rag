import pytest
from pathlib import Path
from backend.models.chunk import Chunk, ChunkMetadata, ContentType
from backend.models.rag import QueryCategory, ScoredChunk
from backend.rag.evidence_gate import EvidenceSufficiencyGate
from backend.vision.image_asset_manager import ImageAsset, ImageAssetManager


def test_evidence_gate_workflow_diagram_evaluation():
    gate = EvidenceSufficiencyGate()

    # Query asking about workflow
    query = "Explain the content creation workflow."
    intent = QueryCategory.FACTUAL

    # Text chunk without visual extraction
    chunk = Chunk(
        id="chk_1",
        text="Build an Agentic workflow that turns any URL into social media posts. Firecrawl scrapes web content.",
        metadata=ChunkMetadata(
            document_id="doc_guidebook",
            source_file="guidebook.pdf",
            file_path="uploads/guidebook.pdf",
            file_hash="hash1",
            document_type="pdf",
            chunk_strategy="recursive",
            page_number=99,
            display_page_number=98,
            page_label="98",
            internal_page_index=98,
            section_title="Multi-agent Content Creation System",
        ),
    )
    scored = ScoredChunk(chunk=chunk, score=0.92)

    res = gate.evaluate(query=query, intent=intent, candidate_chunks=[scored])
    # The gate detects that the query requires workflow/diagram understanding
    assert not res.is_sufficient
    assert "architecture_diagram" in res.missing_evidence_types
    assert 99 in res.pages_to_inspect


def test_evidence_gate_requires_extracted_code_not_a_code_hint():
    """A page labelled as code is not code evidence until its text was extracted."""
    gate = EvidenceSufficiencyGate()
    chunk = Chunk(
        id="code_hint",
        text="The implementation is shown in the image below.",
        metadata=ChunkMetadata(
            document_id="doc_code",
            source_file="code-guide.pdf",
            file_hash="hash-code",
            page_number=12,
            section_title="Implementation",
            has_code=True,
        ),
    )

    res = gate.evaluate(
        query="Show me the exact code for the implementation.",
        intent=QueryCategory.CODE,
        candidate_chunks=[ScoredChunk(chunk=chunk, score=0.9)],
    )

    assert not res.is_sufficient
    assert "code_implementation" in res.missing_evidence_types


def test_image_asset_manager_multi_identifier_lookup(tmp_path):
    mgr = ImageAssetManager(storage_dir=tmp_path)

    # Save a test 100x100 PNG image
    import io
    from PIL import Image

    img = Image.new("RGB", (200, 150), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    asset = mgr.save_image_asset(
        document_id="doc_123",
        internal_page_index=98,
        page_number=99,
        page_label="98",
        display_page_number=98,
        image_bytes=img_bytes,
        section_title="Multi-agent Content Creation System",
        visual_type="diagram_architecture",
    )

    assert asset.asset_id.startswith("ast_")
    assert asset.internal_page_index == 98
    assert asset.physical_page_number == 99
    assert asset.display_page_number == 98
    assert asset.display_label == "98"

    # Lookup by internal index (98)
    found_by_internal = mgr.get_page_asset("doc_123", 98)
    assert found_by_internal is not None
    assert found_by_internal.asset_id == asset.asset_id

    # Lookup by physical page (99)
    found_by_phys = mgr.get_page_asset("doc_123", 99)
    assert found_by_phys is not None
    assert found_by_phys.asset_id == asset.asset_id

    # Lookup by string printed label ("98")
    found_by_str = mgr.get_page_asset("doc_123", "98")
    assert found_by_str is not None
    assert found_by_str.asset_id == asset.asset_id

    # Lookup by asset ID
    found_by_id = mgr.get_asset("doc_123", asset.asset_id)
    assert found_by_id is not None
    assert found_by_id.image_hash == asset.image_hash
