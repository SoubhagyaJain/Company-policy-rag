import pytest
from backend.models.page_identity import PageIdentity
from backend.models.chunk import Chunk, ChunkMetadata, ContentType
from backend.models.document import DocumentMetadata
from backend.models.rag import Citation
from backend.vision.image_asset_manager import ImageAssetManager


def test_page_identity_creation_and_defaults():
    # Case 1: Pure 0-indexed internal page 98
    p1 = PageIdentity.from_indices(internal_page_index=98)
    assert p1.internal_page_index == 98
    assert p1.physical_page_number == 99
    assert p1.display_label == "99"

    # Case 2: 0-indexed 98 with printed display label 98
    p2 = PageIdentity.from_indices(
        internal_page_index=98,
        physical_page_number=99,
        display_page_number=98,
        page_label="98",
    )
    assert p2.internal_page_index == 98
    assert p2.physical_page_number == 99
    assert p2.display_page_number == 98
    assert p2.page_label == "98"
    assert p2.display_label == "98"

    # Case 3: Roman numeral display label
    p3 = PageIdentity.from_indices(
        internal_page_index=3,
        physical_page_number=4,
        display_page_number="iv",
        page_label="iv",
    )
    assert p3.internal_page_index == 3
    assert p3.physical_page_number == 4
    assert p3.display_page_number == "iv"
    assert p3.display_label == "iv"


def test_page_identity_matching():
    p = PageIdentity.from_indices(
        internal_page_index=98,
        physical_page_number=99,
        display_page_number=98,
        page_label="98",
    )
    # User-facing references resolve only through the printed display number.
    assert p.matches(98)
    assert not p.matches(99)
    assert p.matches_physical_page(99)
    assert p.matches_internal_index(98)
    assert not p.matches(100)

    # String matches
    assert p.matches("98")
    assert not p.matches("99")
    assert p.matches("Page 98")
    assert p.matches("p. 98")
    assert p.matches("PAGE 98")
    assert not p.matches("Page 100")


def test_display_lookup_never_matches_internal_or_physical_indices():
    """User page references must not resolve through parser or PDF-sheet indices."""
    p = PageIdentity.from_indices(
        internal_page_index=98,
        physical_page_number=99,
        display_page_number=100,
        page_label="100",
    )

    assert p.matches_display("Page 100")
    assert not p.matches_display("Page 98")
    assert not p.matches_display(99)
    assert p.matches_internal_index(98)
    assert p.matches_physical_page(99)


def test_visual_asset_lookup_requires_an_explicit_page_domain(tmp_path):
    """An offset PDF must not select a physical page for a printed-page request."""
    manager = ImageAssetManager(storage_dir=tmp_path)
    first = manager.save_image_asset(
        document_id="offset_doc",
        internal_page_index=98,
        page_number=99,
        display_page_number=100,
        page_label="100",
        image_bytes=b"first-image",
    )
    second = manager.save_image_asset(
        document_id="offset_doc",
        internal_page_index=99,
        page_number=100,
        display_page_number=101,
        page_label="101",
        image_bytes=b"second-image",
    )

    assert [asset.asset_id for asset in manager.get_page_assets("offset_doc", 100)] == [first.asset_id]
    assert [asset.asset_id for asset in manager.get_page_assets_by_physical_page("offset_doc", 100)] == [second.asset_id]
    assert [asset.asset_id for asset in manager.get_page_assets_by_internal_index("offset_doc", 98)] == [first.asset_id]


def test_chunk_and_citation_page_identity_propagation():
    meta = ChunkMetadata(
        document_id="doc_test",
        source_file="guidebook.pdf",
        file_path="uploads/guidebook.pdf",
        file_hash="hash123",
        document_type="pdf",
        chunk_strategy="recursive",
        internal_page_index=98,
        page_number=99,
        display_page_number=98,
        page_label="98",
        section_title="Multi-agent Content Creation System",
    )
    page_id = meta.get_page_identity()
    assert page_id.internal_page_index == 98
    assert page_id.physical_page_number == 99
    assert page_id.display_page_number == 98
    assert page_id.display_label == "98"

    cit = Citation(
        source_index=1,
        chunk_id="chk_1",
        document_id="doc_test",
        source_file="guidebook.pdf",
        page_number=99,
        internal_page_index=98,
        display_page_number=98,
        page_label="98",
        section_title="Multi-agent Content Creation System",
        snippet="Workflow description...",
        evidence_type="DIAGRAM_ARCHITECTURE",
        visual_asset_id="ast_abc123",
    )
    assert cit.display_page == "98"
    assert cit.evidence_type == "DIAGRAM_ARCHITECTURE"
    assert cit.visual_asset_id == "ast_abc123"
