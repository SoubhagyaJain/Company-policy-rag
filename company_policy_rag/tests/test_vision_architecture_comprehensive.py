import io
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from backend.ingestion.loaders.pdf import PDFLoader
from backend.models.api_dto import IngestionStatus
from backend.models.chunk import Chunk, ChunkMetadata
from backend.models.rag import ScoredChunk
from backend.rag.citations import CitationEngine
from backend.services.document_service import DocumentService
from backend.vision.image_asset_manager import ImageAsset, ImageAssetManager
from backend.vision.vision_cache import VisionCacheManager
from backend.vision.vision_service import VisionCircuitBreaker, VisionService, VisualContentType
from src.config import settings


def _create_sample_pdf_with_image(file_path: Path, printed_page_num: str = "82") -> Path:
    """Create a minimal PDF with 1 page containing text, footer, and an embedded image."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # Insert text and footer
    page.insert_text(
        (50, 72),
        "#8) Human-like Memory for Agents\n\n"
        "The following architecture diagram illustrates the memory engine components.",
        fontsize=12,
    )
    page.insert_text((50, 800), f"DailyDoseofDS.com\n{printed_page_num}", fontsize=10)

    # Insert a dummy image
    img = Image.new("RGB", (400, 300), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    rect = fitz.Rect(50, 150, 450, 450)
    page.insert_image(rect, stream=img_bytes)

    doc.save(str(file_path))
    doc.close()
    return file_path


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_zero_ingestion_retries(temp_dir):
    """Test 1: Ingestion-time visual extraction executes exactly 1 attempt (0 retries)."""
    cache = VisionCacheManager(storage_dir=temp_dir / "cache")
    asset_mgr = ImageAssetManager(storage_dir=temp_dir / "images")
    service = VisionService(cache_manager=cache, image_asset_manager=asset_mgr)

    dummy_img = Image.new("RGB", (200, 200), color="blue")
    buf = io.BytesIO()
    dummy_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    attempt_counter = 0

    def mock_vision(*args, **kwargs):
        nonlocal attempt_counter
        attempt_counter += 1
        raise TimeoutError("Ollama vision completion timed out")

    with patch("backend.vision.vision_service.execute_vision_completion", side_effect=mock_vision):
        with patch.object(service, "is_available", return_value=(True, "Ready")):
            res = service.extract_from_image(
                image_bytes=img_bytes,
                visual_type=VisualContentType.DIAGRAM_ARCHITECTURE,
                document_id="doc_test_zero_retry",
                page_number=83,
                page_label="82",
                is_query_time=False,  # Ingestion mode
            )

    assert res is None
    assert attempt_counter == 1, f"Expected exactly 1 attempt on ingestion (0 retries), but got {attempt_counter}."


def test_controlled_query_time_lazy_retries(temp_dir):
    """Test 2: Query-time lazy visual extraction performs at most 1 retry (2 attempts)."""
    cache = VisionCacheManager(storage_dir=temp_dir / "cache")
    asset_mgr = ImageAssetManager(storage_dir=temp_dir / "images")
    service = VisionService(cache_manager=cache, image_asset_manager=asset_mgr)

    dummy_img = Image.new("RGB", (200, 200), color="green")
    buf = io.BytesIO()
    dummy_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    attempt_counter = 0

    def mock_vision(*args, **kwargs):
        nonlocal attempt_counter
        attempt_counter += 1
        if attempt_counter == 1:
            raise TimeoutError("Ollama vision timed out on attempt 1")
        return "Recovered diagram description on retry."

    with patch("backend.vision.vision_service.execute_vision_completion", side_effect=mock_vision):
        with patch.object(service, "is_available", return_value=(True, "Ready")):
            res = service.extract_from_image(
                image_bytes=img_bytes,
                visual_type=VisualContentType.DIAGRAM_ARCHITECTURE,
                document_id="doc_test_lazy_retry",
                page_number=83,
                page_label="82",
                is_query_time=True,  # Query-time lazy mode
            )

    assert res is not None
    assert res.text == "Recovered diagram description on retry."
    assert attempt_counter == 2, f"Expected 2 attempts for lazy retry, got {attempt_counter}."


def test_negative_cache_ttl_and_bypass(temp_dir):
    """Test 3: Failed vision extractions are negatively cached and bypassed on subsequent calls."""
    cache = VisionCacheManager(storage_dir=temp_dir / "cache")
    asset_mgr = ImageAssetManager(storage_dir=temp_dir / "images")
    service = VisionService(cache_manager=cache, image_asset_manager=asset_mgr)

    dummy_img = Image.new("RGB", (200, 200), color="red")
    buf = io.BytesIO()
    dummy_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    call_counter = 0

    def mock_vision(*args, **kwargs):
        nonlocal call_counter
        call_counter += 1
        raise RuntimeError("GPU out of memory")

    with patch("backend.vision.vision_service.execute_vision_completion", side_effect=mock_vision):
        with patch.object(service, "is_available", return_value=(True, "Ready")):
            # First call -> fails and writes negative cache
            res1 = service.extract_from_image(
                image_bytes=img_bytes,
                visual_type=VisualContentType.DIAGRAM_ARCHITECTURE,
                document_id="doc_neg_cache",
                page_number=1,
                is_query_time=False,
            )
            assert res1 is None
            assert call_counter == 1

            # Second call -> immediately returned from negative cache without calling Ollama
            res2 = service.extract_from_image(
                image_bytes=img_bytes,
                visual_type=VisualContentType.DIAGRAM_ARCHITECTURE,
                document_id="doc_neg_cache",
                page_number=1,
                is_query_time=False,
            )
            assert res2 is None
            assert call_counter == 1, "Expected second call to be blocked by negative cache!"


def test_standalone_image_asset_extraction(temp_dir):
    """Test 4: Standalone image extraction saves full-resolution image to disk directly from PDF."""
    pdf_path = _create_sample_pdf_with_image(temp_dir / "sample.pdf", printed_page_num="82")
    asset_mgr = ImageAssetManager(storage_dir=temp_dir / "images")

    assets = asset_mgr.extract_page_images(
        pdf_path=pdf_path,
        internal_page_index=0,
        page_number=1,
        page_label="82",
        document_id="doc_sample_123",
    )

    assert len(assets) == 1
    asset = assets[0]
    assert asset.document_id == "doc_sample_123"
    assert asset.page_number == 1
    assert asset.page_label == "82"
    assert asset.width == 400
    assert asset.height == 300
    assert Path(asset.file_path).exists()
    assert asset.asset_url == f"/api/documents/doc_sample_123/images/{asset.image_hash}"


def test_image_survives_vision_model_failure(temp_dir):
    """Test 5: Original image asset remains preserved and accessible even if VLM fails."""
    pdf_path = _create_sample_pdf_with_image(temp_dir / "sample.pdf", printed_page_num="82")
    asset_mgr = ImageAssetManager(storage_dir=temp_dir / "images")
    cache = VisionCacheManager(storage_dir=temp_dir / "cache")
    service = VisionService(cache_manager=cache, image_asset_manager=asset_mgr)

    loader = PDFLoader(vision_service=service, image_asset_manager=asset_mgr)

    with patch("backend.vision.vision_service.execute_vision_completion", side_effect=TimeoutError("Ollama vision timed out")):
        raw_docs = loader.load(pdf_path, base_metadata={"document_id": "doc_survive_test"})

    assert len(raw_docs) >= 1
    # Check that image asset was saved to disk
    assets = asset_mgr.list_assets("doc_survive_test")
    assert len(assets) == 1
    assert Path(assets[0].file_path).is_file()
    assert assets[0].file_size_bytes > 0


def test_image_downscaling_for_inference():
    """Test 6: Large images are scaled down for VLM inference copy while keeping aspect ratio."""
    # Create 1600x1200 image
    large_img = Image.new("RGB", (1600, 1200), color="purple")
    buf = io.BytesIO()
    large_img.save(buf, format="PNG")
    orig_bytes = buf.getvalue()

    opt_bytes = ImageAssetManager.get_optimized_inference_bytes(orig_bytes, max_dim=1024)
    opt_img = Image.open(io.BytesIO(opt_bytes))

    assert max(opt_img.width, opt_img.height) <= 1024
    assert opt_img.width == 1024
    assert opt_img.height == 768
    assert len(opt_bytes) < len(orig_bytes)


def test_canonical_page_numbering_detection(temp_dir):
    """Test 7: Printed footer numbers reconcile physical PDF page numbers with human labels."""
    pdf_path = _create_sample_pdf_with_image(temp_dir / "sample_p82.pdf", printed_page_num="82")
    loader = PDFLoader()

    raw_docs = loader.load(pdf_path, base_metadata={"document_id": "doc_p82"})
    assert len(raw_docs) >= 1
    doc_meta = raw_docs[0].metadata

    assert doc_meta.page_number == 1  # Physical 1-based
    assert doc_meta.internal_page_index == 0  # 0-indexed
    assert doc_meta.page_label == "82"  # Printed footer detected!


def test_vision_circuit_breaker():
    """Test 8: Circuit breaker trips after 3 consecutive failures into OPEN mode."""
    breaker = VisionCircuitBreaker(failure_threshold=3, recovery_cooldown=10.0)

    assert breaker.allow_request() is True
    breaker.record_failure()
    assert breaker.allow_request() is True
    breaker.record_failure()
    assert breaker.allow_request() is True
    breaker.record_failure()

    # 3 failures -> Circuit OPEN
    assert breaker.allow_request() is False

    # After success -> Circuit resets
    breaker.record_success()
    assert breaker.allow_request() is True


def test_document_ingestion_reaches_100_percent_ready_immediately(temp_dir):
    """Test 9: Document ingestion with images reaches 100% READY without waiting for vision."""
    pdf_path = _create_sample_pdf_with_image(temp_dir / "sample_ingest.pdf", printed_page_num="82")
    pdf_bytes = pdf_path.read_bytes()

    mock_vec = MagicMock()
    mock_vec.add_chunks.return_value = None
    mock_bm25 = MagicMock()
    mock_bm25.entries = []
    mock_bm25.build_index.return_value = None
    mock_emb = MagicMock()
    mock_emb.embed_chunks.return_value = [[0.1] * 384]

    doc_service = DocumentService(
        vector_store=mock_vec,
        bm25_index=mock_bm25,
        embedding_service=mock_emb,
        storage_dir=str(temp_dir / "uploads"),
    )
    doc_service.image_asset_manager = ImageAssetManager(storage_dir=temp_dir / "images")

    t0 = time.perf_counter()
    res = doc_service.upload_document(
        filename="AI Agents guidebook.pdf",
        content_bytes=pdf_bytes,
        category="General",
    )
    elapsed = time.perf_counter() - t0

    assert res.status == "READY"
    assert res.progress == 100
    assert res.text_ready is True
    assert elapsed < 5.0, f"Ingestion took {elapsed:.2f}s, expected < 5s."


def test_citation_page_label_and_image_url():
    """Test 10: CitationEngine populates page_label, internal_page_index, and image_url."""
    meta = ChunkMetadata(
        document_id="doc_cit_test",
        source_file="guidebook.pdf",
        file_path="/tmp/guidebook.pdf",
        file_hash="hash123",
        document_type="pdf",
        chunk_strategy="adaptive",
        page_number=83,
        internal_page_index=82,
        page_label="82",
        section_title="Human-like Memory for Agents",
        extra={
            "image_url": "/api/documents/doc_cit_test/images/img_abc123",
        },
    )
    chunk = Chunk(id="chunk_1", text="Human-like memory architecture diagram.", metadata=meta)
    scored = ScoredChunk(chunk=chunk, score=0.92, rank=1)

    engine = CitationEngine()
    citations = engine.select_citations(
        answer_text="As shown in [Source 1], the memory system organizes data.",
        generation_chunks=[scored],
    )

    assert len(citations) == 1
    cit = citations[0]
    assert cit.page_number == 83
    assert cit.page_label == "82"
    assert cit.internal_page_index == 82
    assert cit.image_url == "/api/documents/doc_cit_test/images/img_abc123"
