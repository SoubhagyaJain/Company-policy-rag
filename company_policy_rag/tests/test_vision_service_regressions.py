from __future__ import annotations

import io
from unittest.mock import patch

from PIL import Image

from backend.vision.image_asset_manager import ImageAssetManager
from backend.vision.vision_cache import VisionCacheManager
from backend.vision.vision_service import VisionService, VisualContentType


def _image_bytes() -> bytes:
    image = Image.new("RGB", (240, 160), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_cached_table_preserves_table_content_type(tmp_path):
    image_bytes = _image_bytes()
    cache = VisionCacheManager(cache_dir=tmp_path / "cache")
    service = VisionService(
        cache_manager=cache,
        image_asset_manager=ImageAssetManager(storage_dir=tmp_path / "images"),
    )
    image_hash = cache.compute_image_hash(image_bytes)
    cache.set(
        image_hash=image_hash,
        vision_model=service.vision_model,
        extracted_text="| name | value |\n| --- | --- |\n| A | 1 |",
        visual_type=VisualContentType.TABLE_DATA.value,
        document_id="doc_table",
        page_number=3,
    )

    result = service.extract_from_image(
        image_bytes=image_bytes,
        visual_type=VisualContentType.TABLE_DATA,
        document_id="doc_table",
        page_number=3,
    )

    assert result is not None
    assert result.content_type == "table"
    assert result.raw_code is None


def test_figure_asset_is_eligible_for_diagram_request(tmp_path):
    manager = ImageAssetManager(storage_dir=tmp_path / "images")
    asset = manager.save_image_asset(
        document_id="doc_figure",
        internal_page_index=6,
        page_number=7,
        page_label="6",
        image_bytes=_image_bytes(),
        visual_type="figure",
    )
    service = VisionService(
        cache_manager=VisionCacheManager(cache_dir=tmp_path / "cache"),
        image_asset_manager=manager,
    )

    with patch.object(service, "extract_from_image", return_value=None) as extract:
        service.extract_stored_assets(
            document_id="doc_figure",
            page_number=7,
            required_visual_type=VisualContentType.DIAGRAM_ARCHITECTURE,
        )

    assert extract.call_count == 1
    assert extract.call_args.kwargs["page_number"] == asset.physical_page_number
    assert extract.call_args.kwargs["visual_type"] == VisualContentType.DIAGRAM_ARCHITECTURE


def test_explicit_diagram_request_overrides_wrong_page_heuristic(tmp_path):
    import fitz

    image_bytes = _image_bytes()
    pdf_path = tmp_path / "heuristic_mismatch.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Implementation code example")
    page.insert_image(fitz.Rect(72, 100, 312, 260), stream=image_bytes)
    doc.save(pdf_path)
    doc.close()

    service = VisionService(
        cache_manager=VisionCacheManager(cache_dir=tmp_path / "cache"),
        image_asset_manager=ImageAssetManager(storage_dir=tmp_path / "images"),
    )

    with patch.object(service, "extract_from_image", return_value=None) as extract:
        service.process_pdf_page_visuals(
            pdf_path=pdf_path,
            page_number=1,
            page_text="Implementation code example",
            required_visual_type=VisualContentType.DIAGRAM_ARCHITECTURE,
        )

    assert extract.call_count == 1
    assert extract.call_args.kwargs["visual_type"] == VisualContentType.DIAGRAM_ARCHITECTURE
