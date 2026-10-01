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


# ---------------------------------------------------------------------------
# Scanned (text-less) PDF ingestion
# ---------------------------------------------------------------------------


def _scanned_pdf(tmp_path, pages: int = 2):
    import fitz

    pdf_path = tmp_path / "scanned.pdf"
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_image(page.rect, stream=_image_bytes())
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _service(tmp_path) -> VisionService:
    return VisionService(
        cache_manager=VisionCacheManager(cache_dir=tmp_path / "cache"),
        image_asset_manager=ImageAssetManager(storage_dir=tmp_path / "images"),
    )


def test_device_probe_sees_gpu_before_runtime_is_imported(monkeypatch):
    """The probe runs before any load; it must not report cpu just because torch is unimported."""
    from types import SimpleNamespace

    from backend.vision import hf_vision_client as hv

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            mem_get_info=lambda: (4 * 1024**3, 6 * 1024**3),
            current_device=lambda: 0,
        )
    )

    def fake_ensure_runtime():
        monkeypatch.setattr(hv, "torch", fake_torch)
        return True, None

    monkeypatch.setattr(hv, "torch", None)
    monkeypatch.setattr(hv, "_ensure_runtime", fake_ensure_runtime)
    monkeypatch.setattr(hv, "_driver_free_gb", lambda device_index: None)

    assert hv.HFVisionClient()._choose_device() == "cuda"


def test_device_probe_trusts_the_driver_over_torch(monkeypatch):
    """Windows WDDM: torch reports its own budget and misses VRAM held by Ollama."""
    from types import SimpleNamespace

    from backend.vision import hf_vision_client as hv

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            mem_get_info=lambda: (5 * 1024**3, 6 * 1024**3),
            current_device=lambda: 0,
        )
    )
    monkeypatch.setattr(hv, "torch", fake_torch)
    monkeypatch.setattr(hv, "_ensure_runtime", lambda: (True, None))
    monkeypatch.setattr(hv, "_driver_free_gb", lambda device_index: 1.0)

    client = hv.HFVisionClient()
    assert client.gpu_free_gb() == 1.0
    assert client.has_gpu_headroom() is False
    assert client._choose_device() == "cpu"


def test_textless_page_is_transcribed_as_prose(tmp_path):
    from backend.vision import vision_service as vs
    from src.config import settings

    service = _service(tmp_path)
    detection = service.detect_visual_content(
        page_text="", image_bytes=_image_bytes(), image_count=1, image_width=1240, image_height=1754
    )
    assert detection.visual_type == VisualContentType.SCANNED_TEXT

    transcript = "Part 14 - Implement it from scratch: a tiny PyTorch decoder in python"
    with (
        patch.object(service, "is_available", return_value=(True, "ready")),
        patch.object(vs.HFVisionClient, "get_instance") as get_instance,
    ):
        get_instance.return_value.execute.return_value = transcript
        chunk = service.extract_from_image(
            image_bytes=_image_bytes(),
            visual_type=detection.visual_type,
            document_id="doc_scan",
            page_number=1,
        )

    call = get_instance.return_value.execute.call_args.kwargs
    assert call["prompt"] == vs.SCANNED_PAGE_PROMPT
    assert call["max_new_tokens"] == settings.vision_scan_num_predict
    assert call["timeout"] == settings.vision_scan_timeout
    # Prose that mentions code must not be rewritten into a fenced code block.
    assert chunk is not None
    assert chunk.text == transcript
    assert chunk.content_type == "prose"
    assert chunk.visual_type == VisualContentType.SCANNED_TEXT.value


def test_ingestion_gpu_lease_evicts_pinned_ollama_model_and_restores_it(tmp_path):
    from unittest.mock import MagicMock

    from backend.vision import vision_service as vs

    service = _service(tmp_path)
    client = MagicMock()
    client.is_loaded = False
    client.gpu_free_gb.return_value = 1.0
    # No headroom while the chat model is resident, headroom once it is evicted.
    client.has_gpu_headroom.side_effect = [False, True, True]
    loaded = [
        {"name": "qwen2.5:7b", "size_vram": 4_185_758_105, "expires_at": "2319-01-11T11:19:23+05:30"},
        {"name": "cpu-only:1b", "size_vram": 0, "expires_at": "2319-01-11T11:19:23+05:30"},
    ]

    with (
        patch.object(service, "is_available", return_value=(True, "ready")),
        patch.object(vs.HFVisionClient, "get_instance", return_value=client),
        patch.object(vs, "list_loaded_models", side_effect=[loaded, []]),
        patch.object(vs, "unload_model", return_value=True) as unload,
        patch.object(vs, "preload_model", return_value=True) as preload,
    ):
        lease = service.acquire_ingestion_gpu()
        assert lease.evicted == {"qwen2.5:7b": 4_185_758_105}
        assert lease.pinned == ["qwen2.5:7b"]
        unload.assert_called_once_with("qwen2.5:7b")
        preload.assert_not_called()

        service.release_ingestion_gpu(lease)
        client.unload.assert_called_once()
        preload.assert_called_once_with("qwen2.5:7b")


def test_ingestion_gpu_lease_is_a_noop_without_a_gpu(tmp_path):
    from unittest.mock import MagicMock

    from backend.vision import vision_service as vs

    service = _service(tmp_path)
    client = MagicMock()
    client.gpu_free_gb.return_value = None

    with (
        patch.object(service, "is_available", return_value=(True, "ready")),
        patch.object(vs.HFVisionClient, "get_instance", return_value=client),
        patch.object(vs, "list_loaded_models") as list_loaded,
        patch.object(vs, "unload_model") as unload,
    ):
        lease = service.acquire_ingestion_gpu()
        service.release_ingestion_gpu(lease)

    assert lease.evicted == {}
    list_loaded.assert_not_called()
    unload.assert_not_called()
    client.unload.assert_not_called()


def test_scanned_pdf_is_read_through_vision_and_gpu_is_released(tmp_path):
    from backend.ingestion.loaders.pdf import PDFLoader
    from backend.vision.vision_service import IngestionGpuLease, VisualExtractionChunk

    pdf_path = _scanned_pdf(tmp_path, pages=2)
    service = _service(tmp_path)
    lease = IngestionGpuLease(evicted={"qwen2.5:7b": 1})
    progress: list[tuple[int, int]] = []

    def fake_page_visuals(**kwargs):
        assert kwargs["live_inference"] is True
        return [
            VisualExtractionChunk(
                text=f"transcript of page {kwargs['page_number']}",
                content_type="prose",
                visual_type=VisualContentType.SCANNED_TEXT.value,
                page_number=kwargs["page_number"],
            )
        ]

    with (
        patch.object(service, "acquire_ingestion_gpu", return_value=lease),
        patch.object(service, "is_query_time_available", return_value=(True, "Interactive vision is available on cuda.")),
        patch.object(service, "process_pdf_page_visuals", side_effect=fake_page_visuals),
        patch.object(service, "release_ingestion_gpu") as release,
    ):
        documents = PDFLoader(vision_service=service, image_asset_manager=service.image_asset_manager).load(
            pdf_path,
            base_metadata={"document_id": "doc_scan"},
            progress_callback=lambda page, total: progress.append((page, total)),
        )

    assert [doc.content for doc in documents] == ["transcript of page 1", "transcript of page 2"]
    assert all(doc.metadata.extra["is_visual_extraction"] for doc in documents)
    assert progress == [(1, 2), (2, 2)]
    release.assert_called_once_with(lease)


def test_scanned_pdf_failure_names_why_vision_could_not_run(tmp_path):
    import pytest

    from backend.ingestion.loaders.pdf import PDFLoader
    from backend.vision.vision_service import IngestionGpuLease

    pdf_path = _scanned_pdf(tmp_path, pages=1)
    service = _service(tmp_path)
    reason = "CPU-only vision is disabled for interactive queries."

    with (
        patch.object(service, "acquire_ingestion_gpu", return_value=IngestionGpuLease()),
        patch.object(service, "is_query_time_available", return_value=(False, reason)),
        patch.object(service, "release_ingestion_gpu") as release,
        pytest.raises(ValueError, match="No readable text found") as excinfo,
    ):
        PDFLoader(vision_service=service, image_asset_manager=service.image_asset_manager).load(
            pdf_path, base_metadata={"document_id": "doc_scan"}
        )

    assert reason in str(excinfo.value)
    release.assert_called_once()


def test_text_pdf_never_touches_the_gpu_lease(tmp_path):
    import fitz

    from backend.ingestion.loaders.pdf import PDFLoader

    pdf_path = tmp_path / "text.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Leave policy: employees accrue 1.5 days per month.")
    doc.save(pdf_path)
    doc.close()
    service = _service(tmp_path)

    with (
        patch.object(service, "acquire_ingestion_gpu") as acquire,
        patch.object(service, "release_ingestion_gpu") as release,
    ):
        documents = PDFLoader(vision_service=service, image_asset_manager=service.image_asset_manager).load(pdf_path)

    assert len(documents) == 1
    acquire.assert_not_called()
    release.assert_not_called()
