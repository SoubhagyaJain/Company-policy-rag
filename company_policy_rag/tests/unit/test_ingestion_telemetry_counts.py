"""Ingestion telemetry must report real asset and vision counts, not page counts."""

from __future__ import annotations

from unittest.mock import MagicMock

from backend.services.document_service import DocumentService


def test_text_upload_reports_no_visual_assets_or_vision_work(tmp_path, monkeypatch) -> None:
    telemetry = MagicMock()
    monkeypatch.setattr("backend.api.dependencies.get_telemetry_service", lambda: telemetry)
    embeddings = MagicMock()
    embeddings.embed_chunks.side_effect = lambda texts: [[0.1, 0.2] for _ in texts]
    assets = MagicMock()
    assets.list_assets.return_value = []
    service = DocumentService(
        storage_dir=str(tmp_path / "uploads"),
        embedding_service=embeddings,
        image_asset_manager=assets,
        vision_cache_manager=MagicMock(),
    )
    try:
        upload = service.upload_document("policy.txt", b"Employees receive twenty days of annual paid leave.")
        service._ingestion_executor.submit(lambda: None).result(timeout=15)
        assert service.get_ingestion_status(upload.document_id).status == "READY"
    finally:
        service._ingestion_executor.shutdown(wait=True)

    trace = telemetry.record_ingestion_trace.call_args.kwargs
    assert trace["status"] == "READY"
    assert trace["pages_count"] == 1
    assert trace["visual_assets_count"] == 0
    assert trace["vision_success_count"] == 0
