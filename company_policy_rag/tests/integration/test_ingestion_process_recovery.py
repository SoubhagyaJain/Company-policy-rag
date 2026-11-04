"""Offline process-death test: real SQLite/Chroma/BM25, injected test embeddings."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

from backend.services.document_catalog import DocumentCatalog
from backend.services.document_service import DocumentService


def test_abrupt_process_exit_after_vector_write_recovers_without_duplicate_rows(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[2]
    script = r"""
import os
import sys
from unittest.mock import MagicMock
from backend.services.document_service import DocumentService
from src.config import settings
settings.ingestion_lease_seconds = 1.0
embeddings, assets = MagicMock(), MagicMock()
embeddings.embed_chunks.side_effect = lambda texts: [[1.0, 0.5] for _ in texts]
assets.list_assets.return_value = []
service = DocumentService(storage_dir=sys.argv[1], embedding_service=embeddings,
                          image_asset_manager=assets, vision_cache_manager=MagicMock())
update = service._update_job_stage
def die(**kwargs):
    update(**kwargs)
    if kwargs["stage"].value == "VECTOR_INDEX" and kwargs["status"] == "COMPLETED":
        os._exit(71)
service._update_job_stage = die
service.upload_document("policy.txt", b"Employees receive twenty days of annual paid leave.", category="HR")
service.close()
raise SystemExit(3)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "uploads")],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 71, result.stderr[-3000:]
    catalog = DocumentCatalog(tmp_path / "document_catalog.sqlite3")
    row = catalog.documents()[0]
    assert row["state"] == "RUNNING"
    expired_time = catalog.job(row["document_id"])["lease_until"] + 1
    monkeypatch.setattr(
        "backend.services.document_service.DocumentCatalog",
        lambda path: DocumentCatalog(path, clock=lambda: expired_time),
    )
    monkeypatch.setattr("backend.api.dependencies.get_telemetry_service", lambda: MagicMock())
    embeddings, assets = MagicMock(), MagicMock()
    embeddings.embed_chunks.side_effect = lambda texts: [[1.0, 0.5] for _ in texts]
    assets.list_assets.return_value = []
    restarted = DocumentService(
        storage_dir=str(tmp_path / "uploads"),
        embedding_service=embeddings,
        image_asset_manager=assets,
        vision_cache_manager=MagicMock(),
    )
    try:
        restarted._ingestion_executor.submit(lambda: None).result(timeout=30)
        status = restarted.get_ingestion_status(row["document_id"])
        assert status.status == "READY", status.error
        assert restarted.vector_store._collection.count() == status.chunks_indexed
        assert len(restarted.bm25_index.entries) == status.chunks_indexed
        assert restarted._stored_files[row["document_id"]].read_bytes() == (
            b"Employees receive twenty days of annual paid leave."
        )
        assert restarted.catalog.job(row["document_id"])["attempt_count"] == 2
    finally:
        restarted.close()
