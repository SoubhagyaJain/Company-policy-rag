from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from backend.models.api_dto import IngestionStatusResponse
from backend.services.document_catalog import CatalogConflict, DocumentCatalog

DOC = "doc_123456789abc"


def enqueue(catalog, tmp_path):
    source = tmp_path / f"{DOC}_policy.txt"
    source.write_text("Original policy")
    status = IngestionStatusResponse(document_id=DOC, filename="policy.txt", status="TEXT_INDEXING",
                                     current_stage="QUEUED", text_ready=False)
    record = {"document_id": DOC, "filename": "policy.txt", "file_hash": "hash", "chunk_count": 0}
    catalog.enqueue(record, source, {"category": "HR", "chunk_strategy": "recursive"}, status.model_dump())
    return status


def test_catalog_claim_is_atomic_and_wrong_owner_cannot_publish(tmp_path):
    path = tmp_path / "catalog.sqlite3"
    first, second = DocumentCatalog(path), DocumentCatalog(path)
    status = enqueue(first, tmp_path)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda entry: entry[0].claim(DOC, entry[1], 120), [(first, "one"), (second, "two")]))
    assert sum(result is not None for result in results) == 1
    assert not first.searchable(DOC)
    with pytest.raises(CatalogConflict):
        first.finish(DOC, "wrong-owner", status.model_dump())
    assert first.get(DOC)["state"] == "RUNNING"
    assert first.get(DOC)["options"]["category"] == "HR"


def test_expired_attempts_have_history_and_bounded_recovery(tmp_path):
    now = [100.0]
    catalog = DocumentCatalog(tmp_path / "catalog.sqlite3", clock=lambda: now[0])
    enqueue(catalog, tmp_path)
    for expected in range(1, 4):
        assert catalog.claim(DOC, "worker", 10)["attempt"] == expected
        assert catalog.claim(DOC, "other", 10) is None
        now[0] += 11
    assert catalog.claim(DOC, "worker", 10) is None
    assert catalog.get(DOC)["state"] == "FAILED"
    assert catalog.job(DOC)["status"]["can_retry"]
    with catalog.connection() as db:
        attempts = db.execute("SELECT state FROM attempts ORDER BY number").fetchall()
    assert [row[0] for row in attempts] == ["INTERRUPTED"] * 3


def test_failed_sql_transaction_leaves_no_half_enqueued_document(tmp_path):
    catalog = DocumentCatalog(tmp_path / "catalog.sqlite3")
    status = enqueue(catalog, tmp_path)
    catalog.invalidate(DOC, "test failure")
    with pytest.raises(Exception):
        catalog.enqueue(catalog.get(DOC)["record"], tmp_path / "source", {}, status.model_dump())
    assert catalog.get(DOC)["state"] == "FAILED"
    assert catalog.job(DOC)["state"] == "FAILED"


def test_worker_lock_cannot_be_taken_by_second_catalog_instance(tmp_path):
    first = DocumentCatalog(tmp_path / "catalog.sqlite3")
    second = DocumentCatalog(first.path)
    with first.worker_lock() as acquired:
        assert acquired
        with second.worker_lock() as other_acquired:
            assert not other_acquired
    with second.worker_lock() as acquired:
        assert acquired


def test_catalog_backup_and_schema_guard_preserve_state(tmp_path):
    catalog = DocumentCatalog(tmp_path / "catalog.sqlite3")
    enqueue(catalog, tmp_path)
    backup = tmp_path / "backup.sqlite3"
    catalog.backup(backup)
    assert DocumentCatalog(backup).get(DOC)["state"] == "QUEUED"
    with pytest.raises(FileExistsError):
        catalog.backup(backup)
    with catalog.connection() as db:
        db.execute("PRAGMA user_version=99")
    with pytest.raises(RuntimeError, match="schema"):
        DocumentCatalog(catalog.path)
