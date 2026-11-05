from __future__ import annotations

import hashlib
import sqlite3
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.services.document_service import DocumentService
from src.config import settings

CONTENT = b"Employees receive twenty days of annual paid leave. Review travel expenses with HR."


@pytest.fixture
def services(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.api.dependencies.get_telemetry_service", lambda: MagicMock())
    instances = []

    def create(**overrides):
        embeddings, assets = MagicMock(), MagicMock()
        embeddings.embed_chunks.side_effect = lambda texts: [[1.0, 0.5] for _ in texts]
        assets.list_assets.return_value = []
        options = dict(
            storage_dir=str(tmp_path / "uploads"),
            embedding_service=embeddings,
            image_asset_manager=assets,
            vision_cache_manager=MagicMock(),
        )
        options.update(overrides)
        service = DocumentService(**options)
        instances.append(service)
        return service

    yield create
    for service in instances:
        service.close()


def finish(service):
    service._ingestion_executor.submit(lambda: None).result(timeout=20)


class SimulatedWorkerExit(BaseException):
    pass


@pytest.mark.parametrize(
    "stage",
    ["TEXT_EXTRACTION", "SECTION_DETECTION", "CHUNKING", "EMBEDDINGS", "VECTOR_INDEX", "BM25_INDEX", "FINALIZING"],
)
def test_interruption_at_each_stage_recovers_same_id_hash_options_without_duplicate_chunks(
    services, monkeypatch, stage
):
    first = services()
    # Persisted leases are expired at the next real-clock service startup.
    first.catalog.clock = lambda: 100.0
    original = first._update_job_stage

    def crash(**kwargs):
        original(**kwargs)
        if kwargs["stage"].value == stage and kwargs["status"] == "COMPLETED":
            raise SimulatedWorkerExit()

    monkeypatch.setattr(first, "_update_job_stage", crash)
    uploaded = first.upload_document("policy.txt", CONTENT, category="HR", chunk_strategy="recursive")
    finish(first)
    document_id = uploaded.document_id
    assert first.catalog.get(document_id)["state"] == "RUNNING"
    assert not first.catalog.searchable(document_id)
    assert first.vector_store.search([1.0, 0.5]) == []
    assert first.bm25_index.search("annual leave") == []
    monkeypatch.setattr(settings, "ingestion_recovery_enabled", False)
    restarted = services()
    assert restarted.get_ingestion_status(document_id).status != "READY"
    assert not restarted.docstore
    monkeypatch.setattr(settings, "ingestion_recovery_enabled", True)
    restarted.recover_pending_ingestions()
    finish(restarted)
    status = restarted.get_ingestion_status(document_id)
    assert status.status == "READY", status.error
    row = restarted.catalog.get(document_id)
    assert row["options"]["category"] == "HR"
    assert row["options"]["chunk_strategy"] == "recursive"
    assert row["record"]["chunk_strategy"] == "recursive"
    assert row["file_hash"] == hashlib.sha256(CONTENT).hexdigest()
    assert restarted._stored_files[document_id].read_bytes() == CONTENT
    assert restarted.get_document_detail(document_id).category == "HR"
    assert row["expected_chunks"] == restarted.vector_store._collection.count()
    assert len(restarted.bm25_index.entries) == row["expected_chunks"]
    assert len({chunk.id for chunk in restarted.bm25_index.entries}) == row["expected_chunks"]
    with restarted.catalog.connection() as db:
        assert [r[0] for r in db.execute("SELECT state FROM attempts ORDER BY number")] == ["INTERRUPTED", "READY"]


def test_queued_upload_survives_unavailable_executor_and_restarts(services):
    first = services()
    first.close()
    uploaded = first.upload_document("policy.txt", CONTENT)
    assert first.catalog.get(uploaded.document_id)["state"] == "QUEUED"
    restarted = services()
    finish(restarted)
    assert restarted.get_ingestion_status(uploaded.document_id).status == "READY"
    assert restarted._stored_files[uploaded.document_id].read_bytes() == CONTENT


def test_partial_vector_cleanup_failure_cannot_become_ready_after_restart(services, monkeypatch):
    first = services()
    actual_upsert = first.vector_store._collection.upsert

    def partial_write(**kwargs):
        actual_upsert(**kwargs)
        raise OSError("partial write failure")

    monkeypatch.setattr(first.vector_store._collection, "upsert", partial_write)
    actual_delete = first.vector_store._collection.delete
    monkeypatch.setattr(first.vector_store._collection, "delete", MagicMock(side_effect=OSError("cleanup failed")))
    uploaded = first.upload_document("policy.txt", CONTENT, category="HR")
    finish(first)
    assert first.catalog.get(uploaded.document_id)["state"] == "FAILED"
    assert first.vector_store._collection.count() > 0
    assert first.vector_store.search([1.0, 0.5]) == []
    restarted = services()
    assert restarted.get_ingestion_status(uploaded.document_id).status == "FAILED"
    assert not restarted.docstore
    monkeypatch.setattr(first.vector_store._collection, "upsert", actual_upsert)
    monkeypatch.setattr(first.vector_store._collection, "delete", actual_delete)
    restarted.retry_document(uploaded.document_id)
    finish(restarted)
    assert restarted.get_ingestion_status(uploaded.document_id).status == "READY"


def test_changed_original_is_preserved_and_never_automatically_reindexed(services):
    first = services()
    first.close()
    uploaded = first.upload_document("policy.txt", CONTENT)
    source = first._stored_files[uploaded.document_id]
    source.write_bytes(b"Changed source requires review")
    restarted = services()
    finish(restarted)
    status = restarted.get_ingestion_status(uploaded.document_id)
    assert status.status == "FAILED" and "hash" in status.error
    assert restarted.vector_store._collection.count() == 0
    with pytest.raises(ValueError, match="hash"):
        restarted.retry_document(uploaded.document_id)
    assert source.read_bytes() == b"Changed source requires review"
    assert restarted.get_ingestion_status(uploaded.document_id).status == "FAILED"


def test_ready_catalog_survives_restart_and_detects_missing_persistent_vectors(services):
    first = services()
    uploaded = first.upload_document("policy.txt", CONTENT)
    finish(first)
    job_id = first.get_ingestion_status(uploaded.document_id).job_id
    restarted = services()
    assert restarted.get_ingestion_status(uploaded.document_id).job_id == job_id
    assert restarted.get_ingestion_status(uploaded.document_id).status == "READY"
    first.vector_store._collection.delete(where={"document_id": uploaded.document_id})
    damaged = services()
    status = damaged.get_ingestion_status(uploaded.document_id)
    assert status.status == "FAILED" and not status.text_ready
    assert "count" in status.error
    assert damaged.bm25_index.search("annual leave") == []
    assert damaged._stored_files[uploaded.document_id].read_bytes() == CONTENT


def test_legacy_import_is_idempotent_preserves_ids_and_does_not_reindex(services):
    original = services(catalog_enabled=False)
    uploaded = original.upload_document("policy.txt", CONTENT, category="HR")
    finish(original)
    chunks = set(original.docstore)
    original.close()
    imported = services()
    record = imported.get_document_detail(uploaded.document_id)
    assert record.status == "PARTIALLY_INDEXED" and record.text_ready
    assert imported.catalog.get(uploaded.document_id)["state"] == "LEGACY"
    assert imported.catalog.job(uploaded.document_id) is None
    assert set(imported.docstore) == chunks
    assert imported.vector_store._collection.count() == len(chunks)
    imported.embedding_service.embed_chunks.assert_not_called()
    inventory = imported.storage_dir.parent / "catalog_migration_inventory.json"
    saved_inventory = inventory.read_bytes()
    repeated = services()
    assert len(repeated.catalog.documents()) == 1
    assert inventory.read_bytes() == saved_inventory
    assert repeated._stored_files[uploaded.document_id].read_bytes() == CONTENT
    with pytest.raises(ValueError, match="Legacy"):
        repeated.retry_document(uploaded.document_id)


def test_corrupt_catalog_fails_startup_without_replacing_originals(services):
    first = services()
    first.close()
    uploaded = first.upload_document("policy.txt", CONTENT)
    path = first.catalog.path
    path.write_bytes(b"corrupt catalog preserved for diagnosis")
    with pytest.raises(sqlite3.DatabaseError):
        services()
    assert first._stored_files[uploaded.document_id].read_bytes() == CONTENT
    assert path.read_bytes() == b"corrupt catalog preserved for diagnosis"


def test_ambiguous_legacy_originals_are_preserved_for_review(services):
    first = services(catalog_enabled=False)
    uploaded = first.upload_document("policy.txt", CONTENT)
    finish(first)
    other = first.storage_dir / f"{uploaded.document_id}_conflict.txt"
    other.write_bytes(b"Conflicting original")
    imported = services()
    assert imported.catalog.get(uploaded.document_id)["state"] == "REVIEW"
    assert not imported.catalog.searchable(uploaded.document_id)
    assert imported.vector_store.search([1.0, 0.5]) == []
    assert other.read_bytes() == b"Conflicting original"
    assert first._stored_files[uploaded.document_id].read_bytes() == CONTENT
    with pytest.raises(ValueError, match="review"):
        imported.retry_document(uploaded.document_id)


def test_catalog_revocation_invalidates_cached_candidates_and_asset_access(services):
    from backend.api.routes.documents import _require_visible_assets
    from backend.rag.pipeline import RAGPipeline
    from fastapi import HTTPException

    service = services()
    uploaded = service.upload_document("policy.txt", CONTENT)
    finish(service)
    cached_candidates = service.vector_store.search([1.0, 0.5])
    assert cached_candidates
    previous_version = service.vector_store.corpus_version()
    service.catalog.invalidate(uploaded.document_id, "incomplete index")
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.hybrid_retriever = SimpleNamespace(dense_retriever=SimpleNamespace(vector_store=service.vector_store))
    assert pipeline._visible_chunks(cached_candidates) == []
    assert service.vector_store.corpus_version() != previous_version
    with pytest.raises(HTTPException) as exc:
        _require_visible_assets(service, uploaded.document_id)
    assert exc.value.status_code == 404


def test_disabling_catalog_cannot_expose_unpublished_library_data(services):
    first = services()
    first.close()
    uploaded = first.upload_document("policy.txt", CONTENT)
    with pytest.raises(RuntimeError, match="disable the catalog"):
        services(catalog_enabled=False)
    assert first._stored_files[uploaded.document_id].read_bytes() == CONTENT
    assert first.catalog.get(uploaded.document_id)["state"] == "QUEUED"


def test_lease_heartbeat_runs_while_parser_has_no_stage_progress(services, monkeypatch):
    monkeypatch.setattr(settings, "ingestion_lease_seconds", 1.0)
    service = services()
    entered, release, renewed = threading.Event(), threading.Event(), threading.Event()
    execute = service._execute_ingestion_stages
    renew = service.catalog.renew

    def slow_parser(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return execute(*args, **kwargs)

    def heartbeat(*args, **kwargs):
        renew(*args, **kwargs)
        renewed.set()

    monkeypatch.setattr(service, "_execute_ingestion_stages", slow_parser)
    monkeypatch.setattr(service.catalog, "renew", heartbeat)
    uploaded = service.upload_document("policy.txt", CONTENT)
    try:
        assert entered.wait(5)
        initial_deadline = service.catalog.job(uploaded.document_id)["lease_until"]
        assert renewed.wait(5)
        assert service.catalog.job(uploaded.document_id)["lease_until"] > initial_deadline
        assert service.catalog.claim(uploaded.document_id, "other-worker", 1) is None
    finally:
        release.set()
    finish(service)
    assert service.get_ingestion_status(uploaded.document_id).status == "READY"


def test_restart_recovers_expiring_lease_without_browser_polling(services, monkeypatch):
    first = services()
    first.close()
    uploaded = first.upload_document("policy.txt", CONTENT)
    first.catalog.claim(uploaded.document_id, "crashed-worker", 0.5)
    restarted = services()
    completed = threading.Event()
    finish_catalog = restarted.catalog.finish

    def published(*args, **kwargs):
        finish_catalog(*args, **kwargs)
        completed.set()

    monkeypatch.setattr(restarted.catalog, "finish", published)
    assert completed.wait(10)
    assert restarted.catalog.get(uploaded.document_id)["state"] == "READY"
    assert restarted.catalog.job(uploaded.document_id)["attempt_count"] == 2
