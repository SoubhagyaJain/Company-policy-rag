from unittest.mock import MagicMock

from backend.services.document_service import DocumentService


def test_fresh_start_isolates_previous_uploads_and_indexes(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.api.dependencies.get_telemetry_service", lambda: MagicMock())
    embeddings = MagicMock()
    embeddings.embed_chunks.side_effect = lambda texts: [[0.1, 0.2] for _ in texts]
    assets = MagicMock()
    assets.list_assets.return_value = []
    options = dict(
        storage_dir=str(tmp_path / "uploads"),
        embedding_service=embeddings,
        image_asset_manager=assets,
        vision_cache_manager=MagicMock(),
        fresh_start=True,
    )
    first = DocumentService(**options)
    second = None
    content = b"Employees receive twenty days of annual paid leave."
    try:
        upload = first.upload_document("policy.txt", content)
        first._ingestion_executor.submit(lambda: None).result(timeout=15)
        assert first.get_ingestion_status(upload.document_id).status == "READY"
        stored_file = first._stored_files[upload.document_id]

        second = DocumentService(**options)
        assert second.list_documents().total_count == 0
        assert second.docstore == {}
        assert second.bm25_index.entries == []
        assert second.vector_store._collection.count() == 0
        assert second.vector_store.persist_dir != first.vector_store.persist_dir
        assert stored_file.read_bytes() == content

        # Identical files can be uploaded again in the new run.
        second_upload = second.upload_document("policy.txt", content)
        second._ingestion_executor.submit(lambda: None).result(timeout=15)
        assert second.get_ingestion_status(second_upload.document_id).status == "READY"
    finally:
        first._ingestion_executor.shutdown(wait=True)
        if second is not None:
            second._ingestion_executor.shutdown(wait=True)


def test_api_library_mode_and_matching_cache_directory(monkeypatch, tmp_path):
    from backend.api import dependencies
    from src.config import settings

    service = MagicMock()
    service.vector_store.persist_dir = tmp_path / "session" / "chroma"
    constructor = MagicMock(return_value=service)
    cache_constructor = MagicMock()
    monkeypatch.setattr(dependencies, "_document_service", None)
    monkeypatch.setattr(dependencies, "_semantic_cache_manager", None)
    monkeypatch.setattr(dependencies, "DocumentService", constructor)
    monkeypatch.setattr(dependencies, "SemanticCacheManager", cache_constructor)
    monkeypatch.setattr(settings, "document_library_mode", "session")
    assert dependencies.get_document_service() is service
    assert dependencies.get_document_service() is service
    constructor.assert_called_once_with(fresh_start=True)
    dependencies.get_semantic_cache_manager()
    assert cache_constructor.call_args.kwargs["persist_dir"] == service.vector_store.persist_dir


def _service_options(tmp_path):
    embeddings = MagicMock()
    embeddings.embed_chunks.side_effect = lambda texts: [[0.1, 0.2] for _ in texts]
    assets = MagicMock()
    assets.list_assets.return_value = []
    return dict(
        storage_dir=str(tmp_path / "uploads"),
        embedding_service=embeddings,
        image_asset_manager=assets,
        vision_cache_manager=MagicMock(),
    )


def test_persistent_library_survives_a_restart(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.api.dependencies.get_telemetry_service", lambda: MagicMock())
    options = _service_options(tmp_path)
    first = DocumentService(**options)
    restarted = None
    try:
        upload = first.upload_document("policy.txt", b"Employees receive twenty days of annual paid leave.")
        first._ingestion_executor.submit(lambda: None).result(timeout=15)
        assert first.get_ingestion_status(upload.document_id).status == "READY"
        first_chunks = set(first.docstore)

        restarted = DocumentService(**options)
        listed = restarted.list_documents()
        assert [doc.document_id for doc in listed.documents] == [upload.document_id]
        assert set(restarted.docstore) == first_chunks
        assert {chunk.id for chunk in restarted.bm25_index.entries} == first_chunks
        assert restarted.vector_store.persist_dir == tmp_path / "chroma"
        assert not (tmp_path / "sessions").exists()
    finally:
        first._ingestion_executor.shutdown(wait=True)
        if restarted is not None:
            restarted._ingestion_executor.shutdown(wait=True)


def test_vector_write_failure_fails_the_upload_and_cleans_indexes(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.api.dependencies.get_telemetry_service", lambda: MagicMock())
    service = DocumentService(**_service_options(tmp_path))
    try:
        collection = service.vector_store._collection

        def broken_upsert(**kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(collection, "upsert", broken_upsert)
        upload = service.upload_document("policy.txt", b"Employees receive twenty days of annual paid leave.")
        service._ingestion_executor.submit(lambda: None).result(timeout=15)

        status = service.get_ingestion_status(upload.document_id)
        assert status.status == "FAILED"
        assert "Vector index write failed" in (status.error or "")
        assert service.docstore == {}
        assert service.bm25_index.entries == []
        assert service.vector_store.count() == 0
    finally:
        service._ingestion_executor.shutdown(wait=True)
