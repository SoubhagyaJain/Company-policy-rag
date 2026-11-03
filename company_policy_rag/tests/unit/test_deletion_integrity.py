from unittest.mock import MagicMock

import pytest

from backend.embeddings.vector_store import ChromaVectorStore
from backend.models.chunk import Chunk, ChunkMetadata
from backend.services.document_service import DocumentService


def test_persistent_delete_failure_is_reported_and_hidden_after_restart(tmp_path, monkeypatch):
    store = ChromaVectorStore(persist_dir=tmp_path / "chroma")
    chunk = Chunk(id="chunk", text="Employees receive twenty days of annual leave.",
                  metadata=ChunkMetadata(document_id="doc_123456789abc", source_file="policy.txt", chunk_index=0),
                  embedding=[1.0, 0.5])
    store.add_chunks([chunk])
    original_delete = store._collection.delete
    monkeypatch.setattr(store._collection, "delete", MagicMock(side_effect=OSError("disk unavailable")))
    with pytest.raises(RuntimeError, match="delete"):
        store.delete_by_document_id(chunk.metadata.document_id)
    assert store.search([1.0, 0.5]) == []
    restarted = ChromaVectorStore(persist_dir=tmp_path / "chroma")
    assert restarted.search([1.0, 0.5]) == []
    monkeypatch.setattr(store._collection, "delete", original_delete)
    restarted.delete_by_document_id(chunk.metadata.document_id)
    assert restarted._collection.count() == 0


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.api.dependencies.get_telemetry_service", lambda: MagicMock())
    embeddings = MagicMock()
    embeddings.embed_chunks.side_effect = lambda texts: [[1.0, 0.5] for _ in texts]
    result = DocumentService(storage_dir=str(tmp_path / "uploads"), embedding_service=embeddings)
    yield result
    result._ingestion_executor.shutdown(wait=True)


def test_partial_delete_hides_lexical_and_parent_evidence_and_retains_source(service, monkeypatch):
    uploaded = service.upload_document("policy.txt", b"Employees receive twenty days of annual paid leave.")
    service._ingestion_executor.submit(lambda: None).result(timeout=15)
    source = service._stored_files[uploaded.document_id]
    monkeypatch.setattr(service.vector_store._collection, "delete", MagicMock(side_effect=OSError("disk unavailable")))
    with pytest.raises(RuntimeError):
        service.delete_document(uploaded.document_id)
    assert not service.docstore
    assert not service.bm25_index.entries
    assert source.exists()  # recoverable cleanup, not data destruction on failure
    restarted = DocumentService(storage_dir=str(service.storage_dir), embedding_service=service.embedding_service)
    try:
        assert not restarted.docstore
        assert not restarted.bm25_index.entries
        assert restarted.get_document_detail(uploaded.document_id).status == "DELETING"
    finally:
        restarted._ingestion_executor.shutdown(wait=True)
    with pytest.raises(ValueError, match="deletion"):
        service.retry_document(uploaded.document_id)


def test_active_ingestion_cannot_be_deleted(service):
    from backend.models.api_dto import IngestionStatusResponse
    service._ingestion_jobs["doc_123456789abc"] = IngestionStatusResponse(
        document_id="doc_123456789abc", filename="policy.txt", status="TEXT_INDEXING")
    with pytest.raises(RuntimeError, match="index"):
        service.delete_document("doc_123456789abc")


def test_failed_ingestion_cleanup_does_not_revoke_retry(service, monkeypatch):
    original_save = service.bm25_index.save
    monkeypatch.setattr(service.bm25_index, "save", MagicMock(side_effect=OSError("disk unavailable")))
    uploaded = service.upload_document("policy.txt", b"Employees receive twenty days of annual paid leave.")
    service._ingestion_executor.submit(lambda: None).result(timeout=15)
    assert service.get_ingestion_status(uploaded.document_id).status == "FAILED"
    assert not service.vector_store.is_document_deleted(uploaded.document_id)
    monkeypatch.setattr(service.bm25_index, "save", original_save)
    service.retry_document(uploaded.document_id)
    service._ingestion_executor.submit(lambda: None).result(timeout=15)
    assert service.get_ingestion_status(uploaded.document_id).status == "READY"


def test_api_reports_incomplete_delete_and_allows_cleanup_retry(service, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.api.main import app
    from backend.api.dependencies import get_document_service, get_semantic_cache_manager, get_conversation_state_manager
    uploaded = service.upload_document("policy.txt", b"Employees receive twenty days of annual paid leave.")
    service._ingestion_executor.submit(lambda: None).result(timeout=15)
    original_delete = service.vector_store._collection.delete
    monkeypatch.setattr(service.vector_store._collection, "delete", MagicMock(side_effect=OSError("disk unavailable")))
    monkeypatch.setitem(app.dependency_overrides, get_document_service, lambda: service)
    monkeypatch.setitem(app.dependency_overrides, get_semantic_cache_manager, lambda: MagicMock())
    monkeypatch.setitem(app.dependency_overrides, get_conversation_state_manager, lambda: MagicMock())
    client = TestClient(app, client=("127.0.0.1", 50000))
    url = f"/api/documents/{uploaded.document_id}"
    response = client.delete(url)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "DOCUMENT_DELETE_INCOMPLETE"
    assert client.get(url + "/assets").status_code == 404
    monkeypatch.setattr(service.vector_store._collection, "delete", original_delete)
    assert client.delete(url).json()["status"] == "deleted"


def test_pipeline_filters_revoked_cached_and_reused_evidence(tmp_path):
    from types import SimpleNamespace
    from backend.models.rag import ScoredChunk
    from backend.rag.pipeline import RAGPipeline
    store = ChromaVectorStore(persist_dir=tmp_path / "chroma")
    chunk = Chunk(id="old", text="Old policy", metadata=ChunkMetadata(
        document_id="doc_123456789abc", source_file="policy.txt", chunk_index=0))
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.hybrid_retriever = SimpleNamespace(dense_retriever=SimpleNamespace(vector_store=store))
    store.mark_document_deleted(chunk.metadata.document_id)
    assert pipeline._visible_chunks([ScoredChunk(chunk=chunk, score=1)]) == []
