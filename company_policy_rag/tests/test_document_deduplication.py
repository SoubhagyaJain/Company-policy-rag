from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backend.services.document_service import DocumentService, DuplicateDocumentError


class _Collection:
    def __init__(self, metadatas: list[dict[str, Any]]) -> None:
        self.metadatas = metadatas

    def get(self, include: list[str]) -> dict[str, Any]:
        assert include == ["metadatas"]
        return {"metadatas": self.metadatas}


class _VectorStore:
    def __init__(self, metadatas: list[dict[str, Any]]) -> None:
        self._collection = _Collection(metadatas)
        self.deleted: list[str] = []

    def delete_by_document_id(self, document_id: str) -> None:
        self.deleted.append(document_id)


class _BM25:
    def __init__(self) -> None:
        self.entries: list[Any] = []
        self.deleted: list[str] = []
        self.saved = 0

    def remove_by_document_id(self, document_id: str) -> None:
        self.deleted.append(document_id)

    def save(self) -> None:
        self.saved += 1


class _Assets:
    def delete_document_assets(self, document_id: str) -> int:
        return 0


class _VisionCache:
    def delete_by_document_id(self, document_id: str) -> int:
        return 0


def _service(tmp_path: Path, records: list[dict[str, Any]]) -> DocumentService:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    for record in records:
        (uploads / f"{record['document_id']}_{record['source_file']}").write_bytes(record["content"])
    metadatas = [
        {
            "document_id": record["document_id"],
            "source_file": record["source_file"],
            "document_type": "txt",
            "category": "Test",
            "page_number": 1,
        }
        for record in records
        if record.get("indexed", True)
    ]
    return DocumentService(
        vector_store=_VectorStore(metadatas),  # type: ignore[arg-type]
        bm25_index=_BM25(),  # type: ignore[arg-type]
        embedding_service=object(),  # type: ignore[arg-type]
        image_asset_manager=_Assets(),  # type: ignore[arg-type]
        vision_cache_manager=_VisionCache(),  # type: ignore[arg-type]
        storage_dir=str(uploads),
    )


def test_upload_rejects_existing_exact_content(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        [{"document_id": "doc_aaaaaaaaaaaa", "source_file": "policy.txt", "content": b"same bytes"}],
    )

    with pytest.raises(DuplicateDocumentError) as exc_info:
        service.upload_document("renamed-policy.txt", b"same bytes")

    assert exc_info.value.document_id == "doc_aaaaaaaaaaaa"
    assert exc_info.value.file_hash == service._documents["doc_aaaaaaaaaaaa"]["file_hash"]
    assert len(list((tmp_path / "uploads").iterdir())) == 1


def test_deduplicate_removes_only_exact_duplicate_across_layers(tmp_path: Path) -> None:
    records = [
        {"document_id": "doc_aaaaaaaaaaaa", "source_file": "old.txt", "content": b"identical"},
        {"document_id": "doc_bbbbbbbbbbbb", "source_file": "new.txt", "content": b"identical"},
        {"document_id": "doc_cccccccccccc", "source_file": "other.txt", "content": b"different"},
    ]
    service = _service(tmp_path, records)

    preview = service.deduplicate_documents(dry_run=True)
    assert preview["duplicate_groups"] == 1
    assert preview["duplicates_found"] == 1
    assert preview["duplicates_removed"] == 0

    result = service.deduplicate_documents(dry_run=False)
    assert result["duplicates_removed"] == 1
    assert len(list((tmp_path / "uploads").glob("doc_*_*"))) == 2
    assert service.get_duplicate_groups() == []
    removed_id = result["removed"][0]["document_id"]
    assert removed_id in service.vector_store.deleted
    assert removed_id in service.bm25_index.deleted
    assert service.bm25_index.saved == 1


def test_same_filename_with_different_content_is_preserved(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        [
            {"document_id": "doc_aaaaaaaaaaaa", "source_file": "policy.txt", "content": b"version one"},
            {"document_id": "doc_bbbbbbbbbbbb", "source_file": "policy.txt", "content": b"version two"},
        ],
    )

    assert service.get_duplicate_groups() == []
    assert service.deduplicate_documents(dry_run=False)["duplicates_removed"] == 0
    assert len(list((tmp_path / "uploads").glob("doc_*_*"))) == 2
