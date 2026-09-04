from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from backend.services.document_service import DocumentService


class _LegacyCollection:
    def get(self, include: list[str]) -> dict[str, Any]:
        assert include == ["documents", "metadatas"]
        return {
            "ids": ["legacy_chunk_1"],
            "documents": ["Legacy indexed content"],
            "metadatas": [
                {
                    "document_id": "None",
                    "source_file": "legacy-policy.pdf",
                    "document_type": "pdf",
                    "category": "Policy",
                    "chunk_index": 0,
                }
            ],
        }


class _LegacyVectorStore:
    def __init__(self) -> None:
        self._collection = _LegacyCollection()
        self.deleted_ids: list[str] = []
        self.deleted_sources: list[str] = []

    def delete_by_document_id(self, document_id: str) -> None:
        self.deleted_ids.append(document_id)

    def delete_by_source(self, source_file: str) -> None:
        self.deleted_sources.append(source_file)


class _BM25:
    def __init__(self) -> None:
        self.entries: list[Any] = []

    def build_index(self, chunks: list[Any]) -> None:
        self.entries = list(chunks)

    def remove_by_document_id(self, document_id: str) -> None:
        self.entries = [c for c in self.entries if c.metadata.document_id != document_id]

    def save(self) -> None:
        pass


class _StrictAssets:
    def delete_document_assets(self, document_id: str) -> int:
        if not re.fullmatch(r"doc_[0-9a-f]{12}", document_id):
            raise ValueError(f"Invalid document ID: {document_id}")
        return 0


class _VisionCache:
    def delete_by_document_id(self, document_id: str) -> int:
        return 0


def test_legacy_index_document_gets_canonical_id_and_deletes_by_source(tmp_path: Path) -> None:
    vector_store = _LegacyVectorStore()
    service = DocumentService(
        vector_store=vector_store,  # type: ignore[arg-type]
        bm25_index=_BM25(),  # type: ignore[arg-type]
        embedding_service=object(),  # type: ignore[arg-type]
        image_asset_manager=_StrictAssets(),  # type: ignore[arg-type]
        vision_cache_manager=_VisionCache(),  # type: ignore[arg-type]
        storage_dir=str(tmp_path / "uploads"),
    )

    listed = service.list_documents()
    assert listed.total_count == 1
    document_id = listed.documents[0].document_id
    assert re.fullmatch(r"doc_[0-9a-f]{12}", document_id)
    assert document_id != "None"

    result = service.delete_document(document_id)

    assert result is not None
    assert result["status"] == "deleted"
    assert vector_store.deleted_ids == []
    assert vector_store.deleted_sources == ["legacy-policy.pdf"]
    assert service.list_documents().total_count == 0
