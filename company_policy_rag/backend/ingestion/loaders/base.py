from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.models.document import DocumentCategory, DocumentMetadata, DocumentType, RawDocument
from backend.utils.hashing import compute_file_hash


class BaseLoader(ABC):
    """Abstract Base Class for document loaders."""

    @abstractmethod
    def load(
        self,
        file_path: Path,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RawDocument]:
        """Load file content into RawDocument instances preserving hierarchy and metadata."""
        pass

    @abstractmethod
    def supports(self, file_path: Path) -> bool:
        """Return True if loader handles this file extension or format."""
        pass

    def _build_base_metadata(
        self,
        file_path: Path,
        doc_type: DocumentType,
        custom_meta: Optional[Dict[str, Any]] = None,
    ) -> DocumentMetadata:
        rel_path = str(file_path)
        file_hash = compute_file_hash(file_path) if file_path.exists() else "nofile"
        category = "general"
        if custom_meta and "category" in custom_meta:
            category = custom_meta["category"]

        meta_dict: Dict[str, Any] = {
            "source_file": file_path.name,
            "file_path": rel_path,
            "file_hash": file_hash,
            "document_type": doc_type,
            "category": category,
        }
        if custom_meta:
            for k, v in custom_meta.items():
                if k not in meta_dict:
                    meta_dict[k] = v

        return DocumentMetadata(**meta_dict)
