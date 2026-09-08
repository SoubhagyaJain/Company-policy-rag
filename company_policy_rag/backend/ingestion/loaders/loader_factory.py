from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.ingestion.loaders.base import BaseLoader
from backend.ingestion.loaders.csv import CSVLoader
from backend.ingestion.loaders.docx import DocxLoader
from backend.ingestion.loaders.html import HTMLLoader
from backend.ingestion.loaders.json import JSONLoader
from backend.ingestion.loaders.markdown import MarkdownLoader
from backend.ingestion.loaders.pdf import PDFLoader
from backend.ingestion.loaders.office import SpreadsheetLoader, PresentationLoader
from backend.ingestion.loaders.validation import MAX_DOCUMENT_BYTES
from backend.ingestion.loaders.txt import TxtLoader
from backend.models.document import RawDocument


class LoaderFactory:
    """Factory registry for document loaders."""

    def __init__(self) -> None:
        self.loaders: list[BaseLoader] = [
            PDFLoader(),
            SpreadsheetLoader(),
            PresentationLoader(),
            DocxLoader(),
            MarkdownLoader(),
            HTMLLoader(),
            CSVLoader(),
            JSONLoader(),
            TxtLoader(),  # Fallback for plain text files
        ]

    def register_loader(self, loader: BaseLoader) -> None:
        """Register a new loader at high priority."""
        self.loaders.insert(0, loader)

    def get_loader_for_file(self, file_path: Path) -> BaseLoader:
        """Find the matching loader for a file path."""
        for loader in self.loaders:
            if loader.supports(file_path):
                return loader
        raise ValueError(
            f"Unsupported document format: {file_path.suffix or '(no extension)'}. Export to PDF, DOCX, XLSX, PPTX, TXT, MD, HTML, CSV, TSV, JSON or JSONL."
        )

    def load_document(
        self,
        file_path: Path,
        base_metadata: dict[str, Any] | None = None,
    ) -> list[RawDocument]:
        """Convenience method to load a document file using the appropriate loader."""
        loader = self.get_loader_for_file(file_path)
        if file_path.stat().st_size > MAX_DOCUMENT_BYTES:
            raise ValueError("Document exceeds the 100MB limit.")
        documents = loader.load(file_path, base_metadata=base_metadata)
        if not any(doc.content.strip() for doc in documents):
            raise ValueError(
                "No readable text found. For scans or image-only documents, run OCR and upload a searchable PDF."
            )
        return documents


# Global factory instance
_default_factory = LoaderFactory()


def get_loader_for_file(file_path: Path) -> BaseLoader:
    return _default_factory.get_loader_for_file(file_path)


def load_document(
    file_path: Path,
    base_metadata: dict[str, Any] | None = None,
) -> list[RawDocument]:
    return _default_factory.load_document(file_path, base_metadata=base_metadata)
