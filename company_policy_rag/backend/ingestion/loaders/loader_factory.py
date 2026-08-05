from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.ingestion.loaders.base import BaseLoader
from backend.ingestion.loaders.csv import CSVLoader
from backend.ingestion.loaders.docx import DocxLoader
from backend.ingestion.loaders.html import HTMLLoader
from backend.ingestion.loaders.json import JSONLoader
from backend.ingestion.loaders.markdown import MarkdownLoader
from backend.ingestion.loaders.pdf import PDFLoader
from backend.ingestion.loaders.txt import TxtLoader
from backend.models.document import RawDocument


class LoaderFactory:
    """Factory registry for document loaders."""

    def __init__(self) -> None:
        self.loaders: List[BaseLoader] = [
            PDFLoader(),
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
        return TxtLoader()

    def load_document(
        self,
        file_path: Path,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RawDocument]:
        """Convenience method to load a document file using the appropriate loader."""
        loader = self.get_loader_for_file(file_path)
        return loader.load(file_path, base_metadata=base_metadata)


# Global factory instance
_default_factory = LoaderFactory()


def get_loader_for_file(file_path: Path) -> BaseLoader:
    return _default_factory.get_loader_for_file(file_path)


def load_document(
    file_path: Path,
    base_metadata: Optional[Dict[str, Any]] = None,
) -> List[RawDocument]:
    return _default_factory.load_document(file_path, base_metadata=base_metadata)
