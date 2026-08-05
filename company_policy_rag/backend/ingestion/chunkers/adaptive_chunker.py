from __future__ import annotations

from typing import List, Optional

from backend.ingestion.chunkers.base import BaseChunker
from backend.ingestion.chunkers.heading_aware import HeadingAwareChunker
from backend.ingestion.chunkers.markdown_aware import MarkdownAwareChunker
from backend.ingestion.chunkers.recursive import RecursiveChunker
from backend.ingestion.chunkers.semantic import SemanticChunker
from backend.ingestion.chunkers.table_aware import TableAwareChunker
from backend.models.chunk import Chunk
from backend.models.document import DocumentType, RawDocument


class AdaptiveChunker(BaseChunker):
    """Adaptive chunker that inspects document properties and selects the optimal chunking strategy."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        override_strategy: Optional[str] = None,
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.override_strategy = override_strategy.lower() if override_strategy else None

        self.recursive_chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.semantic_chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.markdown_chunker = MarkdownAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.heading_chunker = HeadingAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.table_chunker = TableAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def select_chunker(self, document: RawDocument) -> BaseChunker:
        if self.override_strategy:
            strat = self.override_strategy
            if "table" in strat:
                return self.table_chunker
            elif "markdown" in strat:
                return self.markdown_chunker
            elif "heading" in strat:
                return self.heading_chunker
            elif "semantic" in strat:
                return self.semantic_chunker
            elif "recursive" in strat:
                return self.recursive_chunker

        meta = document.metadata
        doc_type = meta.document_type

        if meta.has_tables or "|---" in document.content or ("|" in document.content and "\n|" in document.content):
            return self.table_chunker

        if doc_type in (DocumentType.MARKDOWN, DocumentType.HTML) or meta.has_code or "```" in document.content:
            return self.markdown_chunker

        if meta.section_path is not None or meta.section_title is not None or doc_type in (DocumentType.PDF, DocumentType.DOCX):
            return self.heading_chunker

        if len(document.content.split()) > 200:
            return self.semantic_chunker

        return self.recursive_chunker

    def chunk(self, documents: List[RawDocument]) -> List[Chunk]:
        chunks: List[Chunk] = []

        for doc in documents:
            chunker = self.select_chunker(doc)
            doc_chunks = chunker.chunk([doc])
            chunks.extend(doc_chunks)

        return chunks
