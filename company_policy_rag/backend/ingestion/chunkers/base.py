from __future__ import annotations

from abc import ABC, abstractmethod

from backend.models.chunk import Chunk, ChunkMetadata, ChunkRole, ContentType
from backend.models.document import RawDocument


class BaseChunker(ABC):
    """Abstract Base Class for document chunkers."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, documents: list[RawDocument]) -> list[Chunk]:
        """Split RawDocument list into Chunk list according to strategy."""

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (approx 4 chars per token)."""
        return max(1, len(text) // 4)

    def _create_chunk(
        self,
        text: str,
        document: RawDocument,
        chunk_index: int,
        strategy_name: str,
        section_title: str | None = None,
        section_number: str | None = None,
        section_path: str | None = None,
        section_level: int | None = None,
        node_role: ChunkRole = ChunkRole.STANDALONE,
        parent_id: str | None = None,
        child_ids: list[str] | None = None,
        content_type: ContentType = ContentType.PROSE,
        is_atomic: bool = False,
    ) -> Chunk:
        doc_meta = document.metadata
        chunk_meta = ChunkMetadata(
            document_id=document.id,
            source_file=doc_meta.source_file,
            file_path=doc_meta.file_path,
            file_hash=doc_meta.file_hash,
            document_type=doc_meta.document_type.value,
            category=doc_meta.category,
            chunk_index=chunk_index,
            page_number=doc_meta.page_number,
            section_title=section_title or doc_meta.section_title,
            section_number=section_number or doc_meta.section_number,
            section_path=section_path or doc_meta.section_path,
            section_level=section_level if section_level is not None else doc_meta.section_level,
            chunk_strategy=strategy_name,
            node_role=node_role,
            parent_id=parent_id,
            child_ids=child_ids or [],
            content_type=content_type,
            is_atomic=is_atomic,
            extra=dict(doc_meta.extra),
        )

        return Chunk(
            text=text.strip(),
            metadata=chunk_meta,
            token_count=self._estimate_tokens(text),
        )
