from __future__ import annotations

import re
from abc import ABC, abstractmethod

from backend.models.chunk import Chunk, ChunkMetadata, ChunkRole, ContentType
from backend.models.document import RawDocument


class BaseChunker(ABC):
    """Abstract Base Class for document chunkers with context-enrichment capabilities."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, documents: list[RawDocument]) -> list[Chunk]:
        """Split RawDocument list into Chunk list according to strategy."""

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (approx 4 chars per token)."""
        return max(1, len(text) // 4)

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace and clean up raw extracted document text."""
        # Replace multiple spaces with single space, preserve intentional paragraph breaks
        cleaned = re.sub(r"[ \t]+", " ", text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

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
        final_sec_title = section_title or doc_meta.section_title
        final_sec_num = section_number or doc_meta.section_number
        final_sec_path = section_path or doc_meta.section_path

        # Determine content type based on document metadata hints if default PROSE was passed
        final_content_type = content_type
        if final_content_type == ContentType.PROSE:
            raw_ct = doc_meta.extra.get("content_type")
            if raw_ct == "code" or doc_meta.has_code:
                final_content_type = ContentType.CODE
            elif raw_ct == "table" or doc_meta.has_tables:
                final_content_type = ContentType.TABLE

        image_assets = [dict(asset) for asset in (doc_meta.image_assets or [])]
        visual_asset_ids = [
            str(asset["asset_id"])
            for asset in image_assets
            if asset.get("asset_id")
        ]
        if doc_meta.extra.get("asset_id"):
            extra_asset_id = str(doc_meta.extra["asset_id"])
            if extra_asset_id not in visual_asset_ids:
                visual_asset_ids.append(extra_asset_id)

        chunk_meta = ChunkMetadata(
            document_id=document.id,
            source_file=doc_meta.source_file,
            file_path=doc_meta.file_path,
            file_hash=doc_meta.file_hash,
            document_type=doc_meta.document_type.value,
            category=doc_meta.category,
            department=doc_meta.department,
            effective_date=doc_meta.effective_date,
            policy_id=doc_meta.policy_id,
            key_entities=list(doc_meta.key_entities) if doc_meta.key_entities else [],
            topic_tags=list(doc_meta.topic_tags) if doc_meta.topic_tags else [],
            chunk_index=chunk_index,
            page_number=doc_meta.page_number,
            internal_page_index=doc_meta.internal_page_index,
            display_page_number=doc_meta.display_page_number,
            page_label=doc_meta.page_label,
            section_title=final_sec_title,
            section_number=final_sec_num,
            section_path=final_sec_path,
            section_level=section_level if section_level is not None else doc_meta.section_level,
            chunk_strategy=strategy_name,
            node_role=node_role,
            parent_id=parent_id,
            child_ids=child_ids or [],
            content_type=final_content_type,
            has_code=doc_meta.has_code or final_content_type == ContentType.CODE,
            has_tables=doc_meta.has_tables or final_content_type == ContentType.TABLE,
            is_atomic=is_atomic,
            image_assets=image_assets,
            visual_asset_ids=visual_asset_ids,
            extra=dict(doc_meta.extra),
        )

        cleaned_text = self._clean_text(text)

        return Chunk(
            text=cleaned_text,
            metadata=chunk_meta,
            token_count=self._estimate_tokens(cleaned_text),
        )
