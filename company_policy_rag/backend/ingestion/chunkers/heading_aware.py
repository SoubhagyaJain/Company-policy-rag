from __future__ import annotations

from backend.ingestion.chunkers.base import BaseChunker
from backend.models.chunk import Chunk, ContentType
from backend.models.document import RawDocument
from backend.utils.section_tracker import SectionTracker, parse_section_heading


class HeadingAwareChunker(BaseChunker):
    """Hierarchical heading-aware chunker for legal, policy, and compliance documents."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk(self, documents: list[RawDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for doc in documents:
            if not doc.content.strip():
                continue

            lines = doc.content.splitlines()
            section_tracker = SectionTracker()
            chunk_idx = 0

            current_lines: list[str] = []
            current_ctx = section_tracker.current_context()

            for line in lines:
                heading = parse_section_heading(line)
                if heading:
                    # Flush current lines as chunk
                    if current_lines:
                        text_block = "\n".join(current_lines).strip()
                        if text_block:
                            c = self._create_chunk(
                                text=text_block,
                                document=doc,
                                chunk_index=chunk_idx,
                                strategy_name="heading_aware",
                                section_title=current_ctx.section_title,
                                section_number=current_ctx.section_number,
                                section_path=current_ctx.section_path,
                                section_level=current_ctx.section_level,
                                content_type=ContentType.PROSE,
                            )
                            chunks.append(c)
                            chunk_idx += 1
                        current_lines = []

                    current_ctx = section_tracker.update(heading)

                current_lines.append(line)

            if current_lines:
                text_block = "\n".join(current_lines).strip()
                if text_block:
                    c = self._create_chunk(
                        text=text_block,
                        document=doc,
                        chunk_index=chunk_idx,
                        strategy_name="heading_aware",
                        section_title=current_ctx.section_title,
                        section_number=current_ctx.section_number,
                        section_path=current_ctx.section_path,
                        section_level=current_ctx.section_level,
                        content_type=ContentType.PROSE,
                    )
                    chunks.append(c)

        return chunks
