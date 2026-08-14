from __future__ import annotations

from backend.ingestion.chunkers.base import BaseChunker
from backend.ingestion.chunkers.recursive import RecursiveChunker
from backend.models.chunk import Chunk, ContentType
from backend.models.document import RawDocument
from backend.utils.section_tracker import SectionTracker, parse_section_heading


class HeadingAwareChunker(BaseChunker):
    """Hierarchical heading-aware chunker for legal, policy, and compliance documents."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.recursive_helper = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk(self, documents: list[RawDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []
        min_chunk_chars = 100  # Enforce minimum substantive length
        max_chunk_chars = self.chunk_size * 4

        for doc in documents:
            if not doc.content.strip():
                continue

            lines = doc.content.splitlines()
            section_tracker = SectionTracker()
            chunk_idx = 0

            current_lines: list[str] = []
            current_ctx = section_tracker.current_context()

            def flush_section(text_lines: list[str], ctx) -> list[Chunk]:
                nonlocal chunk_idx
                raw_text = "\n".join(text_lines).strip()
                if not raw_text or len(raw_text) < 40:
                    return []

                res: list[Chunk] = []
                if len(raw_text) <= max_chunk_chars:
                    c = self._create_chunk(
                        text=raw_text,
                        document=doc,
                        chunk_index=chunk_idx,
                        strategy_name="heading_aware",
                        section_title=ctx.section_title,
                        section_number=ctx.section_number,
                        section_path=ctx.section_path,
                        section_level=ctx.section_level,
                        content_type=ContentType.PROSE,
                    )
                    res.append(c)
                    chunk_idx += 1
                else:
                    # Section is large: split recursively with overlap while preserving section metadata
                    sub_splits = self.recursive_helper._split_text(raw_text, self.recursive_helper.separators)
                    for split_text in sub_splits:
                        s_clean = split_text.strip()
                        if s_clean and len(s_clean) >= 40:
                            c = self._create_chunk(
                                text=s_clean,
                                document=doc,
                                chunk_index=chunk_idx,
                                strategy_name="heading_aware",
                                section_title=ctx.section_title,
                                section_number=ctx.section_number,
                                section_path=ctx.section_path,
                                section_level=ctx.section_level,
                                content_type=ContentType.PROSE,
                            )
                            res.append(c)
                            chunk_idx += 1
                return res

            for line in lines:
                heading = parse_section_heading(line)
                if heading:
                    # Only flush if the accumulated lines have enough substantive text
                    accumulated = "\n".join(current_lines).strip()
                    if len(accumulated) >= min_chunk_chars:
                        chunks.extend(flush_section(current_lines, current_ctx))
                        current_lines = []
                    current_ctx = section_tracker.update(heading)

                current_lines.append(line)

            if current_lines:
                chunks.extend(flush_section(current_lines, current_ctx))

        return chunks
