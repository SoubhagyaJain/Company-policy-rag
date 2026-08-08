from __future__ import annotations

import re

from backend.ingestion.chunkers.base import BaseChunker
from backend.models.chunk import Chunk, ContentType
from backend.models.document import RawDocument
from backend.utils.section_tracker import SectionTracker, parse_section_heading

_FENCE_SPLIT_REGEX = re.compile(r"(```[\s\S]*?```)")


class MarkdownAwareChunker(BaseChunker):
    """Markdown-aware chunker protecting code blocks and respecting header boundaries."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk(self, documents: list[RawDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for doc in documents:
            if not doc.content.strip():
                continue

            section_tracker = SectionTracker()
            chunk_idx = 0

            # First split by fenced code blocks vs normal markdown
            parts = _FENCE_SPLIT_REGEX.split(doc.content)

            for part in parts:
                part_stripped = part.strip()
                if not part_stripped:
                    continue

                if part_stripped.startswith("```") and part_stripped.endswith("```"):
                    # Atomic Code Block
                    ctx = section_tracker.current_context()
                    c = self._create_chunk(
                        text=part_stripped,
                        document=doc,
                        chunk_index=chunk_idx,
                        strategy_name="markdown_aware",
                        section_title=ctx.section_title,
                        section_number=ctx.section_number,
                        section_path=ctx.section_path,
                        section_level=ctx.section_level,
                        content_type=ContentType.CODE,
                        is_atomic=True,
                    )
                    chunks.append(c)
                    chunk_idx += 1
                else:
                    # Normal markdown text
                    lines = part_stripped.splitlines()
                    current_block: list[str] = []

                    for line in lines:
                        heading = parse_section_heading(line)
                        if heading:
                            if current_block:
                                block_text = "\n".join(current_block).strip()
                                if block_text:
                                    ctx = section_tracker.current_context()
                                    c = self._create_chunk(
                                        text=block_text,
                                        document=doc,
                                        chunk_index=chunk_idx,
                                        strategy_name="markdown_aware",
                                        section_title=ctx.section_title,
                                        section_number=ctx.section_number,
                                        section_path=ctx.section_path,
                                        section_level=ctx.section_level,
                                        content_type=ContentType.PROSE,
                                    )
                                    chunks.append(c)
                                    chunk_idx += 1
                                current_block = []

                            ctx = section_tracker.update(heading)

                        current_block.append(line)

                    if current_block:
                        block_text = "\n".join(current_block).strip()
                        if block_text:
                            ctx = section_tracker.current_context()
                            c = self._create_chunk(
                                text=block_text,
                                document=doc,
                                chunk_index=chunk_idx,
                                strategy_name="markdown_aware",
                                section_title=ctx.section_title,
                                section_number=ctx.section_number,
                                section_path=ctx.section_path,
                                section_level=ctx.section_level,
                                content_type=ContentType.PROSE,
                            )
                            chunks.append(c)
                            chunk_idx += 1

        return chunks
