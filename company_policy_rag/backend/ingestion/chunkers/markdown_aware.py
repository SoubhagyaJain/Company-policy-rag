from __future__ import annotations

import re

from backend.ingestion.chunkers.base import BaseChunker
from backend.ingestion.chunkers.recursive import RecursiveChunker
from backend.models.chunk import Chunk, ContentType
from backend.models.document import RawDocument
from backend.utils.section_tracker import SectionTracker, parse_section_heading

_FENCE_SPLIT_REGEX = re.compile(r"(```[\s\S]*?```)")


class MarkdownAwareChunker(BaseChunker):
    """Markdown-aware chunker protecting code blocks and respecting hierarchical header boundaries."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.recursive_helper = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk(self, documents: list[RawDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []
        min_chunk_chars = 60
        max_chunk_chars = self.chunk_size * 4

        for doc in documents:
            if not doc.content.strip():
                continue

            section_tracker = SectionTracker()
            chunk_idx = 0

            # Split document by fenced code blocks vs standard markdown text
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
                    # Markdown text: parse lines and headings
                    lines = part_stripped.splitlines()
                    current_block: list[str] = []
                    current_ctx = section_tracker.current_context()

                    def flush_block(text_lines: list[str], ctx) -> list[Chunk]:
                        nonlocal chunk_idx
                        raw = "\n".join(text_lines).strip()
                        if not raw or len(raw) < 30:
                            return []

                        res: list[Chunk] = []
                        if len(raw) <= max_chunk_chars:
                            c = self._create_chunk(
                                text=raw,
                                document=doc,
                                chunk_index=chunk_idx,
                                strategy_name="markdown_aware",
                                section_title=ctx.section_title,
                                section_number=ctx.section_number,
                                section_path=ctx.section_path,
                                section_level=ctx.section_level,
                                content_type=ContentType.PROSE,
                            )
                            res.append(c)
                            chunk_idx += 1
                        else:
                            splits = self.recursive_helper._split_text(raw, self.recursive_helper.separators)
                            for s in splits:
                                s_clean = s.strip()
                                if s_clean and len(s_clean) >= 30:
                                    c = self._create_chunk(
                                        text=s_clean,
                                        document=doc,
                                        chunk_index=chunk_idx,
                                        strategy_name="markdown_aware",
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
                            accumulated = "\n".join(current_block).strip()
                            if len(accumulated) >= min_chunk_chars:
                                chunks.extend(flush_block(current_block, current_ctx))
                                current_block = []
                            current_ctx = section_tracker.update(heading)

                        current_block.append(line)

                    if current_block:
                        chunks.extend(flush_block(current_block, current_ctx))

        return chunks
