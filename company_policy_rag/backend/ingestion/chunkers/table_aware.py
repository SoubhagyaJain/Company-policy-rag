from __future__ import annotations

import re

from backend.ingestion.chunkers.base import BaseChunker
from backend.models.chunk import Chunk, ContentType
from backend.models.document import RawDocument

_TABLE_BLOCK_REGEX = re.compile(r"((?:^\|[^\n]+\|\n)+)", re.MULTILINE)


class TableAwareChunker(BaseChunker):
    """Table-aware chunker preserving table structure and prepending header context."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk(self, documents: list[RawDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for doc in documents:
            if not doc.content.strip():
                continue

            lines = doc.content.splitlines()
            in_table = False
            table_lines: list[str] = []
            prose_lines: list[str] = []
            chunk_idx = 0

            for line in lines:
                is_table_line = line.strip().startswith("|") and line.strip().endswith("|")

                if is_table_line:
                    if prose_lines:
                        prose_text = "\n".join(prose_lines).strip()
                        if prose_text:
                            c = self._create_chunk(
                                text=prose_text,
                                document=doc,
                                chunk_index=chunk_idx,
                                strategy_name="table_aware",
                                content_type=ContentType.PROSE,
                            )
                            chunks.append(c)
                            chunk_idx += 1
                        prose_lines = []

                    in_table = True
                    table_lines.append(line)
                else:
                    if in_table and table_lines:
                        table_chunks = self._chunk_table_lines(table_lines, doc, chunk_idx)
                        chunks.extend(table_chunks)
                        chunk_idx += len(table_chunks)
                        table_lines = []
                        in_table = False

                    prose_lines.append(line)

            if table_lines:
                table_chunks = self._chunk_table_lines(table_lines, doc, chunk_idx)
                chunks.extend(table_chunks)
                chunk_idx += len(table_chunks)

            if prose_lines:
                prose_text = "\n".join(prose_lines).strip()
                if prose_text:
                    c = self._create_chunk(
                        text=prose_text,
                        document=doc,
                        chunk_index=chunk_idx,
                        strategy_name="table_aware",
                        content_type=ContentType.PROSE,
                    )
                    chunks.append(c)

        return chunks

    def _chunk_table_lines(self, lines: list[str], doc: RawDocument, start_idx: int) -> list[Chunk]:
        table_text = "\n".join(lines).strip()
        tokens = self._estimate_tokens(table_text)

        # If table fits within single chunk budget, keep atomic
        if tokens <= self.chunk_size:
            c = self._create_chunk(
                text=table_text,
                document=doc,
                chunk_index=start_idx,
                strategy_name="table_aware",
                content_type=ContentType.TABLE,
                is_atomic=True,
            )
            return [c]

        # Larger table: split rows while preserving header rows
        header_lines = lines[:2] if len(lines) >= 2 and "---" in lines[1] else lines[:1]
        data_lines = lines[len(header_lines) :]

        chunks: list[Chunk] = []
        current_rows = list(header_lines)
        curr_tokens = self._estimate_tokens("\n".join(current_rows))
        sub_idx = start_idx

        for row in data_lines:
            row_tokens = self._estimate_tokens(row)
            if curr_tokens + row_tokens > self.chunk_size and len(current_rows) > len(header_lines):
                sub_text = "\n".join(current_rows).strip()
                c = self._create_chunk(
                    text=sub_text,
                    document=doc,
                    chunk_index=sub_idx,
                    strategy_name="table_aware",
                    content_type=ContentType.TABLE,
                    is_atomic=False,
                )
                chunks.append(c)
                sub_idx += 1
                current_rows = list(header_lines)
                curr_tokens = self._estimate_tokens("\n".join(current_rows))

            current_rows.append(row)
            curr_tokens += row_tokens

        if len(current_rows) > len(header_lines):
            sub_text = "\n".join(current_rows).strip()
            c = self._create_chunk(
                text=sub_text,
                document=doc,
                chunk_index=sub_idx,
                strategy_name="table_aware",
                content_type=ContentType.TABLE,
                is_atomic=False,
            )
            chunks.append(c)

        return chunks
