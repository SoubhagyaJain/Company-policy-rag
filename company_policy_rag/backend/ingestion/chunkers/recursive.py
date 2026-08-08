from __future__ import annotations

from backend.ingestion.chunkers.base import BaseChunker
from backend.models.chunk import Chunk, ContentType
from backend.models.document import RawDocument


class RecursiveChunker(BaseChunker):
    """Recursive text chunker using hierarchical separators."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, documents: list[RawDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for doc in documents:
            if not doc.content.strip():
                continue

            text_splits = self._split_text(doc.content, self.separators)
            for idx, text in enumerate(text_splits):
                if not text.strip():
                    continue
                c = self._create_chunk(
                    text=text,
                    document=doc,
                    chunk_index=idx,
                    strategy_name="recursive",
                    content_type=ContentType.PROSE,
                )
                chunks.append(c)

        return chunks

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        final_chunks: list[str] = []
        target_len = self.chunk_size * 4  # Convert token size target to char count approx

        separator = separators[-1]
        new_separators: list[str] = []
        for i, s in enumerate(separators):
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                new_separators = separators[i + 1 :]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: list[str] = []
        for s in splits:
            if len(s) < target_len:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator, target_len)
                    final_chunks.extend(merged)
                    good_splits = []
                if new_separators:
                    sub_chunks = self._split_text(s, new_separators)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(s)

        if good_splits:
            merged = self._merge_splits(good_splits, separator, target_len)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: list[str], separator: str, target_len: int) -> list[str]:
        docs: list[str] = []
        current_doc: list[str] = []
        total = 0
        overlap_char_count = self.chunk_overlap * 4

        for d in splits:
            len_d = len(d)
            if total + len_d + (len(separator) if current_doc else 0) > target_len:
                if current_doc:
                    doc_str = separator.join(current_doc)
                    if doc_str.strip():
                        docs.append(doc_str)
                    # Keep overlap
                    while current_doc and total > overlap_char_count:
                        removed = current_doc.pop(0)
                        total -= len(removed) + len(separator)
            current_doc.append(d)
            total += len_d + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            doc_str = separator.join(current_doc)
            if doc_str.strip():
                docs.append(doc_str)

        return docs
