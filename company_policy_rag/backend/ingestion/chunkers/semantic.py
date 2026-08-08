from __future__ import annotations

import re

from backend.ingestion.chunkers.base import BaseChunker
from backend.models.chunk import Chunk, ContentType
from backend.models.document import RawDocument

_SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")


class SemanticChunker(BaseChunker):
    """Semantic chunker grouping coherent sentences and paragraphs into semantic chunks."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        max_sentences_per_chunk: int = 15,
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.max_sentences_per_chunk = max_sentences_per_chunk

    def chunk(self, documents: list[RawDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for doc in documents:
            if not doc.content.strip():
                continue

            paragraphs = [p.strip() for p in doc.content.split("\n\n") if p.strip()]
            current_sentences: list[str] = []
            current_tokens = 0
            chunk_idx = 0

            for para in paragraphs:
                sentences = _SENTENCE_SPLIT_REGEX.split(para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    sent_tokens = self._estimate_tokens(sentence)

                    if current_tokens + sent_tokens > self.chunk_size and current_sentences:
                        # Flush chunk
                        chunk_text = " ".join(current_sentences)
                        c = self._create_chunk(
                            text=chunk_text,
                            document=doc,
                            chunk_index=chunk_idx,
                            strategy_name="semantic",
                            content_type=ContentType.PROSE,
                        )
                        chunks.append(c)
                        chunk_idx += 1

                        # Keep overlap sentences
                        overlap_sentences: list[str] = []
                        overlap_tokens = 0
                        for prev_s in reversed(current_sentences):
                            prev_t = self._estimate_tokens(prev_s)
                            if overlap_tokens + prev_t <= self.chunk_overlap:
                                overlap_sentences.insert(0, prev_s)
                                overlap_tokens += prev_t
                            else:
                                break

                        current_sentences = overlap_sentences
                        current_tokens = overlap_tokens

                    current_sentences.append(sentence)
                    current_tokens += sent_tokens

            if current_sentences:
                chunk_text = " ".join(current_sentences)
                c = self._create_chunk(
                    text=chunk_text,
                    document=doc,
                    chunk_index=chunk_idx,
                    strategy_name="semantic",
                    content_type=ContentType.PROSE,
                )
                chunks.append(c)

        return chunks
