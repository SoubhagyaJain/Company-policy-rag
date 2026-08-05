from __future__ import annotations

import re
from typing import List, Optional, Set
from backend.models.rag import Citation, ScoredChunk
from backend.utils.logging import logger

_SOURCE_TAG_PATTERN = re.compile(r"\[Source\s+([^\]]+)\]", re.IGNORECASE)


class CitationEngine:
    """
    Extracts explicit [Source N] tags from generated answer text and maps them to verified
    retrieved chunk metadata. Provides fallback to relative score thresholding if answer
    lacks source tags.
    """

    @staticmethod
    def extract_source_tags(answer_text: str) -> Set[int]:
        """Parse 1-based [Source N] tags from answer text."""
        indices: Set[int] = set()
        for match in _SOURCE_TAG_PATTERN.finditer(answer_text):
            inner = match.group(1)
            for num_match in re.finditer(r"\b(\d+)\b", inner):
                indices.add(int(num_match.group(1)))
        return indices

    def select_citations(
        self,
        answer_text: str,
        generation_chunks: List[ScoredChunk],
        user_query: Optional[str] = None,
    ) -> List[Citation]:
        """Map answer text [Source N] tags or relevance scores to Citation models."""
        if not generation_chunks:
            return []

        cited_indices = self.extract_source_tags(answer_text)
        citations: List[Citation] = []
        selection_mode = "cited_in_answer"

        if cited_indices:
            for idx in sorted(cited_indices):
                if 1 <= idx <= len(generation_chunks):
                    sc = generation_chunks[idx - 1]
                    meta = sc.chunk.metadata
                    snippet = sc.chunk.text[:250].strip() + ("..." if len(sc.chunk.text) > 250 else "")
                    citations.append(
                        Citation(
                            source_index=idx,
                            chunk_id=sc.chunk.id,
                            document_id=meta.document_id,
                            source_file=meta.source_file,
                            page_number=meta.page_number,
                            section_title=meta.section_title,
                            section_path=meta.section_path,
                            snippet=snippet,
                            relevance_score=sc.rerank_score if sc.rerank_score is not None else sc.score,
                            selection_reason=selection_mode,
                        )
                    )

        if not citations:
            selection_mode = "score_threshold_fallback"
            top_score = max((c.rerank_score if c.rerank_score is not None else c.score) for c in generation_chunks)
            threshold = top_score * 0.45 if top_score > 0 else 0.0
            filtered = [
                c for c in generation_chunks
                if (c.rerank_score if c.rerank_score is not None else c.score) >= threshold
            ]
            if not filtered:
                filtered = sorted(generation_chunks, key=lambda c: c.score, reverse=True)[:1]

            for sc in filtered[:3]:
                idx = sc.rank if sc.rank is not None else 1
                meta = sc.chunk.metadata
                snippet = sc.chunk.text[:250].strip() + ("..." if len(sc.chunk.text) > 250 else "")
                citations.append(
                    Citation(
                        source_index=idx,
                        chunk_id=sc.chunk.id,
                        document_id=meta.document_id,
                        source_file=meta.source_file,
                        page_number=meta.page_number,
                        section_title=meta.section_title,
                        section_path=meta.section_path,
                        snippet=snippet,
                        relevance_score=sc.rerank_score if sc.rerank_score is not None else sc.score,
                        selection_reason=selection_mode,
                    )
                )

        return citations
