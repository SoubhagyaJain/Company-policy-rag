from __future__ import annotations

import math
import re

from backend.models.rag import Citation, ScoredChunk
from backend.utils.section_tracker import is_noise_line

_SOURCE_TAG_PATTERN = re.compile(r"\[Source\s+([^\]]+)\]", re.IGNORECASE)


def _compute_confidence(sc: ScoredChunk) -> float:
    """Normalize rerank logit score or candidate score into [0.05, 0.99] confidence range."""
    raw = sc.rerank_score if sc.rerank_score is not None else sc.score
    if raw is None:
        return 0.75
    try:
        raw_val = float(raw)
        # If score is already a bounded probability [0.0, 1.0]
        if 0.0 <= raw_val <= 1.0 and sc.rerank_score is None:
            return max(0.50, min(0.99, raw_val))
        # CrossEncoder raw logits (-10 to +10) -> Sigmoid
        if raw_val > 15.0:
            return 0.99
        if raw_val < -15.0:
            return 0.15
        prob = 1.0 / (1.0 + math.exp(-raw_val))
        return max(0.20, min(0.99, round(prob, 4)))
    except Exception:
        return 0.80


def _clean_section_title(title: str | None) -> str | None:
    if not title:
        return None
    cleaned = title.strip()
    if is_noise_line(cleaned):
        return None
    return cleaned


class CitationEngine:
    """
    Extracts explicit [Source N] tags from generated answer text and maps them to verified
    retrieved chunk metadata. Provides fallback to relative score thresholding if answer
    lacks source tags.
    """

    @staticmethod
    def extract_source_tags(answer_text: str) -> set[int]:
        """Parse 1-based [Source N] tags from answer text."""
        indices: set[int] = set()
        for match in _SOURCE_TAG_PATTERN.finditer(answer_text):
            inner = match.group(1)
            for num_match in re.finditer(r"\b(\d+)\b", inner):
                indices.add(int(num_match.group(1)))
        return indices

    def select_citations(
        self,
        answer_text: str,
        generation_chunks: list[ScoredChunk],
        user_query: str | None = None,
    ) -> list[Citation]:
        """Map answer text [Source N] tags or relevance scores to Citation models."""
        if not generation_chunks:
            return []

        cited_indices = self.extract_source_tags(answer_text)
        citations: list[Citation] = []
        selection_mode = "cited_in_answer"

        if cited_indices:
            for idx in sorted(cited_indices):
                if 1 <= idx <= len(generation_chunks):
                    sc = generation_chunks[idx - 1]
                    meta = sc.chunk.metadata
                    # Provide verbatim chunk text up to 2000 characters
                    full_text = sc.chunk.text.strip()
                    snippet = full_text[:2000].strip() + ("..." if len(full_text) > 2000 else "")
                    sec_title = _clean_section_title(meta.section_title)

                    img_url = meta.extra.get("image_url") or (meta.image_assets[0].get("asset_url") if meta.image_assets else None)
                    citations.append(
                        Citation(
                            source_index=idx,
                            chunk_id=sc.chunk.id,
                            document_id=meta.document_id,
                            source_file=meta.source_file,
                            page_number=meta.page_number,
                            internal_page_index=meta.internal_page_index,
                            page_label=meta.page_label,
                            section_title=sec_title,
                            section_path=meta.section_path if sec_title else None,
                            snippet=snippet,
                            relevance_score=_compute_confidence(sc),
                            selection_reason=selection_mode,
                            image_url=img_url,
                            image_assets=meta.image_assets,
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
                full_text = sc.chunk.text.strip()
                snippet = full_text[:2000].strip() + ("..." if len(full_text) > 2000 else "")
                sec_title = _clean_section_title(meta.section_title)
                img_url = meta.extra.get("image_url") or (meta.image_assets[0].get("asset_url") if meta.image_assets else None)

                citations.append(
                    Citation(
                        source_index=idx,
                        chunk_id=sc.chunk.id,
                        document_id=meta.document_id,
                        source_file=meta.source_file,
                        page_number=meta.page_number,
                        internal_page_index=meta.internal_page_index,
                        page_label=meta.page_label,
                        section_title=sec_title,
                        section_path=meta.section_path if sec_title else None,
                        snippet=snippet,
                        relevance_score=_compute_confidence(sc),
                        selection_reason=selection_mode,
                        image_url=img_url,
                        image_assets=meta.image_assets,
                    )
                )

        return citations
