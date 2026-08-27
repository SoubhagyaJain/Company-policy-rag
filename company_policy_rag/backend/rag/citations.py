from __future__ import annotations

import math
import re
from typing import Any

from backend.models.rag import Citation, ScoredChunk
from backend.utils.section_tracker import is_noise_line

_SOURCE_TAG_PATTERN = re.compile(r"\[(?:Visual\s+)?Source\s+([^\]]+)\]", re.IGNORECASE)


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
    Extracts explicit [Source N] and [Visual Source N] tags from generated answer text
    and maps them to verified retrieved chunk metadata with canonical PageIdentity.
    """

    @staticmethod
    def extract_source_tags(answer_text: str) -> set[int]:
        """Parse 1-based [Source N] and [Visual Source N] tags from answer text."""
        indices: set[int] = set()
        for match in _SOURCE_TAG_PATTERN.finditer(answer_text):
            inner = match.group(1)
            for num_match in re.finditer(r"\b(\d+)\b", inner):
                indices.add(int(num_match.group(1)))
        return indices

    def _build_citation_from_chunk(
        self,
        idx: int,
        sc: ScoredChunk,
        selection_reason: str,
    ) -> Citation:
        meta = sc.chunk.metadata
        extra = meta.extra or {}
        full_text = sc.chunk.text.strip()
        snippet = full_text[:2000].strip() + ("..." if len(full_text) > 2000 else "")
        sec_title = _clean_section_title(meta.section_title)

        page_id = meta.get_page_identity()

        # Determine evidence type
        is_visual = (
            extra.get("is_visual_extraction", False)
            or "diagram" in str(meta.content_type).lower()
            or extra.get("visual_type") in ("diagram_architecture", "code_screenshot", "table_data", "figure", "image")
        )
        if is_visual:
            raw_vtype = extra.get("visual_type", "diagram_architecture").upper()
            if "CODE" in raw_vtype or "```" in sc.chunk.text or "def " in sc.chunk.text or "kickoff" in sc.chunk.text:
                evidence_type = "CODE_SCREENSHOT"
            elif "TABLE" in raw_vtype:
                evidence_type = "TABLE_DATA"
            elif "FIGURE" in raw_vtype:
                evidence_type = "FIGURE"
            else:
                evidence_type = "DIAGRAM_ARCHITECTURE"
        elif "```" in sc.chunk.text or str(meta.content_type).lower() in ("code", "contenttype.code") or "def " in sc.chunk.text or "kickoff" in sc.chunk.text:
            evidence_type = "CODE"
        elif "table" in str(meta.content_type).lower() or "|---" in sc.chunk.text:
            evidence_type = "TABLE_DATA"
        else:
            evidence_type = "TEXT"

        # Resolve visual asset id and URL
        asset_id = extra.get("asset_id")
        if not asset_id and meta.visual_asset_ids:
            asset_id = meta.visual_asset_ids[0]
        elif not asset_id and meta.image_assets:
            asset_id = meta.image_assets[0].get("asset_id")

        img_url = None
        if asset_id and meta.document_id:
            img_url = f"/api/documents/{meta.document_id}/visual-assets/{asset_id}"
        elif extra.get("image_url"):
            img_url = extra.get("image_url")
        elif meta.image_assets:
            img_url = meta.image_assets[0].get("asset_url")

        visual_status = "VISION_READY" if is_visual else ("ASSET_AVAILABLE" if (asset_id or meta.image_assets) else None)

        return Citation(
            source_index=idx,
            chunk_id=sc.chunk.id,
            document_id=meta.document_id,
            source_file=meta.source_file,
            document_name=meta.source_file,
            page_number=page_id.physical_page_number,
            internal_page_index=page_id.internal_page_index,
            display_page_number=page_id.display_page_number,
            page_label=page_id.page_label,
            section_title=sec_title,
            section_path=meta.section_path if sec_title else None,
            snippet=snippet,
            relevance_score=_compute_confidence(sc),
            selection_reason=selection_reason,
            evidence_type=evidence_type,
            visual_asset_id=asset_id,
            visual_status=visual_status,
            image_url=img_url,
            image_assets=meta.image_assets,
        )

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
                    cit = self._build_citation_from_chunk(idx, sc, selection_mode)
                    citations.append(cit)

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
                cit = self._build_citation_from_chunk(idx, sc, selection_mode)
                citations.append(cit)

        return citations
