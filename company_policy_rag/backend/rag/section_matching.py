"""Preserve explicitly named sections when semantic ranking favors generic prose."""
from __future__ import annotations

import re
import unicodedata

from backend.models.rag import ScoredChunk
from backend.utils.section_tracker import is_noise_line, parse_section_heading


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def named_section_matches(query: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    """Find multiword section names explicitly mentioned in the question.

    Inspect source headings as well as metadata so previously indexed files with
    stale section metadata still work. Contents-page entries are not evidence.
    """
    normalized_query = f" {_normalize(query)} "
    generic = {"tech stack", "technology stack", "expected output", "setup llm", "system overview"}
    matches = []
    for scored in chunks:
        chunk = scored.chunk
        if (chunk.metadata.extra or {}).get("visual_status") == "ASSET_AVAILABLE":
            continue
        lines = chunk.text.splitlines()
        if any(re.search(r"\.{3,}\s*\d+\s*$", line) for line in lines):
            continue
        metadata_title = chunk.metadata.section_title or ""
        titles = [metadata_title]
        source_titles: list[str] = []
        for line in lines:
            heading = parse_section_heading(line)
            if heading:
                titles.append(heading.section_title)
                source_titles.append(heading.section_title)
        names = [_normalize(title) for title in titles if not is_noise_line(title)]
        matching_names = [
            name for name in names
            if 2 <= len(name.split()) <= 12 and name not in generic
            and f" {name} " in normalized_query
        ]
        longest = max((len(name.split()) for name in matching_names), default=0)
        if longest:
            source_heading_match = any(
                _normalize(title) in matching_names for title in source_titles
            )
            content_words = len(_normalize(chunk.text).split())
            relevance = scored.rerank_score if scored.rerank_score is not None else scored.score
            matches.append((longest, source_heading_match, content_words, relevance or 0.0, scored))
    matches.sort(key=lambda item: item[:4], reverse=True)
    return [item[-1] for item in matches]


def prioritize_named_sections(query: str, ranked: list[ScoredChunk], pool: list[ScoredChunk]) -> list[ScoredChunk]:
    ranked_by_id = {sc.chunk.id: sc for sc in ranked}
    preferred = [ranked_by_id.get(sc.chunk.id, sc) for sc in named_section_matches(query, pool)]
    preferred_ids = {sc.chunk.id for sc in preferred}
    return preferred + [sc for sc in ranked if sc.chunk.id not in preferred_ids]
