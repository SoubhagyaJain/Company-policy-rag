"""Citation cards and grounding badge."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from src.config import settings
from src.prompts import resolve_grounding_mode
from src.utils import (
    format_citation_excerpt,
    format_citation_primary,
    prepare_citations_for_display,
    shorten_source_filename,
)


def render_grounding_badge() -> None:
    mode = resolve_grounding_mode()
    css_class = "status-pill strict" if mode == "strict" else "status-pill"
    label = "Strict grounding" if mode == "strict" else "Balanced grounding"
    st.markdown(f'<span class="{css_class}">{label}</span>', unsafe_allow_html=True)


def render_sources_compact(citations: list[dict[str, Any]]) -> None:
    """Single collapsed expander for all sources (ChatGPT-style)."""
    prepared = prepare_citations_for_display(citations)
    if not prepared:
        return

    noun = "source" if len(prepared) == 1 else "sources"
    with st.expander(f"📚 {len(prepared)} {noun}", expanded=False):
        for i, citation in enumerate(prepared, 1):
            label = format_citation_primary(citation)
            ev_type = citation.get("evidence_type") or ("DIAGRAM" if citation.get("image_url") or citation.get("page_images") else "TEXT")
            type_badge = f"`[{ev_type}]` " if ev_type else ""
            if settings.citation_show_relevance_score and citation.get("score") is not None:
                label = f"{label} · {citation['score']:.2f}"
            st.markdown(f"**{i}.** {type_badge}{label}")
            source_file = citation.get("source_file", "unknown")
            st.caption(f"`{shorten_source_filename(source_file)}`")
            if settings.citation_show_excerpts:
                st.caption(format_citation_excerpt(citation, max_len=280))


def render_sources(citations: list[dict[str, Any]]) -> None:
    prepared = prepare_citations_for_display(citations)
    if not prepared:
        return

    noun = "source" if len(prepared) == 1 else "sources"
    st.markdown(
        f'<p class="source-header">📚 {len(prepared)} {noun} — expand to verify against the document</p>',
        unsafe_allow_html=True,
    )

    for i, citation in enumerate(prepared, 1):
        label = format_citation_primary(citation)
        if settings.citation_show_relevance_score and citation.get("score") is not None:
            label = f"{label} · score {citation['score']:.2f}"

        with st.expander(f"{i}. {label}", expanded=False):
            reason = citation.get("selection_reason")
            if reason == "cited_in_answer":
                st.caption("✓ Cited in answer")
            elif reason == "score_threshold_fallback":
                st.caption("⚠ High relevance fallback (no [Source N] tag in answer)")

            source_file = citation.get("source_file", "unknown")
            st.markdown(f"**Document:** `{shorten_source_filename(source_file)}`")

            section_path = citation.get("section_path")
            if section_path:
                st.markdown(f"**Section:** {section_path}")
            elif citation.get("section_title"):
                st.markdown(f"**Section:** {citation.get('section_title')}")

            page = citation.get("page_number")
            if page is not None:
                st.markdown(f"**Page:** {page}")

            if settings.citation_show_excerpts:
                st.markdown("**Excerpt**")
                st.caption(format_citation_excerpt(citation, max_len=500))

            page_images = citation.get("page_images") or []
            if page_images:
                st.markdown("**Page visuals**")
                for img_path in page_images:
                    path = Path(img_path)
                    if path.is_file():
                        st.image(str(path), caption=path.name, use_container_width=True)