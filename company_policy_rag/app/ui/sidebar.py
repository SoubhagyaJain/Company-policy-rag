"""Shared sidebar controls and status."""

from __future__ import annotations

import streamlit as st

from src.config import settings
from src.memory import memory_stats
from src.prompts import resolve_grounding_mode

from app.ui.session import apply_grounding_mode, ensure_session_state, index_health


def render_sidebar_controls() -> None:
    ensure_session_state()
    with st.sidebar:
        st.title("Settings")
        st.caption("Company policy & legal document assistant")

        grounding_choice = st.radio(
            "Grounding mode",
            options=["balanced", "strict"],
            format_func=lambda x: (
                "Balanced (helpful synthesis)"
                if x == "balanced"
                else "Strict (max faithfulness)"
            ),
            index=0 if resolve_grounding_mode() == "balanced" else 1,
            help="Balanced answers from related excerpts; Strict abstains unless fully supported.",
        )
        apply_grounding_mode(grounding_choice)

        # Thinking Detail Level (Milestone 4 requirement)
        st.divider()
        st.subheader("Thinking Experience")
        thinking_levels = ["off", "compact", "standard", "detailed"]
        thinking_labels = {
            "off": "Off (No reasoning UI)",
            "compact": "Compact (Core milestones)",
            "standard": "Standard (All stages)",
            "detailed": "Detailed (Metrics & durations)",
        }
        current_thk = st.session_state.get("thinking_detail_level", "standard")
        thk_choice = st.radio(
            "Reasoning detail",
            options=thinking_levels,
            format_func=lambda x: thinking_labels[x],
            index=thinking_levels.index(current_thk) if current_thk in thinking_levels else 2,
            help="Control the granularity of safe thinking milestones and duration displays.",
        )
        if thk_choice != st.session_state.get("thinking_detail_level"):
            st.session_state.thinking_detail_level = thk_choice

        # LLM Temperature
        st.divider()
        st.subheader("Model Parameters")
        settings.llm_temperature = st.slider(
            "LLM temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(settings.llm_temperature),
            step=0.05,
        )

        # Citations
        st.divider()
        st.subheader("Citations")
        settings.show_citations = st.toggle(
            "Show sources",
            value=settings.show_citations,
        )
        settings.citation_show_excerpts = st.toggle(
            "Show excerpts",
            value=settings.citation_show_excerpts,
        )
        settings.citation_show_relevance_score = st.toggle(
            "Show relevance scores",
            value=settings.citation_show_relevance_score,
        )

        st.divider()
        st.subheader("Search scope")
        scope_options = ["all", "policy", "guidebook"]
        scope_labels = {
            "all": "All corpora",
            "policy": "Handbook only",
            "guidebook": "Guidebook only",
        }
        current = st.session_state.get("corpus_scope", "all")
        scope_choice = st.radio(
            "Corpus filter",
            options=scope_options,
            format_func=lambda x: scope_labels[x],
            index=scope_options.index(current) if current in scope_options else 0,
        )
        if scope_choice != st.session_state.corpus_scope:
            st.session_state.corpus_scope = scope_choice
            st.session_state.initialized = False

        st.divider()
        st.subheader("Chat mode")
        mode_options = ["direct", "agent"]
        mode_labels = {
            "direct": "Direct (faster)",
            "agent": "Agent (follow-ups)",
        }
        current_mode = st.session_state.get("chat_mode", "direct")
        mode_choice = st.radio(
            "Response mode",
            options=mode_options,
            format_func=lambda x: mode_labels[x],
            index=mode_options.index(current_mode) if current_mode in mode_options else 0,
            help="Direct skips ReAct routing for faster single-shot Q&A. "
            "Agent uses memory for multi-turn follow-ups.",
        )
        if mode_choice != st.session_state.chat_mode:
            st.session_state.chat_mode = mode_choice

        st.divider()
        st.subheader("Session")
        if st.button("Clear chat history", use_container_width=True):
            from app.ui.session import clear_chat_session
            clear_chat_session()
            st.rerun()

        st.page_link("pages/2_Documents.py", label="Manage documents →", icon="📄")
        st.page_link("pages/3_System_Health.py", label="System health →", icon="🔧")


def render_sidebar_status() -> None:
    with st.sidebar:
        st.divider()
        st.subheader("System status")
        health = index_health()
        if health.get("ready"):
            st.metric("Indexed chunks", health.get("count", 0))
            if health.get("last_updated"):
                st.caption(f"Index updated: {health['last_updated']}")
            if st.session_state.get("initialized"):
                st.caption(f"LLM: `{settings.llm_model}`")
                if settings.enable_reranker:
                    st.caption(
                        f"Retrieve: k={settings.retrieval_candidate_k} → "
                        f"top {settings.reranker_top_n}"
                    )
                mem = memory_stats(st.session_state.get("memory"))
                if mem.get("enabled"):
                    st.caption(f"Memory: {mem.get('turns', 0)} turns")
                scope = st.session_state.get("corpus_scope", "all")
                if scope != "all":
                    st.caption(f"Scope: **{scope}**")
            else:
                st.caption("Agent loading…")
        else:
            st.warning("No index — open Documents to upload or index.")