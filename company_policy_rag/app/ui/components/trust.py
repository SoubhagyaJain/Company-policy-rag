"""Per-answer trust and latency panel."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.prompts import LOW_CONFIDENCE_MESSAGE


def citation_quality_summary(citations: list[dict[str, Any]]) -> tuple[int, int]:
    cited = sum(1 for c in citations if c.get("selection_reason") == "cited_in_answer")
    fallback = sum(
        1 for c in citations if c.get("selection_reason") == "score_threshold_fallback"
    )
    return cited, fallback


def render_trust_panel(
    *,
    timing: dict[str, Any] | None,
    citations: list[dict[str, Any]],
    answer: str,
    grounding_mode: str | None = None,
    reasoning_summary: dict[str, Any] | None = None,
    expanded: bool = False,
) -> None:
    cited, fallback = citation_quality_summary(citations)
    low_conf = LOW_CONFIDENCE_MESSAGE in answer
    degraded = reasoning_summary.get("degraded_stages") if reasoning_summary else []

    with st.expander("Trust & performance", expanded=expanded or low_conf or bool(degraded)):
        if grounding_mode == "strict":
            st.caption("🔒 **Strict Grounding**: Answers must be fully supported by verified document context.")

        if low_conf:
            st.warning("⚠ Low-confidence answer — review sources carefully.")

        if degraded:
            st.warning(f"⚠ Operating with degraded components: **{', '.join(degraded)}**. Fallback strategies were applied.")

        if timing:
            cols = st.columns(4)
            e2e = timing.get("e2e_ms", 0) or timing.get("total_latency_ms", 0)
            retrieve = timing.get("retrieve_total_ms", 0) or timing.get("stage_timings", {}).get("retrieval", 0)
            generate = timing.get("generation_ms", 0) or timing.get("stage_timings", {}).get("generation", 0)
            guard = timing.get("faithfulness_guard_ms", 0) or timing.get("stage_timings", {}).get("guard", 0)

            cols[0].metric("E2E", f"{e2e:.0f} ms")
            cols[1].metric("Retrieve", f"{retrieve:.0f} ms")
            cols[2].metric("Generate", f"{generate:.0f} ms")
            cols[3].metric("Guard", f"{guard:.0f} ms" if guard else "—")

        if reasoning_summary:
            st.divider()
            st.caption("**Reasoning Trace Summary**")
            info_cols = st.columns(3)
            info_cols[0].markdown(f"**Intent:** `{reasoning_summary.get('intent', 'factual')}`")
            info_cols[1].markdown(f"**Answer Mode:** `{reasoning_summary.get('answer_mode', 'DIRECT')}`")
            info_cols[2].markdown(f"**Evidence:** `{reasoning_summary.get('evidence_status', 'DIRECT')}`")

            flags = []
            if reasoning_summary.get("is_follow_up"):
                flags.append("✓ Follow-up resolved")
            if reasoning_summary.get("reused_previous_evidence"):
                flags.append("✓ Evidence reused")
            if reasoning_summary.get("used_visual_evidence"):
                flags.append("✓ Visual evidence inspected")
            if flags:
                st.caption(" · ".join(flags))

        if citations:
            st.caption(
                f"Sources: {cited} cited in answer"
                + (f", {fallback} score fallback" if fallback else "")
            )
            if fallback and not cited:
                st.info(
                    "Answer had no [Source N] tags — showing high-relevance fallback sources."
                )