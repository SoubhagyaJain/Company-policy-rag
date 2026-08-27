"""Chat welcome, history, and ChatGPT-style turn handling."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import streamlit as st

from src.agent import AgentTurnResult
from src.config import settings
from src.prompts import LOW_CONFIDENCE_MESSAGE, resolve_grounding_mode

from app.ui.components.citations import render_sources_compact
from app.ui.components.trust import render_trust_panel
from app.ui.session import index_health, run_agent_turn, run_direct_turn

SUGGESTED_PROMPTS = [
    "How many sick days do I get?",
    "What is the dress code policy?",
    "What are the six building blocks of AI agents?",
    "What is the vacation benefits policy?",
]

TYPING_INDICATOR_HTML = (
    '<div class="typing-indicator" aria-label="Generating answer">'
    "<span></span><span></span><span></span></div>"
)


def stream_answer_chunks(text: str, *, words_per_chunk: int = 3) -> Iterator[str]:
    """Yield answer text in small chunks for st.write_stream."""
    words = text.split()
    if not words:
        if text:
            yield text
        return
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i : i + words_per_chunk])
        if i + words_per_chunk < len(words):
            chunk += " "
        yield chunk


def apply_queue_user_prompt(session: dict[str, Any], prompt: str) -> None:
    """Append user message and defer generation (testable without Streamlit)."""
    session.setdefault("messages", []).append({"role": "user", "content": prompt})
    session["pending_user_prompt"] = prompt


def apply_complete_assistant_turn(
    session: dict[str, Any],
    turn: AgentTurnResult,
    *,
    user_prompt: str,
) -> None:
    """Persist assistant message and clear pending state."""
    session.setdefault("messages", []).append(
        {
            "role": "assistant",
            "content": turn.answer,
            "citations": turn.citations,
            "thinking_events": turn.thinking_events,
            "reasoning_summary": turn.reasoning_summary,
            "timing": turn.timing,
            "grounding_mode": turn.grounding_mode,
            "low_confidence": turn.low_confidence,
            "user_prompt": user_prompt,
        }
    )
    session["pending_user_prompt"] = None


def queue_user_prompt(prompt: str) -> None:
    apply_queue_user_prompt(st.session_state, prompt)


def complete_assistant_turn(turn: AgentTurnResult, *, user_prompt: str) -> None:
    apply_complete_assistant_turn(st.session_state, turn, user_prompt=user_prompt)


def render_thinking_history(
    thinking_events: list[dict[str, Any]],
    reasoning_summary: dict[str, Any] | None = None,
    detail_level: str = "standard",
) -> None:
    """Render collapsible reasoning expander for past turns in chat history."""
    if detail_level == "off" or not thinking_events:
        return

    total_ms = 0.0
    if reasoning_summary and isinstance(reasoning_summary, dict):
        total_ms = float(reasoning_summary.get("total_duration_ms") or 0.0)
    if total_ms <= 0.0:
        total_ms = sum(float(e.get("duration_ms") or 0.0) for e in thinking_events)

    duration_str = f" for {total_ms / 1000:.1f}s" if total_ms > 0 else ""
    header = f"💭 Thought{duration_str}"

    with st.expander(header, expanded=False):
        for ev in thinking_events:
            stage = ev.get("stage", "")
            status = ev.get("status", "completed")
            title = ev.get("title", stage.replace("_", " ").title())
            summary = ev.get("summary", "")
            dur = float(ev.get("duration_ms") or 0.0)
            details = ev.get("details") or {}

            dur_str = f" `{dur:.0f}ms`" if dur > 0 and detail_level == "detailed" else ""

            if status == "warning" or stage == "degraded":
                st.markdown(f"**⚠ {title}**{dur_str}")
                if summary:
                    st.caption(f"_{summary}_")
            elif status == "skipped":
                st.markdown(
                    f"<span style='color: var(--text-muted); font-size: 0.85rem;'>⏭ {title}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"**✓ {title}**{dur_str}")
                if summary:
                    st.caption(summary)
                if detail_level == "detailed" and details:
                    safe_meta = []
                    if "candidate_count" in details:
                        safe_meta.append(f"candidates: {details['candidate_count']}")
                    if "source_count" in details:
                        safe_meta.append(f"sources: {details['source_count']}")
                    if "active_topic" in details and details["active_topic"]:
                        safe_meta.append(f"topic: {details['active_topic']}")
                    if "evidence_status" in details:
                        safe_meta.append(f"evidence: {details['evidence_status']}")
                    if safe_meta:
                        st.caption(" · ".join(safe_meta))


def render_welcome() -> None:
    if st.session_state.messages:
        return

    health = index_health()
    st.markdown(
        """
        <div class="welcome-hero">
          <h3>Ask about handbook policies or guidebook content</h3>
          <p>Answers are grounded in indexed PDFs. Sources appear below each reply.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    cols[0].metric("Indexed chunks", health.get("count", 0))
    cols[1].metric("Grounding", resolve_grounding_mode().title())
    cols[2].metric("Citations", "On" if settings.show_citations else "Off")


def render_suggested_prompts() -> None:
    if st.session_state.messages:
        return
    st.caption("Try a suggested question:")
    cols = st.columns(2)
    for i, prompt in enumerate(SUGGESTED_PROMPTS):
        with cols[i % 2]:
            if st.button(prompt, key=f"suggest_{i}", use_container_width=True):
                queue_user_prompt(prompt)
                st.rerun()


def _render_assistant_extras(msg: dict[str, Any]) -> None:
    detail_level = st.session_state.get("thinking_detail_level", "standard")
    thinking_events = msg.get("thinking_events") or []
    reasoning_summary = msg.get("reasoning_summary")
    citations = msg.get("citations") or []

    if thinking_events and detail_level != "off":
        render_thinking_history(thinking_events, reasoning_summary, detail_level=detail_level)

    if settings.show_citations and citations:
        render_sources_compact(citations)

    render_trust_panel(
        timing=msg.get("timing"),
        citations=citations,
        answer=msg.get("content", ""),
        grounding_mode=msg.get("grounding_mode"),
        reasoning_summary=reasoning_summary,
        expanded=bool(msg.get("low_confidence")),
    )


def render_chat_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] != "assistant":
                continue
            _render_assistant_extras(msg)


def _run_turn(prompt: str, agent, memory) -> AgentTurnResult:
    chat_mode = st.session_state.get("chat_mode", "direct")
    if chat_mode == "agent":
        return run_agent_turn(agent, prompt, memory)
    return run_direct_turn(prompt)


def process_pending_turn(prompt: str, agent, memory) -> None:
    """Generate assistant reply for the queued user prompt with live st.status milestone tracking."""
    detail_level = st.session_state.get("thinking_detail_level", "standard")
    show_thinking = (detail_level != "off")

    with st.chat_message("assistant"):
        if show_thinking:
            status_box = st.status("Reasoning over document corpus...", expanded=True)
            status_box.write("🔍 Resolving conversation context & follow-up intent...")
        else:
            status_box = None

        try:
            turn = _run_turn(prompt, agent, memory)
            if status_box:
                if turn.thinking_events:
                    for ev in turn.thinking_events:
                        stg = ev.get("stage", "")
                        stt = ev.get("status", "completed")
                        ttl = ev.get("title", stg.replace("_", " ").title())
                        smm = ev.get("summary", "")
                        if stt == "warning" or stg == "degraded":
                            status_box.markdown(f"**⚠ {ttl}** — {smm}")
                        elif stt == "skipped":
                            status_box.markdown(
                                f"<span style='color: var(--text-muted);'>⏭ {ttl}</span>",
                                unsafe_allow_html=True,
                            )
                        else:
                            status_box.markdown(f"**✓ {ttl}**" + (f": {smm}" if smm else ""))
                else:
                    status_box.write(f"✓ Retrieved verified sources ({len(turn.citations)} citations)")
                    status_box.write(f"✓ Mode: {turn.grounding_mode.title()} grounding")

                total_ms = 0.0
                if turn.reasoning_summary and isinstance(turn.reasoning_summary, dict):
                    total_ms = float(turn.reasoning_summary.get("total_duration_ms") or 0.0)
                if total_ms <= 0.0 and turn.timing:
                    total_ms = float(turn.timing.get("e2e_ms") or turn.timing.get("total_latency_ms") or 0.0)
                dur_s = total_ms / 1000.0 if total_ms > 0 else 0.5
                status_box.update(
                    label=f"Thought for {dur_s:.1f}s",
                    state="complete",
                    expanded=False,
                )
        except Exception as exc:
            turn = AgentTurnResult(
                answer=f"Sorry, something went wrong: {exc}",
                citations=[],
                timing=None,
                grounding_mode=resolve_grounding_mode(),
                low_confidence=False,
            )
            if status_box:
                status_box.update(label="Thinking error", state="error", expanded=False)

        st.write_stream(stream_answer_chunks(turn.answer))

        if turn.low_confidence or LOW_CONFIDENCE_MESSAGE in turn.answer:
            st.caption("⚠ Review cited sources — answer could not be fully verified.")
        if settings.show_citations and turn.citations:
            render_sources_compact(turn.citations)
        render_trust_panel(
            timing=turn.timing,
            citations=turn.citations,
            answer=turn.answer,
            grounding_mode=turn.grounding_mode,
            reasoning_summary=turn.reasoning_summary,
            expanded=turn.low_confidence,
        )

    complete_assistant_turn(turn, user_prompt=prompt)
    st.rerun()


def render_chat_interface(agent, memory) -> None:
    """Main chat loop: history, pending generation, and input."""
    st.markdown('<div class="chat-thread">', unsafe_allow_html=True)

    in_conversation = bool(st.session_state.messages)
    if not in_conversation:
        render_welcome()
        render_suggested_prompts()

    render_chat_history()

    pending = st.session_state.get("pending_user_prompt")
    if pending:
        process_pending_turn(pending, agent, memory)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if prompt := st.chat_input("Ask a policy question…"):
        queue_user_prompt(prompt)
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)