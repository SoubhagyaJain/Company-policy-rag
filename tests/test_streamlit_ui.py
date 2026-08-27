"""Pure-helper tests for Streamlit UI modules (no Streamlit runtime)."""

from __future__ import annotations

import json
from pathlib import Path

from src.agent import AgentTurnResult

from app.ui.components.chat import (
    SUGGESTED_PROMPTS,
    apply_complete_assistant_turn,
    apply_queue_user_prompt,
    stream_answer_chunks,
)
from app.ui.components.health import format_eval_metrics, load_last_eval_run, probe_ollama_tags
from app.ui.components.trust import citation_quality_summary
from app.ui.session import corpus_scope_filters


def test_suggested_prompts_non_empty_and_bounded():
    assert len(SUGGESTED_PROMPTS) >= 3
    assert all(isinstance(p, str) and len(p) > 10 for p in SUGGESTED_PROMPTS)


def test_citation_quality_summary():
    citations = [
        {"selection_reason": "cited_in_answer"},
        {"selection_reason": "cited_in_answer"},
        {"selection_reason": "score_threshold_fallback"},
    ]
    cited, fallback = citation_quality_summary(citations)
    assert cited == 2
    assert fallback == 1


def test_corpus_scope_filters_all_returns_none():
    assert corpus_scope_filters("all") is None
    assert corpus_scope_filters(None) is None


def test_corpus_scope_filters_policy_returns_metadata():
    filters = corpus_scope_filters("policy")
    assert filters is not None
    assert "source_file" in filters


def test_load_last_eval_run_missing_file(tmp_path: Path):
    assert load_last_eval_run(tmp_path / "missing.json") is None


def test_load_last_eval_run_returns_last(tmp_path: Path):
    path = tmp_path / "evaluation_results.json"
    path.write_text(
        json.dumps({"runs": [{"run_id": "a"}, {"run_id": "b", "aggregate": {"faithfulness": 0.8}}]}),
        encoding="utf-8",
    )
    last = load_last_eval_run(path)
    assert last is not None
    assert last["run_id"] == "b"


def test_format_eval_metrics():
    metrics = format_eval_metrics(
        {"aggregate": {"faithfulness": 0.807, "answer_relevancy": 0.766, "hit_rate": 0.886}}
    )
    assert metrics["Faithfulness"] == "0.807"
    assert metrics["Answer relevancy"] == "0.766"
    assert metrics["Hit rate"] == "0.886"


def test_stream_answer_chunks_reconstructs_text():
    text = "The dress code requires business casual attire."
    chunks = list(stream_answer_chunks(text, words_per_chunk=2))
    assert "".join(chunks) == text


def test_queue_user_prompt_sets_pending():
    session: dict = {"messages": []}
    apply_queue_user_prompt(session, "Hello")
    assert session["messages"][-1]["content"] == "Hello"
    assert session["pending_user_prompt"] == "Hello"


def test_complete_assistant_turn_persists_thinking_events_and_summary():
    session: dict = {
        "messages": [{"role": "user", "content": "Tell me about sick days"}],
        "pending_user_prompt": "Tell me about sick days",
    }
    mock_events = [
        {"id": "1", "stage": "retrieval", "status": "completed", "title": "Retrieval", "summary": "Found 4 passages", "duration_ms": 25.0},
        {"id": "2", "stage": "evidence_verification", "status": "completed", "title": "Verification", "summary": "Verified 4/4", "duration_ms": 12.0},
    ]
    mock_summary = {
        "intent": "factual",
        "answer_mode": "DIRECT",
        "is_follow_up": False,
        "evidence_status": "DIRECT",
        "degraded_stages": [],
        "total_duration_ms": 37.0,
    }
    turn = AgentTurnResult(
        answer="You get 10 sick days per year.",
        citations=[{"source_file": "handbook.pdf", "page_number": 12, "evidence_type": "TEXT"}],
        timing={"e2e_ms": 250.0},
        thinking_events=mock_events,
        reasoning_summary=mock_summary,
    )
    apply_complete_assistant_turn(session, turn, user_prompt="Tell me about sick days")
    assert len(session["messages"]) == 2
    assert session["pending_user_prompt"] is None
    assistant_msg = session["messages"][1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["thinking_events"] == mock_events
    assert assistant_msg["reasoning_summary"] == mock_summary
    assert assistant_msg["citations"][0]["evidence_type"] == "TEXT"


def test_probe_ollama_tags_invalid_host():
    ok, models, err = probe_ollama_tags("http://127.0.0.1:1", timeout=0.5)
    assert not ok
    assert models == []
    assert err