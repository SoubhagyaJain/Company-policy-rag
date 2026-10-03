"""Generation options must actually reach Ollama, and token counts must come back."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import ollama
import pytest
from llama_index.llms.ollama import Ollama

from backend.rag.llm_client import (
    StreamingCompletion,
    complete_text,
    estimate_tokens,
    fit_output_budget,
)
from backend.rag.pipeline import _answer_thinking, _answer_thinking_budget


def _response(content: str = "", done: bool = True, **counts):
    return SimpleNamespace(
        message=SimpleNamespace(content=content, thinking=counts.get("thinking", "")),
        done=done,
        done_reason=counts.get("done_reason", "stop" if done else None),
        prompt_eval_count=counts.get("prompt_eval_count"),
        eval_count=counts.get("eval_count"),
        prompt_eval_duration=counts.get("prompt_eval_duration"),
        eval_duration=counts.get("eval_duration"),
    )


@pytest.fixture
def ollama_llm() -> Ollama:
    return Ollama(model="qwen2.5:7b", temperature=0.1, context_window=4096, keep_alive=-1)


def test_complete_sends_num_predict_and_temperature(monkeypatch, ollama_llm) -> None:
    calls: list[dict] = []

    def fake_chat(self, **kwargs):
        calls.append(kwargs)
        return _response("Answer. [Source 1]", prompt_eval_count=812, eval_count=41, eval_duration=2_000_000_000)

    monkeypatch.setattr(ollama.Client, "chat", fake_chat)

    text, usage = complete_text(ollama_llm, "prompt text", temperature=0.05, max_tokens=320)

    assert text == "Answer. [Source 1]"
    options = calls[0]["options"]
    assert options["num_predict"] == 320
    assert options["temperature"] == 0.05
    assert options["num_ctx"] == 4096
    assert calls[0]["model"] == "qwen2.5:7b"
    assert calls[0]["stream"] is False
    assert usage.prompt_tokens == 812
    assert usage.completion_tokens == 41
    assert usage.generation_ms == 2000.0
    assert usage.to_dict()["prompt_overflow"] is False


def test_prompt_overflow_is_flagged_from_real_counts(monkeypatch, ollama_llm) -> None:
    monkeypatch.setattr(ollama.Client, "chat", lambda self, **kw: _response("x", prompt_eval_count=4096, eval_count=1))
    _, usage = complete_text(ollama_llm, "long prompt", max_tokens=64)
    assert usage.prompt_overflow is True


def test_streaming_applies_options_and_records_usage(monkeypatch, ollama_llm) -> None:
    captured: dict = {}

    def fake_chat(self, **kwargs):
        captured.update(kwargs)
        return iter(
            [
                _response("Hel", done=False),
                _response("lo", done=False),
                _response("", done=True, prompt_eval_count=500, eval_count=2),
            ]
        )

    monkeypatch.setattr(ollama.Client, "chat", fake_chat)

    stream = StreamingCompletion(ollama_llm, "p", temperature=0.2, max_tokens=700)
    assert "".join(stream) == "Hello"
    assert captured["stream"] is True
    assert captured["options"]["num_predict"] == 700
    assert stream.usage.prompt_tokens == 500
    assert stream.usage.completion_tokens == 2


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("compact", False), ("standard", True), ("detailed", True)],
)
def test_qwen_thinking_follows_frontend_response_mode(monkeypatch, mode, expected) -> None:
    llm = Ollama(model="qwen3.5:9b", context_window=4096, thinking=False)
    calls: list[dict] = []

    def fake_chat(self, **kwargs):
        calls.append(kwargs)
        if kwargs["stream"]:
            return iter([_response("Hello", done=True)])
        return _response("Hello")

    monkeypatch.setattr(ollama.Client, "chat", fake_chat)
    think = _answer_thinking(llm.model, mode)

    assert complete_text(llm, "prompt", think=think)[0] == "Hello"
    assert "".join(StreamingCompletion(llm, "prompt", think=think)) == "Hello"
    assert [call["think"] for call in calls] == [expected, expected]


def test_other_models_keep_existing_thinking_setting() -> None:
    assert _answer_thinking("qwen2.5:7b", "detailed") is None
    assert _answer_thinking_budget("qwen3.5:9b", "compact") is None
    assert _answer_thinking_budget("qwen3.5:9b", "standard") == 128
    assert _answer_thinking_budget("qwen3.5:9b", "detailed") == 256


def test_bounded_thinking_preserves_reasoning_and_returns_an_answer(monkeypatch) -> None:
    llm = Ollama(model="qwen3.5:9b", context_window=4096)
    calls: list[dict] = []

    def fake_chat(self, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _response("", thinking="2 plus 2 is 4", done_reason="length", prompt_eval_count=20, eval_count=128)
        return _response("4.", prompt_eval_count=45, eval_count=3)

    monkeypatch.setattr(ollama.Client, "chat", fake_chat)
    answer, usage = complete_text(llm, "What is 2 + 2?", max_tokens=700, think=True, thinking_budget=128)

    assert answer == "4."
    assert [call["think"] for call in calls] == [True, False]
    assert calls[0]["options"]["num_predict"] == 128
    assert calls[1]["options"]["num_predict"] == 524
    assert calls[1]["messages"][1]["thinking"] == "2 plus 2 is 4"
    assert usage.prompt_tokens == 65
    assert usage.completion_tokens == 131
    assert usage.to_dict()["prompt_overflow"] is False


def test_bounded_thinking_streams_only_final_answer(monkeypatch) -> None:
    llm = Ollama(model="qwen3.5:9b", context_window=4096)
    calls: list[dict] = []

    def fake_chat(self, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return iter([
                _response("", done=False, thinking="private "),
                _response("", thinking="reasoning", done_reason="length", prompt_eval_count=20, eval_count=128),
            ])
        return iter([
            _response("Final", done=False),
            _response(" answer", done=False),
            _response("", done=True, prompt_eval_count=45, eval_count=2),
        ])

    monkeypatch.setattr(ollama.Client, "chat", fake_chat)
    stream = StreamingCompletion(llm, "Question", max_tokens=700, think=True, thinking_budget=128)

    assert "".join(stream) == "Final answer"
    assert [call["think"] for call in calls] == [True, False]
    assert stream.usage.completion_tokens == 130


def test_bounded_thinking_keeps_an_early_completed_answer(monkeypatch) -> None:
    llm = Ollama(model="qwen3.5:9b", context_window=4096)
    calls: list[dict] = []

    def fake_chat(self, **kwargs):
        calls.append(kwargs)
        return _response("The answer is 4.", thinking="Brief check.", eval_count=20)

    monkeypatch.setattr(ollama.Client, "chat", fake_chat)
    answer, usage = complete_text(llm, "What is 2 + 2?", max_tokens=700, think=True, thinking_budget=128)

    assert answer == "The answer is 4."
    assert len(calls) == 1
    assert usage.num_predict == 700


def test_bounded_thinking_can_cancel_during_reasoning(monkeypatch) -> None:
    llm = Ollama(model="qwen3.5:9b", context_window=4096)
    cancel = threading.Event()
    closed = threading.Event()
    calls: list[dict] = []

    def reasoning_stream():
        try:
            for _ in range(100):
                cancel.set()
                yield _response("", done=False, thinking="thought")
        finally:
            closed.set()

    def fake_chat(self, **kwargs):
        calls.append(kwargs)
        return reasoning_stream()

    monkeypatch.setattr(ollama.Client, "chat", fake_chat)
    stream = StreamingCompletion(llm, "Question", max_tokens=700, think=True, thinking_budget=128, cancel_event=cancel)

    assert list(stream) == []
    assert stream.cancelled is True
    assert closed.is_set()
    assert len(calls) == 1


def test_streaming_cancel_stops_and_closes_the_http_stream(monkeypatch, ollama_llm) -> None:
    closed = threading.Event()
    cancel = threading.Event()

    def token_stream():
        try:
            for index in range(1000):
                yield _response(f"t{index} ", done=False)
        finally:
            closed.set()

    monkeypatch.setattr(ollama.Client, "chat", lambda self, **kw: token_stream())

    stream = StreamingCompletion(ollama_llm, "p", max_tokens=100, cancel_event=cancel)
    received = []
    for delta in stream:
        received.append(delta)
        if len(received) == 3:
            cancel.set()

    assert len(received) == 3
    assert stream.cancelled is True
    assert closed.is_set()


def test_non_ollama_llm_keeps_the_complete_call_contract() -> None:
    llm = MagicMock()
    llm.complete.return_value = "  mocked answer  "

    text, usage = complete_text(llm, "prompt", temperature=0.0, max_tokens=256)

    assert text == "mocked answer"
    llm.complete.assert_called_once_with("prompt", temperature=0.0, max_new_tokens=256)
    assert usage.prompt_tokens is None
    assert usage.num_predict == 256


def test_llm_without_kwargs_support_falls_back_to_plain_complete() -> None:
    class PlainLLM:
        def complete(self, prompt):
            return f"echo:{prompt}"

    text, _ = complete_text(PlainLLM(), "hi", temperature=0.1, max_tokens=10)
    assert text == "echo:hi"


def test_output_budget_shrinks_to_fit_the_context_window(ollama_llm) -> None:
    prompt = "word " * 2400  # ~3,360 estimated tokens
    budget = fit_output_budget(ollama_llm, prompt, 1200)
    assert budget < 1200
    assert estimate_tokens(prompt) + budget <= 4096
    assert fit_output_budget(ollama_llm, "short", 700) == 700


def test_output_budget_never_drops_below_the_floor(ollama_llm) -> None:
    assert fit_output_budget(ollama_llm, "word " * 5000, 700) == 256


def test_token_estimate_is_conservative_for_code_heavy_text() -> None:
    # Calibrated: code/URL-heavy guidebook text ran 1.89 tokens per word on qwen2.5.
    code = "def get_rate(currency: str) -> float:\n    return requests.get(f'https://api.example.com/{currency}').json()['rate']\n" * 20
    words = len(code.split())
    assert estimate_tokens(code) >= int(words * 1.4)
