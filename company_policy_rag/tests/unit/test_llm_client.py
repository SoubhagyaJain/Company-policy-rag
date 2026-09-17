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


def _response(content: str = "", done: bool = True, **counts):
    return SimpleNamespace(
        message=SimpleNamespace(content=content),
        done=done,
        done_reason="stop" if done else None,
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
