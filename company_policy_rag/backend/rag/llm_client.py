"""Per-request generation options and real token accounting for the local LLM.

``llama_index.llms.ollama.Ollama.complete`` / ``stream_complete`` accept
``**kwargs`` but send ``options=self._model_kwargs`` (constructor temperature
and ``num_ctx`` only), so a per-call ``temperature`` or ``max_new_tokens`` was
silently dropped and ``num_predict`` was never set. For Ollama-backed LLMs these
helpers call the ollama client directly with the options merged in and return
Ollama's own token counts. Any other LLM object (tests, other providers) keeps
the previous ``complete(prompt, temperature=..., max_new_tokens=...)`` call.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

# Calibrated on qwen2.5:7b over the eval corpora: prose ran 1.36 tokens/word
# (4.7 chars/token) and code-heavy guidebook text 1.89 tokens/word
# (5.7 chars/token incl. URLs). The max of both views over-estimates slightly,
# which is the safe direction for a context-window guard.
_TOKENS_PER_WORD = 1.4
_CHARS_PER_TOKEN = 4.5


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for budget checks (not billing)."""
    if not text:
        return 0
    return int(max(len(text.split()) * _TOKENS_PER_WORD, len(text) / _CHARS_PER_TOKEN)) + 1


@dataclass
class LLMUsage:
    """Token counts reported by Ollama for one generation call."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    num_ctx: int | None = None
    num_predict: int | None = None
    estimated_prompt_tokens: int | None = None
    prompt_ms: float | None = None
    generation_ms: float | None = None
    done_reason: str | None = None

    @property
    def prompt_overflow(self) -> bool:
        """True when the prompt alone reached the context window (Ollama truncates it)."""
        if self.num_ctx is None:
            return False
        prompt = self.prompt_tokens if self.prompt_tokens is not None else self.estimated_prompt_tokens
        return prompt is not None and prompt >= self.num_ctx

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["prompt_overflow"] = self.prompt_overflow
        return data


def _is_ollama_llm(llm: Any) -> bool:
    try:
        from llama_index.llms.ollama import Ollama
    except Exception:  # pragma: no cover - optional dependency
        return False
    target = getattr(llm, "_target_llm", llm)  # pipeline._LLMProxy wraps a shared client
    # Relies on llama-index-llms-ollama internals (client, _model_kwargs); if a
    # future release renames them, fall back to the generic complete() path.
    return (
        isinstance(target, Ollama)
        and hasattr(type(target), "client")
        and hasattr(type(target), "_model_kwargs")
    )


def context_window(llm: Any) -> int | None:
    value = getattr(getattr(llm, "_target_llm", llm), "context_window", None)
    return int(value) if isinstance(value, int) and value > 0 else None


def _options(llm: Any, temperature: float | None, max_tokens: int | None) -> dict[str, Any]:
    target = getattr(llm, "_target_llm", llm)
    options = dict(getattr(target, "_model_kwargs", {}) or {})
    if temperature is not None:
        options["temperature"] = float(temperature)
    if max_tokens is not None:
        options["num_predict"] = int(max_tokens)
    return options


def _usage_from(response: Any, options: dict[str, Any], estimated: int) -> LLMUsage:
    def _ms(value: Any) -> float | None:
        return round(value / 1_000_000, 2) if isinstance(value, (int, float)) else None

    return LLMUsage(
        prompt_tokens=getattr(response, "prompt_eval_count", None),
        completion_tokens=getattr(response, "eval_count", None),
        num_ctx=options.get("num_ctx"),
        num_predict=options.get("num_predict"),
        estimated_prompt_tokens=estimated,
        prompt_ms=_ms(getattr(response, "prompt_eval_duration", None)),
        generation_ms=_ms(getattr(response, "eval_duration", None)),
        done_reason=getattr(response, "done_reason", None),
    )


def _model_name(llm: Any) -> str:
    return str(getattr(llm, "model", "") or getattr(getattr(llm, "_target_llm", None), "model", ""))


def complete_text(
    llm: Any,
    prompt: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, LLMUsage]:
    """Run one non-streaming completion with the options actually applied."""
    estimated = estimate_tokens(prompt)
    if _is_ollama_llm(llm):
        target = getattr(llm, "_target_llm", llm)
        options = _options(llm, temperature, max_tokens)
        response = target.client.chat(
            model=_model_name(llm),
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options=options,
            keep_alive=getattr(target, "keep_alive", None),
        )
        text = str(response.message.content or "").strip()
        return text, _usage_from(response, options, estimated)

    try:
        raw = llm.complete(prompt, temperature=temperature, max_new_tokens=max_tokens)
    except TypeError:
        raw = llm.complete(prompt)
    return str(raw).strip(), LLMUsage(num_predict=max_tokens, estimated_prompt_tokens=estimated)


class StreamingCompletion:
    """Iterate text deltas; ``usage`` is filled in once the stream finishes.

    ``close()`` (or setting ``cancel_event``) stops iteration and closes the
    underlying HTTP stream so Ollama stops generating.
    """

    def __init__(
        self,
        llm: Any,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self._llm = llm
        self._prompt = prompt
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._cancel_event = cancel_event
        self._stream: Any = None
        self.cancelled = False
        self.usage = LLMUsage(num_predict=max_tokens, estimated_prompt_tokens=estimate_tokens(prompt))

    def __iter__(self) -> Iterator[str]:
        if _is_ollama_llm(self._llm):
            yield from self._iter_ollama()
        else:
            yield from self._iter_generic()

    def _cancelled(self) -> bool:
        if self._cancel_event is not None and self._cancel_event.is_set():
            self.cancelled = True
            return True
        return False

    def _iter_ollama(self) -> Iterator[str]:
        target = getattr(self._llm, "_target_llm", self._llm)
        options = _options(self._llm, self._temperature, self._max_tokens)
        self._stream = target.client.chat(
            model=_model_name(self._llm),
            messages=[{"role": "user", "content": self._prompt}],
            stream=True,
            options=options,
            keep_alive=getattr(target, "keep_alive", None),
        )
        try:
            for part in self._stream:
                if self._cancelled():
                    return
                delta = str(getattr(getattr(part, "message", None), "content", "") or "")
                if getattr(part, "done", False):
                    self.usage = _usage_from(part, options, self.usage.estimated_prompt_tokens or 0)
                if delta:
                    yield delta
        finally:
            self.close()

    def _iter_generic(self) -> Iterator[str]:
        try:
            stream = self._llm.stream_complete(
                self._prompt,
                temperature=self._temperature,
                max_new_tokens=self._max_tokens,
            )
        except TypeError:
            stream = self._llm.stream_complete(self._prompt)
        self._stream = stream
        try:
            for part in stream:
                if self._cancelled():
                    return
                delta = getattr(part, "delta", None)
                if delta is None:
                    delta = getattr(part, "text", None)
                if delta is None:
                    delta = str(part)
                delta = str(delta)
                if delta:
                    yield delta
        finally:
            self.close()

    def close(self) -> None:
        stream, self._stream = self._stream, None
        closer = getattr(stream, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass


def supports_streaming(llm: Any) -> bool:
    return _is_ollama_llm(llm) or hasattr(llm, "stream_complete")


def fit_output_budget(llm: Any, prompt: str, max_tokens: int, *, floor: int = 256) -> int:
    """Shrink ``max_tokens`` so prompt + output stays inside ``num_ctx``.

    Ollama silently truncates the *start* of an over-long prompt (instructions
    and the top-ranked sources). Never go below ``floor``; the caller records
    ``prompt_overflow`` from the real counts if the prompt itself is too large.
    """
    window = context_window(llm)
    if window is None:
        return max_tokens
    available = window - estimate_tokens(prompt) - 32
    return max(min(max_tokens, available), min(floor, max_tokens))
