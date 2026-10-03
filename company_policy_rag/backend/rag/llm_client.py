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
    _prompt_overflow: bool | None = None

    @property
    def prompt_overflow(self) -> bool:
        """True when the prompt alone reached the context window (Ollama truncates it)."""
        if self._prompt_overflow is not None:
            return self._prompt_overflow
        if self.num_ctx is None:
            return False
        prompt = self.prompt_tokens if self.prompt_tokens is not None else self.estimated_prompt_tokens
        return prompt is not None and prompt >= self.num_ctx

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("_prompt_overflow")
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


def _sum_optional(first: int | float | None, second: int | float | None) -> int | float | None:
    if first is None and second is None:
        return None
    return (first or 0) + (second or 0)


def _combined_usage(first: LLMUsage, second: LLMUsage, max_tokens: int) -> LLMUsage:
    return LLMUsage(
        prompt_tokens=_sum_optional(first.prompt_tokens, second.prompt_tokens),
        completion_tokens=_sum_optional(first.completion_tokens, second.completion_tokens),
        num_ctx=second.num_ctx,
        num_predict=max_tokens,
        estimated_prompt_tokens=_sum_optional(first.estimated_prompt_tokens, second.estimated_prompt_tokens),
        prompt_ms=_sum_optional(first.prompt_ms, second.prompt_ms),
        generation_ms=_sum_optional(first.generation_ms, second.generation_ms),
        done_reason=second.done_reason,
        _prompt_overflow=first.prompt_overflow or second.prompt_overflow,
    )


def _thinking_options(options: dict[str, Any], thinking_budget: int) -> dict[str, Any]:
    max_tokens = int(options["num_predict"])
    return {**options, "num_predict": min(thinking_budget, max(1, max_tokens // 3))}


def _bounded_thinking_followup(
    prompt: str,
    options: dict[str, Any],
    first_content: str,
    thought: str,
    first_usage: LLMUsage,
) -> tuple[str | None, list[dict[str, str]], dict[str, Any]]:
    """Prepare a final-answer turn when a bounded thinking turn ran out."""
    max_tokens = int(options["num_predict"])
    if first_content and first_usage.done_reason != "length":
        return first_content, [], options

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": first_content, "thinking": thought},
        {
            "role": "user",
            "content": "Provide the final answer to the original request now, using the supplied evidence. Do not describe private reasoning.",
        },
    ]
    # The second prompt includes the first turn's thinking and a short follow-up.
    # Reserve some context for those additions as well as the visible answer.
    final_options = {
        **options,
        "num_predict": max(1, max_tokens - (first_usage.completion_tokens or max_tokens // 3) - 48),
    }
    return None, messages, final_options


def _bounded_thinking_start(
    llm: Any,
    prompt: str,
    options: dict[str, Any],
    thinking_budget: int,
) -> tuple[str | None, LLMUsage, list[dict[str, str]], dict[str, Any]]:
    """Give Qwen a short thinking turn, then prepare a final-answer turn if needed."""
    target = getattr(llm, "_target_llm", llm)
    first_options = _thinking_options(options, thinking_budget)
    first = target.client.chat(
        model=_model_name(llm),
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        think=True,
        options=first_options,
        keep_alive=getattr(target, "keep_alive", None),
    )
    first_usage = _usage_from(first, first_options, estimate_tokens(prompt))
    first_content = str(first.message.content or "").strip()
    thought = str(getattr(first.message, "thinking", "") or "")
    ready, messages, final_options = _bounded_thinking_followup(
        prompt, options, first_content, thought, first_usage
    )
    return ready, first_usage, messages, final_options


def complete_text(
    llm: Any,
    prompt: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    think: bool | None = None,
    thinking_budget: int | None = None,
) -> tuple[str, LLMUsage]:
    """Run one non-streaming completion with the options actually applied."""
    estimated = estimate_tokens(prompt)
    if _is_ollama_llm(llm):
        target = getattr(llm, "_target_llm", llm)
        options = _options(llm, temperature, max_tokens)
        first_usage: LLMUsage | None = None
        messages = [{"role": "user", "content": prompt}]
        if think is True and thinking_budget is not None and max_tokens is not None and max_tokens > 64:
            ready, first_usage, messages, options = _bounded_thinking_start(
                llm, prompt, options, thinking_budget
            )
            if ready is not None:
                first_usage.num_predict = max_tokens
                return ready, first_usage
        response = target.client.chat(
            model=_model_name(llm),
            messages=messages,
            stream=False,
            think=False if first_usage is not None else (think if think is not None else getattr(target, "thinking", None)),
            options=options,
            keep_alive=getattr(target, "keep_alive", None),
        )
        text = str(response.message.content or "").strip()
        usage = _usage_from(response, options, estimate_tokens(str(messages)) if first_usage is not None else estimated)
        if first_usage is not None:
            usage = _combined_usage(first_usage, usage, max_tokens)
        return text, usage

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
        think: bool | None = None,
        thinking_budget: int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self._llm = llm
        self._prompt = prompt
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._think = think
        self._thinking_budget = thinking_budget
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
        first_usage: LLMUsage | None = None
        messages = [{"role": "user", "content": self._prompt}]
        if self._cancelled():
            return
        if self._think is True and self._thinking_budget is not None and self._max_tokens is not None and self._max_tokens > 64:
            first_options = _thinking_options(options, self._thinking_budget)
            first_content_parts: list[str] = []
            thought_parts: list[str] = []
            self._stream = target.client.chat(
                model=_model_name(self._llm),
                messages=messages,
                stream=True,
                think=True,
                options=first_options,
                keep_alive=getattr(target, "keep_alive", None),
            )
            try:
                for part in self._stream:
                    if self._cancelled():
                        return
                    first_content_parts.append(str(getattr(part.message, "content", "") or ""))
                    thought_parts.append(str(getattr(part.message, "thinking", "") or ""))
                    if getattr(part, "done", False):
                        first_usage = _usage_from(part, first_options, estimate_tokens(self._prompt))
            finally:
                self.close()
            if self._cancelled():
                return
            if first_usage is None:
                return
            ready, messages, options = _bounded_thinking_followup(
                self._prompt,
                options,
                "".join(first_content_parts).strip(),
                "".join(thought_parts),
                first_usage,
            )
            if ready is not None:
                first_usage.num_predict = self._max_tokens
                self.usage = first_usage
                yield ready
                return
        self._stream = target.client.chat(
            model=_model_name(self._llm),
            messages=messages,
            stream=True,
            think=False if first_usage is not None else (self._think if self._think is not None else getattr(target, "thinking", None)),
            options=options,
            keep_alive=getattr(target, "keep_alive", None),
        )
        try:
            for part in self._stream:
                if self._cancelled():
                    return
                delta = str(getattr(getattr(part, "message", None), "content", "") or "")
                if getattr(part, "done", False):
                    usage = _usage_from(
                        part,
                        options,
                        estimate_tokens(str(messages)) if first_usage is not None else self.usage.estimated_prompt_tokens or 0,
                    )
                    self.usage = _combined_usage(first_usage, usage, self._max_tokens) if first_usage is not None else usage
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
