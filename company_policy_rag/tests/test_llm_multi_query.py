"""LLM-based multi-query decomposition (H2).

Replaces the corpus-overfit keyword tables with one LLM decomposition call,
keeping the heuristic tables as a fallback. All tests use a fake LLM — no Ollama.
"""

from __future__ import annotations

import json

from backend.rag.multi_query import MultiQueryGenerator, _parse_subqueries


class _LLM:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0

    def complete(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        return self.payload


# ── _parse_subqueries ───────────────────────────────────────────────────────

def test_parse_json_array() -> None:
    out = _parse_subqueries('["what is the leave policy", "how do I expense travel"]', 8)
    assert out == ["what is the leave policy", "how do I expense travel"]


def test_parse_json_with_surrounding_prose() -> None:
    raw = 'Sure, here you go:\n["query one here", "query two here"]\nHope that helps'
    assert _parse_subqueries(raw, 8) == ["query one here", "query two here"]


def test_parse_newline_fallback() -> None:
    raw = "1. first search query\n2. second search query\n- third one"
    out = _parse_subqueries(raw, 8)
    assert out == ["first search query", "second search query", "third one"]


def test_parse_dedupes_and_caps() -> None:
    out = _parse_subqueries('["a query", "a query", "b query", "c query"]', 2)
    assert out == ["a query", "b query"]  # deduped, then capped at 2


# ── generate_subqueries: LLM path ───────────────────────────────────────────

def test_llm_path_used_when_available() -> None:
    llm = _LLM(json.dumps(["leave accrual policy", "carry over unused leave"]))
    gen = MultiQueryGenerator(llm=llm)
    out = gen.generate_subqueries("How does leave accrual and carry-over work?")
    assert llm.calls == 1
    assert out[0] == "How does leave accrual and carry-over work?"  # original first
    assert "leave accrual policy" in out
    assert "carry over unused leave" in out


def test_llm_result_cached_across_calls() -> None:
    llm = _LLM(json.dumps(["q one", "q two"]))
    gen = MultiQueryGenerator(llm=llm)
    q = "some comprehensive question"
    gen.generate_subqueries(q)
    gen.generate_subqueries(q)
    assert llm.calls == 1  # memoized, no second LLM hit (retry-loop safe)


# ── fallback to heuristic ───────────────────────────────────────────────────

def test_no_llm_uses_heuristic() -> None:
    gen = MultiQueryGenerator()  # no llm
    out = gen.generate_subqueries("What are the six building blocks of AI agents?")
    # Heuristic building-blocks expansion fires.
    assert any("building block" in q.lower() for q in out)


def test_garbage_llm_falls_back_to_heuristic() -> None:
    llm = _LLM("the model rambled with no array")
    gen = MultiQueryGenerator(llm=llm)
    out = gen.generate_subqueries("What are the six building blocks of AI agents?")
    assert llm.calls == 1
    assert any("building block" in q.lower() for q in out)  # heuristic used


def test_llm_exception_falls_back_to_heuristic() -> None:
    class _Boom:
        def complete(self, *a, **k):
            raise RuntimeError("ollama down")
    gen = MultiQueryGenerator(llm=_Boom())
    out = gen.generate_subqueries("What are the six building blocks of AI agents?")
    assert any("building block" in q.lower() for q in out)


def test_disabled_flag_forces_heuristic(monkeypatch) -> None:
    import backend.rag.multi_query as mq
    monkeypatch.setattr(mq.settings, "enable_llm_multi_query", False, raising=False)
    llm = _LLM(json.dumps(["should not be used"]))
    gen = MultiQueryGenerator(llm=llm)
    out = gen.generate_subqueries("What are the six building blocks of AI agents?")
    assert llm.calls == 0  # flag off -> LLM never called
    assert any("building block" in q.lower() for q in out)


def test_empty_query_returns_empty() -> None:
    gen = MultiQueryGenerator(llm=_LLM("[]"))
    assert gen.generate_subqueries("   ") == []


def test_respects_max_queries() -> None:
    llm = _LLM(json.dumps([f"query number {i}" for i in range(20)]))
    gen = MultiQueryGenerator(llm=llm)
    out = gen.generate_subqueries("a big comprehensive question", max_queries=5)
    assert len(out) <= 5
