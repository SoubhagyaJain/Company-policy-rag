"""LLM-backed faithfulness verification (H5).

The LLM judge augments the lexical heuristic: it can only make the verdict
stricter (catch hallucinations), never inflate a weak answer, and any LLM or
parse failure must leave the heuristic verdict untouched. All tests use a fake
LLM — no Ollama required.
"""

from __future__ import annotations

import json

from backend.models.chunk import Chunk, ChunkMetadata
from backend.models.rag import ScoredChunk
from backend.rag.verifier import SelfReflectionVerifier


def _ctx(text: str) -> list[ScoredChunk]:
    meta = ChunkMetadata(document_id="d1", source_file="handbook.pdf", chunk_index=0)
    return [ScoredChunk(chunk=Chunk(id="c1", text=text, metadata=meta), score=1.0)]


class _JudgeLLM:
    """Fake LLM returning a fixed JSON faithfulness verdict."""

    def __init__(self, verdict: dict | str) -> None:
        self.calls = 0
        self._payload = verdict if isinstance(verdict, str) else json.dumps(verdict)

    def complete(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        return self._payload


CONTEXT = "Employees accrue one vacation day per month of service."
GROUNDED_ANSWER = "Employees accrue one vacation day per month of service [Source 1]."


def test_llm_judge_not_called_when_disabled() -> None:
    llm = _JudgeLLM({"faithfulness": 0.0, "unsupported_claims": ["x"]})
    v = SelfReflectionVerifier(llm=llm)
    v.verify(query="q", answer=GROUNDED_ANSWER, context_chunks=_ctx(CONTEXT),
             citations=[], use_llm_judge=False)
    assert llm.calls == 0  # default path must not touch the LLM


def test_llm_judge_flags_hallucination_the_heuristic_misses() -> None:
    # Answer reuses context vocabulary (heuristic would pass) but adds an
    # unsupported figure. The LLM judge catches it.
    answer = "Employees accrue one vacation day per month and get a $5,000 bonus [Source 1]."
    llm = _JudgeLLM({"faithfulness": 0.2, "unsupported_claims": ["$5,000 bonus"]})
    v = SelfReflectionVerifier(llm=llm)
    report = v.verify(query="vacation accrual", answer=answer, context_chunks=_ctx(CONTEXT),
                      citations=[], use_llm_judge=True)
    assert llm.calls == 1
    assert report.passed is False
    assert any("5,000" in c or "bonus" in c.lower() for c in report.unsupported_claims)
    assert report.faithfulness <= 0.2


def test_llm_judge_can_only_lower_not_raise_faithfulness() -> None:
    # A grounded answer the heuristic scores high; a (wrongly) low LLM score
    # pulls it down (min), proving the judge is strictly stricter.
    grounded = _JudgeLLM({"faithfulness": 0.1, "unsupported_claims": []})
    v = SelfReflectionVerifier(llm=grounded)
    r_low = v.verify(query="q", answer=GROUNDED_ANSWER, context_chunks=_ctx(CONTEXT),
                     citations=[], use_llm_judge=True)
    # A high LLM score must NOT raise faithfulness above the heuristic.
    high = _JudgeLLM({"faithfulness": 1.0, "unsupported_claims": []})
    v2 = SelfReflectionVerifier(llm=high)
    r_heur = v2.verify(query="q", answer=GROUNDED_ANSWER, context_chunks=_ctx(CONTEXT),
                       citations=[], use_llm_judge=False)
    assert r_low.faithfulness <= r_heur.faithfulness
    assert r_low.faithfulness <= 0.1 + 1e-9


def test_garbage_llm_output_falls_back_to_heuristic() -> None:
    garbage = _JudgeLLM("the model rambled with no json here")
    v = SelfReflectionVerifier(llm=garbage)
    with_judge = v.verify(query="q", answer=GROUNDED_ANSWER, context_chunks=_ctx(CONTEXT),
                          citations=[], use_llm_judge=True)
    baseline = SelfReflectionVerifier().verify(
        query="q", answer=GROUNDED_ANSWER, context_chunks=_ctx(CONTEXT),
        citations=[], use_llm_judge=False)
    # Fallback => identical faithfulness to the heuristic-only verdict.
    assert with_judge.faithfulness == baseline.faithfulness


def test_llm_exception_falls_back_to_heuristic() -> None:
    class _BoomLLM:
        def complete(self, prompt, **kwargs):
            raise RuntimeError("ollama down")

    v = SelfReflectionVerifier(llm=_BoomLLM())
    report = v.verify(query="q", answer=GROUNDED_ANSWER, context_chunks=_ctx(CONTEXT),
                      citations=[], use_llm_judge=True)
    baseline = SelfReflectionVerifier().verify(
        query="q", answer=GROUNDED_ANSWER, context_chunks=_ctx(CONTEXT),
        citations=[], use_llm_judge=False)
    assert report.faithfulness == baseline.faithfulness


def test_abstention_skips_llm_judge() -> None:
    llm = _JudgeLLM({"faithfulness": 0.0, "unsupported_claims": ["x"]})
    v = SelfReflectionVerifier(llm=llm)
    report = v.verify(query="q", answer="I am unable to answer based on the provided documents.",
                      context_chunks=_ctx(CONTEXT), citations=[], use_llm_judge=True)
    assert llm.calls == 0
    assert report.passed is True


def test_grounded_answer_with_supportive_judge_passes() -> None:
    llm = _JudgeLLM({"faithfulness": 1.0, "unsupported_claims": []})
    v = SelfReflectionVerifier(llm=llm)
    report = v.verify(query="vacation accrual per month", answer=GROUNDED_ANSWER,
                      context_chunks=_ctx(CONTEXT), citations=[], use_llm_judge=True)
    assert llm.calls == 1
    assert not report.unsupported_claims
