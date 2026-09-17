from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

from backend.evaluation.answer_eval import (
    AnswerCase,
    compare,
    is_abstention,
    keyword_recall,
    load_cases,
    score_response,
    summarize,
    write_spot_check,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _response(answer: str, context_ids: list[str], cited: list[str], fallback: list[str] | None = None):
    citations = [SimpleNamespace(chunk_id=cid, selection_reason="cited_in_answer") for cid in cited]
    citations += [SimpleNamespace(chunk_id=cid, selection_reason="score_threshold_fallback") for cid in fallback or []]
    return SimpleNamespace(
        answer=answer,
        context_chunks=[SimpleNamespace(chunk=SimpleNamespace(id=cid)) for cid in context_ids],
        citations=citations,
        trace=SimpleNamespace(
            llm_usage={"prompt_tokens": 1200, "completion_tokens": 60, "prompt_overflow": False},
            fallback_reason="none",
            faithfulness_passed=True,
            query_scope="global",
            retrieval_stages={"policy_block_tokens": 0},
        ),
    )


def test_keyword_recall_accepts_alternatives_and_word_boundaries() -> None:
    assert keyword_recall("Employees get 20 days.", ["20|twenty"]) == 1.0
    assert keyword_recall("Employees get twenty days.", ["20|twenty", "31 March"]) == 0.5
    assert keyword_recall("Employees get 200 days.", ["20|twenty"]) == 0.0
    assert math.isnan(keyword_recall("anything", []))


def test_abstention_detection() -> None:
    assert is_abstention("I could not find this information in the provided document.")
    assert is_abstention("The handbook does not mention a dental plan.")
    assert not is_abstention("Full-time employees receive 20 paid annual-leave days [Source 1].")


def test_score_response_citation_and_context_metrics() -> None:
    case = AnswerCase("q", "handbook", "How many days?", {"nsl_04": 2}, keywords=["20|twenty"])
    row = score_response(
        case,
        _response("You get 20 days [Source 1] [Source 3].", ["nsl_04", "nsl_02", "nsl_06"], ["nsl_04", "nsl_06"]),
        latency_ms=4200.0,
    )
    assert row["context_hit"] == 1.0
    assert row["keyword_recall"] == 1.0
    assert row["tagged_citation"] == 1.0
    assert row["citation_precision"] == 0.5
    assert row["citation_in_context"] == 1.0
    assert row["abstention_correct"] == 1.0
    assert row["prompt_tokens"] == 1200


def test_fallback_citations_do_not_count_as_tagged() -> None:
    case = AnswerCase("q", "handbook", "How many days?", {"nsl_04": 2}, keywords=["20"])
    row = score_response(case, _response("You get 20 days.", ["nsl_04"], [], fallback=["nsl_04"]), latency_ms=1.0)
    assert row["tagged_citation"] == 0.0
    assert math.isnan(row["citation_precision"])


def test_abstention_cases_score_correct_only_when_abstaining() -> None:
    case = AnswerCase("dental", "handbook", "Dental plan?", {}, should_abstain=True)
    good = score_response(case, _response("The handbook does not mention a dental plan.", ["nsl_00"], []), latency_ms=1.0)
    bad = score_response(case, _response("Northstar offers a PPO plan [Source 1].", ["nsl_00"], ["nsl_00"]), latency_ms=1.0)
    assert good["abstention_correct"] == 1.0
    assert bad["abstention_correct"] == 0.0
    assert math.isnan(good["context_hit"])


def test_aspect_coverage_for_multi_part_questions() -> None:
    case = AnswerCase(
        "multi", "handbook", "leave and meals?", {"nsl_04": 2, "nsl_02": 2},
        aspects={"leave": {"nsl_04"}, "meals": {"nsl_02"}},
    )
    row = score_response(case, _response("20 days [Source 1].", ["nsl_04"], ["nsl_04"]), latency_ms=1.0)
    assert row["context_coverage"] == 0.5


def test_summarize_and_compare(tmp_path) -> None:
    case = AnswerCase("q", "handbook", "How many days?", {"nsl_04": 2}, keywords=["20"])
    base = [dict(score_response(case, _response("unknown", ["nsl_02"], []), latency_ms=5000.0), config="a")]
    cand = [dict(score_response(case, _response("20 days [Source 1]", ["nsl_04"], ["nsl_04"]), latency_ms=4000.0), config="b")]

    summary = summarize(cand)
    assert summary["n"] == 1 and summary["keyword_recall"] == 1.0 and summary["latency_p50_ms"] == 4000.0

    diff = compare(base, cand)
    assert diff["keyword_recall"]["delta"] == 1.0
    assert diff["context_hit"]["wins"] == 1
    assert diff["latency_ms"]["wins"] == 1  # candidate is faster

    path = write_spot_check(cand, tmp_path / "spot.csv")
    assert "human_correct" in path.read_text(encoding="utf-8")


def test_handbook_labels_load_and_reference_real_chunks() -> None:
    import json

    cases = load_cases("handbook", PROJECT_ROOT / "data/eval/retrieval/handbook_labels.json")
    corpus_ids = {c["id"] for c in json.loads((PROJECT_ROOT / "data/eval/retrieval/handbook_corpus.json").read_text(encoding="utf-8"))}
    assert len(cases) == 24
    assert sum(c.should_abstain for c in cases) == 3
    for case in cases:
        assert case.relevant_ids <= corpus_ids
        assert case.should_abstain or case.relevant_ids


def test_negative_policy_facts_are_not_abstentions() -> None:
    assert not is_abstention("No, contractors are not covered by the employee parental leave benefit [Source 1].")
    assert not is_abstention("Customer data must not be copied to unapproved AI services.")
    assert is_abstention("Dental coverage is not mentioned in the handbook.")
    assert is_abstention("The provided documents do not specify a 401(k) match.")
    assert is_abstention("Based on the provided evidence, the employee handbook does not explicitly mention the owner.")


def test_trailing_hedge_after_an_answer_is_not_an_abstention() -> None:
    answer = (
        "Contractors may work remotely only when their sponsor and service agreement permit it [Source 1].\n\n"
        "- Sponsor approval is required.\n\n"
        "The handbook does not provide any additional exceptions for contractors."
    )
    assert not is_abstention(answer)
