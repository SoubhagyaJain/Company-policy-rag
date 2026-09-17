from __future__ import annotations

import math

import pytest

from backend.evaluation.retrieval_eval import (
    QueryLabels,
    context_coverage,
    evidence_flow,
    paired_bootstrap_ci,
    ranking_metrics,
    rerank_outcome,
    sign_flip_p_value,
    stage_rankings,
    win_loss_tie,
)


def _labels(graded: dict[str, int], aspects: dict[str, set[str]] | None = None) -> QueryLabels:
    return QueryLabels(query_id="q", question="q?", graded=graded, aspects=aspects or {})


def test_stage_rankings_append_unscored_candidates_after_reranked_pool() -> None:
    stages = {
        "subqueries": [{"dense": ["a", "b"], "bm25": ["b", "c"], "fused": ["b", "a", "c"]}],
        "candidates": ["b", "a", "c", "d"],
        "pre_rerank": ["b", "a", "c", "d"],
        "reranker_enabled": True,
        "rerank": [{"pool": ["b", "a"], "prefilter": ["a", "b"], "postfilter": ["a"]}],
        "post_rerank": ["a"],
        "final_context": ["a"],
    }
    rankings = stage_rankings(stages)
    assert rankings["post_rerank_prefilter"] == ["a", "b", "c", "d"]
    assert rankings["post_filter"] == ["a"]
    assert rankings["dense"] == ["a", "b"]


def test_stage_rankings_without_reranker_keep_fused_order() -> None:
    stages = {"candidates": ["x", "y"], "pre_rerank": ["y", "x"], "reranker_enabled": False, "post_rerank": ["y"]}
    assert stage_rankings(stages)["post_rerank_prefilter"] == ["y", "x"]


def test_ranking_metrics_graded_ndcg_and_recall() -> None:
    labels = _labels({"a": 2, "b": 1})
    perfect = ranking_metrics(["a", "b", "z"], labels)
    swapped = ranking_metrics(["b", "a", "z"], labels)
    assert perfect["ndcg@10"] == pytest.approx(1.0)
    assert swapped["ndcg@10"] < 1.0
    assert perfect["mrr"] == 1.0
    assert perfect["recall@5"] == 1.0
    assert ranking_metrics(["z", "z2", "b"], labels)["mrr"] == pytest.approx(1 / 3)
    assert ranking_metrics(["z"], labels)["hit@6"] == 0.0


def test_ranking_metrics_coverage_counts_aspects() -> None:
    labels = _labels({"a": 2}, aspects={"one": {"a"}, "two": {"q"}})
    assert ranking_metrics(["a"], labels)["coverage@6"] == 0.5
    assert math.isnan(ranking_metrics(["a"], _labels({"a": 2}))["coverage@6"])


def _outcome(pre: list[str], post: list[str], pool: list[str], relevant: dict[str, int], post_filter=None) -> str:
    rankings = {
        "candidates": pre,
        "pre_rerank": pre,
        "post_rerank_prefilter": post,
        "post_filter": post_filter if post_filter is not None else post[:6],
    }
    stages = {"reranker_enabled": True, "rerank": [{"pool": pool}]}
    return rerank_outcome(rankings, stages, _labels(relevant))


def test_rerank_outcome_buckets() -> None:
    pre = [f"c{i}" for i in range(10)]
    assert _outcome(pre, pre, pre, {"missing": 2}) == "not_retrieved"
    assert _outcome(pre, pre, pre[:5], {"c8": 2}) == "outside_pool"
    fixed_post = ["c8"] + [c for c in pre if c != "c8"]
    assert _outcome(pre, fixed_post, pre, {"c8": 2}) == "fixed"
    harmed_post = [c for c in pre if c != "c0"] + ["c0"]
    assert _outcome(pre, harmed_post, pre, {"c0": 2}) == "harmed"
    improved_post = ["c3"] + [c for c in pre if c != "c3"]
    assert _outcome(pre, improved_post, pre, {"c3": 2}) == "improved"
    assert _outcome(pre, pre, pre, {"c3": 2}) == "unchanged"
    assert _outcome(pre, pre, pre, {"c3": 2}, post_filter=["c0"]) == "filter_dropped"


def test_paired_statistics() -> None:
    baseline = [0.0, 0.5, 1.0, 0.5, 0.0, 1.0]
    better = [1.0, 1.0, 1.0, 1.0, 0.5, 1.0]
    ci = paired_bootstrap_ci(baseline, better, resamples=2000)
    assert ci["delta"] == pytest.approx(sum(b - a for a, b in zip(baseline, better)) / 6)
    assert ci["ci_low"] > 0
    assert sign_flip_p_value(baseline, baseline) == 1.0
    assert win_loss_tie(baseline, better) == {"wins": 4, "losses": 0, "ties": 2}


def test_stage_rankings_include_downstream_context_stages() -> None:
    stages = {
        "candidates": ["a", "b", "p"],
        "reranker_enabled": False,
        "post_rerank": ["a", "b"],
        "governing_selection": ["p", "a"],
        "post_governing": ["p", "a"],
        "post_packing": ["p", "a"],
        "final_context": ["p"],
    }
    rankings = stage_rankings(stages)
    assert rankings["governing_selection"] == ["p", "a"]
    assert rankings["post_governing"] == ["p", "a"]
    assert rankings["post_packing"] == ["p", "a"]
    assert rankings["final_context"] == ["p"]


def test_evidence_flow_counts_discarded_and_added_evidence() -> None:
    labels = _labels({"a": 2, "b": 1, "p": 1})
    flow = evidence_flow({"post_filter": ["a", "b", "z"], "final_context": ["p", "z"]}, labels)
    assert flow["relevant_discarded"] == 2.0  # a and b handed on, both gone
    assert flow["relevant_added"] == 1.0  # p pulled in from the pool
    assert flow["top1_discarded"] == 1.0
    assert flow["top1_relevant_discarded"] == 1.0
    assert flow["lost_all_evidence"] == 0.0  # p is still relevant
    kept = evidence_flow({"post_filter": ["a", "b"], "final_context": ["x", "a"]}, labels)
    assert kept["relevant_discarded"] == 1.0
    assert kept["top1_discarded"] == 0.0
    lost = evidence_flow({"post_filter": ["b"], "final_context": ["x"]}, _labels({"b": 1}))
    assert lost["lost_all_evidence"] == 1.0
    assert lost["top1_relevant_discarded"] == 1.0


def test_context_coverage_is_graded_gain_over_ideal_for_context_size() -> None:
    labels = _labels({"a": 2, "b": 1, "c": 1})
    assert context_coverage(["a"], labels) == pytest.approx(1.0)  # best single slot
    assert context_coverage(["b", "x"], labels) == pytest.approx(1 / 3)
    assert context_coverage(["a", "b", "c", "x"], labels) == pytest.approx(1.0)
    assert math.isnan(context_coverage(["a"], _labels({})))
