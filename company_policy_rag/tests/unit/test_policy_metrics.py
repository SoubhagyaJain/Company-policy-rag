from backend.evaluation.policy_metrics import (
    aggregate_policy_metrics,
    evaluate_policy_case,
    retrieval_ranking_metrics,
)
from scripts.build_policy_reliability_dataset import build_cases


def test_retrieval_metrics_reward_early_governing_clause() -> None:
    metrics = retrieval_ranking_metrics(["GENERAL SAFETY", "PRESCRIBED DRUGS"], ["PRESCRIBED DRUGS"])
    assert metrics["mrr"] == 0.5
    assert metrics["recall_at_k"] == 1.0
    assert 0.6 < metrics["ndcg_at_k"] < 0.7


def test_policy_case_scores_clause_conditions_and_abstention() -> None:
    case = {
        "relevant_sections": ["AFTER HOURS CALLS"],
        "expected_primary_section": "AFTER HOURS CALLS",
        "expected_conditions": ["eight hours", "overtime"],
        "should_abstain": False,
    }
    result = {
        "retrieved_sections": ["AFTER HOURS CALLS"],
        "selected_primary_section": "AFTER HOURS CALLS",
        "answer": "The eight hours must be preserved; the affected period is overtime.",
    }
    metrics = evaluate_policy_case(case, result)
    assert metrics["governing_clause_accuracy"] == 1.0
    assert metrics["condition_preservation"] == 1.0
    assert metrics["abstention_accuracy"] == 1.0
    assert aggregate_policy_metrics([metrics])["mrr"] == 1.0


def test_policy_reliability_dataset_has_100_balanced_cases() -> None:
    cases = build_cases()
    assert len(cases) == 100
    assert len({case["category"] for case in cases}) == 10

