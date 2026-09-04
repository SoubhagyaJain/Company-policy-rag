from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from typing import Any


def _normal(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def retrieval_ranking_metrics(
    retrieved_sections: Sequence[str],
    relevant_sections: Iterable[str],
    *,
    k: int = 10,
) -> dict[str, float]:
    """Compute MRR, binary NDCG@K, and Recall@K for a policy query."""
    relevant = {_normal(section) for section in relevant_sections if section}
    ranked = [_normal(section) for section in retrieved_sections[:k]]
    hits = [1 if item in relevant else 0 for item in ranked]
    first_rank = next((index + 1 for index, hit in enumerate(hits) if hit), None)
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / math.log2(index + 2) for index in range(ideal_hits))
    return {
        "mrr": 0.0 if first_rank is None else 1.0 / first_rank,
        "ndcg_at_k": 0.0 if idcg == 0 else dcg / idcg,
        "recall_at_k": 1.0 if not relevant else len(set(ranked).intersection(relevant)) / len(relevant),
    }


def evaluate_policy_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, float]:
    """Score governing-clause selection and answer constraint preservation."""
    ranking = retrieval_ranking_metrics(
        result.get("retrieved_sections", []),
        case.get("relevant_sections", []),
        k=int(result.get("k", 10)),
    )
    expected_primary = _normal(str(case.get("expected_primary_section", "")))
    actual_primary = _normal(str(result.get("selected_primary_section", "")))
    relevant_sections = {
        _normal(str(section)) for section in case.get("relevant_sections", []) if section
    }
    answer = _normal(str(result.get("answer", "")))
    conditions = [_normal(str(item)) for item in case.get("expected_conditions", []) if item]
    expected_abstain = bool(case.get("should_abstain", False))
    abstained = any(
        phrase in answer
        for phrase in ("could not find", "insufficient information", "cannot be determined")
    )
    ranking.update(
        {
            "governing_clause_accuracy": float(bool(expected_primary) and expected_primary == actual_primary),
            "section_accuracy": float(bool(actual_primary) and actual_primary in relevant_sections),
            "condition_preservation": (
                1.0 if not conditions else sum(condition in answer for condition in conditions) / len(conditions)
            ),
            "abstention_accuracy": float(expected_abstain == abstained),
        }
    )
    return ranking


def aggregate_policy_metrics(scored_cases: Sequence[dict[str, float]]) -> dict[str, float]:
    if not scored_cases:
        return {}
    keys = sorted({key for case in scored_cases for key in case})
    return {
        key: round(sum(case.get(key, 0.0) for case in scored_cases) / len(scored_cases), 4)
        for key in keys
    }
