"""Backend-native retrieval evaluation.

Turns ``RAGTrace.retrieval_stages`` (recorded by ``RAGPipeline``) into per-stage
rankings, scores them against graded chunk-level relevance labels, and compares
configurations with paired statistics. Used by ``scripts/eval_retrieval_backend.py``.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Order matters: this is the order stages run in the pipeline.
STAGES: tuple[str, ...] = (
    "dense",
    "bm25",
    "fusion",
    "candidates",
    "pre_rerank",
    "post_rerank_prefilter",
    "post_filter",
    "governing_selection",
    "post_governing",
    "post_packing",
    "final_context",
)
# Stages after the rerank hand-off, in pipeline order.
DOWNSTREAM_STAGES: tuple[str, ...] = ("post_filter", "post_governing", "post_packing", "final_context")
RECALL_KS: tuple[int, ...] = (5, 10, 20)
HIT_CUTOFF = 6
NDCG_K = 10


@dataclass
class QueryLabels:
    """Graded relevance for one query. Grades: 2 = directly answers, 1 = partial."""

    query_id: str
    question: str
    graded: dict[str, int]
    # Coverage aspects (e.g. each item an enumeration asks for) -> chunk ids covering it.
    aspects: dict[str, set[str]] = field(default_factory=dict)
    category: str = ""
    style: str = ""

    @property
    def relevant(self) -> set[str]:
        return {cid for cid, grade in self.graded.items() if grade > 0}


def load_labels(path: str | Path) -> tuple[dict[str, Any], list[QueryLabels]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    queries = [
        QueryLabels(
            query_id=item["id"],
            question=item["question"],
            graded={cid: int(grade) for cid, grade in (item.get("relevant") or {}).items()},
            aspects={name: set(ids) for name, ids in (item.get("aspects") or {}).items()},
            category=item.get("category", ""),
            style=item.get("style", ""),
        )
        for item in data["queries"]
    ]
    meta = {key: value for key, value in data.items() if key != "queries"}
    return meta, queries


# ── Stage rankings ─────────────────────────────────────────────────────────


def _dedupe(ids: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for cid in ids:
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def _interleave(lists: Sequence[Sequence[str]]) -> list[str]:
    merged: list[str] = []
    for tier in range(max((len(x) for x in lists), default=0)):
        for ranked in lists:
            if tier < len(ranked):
                merged.append(ranked[tier])
    return _dedupe(merged)


def stage_rankings(stages: dict[str, Any]) -> dict[str, list[str]]:
    """Ranked chunk ids per evaluation stage from ``RAGTrace.retrieval_stages``.

    ``post_rerank_prefilter`` is a full ranking: the cross-encoder's order over
    the scored pool, followed by unscored candidates in fused order, so recall
    at depth stays comparable with the no-reranker configuration.
    ``post_filter`` is what the rerank stage actually hands on (after the score
    filter and top-n cut).
    """
    subqueries = stages.get("subqueries") or []
    primary = subqueries[0] if subqueries else {}
    candidates = list(stages.get("candidates") or [])
    pre = list(stages.get("pre_rerank") or candidates)
    traces = [t for t in (stages.get("rerank") or []) if t.get("prefilter") is not None]

    if stages.get("reranker_enabled") is False or not traces:
        prefilter = list(pre)
    elif len(traces) == 1:
        scored = list(traces[0]["prefilter"])
        prefilter = scored + [cid for cid in pre if cid not in set(scored)]
    else:
        scored = _interleave([t["prefilter"] for t in traces])
        prefilter = scored + [cid for cid in pre if cid not in set(scored)]

    return {
        "dense": list(primary.get("dense") or []),
        "bm25": list(primary.get("bm25") or []),
        "fusion": list(primary.get("fused") or []),
        "candidates": candidates,
        "pre_rerank": pre,
        "post_rerank_prefilter": prefilter,
        "post_filter": list(stages.get("post_rerank") or []),
        "governing_selection": list(stages.get("governing_selection") or []),
        "post_governing": list(stages.get("post_governing") or []),
        "post_packing": list(stages.get("post_packing") or []),
        "final_context": list(stages.get("final_context") or []),
    }


# ── Metrics ────────────────────────────────────────────────────────────────


def first_relevant_rank(ranked: Sequence[str], relevant: set[str]) -> int | None:
    return next((i + 1 for i, cid in enumerate(ranked) if cid in relevant), None)


def ranking_metrics(ranked: Sequence[str], labels: QueryLabels) -> dict[str, float]:
    relevant = labels.relevant
    graded = labels.graded
    metrics: dict[str, float] = {}
    for k in RECALL_KS:
        metrics[f"recall@{k}"] = (
            len(relevant.intersection(ranked[:k])) / len(relevant) if relevant else math.nan
        )
    top_hit = ranked[:HIT_CUTOFF]
    metrics[f"hit@{HIT_CUTOFF}"] = float(bool(relevant.intersection(top_hit)))
    metrics[f"direct_hit@{HIT_CUTOFF}"] = float(
        any(graded.get(cid, 0) >= 2 for cid in top_hit)
    )
    first = first_relevant_rank(ranked, relevant)
    metrics["mrr"] = 0.0 if first is None else 1.0 / first
    dcg = sum(
        (2 ** graded.get(cid, 0) - 1) / math.log2(i + 2) for i, cid in enumerate(ranked[:NDCG_K])
    )
    ideal = sorted(graded.values(), reverse=True)[:NDCG_K]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    metrics[f"ndcg@{NDCG_K}"] = dcg / idcg if idcg > 0 else math.nan
    if labels.aspects:
        present = set(ranked[:HIT_CUTOFF])
        covered = sum(1 for ids in labels.aspects.values() if ids & present)
        metrics[f"coverage@{HIT_CUTOFF}"] = covered / len(labels.aspects)
    else:
        metrics[f"coverage@{HIT_CUTOFF}"] = math.nan
    return metrics


def context_coverage(context: Sequence[str], labels: QueryLabels) -> float:
    """Graded gain in the context over the best gain that many slots could hold.

    Uses the context size, capped at ``HIT_CUTOFF``, so a short context is not
    penalised for slots it never had.
    """
    slots = min(HIT_CUTOFF, max(len(context), 1))
    ideal = sum(sorted(labels.graded.values(), reverse=True)[:slots])
    if ideal <= 0:
        return math.nan
    gained = sum(labels.graded.get(cid, 0) for cid in _dedupe(context)[:slots])
    return gained / ideal


def evidence_flow(rankings: dict[str, list[str]], labels: QueryLabels) -> dict[str, float]:
    """What happens to the rerank hand-off between context assembly and the LLM.

    relevant_discarded: relevant chunks handed on by the rerank stage that are
    absent from the final context. relevant_added: relevant chunks in the final
    context that the hand-off did not contain (pulled from the candidate pool).
    top1_discarded / top1_relevant_discarded: the hand-off's rank-1 chunk (any,
    or only when relevant) is absent from the final context.
    """
    relevant = labels.relevant
    handoff = rankings.get("post_filter") or []
    final = rankings.get("final_context") or []
    final_set = set(final)
    handoff_rel = relevant.intersection(handoff)
    final_rel = relevant.intersection(final)
    top1 = handoff[0] if handoff else None
    return {
        "handoff_relevant": float(len(handoff_rel)),
        "final_relevant": float(len(final_rel)),
        "relevant_discarded": float(len(handoff_rel - final_rel)),
        "relevant_added": float(len(final_rel - handoff_rel)),
        "top1_discarded": float(top1 is not None and top1 not in final_set),
        "top1_relevant_discarded": float(top1 is not None and top1 in relevant and top1 not in final_set),
        "lost_all_evidence": float(bool(handoff_rel) and not final_rel),
        "context_coverage": context_coverage(final, labels),
        "context_size": float(len(final)),
    }


def rerank_outcome(rankings: dict[str, list[str]], stages: dict[str, Any], labels: QueryLabels) -> str:
    """Classify what the rerank stage did for one query.

    not_retrieved: no relevant chunk in the candidate pool at all.
    outside_pool: relevant chunks retrieved, but none inside the scored pool.
    fixed / harmed: first relevant chunk moved into / out of the top 6.
    improved / worsened: both in the top 6, first relevant rank moved up / down.
    unchanged / both_miss: same rank / relevant chunk outside the top 6 either way.
    filter_dropped: ranked in the top 6 by the cross-encoder but removed by the score filter.
    """
    relevant = labels.relevant
    if not relevant.intersection(rankings["candidates"]):
        return "not_retrieved"
    traces = stages.get("rerank") or []
    if stages.get("reranker_enabled") is False or not traces:
        return "no_reranker"
    pool = set().union(*(set(t.get("pool") or []) for t in traces))
    if not relevant.intersection(pool):
        return "outside_pool"
    pre = first_relevant_rank(rankings["pre_rerank"], relevant)
    post = first_relevant_rank(rankings["post_rerank_prefilter"], relevant)
    in_pre = pre is not None and pre <= HIT_CUTOFF
    in_post = post is not None and post <= HIT_CUTOFF
    if in_post and not relevant.intersection(rankings["post_filter"]):
        return "filter_dropped"
    if not in_pre and in_post:
        return "fixed"
    if in_pre and not in_post:
        return "harmed"
    if not in_pre and not in_post:
        return "both_miss"
    if post < pre:
        return "improved"
    if post > pre:
        return "worsened"
    return "unchanged"


# ── Paired statistics ──────────────────────────────────────────────────────


def _finite_pairs(a: Sequence[float], b: Sequence[float]) -> list[tuple[float, float]]:
    return [(x, y) for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]


def paired_bootstrap_ci(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 13,
) -> dict[str, float]:
    """Mean of (candidate - baseline) with a percentile bootstrap CI over queries."""
    pairs = _finite_pairs(baseline, candidate)
    if not pairs:
        return {"delta": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0}
    deltas = [y - x for x, y in pairs]
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    lo = means[int(math.floor(alpha / 2 * resamples))]
    hi = means[min(resamples - 1, int(math.ceil((1 - alpha / 2) * resamples)) - 1)]
    return {"delta": sum(deltas) / n, "ci_low": lo, "ci_high": hi, "n": n}


def sign_flip_p_value(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    resamples: int = 20_000,
    seed: int = 13,
) -> float:
    """Two-sided paired randomization test on the mean difference."""
    deltas = [y - x for x, y in _finite_pairs(baseline, candidate)]
    if not deltas or all(d == 0 for d in deltas):
        return 1.0
    observed = abs(sum(deltas))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(resamples):
        total = sum(d if rng.random() < 0.5 else -d for d in deltas)
        if abs(total) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (resamples + 1)


def win_loss_tie(
    baseline: Sequence[float], candidate: Sequence[float], eps: float = 1e-9
) -> dict[str, int]:
    wins = losses = ties = 0
    for x, y in _finite_pairs(baseline, candidate):
        if y > x + eps:
            wins += 1
        elif y < x - eps:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties}


def mean(values: Sequence[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else math.nan


def percentile(values: Sequence[float], q: float) -> float:
    finite = sorted(v for v in values if not math.isnan(v))
    if not finite:
        return math.nan
    idx = min(len(finite) - 1, max(0, int(round(q * (len(finite) - 1)))))
    return finite[idx]
