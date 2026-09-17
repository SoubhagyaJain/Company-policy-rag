"""Answer-level evaluation of the shipped pipeline.

``retrieval_eval`` stops at the formatted context. This module scores what a
user actually receives from ``RAGPipeline.query``: the answer text, its
citations, whether it abstains when it should, latency, and the token counts
Ollama reported. Used by ``scripts/eval_answers_backend.py``.

Metrics are deliberately simple and deterministic so runs are comparable:

- ``context_hit`` / ``context_coverage``: a labelled relevant chunk reached the
  prompt (coverage is over coverage aspects when the label has them).
- ``tagged_citation``: the answer carries at least one [Source N] tag.
- ``citation_precision``: share of cited chunks that are labelled relevant.
- ``citation_in_context``: share of cited chunks that were actually in the prompt.
- ``keyword_recall``: share of answer facts (``keywords``; ``a|b`` = either)
  present in the answer.
- ``abstention_correct``: abstained exactly when ``should_abstain``.
- ``clean_format``: the answer does not copy a prompt source header line.
- ``faithfulness`` (optional): an LLM judge over the prompt context.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.evaluation.retrieval_eval import (
    mean,
    paired_bootstrap_ci,
    percentile,
    sign_flip_p_value,
    win_loss_tie,
)

# Abstention = a statement about the evidence, not a negative policy fact:
# "contractors are not covered by the benefit" is an answer, while "the handbook
# does not mention a dental plan" is an abstention.
_EVIDENCE_SUBJECT = r"(?:document|documents|handbook|guidebook|context|evidence|sources?|provided (?:text|information|documents?)|retrieved (?:text|excerpts?|sources?))"
_ABSTENTION_RE = re.compile(
    r"could not find|couldn't find|cannot find|can't find|unable to (?:answer|find|locate)|"
    r"no (?:information|details|mention) (?:about|on|of|regarding)|"
    rf"{_EVIDENCE_SUBJECT} (?:does|do) not (?:explicitly |specifically |directly |clearly )?(?:contain|mention|specify|include|provide|say|state|cover|address)|"
    rf"(?:not|isn't|is not|are not) (?:mentioned|specified|covered|included|provided|stated|addressed) in the {_EVIDENCE_SUBJECT}|"
    rf"not (?:available|found) in the {_EVIDENCE_SUBJECT}",
    re.IGNORECASE,
)


@dataclass
class AnswerCase:
    case_id: str
    corpus: str
    question: str
    relevant: dict[str, int]
    keywords: list[str] = field(default_factory=list)
    aspects: dict[str, set[str]] = field(default_factory=dict)
    category: str = ""
    should_abstain: bool = False

    @property
    def relevant_ids(self) -> set[str]:
        return {cid for cid, grade in self.relevant.items() if grade > 0}


def load_cases(corpus: str, path: str | Path) -> list[AnswerCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        AnswerCase(
            case_id=item["id"],
            corpus=corpus,
            question=item["question"],
            relevant={cid: int(grade) for cid, grade in (item.get("relevant") or {}).items()},
            keywords=[str(k) for k in item.get("keywords") or [] if str(k).strip()],
            aspects={name: set(ids) for name, ids in (item.get("aspects") or {}).items()},
            category=item.get("category", ""),
            should_abstain=bool(item.get("should_abstain", False)),
        )
        for item in data["queries"]
    ]


_LEAD_SENTENCES = 2
_HEADER_ECHO_RE = re.compile(r"Evidence Type:|\bFile:\s*\S+\s*\||</?(?:visual_)?source\b", re.IGNORECASE)


def echoes_source_header(answer: str) -> bool:
    """The answer copied a prompt source header / metadata line."""
    return bool(_HEADER_ECHO_RE.search(answer or ""))


def is_abstention(answer: str) -> bool:
    """True when the answer *opens* by saying the evidence lacks the answer.

    Only the lead is checked: a real answer often ends with a hedge such as
    "the handbook does not provide any further exceptions", which is not an
    abstention.
    """
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", (answer or "").strip()) if s.strip()]
    return bool(_ABSTENTION_RE.search(" ".join(sentences[:_LEAD_SENTENCES])))


def keyword_recall(answer: str, keywords: Sequence[str]) -> float:
    if not keywords:
        return math.nan
    text = (answer or "").casefold()
    hits = 0
    for keyword in keywords:
        options = [opt.strip().casefold() for opt in keyword.split("|") if opt.strip()]
        if any(re.search(rf"(?<![\w]){re.escape(opt)}(?![\w])", text) for opt in options):
            hits += 1
    return hits / len(keywords)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def score_response(case: AnswerCase, response: Any, *, latency_ms: float) -> dict[str, Any]:
    """Score one ``RAGResponse`` against its case. Returns a JSON-serialisable row."""
    trace = response.trace
    context_ids = [sc.chunk.id for sc in response.context_chunks]
    context_set = set(context_ids)
    tagged = [c for c in response.citations if getattr(c, "selection_reason", "") == "cited_in_answer"]
    cited_ids = [c.chunk_id for c in tagged]
    relevant = case.relevant_ids
    abstained = is_abstention(response.answer)

    if case.aspects:
        covered = sum(1 for ids in case.aspects.values() if ids & context_set)
        coverage = covered / len(case.aspects)
    else:
        coverage = math.nan if not relevant else float(bool(relevant & context_set))

    usage = dict(getattr(trace, "llm_usage", {}) or {})
    return {
        "corpus": case.corpus,
        "case_id": case.case_id,
        "category": case.category,
        "question": case.question,
        "answer": response.answer,
        "should_abstain": case.should_abstain,
        "abstained": abstained,
        "abstention_correct": float(abstained == case.should_abstain),
        "context_ids": context_ids,
        "cited_ids": cited_ids,
        "context_hit": math.nan if not relevant else float(bool(relevant & context_set)),
        "context_coverage": coverage,
        "tagged_citation": float(bool(tagged)) if not case.should_abstain else math.nan,
        "citation_precision": _ratio(sum(1 for cid in cited_ids if cid in relevant), len(cited_ids))
        if relevant
        else math.nan,
        "citation_in_context": _ratio(sum(1 for cid in cited_ids if cid in context_set), len(cited_ids)),
        "keyword_recall": keyword_recall(response.answer, case.keywords),
        "clean_format": float(not echoes_source_header(response.answer)),
        "latency_ms": round(latency_ms, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_overflow": bool(usage.get("prompt_overflow", False)),
        "fallback_reason": getattr(trace, "fallback_reason", None),
        "verification_passed": getattr(trace, "faithfulness_passed", None),
        "query_scope": getattr(trace, "query_scope", None),
        "policy_block_tokens": (getattr(trace, "retrieval_stages", {}) or {}).get("policy_block_tokens"),
    }


QUALITY_METRICS: tuple[str, ...] = (
    "context_hit",
    "context_coverage",
    "keyword_recall",
    "tagged_citation",
    "citation_precision",
    "citation_in_context",
    "abstention_correct",
    "clean_format",
    "faithfulness",
)


def _metric(row: dict[str, Any], name: str) -> float:
    value = row.get(name)
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(r["latency_ms"]) for r in rows if r.get("latency_ms") is not None]
    prompt_tokens = [float(r["prompt_tokens"]) for r in rows if r.get("prompt_tokens") is not None]
    summary: dict[str, Any] = {"n": len(rows)}
    for name in QUALITY_METRICS:
        summary[name] = mean([_metric(r, name) for r in rows])
    summary["latency_p50_ms"] = percentile(latencies, 0.5)
    summary["latency_p95_ms"] = percentile(latencies, 0.95)
    summary["prompt_tokens_p50"] = percentile(prompt_tokens, 0.5)
    summary["prompt_tokens_max"] = max(prompt_tokens) if prompt_tokens else math.nan
    summary["prompt_overflow_rate"] = mean([float(bool(r.get("prompt_overflow"))) for r in rows])
    return summary


def compare(baseline: Sequence[dict[str, Any]], candidate: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Paired comparison by (corpus, case_id)."""
    base = {(r["corpus"], r["case_id"]): r for r in baseline}
    cand = {(r["corpus"], r["case_id"]): r for r in candidate}
    keys = sorted(base.keys() & cand.keys())
    out: dict[str, Any] = {"n": len(keys)}
    for name in (*QUALITY_METRICS, "latency_ms"):
        b = [_metric(base[k], name) for k in keys]
        c = [_metric(cand[k], name) for k in keys]
        stats = paired_bootstrap_ci(b, c)
        stats["p"] = sign_flip_p_value(b, c)
        # A latency "win" is the candidate being faster.
        stats.update(win_loss_tie(c, b) if name == "latency_ms" else win_loss_tie(b, c))
        out[name] = stats
    return out


def write_spot_check(rows: Sequence[dict[str, Any]], path: str | Path, limit: int = 50) -> Path:
    """CSV for human review: question, answer, cited ids, blank judgment columns."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["corpus", "case_id", "question", "answer", "cited_ids", "context_ids",
             "human_correct(0/1)", "human_faithful(0/1)", "notes"]
        )
        for row in list(rows)[:limit]:
            writer.writerow(
                [row["corpus"], row["case_id"], row["question"], row["answer"],
                 " ".join(row.get("cited_ids") or []), " ".join(row.get("context_ids") or []), "", "", ""]
            )
    return target


def format_markdown(title: str, groups: dict[str, dict[str, Any]], comparisons: dict[str, dict[str, Any]]) -> str:
    def fmt(value: Any, digits: int = 3) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "n/a"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines = [f"# {title}", ""]
    header = ["Group", "n", *QUALITY_METRICS, "latency p50 ms", "latency p95 ms", "prompt tok p50", "overflow"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for group, s in groups.items():
        lines.append(
            "| " + " | ".join(
                [group, str(s["n"]), *(fmt(s[m]) for m in QUALITY_METRICS),
                 fmt(s["latency_p50_ms"], 0), fmt(s["latency_p95_ms"], 0),
                 fmt(s["prompt_tokens_p50"], 0), fmt(s["prompt_overflow_rate"])]
            ) + " |"
        )
    for label, comparison in comparisons.items():
        lines += ["", f"## {label} (n={comparison['n']})", "",
                  "| Metric | Delta | 95% CI | p | W/L/T |", "|---|---|---|---|---|"]
        for metric in (*QUALITY_METRICS, "latency_ms"):
            stats = comparison[metric]
            lines.append(
                f"| {metric} | {fmt(stats['delta'])} | [{fmt(stats['ci_low'])}, {fmt(stats['ci_high'])}] | "
                f"{fmt(stats['p'])} | {stats['wins']}/{stats['losses']}/{stats['ties']} |"
            )
    return "\n".join(lines) + "\n"
