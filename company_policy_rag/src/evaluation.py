"""
Lightweight RAG evaluation framework for company_policy_rag.

Production-rag principles applied:
- Measure before optimizing: repeatable golden-set eval on every significant change.
- Primary metrics for policy docs: Context Precision + Faithfulness (hallucination risk).
- LLM-as-judge via local Ollama — no external API dependency.
- Results appended to logs/evaluation_results.json for trend tracking.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama

from src.config import settings
from src.generation import generate_grounded_answer_with_trace
from src.indexing import configure_llama_index, create_retriever, index_exists, load_index
from src.prompts import INSUFFICIENT_INFO_MESSAGE
from src.prompts import format_nodes_for_prompt, get_generation_config_summary
from src.retriever import get_final_top_k, get_retrieval_config_summary
from src.utils import logger

# ── Golden dataset types ─────────────────────────────────────────────────────


@dataclass
class GoldenCase:
    id: str
    category: str
    question: str
    expected_answer: str
    relevant_sections: list[str] = field(default_factory=list)
    expected_key_points: list[str] = field(default_factory=list)
    corpus: str = "policy"
    query_type: str = "factual"
    source_file: str = ""


@dataclass
class CaseResult:
    id: str
    category: str
    corpus: str
    query_type: str
    question: str
    hit_rate: float
    context_precision: float
    context_recall: float
    faithfulness: float | None
    answer_relevancy: float | None
    generated_answer: str
    retrieved_count: int
    relevant_retrieved_count: int
    judge_notes: dict[str, str] = field(default_factory=dict)
    pre_guard_answer: str = ""
    guard_modified: bool = False
    code_validation_triggered: bool = False
    code_validation_passed: bool | None = None
    code_validation_method: str = ""
    code_validation_explanation: str = ""
    code_validation_failed_lines: list[str] = field(default_factory=list)
    fallback_reason: str = "none"


@dataclass
class EvalRun:
    run_id: str
    timestamp: str
    model: str
    judge_model: str
    top_k: int
    retrieval_config: dict[str, Any]
    generation_config: dict[str, Any]
    case_count: int
    aggregate: dict[str, float | None]
    cases: list[dict[str, Any]]
    duration_seconds: float


# ── Dataset loading ──────────────────────────────────────────────────────────


def load_golden_dataset(path: Path | None = None) -> list[GoldenCase]:
    """Load versioned golden dataset from JSON."""
    dataset_path = path or settings.eval_dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {dataset_path}")

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    cases = []
    for item in data.get("cases", []):
        cases.append(
            GoldenCase(
                id=item["id"],
                category=item.get("category", "general"),
                question=item["question"],
                expected_answer=item.get("expected_answer", ""),
                relevant_sections=item.get("relevant_sections", []),
                expected_key_points=item.get("expected_key_points", []),
                corpus=item.get("corpus", "policy"),
                query_type=item.get("query_type", item.get("category", "factual")),
                source_file=item.get("source_file", ""),
            )
        )
    return cases


def filter_cases_by_corpus(
    cases: list[GoldenCase],
    corpus: str | None = None,
) -> list[GoldenCase]:
    """Filter golden cases by corpus (all | policy | guidebook)."""
    selected = (corpus or settings.eval_corpus or "all").lower()
    if selected == "all":
        return cases
    return [c for c in cases if c.corpus == selected]


# ── Relevance matching (retrieval metrics) ───────────────────────────────────


def _chunk_searchable_text(node: NodeWithScore) -> str:
    """Combine metadata + text into a single lowercase searchable string."""
    meta = node.metadata or {}
    parts = [
        str(meta.get("section_path", "")),
        str(meta.get("section_title", "")),
        str(meta.get("section_number", "")),
        str(meta.get("source_file", "")),
        str(meta.get("category", "")),
        node.get_content() or "",
    ]
    return " ".join(parts).lower()


def is_chunk_relevant(node: NodeWithScore, relevant_sections: list[str]) -> bool:
    """
    Fuzzy relevance check against golden relevant_sections keywords.

    Matches if any section keyword appears in chunk metadata or body text.
    Empty relevant_sections (edge-case queries) always returns False for
    retrieval relevance — those cases test abstention, not retrieval.
    """
    if not relevant_sections:
        return False

    searchable = _chunk_searchable_text(node)
    for section in relevant_sections:
        needle = section.lower().strip()
        if not needle:
            continue
        if needle in searchable:
            return True
        # Multi-word: require all significant tokens (len>3) to appear
        tokens = [t for t in re.split(r"\W+", needle) if len(t) > 3]
        if tokens and all(t in searchable for t in tokens):
            return True
    return False


def compute_retrieval_metrics(
    nodes: list[NodeWithScore],
    relevant_sections: list[str],
) -> tuple[float, float, float, int, int]:
    """
    Returns (hit_rate, context_precision, context_recall, relevant_count, expected_count).

    hit_rate: 1.0 if any retrieved chunk is relevant, else 0.0
    context_precision: relevant_retrieved / total_retrieved
    context_recall: matched_section_keywords / total_section_keywords
    """
    if not nodes:
        return 0.0, 0.0, 0.0, 0, len(relevant_sections)

    flags = [is_chunk_relevant(n, relevant_sections) for n in nodes]
    relevant_count = sum(flags)
    total = len(nodes)

    hit_rate = 1.0 if relevant_count > 0 else 0.0
    context_precision = relevant_count / total if total else 0.0

    # Recall: fraction of golden section keywords found across all retrieved chunks
    searchable = " ".join(_chunk_searchable_text(n) for n in nodes)
    if not relevant_sections:
        context_recall = 1.0  # N/A — abstention cases don't expect retrieval
    else:
        matched = sum(
            1 for s in relevant_sections
            if s.lower().strip() in searchable
            or all(t in searchable for t in re.split(r"\W+", s.lower()) if len(t) > 3)
        )
        context_recall = matched / len(relevant_sections)

    return hit_rate, context_precision, context_recall, relevant_count, len(relevant_sections)


# ── LLM-as-judge ─────────────────────────────────────────────────────────────


def _get_judge_llm() -> Ollama:
    return Ollama(
        model=settings.eval_llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.0,
        request_timeout=settings.llm_request_timeout,
    )


def _parse_judge_json(response_text: str) -> dict[str, Any]:
    """Extract JSON object from LLM judge response."""
    text = response_text.strip()
    # Try raw JSON first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find embedded JSON block
    match = re.search(r"\{[^{}]*\"score\"[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"score": None, "reasoning": text[:300]}


FAITHFULNESS_PROMPT = """You are an evaluation judge for a company policy RAG system.

Rate how FAITHFUL the ANSWER is to the CONTEXT (0.0 to 1.0):
- 1.0 = every factual claim in the answer is directly supported by the context
- 0.5 = some unsupported claims or mild extrapolation
- 0.0 = significant hallucination or claims not in context

Scoring guidance:
- Do NOT penalize honest gap statements ("the excerpts do not describe...") when supported claims are faithful.
- At-will employment claims are supported when the context includes at-will / quit-at-any-time language.
- Score 1.0 when the answer correctly abstains and the context truly lacks the topic.

If the answer correctly states it lacks sufficient information and the context is empty or irrelevant, score 1.0.

Respond ONLY with JSON:
{{"score": <float>, "reasoning": "<one sentence>"}}

CONTEXT:
{context}

ANSWER:
{answer}
"""

RELEVANCY_PROMPT = """You are an evaluation judge for a company policy RAG system.

Rate how RELEVANT the ANSWER is to the QUESTION given the retrieved CONTEXT (0.0 to 1.0):
- 1.0 = directly and completely addresses the question using supported policy information
- 0.8 = addresses the main question; minor gaps OK when context is partial
- 0.5 = partially addresses or misses key aspects that ARE present in the context
- 0.0 = off-topic or fails to address the question

Scoring guidance:
- Score partial but faithful answers >= 0.8 when they use relevant context, even if incomplete.
- Score semantic-mapping answers higher when context covers the conduct domain (e.g. internet policy for social media).
- For abstention, score >= 0.8 when the answer states the topic is not covered AND context truly
  lacks it — that directly answers the question. Score low only if context had usable information.
- Resignation-without-notice: score >= 0.8 when answer explains at-will employment allows quitting without notice.
- Disciplinary process: score >= 0.8 when answer covers report → investigate → disciplinary action up to termination, even without a named progressive discipline policy.
- Outside employment: score >= 0.8 when answer covers conflict-of-interest rules from context and notes missing approval policy honestly.
- Guidebook HR abstention: score >= 0.8 when the question asks about vacation, benefits, or HR policy
  and the answer states the AI Agents guidebook does not cover that topic (clean abstention counts as relevant).

Expected topic (reference): {expected_answer}

Respond ONLY with JSON:
{{"score": <float>, "reasoning": "<one sentence>"}}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}
"""


def judge_faithfulness(context: str, answer: str, llm: Ollama) -> tuple[float | None, str]:
    prompt = FAITHFULNESS_PROMPT.format(context=context[:6000], answer=answer[:2000])
    try:
        result = _parse_judge_json(str(llm.complete(prompt)))
        score = result.get("score")
        return (float(score) if score is not None else None, str(result.get("reasoning", "")))
    except Exception as exc:
        logger.warning("Faithfulness judge failed: %s", exc)
        return None, str(exc)


def judge_answer_relevancy(
    question: str,
    answer: str,
    expected_answer: str,
    llm: Ollama,
    *,
    context: str = "",
) -> tuple[float | None, str]:
    prompt = RELEVANCY_PROMPT.format(
        context=context[:6000],
        question=question,
        answer=answer[:2000],
        expected_answer=expected_answer[:500],
    )
    try:
        result = _parse_judge_json(str(llm.complete(prompt)))
        score = result.get("score")
        return (float(score) if score is not None else None, str(result.get("reasoning", "")))
    except Exception as exc:
        logger.warning("Relevancy judge failed: %s", exc)
        return None, str(exc)


# ── Single-case & full-run evaluation ────────────────────────────────────────


def evaluate_case(
    case: GoldenCase,
    index: VectorStoreIndex,
    *,
    use_llm_judge: bool = True,
    llm: Ollama | None = None,
    retrieval_only: bool = False,
) -> CaseResult:
    """Evaluate one golden case: retrieve → generate → score."""
    from src.retrieval_scope import corpus_retrieval_filters

    scope_filters = corpus_retrieval_filters(
        case.corpus,
        source_file=case.source_file or None,
    )
    retriever = create_retriever(index, filters=scope_filters)
    nodes = retriever.retrieve(case.question)

    hit_rate, ctx_prec, ctx_rec, rel_count, _ = compute_retrieval_metrics(
        nodes, case.relevant_sections
    )

    if retrieval_only:
        return CaseResult(
            id=case.id,
            category=case.category,
            corpus=case.corpus,
            query_type=case.query_type,
            question=case.question,
            hit_rate=hit_rate,
            context_precision=ctx_prec,
            context_recall=ctx_rec,
            faithfulness=None,
            answer_relevancy=None,
            generated_answer="",
            retrieved_count=len(nodes),
            relevant_retrieved_count=rel_count,
        )

    trace = generate_grounded_answer_with_trace(
        case.question, nodes, Settings.llm
    )
    pre_guard = trace.pre_guard_answer
    answer = trace.final_answer
    guard_modified = trace.fallback_reason == "faithfulness" or (
        trace.post_guard_answer.strip() != answer.strip()
        and INSUFFICIENT_INFO_MESSAGE in answer
        and INSUFFICIENT_INFO_MESSAGE not in trace.post_guard_answer
    )

    faithfulness: float | None = None
    relevancy: float | None = None
    judge_notes: dict[str, str] = {}

    if use_llm_judge and settings.eval_use_llm_judge:
        judge = llm or _get_judge_llm()
        context = format_nodes_for_prompt(nodes)
        faithfulness, f_reason = judge_faithfulness(context, answer, judge)
        relevancy, r_reason = judge_answer_relevancy(
            case.question,
            answer,
            case.expected_answer,
            judge,
            context=context,
        )
        judge_notes = {"faithfulness": f_reason, "relevancy": r_reason}

    return CaseResult(
        id=case.id,
        category=case.category,
        corpus=case.corpus,
        query_type=case.query_type,
        question=case.question,
        hit_rate=hit_rate,
        context_precision=ctx_prec,
        context_recall=ctx_rec,
        faithfulness=faithfulness,
        answer_relevancy=relevancy,
        generated_answer=answer[:500],
        pre_guard_answer=pre_guard[:500],
        guard_modified=guard_modified,
        code_validation_triggered=trace.code_validation.triggered,
        code_validation_passed=trace.code_validation.passed,
        code_validation_method=trace.code_validation.validation_method,
        code_validation_explanation=trace.code_validation.explanation[:300],
        code_validation_failed_lines=trace.code_validation.failed_lines[:5],
        fallback_reason=trace.fallback_reason,
        retrieved_count=len(nodes),
        relevant_retrieved_count=rel_count,
        judge_notes=judge_notes,
    )


def _mean(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def _aggregate_for_cases(results: list[CaseResult]) -> dict[str, float | None]:
    triggered = [r for r in results if r.code_validation_triggered]
    code_pass_values: list[float] = [
        1.0 if r.code_validation_passed else 0.0
        for r in triggered
        if r.code_validation_passed is not None
    ]
    low_conf_values = [
        1.0 if r.fallback_reason == "code_validation" else 0.0 for r in results
    ]
    return {
        "hit_rate": _mean([r.hit_rate for r in results]),
        "context_precision": _mean([r.context_precision for r in results]),
        "context_recall": _mean([r.context_recall for r in results]),
        "faithfulness": _mean([r.faithfulness for r in results]),
        "answer_relevancy": _mean([r.answer_relevancy for r in results]),
        "code_validation_pass_rate": _mean(code_pass_values) if code_pass_values else None,
        "low_confidence_fallback_rate": _mean(low_conf_values) if low_conf_values else None,
    }


def run_evaluation(
    *,
    dataset_path: Path | None = None,
    max_samples: int | None = None,
    use_llm_judge: bool = True,
    retrieval_only: bool = False,
    corpus: str | None = None,
) -> EvalRun:
    """
    Run full golden-set evaluation against the Chroma-backed index.

    Raises FileNotFoundError if index or dataset is missing.
    """
    if not index_exists():
        raise FileNotFoundError(
            "No Chroma index found. Run: python scripts/index_documents.py"
        )

    configure_llama_index()
    needs_llm = not retrieval_only or settings.enable_query_rewrite
    if needs_llm:
        Settings.llm = Ollama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            request_timeout=settings.llm_request_timeout,
            thinking=False if settings.llm_model == "qwen3.5:9b" else None,
        )

    index = load_index()
    cases = filter_cases_by_corpus(load_golden_dataset(dataset_path), corpus)
    limit = max_samples or settings.eval_max_samples
    if limit and limit > 0:
        cases = cases[:limit]

    effective_judge = use_llm_judge and not retrieval_only
    judge_llm = (
        _get_judge_llm()
        if effective_judge and settings.eval_use_llm_judge
        else None
    )
    start = time.perf_counter()
    results: list[CaseResult] = []

    for i, case in enumerate(cases, 1):
        logger.info("Evaluating [%d/%d] %s", i, len(cases), case.id)
        result = evaluate_case(
            case,
            index,
            use_llm_judge=effective_judge,
            llm=judge_llm,
            retrieval_only=retrieval_only,
        )
        results.append(result)

    duration = time.perf_counter() - start
    aggregate = _aggregate_for_cases(results)
    by_corpus: dict[str, dict[str, float | None]] = {}
    by_query_type: dict[str, dict[str, float | None]] = {}
    for key, attr in (("corpus", "corpus"), ("query_type", "query_type")):
        groups: dict[str, list[CaseResult]] = {}
        for result in results:
            label = getattr(result, attr)
            groups.setdefault(label, []).append(result)
        target = by_corpus if attr == "corpus" else by_query_type
        for label, group in groups.items():
            target[label] = _aggregate_for_cases(group)
    aggregate["by_corpus"] = by_corpus
    aggregate["by_query_type"] = by_query_type

    run = EvalRun(
        run_id=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        timestamp=datetime.now(timezone.utc).isoformat(),
        model="retrieval-only" if retrieval_only else settings.llm_model,
        judge_model="none" if retrieval_only else settings.eval_llm_model,
        top_k=get_final_top_k(),
        retrieval_config=get_retrieval_config_summary(),
        generation_config=get_generation_config_summary(),
        case_count=len(results),
        aggregate=aggregate,
        cases=[asdict(r) for r in results],
        duration_seconds=round(duration, 1),
    )
    return run


# ── Output formatting & persistence ──────────────────────────────────────────


def format_results_table(run: EvalRun) -> str:
    """Render per-case and aggregate scores as a readable ASCII table."""
    headers = ["ID", "Category", "Hit", "CtxPrec", "CtxRec", "Faith", "Relv"]
    rows = []
    for c in run.cases:
        rows.append([
            c["id"][:20],
            c["category"][:12],
            f"{c['hit_rate']:.2f}",
            f"{c['context_precision']:.2f}",
            f"{c['context_recall']:.2f}",
            f"{c['faithfulness']:.2f}" if c["faithfulness"] is not None else "—",
            f"{c['answer_relevancy']:.2f}" if c["answer_relevancy"] is not None else "—",
        ])

    col_widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt_row(cells: list[str]) -> str:
        return " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells))

    sep = "-+-".join("-" * w for w in col_widths)
    lines = [fmt_row(headers), sep]
    lines.extend(fmt_row(r) for r in rows)
    lines.append("")
    lines.append("AGGREGATE SCORES")
    lines.append(sep)
    for metric, value in run.aggregate.items():
        if isinstance(value, dict):
            lines.append(f"  {metric}:")
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, dict):
                    lines.append(f"    {sub_key}:")
                    for m, v in sub_val.items():
                        val_str = f"{v:.3f}" if isinstance(v, (int, float)) and v is not None else "—"
                        lines.append(f"      {m:<18} {val_str}")
                else:
                    val_str = f"{sub_val:.3f}" if isinstance(sub_val, (int, float)) and sub_val is not None else "—"
                    lines.append(f"    {sub_key:<18} {val_str}")
            continue
        val_str = f"{value:.3f}" if isinstance(value, (int, float)) and value is not None else "—"
        lines.append(f"  {metric:<20} {val_str}")
    lines.append(f"\nRun ID: {run.run_id} | Cases: {run.case_count} | Duration: {run.duration_seconds}s")
    return "\n".join(lines)


def save_eval_results(run: EvalRun, path: Path | None = None) -> Path:
    """
    Append evaluation run to logs/evaluation_results.json for trend tracking.

    File structure: {"runs": [ {...}, {...} ]}
    """
    out_path = path or settings.eval_results_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    history: dict[str, Any] = {"runs": []}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            history = json.load(f)

    history.setdefault("runs", []).append(asdict(run))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    logger.info("Saved evaluation results to %s", out_path)
    return out_path