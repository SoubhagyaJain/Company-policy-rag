#!/usr/bin/env python3
"""Reproducible before/after benchmark for multi-turn RAG conversation logic.

The benchmark holds the corpus, BM25 retriever, top-k, answer model, and judge
constant. Only the conversation layer changes:

* baseline: the legacy QueryRewriter fallback that appends the latest user turn
* improved: the production ConversationInterpreter and its retrieval policy

The default run needs no model and is suitable for CI. Pass ``--with-generation``
to measure answer citations, end-to-end latency, and an LLM-judged unsupported-
claim rate with the configured local Ollama model.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.chunk import Chunk, ChunkMetadata  # noqa: E402
from backend.models.conversation import ConversationRAGState, ConversationTurn  # noqa: E402
from backend.models.rag import Citation, ScoredChunk  # noqa: E402
from backend.rag.conversation_interpreter import (  # noqa: E402
    ConversationInterpreter,
    RetrievalDecision,
)
from backend.rag.query_rewrite import QueryRewriter  # noqa: E402
from backend.retrieval.bm25 import BM25SearchIndex  # noqa: E402

SECTION_IDS = {
    "1": "parental_maternity_leave",
    "2": "business_travel_expenses",
    "3": "remote_work",
    "4": "annual_leave_probation",
    "5": "vpn_access_recovery",
    "6": "information_security_data_handling",
}
SOURCE_TAG_RE = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    interpolated = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(interpolated, 3)


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "samples": len(values),
        "mean_ms": round(statistics.fmean(values), 3) if values else None,
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_document(path: Path) -> tuple[list[Chunk], dict[str, Chunk]]:
    """Turn the demo Markdown's H2 sections into production Chunk models."""
    text = path.read_text(encoding="utf-8")
    matches = list(re.finditer(r"^##\s+(\d+)\.\s+(.+?)\s*$", text, flags=re.MULTILINE))
    chunks: list[Chunk] = []
    by_section: dict[str, Chunk] = {}
    for index, match in enumerate(matches):
        section_number, title = match.groups()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        section_id = SECTION_IDS.get(section_number)
        if not section_id:
            raise ValueError(f"No stable section ID configured for section {section_number}")
        chunk = Chunk(
            id=f"demo_{section_id}",
            text=body,
            metadata=ChunkMetadata(
                document_id="northstar-handbook-2026",
                source_file=path.name,
                file_path=str(path),
                document_type="markdown",
                category="demo_employee_policy",
                policy_id="NSL-HR-2026",
                effective_date="2026-01-01",
                chunk_index=index,
                section_number=section_number,
                section_title=title,
                section_path=f"{section_number}. {title}",
                extra={"benchmark_section_id": section_id, "fictional_demo": True},
            ),
            token_count=len(body.split()),
        )
        chunks.append(chunk)
        by_section[section_id] = chunk
    if not chunks:
        raise ValueError(f"No numbered H2 sections found in {path}")
    return chunks, by_section


def load_dataset(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "1.0" or not isinstance(data.get("cases"), list):
        raise ValueError("Conversation benchmark must use schema_version 1.0 and contain cases")
    case_ids = [case.get("id") for case in data["cases"]]
    if None in case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("Every benchmark case needs a unique non-empty id")
    return data


def _section_id(chunk: Chunk) -> str:
    return str((chunk.metadata.extra or {}).get("benchmark_section_id", ""))


def _scored(chunk: Chunk, rank: int = 1, score: float = 1.0) -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=score, sparse_score=score, rank=rank)


def _citation(chunk: Chunk, index: int = 1) -> Citation:
    return Citation(
        source_index=index,
        chunk_id=chunk.id,
        document_id=chunk.metadata.document_id,
        source_file=chunk.metadata.source_file,
        document_name="Northstar Labs Employee Handbook",
        section_title=chunk.metadata.section_title,
        section_path=chunk.metadata.section_path,
        snippet=chunk.text[:400],
        relevance_score=1.0,
    )


def build_state(case: dict[str, Any], sections: dict[str, Chunk]) -> ConversationRAGState:
    turns: list[ConversationTurn] = []
    for index, item in enumerate(case.get("history", []), start=1):
        section_id = item.get("evidence_section")
        evidence = [_scored(sections[section_id])] if section_id else []
        citations = [_citation(sections[section_id])] if section_id else []
        turns.append(
            ConversationTurn(
                turn_id=f"{case['id']}_turn_{index}",
                user_query=item["user_query"],
                resolved_query=item.get("resolved_query", item["user_query"]),
                active_topic=item.get("active_topic"),
                active_entities=item.get("active_entities", []),
                evidence_status="DIRECT" if evidence else "MISSING",
                retrieved_chunks=evidence,
                citations=citations,
                answer=item.get("answer", ""),
            )
        )
    last = turns[-1] if turns else None
    return ConversationRAGState(
        conversation_id=f"benchmark_{case['id']}",
        last_user_query=last.user_query if last else None,
        last_resolved_query=last.resolved_query if last else None,
        active_topic=last.active_topic if last else None,
        active_entities=list(last.active_entities) if last else [],
        previous_evidence_status=last.evidence_status if last else None,
        previous_retrieved_chunks=list(last.retrieved_chunks) if last else [],
        previous_citations=list(last.citations) if last else [],
        last_answer=last.answer if last else None,
        turns=turns,
    )


def _dedupe_ranked(groups: list[list[ScoredChunk]], top_k: int) -> list[ScoredChunk]:
    best: dict[str, ScoredChunk] = {}
    first_seen: dict[str, int] = {}
    cursor = 0
    for group in groups:
        for item in group:
            cursor += 1
            first_seen.setdefault(item.chunk.id, cursor)
            current = best.get(item.chunk.id)
            if current is None or item.score > current.score:
                best[item.chunk.id] = item
    ordered = sorted(best.values(), key=lambda item: (-item.score, first_seen[item.chunk.id]))[:top_k]
    for rank, item in enumerate(ordered, start=1):
        item.rank = rank
    return ordered


def run_baseline(
    message: str,
    state: ConversationRAGState,
    index: BM25SearchIndex,
    top_k: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    rewritten = QueryRewriter(enable_llm_rewrite=False).rewrite(
        message,
        history=state.to_history_messages(max_turns=4),
    ).rewritten_query
    results = index.search(rewritten, top_k=top_k)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "action": RetrievalDecision.RETRIEVE.value,
        "query": rewritten,
        "results": results,
        "logic_retrieval_ms": elapsed_ms,
    }


def run_improved(
    message: str,
    state: ConversationRAGState,
    index: BM25SearchIndex,
    top_k: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    interpreter = ConversationInterpreter(llm=None)
    interpretation = interpreter.interpret(message, state)
    action = interpretation.retrieval_decision
    if action == RetrievalDecision.REUSE_PREVIOUS:
        turn = interpreter.get_trusted_turn(state, interpretation.reuse_turn_id)
        results = list(turn.retrieved_chunks[:top_k]) if turn else []
    elif action == RetrievalDecision.ASK_CLARIFICATION:
        results = []
    elif action == RetrievalDecision.NO_RETRIEVAL:
        results = []
    elif action == RetrievalDecision.DECOMPOSE:
        groups = [index.search(query, top_k=top_k) for query in interpretation.sub_queries]
        results = _dedupe_ranked(groups, top_k)
    else:
        results = index.search(interpretation.standalone_query, top_k=top_k)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "action": action.value,
        "query": interpretation.standalone_query,
        "sub_queries": interpretation.sub_queries,
        "clarification_question": interpretation.clarification_question,
        "results": results,
        "logic_retrieval_ms": elapsed_ms,
    }


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float = 180.0) -> None:
        self.url = f"{base_url.rstrip('/')}/api/generate"
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, *, num_predict: int = 180) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json" if prompt.startswith("You are a strict grounding judge") else None,
                "options": {"temperature": 0, "seed": 42, "num_predict": num_predict},
                "keep_alive": "10m",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                answer = str(body.get("response", "")).strip()
                if not answer:
                    raise RuntimeError("Ollama returned an empty response")
                return answer
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc
                if attempt == 0:
                    time.sleep(0.5)
        raise RuntimeError(f"Ollama generation failed after retry: {last_error}")


def _source_context(results: list[ScoredChunk]) -> str:
    if not results:
        return "NO VERIFIED SOURCES"
    blocks = []
    for index, item in enumerate(results, start=1):
        meta = item.chunk.metadata
        blocks.append(f"[Source {index}] {meta.section_path}\n{item.chunk.text}")
    return "\n\n".join(blocks)


def generate_answer(client: OllamaClient, question: str, results: list[ScoredChunk]) -> tuple[str, float]:
    prompt = (
        "Answer the user's employee-policy question using only the verified sources below. "
        "Every factual sentence must end with the matching [Source N] tag. If the sources do not answer "
        "the question, say that the available sources do not contain the answer. Use 2-4 concise sentences.\n\n"
        f"Question: {question}\n\nVerified sources:\n{_source_context(results)}\n\nAnswer:"
    )
    started = time.perf_counter()
    answer = client.generate(prompt)
    return answer, (time.perf_counter() - started) * 1000


def judge_answers(
    client: OllamaClient,
    baseline_answer: str,
    baseline_results: list[ScoredChunk],
    improved_answer: str,
    improved_results: list[ScoredChunk],
) -> dict[str, Any]:
    prompt = (
        "You are a strict grounding judge. Evaluate each answer only against its own sources. Split each answer "
        "into atomic factual claims. A claim is unsupported when its own sources do not entail it. Ignore citation "
        "markers and non-factual uncertainty statements. Do not penalize a source-supported but irrelevant answer; "
        "retrieval accuracy measures relevance separately. Return JSON only with this exact shape: "
        '{"baseline":{"claim_count":0,"unsupported_claim_count":0,"unsupported_claims":[]},'
        '"improved":{"claim_count":0,"unsupported_claim_count":0,"unsupported_claims":[]}}.\n\n'
        f"BASELINE SOURCES:\n{_source_context(baseline_results)}\n"
        f"BASELINE ANSWER:\n{baseline_answer}\n\n"
        f"IMPROVED SOURCES:\n{_source_context(improved_results)}\n"
        f"IMPROVED ANSWER:\n{improved_answer}"
    )
    raw = client.generate(prompt, num_predict=260)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise RuntimeError(f"Grounding judge returned invalid JSON: {raw[:200]}") from exc
        parsed = json.loads(match.group(0))
    normalized: dict[str, Any] = {}
    for system in ("baseline", "improved"):
        item = parsed.get(system, {}) if isinstance(parsed, dict) else {}
        claim_count = max(0, int(item.get("claim_count", 0)))
        unsupported = max(0, min(claim_count, int(item.get("unsupported_claim_count", 0))))
        normalized[system] = {
            "claim_count": claim_count,
            "unsupported_claim_count": unsupported,
            "unsupported_claims": [str(value) for value in item.get("unsupported_claims", [])][:10],
        }
    return normalized


def _score_case(case: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    expected_sections = set(case.get("expected_sections", []))
    sections = [_section_id(item.chunk) for item in run["results"]]
    first_relevant_rank = next(
        (rank for rank, section in enumerate(sections, start=1) if section in expected_sections),
        None,
    )
    accepted_actions = set(case.get("accepted_actions", [case["expected_action"]]))
    lowered_query = run["query"].lower()
    query_terms = [str(term).lower() for term in case.get("expected_query_terms", [])]
    return {
        "action_correct": run["action"] in accepted_actions,
        "retrieval_hit": first_relevant_rank is not None if expected_sections else None,
        "reciprocal_rank": round(1 / first_relevant_rank, 4) if first_relevant_rank else (0.0 if expected_sections else None),
        "query_term_coverage": (
            round(sum(term in lowered_query for term in query_terms) / len(query_terms), 4)
            if query_terms
            else None
        ),
        "retrieved_sections": sections,
    }


def _score_citations(answer: str, results: list[ScoredChunk], expected_sections: set[str]) -> dict[str, Any]:
    tags = [int(value) for value in SOURCE_TAG_RE.findall(answer)]
    cited_sections: list[str | None] = []
    for tag in tags:
        cited_sections.append(_section_id(results[tag - 1].chunk) if 0 < tag <= len(results) else None)
    correct = sum(section in expected_sections for section in cited_sections)
    return {
        "citation_count": len(tags),
        "correct_citation_count": correct,
        "citation_correct": bool(tags) and correct == len(tags),
        "cited_sections": cited_sections,
    }


def _aggregate(case_rows: list[dict[str, Any]], system: str, with_generation: bool) -> dict[str, Any]:
    rows = [row[system] for row in case_rows]
    retrieval_rows = [row for row in rows if row["metrics"]["retrieval_hit"] is not None]
    term_rows = [row for row in rows if row["metrics"]["query_term_coverage"] is not None]
    summary: dict[str, Any] = {
        "case_count": len(rows),
        "retrieval_cases": len(retrieval_rows),
        "retrieval_hit_at_3": round(
            sum(bool(row["metrics"]["retrieval_hit"]) for row in retrieval_rows) / len(retrieval_rows), 4
        ),
        "mean_reciprocal_rank": round(
            statistics.fmean(float(row["metrics"]["reciprocal_rank"]) for row in retrieval_rows), 4
        ),
        "retrieval_policy_accuracy": round(
            sum(bool(row["metrics"]["action_correct"]) for row in rows) / len(rows), 4
        ),
        "standalone_query_term_coverage": round(
            statistics.fmean(float(row["metrics"]["query_term_coverage"]) for row in term_rows), 4
        ),
        "conversation_logic_and_retrieval_latency": _latency_summary(
            [float(row["logic_retrieval_ms"]) for row in rows]
        ),
        "citation_correctness": None,
        "citation_answer_pass_rate": None,
        "hallucination_rate": None,
        "answer_generation_latency": None,
        "end_to_end_latency": None,
    }
    if with_generation:
        answer_rows = [row for row in rows if row.get("answer") is not None]
        emitted = sum(int(row["citations"]["citation_count"]) for row in answer_rows)
        correct = sum(int(row["citations"]["correct_citation_count"]) for row in answer_rows)
        claims = sum(int(row["judge"]["claim_count"]) for row in answer_rows)
        unsupported = sum(int(row["judge"]["unsupported_claim_count"]) for row in answer_rows)
        generation_values = [float(row["generation_ms"]) for row in answer_rows]
        end_to_end_values = [
            float(row["logic_retrieval_ms"]) + float(row["generation_ms"]) for row in answer_rows
        ]
        summary.update(
            {
                "citation_correctness": round(correct / emitted, 4) if emitted else 0.0,
                "citation_answer_pass_rate": round(
                    sum(bool(row["citations"]["citation_correct"]) for row in answer_rows) / len(answer_rows), 4
                ),
                "hallucination_rate": round(unsupported / claims, 4) if claims else 0.0,
                "answer_generation_latency": _latency_summary(generation_values),
                "end_to_end_latency": _latency_summary(end_to_end_values),
                "factual_claim_count": claims,
                "unsupported_claim_count": unsupported,
            }
        )
    return summary


def benchmark(
    dataset_path: Path,
    document_path: Path,
    *,
    top_k: int = 3,
    with_generation: bool = False,
    ollama_url: str = "http://localhost:11434",
    model: str = "qwen2.5:7b",
    hardware_note: str | None = None,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    chunks, sections = load_document(document_path)
    index = BM25SearchIndex(storage_dir="unused-benchmark-index")
    index.build_index(chunks)
    client = OllamaClient(ollama_url, model) if with_generation else None
    case_rows: list[dict[str, Any]] = []

    # Exclude model loading from measured requests. Alternate answer order per
    # case so one system does not receive a systematic warm-cache advantage.
    if client:
        client.generate("Reply with the single word ready.", num_predict=3)

    for position, case in enumerate(dataset["cases"], start=1):
        state = build_state(case, sections)
        runs = {
            "baseline": run_baseline(case["user_message"], state, index, top_k),
            "improved": run_improved(case["user_message"], state, index, top_k),
        }
        row: dict[str, Any] = {
            "id": case["id"],
            "behavior": case["behavior"],
            "user_message": case["user_message"],
            "expected_action": case["expected_action"],
            "expected_sections": case.get("expected_sections", []),
        }
        for system, run in runs.items():
            row[system] = {
                "action": run["action"],
                "query": run["query"],
                "sub_queries": run.get("sub_queries", []),
                "clarification_question": run.get("clarification_question"),
                "logic_retrieval_ms": round(float(run["logic_retrieval_ms"]), 3),
                "metrics": _score_case(case, run),
            }

        should_answer = case["expected_action"] != RetrievalDecision.ASK_CLARIFICATION.value
        if client and should_answer:
            print(f"[{position}/{len(dataset['cases'])}] Generating {case['id']}...", flush=True)
            answer_order = ("improved", "baseline") if position % 2 else ("baseline", "improved")
            for system in answer_order:
                run = runs[system]
                answer, elapsed_ms = generate_answer(client, run["query"], run["results"])
                row[system]["answer"] = answer
                row[system]["generation_ms"] = round(elapsed_ms, 3)
                row[system]["citations"] = _score_citations(
                    answer,
                    run["results"],
                    set(case.get("expected_sections", [])),
                )
            judged = judge_answers(
                client,
                row["baseline"]["answer"],
                runs["baseline"]["results"],
                row["improved"]["answer"],
                runs["improved"]["results"],
            )
            row["baseline"]["judge"] = judged["baseline"]
            row["improved"]["judge"] = judged["improved"]
        elif client:
            for system in ("baseline", "improved"):
                row[system]["answer"] = None
                row[system]["generation_ms"] = None
                row[system]["citations"] = None
                row[system]["judge"] = None
        case_rows.append(row)

    return {
        "schema_version": "1.0",
        "benchmark": dataset["name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "dataset": _display_path(dataset_path),
            "document": _display_path(document_path),
            "corpus_sections": len(chunks),
            "top_k": top_k,
            "generation_enabled": with_generation,
            "answer_and_judge_model": model if with_generation else None,
            "temperature": 0,
            "seed": 42,
            "baseline": "legacy QueryRewriter fallback with latest user turn",
            "improved": "production ConversationInterpreter deterministic policy path",
            "retriever": "production BM25SearchIndex (identical for both systems)",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unavailable",
            "hardware_note": hardware_note or "not specified",
        },
        "systems": {
            "baseline": _aggregate(case_rows, "baseline", with_generation),
            "improved": _aggregate(case_rows, "improved", with_generation),
        },
        "cases": case_rows,
        "limitations": [
            "The corpus and cases are fictional and intentionally small; these scores do not predict every production corpus.",
            "Hallucination rate is an unsupported-claim proxy judged by the same local model family, not a human adjudication.",
            "Latency is hardware-dependent and represents one local run; logic/retrieval timings are more stable than generation timings.",
        ],
    }


def _metric(value: Any, *, percent: bool = True) -> str:
    if value is None:
        return "Not measured"
    return f"{float(value) * 100:.1f}%" if percent else f"{float(value):.2f} ms"


def render_report(result: dict[str, Any]) -> str:
    baseline = result["systems"]["baseline"]
    improved = result["systems"]["improved"]
    generated = result["configuration"]["generation_enabled"]
    rows = [
        ("Retrieval hit@3", baseline["retrieval_hit_at_3"], improved["retrieval_hit_at_3"], True),
        ("Mean reciprocal rank", baseline["mean_reciprocal_rank"], improved["mean_reciprocal_rank"], True),
        ("Retrieval-policy accuracy", baseline["retrieval_policy_accuracy"], improved["retrieval_policy_accuracy"], True),
        ("Standalone-query term coverage", baseline["standalone_query_term_coverage"], improved["standalone_query_term_coverage"], True),
        ("Citation correctness", baseline["citation_correctness"], improved["citation_correctness"], True),
        ("Answers with only correct citations", baseline["citation_answer_pass_rate"], improved["citation_answer_pass_rate"], True),
        ("Unsupported-claim rate", baseline["hallucination_rate"], improved["hallucination_rate"], True),
        (
            "Logic + retrieval p50",
            baseline["conversation_logic_and_retrieval_latency"]["p50_ms"],
            improved["conversation_logic_and_retrieval_latency"]["p50_ms"],
            False,
        ),
        (
            "End-to-end p50",
            baseline["end_to_end_latency"]["p50_ms"] if generated else None,
            improved["end_to_end_latency"]["p50_ms"] if generated else None,
            False,
        ),
    ]
    table = ["| Metric | Before | After |", "|---|---:|---:|"]
    for label, before, after, percent in rows:
        table.append(f"| {label} | {_metric(before, percent=percent)} | {_metric(after, percent=percent)} |")

    case_table = [
        "| Case | Behavior | Before action / hit | After action / hit |",
        "|---|---|---|---|",
    ]
    for case in result["cases"]:
        before_hit = case["baseline"]["metrics"]["retrieval_hit"]
        after_hit = case["improved"]["metrics"]["retrieval_hit"]
        before_label = "n/a" if before_hit is None else ("hit" if before_hit else "miss")
        after_label = "n/a" if after_hit is None else ("hit" if after_hit else "miss")
        case_table.append(
            f"| `{case['id']}` | {case['behavior']} | {case['baseline']['action']} / {before_label} | "
            f"{case['improved']['action']} / {after_label} |"
        )

    mode_note = (
        f"Answers and grounding judgments were generated locally with `{result['configuration']['answer_and_judge_model']}` "
        "at temperature 0 and seed 42."
        if generated
        else "This CI-safe run measures conversation policy, query quality, retrieval accuracy, and retrieval latency only. Run with `--with-generation` for citation, end-to-end latency, and unsupported-claim measurements."
    )
    return "\n".join(
        [
            "# Conversation RAG benchmark",
            "",
            f"Generated: `{result['generated_at']}`",
            "",
            "## Method",
            "",
            "This benchmark changes only the conversation layer. The before and after systems share the same fictional handbook, production BM25 implementation, top-3 retrieval, answer prompt, and answer model. The before path uses the legacy query rewriter; the after path uses the production `ConversationInterpreter` and its retrieval policy.",
            "",
            mode_note,
            f"Run environment: `{result['environment']['platform']}`, Python `{result['environment']['python']}`; {result['environment']['hardware_note']}.",
            "",
            "## Results",
            "",
            *table,
            "",
            "Retrieval accuracy is hit@3 over cases that require evidence. Citation correctness is the share of emitted citation tags that resolve to an expected source section. Unsupported-claim rate is the LLM-judged share of factual claims that the supplied sources do not entail. Latency values are from this machine and one run.",
            "",
            "## Case-level evidence",
            "",
            *case_table,
            "",
            "## Reproduce",
            "",
            "```bash",
            "python scripts/benchmark_conversation.py --with-generation",
            "```",
            "",
            "The machine-readable output includes rewritten queries, retrieved section IDs, generated answers, citation mappings, judge counts, and per-case timings in `data/eval/conversation_benchmark_results.json`.",
            "",
            "## Limits",
            "",
            *[f"- {item}" for item in result["limitations"]],
            "",
        ]
    )


def assert_minimums(result: dict[str, Any]) -> None:
    baseline = result["systems"]["baseline"]
    improved = result["systems"]["improved"]
    failures: list[str] = []
    if improved["retrieval_hit_at_3"] < 0.90:
        failures.append("improved retrieval hit@3 is below 90%")
    if improved["retrieval_policy_accuracy"] < 0.90:
        failures.append("improved retrieval-policy accuracy is below 90%")
    if improved["retrieval_hit_at_3"] < baseline["retrieval_hit_at_3"]:
        failures.append("improved retrieval hit@3 regressed below baseline")
    if improved["standalone_query_term_coverage"] < baseline["standalone_query_term_coverage"]:
        failures.append("improved standalone-query term coverage regressed below baseline")
    if failures:
        raise SystemExit("Benchmark gate failed: " + "; ".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/eval/conversation_benchmark.json",
    )
    parser.add_argument(
        "--document",
        type=Path,
        default=PROJECT_ROOT / "data/demo/sample_employee_handbook.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/eval/conversation_benchmark_results.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "docs/BENCHMARK_RESULTS.md",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--with-generation", action="store_true")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument(
        "--hardware-note",
        default=None,
        help="Human-readable CPU/GPU note recorded with the benchmark result.",
    )
    parser.add_argument("--assert-minimums", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = benchmark(
        args.dataset.resolve(),
        args.document.resolve(),
        top_k=max(1, args.top_k),
        with_generation=args.with_generation,
        ollama_url=args.ollama_url,
        model=args.model,
        hardware_note=args.hardware_note,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.report.write_text(render_report(result), encoding="utf-8")
    if args.assert_minimums:
        assert_minimums(result)
    baseline = result["systems"]["baseline"]
    improved = result["systems"]["improved"]
    print(
        "Conversation benchmark complete: "
        f"hit@3 {baseline['retrieval_hit_at_3']:.1%} -> {improved['retrieval_hit_at_3']:.1%}; "
        f"policy {baseline['retrieval_policy_accuracy']:.1%} -> {improved['retrieval_policy_accuracy']:.1%}"
    )
    print(f"Results: {args.output}")
    print(f"Report:  {args.report}")


if __name__ == "__main__":
    main()
