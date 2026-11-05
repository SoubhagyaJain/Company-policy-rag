#!/usr/bin/env python3
"""Answer-level evaluation of the backend RAG pipeline against a live Ollama model.

Runs ``RAGPipeline.query`` (the same entry point the chat API uses, without
conversation state or caches) for every labelled question and scores the
answer, its citations, abstention, latency, and Ollama's token counts. See
``backend/evaluation/answer_eval.py`` for metric definitions.

  python scripts/eval_answers_backend.py run \
      --corpus handbook=data/eval/retrieval/handbook_corpus.json \
      --queries handbook=data/eval/retrieval/handbook_labels.json \
      --config defaults --out logs/answer_eval/current

  # Same questions with a setting changed, then a paired comparison:
  python scripts/eval_answers_backend.py run ... --config legacy_context \
      --set CONTEXT_ASSEMBLY_MODE=governing --out logs/answer_eval/current
  python scripts/eval_answers_backend.py summarize --results logs/answer_eval/current \
      --compare defaults:legacy_context --spot-check logs/answer_eval/current/spot_check.csv

Corpora are chunk snapshots (``corpus.json``) indexed with the production
embedding model into ``storage/retrieval_eval`` (shared with the retrieval
harness). Nothing is written to the live document library.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import platform
import subprocess
import urllib.request
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.evaluation.answer_eval import (  # noqa: E402
    compare,
    format_markdown,
    load_cases,
    score_response,
    summarize,
    write_spot_check,
)
from src.config import Settings, settings  # noqa: E402

_FORCED = {
    "retrieval_cache_enabled": False,
    "semantic_cache_enabled": False,
    "vision_enabled": False,
    "enable_lazy_vision_fallback": False,
    "record_retrieval_stages": True,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _field_for_alias(key: str) -> str:
    for name, info in Settings.model_fields.items():
        if name == key.lower() or (info.alias or "").upper() == key.upper():
            return name
    raise SystemExit(f"Unknown setting: {key}")


def _coerce(name: str, raw: str) -> Any:
    current = getattr(settings, name)
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _build_llm(model: str):
    from llama_index.llms.ollama import Ollama

    return Ollama(
        base_url=settings.ollama_base_url,
        model=model,
        temperature=settings.llm_temperature,
        request_timeout=settings.llm_request_timeout,
        context_window=settings.llm_context_window,
        keep_alive=-1,
    )


def cmd_run(args: argparse.Namespace) -> int:
    import eval_retrieval_backend as retrieval_harness

    from backend.rag.pipeline import RAGPipeline
    from backend.rag.verifier import SelfReflectionVerifier
    from backend.retrieval.reranker import CrossEncoderReranker

    from backend.rag.response_modes import RESPONSE_MODES

    if args.context_chunks:
        mode = RESPONSE_MODES[args.response_mode]
        RESPONSE_MODES[args.response_mode] = replace(mode, retrieval_top_k=max(15, mode.retrieval_top_k), rerank_top_k=args.context_chunks)
    overrides = dict(_FORCED)
    for item in args.set or []:
        key, _, raw = item.partition("=")
        name = _field_for_alias(key.strip())
        overrides[name] = _coerce(name, raw.strip())
    for name, value in overrides.items():
        setattr(settings, name, value)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done = {(r["config"], r["corpus"], r["case_id"]) for r in _read_jsonl(results_path)}

    metadata = {"python": platform.python_version(), "platform": platform.platform(), "model": args.model or settings.llm_model, "context_chunks_limit": args.context_chunks, "response_mode": args.response_mode, "forced_settings": overrides, "note": "run each configuration in a fresh process; warmup is excluded"}
    try:
        with urllib.request.urlopen(settings.ollama_base_url.rstrip("/") + "/api/ps", timeout=3) as response:
            metadata["ollama_ps_before"] = json.load(response)
    except Exception as exc:
        metadata["ollama_ps_error"] = str(exc)
    try:
        metadata["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        metadata["nvidia_smi"] = None
    (out_dir / (args.config + "_environment.json")).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    llm = _build_llm(args.model or settings.llm_model)
    judge = _build_llm(args.judge_model) if args.judge_model else None
    verifier = SelfReflectionVerifier()
    score_cache = retrieval_harness.RerankScoreCache(Path(args.score_cache))

    corpora = dict(item.split("=", 1) for item in args.corpus)
    query_files = dict(item.split("=", 1) for item in args.queries)
    for corpus, corpus_path in corpora.items():
        cases = load_cases(corpus, query_files[corpus])
        if args.limit:
            cases = cases[: args.limit]
        env = retrieval_harness.build_environment(corpus, corpus_path, Path(args.workdir), score_cache)
        env.hybrid.min_chunk_words = int(settings.min_chunk_words)
        pipeline = RAGPipeline(
            hybrid_retriever=env.hybrid,
            reranker=CrossEncoderReranker(
                model_name=settings.reranker_model,
                top_n=settings.reranker_top_n,
                device=settings.reranker_device,
                min_ratio=settings.rerank_min_score_ratio,
                pool_size=settings.reranker_pool_size,
                score_filter_enabled=settings.enable_rerank_score_filter,
                min_keep=settings.rerank_min_keep,
            ),
            docstore=env.pipeline.docstore,
            llm=llm,
            semantic_cache=None,
        )
        started = time.perf_counter()
        if args.warmup and cases:
            pipeline.query(cases[0].question, response_mode=args.response_mode)
        for case in cases:
            if (args.config, corpus, case.case_id) in done and not args.force:
                continue
            t0 = time.perf_counter()
            first_text = []
            def on_text(delta: str) -> None:
                if delta and not first_text:
                    first_text.append((time.perf_counter() - t0) * 1000)
            response = pipeline.query(case.question, response_mode=args.response_mode,
                                      stream_callback=on_text if args.stream else None)
            row = score_response(case, response, latency_ms=(time.perf_counter() - t0) * 1000)
            row["config"] = args.config
            row["settings"] = {k: overrides[k] for k in sorted(overrides) if k not in _FORCED}
            row["first_callback_ms"] = round(first_text[0], 2) if first_text else None
            row["stream_requested"] = args.stream
            row["context_limit"] = args.context_chunks
            row["final_context_count"] = len(response.context_chunks)
            row["context_tokens_estimated"] = response.trace.context_tokens
            row["stage_timings_ms"] = response.trace.stage_timings_ms
            row["stage_status"] = response.trace.stage_status
            row["llm_usage"] = response.trace.llm_usage
            try:
                with urllib.request.urlopen(settings.ollama_base_url.rstrip("/") + "/api/ps", timeout=3) as ps_response:
                    row["ollama_ps_after"] = json.load(ps_response)
            except Exception:
                row["ollama_ps_after"] = None
            if judge is not None and not row["abstained"] and response.context_chunks:
                verdict = verifier._evaluate_faithfulness_llm(response.answer, response.context_chunks, judge)
                row["faithfulness"] = verdict[0] if verdict else None
                row["unsupported_claims"] = verdict[1] if verdict else []
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_json_safe(row)) + "\n")
            print(
                f"{args.config}/{corpus}/{case.case_id}: kw={row['keyword_recall']} "
                f"cite_prec={row['citation_precision']} abstain_ok={row['abstention_correct']} "
                f"{row['latency_ms']:.0f} ms",
                flush=True,
            )
        print(f"{args.config}/{corpus}: {len(cases)} cases in {time.perf_counter() - started:.1f}s", flush=True)
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    from backend.evaluation.answer_eval import echoes_source_header, is_abstention

    rows = _read_jsonl(Path(args.results) / "results.jsonl")
    for row in rows:
        # Re-derive text-only judgments so detector fixes apply to stored runs.
        row["abstained"] = is_abstention(row.get("answer", ""))
        row["abstention_correct"] = float(row["abstained"] == bool(row.get("should_abstain")))
        row["clean_format"] = float(not echoes_source_header(row.get("answer", "")))
        for key, value in list(row.items()):
            if value is None and key in {"context_hit", "context_coverage", "tagged_citation",
                                         "citation_precision", "citation_in_context", "keyword_recall",
                                         "faithfulness"}:
                row[key] = math.nan
    configs = list(dict.fromkeys(r["config"] for r in rows))
    groups: dict[str, dict[str, Any]] = {}
    for config in configs:
        config_rows = [r for r in rows if r["config"] == config]
        corpora = list(dict.fromkeys(r["corpus"] for r in config_rows))
        for corpus in corpora:
            groups[f"{config} / {corpus}"] = summarize([r for r in config_rows if r["corpus"] == corpus])
        if len(corpora) > 1:
            groups[f"{config} / ALL"] = summarize(config_rows)
    comparisons: dict[str, dict[str, Any]] = {}
    for spec in (args.compare or "").split(","):
        if not spec.strip():
            continue
        candidate, _, baseline = spec.partition(":")
        comparisons[f"{candidate} vs {baseline}"] = compare(
            [r for r in rows if r["config"] == baseline],
            [r for r in rows if r["config"] == candidate],
        )
    out = Path(args.results)
    (out / "summary.json").write_text(
        json.dumps(_json_safe({"groups": groups, "comparisons": comparisons}), indent=2), encoding="utf-8"
    )
    markdown = format_markdown(f"Answer evaluation: {out.name}", groups, comparisons)
    (out / "summary.md").write_text(markdown, encoding="utf-8")
    if args.spot_check:
        first = configs[0] if configs else None
        write_spot_check([r for r in rows if r["config"] == first], args.spot_check, limit=args.spot_check_limit)
    print(markdown)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--corpus", action="append", required=True, help="name=path/to/corpus.json")
    run.add_argument("--queries", action="append", required=True, help="name=path/to/labels.json")
    run.add_argument("--config", required=True, help="label stored on every result row")
    run.add_argument("--set", action="append", help="SETTING=value override, e.g. CONTEXT_ASSEMBLY_MODE=governing")
    run.add_argument("--out", required=True)
    run.add_argument("--model", help="Ollama model (default: OLLAMA_LLM_MODEL)")
    run.add_argument("--judge-model", help="Ollama model for the faithfulness judge (off when omitted)")
    run.add_argument("--response-mode", default="standard")
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--force", action="store_true")
    run.add_argument("--context-chunks", type=int, choices=[3, 5, 10], help="maximum final evidence chunks; other standard-mode budgets stay fixed")
    run.add_argument("--warmup", action="store_true", help="run one unrecorded query before measurement")
    run.add_argument("--stream", action="store_true", help="measure time to first pipeline callback (not network delivery)")
    run.add_argument("--workdir", default=str(PROJECT_ROOT / "storage" / "retrieval_eval"))
    run.add_argument("--score-cache", default=str(PROJECT_ROOT / "storage" / "retrieval_eval" / "rerank_scores.json"))
    run.set_defaults(func=cmd_run)

    summary = sub.add_parser("summarize")
    summary.add_argument("--results", required=True)
    summary.add_argument("--compare", help="candidate:baseline[,candidate:baseline]")
    summary.add_argument("--spot-check", help="write a human review CSV for the first config")
    summary.add_argument("--spot-check-limit", type=int, default=50)
    summary.set_defaults(func=cmd_summarize)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
