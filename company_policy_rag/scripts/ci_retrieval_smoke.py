#!/usr/bin/env python3
"""CI gate: backend retrieval -> final prompt context on a small labelled corpus.

Runs ``RAGPipeline.run_retrieval_stages`` (the production retrieval, context
assembly and packing path; no LLM) with the default settings over the handbook
fixture and fails when context hit or coverage drops below the committed
baseline. Replaces the old smoke gate that exercised the legacy ``src/`` stack.

  python scripts/ci_retrieval_smoke.py
  python scripts/ci_retrieval_smoke.py --write-baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CORPUS = PROJECT_ROOT / "data" / "eval" / "retrieval" / "handbook_corpus.json"
LABELS = PROJECT_ROOT / "data" / "eval" / "retrieval" / "handbook_labels.json"
BASELINE = PROJECT_ROOT / "data" / "eval" / "retrieval" / "handbook_smoke_baseline.json"
CONTEXT_K = 6
# One lost query out of 21 is -0.048; the gate must catch that.
TOLERANCE = 0.03


def measure(workdir: Path) -> dict[str, float]:
    import eval_retrieval_backend as retrieval_harness

    from backend.evaluation.answer_eval import load_cases
    from src.config import settings

    for key, value in {
        "retrieval_cache_enabled": False,
        "semantic_cache_enabled": False,
        "vision_enabled": False,
        "enable_lazy_vision_fallback": False,
        "enable_llm_multi_query": False,
        "record_retrieval_stages": True,
    }.items():
        setattr(settings, key, value)

    env = retrieval_harness.build_environment(
        "handbook", str(CORPUS), workdir, retrieval_harness.RerankScoreCache(workdir / "scores.json")
    )
    env.hybrid.min_chunk_words = int(settings.min_chunk_words)

    hits: list[float] = []
    top2: list[float] = []
    reciprocal_ranks: list[float] = []
    coverages: list[float] = []
    for case in load_cases("handbook", LABELS):
        if not case.relevant_ids:
            continue
        ctx = env.pipeline.run_retrieval_stages(case.question)
        context = list(ctx.retrieval_stages.get("final_context") or [])[:CONTEXT_K]
        present = set(context)
        hits.append(float(bool(case.relevant_ids & present)))
        top2.append(float(bool(case.relevant_ids & set(context[:2]))))
        first = next((rank for rank, cid in enumerate(context, start=1) if cid in case.relevant_ids), None)
        reciprocal_ranks.append(1.0 / first if first else 0.0)
        if case.aspects:
            coverages.append(sum(1 for ids in case.aspects.values() if ids & present) / len(case.aspects))
        else:
            coverages.append(float(bool(case.relevant_ids & present)))
        if not case.relevant_ids & present:
            print(f"MISS {case.case_id}: context={context}", flush=True)
    return {
        "queries": len(hits),
        "context_hit@6": round(sum(hits) / len(hits), 4),
        # The fixture is small (7 chunks), so order inside the context is the
        # signal that catches ranking/assembly regressions.
        "context_hit@2": round(sum(top2) / len(top2), 4),
        "context_mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4),
        "context_coverage@6": round(sum(coverages) / len(coverages), 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--workdir", help="index directory (default: a temp dir)")
    args = parser.parse_args(argv)

    # Chroma keeps index files open on Windows; a failed temp cleanup is not a gate failure.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        workdir = Path(args.workdir) if args.workdir else Path(tmp)
        metrics = measure(workdir)
    print(json.dumps(metrics, indent=2))

    if args.write_baseline:
        BASELINE.write_text(json.dumps({**metrics, "tolerance": TOLERANCE}, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {BASELINE}")
        return 0

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    tolerance = float(baseline.get("tolerance", TOLERANCE))
    failures = [
        f"{name}: {metrics[name]:.3f} < baseline {baseline[name]:.3f} - {tolerance}"
        for name in ("context_hit@6", "context_hit@2", "context_mrr", "context_coverage@6")
        if metrics[name] < float(baseline[name]) - tolerance
    ]
    if failures:
        print("Retrieval smoke gate FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("Retrieval smoke gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
