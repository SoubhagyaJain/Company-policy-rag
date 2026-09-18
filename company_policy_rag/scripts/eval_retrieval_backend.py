#!/usr/bin/env python3
"""Backend-native retrieval evaluation and reranker ablation.

Runs ``RAGPipeline.run_retrieval_stages`` -- the production retrieval path
(routing, multi-query, dense + BM25 + RRF, reranking, governing-clause
selection, context packing) -- once per configuration over a labelled query
set, records the ranked chunk ids at every stage, and scores them.

Subcommands
  run               execute configurations, append rankings to results.jsonl
  pool              export blind judgment pools from run results
  import-judgments  turn pool judgments into a graded labels file
  bm25-grid         component-level BM25 k1/b/stemming/field sweep
  summarize         metrics per stage, reranker outcomes, latency, paired CIs

Typical flow
  python scripts/eval_retrieval_backend.py run --corpus guidebook=<corpus.json> \
      --queries guidebook=data/eval/retrieval/guidebook_labels.json --suite headline --out logs/retrieval_eval/main
  python scripts/eval_retrieval_backend.py summarize --results logs/retrieval_eval/main \
      --labels guidebook=data/eval/retrieval/guidebook_labels.json

Every run disables the retrieval cache, semantic cache, vision fallback, and
LLM multi-query so results are deterministic and uncached.
Cross-encoder scores are cached on disk (they depend only on query, chunk text,
and model); reranker latency is therefore reported from measured uncached
per-pair cost, never from cache hits.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Load sentence-transformers' native libraries on the main thread first:
# importing them for the first time inside a worker thread crashes on Windows.
import sentence_transformers  # noqa: E402,F401

from backend.embeddings.embeddings import EmbeddingService  # noqa: E402
from backend.embeddings.vector_store import ChromaVectorStore  # noqa: E402
from backend.evaluation.retrieval_eval import (  # noqa: E402
    DOWNSTREAM_STAGES,
    HIT_CUTOFF,
    NDCG_K,
    RECALL_KS,
    STAGES,
    QueryLabels,
    evidence_flow,
    first_relevant_rank,
    load_labels,
    mean,
    paired_bootstrap_ci,
    percentile,
    ranking_metrics,
    rerank_outcome,
    sign_flip_p_value,
    win_loss_tie,
)
from backend.models.chunk import Chunk  # noqa: E402
from backend.rag.pipeline import RAGPipeline  # noqa: E402
from backend.retrieval.bm25 import DEFAULT_METADATA_FIELDS, BM25SearchIndex  # noqa: E402
from backend.retrieval.hybrid import HybridRetriever  # noqa: E402
from backend.retrieval.reranker import CrossEncoderReranker  # noqa: E402
from backend.retrieval.vector import DenseVectorRetriever  # noqa: E402
from src.config import settings  # noqa: E402

RERANKER_MODELS = {"base": "BAAI/bge-reranker-base", "large": "BAAI/bge-reranker-large"}
LEGACY_BM25_FIELDS = ",".join(DEFAULT_METADATA_FIELDS)
SHORT_CHUNK_WORDS = 5


# ── Configurations ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    reranker: str | None = None  # None | "base" | "large"
    score_filter: bool = False
    ratio: float | None = None  # None: per-category router ratio
    min_keep: int = 1
    pool: int = 0  # 0: legacy max(top_n*4, 20); -1: every candidate
    depth: int = 0  # 0: response-mode depth
    depth_mode: str = "response_mode"
    rrf_k: int = 60
    merge: str = "max_score"
    min_chunk_words: int = 0
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_stemming: bool = False
    bm25_fields: str = LEGACY_BM25_FIELDS
    scope_mode: str = "strict"  # SCOPE_UNBOUND_REFERENCE_MODE
    context_mode: str = "governing"  # CONTEXT_ASSEMBLY_MODE
    anchor_k: int = 2  # CONTEXT_RANK_ANCHOR_K

    def first_stage_key(self) -> tuple:
        """Everything except the reranker settings: identifies the no-reranker twin."""
        return (
            self.scope_mode,
            self.context_mode,
            self.anchor_k,
            self.depth,
            self.depth_mode,
            self.rrf_k,
            self.merge,
            self.min_chunk_words,
            self.bm25_k1,
            self.bm25_b,
            self.bm25_stemming,
            self.bm25_fields,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalConfig":
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})


CURRENT = RetrievalConfig(name="current")


def _load_tuned(path: str | None) -> RetrievalConfig:
    if not path:
        raise SystemExit("--tuned-config is required for suites that use the tuned first stage")
    return RetrievalConfig.from_dict({"name": "tuned", **json.loads(Path(path).read_text(encoding="utf-8"))})


def suite_configs(suite: str, tuned_path: str | None) -> list[RetrievalConfig]:
    cur = CURRENT
    if suite == "headline":
        return [
            replace(cur, name="A_current_norerank"),
            replace(cur, name="B_current_base_nofilter", reranker="base"),
            replace(cur, name="C_current_base_filter", reranker="base", score_filter=True, min_keep=3),
            replace(cur, name="C0_current_base_filter_legacy_keep1", reranker="base", score_filter=True, min_keep=1),
            replace(cur, name="D_current_large_nofilter", reranker="large"),
        ]
    if suite == "depth":
        out = []
        for depth in (4, 8, 15, 30):
            for rr in (None, "base", "large"):
                out.append(replace(cur, name=f"depth{depth}_{rr or 'norerank'}", depth=depth, reranker=rr))
        return out
    if suite == "pool":
        return [
            replace(cur, name=f"depth30_{rr}_pool{'all' if pool < 0 else pool}", depth=30, reranker=rr, pool=pool)
            for rr in ("base", "large")
            for pool in (20, 40, -1)
        ]
    if suite == "filter":
        out = []
        for rr in ("base", "large"):
            out.append(replace(cur, name=f"{rr}_filter_off", reranker=rr))
            for ratio in (0.25, 0.45):
                out.append(
                    replace(cur, name=f"{rr}_filter{ratio}", reranker=rr, score_filter=True, ratio=ratio, min_keep=3)
                )
        return out
    if suite == "rrf_k":
        return [replace(cur, name=f"rrf_k{k}", rrf_k=k) for k in (10, 30, 60)]
    if suite == "first_stage_ablation":
        # One change at a time on top of the current first stage, no reranker.
        return [
            replace(cur, name="fs_current"),
            replace(cur, name="fs_depth_max", depth_mode="max"),
            replace(cur, name="fs_merge_rrf", merge="rrf"),
            replace(cur, name="fs_min_words6", min_chunk_words=SHORT_CHUNK_WORDS),
            replace(cur, name="fs_bm25_text_only", bm25_fields=""),
            replace(cur, name="fs_bm25_section_fields", bm25_fields="section_path,section_title,section_number"),
            replace(cur, name="fs_bm25_stem", bm25_stemming=True),
            replace(cur, name="fs_rrf_k10", rrf_k=10),
            replace(cur, name="fs_rrf_k30", rrf_k=30),
        ]
    if suite == "tuned":
        tuned = _load_tuned(tuned_path)
        return [
            replace(tuned, name="E_tuned_norerank", reranker=None, score_filter=False),
            replace(tuned, name="F_tuned_base_nofilter", reranker="base", score_filter=False),
            replace(tuned, name="F_tuned_large_nofilter", reranker="large", score_filter=False),
            replace(tuned, name="F_tuned_base_filter", reranker="base", score_filter=True, min_keep=3),
            replace(tuned, name="F_tuned_large_filter", reranker="large", score_filter=True, min_keep=3),
        ]
    if suite == "downstream":
        # Tuned first stage (skip chunks <= 5 words) with legacy vs fixed
        # downstream stages, plus one-fix-at-a-time and anchor-depth checks.
        tuned = replace(cur, min_chunk_words=SHORT_CHUNK_WORDS)
        fixed = replace(tuned, scope_mode="resolve", context_mode="rank_anchor", anchor_k=2)
        out = []
        for label, rr in (("A", None), ("B", "base"), ("C", "large")):
            model = rr or "norerank"
            out.append(replace(tuned, name=f"{label}_legacy_{model}", reranker=rr))
            out.append(replace(fixed, name=f"{label}_fixed_{model}", reranker=rr))
            out.append(replace(tuned, name=f"{label}_scopefix_{model}", reranker=rr, scope_mode="resolve"))
            out.append(
                replace(tuned, name=f"{label}_contextfix_{model}", reranker=rr, context_mode="rank_anchor", anchor_k=2)
            )
            for k in (1, 3):
                out.append(replace(fixed, name=f"{label}_fixed_anchor{k}_{model}", reranker=rr, anchor_k=k))
        for label, rr in (("B", "base"), ("C", "large")):
            out.append(replace(fixed, name=f"{label}_fixed_{rr}_filter", reranker=rr, score_filter=True, min_keep=3))
        return out
    if suite == "tuned_sweep":
        tuned = _load_tuned(tuned_path)
        out = []
        for depth in (4, 8, 15, 30):
            for rr in (None, "base", "large"):
                out.append(replace(tuned, name=f"tuned_depth{depth}_{rr or 'norerank'}", depth=depth, reranker=rr))
        for rr in ("base", "large"):
            for pool in (20, 40, -1):
                out.append(
                    replace(tuned, name=f"tuned_depth30_{rr}_pool{'all' if pool < 0 else pool}", depth=30, reranker=rr, pool=pool)
                )
            for ratio in (0.25, 0.45):
                out.append(
                    replace(tuned, name=f"tuned_{rr}_filter{ratio}", reranker=rr, score_filter=True, ratio=ratio, min_keep=3)
                )
        return out
    raise SystemExit(f"unknown suite {suite!r}")


# ── Environment ────────────────────────────────────────────────────────────


def load_corpus(path: str | Path) -> list[Chunk]:
    return [Chunk(**item) for item in json.loads(Path(path).read_text(encoding="utf-8"))]


def _corpus_fingerprint(chunks: list[Chunk]) -> str:
    digest = hashlib.sha1()
    for chunk in chunks:
        digest.update(chunk.id.encode("utf-8"))
        digest.update((chunk.text or "").encode("utf-8"))
    return digest.hexdigest()[:12]


def _pair_key(query: str, text: str) -> str:
    return hashlib.sha1(f"{query}\x1f{text}".encode("utf-8")).hexdigest()


class RerankScoreCache:
    """Disk cache of cross-encoder scores plus measured uncached cost per pair."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.scores: dict[str, dict[str, float]] = {}
        self.cost: dict[str, dict[str, float]] = {}
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.scores = data.get("scores", {})
            self.cost = data.get("cost", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"scores": self.scores, "cost": self.cost}), encoding="utf-8")
        tmp.replace(self.path)

    def ms_per_pair(self, model: str, device: str) -> float:
        cost = (self.cost.get(model) or {}).get(device) or {}
        pairs = cost.get("pairs", 0)
        return cost.get("ms", 0.0) / pairs if pairs else math.nan

    def record_cost(self, model: str, device: str, pairs: int, ms: float) -> None:
        cost = self.cost.setdefault(model, {}).setdefault(device, {"pairs": 0, "ms": 0.0})
        cost["pairs"] += pairs
        cost["ms"] += ms

    def wrap(self, reranker: CrossEncoderReranker) -> None:
        reranker._init_model()
        model = reranker._model
        if model is None:
            raise SystemExit(f"reranker model {reranker.model_name} failed to load")
        if getattr(model, "_eval_score_cache", False):
            return
        name = reranker.model_name
        device = reranker.device
        store = self.scores.setdefault(name, {})
        raw_predict = model.predict
        # Warm the model so first-call overhead is not billed as per-pair cost.
        raw_predict([["warm up", "warm up passage"]] * 4, batch_size=4, show_progress_bar=False)

        def predict(pairs, **kwargs):
            keys = [_pair_key(q, t) for q, t in pairs]
            missing = [i for i, key in enumerate(keys) if key not in store]
            if missing:
                t0 = time.perf_counter()
                out = raw_predict([pairs[i] for i in missing], **kwargs)
                elapsed = (time.perf_counter() - t0) * 1000
                out = out.tolist() if hasattr(out, "tolist") else list(out)
                for i, score in zip(missing, out):
                    store[keys[i]] = float(score)
                self.record_cost(name, device, len(missing), elapsed)
            return [store[key] for key in keys]

        model.predict = predict
        model._eval_score_cache = True


@dataclass
class Environment:
    corpus: str
    chunks: list[Chunk]
    bm25: BM25SearchIndex
    hybrid: HybridRetriever
    pipeline: RAGPipeline
    rerankers: dict[str, CrossEncoderReranker]
    score_cache: RerankScoreCache
    score_device: str = "cpu"

    def reranker(self, key: str) -> CrossEncoderReranker:
        if key not in self.rerankers:
            rr = CrossEncoderReranker(
                model_name=RERANKER_MODELS[key],
                top_n=int(settings.reranker_top_n),
                device=self.score_device,
                min_ratio=float(settings.rerank_min_score_ratio),
            )
            self.score_cache.wrap(rr)
            self.rerankers[key] = rr
        return self.rerankers[key]


def build_environment(
    corpus: str, corpus_path: str, workdir: Path, score_cache: RerankScoreCache, score_device: str = "cpu"
) -> Environment:
    chunks = load_corpus(corpus_path)
    fingerprint = _corpus_fingerprint(chunks)
    index_dir = workdir / "indexes" / f"{corpus}_{fingerprint}"
    embedder = EmbeddingService(cache_enabled=False)  # uncached query embedding = honest latency

    store = ChromaVectorStore(collection_name="retrieval_eval", persist_dir=str(index_dir / "chroma"))
    if store.count() != len(chunks):
        store.clear()
        missing = [c for c in chunks if not c.embedding]
        if missing:
            vectors = embedder.embed_chunks([c.text or "" for c in missing])
            for chunk, vector in zip(missing, vectors):
                chunk.embedding = vector
        for start in range(0, len(chunks), 256):
            store.add_chunks(chunks[start : start + 256])
    else:
        # Vectors are already persisted; register the full chunk objects so
        # search returns complete metadata, exactly as a live session does.
        store.add_chunks([c.model_copy(update={"embedding": None}) for c in chunks])

    bm25 = BM25SearchIndex(storage_dir=str(index_dir / "bm25"))
    bm25.build_index(chunks)
    dense = DenseVectorRetriever(vector_store=store, embedding_service=embedder)
    hybrid = HybridRetriever(dense_retriever=dense, bm25_index=bm25, reranker=object())
    # A live session's pipeline holds its documents' chunks: scope resolution
    # reads indexed document names from the docstore.
    docstore = {c.id: c.model_copy(update={"embedding": None}) for c in chunks}
    pipeline = RAGPipeline(hybrid_retriever=hybrid, reranker=CrossEncoderReranker(), docstore=docstore, llm=None)
    return Environment(corpus, chunks, bm25, hybrid, pipeline, {}, score_cache, score_device)


_FORCED_SETTINGS = {
    "retrieval_cache_enabled": False,
    "semantic_cache_enabled": False,
    "vision_enabled": False,
    "enable_lazy_vision_fallback": False,
    "enable_llm_multi_query": False,
    "enable_query_rewrite": False,
    "record_retrieval_stages": True,
}


@contextlib.contextmanager
def configured(env: Environment, cfg: RetrievalConfig) -> Iterator[None]:
    overrides = dict(_FORCED_SETTINGS)
    overrides.update(
        {
            "enable_reranker": cfg.reranker is not None,
            "rerank_score_ratio_override": cfg.ratio,
            "retrieval_depth_override": cfg.depth,
            "retrieval_depth_mode": cfg.depth_mode,
            "hybrid_rrf_k": cfg.rrf_k,
            "subquery_merge_mode": cfg.merge,
            "scope_unbound_reference_mode": cfg.scope_mode,
            "context_assembly_mode": cfg.context_mode,
            "context_rank_anchor_k": cfg.anchor_k,
        }
    )
    saved = {key: getattr(settings, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(settings, key, value)
        env.bm25.reconfigure(
            k1=cfg.bm25_k1,
            b=cfg.bm25_b,
            stemming=cfg.bm25_stemming,
            metadata_fields=cfg.bm25_fields,
            skip_zero_scores=True,
        )
        env.hybrid.min_chunk_words = cfg.min_chunk_words
        if cfg.reranker is not None:
            rr = env.reranker(cfg.reranker)
            rr.score_filter_enabled = cfg.score_filter
            rr.pool_size = cfg.pool
            rr.min_keep = cfg.min_keep
            env.pipeline.reranker = rr
        yield
    finally:
        for key, value in saved.items():
            setattr(settings, key, value)


# ── run ────────────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_config(env: Environment, cfg: RetrievalConfig, queries: list[QueryLabels], response_mode: str) -> list[dict[str, Any]]:
    from backend.evaluation.retrieval_eval import stage_rankings

    rows = []
    with configured(env, cfg):
        for q in queries:
            t0 = time.perf_counter()
            ctx = env.pipeline.run_retrieval_stages(q.question, response_mode=response_mode)
            wall_ms = (time.perf_counter() - t0) * 1000
            stages = ctx.retrieval_stages
            if stages.get("retrieval_cache_hit"):
                raise SystemExit("retrieval cache hit during an experiment run")
            traces = stages.get("rerank") or []
            rows.append(
                {
                    "corpus": env.corpus,
                    "config": cfg.name,
                    "config_params": asdict(cfg),
                    "query_id": q.query_id,
                    "category": str(getattr(ctx.classification, "category", "")),
                    "fast_path": ctx.is_fast_path,
                    "strategy": stages.get("strategy", {}),
                    "sub_query_count": len(stages.get("sub_queries") or []),
                    "candidate_count": stages.get("candidate_count", 0),
                    "reranker_enabled": stages.get("reranker_enabled"),
                    "rankings": stage_rankings(stages),
                    "rerank_pools": [t.get("pool") or [] for t in traces],
                    "rerank_models": sorted({t.get("model", "") for t in traces}),
                    "context_tokens": stages.get("context_tokens", 0),
                    "context_assembly_mode": stages.get("context_assembly_mode"),
                    "timings": {
                        "wall_ms": round(wall_ms, 2),
                        "retrieval_total_ms": ctx.stage_timings.get("retrieval_total", 0.0),
                        "retrieval_ms": stages.get("retrieval_ms", 0.0),
                        "rerank_stage_ms": stages.get("rerank_ms", 0.0),
                        "rerank_model_ms_observed": round(sum(t.get("model_ms", 0.0) for t in traces), 2),
                        "rerank_pairs": sum(int(t.get("pool_size", 0)) for t in traces),
                    },
                }
            )
    return rows


def cmd_run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done = {(r["corpus"], r["config"]) for r in _read_jsonl(results_path)}
    score_cache = RerankScoreCache(Path(args.score_cache))
    workdir = Path(args.workdir)

    configs: list[RetrievalConfig] = []
    for suite in args.suite or []:
        configs.extend(suite_configs(suite, args.tuned_config))
    if args.configs:
        configs.extend(RetrievalConfig.from_dict(c) for c in json.loads(Path(args.configs).read_text(encoding="utf-8")))

    corpora = dict(item.split("=", 1) for item in args.corpus)
    query_files = dict(item.split("=", 1) for item in args.queries)
    for corpus, corpus_path in corpora.items():
        _, queries = load_labels(query_files[corpus])
        if args.limit:
            queries = queries[: args.limit]
        env = build_environment(corpus, corpus_path, workdir, score_cache, args.score_device)
        for cfg in configs:
            if (corpus, cfg.name) in done and not args.force:
                print(f"skip {corpus}/{cfg.name} (already in results)")
                continue
            t0 = time.perf_counter()
            rows = run_config(env, cfg, queries, args.response_mode)
            with results_path.open("a", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
            score_cache.save()
            print(f"{corpus}/{cfg.name}: {len(rows)} queries in {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


# ── pool / import-judgments ────────────────────────────────────────────────


def _word_count(text: str) -> int:
    return len((text or "").split())


def cmd_pool(args: argparse.Namespace) -> int:
    rows = [r for r in _read_jsonl(Path(args.results) / "results.jsonl") if r["corpus"] == args.corpus]
    chunks = {c.id: c for c in load_corpus(args.corpus_path)}
    _, queries = load_labels(args.queries)
    depth_by_stage = {}
    for spec in args.stages.split(","):
        name, _, depth = spec.partition(":")
        depth_by_stage[name] = int(depth or args.depth)

    pools = []
    lines = [f"# Judgment pool: {args.corpus}", ""]
    for q in queries:
        ids: set[str] = set(q.graded)  # keep any existing labels in the pool
        for row in rows:
            if row["query_id"] != q.query_id:
                continue
            for stage, depth in depth_by_stage.items():
                ids.update(row["rankings"].get(stage, [])[:depth])
        ordered = sorted(ids, key=lambda cid: hashlib.sha1(f"{q.query_id}{cid}".encode()).hexdigest())
        items = []
        lines += [f"## {q.query_id}", f"Q: {q.question}", ""]
        for n, cid in enumerate(ordered, start=1):
            chunk = chunks[cid]
            short = _word_count(chunk.text) <= SHORT_CHUNK_WORDS
            items.append({"n": n, "chunk_id": cid, "short": short})
            text = " ".join((chunk.text or "").split())
            if short:
                lines.append(f"[{n}] (<= {SHORT_CHUNK_WORDS} words, auto 0) {text}")
            else:
                title = chunk.metadata.section_title or ""
                lines.append(f"[{n}] ({title}) {text[: args.chars]}")
        lines.append("")
        pools.append({"query_id": q.query_id, "question": q.question, "items": items})
    Path(args.out_json).write_text(json.dumps({"corpus": args.corpus, "pools": pools}, indent=1), encoding="utf-8")
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    sizes = [len(p["items"]) for p in pools]
    print(f"pooled {sum(sizes)} chunks over {len(pools)} queries (mean {sum(sizes) / max(1, len(sizes)):.1f})")
    return 0


def _needle_hits(chunks: list[Chunk], needle: str) -> set[str]:
    n = needle.lower().strip()
    tokens = [t for t in re.split(r"\W+", n) if len(t) > 3]
    hits = set()
    for chunk in chunks:
        meta = chunk.metadata
        text = " ".join(
            str(x or "") for x in (chunk.text, meta.section_path, meta.section_title, meta.section_number, meta.source_file, meta.category)
        ).lower()
        if n in text or (tokens and all(t in text for t in tokens)):
            hits.add(chunk.id)
    return hits


def cmd_import_judgments(args: argparse.Namespace) -> int:
    pool = json.loads(Path(args.pool).read_text(encoding="utf-8"))
    judgments = json.loads(Path(args.judgments).read_text(encoding="utf-8"))
    queries_meta, queries = load_labels(args.queries)
    by_id = {q.query_id: q for q in queries}
    chunks = load_corpus(args.corpus_path)
    source = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    source_items = {item["id"]: item for item in source["queries"]}

    out_queries = []
    for entry in pool["pools"]:
        qid = entry["query_id"]
        grades = judgments.get(qid, {})
        by_n = {str(item["n"]): item for item in entry["items"]}
        relevant: dict[str, int] = {}
        for n, grade in grades.items():
            item = by_n[str(n)]
            if int(grade) > 0 and not item["short"]:
                relevant[item["chunk_id"]] = int(grade)
        src = source_items[qid]
        # A question written from a specific passage always counts that passage
        # as directly relevant, whether or not any system retrieved it.
        if src.get("source_chunk_id"):
            relevant[src["source_chunk_id"]] = max(relevant.get(src["source_chunk_id"], 0), 2)
        keywords = src.get("keywords") or []
        limit = max(3, int(0.10 * len(chunks)))
        aspects = {}
        for needle in keywords:
            hits = _needle_hits(chunks, needle)
            if 0 < len(hits) <= limit:
                aspects[needle] = sorted(hits)
        out_queries.append(
            {
                **{k: v for k, v in src.items() if k not in ("relevant", "aspects")},
                "relevant": relevant,
                "aspects": aspects,
                "judged": [item["chunk_id"] for item in entry["items"]],
            }
        )
        if qid not in by_id:
            raise SystemExit(f"judged query {qid} not in {args.queries}")
    meta = {k: v for k, v in source.items() if k != "queries"}
    meta["judgment_pool"] = {
        "stages": args.pool_note,
        "judged_chunks": sum(len(p["items"]) for p in pool["pools"]),
    }
    meta.update(queries_meta.get("judgment_meta", {}))
    Path(args.out).write_text(json.dumps({**meta, "queries": out_queries}, indent=1), encoding="utf-8")
    judged = sum(len(q["relevant"]) for q in out_queries)
    print(f"wrote {len(out_queries)} queries, {judged} relevant judgments -> {args.out}")
    return 0


def cmd_keyword_labels(args: argparse.Namespace) -> int:
    """Secondary label set: chunks matching the query's specific keywords (grade 1)."""
    source = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    chunks = load_corpus(args.corpus_path)
    limit = max(3, int(0.10 * len(chunks)))
    out = []
    for item in source["queries"]:
        relevant: dict[str, int] = {}
        aspects = {}
        for needle in item.get("keywords") or []:
            hits = _needle_hits(chunks, needle)
            if 0 < len(hits) <= limit:
                aspects[needle] = sorted(hits)
                relevant.update({cid: 1 for cid in hits})
        out.append({**item, "relevant": relevant, "aspects": aspects})
    meta = {k: v for k, v in source.items() if k != "queries"}
    meta["label_source"] = "keyword needles matching <=10% of the corpus (secondary, lexical-biased)"
    Path(args.out).write_text(json.dumps({**meta, "queries": out}, indent=1), encoding="utf-8")
    print(f"wrote keyword labels for {len(out)} queries -> {args.out}")
    return 0


# ── bm25-grid ──────────────────────────────────────────────────────────────


def cmd_bm25_grid(args: argparse.Namespace) -> int:
    rows = []
    for spec in args.labels:
        corpus, _, rest = spec.partition("=")
        labels_path, _, corpus_path = rest.partition("@")
        _, queries = load_labels(labels_path)
        queries = [q for q in queries if q.relevant]
        index = BM25SearchIndex(storage_dir=str(Path(args.workdir) / "bm25_grid"))
        index.build_index(load_corpus(corpus_path))
        for stemming in (False, True):
            for fields_value in (LEGACY_BM25_FIELDS, "section_path,section_title,section_number", ""):
                for k1 in (0.6, 0.9, 1.2, 1.5, 2.0):
                    for b in (0.3, 0.5, 0.75, 0.9):
                        index.reconfigure(k1=k1, b=b, stemming=stemming, metadata_fields=fields_value)
                        metrics = [
                            ranking_metrics([h.chunk.id for h in index.search(q.question, top_k=30)], q) for q in queries
                        ]
                        rows.append(
                            {
                                "corpus": corpus,
                                "stemming": stemming,
                                "fields": fields_value or "(text only)",
                                "k1": k1,
                                "b": b,
                                **{key: mean([m[key] for m in metrics]) for key in ("mrr", f"ndcg@{NDCG_K}", "recall@10", "recall@20")},
                            }
                        )
    Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    corpora = sorted({r["corpus"] for r in rows})
    by_setting: dict[tuple, list[dict]] = {}
    for r in rows:
        by_setting.setdefault((r["stemming"], r["fields"], r["k1"], r["b"]), []).append(r)
    ranked = sorted(
        by_setting.items(),
        key=lambda kv: -mean([r[f"ndcg@{NDCG_K}"] for r in kv[1]]),
    )
    print(f"{'stem':5} {'fields':42} {'k1':>4} {'b':>4} " + " ".join(f"{c[:10]:>22}" for c in corpora) + "   mean nDCG")
    for (stem, flds, k1, b), group in ranked[: args.top]:
        cells = []
        for c in corpora:
            r = next(x for x in group if x["corpus"] == c)
            cells.append(f"nDCG {r[f'ndcg@{NDCG_K}']:.3f} MRR {r['mrr']:.3f}")
        print(f"{str(stem):5} {flds[:42]:42} {k1:>4} {b:>4} " + " ".join(f"{x:>22}" for x in cells) + f"   {mean([g[f'ndcg@{NDCG_K}'] for g in group]):.3f}")
    legacy = by_setting[(False, LEGACY_BM25_FIELDS, 1.5, 0.75)]
    print("legacy (k1=1.5 b=0.75 all fields, no stem): " + ", ".join(f"{r['corpus']} nDCG {r[f'ndcg@{NDCG_K}']:.3f} MRR {r['mrr']:.3f}" for r in legacy))
    return 0


# ── latency ────────────────────────────────────────────────────────────────


def cmd_latency(args: argparse.Namespace) -> int:
    """Measure uncached cross-encoder cost per pair on a device over real candidate pools."""
    import torch

    rows = _read_jsonl(Path(args.results) / "results.jsonl")
    corpora = dict(item.split("=", 1) for item in args.corpus)
    texts = {corpus: {c.id: c.text or "" for c in load_corpus(path)} for corpus, path in corpora.items()}
    questions = {}
    for spec in args.queries:
        corpus, _, path = spec.partition("=")
        _, qs = load_labels(path)
        questions.update({(corpus, q.query_id): q.question for q in qs})
    samples = []
    seen = set()
    for row in rows:
        key = (row["corpus"], row["query_id"])
        if row["corpus"] not in texts or key in seen or key not in questions:
            continue
        seen.add(key)
        ids = row["rankings"]["candidates"][: args.pool]
        if len(ids) == args.pool:
            samples.append([[questions[key], texts[row["corpus"]][cid]] for cid in ids])
    random.Random(7).shuffle(samples)
    samples = samples[: args.samples]
    score_cache = RerankScoreCache(Path(args.score_cache))
    for key in args.models.split(","):
        rr = CrossEncoderReranker(model_name=RERANKER_MODELS[key], device=args.device)
        rr._init_model()
        model = rr._model
        model.predict([["warm up", "warm up passage"]] * 8, batch_size=8, show_progress_bar=False)
        total_pairs = 0
        total_ms = 0.0
        per_query = []
        for pairs in samples:
            t0 = time.perf_counter()
            with torch.inference_mode():
                model.predict(pairs, batch_size=min(len(pairs), 16), show_progress_bar=False)
            if args.device == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            total_pairs += len(pairs)
            total_ms += ms
            per_query.append(ms)
        score_cache.cost.setdefault(rr.model_name, {})[args.device] = {"pairs": total_pairs, "ms": total_ms}
        print(
            f"{rr.model_name} on {args.device}: {total_ms / total_pairs:.1f} ms/pair over {total_pairs} pairs "
            f"({len(samples)} queries, pool <= {args.pool}); per-query p50 {percentile(per_query, 0.5):.0f} ms"
        )
    score_cache.save()
    return 0


# ── summarize ──────────────────────────────────────────────────────────────


RANKING_STAGE = "post_rerank_prefilter"  # full ranking after the rerank stage
HANDOFF_STAGE = "post_filter"  # what the rerank stage passes on
CONTEXT_STAGE = "final_context"


def _fmt(value: float, digits: int = 3) -> str:
    return "  n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def _score_rows(rows: list[dict[str, Any]], labels: dict[str, dict[str, QueryLabels]]) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        q = labels.get(row["corpus"], {}).get(row["query_id"])
        if q is None or not q.relevant:
            continue
        rankings = row["rankings"]
        stage_metrics = {stage: ranking_metrics(rankings.get(stage, []), q) for stage in STAGES}
        stages_view = {
            "reranker_enabled": row["reranker_enabled"],
            "rerank": [{"pool": pool, "prefilter": []} for pool in row["rerank_pools"]],
        }
        scored.append(
            {
                **row,
                "labels": q,
                "metrics": stage_metrics,
                "flow": evidence_flow(rankings, q),
                "outcome": rerank_outcome(rankings, stages_view, q),
            }
        )
    return scored


def _latency(rows: list[dict[str, Any]], score_cache: RerankScoreCache, device: str) -> dict[str, float]:
    est_rerank = []
    e2e = []
    retrieval = []
    for row in rows:
        t = row["timings"]
        models = row.get("rerank_models") or []
        ms_pair = mean([score_cache.ms_per_pair(m, device) for m in models]) if models else 0.0
        rerank_ms = t["rerank_pairs"] * ms_pair if models else 0.0
        est_rerank.append(rerank_ms)
        retrieval.append(t["retrieval_ms"])
        e2e.append(t["retrieval_total_ms"] - t["rerank_model_ms_observed"] + rerank_ms)
    return {
        "retrieval_p50": percentile(retrieval, 0.5),
        "rerank_p50": percentile(est_rerank, 0.5),
        "rerank_p95": percentile(est_rerank, 0.95),
        "e2e_p50": percentile(e2e, 0.5),
        "e2e_p95": percentile(e2e, 0.95),
        "pairs_mean": mean([row["timings"]["rerank_pairs"] for row in rows]),
    }


def _twin_baseline(cfg: dict[str, Any], configs: dict[str, dict[str, Any]]) -> str | None:
    key = RetrievalConfig.from_dict(cfg).first_stage_key()
    for name, other in configs.items():
        if other.get("reranker") is None and RetrievalConfig.from_dict(other).first_stage_key() == key:
            return name
    return None


def cmd_summarize(args: argparse.Namespace) -> int:
    results = _read_jsonl(Path(args.results) / "results.jsonl")
    score_cache = RerankScoreCache(Path(args.score_cache))
    label_sets: dict[str, dict[str, QueryLabels]] = {}
    for spec in args.labels:
        corpus, _, path = spec.partition("=")
        _, queries = load_labels(path)
        label_sets[corpus] = {q.query_id: q for q in queries}
    scored = _score_rows(results, label_sets)
    only = set(args.only.split(",")) if args.only else None

    config_order: list[str] = []
    config_params: dict[str, dict[str, Any]] = {}
    for row in scored:
        if row["config"] not in config_params:
            config_order.append(row["config"])
            config_params[row["config"]] = row["config_params"]
    if only:
        config_order = [c for c in config_order if c in only]

    groups = sorted({r["corpus"] for r in scored}) + (["ALL"] if len({r["corpus"] for r in scored}) > 1 else [])
    out: dict[str, Any] = {"label_files": args.labels, "groups": {}}
    md: list[str] = [f"# Retrieval evaluation: {Path(args.results).name}", ""]
    md.append(f"Labels: {', '.join(args.labels)}")
    md.append(
        f"Ranking metrics use `{RANKING_STAGE}` (reranker order over the scored pool, then unscored candidates); "
        f"hand-off metrics use `{HANDOFF_STAGE}` (after score filter and top-n); context metrics use `{CONTEXT_STAGE}`."
    )
    md.append("")

    for group in groups:
        rows_g = [r for r in scored if group == "ALL" or r["corpus"] == group]
        n_queries = len({(r["corpus"], r["query_id"]) for r in rows_g})
        md += [f"## {group} ({n_queries} labelled queries)", ""]
        md.append(
            "| Config | Hit@6 | MRR | nDCG@10 | R@5 | R@10 | R@20 | Handoff Hit@6 | Handoff n | Context Hit@6 | Cov@6 | Rerank p50 ms | Retrieval+rerank p50 ms |"
        )
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        group_summary = {}
        per_config: dict[str, list[dict[str, Any]]] = {}
        for name in config_order:
            rows_c = sorted(
                (r for r in rows_g if r["config"] == name), key=lambda r: (r["corpus"], r["query_id"])
            )
            if not rows_c:
                continue
            per_config[name] = rows_c

            def m(stage: str, key: str) -> float:
                return mean([r["metrics"][stage][key] for r in rows_c])

            lat = _latency(rows_c, score_cache, args.latency_device)
            summary = {
                "hit@6": m(RANKING_STAGE, f"hit@{HIT_CUTOFF}"),
                "mrr": m(RANKING_STAGE, "mrr"),
                "ndcg@10": m(RANKING_STAGE, f"ndcg@{NDCG_K}"),
                **{f"recall@{k}": m(RANKING_STAGE, f"recall@{k}") for k in RECALL_KS},
                "handoff_hit@6": m(HANDOFF_STAGE, f"hit@{HIT_CUTOFF}"),
                "handoff_mrr": m(HANDOFF_STAGE, "mrr"),
                "handoff_n": mean([len(r["rankings"][HANDOFF_STAGE]) for r in rows_c]),
                "context_hit@6": m(CONTEXT_STAGE, f"hit@{HIT_CUTOFF}"),
                "coverage@6": m(RANKING_STAGE, f"coverage@{HIT_CUTOFF}"),
                "stage_means": {
                    stage: {key: mean([r["metrics"][stage][key] for r in rows_c]) for key in rows_c[0]["metrics"][stage]}
                    for stage in STAGES
                },
                "outcomes": {},
                "latency": lat,
                "n": len(rows_c),
            }
            for r in rows_c:
                summary["outcomes"][r["outcome"]] = summary["outcomes"].get(r["outcome"], 0) + 1
            group_summary[name] = summary
            md.append(
                f"| {name} | {_fmt(summary['hit@6'])} | {_fmt(summary['mrr'])} | {_fmt(summary['ndcg@10'])} | "
                f"{_fmt(summary['recall@5'])} | {_fmt(summary['recall@10'])} | {_fmt(summary['recall@20'])} | "
                f"{_fmt(summary['handoff_hit@6'])} | {_fmt(summary['handoff_n'], 1)} | {_fmt(summary['context_hit@6'])} | "
                f"{_fmt(summary['coverage@6'])} | {_fmt(lat['rerank_p50'], 0)} | {_fmt(lat['e2e_p50'], 0)} |"
            )
        md.append("")

        # Stage funnel for each config: where relevant evidence is found / lost.
        md += ["### Stage funnel (Hit@6 / Recall@20 per stage)", ""]
        md.append("| Config | " + " | ".join(STAGES) + " |")
        md.append("|---|" + "---|" * len(STAGES))
        for name, summary in group_summary.items():
            cells = [
                f"{_fmt(summary['stage_means'][s][f'hit@{HIT_CUTOFF}'], 2)} / {_fmt(summary['stage_means'][s]['recall@20'], 2)}"
                for s in STAGES
            ]
            md.append(f"| {name} | " + " | ".join(cells) + " |")
        md.append("")

        md += ["### Evidence flow after the rerank hand-off", ""]
        md.append(
            "Hit@6 / MRR / nDCG@10 at each stage. Discard counts are totals over queries; "
            "coverage is graded gain in the final context over the ideal for that many slots."
        )
        md.append("")
        md.append(
            "| Config | Hand-off | Governing | Packing | Final context | Rel. discarded | Rel. added | #1 discarded | "
            "#1 relevant discarded | Lost all evidence | Coverage | Ctx chunks | E2E p50 CPU ms | E2E p50 GPU ms |"
        )
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, rows_c in per_config.items():
            cells = []
            for stage in DOWNSTREAM_STAGES:
                vals = [mean([r["metrics"][stage][k] for r in rows_c]) for k in (f"hit@{HIT_CUTOFF}", "mrr", f"ndcg@{NDCG_K}")]
                cells.append(" / ".join(_fmt(v, 2) for v in vals))

            def flow_total(key: str) -> int:
                return int(sum(r["flow"][key] for r in rows_c))

            gpu = _latency(rows_c, score_cache, "cuda")
            flow_summary: dict[str, Any] = {
                key: sum(r["flow"][key] for r in rows_c)
                for key in ("relevant_discarded", "relevant_added", "top1_discarded", "top1_relevant_discarded", "lost_all_evidence")
            }
            flow_summary["context_coverage"] = mean([r["flow"]["context_coverage"] for r in rows_c])
            flow_summary["context_size"] = mean([r["flow"]["context_size"] for r in rows_c])
            flow_summary["stage_means"] = {
                stage: {k: mean([r["metrics"][stage][k] for r in rows_c]) for k in (f"hit@{HIT_CUTOFF}", "mrr", f"ndcg@{NDCG_K}")}
                for stage in DOWNSTREAM_STAGES
            }
            flow_summary["latency_gpu"] = gpu
            group_summary[name]["flow"] = flow_summary
            md.append(
                f"| {name} | " + " | ".join(cells) + f" | {flow_total('relevant_discarded')} | {flow_total('relevant_added')} | "
                f"{flow_total('top1_discarded')} | {flow_total('top1_relevant_discarded')} | {flow_total('lost_all_evidence')} | "
                f"{_fmt(flow_summary['context_coverage'])} | {_fmt(flow_summary['context_size'], 1)} | "
                f"{_fmt(group_summary[name]['latency']['e2e_p50'], 0)} | {_fmt(gpu['e2e_p50'], 0)} |"
            )
        md.append("")

        explicit: dict[str, Any] = {}
        if args.compare:
            md += ["### Paired comparisons (explicit)", ""]
            md.append("| Candidate | Baseline | Metric | Delta | 95% CI | p (sign-flip) | W/L/T |")
            md.append("|---|---|---|---|---|---|---|")
            metric_specs = (
                ("Ranking MRR", lambda r: r["metrics"][RANKING_STAGE]["mrr"]),
                ("Ranking nDCG@10", lambda r: r["metrics"][RANKING_STAGE][f"ndcg@{NDCG_K}"]),
                ("Hand-off Hit@6", lambda r: r["metrics"][HANDOFF_STAGE][f"hit@{HIT_CUTOFF}"]),
                ("Context Hit@6", lambda r: r["metrics"][CONTEXT_STAGE][f"hit@{HIT_CUTOFF}"]),
                ("Context MRR", lambda r: r["metrics"][CONTEXT_STAGE]["mrr"]),
                ("Context nDCG@10", lambda r: r["metrics"][CONTEXT_STAGE][f"ndcg@{NDCG_K}"]),
                ("Context coverage", lambda r: r["flow"]["context_coverage"]),
                ("Relevant discarded", lambda r: r["flow"]["relevant_discarded"]),
            )
            for spec in args.compare.split(","):
                cand_name, _, base_name = spec.partition(":")
                if cand_name not in per_config or base_name not in per_config:
                    continue
                base_rows = {(r["corpus"], r["query_id"]): r for r in per_config[base_name]}
                cand_rows = {(r["corpus"], r["query_id"]): r for r in per_config[cand_name]}
                keys = sorted(set(base_rows) & set(cand_rows))
                result: dict[str, Any] = {"baseline": base_name, "candidate": cand_name}
                for label, getter in metric_specs:
                    a = [getter(base_rows[k]) for k in keys]
                    b = [getter(cand_rows[k]) for k in keys]
                    ci = paired_bootstrap_ci(a, b)
                    pval = sign_flip_p_value(a, b)
                    wlt = win_loss_tie(a, b)
                    result[label] = {**ci, "p": pval, **wlt}
                    md.append(
                        f"| {cand_name} | {base_name} | {label} | {ci['delta']:+.3f} | [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}] | "
                        f"{pval:.3f} | {wlt['wins']}/{wlt['losses']}/{wlt['ties']} |"
                    )
                ids_changed = sum(
                    set(base_rows[k]["rankings"][CONTEXT_STAGE]) != set(cand_rows[k]["rankings"][CONTEXT_STAGE]) for k in keys
                )
                rel_changed = sum(
                    (set(base_rows[k]["rankings"][CONTEXT_STAGE]) & base_rows[k]["labels"].relevant)
                    != (set(cand_rows[k]["rankings"][CONTEXT_STAGE]) & cand_rows[k]["labels"].relevant)
                    for k in keys
                )
                result["context_ids_changed"] = ids_changed
                result["context_relevant_set_changed"] = rel_changed
                result["n"] = len(keys)
                md.append(f"| {cand_name} | {base_name} | Final context chunk set differs | {ids_changed}/{len(keys)} queries | | | |")
                md.append(
                    f"| {cand_name} | {base_name} | Relevant chunks in final context differ | {rel_changed}/{len(keys)} queries | | | |"
                )
                explicit[spec] = result
            md.append("")

        md += ["### Reranker outcomes per query (first relevant chunk, top-6 window)", ""]
        outcome_keys = ["not_retrieved", "outside_pool", "fixed", "improved", "unchanged", "worsened", "harmed", "both_miss", "filter_dropped"]
        md.append("| Config | " + " | ".join(outcome_keys) + " |")
        md.append("|---|" + "---|" * len(outcome_keys))
        for name, summary in group_summary.items():
            if config_params[name].get("reranker") is None:
                continue
            md.append(f"| {name} | " + " | ".join(str(summary["outcomes"].get(k, 0)) for k in outcome_keys) + " |")
        md.append("")

        md += ["### Paired comparisons vs the same first stage without reranker", ""]
        md.append("| Config | vs | Metric | Delta | 95% CI | p (sign-flip) | W/L/T |")
        md.append("|---|---|---|---|---|---|---|")
        comparisons = {}
        for name in group_summary:
            params = config_params[name]
            baseline = args.baseline if (args.baseline and params.get("reranker") is None and name != args.baseline) else None
            if params.get("reranker") is not None:
                baseline = _twin_baseline(params, {k: config_params[k] for k in group_summary})
            if not baseline or baseline == name or baseline not in per_config:
                continue
            base_rows = {(r["corpus"], r["query_id"]): r for r in per_config[baseline]}
            cand_rows = {(r["corpus"], r["query_id"]): r for r in per_config[name]}
            keys = sorted(set(base_rows) & set(cand_rows))
            comparisons[name] = {"baseline": baseline}
            for label, stage, key in (
                ("MRR", RANKING_STAGE, "mrr"),
                ("nDCG@10", RANKING_STAGE, f"ndcg@{NDCG_K}"),
                ("Hit@6", RANKING_STAGE, f"hit@{HIT_CUTOFF}"),
                ("Handoff Hit@6", HANDOFF_STAGE, f"hit@{HIT_CUTOFF}"),
                ("Context Hit@6", CONTEXT_STAGE, f"hit@{HIT_CUTOFF}"),
            ):
                a = [base_rows[k]["metrics"][stage][key] for k in keys]
                b = [cand_rows[k]["metrics"][stage][key] for k in keys]
                ci = paired_bootstrap_ci(a, b)
                p = sign_flip_p_value(a, b)
                wlt = win_loss_tie(a, b)
                comparisons[name][label] = {**ci, "p": p, **wlt}
                md.append(
                    f"| {name} | {baseline} | {label} | {ci['delta']:+.3f} | [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}] | "
                    f"{p:.3f} | {wlt['wins']}/{wlt['losses']}/{wlt['ties']} |"
                )
        md.append("")
        out["groups"][group] = {"configs": group_summary, "comparisons": comparisons, "explicit_comparisons": explicit}

    if args.per_query:
        md += ["## Per-query first relevant rank (ranking stage)", ""]
        names = [n for n in config_order if n in (args.per_query.split(","))]
        md.append("| Corpus | Query | " + " | ".join(names) + " |")
        md.append("|---|---|" + "---|" * len(names))
        index = {(r["config"], r["corpus"], r["query_id"]): r for r in scored}
        for corpus, qid in sorted({(r["corpus"], r["query_id"]) for r in scored}):
            cells = []
            for n in names:
                r = index.get((n, corpus, qid))
                if r is None:
                    cells.append("-")
                    continue
                rank = next(
                    (i + 1 for i, cid in enumerate(r["rankings"][RANKING_STAGE]) if cid in r["labels"].relevant), None
                )
                cells.append(f"{rank if rank else '>'}{'' if r['outcome'] in ('no_reranker', 'unchanged') else ' ' + r['outcome']}")
            md.append(f"| {corpus} | {qid} | " + " | ".join(cells) + " |")
        md.append("")

    if args.per_query_context:
        names = [n for n in config_order if n in args.per_query_context.split(",")]
        md += [
            "## Per-query evidence: hand-off first relevant rank -> final-context first relevant rank",
            "",
            "`-` = no relevant chunk at that stage. Only queries where the listed configs differ are shown.",
            "",
        ]
        md.append("| Corpus | Query | " + " | ".join(names) + " |")
        md.append("|---|---|" + "---|" * len(names))
        index = {(r["config"], r["corpus"], r["query_id"]): r for r in scored}

        def cell(r: dict[str, Any] | None) -> str:
            if r is None:
                return "n/a"
            rel = r["labels"].relevant
            h = first_relevant_rank(r["rankings"][HANDOFF_STAGE], rel)
            f = first_relevant_rank(r["rankings"][CONTEXT_STAGE], rel)
            return f"{h or '-'} -> {f or '-'}"

        for corpus, qid in sorted({(r["corpus"], r["query_id"]) for r in scored}):
            cells = [cell(index.get((n, corpus, qid))) for n in names]
            if len(set(cells)) > 1:
                md.append(f"| {corpus} | {qid} | " + " | ".join(cells) + " |")
        md.append("")

    out_path = Path(args.out or (Path(args.results) / "summary"))
    out_path.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    out_path.with_suffix(".json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("\n".join(md))
    return 0


# ── CLI ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    default_workdir = str(PROJECT_ROOT / "storage" / "retrieval_eval")
    default_cache = str(PROJECT_ROOT / "storage" / "retrieval_eval" / "rerank_scores.json")

    p = sub.add_parser("run")
    p.add_argument("--corpus", action="append", required=True, help="name=path/to/corpus.json")
    p.add_argument("--queries", action="append", required=True, help="name=path/to/labels.json")
    p.add_argument("--suite", action="append", help="headline|depth|pool|filter|rrf_k|first_stage_ablation|tuned|tuned_sweep|downstream")
    p.add_argument("--configs", help="JSON list of RetrievalConfig dicts")
    p.add_argument("--tuned-config", help="JSON dict of first-stage parameters for tuned suites")
    p.add_argument("--out", required=True)
    p.add_argument("--workdir", default=default_workdir)
    p.add_argument("--score-cache", default=default_cache)
    p.add_argument("--response-mode", default="standard")
    p.add_argument(
        "--score-device",
        default="cpu",
        help="device used to compute cross-encoder scores for ranking (latency is measured separately)",
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("pool")
    p.add_argument("--results", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--corpus-path", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--stages", default="dense:10,bm25:10,candidates:10,post_rerank_prefilter:10")
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--chars", type=int, default=500)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    p.set_defaults(func=cmd_pool)

    p = sub.add_parser("import-judgments")
    p.add_argument("--pool", required=True)
    p.add_argument("--judgments", required=True, help='{"query_id": {"<n>": grade}}')
    p.add_argument("--queries", required=True)
    p.add_argument("--corpus-path", required=True)
    p.add_argument("--pool-note", default="")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_import_judgments)

    p = sub.add_parser("keyword-labels")
    p.add_argument("--queries", required=True)
    p.add_argument("--corpus-path", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_keyword_labels)

    p = sub.add_parser("bm25-grid")
    p.add_argument("--labels", action="append", required=True, help="name=labels.json@corpus.json")
    p.add_argument("--workdir", default=default_workdir)
    p.add_argument("--out", required=True)
    p.add_argument("--top", type=int, default=15)
    p.set_defaults(func=cmd_bm25_grid)

    p = sub.add_parser("latency")
    p.add_argument("--results", required=True)
    p.add_argument("--corpus", action="append", required=True, help="name=path/to/corpus.json")
    p.add_argument("--queries", action="append", required=True, help="name=path/to/labels.json")
    p.add_argument("--models", default="base,large")
    p.add_argument("--device", default="cpu")
    p.add_argument("--pool", type=int, default=24)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--score-cache", default=default_cache)
    p.set_defaults(func=cmd_latency)

    p = sub.add_parser("summarize")
    p.add_argument("--results", required=True)
    p.add_argument("--labels", action="append", required=True, help="name=labels.json")
    p.add_argument("--score-cache", default=default_cache)
    p.add_argument("--latency-device", default="cpu")
    p.add_argument("--only", help="comma-separated config names")
    p.add_argument("--baseline", help="baseline config for no-reranker comparisons")
    p.add_argument("--per-query", help="comma-separated config names for the per-query table")
    p.add_argument("--per-query-context", help="comma-separated config names for the per-query evidence table")
    p.add_argument("--compare", help="comma-separated candidate:baseline config pairs")
    p.add_argument("--out")
    p.set_defaults(func=cmd_summarize)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
