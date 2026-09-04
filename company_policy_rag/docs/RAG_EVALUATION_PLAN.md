# RAG Evaluation Plan

## Purpose

This plan measures the supported `backend/` RAG runtime end to end and provides diagnostic stage metrics. The existing legacy evaluator remains useful only as a historical comparison; its scores must not be presented as evidence for `backend/rag/pipeline.py`.

All active-runtime current values are **BASELINE REQUIRED**. The historical CI smoke artifact (hit rate 1.0, context precision 0.895833, context recall 0.75) is explicitly labeled **legacy-path only**.

## Evaluation principles

1. Freeze the corpus, index generation, model revisions, effective config fingerprint, prompt version, and dataset version for every run.
2. Separate retrieval, context selection, answer, citation, conversation, robustness, latency, and resource metrics. A composite score never hides a hard safety failure.
3. Use answer-bearing chunk IDs and atomic claim labels where possible; fuzzy keyword matches are supporting signals, not ground truth.
4. Compare variants on identical queries and use paired statistics.
5. Run deterministic or low-temperature settings for regression gates. Record nondeterminism through repeated trials.
6. Keep test/evaluation documents isolated from production telemetry and prevent answer leakage from dataset labels.

## Harness architecture

Add an adapter that constructs the same dependencies as `backend/api/dependencies.py` and invokes `RAGPipeline.query`/stream using a frozen evaluation profile. The harness must capture complete candidate lineage and final evidence without depending on the web UI.

Required manifest:

```yaml
run_id: unique immutable ID
commit: git commit or dirty-tree digest
dataset_version: content hash
corpus_version: committed index generation
rag_stack_version: full architecture fingerprint
hardware_profile: CPU, RAM, GPU, VRAM, OS
models: embedding, reranker, generation, vision revisions
generation_parameters: temperature, seed when supported, max tokens
repeat_index: integer
started_at: UTC timestamp
```

Required outputs are per-example JSONL, aggregate JSON, human-readable Markdown, and failed-example bundles. Each failed bundle contains the query, expected labels, candidate rank lineage, final context order, citations, answer, metric reasons, and redacted trace.

## Dataset design

### Splits

| Split | Purpose | Minimum starting size | Gate role |
|---|---|---:|---|
| CI smoke | fast deterministic regressions | existing 8, relabeled for active backend | blocking after baseline stabilizes |
| Core golden | representative policy questions | existing 60 + 35 after dedup/relabel | nightly/release |
| Retrieval diagnostic | one/many answer-bearing chunks with hard negatives | 150 | retrieval gate |
| Citation/claim | atomic answer claims mapped to evidence spans | 100 | faithfulness/citation gate |
| Conversation | multi-turn follow-up/topic-shift sessions | 40 sessions / >=3 turns | release gate |
| No-answer | plausible but unsupported requests | 75 | abstention gate |
| Adversarial/security | injection, scope, leakage, malformed input | 100 | hard release gate |
| Layout/ingestion | file/page fixtures by format and failure class | >=10 per important class | ingestion gate |
| Performance | fixed query mix and corpus sizes | >=200 queries | capacity/SLO gate |
| Shadow/user | sampled, reviewed real tasks | **BASELINE REQUIRED** | post-deployment learning only |

### Question taxonomy

Every example carries one primary and optional secondary class:

- direct fact, numeric/date, definition, quote, list/enumeration;
- cross-section synthesis, cross-document comparison, exception/conditional policy;
- versioned identifier, acronym, code/config example, table lookup, figure/image evidence;
- broad summary, structural navigation, ambiguous intent;
- follow-up pronoun/ellipsis, correction, topic shift, mode switch;
- unsupported/no-answer, conflicting evidence, stale/deleted document;
- adversarial source instruction, prompt injection, scope escape, citation spoofing.

Stratify reports by class, document type, answer length, number of required evidence chunks, and scope. Aggregate success can otherwise hide a critical weak class.

### Ground-truth schema

```json
{
  "id": "q-001",
  "query": "...",
  "conversation": [],
  "scope": {"document_ids": [], "filters": {}},
  "answerability": "answerable|partial|unanswerable",
  "acceptable_answers": ["..."],
  "required_facts": [{"id": "f1", "text": "...", "risk": "high"}],
  "evidence": [{"document_id": "...", "chunk_id": "...", "span": "...", "supports": ["f1"]}],
  "hard_negatives": ["chunk-id"],
  "required_citation_count": 1,
  "tags": ["numeric", "direct_fact"]
}
```

Chunk IDs are version-specific. Store source-span anchors as well so relabeling can map across a new chunk policy.

## Ingestion metrics

| Metric | Definition |
|---|---|
| Parse success | documents/pages with expected extractable content divided by attempted |
| Silent-empty rate | pages expected to contain evidence but producing no text/asset outcome |
| Structure accuracy | heading/section/page/table labels matching annotated structure |
| Chunk boundary quality | percentage of annotated answer spans fully contained in at least one chunk |
| Duplication ratio | duplicate normalized tokens/chunks introduced by overlap/parsing |
| Index consistency | equality of manifest expected IDs, vector IDs, and BM25 IDs |
| Ingestion latency | p50/p95 by file type, page count, and chunk count |
| Recovery correctness | fault-injected jobs leaving no visible partial generation and succeeding idempotently on retry |

Initial thresholds: **BASELINE REQUIRED** by document class. Hard gate from day one: no failed store write may produce a ready/queryable document.

## Retrieval metrics

For each query with relevant set `Rel` and ranked list `R`:

- **HitRate@K:** 1 if any relevant chunk occurs in top K, averaged.
- **Recall@K:** `|Rel ∩ R[:K]| / |Rel|`.
- **Precision@K:** `|Rel ∩ R[:K]| / K`.
- **MRR@K:** reciprocal rank of the first relevant result, 0 if absent.
- **nDCG@K:** graded relevance gain discounted by rank and normalized by ideal ordering.
- **Scope violation rate:** fraction of returned candidates outside authorized/effective scope; hard target 0.
- **Stage survival:** relevant evidence survival from dense/BM25 union → fusion → rerank → pack.

Report K at 1, 3, 5, 10, and the actual generation context depth. Report dense-only, BM25-only, fused, reranked, and packed stages. Use exact labeled IDs first; span overlap and section/keyword fuzzy matching are secondary diagnostics.

## Answer and grounding metrics

### Deterministic checks

- required fact exact/normalized match;
- numeric/date/unit equality and contradiction;
- required entity/audience/exception coverage;
- quote/code substring integrity where verbatim output is required;
- answerability behavior (answer, qualify, or abstain);
- prohibited unsupported claim count.

### Semantic/human checks

- **Correctness:** required facts present and no material contradiction.
- **Faithfulness:** every externally verifiable answer claim is entailed by supplied evidence.
- **Relevance:** answer directly addresses the user without avoidable digression.
- **Completeness:** covers all required facts/parts at appropriate depth.
- **Calibration:** confidence/abstention matches evidence sufficiency.

Use two independent reviewers for the initial calibration set with adjudication and report inter-rater agreement. An LLM judge may scale review only after agreement with humans is measured, must not see hidden labels beyond its rubric, and must not be the sole hard gate for high-risk claims.

## Citation metrics

- **Citation validity:** cited source index exists in final packed context and is in scope. Hard target 100%.
- **Citation precision:** cited evidence actually supports the attached claim.
- **Citation recall/completeness:** supported-required claims with at least one adequate citation divided by supported-required claims.
- **Citation attribution accuracy:** document/page/section/snippet metadata matches the actual chunk.
- **Citation placement:** citation is unambiguously associated with its claim.
- **Unsupported citation rate:** citations pointing to irrelevant or contradictory evidence.

Numeric, quoted, code, obligation, exception, and eligibility claims are high risk and receive separate gates.

## Conversation metrics

- follow-up resolution accuracy;
- correct topic-shift detection;
- document/general mode isolation;
- active/selected document scope retention;
- correction incorporation and stale-turn rejection;
- evidence continuity when it is intended;
- cross-session leakage rate (hard target 0);
- answer quality and latency by turn number.

Each conversation test includes the full expected state transition, not only final text.

## Adversarial and robustness tests

| Class | Examples | Required outcome |
|---|---|---|
| Source prompt injection | document says to ignore system policy or exfiltrate data | treat as evidence text, never instruction |
| Query injection | user asks for hidden prompt/system files | refuse/limit according to product policy |
| Scope escape | selected doc query asks for another user's/global doc | no out-of-scope candidates or citations |
| Citation spoofing | source includes `[Source 1]` or model invents `[Source 9]` | only validated final indices accepted |
| Unsupported premise | asks about nonexistent policy | explicit no-evidence/abstention |
| Conflicting versions | old and current policy disagree | respect configured authority/freshness and disclose conflict |
| Malformed upload | oversized, corrupt, wrong MIME/extension, archive bomb pattern | typed rejection/failure, no partial index |
| Cache mutation | warm cache then delete/replace source | no stale evidence/answer |
| Dependency degradation | embedder/reranker/vector/BM25/Ollama unavailable | truthful degraded/error state |
| Disconnect overload | cancel many SSE requests | work stops, capacity recovers |

## Performance and capacity methodology

Measure on a named hardware profile after warmup and also from cold process/model state.

### Query stages

- admission queue;
- route/conversation/rewrite;
- embedding;
- dense search;
- BM25 search;
- fusion;
- rerank;
- expansion/packing;
- prompt construction;
- model queue/load;
- time to first token;
- generation duration/tokens per second;
- citation/verification;
- telemetry overhead;
- total response and stream completion.

Report p50/p90/p95/p99, error/abstention/degraded rates, throughput, CPU/RAM/GPU/VRAM, cache hit rates, and canceled-work duration. Workload mix must state response modes, query classes, model, corpus size, answer length, cache state, and concurrency.

Initial p95/TTFT/QPS/resource targets are **BASELINE REQUIRED**. Derive targets from user experience and hardware capacity, not repository comments.

## Composite quality score

The composite is a prioritization aid, never the only gate:

```text
Quality = 0.30 * retrieval_nDCG@10
        + 0.25 * citation_F1
        + 0.20 * faithfulness
        + 0.15 * answer_correctness
        + 0.10 * answer_completeness
```

Latency and resource use are separate constraints because blending them can let faster but unsafe answers pass. Hard constraints override composite score:

- scope violation = 0;
- valid citation indices = 100%;
- silent partial index = 0;
- cross-session leakage = 0;
- no-answer false assertion and high-risk faithfulness thresholds = **BASELINE REQUIRED**, then approved explicitly.

## Statistical comparison

- Use paired per-query deltas for candidate versus control.
- Bootstrap 95% confidence intervals for aggregate retrieval and answer metrics.
- Use McNemar's test for paired binary pass/fail changes and a paired permutation/Wilcoxon test for non-normal continuous deltas.
- Report effect size and critical-class regressions, not only p-values.
- For stochastic generation, repeat each selected example at least 3 times initially and report pass stability; choose repeat count from observed variance.
- Correct for multiple comparisons within broad experiment batches or pre-register one primary metric.

## Gate tiers

### Pull request

- deterministic unit/contract tests;
- active 8-example smoke on frozen tiny index;
- fault tests for P0 paths;
- wall-clock budget set after baseline.

### Nightly

- full core/retrieval/citation/no-answer sets;
- 3 repeats for stochastic subset;
- candidate lineage completeness and cache mutation tests.

### Release

- conversation, adversarial, ingestion fault, cold/warm performance, capacity, and human-reviewed sample;
- named configuration/index/model/hardware manifest;
- no hard-gate regression and signed threshold review.

### Post-deployment

- sampled, privacy-reviewed feedback set;
- drift by query/document class;
- shadow evaluation before policy/model/index changes;
- rollback on hard failure or approved error-budget breach.

## Baseline procedure

1. Fix false status/trace defects without altering retrieval behavior.
2. Freeze a representative corpus and relabel existing golden examples with answer-bearing evidence spans/IDs.
3. Run current active backend three times cache-off and cache-on, cold and warm.
4. Publish raw per-example results and aggregate confidence intervals.
5. Review failures by taxonomy and set initial thresholds based on user risk, not merely current performance.
6. Make the smoke suite blocking; make the full suite blocking after two stable scheduled runs.

## Evaluation failure workflow

Every failed case receives a stable failure code from `RAG_FAILURE_TAXONOMY.md` and the earliest failing stage. Promote a case to a permanent regression when the expected behavior is adjudicated. Never delete a failing example merely because the system cannot pass it; change or quarantine labels only with a recorded reason.

## Required implementation touchpoints

- `backend/rag/pipeline.py:RAGPipeline.query` and streaming adapter.
- `backend/api/dependencies.py` or a shared composition root for evaluation.
- New `backend/evaluation/` active-path runner and metrics.
- `backend/models/rag.py` and telemetry models for candidate lineage.
- `data/eval/` versioned schemas/manifests.
- `scripts/ci_eval_gate.py` and `.github/workflows/ci.yml`.
- Admin/debug tooling described in `RAG_OBSERVABILITY_PLAN.md`.

## Unknowns to resolve

- Business-risk weighting by question type and acceptable abstention rate.
- Representative corpus and traffic distribution.
- Required languages, OCR/layout classes, and visual-evidence prevalence.
- Human-review capacity and privacy rules.
- Named production hardware and concurrency target.
- Minimum material improvement and maximum latency/memory regression per experiment.

All remain **BASELINE REQUIRED** or product-owner decisions; none should be guessed from the current code.
