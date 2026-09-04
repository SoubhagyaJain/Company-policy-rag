# RAG Observability Plan

## Objective

Make every operational failure detectable and every evaluated answer scientifically reconstructable, while minimizing sensitive text retention. The current RAG trace and SQLite telemetry are the starting point; this plan extends them rather than requiring a new observability platform.

## Two telemetry planes

### Operational plane — always on

Low-cardinality counters, histograms, gauges, state/version identifiers, and typed failure codes. Store no full query, document text, context, or answer by default.

### Debug/evaluation plane — sampled or explicit

Full stage lineage and redacted previews for a short, approved retention period. Evaluation runs may retain fixture text because the corpus is controlled. Production debug capture requires access control, reason, sampling decision, expiry, and audit event.

## Identity and correlation

One `request_id` is accepted from a safe-format client header or generated at the API edge, then propagated to:

- HTTP response and every SSE event;
- `ChatService`, `RAGPipeline`, model client, ingestion and telemetry calls;
- `trace_id`, with a distinct `conversation_id` and `turn_id`;
- all logs and failure records.

Do not independently generate a second query request ID. For ingestion, use `job_id`, `document_id`, source hash, and index generation; never use an original filename as the only identity.

## Canonical query trace

```json
{
  "schema_version": 2,
  "request_id": "req_...",
  "trace_id": "trace_...",
  "conversation_id": "hashed-or-opaque",
  "turn_id": 3,
  "started_at": "UTC",
  "status": "completed|abstained|degraded|canceled|failed",
  "failure_code": null,
  "versions": {
    "commit": "...",
    "corpus_generation": "...",
    "rag_stack": "...",
    "prompt": "...",
    "embedding": "model@revision",
    "reranker": "model@revision",
    "generation": "model@revision"
  },
  "request": {
    "query_hash": "...",
    "redacted_preview": "optional sampled value",
    "response_mode": "standard",
    "scope_hash": "...",
    "filter_keys": [],
    "route": "factual",
    "route_confidence": 0.0,
    "rewrite_count": 1,
    "cache_eligible": true
  },
  "retrieval": {
    "dense": [{"chunk_id": "...", "rank": 1, "score": 0.0}],
    "bm25": [{"chunk_id": "...", "rank": 1, "score": 0.0}],
    "fusion": [{"chunk_id": "...", "rank": 1, "contributions": {}}],
    "rerank": [{"chunk_id": "...", "rank": 1, "raw_score": 0.0, "decision": "kept"}],
    "packed": [{"source_index": 1, "chunk_id": "...", "text_hash": "...", "tokens": 0}],
    "relaxations": [],
    "degraded_modes": []
  },
  "generation": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "first_token_ms": 0,
    "generation_ms": 0,
    "tokens_per_second": 0,
    "finish_reason": "stop",
    "canceled_at_ms": null
  },
  "citations": [{"claim_id": "c1", "source_index": 1, "chunk_id": "...", "status": "supported"}],
  "verification": {
    "grounding_status": "pass|fail|unavailable",
    "faithfulness": 0.0,
    "citation_completeness": 0.0,
    "reasons": []
  },
  "cache": {"retrieval": "miss", "answer": "miss", "key_version": "..."},
  "timings_ms": {},
  "resource": {"queue_ms": 0, "cpu_peak": null, "ram_peak_mb": null, "gpu_peak_mb": null}
}
```

Candidate arrays are required for evaluation and sampled debug traces, not for every production query. Aggregate counts/timings are always on.

## Stage timing contract

Use a monotonic clock and record duration plus outcome for:

1. API queue/validation/auth;
2. conversation resolution;
3. routing;
4. rewrite/multi-query planning;
5. query embedding;
6. dense search;
7. BM25 search;
8. fusion;
9. filter relaxation/scope validation;
10. reranking;
11. parent/adjacent expansion and vision;
12. context packing;
13. semantic answer-cache read/write;
14. model admission/load/queue;
15. prompt build;
16. time to first token and generation;
17. citation extraction;
18. verification/retry;
19. telemetry enqueue/write;
20. total response and stream completion.

Nested stages must reconcile with total time within a documented tolerance. Never compute generation throughput from end-to-end duration. Token counts use the actual model tokenizer when available and explicitly label estimates otherwise.

## Ingestion trace

Each job records:

- source hash, sanitized media type, size bucket, page count, job attempt;
- parser and chunk-policy versions;
- per-page text length, image/table/code flags, textless/partial outcome;
- raw document, chunk, vector, and lexical-entry expected/actual counts;
- embedding dimension/model/fallback state;
- stage start/end/duration/retry/error code;
- staging generation, validation results, commit or rollback outcome;
- cache invalidation generation and duration.

Do not report visual/vision success from raw-document count. A field is `not_run`, `unavailable`, `success`, `partial`, or `failed`.

## Operational metrics

Suggested metric names are semantic, not tied to a vendor.

### Requests

- `rag_query_total{status,route,response_mode,cache,degraded}`
- `rag_query_duration_ms{stage,status}` histogram
- `rag_ttft_ms{model,cache}` histogram
- `rag_active_requests`, `rag_model_queue_depth`, `rag_model_leases`
- `rag_cancellation_total{stage,outcome}` and cancellation-stop latency
- `rag_abstention_total{reason}`

### Retrieval/evidence

- candidate counts by stage and relevant-survival metrics in evaluation
- `rag_scope_violation_total` (alert immediately)
- reranker keep/reject counts and score distributions by version
- context tokens/chunks, truncation/expansion reasons
- citation valid/invalid/unsupported/completeness counts
- verification outcome and reason, never only a floating score

### Caches

- request/hit/miss/reject/write/error by cache and key version
- cache lookup/write latency and entry count
- stale-hit detector count (hard target zero)
- invalidation count/duration/failure by corpus generation

### Models/resources

- model loaded/warm/degraded states and load duration
- actual fallback usage by component
- prompt/completion tokens and generation tokens/s
- process CPU/RAM, GPU utilization/VRAM/temperature where available
- model queue wait, inference concurrency, timeout, unload/eviction

### Ingestion/index

- jobs by stage/outcome/file class, stage latency, retry count
- parse partial/textless/error pages
- expected versus actual vector/BM25 counts
- committed/staging generations and age
- index mutation/rebuild duration and query impact

### Telemetry system

- enqueue/write/error/drop counts, queue depth, oldest item age
- flush duration and unflushed records at shutdown
- debug sample count/expiry/delete failures

## Readiness, liveness, and degradation

### Liveness

The API event loop/process can answer a cheap endpoint. It does not claim RAG readiness.

### Readiness

A composed probe returns each dependency independently:

- active committed corpus generation readable;
- vector and BM25 generation/count consistency;
- configured embedding/reranker/text model loaded or permitted lazy state;
- Ollama reachable and selected model available;
- telemetry state (non-blocking degradation is explicit);
- admission controller accepting work;
- active configuration fingerprint.

Overall readiness is derived from required components and deployment profile. Remove hard-coded `true` fields. Degraded fallbacks are visible and strict mode may mark not ready.

## Dashboards

### Product health

Traffic, completed/abstained/degraded/failed/canceled, TTFT/total latency percentiles, cache hit rate, model queue depth, and current versions.

### Retrieval quality

Nightly/release HitRate/MRR/nDCG by question class; stage survival; citation precision/recall; failure-code trend. Do not calculate online relevance without labels.

### Ingestion/index

Job status funnel, partial/textless pages, consistency mismatches, rebuild/commit latency, generation age, cache invalidation.

### Local model capacity

Model residency/load, queue and active leases, inference latency/tokens, CPU/RAM/GPU/VRAM, cancellation waste, vision contention.

### Telemetry integrity

Queue depth/drops, missing trace fields, request/trace correlation failures, redaction failures, schema versions.

## Initial alerts

Absolute correctness alerts do not require baselining:

- any scope violation;
- any ready document with vector/BM25/manifest mismatch;
- any accepted out-of-range citation;
- any stale-cache detector hit;
- any hard-coded/contradictory readiness or grounding state found by canary;
- telemetry redaction failure on a known sensitive pattern.

Capacity/latency/error-rate alert thresholds are **BASELINE REQUIRED**. Establish normal ranges on named hardware, then use sustained multi-window alerts rather than reacting to one slow local-model request.

## Scientific reliability controls

- Schema and metric definitions are versioned; changing a definition starts a new series.
- Trace clocks are monotonic for durations and UTC for correlation.
- Sampling is deterministic from trace ID and recorded so rates can be weighted correctly.
- Missing values remain null with a reason; never replace them with zero or success.
- Health probes and instrumentation have fault-injection tests.
- Every evaluation report verifies trace completeness and effective config/index/model fingerprints.
- Scores are labeled heuristic, judge-based, or human; uncalibrated heuristic scores are not presented as probabilities.

## Privacy, security, and retention

Classify queries, document text, context, answers, filenames, user/session identity, and prompts as potentially sensitive.

- Hash or tokenize user/session IDs; do not log bearer credentials or raw headers.
- Default logs contain query hash/length/class, not full query.
- Redact email, phone, employee/customer identifiers, secrets, and configured policy terms from previews.
- Full debug traces require explicit role, purpose, bounded sample, and expiry.
- Encrypt storage and backups when required by deployment policy.
- Define deletion propagation across source file, indexes, caches, traces, and backups.
- Retention periods and authorized viewers are **BASELINE REQUIRED** through product/security review.

## Trace debugger

Extend the existing admin experience only after access control is appropriate. A trace view should show:

```text
Request/versions/status
  Route → rewrite → effective scope
  Dense ranks ┐
              ├→ RRF → rerank decisions → final source order
  BM25 ranks ┘
  Prompt/context token budget and text hashes
  Answer claims → citation → supporting span → validation reason
  Timings, cache decisions, degraded states, retries, failure code
```

The view must make rejected candidates and filter relaxation visible; showing only final citations cannot diagnose retrieval misses.

## Implementation sequence and files

1. Fix truth defects in `backend/rag/pipeline.py` and health/admin routes.
2. Propagate one request ID through `backend/api/main.py`, chat routes/service, pipeline, SSE, and telemetry.
3. Extend `backend/models/rag.py` and `backend/models/telemetry_models.py` with versioned lineage and typed outcomes.
4. Instrument dense, BM25, hybrid, reranker, context, caches, Ollama, and ingestion stages at their owning functions.
5. Update `backend/services/telemetry_service.py` and telemetry DB migration/lifecycle; flush on API shutdown.
6. Add privacy filters and separate operational/debug persistence.
7. Build evaluation/debug reports, then dashboards/alerts from stable definitions.

## Acceptance criteria

- All API/SSE/log/telemetry records for a request share one correlation ID.
- All correctness states are derived and fault-tested; missing means unknown, never success.
- Evaluation traces contain 100% of required candidate lineage and version fields.
- Operational trace overhead, drop rate, flush reliability, and storage volume meet targets **BASELINE REQUIRED**.
- No known sensitive fixture appears in default operational logs.
- A retrieval miss can be assigned to the earliest failing stage without rerunning under ad hoc logging.
