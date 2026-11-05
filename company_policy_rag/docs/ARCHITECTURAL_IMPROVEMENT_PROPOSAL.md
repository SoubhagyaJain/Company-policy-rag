# Architecture roadmap and implementation status

The SQLite catalog and initial ingestion recovery in section 1 are now
implemented. See CATALOG_AND_RECOVERY.md for code behavior, validation and the
preservation/rollback runbook. The remaining sections are proposals, not implemented.

The critical batch retains FastAPI, Next.js, Chroma, BM25, Ollama, and a local
deployment. The small deletion journal and embedding identity file are containment
measures. The new SQLite catalog records lifecycle and recovery state; versioned
index generations remain future work. No performance gains are claimed without
measurement.

## 1. Durable SQLite document catalog and reliable ingestion recovery — P1

Store document identity, original-file hash/path, jobs, attempts, expected chunk
counts, ingestion options, and lifecycle transitions in a separate library DB.
Keep telemetry independently disposable. Use a single worker first; persist job
claims and retry state rather than adopting a distributed queue prematurely.

Migration: back up files and indexes; import current IDs without regenerating
them; reconcile files, vectors, BM25, and deletion markers into an inventory.
Classify incomplete and ambiguous records for operator review. Import must be
idempotent and must not automatically reindex unknown libraries.

Compatibility: retain document/status routes and response fields; add lifecycle
detail incrementally. A saved job must recover with its original options after
restart. Avoid interpreting a partial vector index as a complete document.

Rollback: retain old files/indexes and the pre-import inventory. Gate catalog
reads behind a rollout flag; revert to the preserved generation, never rebuild
the old state by destructively overwriting it.

Tests: interrupt each ingestion stage; restart with queued/running jobs; retry
after parser, embedding, disk and index failures; repeat imports; preserve IDs,
source hashes, deletion state, and API pagination/status semantics.

## 2. Versioned index generations — P1; depends on catalog

Build derived artifacts into a staging generation, validate them, then atomically
publish its catalog pointer. Retrieval sees only published generations. Deletion
deactivates a document before retryable physical purging.

Migration: tag the verified current index as the initial generation. Unknown
embedding provenance requires a verification/adoption procedure before enabling
dense retrieval. Future model changes build a separate generation from retained
originals; no in-place replacement of the existing index.

Compatibility: preserve external document IDs and citation URLs; add generation
identity internally. Readers must resolve one coherent snapshot. Legacy source
filename deletion markers must be translated carefully to avoid blocking a new
document with a reused filename.

Rollback: switch the active pointer to a retained, validated generation. Keep
cleanup separate from publication, with an explicit retention policy.

Tests: crashes before/after publication; concurrent upload/delete/query; corrupted
generation counts; model/dimension changes; no staged or deleted evidence in
new responses; citation identity across generation switches.

## 3. Generation-bound semantic caching — P1; depends on snapshots

Capture the generation used for retrieval and carry it through cache keys and
conversation evidence. Never resolve an old answer's generation at delayed-write
time. Revalidate source generations before publishing derived state. Bound entry
count, age, and obsolete-generation retention.

Migration: invalidate legacy cache entries without rewriting original documents.
Namespace new keys by generation, model revision, prompt, response depth and scope.
Keep the existing critical-fact guards on semantic reuse.

Compatibility: lower cache hit rate during rollout is acceptable; cached answers
must preserve the same final-answer/citation contract as newly generated ones.
Conversation history shown to the user is distinct from reusable evidence.

Rollback: disable the new cache namespace; never promote old entries into a new
generation. Source indexes remain unchanged.

Tests: deterministic barriers around delete/clear/delayed put; concurrent source
changes; stale conversation snapshots; negation/numeric/audience minimal pairs;
cache eviction and bounded-memory behavior.

## 4. Explicit verification and delivery outcomes — P1

Introduce accepted, partial, abstained, verification-unavailable, and interrupted
outcomes. Separate retry budget from permission to release an answer. Calibrate
strict/high-risk delivery using held-out supported and contradicted claims.
Retain provisional streaming where its product contract permits it.

Migration: add an optional outcome field while preserving current SSE event names
and final text/citations. Begin in shadow mode and compare with labeled answers.
Enable strict withholding only after evaluating false rejection and latency.

Compatibility: the critical batch fixes canonical final text but deliberately
does not change the verifier's current acceptance policy. UI and telemetry must
distinguish verification unavailable from verified success.

Rollback: disable outcome enforcement while preserving truthful outcome metadata;
keep canonical completion replacement enabled.

Tests: unsupported numbers, conditions, exceptions, contradictory rules, malformed
judge output, judge timeout, zero retries, deterministic answer repair, synchronous
and SSE equivalence, and final citation support.

## 5. Bounded concurrency and GPU coordination — P1/P2

Add bounded admission for chat and ingestion and shared retrieval executors.
Serialize conflicting turns per session or use state revision checks. Coordinate
chat, scanned-page ingestion, model switching and unload through one resource
lease policy; add request deadlines and cancellation propagation.

Migration: measure current cold/warm latency, queue wait, throughput and memory;
introduce conservative configurable limits. Keep one API process until shared
state semantics are established. Celery remains optional until it performs real
durable work that warrants its operational cost.

Compatibility: define retryable busy responses and queued progress. Browser retries
need idempotency keys before retrying requests that may already have executed.

Rollback: restore previous admission settings without dropping durable jobs or
in-progress source artifacts. Release model leases during shutdown.

Tests: same-session races, independent sessions, overload, slow consumers,
cancellation before first text, ingestion/chat GPU contention, process shutdown,
and bounded queue/thread/model-memory growth on representative hardware.

## 6. Truthful health and telemetry — P1

Separate liveness, required readiness and optional capability status. Use a single
request ID throughout HTTP, SSE, model work, telemetry and errors. Report telemetry
queue drops/write failures, add controlled draining, and aggregate dashboards in
SQL. Define redaction and retention for queries, answers and browser history.

Migration: add fields before removing hard-coded legacy ones. Existing metrics
remain available with documented meanings. Telemetry failure must not silently
be represented as complete audit durability.

Rollback: keep old dashboards through an adapter; disable payload capture without
removing operational counters. Preserve source data independently of telemetry.

Tests: dependency outages, optional vision disabled, queue saturation, disk errors,
shutdown drain, identifier correlation, redaction, retention, and SQL aggregates
compared against a small labeled fixture.

## 7. Incremental pipeline modularization — P2

Extract typed QueryPlan, EvidenceSnapshot, GeneratedAnswer, VerificationDecision
and FinalResult contracts. Resolve adapter compatibility during construction,
rather than catching TypeError and rerunning a request. Keep measured retrieval
and context-assembly behavior stable while extracting modules.

Migration: extract one boundary at a time behind existing entry points. Inventory
shared src utilities before retiring legacy orchestration. Declare which frontend
is maintained and preserve CLI consumers through adapters where required.

Compatibility: no framework migration or mandatory new infrastructure. Preserve
response modes, citations, deterministic follow-ups and disabled expensive stages.

Rollback: revert individual extraction commits; retain interface adapters until
parity tests pass. Avoid combining structural changes with ranking changes.

Tests: identical evidence IDs, ordering, source labels, final answers, cancellation
and errors through HTTP, SSE, CLI and evaluation adapters. Any ranking change needs
a separate held-out quality evaluation.

## Suggested sequence

Reliable ingestion recovery also needs explicit per-stage checkpoints: resume only
from validated artifacts, reclaim expired worker leases after restart, and record
whether an attempt is retryable. This depends on the catalog and generation work.
Migration imports unfinished jobs as interrupted rather than ready. Keep the
original attempt and source file for rollback; recovery tests must terminate the
worker at each stage and prove retries do not duplicate chunks or publish partial
indexes. Parser/model retries must have limits and preserve the original options.

Catalog and recovery → generation publication → snapshot-bound caches → calibrated
delivery outcomes. Health correctness and resource measurements can proceed
alongside that work. Introduce admission limits before increasing concurrency;
modularize gradually as the contracts become stable.
