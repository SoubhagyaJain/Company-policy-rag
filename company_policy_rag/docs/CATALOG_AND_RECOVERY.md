# SQLite document catalog and ingestion recovery

Branch: `codex/production-critical-corrections`. Initial checkpoint: `9acb161`;
recovery and compatibility checkpoint: `f2316e0`. Docker is excluded.
No existing user library was migrated or reindexed during implementation;
regressions use disposable temporary libraries. Pre-existing user edits remain
separate. No dependency was added: the catalog uses Python's `sqlite3`.

## What changed

`backend/services/document_catalog.py` owns a separate
`<library_root>/document_catalog.sqlite3` database. It uses WAL, FULL synchronous
transactions, foreign keys and a schema-version guard. Telemetry cleanup cannot
erase document identity or ingestion jobs. Unsupported/corrupt catalog databases
fail startup; they are never replaced with an empty catalog or memory fallback.

Tables retain original document IDs, source paths and SHA-256 hashes, metadata,
ingestion options, expected chunk counts, current and previous jobs, individual
attempts, lease owners/deadlines and lifecycle events. Job status includes stage
progress and failure details. New jobs and document state commit together.
Explicit retries create new job IDs while retaining prior attempt history.

`backend/services/document_service.py` integrates upload, status, retry, startup,
completion and deletion. Source files are staged, flushed and renamed before a
durable job is acknowledged. An unavailable executor leaves a queued job and its
source available for restart recovery. It no longer discards the accepted bytes.

Publication requires successful vector and BM25 writes plus the expected counts.
The final record and READY job state commit atomically. Startup validates READY
source hashes and vector counts; a missing/changed source or incomplete persistent
index becomes FAILED with an explanation rather than being inferred READY from
the presence of some vectors.

`backend/embeddings/vector_store.py`, `backend/retrieval/bm25.py` and
`backend/rag/pipeline.py` filter unfinished/failed/review/deleted catalog records.
Visibility also applies to cached candidates, reusable evidence and asset routes
in `backend/api/routes/documents.py`. Corpus fingerprints include the catalog
lifecycle revision to invalidate cached results after visibility changes. This
is a visibility guard, not the proposed versioned-generation cache architecture.

## Recovery behavior

The existing single-worker executor remains. A transaction claims each attempt;
an independent heartbeat renews the lease even while parsing or embedding has no
stage progress. A library-scoped OS file lock serializes writers across processes
because vector and BM25 stores cannot enforce lease fencing themselves. Process
death releases that lock. An expired lease alone cannot start a second writer
while the original writer still holds it.

Queued jobs resume at startup. Jobs left RUNNING wait for lease expiry, then a
single daemon wakeup resumes them without requiring browser polling. Status
polling also checks for recovery. Shutdown cancels the wakeup and drains the
worker through `DocumentService.close()`.

Recovery verifies the exact original path/hash and saved embedding configuration,
removes only that API job's partial derived rows, and replays parsing, chunking,
embedding and indexing using its saved category and chunk strategy. It does not
trust or reuse incomplete intermediate output. Source bytes and document ID stay
the same. Each crash recovery is a new recorded attempt. Three automatic attempts
are allowed by default; exhaustion becomes FAILED and requires explicit retry.
Ordinary parser/model/disk failures also remain FAILED for explicit retry, avoiding
an automatic loop on an unavailable dependency or invalid file.

## Lifecycle and compatibility

| Catalog state | API/search behavior |
|---|---|
| QUEUED / RUNNING | Existing TEXT_INDEXING status and progress; excluded from search/assets |
| READY | Existing READY response; searchable after committed publication |
| FAILED | Source retained, error recorded, existing retry endpoint available when identity is safe |
| LEGACY | Existing evidence retained; PARTIALLY_INDEXED with a completeness warning; no automatic reindex |
| REVIEW | Ambiguous legacy originals preserved and hidden; operator review required |
| DELETING / DELETED | Hidden; deletion journal retained, failed cleanup retryable; catalog identity/audit retained |

HTTP routes, response fields, upload HTTP 201, polling, document IDs and citation
URLs remain. Upload still returns TEXT_INDEXING while work runs. Retrying a READY
document remains idempotent. Two older tests that assumed immediate synchronous
readiness now poll the existing status endpoint. `frontend` needs no contract or
component changes.

## Configuration

| Setting | Default | Purpose |
|---|---|---|
| DOCUMENT_CATALOG_ENABLED | true | Use the durable lifecycle catalog |
| INGESTION_RECOVERY_ENABLED | true | Resume queued/interrupted jobs; false pauses automatic recovery, not new uploads |
| INGESTION_LEASE_SECONDS | 120 | Lease lifetime; heartbeat runs at one third of this interval |
| INGESTION_MAX_ATTEMPTS | 3 | Automatic attempt limit per job; explicit retry starts another recorded job |

Keep one API serving process per library. The OS lock protects writers during
recovery and accidental overlapping starts; it does not provide coherent shared
in-memory registries for a multi-process API deployment.

## Preservation, rollout and rollback

1. Stop the API and ingestion work. Preserve a complete copy of the existing
   library root: uploads, Chroma, BM25, content hashes, any catalog and its WAL,
   and migration inventory. Preserve configured external image/vision directories
   too. Use a new backup directory, compare IDs/counts and source SHA-256 hashes,
   and keep this pre-upgrade copy unchanged. No Docker work is required.
2. On the first catalog-enabled start, an atomically written
   `catalog_migration_inventory.json` records observed metadata and each discovered
   source path/hash. Import is idempotent and adds rows without regenerating IDs
   or reindexing legacy data. Repeated starts keep the original inventory.
3. Legacy vector rows without a historical completion record are marked LEGACY,
   with existing evidence retained and completeness explicitly unverified.
   Stored files without vectors become FAILED. Multiple originals sharing an ID
   become REVIEW. Nothing guesses which conflicting original should win.
4. Inspect the catalog and inventory before approving any legacy rebuild. Unknown
   embedding provenance remains subject to the prior embedding-identity guard.
   Verified legacy adoption and generation-based rebuild tooling are future work.
5. For rollback, stop serving and use the preserved pre-upgrade library with the
   previous code. Keep the new library intact for diagnosis. Disabling the catalog
   against a library containing unpublished, failed, review or deleting records
   raises an error so partial rows cannot become visible through legacy restore.

A consistent **catalog-only** backup is available through
`DocumentCatalog.backup(new_path)`. It uses SQLite's backup API to include WAL
state and refuses an existing destination. It is not a complete source/index
backup. Do not delete the live WAL or copy only the database while it is active.

If an original hash changes, restore the verified original from a preserved copy
or upload the intended revision as a new document. Do not edit catalog hashes to
make a mismatched file pass. Ambiguous source identity and index-only legacy
records require operator review. Incomplete `.upload` files from a crash before
acknowledgment remain preserved and are not automatically ingested.

## Validation

Final validation completed successfully:

| Command | Result |
|---|---|
| `.venv/Scripts/python.exe scripts/run_core_tests.py` | **665 passed, 1 skipped**, 98.33 seconds. Skip: optional `datasets` dependency in a training dry-run test. |
| Focused catalog/recovery/process/API/storage checks | **61 passed** before the last autonomous-wakeup test; all are included in the final core run. |
| `npm test` from `frontend` | **221 passed, 0 failed** |
| `node node_modules/typescript/bin/tsc --noEmit --incremental false --pretty false` from `frontend` | Exit 0 |
| Selected changed/new Python source and tests: Ruff | All checks passed |
| `git diff --check` | Passed |

```text
.venv/Scripts/python.exe scripts/run_core_tests.py
```

The core manifest now includes `tests/unit/test_document_catalog.py`,
`tests/unit/test_ingestion_recovery.py`,
`tests/integration/test_ingestion_process_recovery.py`, the document API tests,
decoupled-ingestion API tests and storage-service tests. Coverage includes:

- Atomic claims, rollback on SQL failure, wrong-owner publication, bounded
  attempts, OS locking, backup and unsupported schema.
- Interruption at extraction, section detection, chunking, embedding, vector,
  BM25 and finalization stages, followed by restart with preserved ID/hash/options
  and no duplicate chunks.
- Actual abrupt process exit after a persistent vector write, then recovery.
- Live heartbeat during stalled stage progress and lease-expiry recovery without
  browser polling.
- Unavailable executor, partial writes plus failed cleanup, missing vectors,
  changed original, corrupt catalog, idempotent legacy migration and ambiguous
  originals preserved for review.
- Cache/evidence/asset visibility, deletion and retry compatibility, and rollback
  refusing to expose unpublished rows.

The final core includes the autonomous lease-expiry wakeup test. An earlier
expanded run passed 623 tests. The two stale synchronous API expectations and the
old discard-on-queue-failure expectation were corrected to test the actual async
and durable-queue behavior; no test failures remain in the selected suites.
The initial additional API command referenced a nonexistent extended-format test
file and collected nothing; it was corrected to the actual document API paths.

Local raw outputs are in `logs/catalog_core_final_validation.txt`,
`logs/catalog_final_targeted_validation.txt`, `logs/catalog_frontend_validation.txt`
and `logs/catalog_typescript_validation.txt`. These ignored logs remain on disk;
the results, regression code and this runbook are committed. The core emitted
980 dependency deprecation warnings (Chroma/Pydantic and Starlette/httpx).
No Docker or real model/GPU integration was run.

Exact selected lint command, from the application directory:

```text
.venv/Scripts/python.exe -m ruff check backend/services/document_catalog.py backend/services/document_service.py backend/embeddings/vector_store.py backend/retrieval/bm25.py backend/rag/pipeline.py backend/api/routes/documents.py src/config.py scripts/run_core_tests.py tests/unit/test_document_catalog.py tests/unit/test_ingestion_recovery.py tests/integration/test_ingestion_process_recovery.py tests/unit/test_document_ingestion_hardening.py tests/test_ingestion_decoupled_architecture.py --output-format concise
```

## Remaining scope

This phase does not implement index generations, atomic publication across all
storage engines, calibrated answer-delivery outcomes, GPU scheduling, telemetry
redesign or pipeline modularization. No real model/GPU or power-failure test is
claimed. Filesystem loss/corruption still requires preserved originals/backups;
restart validation prevents it from silently becoming READY. In-flight answers
already generating are not revoked retroactively. Single-process API serving
and local filesystems are the supported boundary.
