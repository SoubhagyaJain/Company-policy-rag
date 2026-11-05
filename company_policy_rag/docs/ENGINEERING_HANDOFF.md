# Engineering handoff and final validation

Live-model follow-up: the app was started and tested without Docker using
`qwen3.5:9b` and `qwen3-embedding:0.6b`. Document upload, hybrid retrieval,
streaming, source viewing and restart persistence worked. Five completed queries
also exposed citation/semantic verification gaps, completeness false negatives
and interrupted-draft recovery limitations. See LIVE_RAG_VALIDATION.md for the
measured results and preserved evidence; answer quality is not all-green.

Subsequent phase: SQLite document catalog and ingestion recovery are implemented
in checkpoints `9acb161` and `f2316e0`, with final verification documented in
CATALOG_AND_RECOVERY.md. Final core: **665 passed, 1 optional dependency skip**;
frontend: **221 passed**, TypeScript and selected lint passed. Docker remains
excluded. The critical-batch results below are retained as historical evidence;
catalog/recovery is no longer proposal-only.

Branch: `codex/production-critical-corrections`, based on `1182a79`.
All six critical correction areas are implemented and offline validation passes.
Docker container recreation remains unvalidated and was excluded from further
work at the user's request. No application containers were recreated.

No existing user library was migrated, deleted, replaced, or reindexed. Existing
document IDs, originals and indexes were preserved. No runtime dependencies were
added. The user's pre-existing API startup-log edit, vision checkpoint checks,
packaging edits, notes and untracked readiness test remain separate and intact.

## Saved checkpoints

| Commit | Correction | Incremental validation |
|---|---|---|
| fe72a1e | Correct Docker library mount at its existing path | 6 offline tests; opt-in container test added |
| 3992edc | Retryable deletion and durable retrieval revocation | 55 targeted tests |
| 924e96e | Canonical final answer, citations, history, provisional UI | 31 backend, 220 frontend tests and TypeScript |
| f86d7c3 | Embedding validation, identity, required API persistence | 69 targeted tests; original core suite 439 passed |
| 6484296 | Local network, origin/host and administrative containment | 44 targeted backend, 221 frontend tests and TypeScript |
| e94dd27 | Artifact-integrity implementation and recovery checkpoint | Initial failure regressions passed; follow-up failures recorded |
| d6506a9 | Finish artifact correctness, exact tag verification, offline core manifest | Final combined core: 603 passed, 1 skipped |

## Verified defects and implemented behavior

Paths below are relative to the application directory `company_policy_rag`.

| Area | Files and entry points | Before | After and regression evidence |
|---|---|---|---|
| A | `docker-compose.yml`, `docker/Dockerfile.backend` | API library root did not match the persistent bind mount | API and worker mount existing `app/storage`; offline contract tests in `tests/unit/test_deployment_persistence.py`. Two-container restore test added in `tests/integration/test_container_library_recreation.py`; execution pending. |
| B | `backend/embeddings/vector_store.py`: `mark_document_deleted`, `delete_by_document_id`, `delete_by_source`; `backend/services/document_service.py`: `delete_document`, `retry_document`; `backend/rag/pipeline.py`: `_document_revoked`, `_visible_chunks`; `backend/api/routes/documents.py` | Persistent delete failure could look successful and stale evidence remained retrievable | Durable revocation before physical cleanup; failure returns incomplete cleanup, source retained for retry, dense/BM25/cache/reused evidence/assets hidden, restart preserves revocation. `tests/unit/test_deletion_integrity.py` injects persistent failures and tests restart/retry, conflicts and ingestion compensation. |
| C | `backend/services/chat_service.py`: `stream_query`; `frontend/lib/api-client.ts`, `frontend/lib/chat-completion.ts`: `completeMessage`; `frontend/lib/types.ts`, `frontend/hooks/useChatStream.ts`, `frontend/components/space/SpaceMessage.tsx` | Draft deltas could remain after the pipeline repaired the final answer/citations | Authoritative completion replaces text, including empty text, and citations in SSE/UI/telemetry/history/state. Drafts and interrupted streams are identified. `tests/unit/test_canonical_answer_delivery.py` and `frontend/tests/canonical_delivery.test.ts` test the actual stream transport and completion behavior. |
| D | `backend/embeddings/embeddings.py`: `validate_embedding`, `EmbeddingService`; `backend/embeddings/vector_store.py`: initialization and `bind_embedding_model`; `backend/retrieval/vector.py`; `backend/services/document_service.py` | Short/malformed vectors could be padded or synthetic; required persistence could silently become memory-only | Exact batch cardinality, numeric/finite/nonzero/dimension validation; explicit test-only hash vectors; model/device-specific loaders; API persistence required; identity manifest prevents mixing incompatible models. `tests/unit/test_embedding_integrity.py` covers fail-closed behavior and preserved old indexes. |
| E | `src/config.py`; `backend/api/security.py`: `require_admin`; `backend/api/main.py`: `create_app`; route guards in `backend/api/routes/chat.py`, `documents.py`, `models.py`; `docker-compose.yml`; `frontend/lib/api-client.ts` | Broad network/browser access and unrestricted administrative actions | Loopback publication, internal Redis, explicit hosts/origins, trusted-peer or configured-key administrative access; optional browser key sent only on administrative calls. `tests/unit/test_security_containment.py` tests rejected origins, hosts, remote peers and missing keys. Existing API tests use real loopback test peers. |
| F | `src/finetuning/merger.py`: `merge_lora_weights`; `gguf_exporter.py`: `convert_to_gguf`; `ollama_registrar.py`: registration and verification; `src/finetuning/__init__.py`; `scripts/export_and_register_ollama.py`, `scripts/run_finetune_pipeline.py` | Failed merge/export could synthesize apparently successful artifacts; CLI registration failure could return success | Real operations only, staged publication, preserved metadata, no overwrites, explicit registration completion and exact requested tag, nonzero failure exit. Fixtures moved to `tests/fixtures/model_artifacts.py`. `tests/unit/test_model_artifact_integrity.py` tests production failure paths without downloading models. |

The expanded artifact run exposed a negative CUDA index bug and older tests
patching the wrong imported `urlopen` symbol; both are fixed. A dataset dry-run
test now explicitly skips when its optional dependency is absent. New correction
regressions are included in `scripts/run_core_tests.py` for routine CI.

## Final checks executed

Commands run from the application directory unless specified otherwise:

| Command | Result |
|---|---|
| `.venv/Scripts/python.exe scripts/run_core_tests.py` | **603 passed, 1 skipped**, 80.69 seconds; no failures. Skip: optional `datasets` package for the dataset dry-run test. 153 dependency deprecation warnings. |
| `npm test` from `frontend` | **221 passed, 0 failed** |
| `node node_modules/typescript/bin/tsc --noEmit --incremental false --pretty false` from `frontend` | Exit 0 |
| `.venv/Scripts/python.exe scripts/production_retrieval_smoke.py --assert-minimums --output logs/critical_retrieval_smoke.json` | Exit 0; 8 cases, hit@3 = 1.0, MRR = 0.8542, floor = 0.8. Explicit test-only hash embeddings; this is not a real-model quality or performance claim. |
| `.venv/Scripts/python.exe -m pytest tests/integration/test_container_library_recreation.py -q -rs` | **1 skipped** because `RAG_CONTAINER_TEST_IMAGE` was not configured; Docker recreation not performed. |
| Selected changed/new Python source and artifact fixtures: `ruff check` | All checks passed; exact command below. |
| `git diff --check` | Passed |

```text
.venv/Scripts/python.exe -m ruff check backend/api/security.py backend/embeddings/embeddings.py src/finetuning/__init__.py src/finetuning/merger.py src/finetuning/gguf_exporter.py src/finetuning/ollama_registrar.py scripts/export_and_register_ollama.py scripts/run_finetune_pipeline.py scripts/run_core_tests.py scripts/production_retrieval_smoke.py tests/unit/test_model_artifact_integrity.py tests/fixtures/model_artifacts.py
```

Local raw outputs are saved under `logs/critical_core_validation.txt`,
`logs/critical_frontend_validation.txt`, `logs/critical_typescript_validation.txt`,
`logs/critical_smoke_validation.txt` and `logs/critical_retrieval_smoke.json`.
Logs are ignored by Git; these results and the regression code are committed.
The entire optional/hardware-dependent repository suite was not run. Real
Ollama/GPU training, merging, llama.cpp conversion and runtime model loading
still require explicit integration validation on suitable hardware/toolchains.

## Compatibility, recovery and remaining risks

Supported API routes, SSE event names, document IDs and citation URLs remain.
Intentional changes make previously misleading success fail explicitly: delete
409/503, interrupted streams, invalid embeddings, unavailable persistence,
unknown dense-index provenance, unauthorized administration, model simulation
and artifact overwrite. See `CRITICAL_CORRECTIONS.md` for configuration,
preservation and rollback steps before deploying to an existing installation.

The deletion journal is single-process containment; an answer already generating
may complete after deletion. Unknown legacy embedding provenance requires verified
adoption before dense reuse; existing indexes remain unchanged and lexical search
is available. Model revision/digest pinning is deferred. Header/file validation
does not establish inference usability. Private Docker peers are trusted behind
loopback bindings; public deployment needs an authenticated ingress. The legacy
root Dockerfile/release route has not been redesigned.

## Proposed next milestones; not implemented

`ARCHITECTURAL_IMPROVEMENT_PROPOSAL.md` supplies migrations, compatibility risks,
rollback and tests for the requested architectural improvements. Its catalog and
initial recovery milestone has since been implemented; the remaining work is:

1. Versioned index generations and verified legacy provenance adoption.
2. Generation-bound semantic caches and conversation evidence.
3. Explicit, calibrated verification and delivery outcomes.
4. Truthful health/telemetry and bounded admission/GPU resource coordination.
5. Incremental pipeline modularization after the contracts stabilize.

Docker recreation is the remaining deployment validation step when the operator
chooses to resume it. The larger architecture remains a separate future batch.
