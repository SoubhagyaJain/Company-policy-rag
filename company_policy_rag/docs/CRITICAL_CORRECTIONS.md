# Critical corrections: validation and deployment

The application keeps the existing `app/storage` library location, document IDs,
and indexes. Compose now mounts that exact directory. No migration or reindex is
performed automatically. The separate `storage` mount is retained.

## Existing installations: preserve before deploying

Do not recreate an old container before preserving its writable-layer library.
Stop requests and wait for ingestion to finish. Stop (do not remove) the old
`rag-backend` container, and use `docker cp rag-backend:/app/app/storage <backup>`
to copy its library to a **new, empty backup directory**. Preserve the old Compose
file, container, and all named volumes. Inventory files and compare SHA-256 hashes
between the copy and original before proceeding.

If the host's `app/storage` is empty, restore the verified backup there. If it
already contains data, compare document IDs and hashes and reconcile into a
separate staging directory; never overwrite or merge conflicting libraries
automatically. Test the preserved copy first. Deploy only after checking source
bytes, document IDs, Chroma counts, BM25 retrieval, and citations. Keep the old
container and backup until the new deployment has been verified.

Rollback uses the preserved old Compose configuration and stopped container, or
the verified backup. Do not use `docker compose down -v` during this process.

## Persistence checks

Offline: `python -m pytest tests/unit/test_deployment_persistence.py tests/unit/test_clean_startup.py -q`.

Explicit Docker integration (uses only a disposable test directory):

```powershell
docker build -f docker/Dockerfile.backend -t policy-rag-persistence-test .
$env:RAG_CONTAINER_TEST_IMAGE = 'policy-rag-persistence-test'
python -m pytest tests/integration/test_container_library_recreation.py -q
```

The integration test creates two separate containers with networking disabled.
It ingests using deterministic test embeddings in the first and checks original
bytes, document IDs, vector rows, and restored BM25 chunks in the second.

## Deletion correction

Persistent deletion errors now return an incomplete-cleanup error. A small atomic,
fsynced deletion journal revokes dense, lexical, cached-candidate, reused-evidence,
and asset access while keeping source files for cleanup retry. Restart preserves
revocation. Active indexing returns a conflict rather than racing deletion.
Compensating cleanup after failed ingestion does not revoke indexing retries.
This journal is single-process containment, not the proposed versioned catalog.
Requests already generating when deletion starts may still complete.

Validation: 55 tests passed across deletion integrity, legacy deletion, ingestion
hardening, document APIs, retrieval invalidation, and deduplication.

## Canonical answer correction

Pipeline completion text and citations now replace provisional deltas in the SSE
completion, telemetry, server history, structured conversation state, and browser
message. Empty final text is authoritative. Drafts are labeled in the active UI;
truncated streams and SSE errors fail rather than silently completing, and a
partially received answer is not automatically retried. Existing event names and
chunk delivery remain. This does not change the verifier acceptance policy.

Validation: 31 backend tests passed (canonical delivery, cancellation, model
switching, conversation scenarios); 220 frontend tests passed; TypeScript passed.
Two old frontend expectations were updated: server errors now reject the stream,
and an empty final answer clears draft text.

## Embedding and persistence correction

Provider results must have exact cardinality, finite numeric values, nonzero
norms, and consistent dimensions before caching or indexing. Synthetic hash
vectors now require the explicit test-only constructor option. The offline smoke
uses that option; ordinary serving does not. HF instances are keyed by model and
device. API-created vector stores require persistence and fail startup clearly
on initialization failure. New API indexes acquire a small embedding-identity
manifest; known incompatible models are rejected.

Existing nonempty indexes without provenance are preserved unchanged. Dense
queries and additional indexing into them require verified provenance; lexical
retrieval remains available. No model identity is guessed, no index is replaced,
and no automatic reindex occurs. Model revision/digest pinning and a verified
legacy adoption tool remain follow-up work.

Validation: 69 targeted tests passed across embedding integrity, persistent
startup, ingestion, deletion, retrieval quality, and the offline retrieval smoke.

## Local security containment

Compose publishes the frontend and API only on 127.0.0.1; Redis is internal.
Explicit API_ALLOWED_ORIGINS and API_ALLOWED_HOSTS replace wildcard ingress.
Untrusted browser origins are rejected before routing, including simple requests.
Administrative routes, model switching, bulk deduplication, and bulk session
deletion require a trusted socket peer, or ADMIN_TOKEN when configured. Source
runs trust loopback only. Compose explicitly trusts private container peers
behind its loopback-only published ports. This is a single-host boundary, not
multi-user authentication. Do not expose these containers through a public proxy
without an operator key and an appropriate authenticated ingress.

ADMIN_TOKEN is optional for the default host-only setup. API clients can send an
Authorization: Bearer header. The browser client supports setAdminToken(), or the
operator can set sessionStorage key rag_admin_token for the current browser tab.
Tokens are never embedded in NEXT_PUBLIC build settings or sent to ordinary chat
or document requests. Tests now use an explicit loopback socket peer.

Validation: 44 security, deployment, administrative, storage, and model API tests
passed; all 221 frontend tests and TypeScript passed. A model-selection test
that assumed an installed Ollama model now injects its model inventory.

## Model artifact correction

`src/finetuning/merger.py:merge_lora_weights` now requires the real HF/PEFT
operation. It stages output, checks exported metadata and weight/tokenizer files,
and publishes only after successful validation. Exporter metadata is retained;
the previous synthetic fallback and hard-coded replacement config are removed.
Existing model output directories are refused rather than overwritten.

`src/finetuning/gguf_exporter.py:convert_to_gguf` requires llama.cpp tools, stages
conversion/quantization, and checks the result before publishing. Missing tools,
converter errors, missing output and invalid headers raise failures. The
`allow_simulation=True` behavior is intentionally removed: that option now raises.
Synthetic artifact builders live only in `tests/fixtures/model_artifacts.py` and
are used through explicitly requested test fixtures. GGUF header checks are not
proof that a model can load or produce valid inference.

Both export CLI scripts return nonzero when requested registration or verification
fails. REST registration requires explicit final success; an HTTP 200, empty body,
progress message or error body is insufficient. Verification requires the exact
requested tag, with only the implicit `:latest` alias accepted. An unrelated tag
of the same model no longer counts as success. Training-only imports are lazy so
export-only operations do not require the optional dataset package.

Regression coverage includes failed merge with no published output, preserved
metadata and existing artifacts, unavailable converter, successful exit without
output, malformed registration responses, wrong tags, and nonzero CLI failures.
The final combined core, including artifact and exact-tag regressions, passed
603 tests with one optional-dataset skip. Frontend tests passed 221/221 and
TypeScript passed. Full commands and limitations are in ENGINEERING_HANDOFF.md.

## Compatibility and practical limits

Document IDs, original bytes, external citation routes, SSE event names and the
local FastAPI/Next.js/Chroma/BM25/Ollama architecture remain. Existing user data
was not migrated or reindexed. No runtime dependencies were added.

Intentional behavior changes: deletion can return 409/503 until cleanup completes;
interrupted SSE fails and remains marked interrupted; invalid embeddings and
required-storage initialization failures now raise; unknown legacy embedding
provenance disables dense use until verified; remote administration needs the
configured trust boundary; model output overwrite and simulation are refused.

The deletion journal is single-process containment. Multi-process consistency,
snapshot-bound caches and already-generating answers need the proposed catalog
and generation work. A filename deletion marker can block legacy data reusing
that filename; the catalog migration must reconcile identity explicitly.
Legacy root Dockerfile/release routes were not redesigned. Docker recreation and
real model/GPU/llama.cpp inference validation remain separate operator checks.

See ARCHITECTURAL_IMPROVEMENT_PROPOSAL.md for the larger changes that are proposed
only, including migration, rollback, compatibility and regression requirements.
