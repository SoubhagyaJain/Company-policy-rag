# Recovery checkpoint

Branch: `codex/production-critical-corrections`.

The six requested correction areas are implemented; artifact integrity is still
being validated at this checkpoint. No existing user library was migrated,
deleted, replaced, or reindexed. Pre-existing edits to API startup logging,
`backend/vision/hf_vision_client.py`, `pyproject.toml`, and the user's untracked
notes/readiness test remain separate from these commits.

Validated checkpoints:

| Commit | Correction | Validation |
|---|---|---|
| fe72a1e | Correct Docker library mount, preserve current host paths | 6 offline tests; container test added |
| 3992edc | Retryable deletion and durable retrieval revocation | 55 targeted tests |
| 924e96e | Canonical final answer, citations, history, provisional UI | 31 backend and 220 frontend tests; TypeScript |
| f86d7c3 | Embedding validation, identity, required API persistence | 69 targeted tests; core suite 439 passed |
| 6484296 | Local network, origin/host and administrative containment | 44 targeted tests; 221 frontend tests; TypeScript |

The five new artifact-integrity regressions pass. Expanded artifact/registration
checks found seven failures: a negative CUDA index bug, five tests patching the
wrong imported `urlopen` symbol (and therefore contacting real Ollama), and one
dataset dry-run test requiring an uninstalled optional `datasets` package.
These are the immediate follow-up before marking artifact validation complete.

Docker validation is pending and explicitly excluded from further work at the
user's request. The engine was unavailable; no application containers were
recreated. See CRITICAL_CORRECTIONS.md for preservation and rollback procedures.

Remaining work:

1. Finish the expanded artifact tests without installing training dependencies.
2. Add the new deterministic regressions to the routine core test manifest.
3. Run combined backend checks, frontend tests, TypeScript, and source lint.
4. Write a separate proposal for the eight deferred architectural improvements.
5. Update this handoff with final results and report implemented versus proposed.

Useful commands (from the application directory):

```text
.venv/Scripts/python.exe -m pytest tests/unit/test_model_artifact_integrity.py tests/unit/test_export_merge_modelfile.py tests/unit/test_adversarial_merger_gguf.py tests/unit/test_ollama_registration.py -q
.venv/Scripts/python.exe scripts/run_core_tests.py
```

Frontend: `npm test` and `node node_modules/typescript/bin/tsc --noEmit --incremental false --pretty false`.
