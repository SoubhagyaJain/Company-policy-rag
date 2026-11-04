# Real-model RAG validation — 4 November 2026

This is an in-progress checkpoint. Docker is excluded. The application is being
tested against real installed Ollama models, not deterministic embedding/model
fixtures. Do not treat the end-to-end answer quality as an all-green result.

## Runtime and preservation

- UI: `http://127.0.0.1:3000`; FastAPI: `http://127.0.0.1:8000`.
- Models: `qwen3.5:9b` and `qwen3-embedding:0.6b`.
- Document: the actual repository runbook `docs/CATALOG_AND_RECOVERY.md`.
- Validation document ID: `doc_692ccb38b20f`.
- Original SHA-256: `7a81ddf4c0a34e588dd056931f7e6777bc3e3fd7529853c049fba8eb794c7149`.
- Persistent validation library, runtime logs, source backup and JSON evidence:
  `logs/live_rag_20261104_100039/`.
- The existing `app/storage` library was copied and hashed before testing.
  Validation uses its own library and telemetry paths. No existing index was
  adopted, replaced or reindexed. No model or dependency was downloaded.

The browser upload returned HTTP 201 with TEXT_INDEXING. Real embeddings took
13.64 seconds; ingestion completed in 13.87 seconds. The catalog committed READY
with 10 expected chunks; Chroma contains 10 vectors of dimension 1024, BM25 has
10 chunks and the source hash matches. These are measured results for this file,
not throughput or production-scale claims.

Both actual model runners loaded. Redis is unavailable and uses the existing
in-memory fallback. Optional Qwen3-VL weights are incomplete, so scanned-image
and vision RAG are not validated. Text ingestion/retrieval remains available.

## Observed browser answers

| Case | Result | Evidence |
|---|---|---|
| Default lease and automatic attempt limit | Facts 120 and 3 were correct; answer marked UNVERIFIED | `tr_5573181834a5`, 43.13 seconds, 251 completion tokens. Extra lifecycle claim cited the table rather than the supporting prose. |
| Precise table-only follow-up | Correct values and source; completeness false negative | `tr_7f3067d2367a`, 31.82 seconds, 166 completion tokens. Completeness 0.333 despite no missing-aspect/citation errors; query formatting instructions counted against lexical coverage. |
| Ordinary lease-lifetime question | Correct primary fact; extra prose had citation mismatches | Third trace saved in `traces_after_three_questions.json`; UI showed UNVERIFIED with citation diagnostics. |

The browser source drawer successfully opened the configuration table. It
contains both defaults. A sentence that also describes exhaustion/failure needs
the recovery-prose citation as well; the table alone does not support that whole
sentence. The verifier correctly flags those citation mismatches.

The completeness false negative needs a separate correction that separates
semantic requirements from presentation instructions and handles technical
identifiers. Lowering thresholds or disabling verification is not a valid fix.
The broader verified/partial/abstained delivery policy remains a planned phase.

## Remaining checks at this checkpoint

Normal prose answer, unsupported-information behavior, restart persistence,
browser history reload, final screenshots and final service health are pending.
The runtime was restarted after the initial foreground processes stopped between
user turns. `scripts/start_live_rag_validation.py` now starts explicit hidden,
detached processes with saved logs and refuses to kill an existing server.

To restart this saved validation runtime when both ports are free, from the
application directory:

```powershell
.\.venv\Scripts\python.exe scripts/start_live_rag_validation.py
```

It uses `logs/live_rag_latest.txt`; process IDs and the active library path are
saved in that run's `runtime_state.json`. These service settings are process-local
and do not overwrite the user's `.env`.
