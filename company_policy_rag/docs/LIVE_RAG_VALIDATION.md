# Real-model RAG validation — 4 November 2026

The application was started and tested against real installed Ollama models,
without deterministic embedding/model fixtures. Docker is excluded. Operational
text RAG passes; answer trustworthiness needs further work. This is not an
all-green answer-quality result, and no verification thresholds were weakened.

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
| Queued jobs and source files when executor is unavailable | Primary behavior correct; current verifier passed at 0.920 | Synchronous real-model response in `prose_live_response.json`; 45.80 seconds, 376 completion tokens. Manual review found a cleanup-verb error described below. |
| Employee dental-insurance reimbursement limit | Correctly refused to invent a limit absent from this document | `unsupported_live_response.json`; 25.38 seconds, current verifier passed at 0.947. Response still includes fallback source cards. |

The browser source drawer successfully opened the configuration table. It
contains both defaults. A sentence that also describes exhaustion/failure needs
the recovery-prose citation as well; the table alone does not support that whole
sentence. The verifier correctly flags those citation mismatches.

The completeness false negative needs a separate correction that separates
semantic requirements from presentation instructions and handles technical
identifiers. Lowering thresholds or disabling verification is not a valid fix.
The broader verified/partial/abstained delivery policy remains a planned phase.

Manual review also found a false positive: the prose answer says the system
"retains partial derived rows" while its source says it **removes** the affected
job's partial derived rows before replay. Its main conclusion about preserving
queued jobs and original files is correct. A 0.920 heuristic score did not catch
the reversed cleanup verb; passing that verifier is not semantic proof.

## Completed operational checks

| Boundary | Result and evidence |
|---|---|
| Browser upload to API | Real Markdown file selected through the file chooser; HTTP 201 then polled READY |
| Source to dense/BM25 indexes | 10 genuine 1024-dimensional vectors, 10 BM25 chunks, expected count 10 and matching source SHA-256 |
| Retrieval to model | Logs show actual dense + BM25 retrieval; completed traces contain real prompt/completion token counts for `qwen3.5:9b` |
| Model/SSE to browser | Three completed streamed answers, provisional-to-final UI transition, citation cards and source drawer observed |
| Browser history | Reload restored all three completed answers, their verification labels and citation cards |
| Backend restart | Same document ID, hash, READY state and 10 chunks restored; evidence in `restart_validation.json` |
| Existing library | All 8 original files still match their pre-test SHA-256 hashes |
| Final runtime | UI HTTP 200; API status ok with 10 chunks; both model runners present |

An additional browser prose stream was interrupted when the browser session
closed. The backend logged client cancellation, and no completed chat trace was
committed for that attempt. Its partial draft remained provisional after browser
reload; this is a further UI recovery issue, not a completed answer. The same
prose question was then completed through the live synchronous API. No parallel
model requests were intentionally introduced.

The first polling helper waited for the interrupted stream and timed out without
sending another model request. The second helper ran the two final synchronous
cases sequentially. These helper failures/interruption are retained in the raw
logs; they are not counted as successful model responses.

The runtime was restarted after the initial processes stopped between user
turns. `scripts/start_live_rag_validation.py` now starts explicit hidden,
detached Windows processes with saved logs and refuses to kill an existing
server. Only this operational launcher and the saved report were added; serving
and verifier code were not changed during this validation.

To restart this saved validation runtime when both ports are free, from the
application directory:

```powershell
.\.venv\Scripts\python.exe scripts/start_live_rag_validation.py
```

It uses `logs/live_rag_latest.txt`; process IDs and the active library path are
saved in that run's `runtime_state.json`. These service settings are process-local
and do not overwrite the user's `.env`.

## Saved evidence and remaining corrections

`validation_summary.json` records five completed query traces, model state,
final HTTP checks, the two final responses and preservation results. Additional
artifacts include `ingestion_status.json`, `persistence_snapshot.json`,
`restored_browser_history.txt`, `final_browser_library.txt`,
`library_ready_after_restart.jpg`, raw query records and backend/frontend logs.
They remain on disk in the validation run directory and are ignored by Git;
this report and the launcher are committed. Launcher Ruff checks passed. The
earlier offline core result remains 665 passed / 1 optional dependency skip;
it was not rerun because application source was unchanged in this phase.

The next corrections, supported by these observations, are:

1. Per-claim citation alignment and semantic verification that catches reversed
   actions such as retains/removes; evaluate on held-out grounded/contradicted
   answers before changing delivery gates.
2. Completeness scoring that separates presentation directives from requested
   facts and tokenizes technical identifiers without losing their identity.
3. Restored unfinished drafts must become interrupted rather than indefinitely
   remaining "awaiting final answer" after their request has disappeared.
4. Correct section metadata for technical Markdown: these chunks were all
   labeled Remaining scope despite coming from different sections.
5. Truthful telemetry/UI labels: the sidebar's 2.4M chunks and 98% grounding are
   static, and progress described cross-encoder scoring although the reranker was
   disabled. The Library view correctly shows the actual 1 document / 10 chunks.

The positive unsupported-information result is one case, not proof of universal
abstention reliability. Scanned PDFs/vision, other documents, other chat models,
load/concurrency and Docker remain untested in this live run.
