# Answer evaluation baseline (2026-09-17)

The first answer-level evaluation of the shipped backend. It runs `RAGPipeline.query()` end to end: retrieval, context assembly, generation with the local LLM, and citations. It compares the new retrieval defaults with the legacy ones.

Tooling:
- `backend/evaluation/answer_eval.py` defines the metrics.
- `scripts/eval_answers_backend.py` runs and summarises the evaluation.

## Setup

| | |
|---|---|
| Corpus | `data/eval/retrieval/handbook_corpus.json`: 7 chunks from the sample employee handbook |
| Questions | `data/eval/retrieval/handbook_labels.json`: 24 cases. 21 are answerable (factual, policy interpretation, procedural, one multi-part); 3 must abstain (dental, 401(k), parking) |
| Model | `qwen2.5:7b` via Ollama; `num_ctx` 4096; temperature 0.1; RTX 4050 Laptop (6 GB) |
| Path | `query(response_mode="standard")`, non-streaming. Retrieval and semantic caches, vision and lazy vision are off. The reranker is off in both configs |
| `defaults` | Current defaults: `CONTEXT_ASSEMBLY_MODE=rank_anchor`, `SCOPE_UNBOUND_REFERENCE_MODE=resolve`, `MIN_CHUNK_WORDS=5`, `ENABLE_QUERY_METADATA_FILTERING=false` |
| `legacy_retrieval` | The pre-hardening values: `governing`, `strict`, `0`, `true` |

## Results

| Config | context hit | coverage | keyword recall | tagged citation | citation precision | citation in context | abstention correct | latency p50 / p95 ms | prompt tok p50 | overflow |
|---|---|---|---|---|---|---|---|---|---|---|
| defaults | 1.000 | 1.000 | 1.000 | 0.952 | 0.925 | 1.000 | 1.000 | 7372 / 13679 | 1263 | 0 |
| legacy_retrieval | 0.952 | 0.952 | 0.952 | 0.857 | 1.000 | 1.000 | 0.958 | 6541 / 13165 | 1263 | 0 |

**Paired comparison (defaults − legacy, n = 24):**

| Metric | Delta | 95% CI | p | W/L/T |
|---|---|---|---|---|
| context_hit | +0.048 | [0.000, 0.143] | 1.00 | 1/0/20 |
| keyword_recall | +0.048 | [0.000, 0.143] | 1.00 | 1/0/20 |
| tagged_citation | +0.095 | [0.000, 0.238] | 0.50 | 2/0/19 |
| citation_precision | −0.083 | [−0.222, 0.000] | 0.50 | 0/2/16 |
| abstention_correct | +0.042 | [0.000, 0.125] | 1.00 | 1/0/23 |
| latency_ms | +837 | [91, 1662] | 0.05 | 5/19/0 (a latency win means faster) |

**Reading the numbers:**
- **Quality.** No difference is significant at n = 24, and the handbook is near the ceiling for both configs. The single legacy loss matches the retrieval study (`docs/DOWNSTREAM_EVIDENCE_LOSS.md`). On `hb_document_owner`, the legacy context dropped `nsl_00` (document control), and the model answered "the handbook does not explicitly mention" the owner.
- **Latency.** The latency gap is output length, not configuration. Prompt tokens are identical (p50 1263), and median completion tokens are 113 vs 113.5. Per-case output varies between runs: `hb_annual_leave_days` produced 135 tokens under defaults and 64 under legacy. Generation dominates end-to-end time, at roughly 7 s for about 110 output tokens.
- **Ceiling.** This corpus cannot separate quality changes. Guidebook and legal need answer keywords before the Phase 4 gates can use answer metrics. Until then, those gates rely on the 70-query retrieval funnel plus this set as a regression check.

## Failures seen under the defaults

| Case | What happened | Follow-up |
|---|---|---|
| `hb_remote_days` | Correct answer. Its closing generic sentence also cites [Source 3] (security, `nsl_06`) → citation precision 0.5. | Phase 4.5 citation support check |
| `hb_contractor_remote` | Correct rule, but cited [Source 2], the parental-leave chunk `nsl_01`, instead of [Source 1], remote work. It also echoed a `[Source 2] File: … \| Section: …` header line. | Phase 4.5; the prompt source-header format (4.2) |
| `hb_multi_part` | Both facts are correct, with no `[Source N]` tags at all. Citation cards fall back to score threshold. | Phase 4.5 / prompt |
| 4 of 24 answers (`hb_remote_days`, `hb_intern_remote`, `hb_customer_data_ai`, `hb_abstain_401k`) | `fallback_reason=retry_exhausted_fallback`: the lexical verifier failed correct answers. With `VERIFICATION_MAX_RETRIES=0` this is only a trace flag; the answer is unchanged. | Verifier false negatives; see the Phase 4.5 support check |

Legacy additionally had one answer (`hb_access_review`) that tagged `**[Source 3: Sample Employee Handbook, …]**`. The citation parser did not recognise that tag, fixed in `c6aa8a7`. Stored rows keep the citations computed at run time, so that case still counts as untagged in the table above.

## Evaluator corrections made while reading these results

- **Abstention detection had false positives.** A negative policy fact ("contractors are *not covered* by the benefit") counted as an abstention, and so did a closing hedge ("the handbook does not provide any additional exceptions"). An abstention is now a statement *about the evidence* ("the handbook does not mention …", "not found in the documents") in the first two sentences. `summarize` re-derives abstention for stored runs, so the table above uses the corrected detector.
- **Latency W/L was inverted.** For `latency_ms`, a win now means the candidate is faster.

## Reproduce

```bash
python scripts/eval_answers_backend.py run --corpus handbook=data/eval/retrieval/handbook_corpus.json --queries handbook=data/eval/retrieval/handbook_labels.json --config defaults --out logs/answer_eval/baseline_2026-09-17
python scripts/eval_answers_backend.py run --corpus handbook=data/eval/retrieval/handbook_corpus.json --queries handbook=data/eval/retrieval/handbook_labels.json --config legacy_retrieval --set CONTEXT_ASSEMBLY_MODE=governing --set SCOPE_UNBOUND_REFERENCE_MODE=strict --set MIN_CHUNK_WORDS=0 --set ENABLE_QUERY_METADATA_FILTERING=true --out logs/answer_eval/baseline_2026-09-17
python scripts/eval_answers_backend.py summarize --results logs/answer_eval/baseline_2026-09-17 --compare defaults:legacy_retrieval --spot-check logs/answer_eval/baseline_2026-09-17/spot_check.csv
```

Each config takes about 3 minutes on the RTX 4050. `logs/` is git-ignored. A human spot-check sheet with empty judgment columns is written to `spot_check.csv`.
