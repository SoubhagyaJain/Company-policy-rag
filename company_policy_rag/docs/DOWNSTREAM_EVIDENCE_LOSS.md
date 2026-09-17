# Downstream evidence loss: does the reranker's gain reach the LLM?

Run date: 2026-09-17. Follows [RERANKER_ABLATION.md](RERANKER_ABLATION.md), same 70 judged queries
(33 guidebook, 37 legal), same harness (`scripts/eval_retrieval_backend.py`, `RAGPipeline.run_retrieval_stages`),
standard response mode, `MIN_CHUNK_WORDS=5` for every A/B/C config. Cross-encoder scores computed on GPU
and cached; latency uses measured per-pair cost (CPU base 158.8 ms, large 542.1 ms; GPU 11.1 / 34.9 ms).
The evaluation pipeline now holds each session's chunks in its docstore, as a live session does
(legacy configs reproduce the previous run's numbers exactly).

## Two measured losses and their causes

### 1. Scope drop ("the guidebook" with no active document)

- `DocumentScopeResolver._CURRENT_DOC_PATTERNS` (`backend/rag/scope_resolver.py:46`) matches
  "the guidebook", "the doc", "this document", "the manual", ...
- With no `active_document_id`/name, step 5 (`elif has_doc_ref:`, `scope_resolver.py:338`, legacy branch
  now at :367) returns `CURRENT_DOCUMENT` with `active_document_id=None` and `allowed_document_ids=[]`.
- `_stage_scope_and_rewrite` adds no document filter (`pipeline.py:1658`), retrieval returns candidates,
  then hard-scope enforcement in `_stage_retrieve` (`pipeline.py:2070`) keeps a chunk only if it matches the
  active id or name. Both are `None`, so every candidate is rejected and counted as cross-document.
- 5/33 guidebook queries returned zero evidence.

### 2. Context assembly replaces the ranked hand-off

- `pipeline.py:2235` calls `governing_clause_selector.select(query, reranked_chunks, candidate_pool=candidate_chunks)`:
  the pool is the whole merged candidate list (up to 40 legal chunks after policy sub-query expansion).
- `_specificity_score` (`policy_reliability.py:281`) scores every pool chunk. Retrieval/reranker evidence
  enters only as `1.5 * tanh(score / 5)` (:288): at most 0.30 points for a sigmoid cross-encoder score,
  about 0.01 for an RRF score. Query-term overlap is worth up to 3.0, any normative word
  ("shall", "must", "entitled", "will be") +2.0, any condition word ("if", "when", "where", "before") +0.75.
- Roles then set the order (`select` :489-504, `order_for_context` :534): primary, up to 3 "exceptions"
  (any "however" / "unless" / "subject to" within 7 points of the top), up to 2 "definitions"
  ("means" / "includes"), then "supporting", truncated to 6. Legal prose is full of those words, so pool
  chunks fill every slot.
- Legacy code then did `if selected_context: reranked_chunks = selected_context` (full replacement).
- `pack_to_token_budget` (`context_compression.py:259`, 1,500 tokens in standard mode) fits 3-5 legal
  chunks of 150-350 words, cutting ranked chunks that survived selection in slots 5-6.

Diagnosis on tuned + large (legacy): legal relevant evidence in hand-off 37/37 -> after selection 32/37 ->
after token budget 29/37; guidebook 1 query lost at selection (the selector returned only 3 chunks).

## Fixes (flags, defaults preserve legacy behavior)

| Flag | Values | Change |
|---|---|---|
| `SCOPE_UNBOUND_REFERENCE_MODE` | `strict` (default, legacy) / `resolve` | An unbound reference binds to the only indexed document, or to the one whose filename contains the named noun ("guidebook", "handbook", "manual", "report", "paper"). Otherwise the query searches globally instead of returning nothing. An active document or filename match still wins. |
| `CONTEXT_ASSEMBLY_MODE` | `governing` (default, legacy) / `rank_anchor` | `merge_governing_context`: the selector's primary rule stays first, the top `CONTEXT_RANK_ANCHOR_K` ranked hand-off chunks follow, the selector's other picks fill the remaining slots. |
| `CONTEXT_RANK_ANCHOR_K` | int, default 2 | Anchor depth. Chosen before the run: the reranker's per-query wins moved the first relevant chunk to rank 1-2. |

New stage records on `RAGTrace.retrieval_stages`: `governing_selection`, `governing_roles`,
`post_governing`, `post_packing` (plus the existing `post_rerank`, `final_context`).

## Results (ALL, 70 queries)

Ranking = reranker order over its pool then unscored candidates; hand-off = rerank-stage output;
context = chunks sent to the LLM. Context MRR/nDCG are over the context order. Coverage = graded gain in
the context over the ideal for that many slots. E2E = retrieval through context formatting, no generation.

| Config | Rank MRR | Rank nDCG@10 | Hand-off Hit@6 | Context Hit@6 | Context MRR | Context nDCG@10 | Coverage | Relevant chunks to LLM | Rel. discarded / handed off | #1 discarded (relevant) | Lost all evidence | E2E p50 CPU | E2E p50 GPU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P current production (no min words, base, filter, legacy) | 0.869 | 0.746 | 0.929 | 0.800 | 0.580 | 0.454 | 0.563 | 125 | 40/136 | 11 (8) | 9 | 2,145 ms | 227 ms |
| A legacy, no reranker | 0.832 | 0.733 | 0.929 | 0.800 | 0.572 | 0.443 | 0.562 | 124 | 59/166 | 22 (15) | 9 | 76 ms | 76 ms |
| B legacy, base | 0.867 | 0.754 | 0.929 | 0.800 | 0.611 | 0.473 | 0.580 | 129 | 60/174 | 12 (8) | 9 | 2,157 ms | 238 ms |
| C legacy, large | 0.881 | 0.766 | 0.929 | 0.800 | 0.621 | 0.478 | 0.585 | 130 | 68/182 | 9 (7) | 9 | 7,153 ms | 541 ms |
| **A fixed, no reranker** | 0.893 | 0.790 | 1.000 | **0.986** | 0.700 | 0.597 | **0.739** | **159** | 41/182 | 0 (0) | 1 | **81 ms** | 81 ms |
| B fixed, base | 0.920 | 0.804 | 1.000 | 0.986 | 0.738 | 0.596 | 0.708 | 154 | 52/191 | 0 (0) | 1 | 2,188 ms | 237 ms |
| C fixed, large | 0.952 | 0.829 | 1.000 | 0.986 | 0.745 | 0.619 | 0.734 | 159 | 58/199 | 0 (0) | 1 | 7,256 ms | 564 ms |
| B fixed, base, filter (keep 3) | 0.920 | 0.804 | 0.986 | 0.986 | 0.731 | 0.593 | 0.706 | 153 | 26/147 | 0 (0) | 1 | 2,298 ms | 271 ms |
| C fixed, large, filter (keep 3) | 0.952 | 0.829 | 1.000 | 0.986 | 0.745 | 0.621 | 0.730 | 158 | 23/148 | 0 (0) | 1 | 7,252 ms | 571 ms |

Stage funnel, Hit@6 / MRR / nDCG@10:

| Config | Hand-off | Governing selection | Packing | Final context |
|---|---|---|---|---|
| C legacy, large | 0.929 / 0.881 / 0.730 | 0.843 / 0.629 / 0.503 | 0.843 / 0.629 / 0.502 | 0.800 / 0.621 / 0.478 |
| C fixed, large | 1.000 / 0.952 / 0.787 | 0.986 / 0.745 / 0.629 | 0.986 / 0.745 / 0.628 | 0.986 / 0.745 / 0.619 |
| A legacy, no reranker | 0.929 / 0.832 / 0.688 | 0.843 / 0.580 / 0.472 | 0.843 / 0.580 / 0.472 | 0.800 / 0.572 / 0.443 |
| A fixed, no reranker | 1.000 / 0.893 / 0.737 | 0.986 / 0.700 / 0.603 | 0.986 / 0.700 / 0.603 | 0.986 / 0.700 / 0.597 |

The remaining hand-off -> context MRR drop after the fix is mostly by design: the selector's primary rule
keeps slot 1, so a rank-1 chunk usually lands at slot 2.

### Paired comparisons (95% bootstrap CI, sign-flip p, W/L/T)

Fixes, same reranker:

| Comparison | Context Hit@6 | Context MRR | Context nDCG@10 | Coverage |
|---|---|---|---|---|
| A fixed vs A legacy | +0.186 [+0.100, +0.286], p<0.001, 13/0 | +0.129 [+0.076, +0.188], 23/2 | +0.154 [+0.105, +0.205], 48/10 | +0.177 [+0.108, +0.252], 27/3 |
| B fixed vs B legacy | +0.186 [+0.100, +0.286], 13/0 | +0.127 [+0.073, +0.189], 22/1 | +0.122 [+0.077, +0.169], 43/8 | +0.129 [+0.066, +0.195], 21/4 |
| C fixed vs C legacy | +0.186 [+0.100, +0.286], 13/0 | +0.125 [+0.074, +0.182], 21/0 | +0.141 [+0.095, +0.191], 45/8 | +0.149 [+0.086, +0.217], 23/3 |
| Scope fix only (C) | +0.057 [+0.014, +0.114], 4/0 | +0.043 [+0.007, +0.093] | +0.030 [+0.002, +0.064] | +0.039 [+0.003, +0.084] |
| Context fix only (C) | +0.114 [+0.043, +0.186], p=0.007, 8/0 | +0.075 [+0.039, +0.113] | +0.102 [+0.065, +0.142] | +0.101 [+0.049, +0.159] |
| A fixed vs P current production | +0.186 [+0.100, +0.286], 13/0 | +0.120 [+0.050, +0.195], 25/5 | +0.144 [+0.088, +0.203], 47/12 | +0.176 [+0.105, +0.251], 27/3 |

Reranker, same downstream:

| Comparison | Ranking MRR | Context Hit@6 | Context MRR | Context nDCG@10 | Coverage | Final chunk set differs / relevant set differs |
|---|---|---|---|---|---|---|
| Base vs none (legacy) | +0.035 [-0.023, +0.093], p=0.27 | 0 (0/0) | +0.039 [-0.006, +0.087] | +0.030 [+0.002, +0.063] | +0.018 [+0.003, +0.035] | 9/70, 6/70 |
| Large vs none (legacy) | +0.049 [-0.002, +0.102], p=0.08 | 0 (0/0) | +0.049 [+0.011, +0.093], p=0.06 | +0.035 [+0.009, +0.066] | +0.023 [+0.007, +0.042] | 12/70, 8/70 |
| **Base vs none (fixed)** | +0.027 [-0.032, +0.087], p=0.40 | 0 (1/1) | +0.038 [-0.007, +0.086], p=0.15 | -0.002 [-0.041, +0.037] | -0.031 [-0.071, +0.009] | 43/70, 27/70 |
| **Large vs none (fixed)** | +0.060 [+0.005, +0.117], p=0.04 | 0 (0/0) | +0.045 [+0.012, +0.085], p=0.03 (7/1) | +0.022 [-0.015, +0.059] | -0.005 [-0.043, +0.032] (12/12) | 48/70, 32/70 |
| **Large vs base (fixed)** | +0.033 [-0.012, +0.082], p=0.22 | 0 (1/1) | +0.007 [-0.019, +0.036], p=0.72 | +0.024 [+0.002, +0.049], p=0.04 | +0.026 [-0.000, +0.057] | 30/70, 18/70 |
| Score filter on vs off (base, fixed) | 0 | 0 | -0.007 | -0.003 | -0.002 | 2/70, 1/70 |

About 170 paired tests are in this summary; at alpha 0.05 roughly 8 false positives are expected. The fix
effects (p<0.001, 0-5 losses) are far outside that; the reranker's context-level effects are not.

### Anchor depth sensitivity (not used to choose the default)

| Config | Context Hit@6 | Context nDCG@10 | Coverage | Lost all evidence |
|---|---|---|---|---|
| none, k=1 / 2 / 3 | 0.971 / 0.986 / 1.000 | 0.560 / 0.597 / 0.602 | 0.692 / 0.739 / 0.746 | 2 / 1 / 0 |
| large, k=1 / 2 / 3 | 0.971 / 0.986 / 1.000 | 0.579 / 0.619 / 0.643 | 0.691 / 0.734 / 0.760 | 2 / 1 / 0 |

k=3 vs k=2: large context nDCG +0.024 [+0.010, +0.040]; no reranker +0.004 [-0.008, +0.017]. More anchors
leave fewer slots for governing-clause exceptions and definitions; this eval has no company-policy queries,
which is what the selector was built for, so validate k=3 on policy QA before adopting it.

## Per-query

Fixes (C fixed vs C legacy), context Hit@6 wins 13, losses 0:
- scope: `agent_building_blocks_count`, `code_available_links`, `code_check_this_out`, `critic_planner_mention`, `multi_agent_orchestration`
- context assembly: `legal_appellate`, `legal_dictatorship_paraphrase`, `legal_legislation_meaning`, `legal_marriage_age_paraphrase`, `legal_sessions_court`, `legal_socialist_word`, `legal_sociological_paraphrase`, `legal_us_congress`

Context MRR 21 wins / 0 losses (large); no reranker 23 / 2 (`memory_block` 0.50->0.33, `legal_double_jeopardy_paraphrase` 0.25->0.20).
Coverage regressions from the fix (large): `currency_tool_example` 0.67->0.56, `legal_equal_pay` 0.43->0.29, `legal_fir_mandatory` 0.60->0.40.
Still lost after the fix: `reflection_pattern` (no reranker and large; relevant chunks at hand-off ranks 3-4, selector returns 3 chunks).

Reranker after the fix, context stage:
- Large vs none, context MRR wins 7: `agent_vs_llm_vs_rag` 0.50->1.00, `guardrails_examples` 0.50->1.00, `react_pattern` 0.33->1.00, `legal_double_jeopardy_paraphrase` 0.20->1.00, `legal_law_reform_need` 0.50->1.00, `memory_block` 0.33->0.50, `workflow_retrieval_step` 0.33->0.50; loss 1: `legal_us_congress` 0.50->0.33. Coverage 12 wins / 12 losses (e.g. `planning_block` 0.75->0.25, `legal_ambedkar_role` 0.80->0.40).
- Base vs none: context Hit win `reflection_pattern`, loss `legal_marriage_age_paraphrase`; context MRR 9 wins / 5 losses; coverage 9 / 15.

## Verdict

- The downstream stages were costing more than the reranker adds. Legacy large: hand-off -> context
  Hit@6 -0.129, MRR -0.260, nDCG@10 -0.252; 68 of 182 handed-off relevant chunks discarded; 9 queries lost
  all evidence; 5 more had none because of scope. The two flagged fixes raise context Hit@6 from 0.800 to
  0.986 with no added latency (p50 76 -> 81 ms).
- After the fixes the reranker changes which chunks reach the LLM (48/70 queries for large) but not how
  much relevant evidence arrives: context Hit@6 is 0.986 with or without it, and relevant chunks delivered
  are 159 (none), 154 (base), 159 (large). Large moves the first relevant chunk earlier (context MRR +0.045).
- Base does not justify +2.1 s CPU / +156 ms GPU. Large over base: context nDCG +0.024, nothing else
  distinguishable, at 3.4x the rerank cost.

## Reproduce

```bash
python scripts/eval_retrieval_backend.py run --corpus guidebook=storage/eval_corpora/guidebook/bm25/corpus.json --corpus legal=storage/eval_corpora/legal/bm25/corpus.json --queries guidebook=data/eval/retrieval/guidebook_labels.json --queries legal=data/eval/retrieval/legal_labels.json --suite downstream --score-device cuda --out logs/retrieval_eval/downstream
python scripts/eval_retrieval_backend.py summarize --results logs/retrieval_eval/downstream --labels guidebook=data/eval/retrieval/guidebook_labels.json --labels legal=data/eval/retrieval/legal_labels.json --compare C_fixed_large:C_legacy_large,C_fixed_large:A_fixed_norerank,B_fixed_base:A_fixed_norerank,C_fixed_large:B_fixed_base --per-query-context A_legacy_norerank,A_fixed_norerank,C_legacy_large,C_fixed_large
```
