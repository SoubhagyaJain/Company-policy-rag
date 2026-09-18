# Reranker ablation: is the cross-encoder earning its latency?

Run date: 2026-09-17. Backend-native: every number below comes from `RAGPipeline.run_retrieval_stages`
(routing, multi-query, dense + BM25 + RRF, reranking, governing-clause selection, context packing),
not from the legacy `src/` LlamaIndex stack that `scripts/evaluate.py` measures.

## Question

After first-stage retrieval is tuned, does the reranker still provide a statistically meaningful
ranking improvement worth its latency?

## Setup

| Item | Value |
|---|---|
| Corpora | AI Agents guidebook (235 chunks), Legal Studies XI textbook (799 chunks); corpus snapshots under `storage/eval_corpora/`, production bge-small vectors loaded into Chroma |
| Queries | 70 with relevant chunks: 33 guidebook (from `golden_dataset_guidebook.json`), 37 legal (written for this eval from sampled passages; 23 lexical, 14 paraphrase) |
| Labels (primary) | Graded chunk-level judgments (2 direct, 1 partial) over pooled candidates: union of top-10 dense, BM25, merged candidates, base- and large-reranked lists at depth 30, and stemmed BM25; 1,702 chunks judged. LLM-assisted (Claude), blind to system and rank, not human-verified. |
| Labels (secondary) | Keyword needles matching <= 10% of the corpus (lexical-biased) |
| Response mode | standard (dense/BM25 depth 8, top 6 handed on) |
| Controlled | retrieval cache, semantic cache, vision fallback, graph expansion, LLM multi-query all off |
| Stats | paired bootstrap 95% CI (10k resamples), two-sided sign-flip test (20k), per-query W/L/T |
| Latency | cross-encoder cost measured uncached on the production device (CPU): base 158.8 ms/pair, large 542.1 ms/pair; GPU (RTX 4050) 11.1 / 34.9 ms/pair. Scores for ranking were computed on GPU and cached. |

Metric stages: **ranking** = reranker order over its pool followed by unscored candidates;
**hand-off** = what the rerank stage passes on (after score filter and top-n);
**context** = final packed context sent to the LLM.

## Headline (ALL, 70 queries, graded labels)

| Config | Hit@6 | MRR | nDCG@10 | R@20 | Hand-off chunks | Context Hit@6 | Rerank p50 (CPU) |
|---|---|---|---|---|---|---|---|
| A current, no reranker | 0.929 | 0.801 | 0.712 | 0.825 | 5.6 | 0.800 | 0 ms |
| B current + base, filter off | 0.929 | 0.869 | 0.746 | 0.826 | 5.5 | 0.800 | 2,065 ms |
| C current + base, filter on (min_keep 3) | 0.929 | 0.869 | 0.746 | 0.826 | 3.4 | 0.800 | 2,065 ms |
| C0 current + base, legacy filter (min_keep 1) | 0.929 | 0.869 | 0.746 | 0.826 | 2.5 (hand-off Hit@6 0.886) | 0.800 | 2,065 ms |
| D current + large, filter off | 0.929 | 0.864 | 0.752 | 0.825 | 5.5 | 0.800 | 7,048 ms |
| E tuned (skip chunks <= 5 words), no reranker | 0.929 | 0.832 | 0.733 | 0.839 | 5.6 | 0.800 | 0 ms |
| F tuned + base, filter off | 0.929 | 0.867 | 0.754 | 0.841 | 5.5 | 0.800 | 2,065 ms |
| F tuned + large, filter off | 0.929 | 0.881 | 0.766 | 0.839 | 5.5 | 0.800 | 7,048 ms |

Retrieval without reranking: p50 about 65 ms.

### Paired deltas

| Comparison | MRR delta [95% CI], p | nDCG@10 delta [95% CI], p | Hit@6 / hand-off / context |
|---|---|---|---|
| B base vs A | +0.068 [+0.008, +0.131], 0.035 (W/L 14/5) | +0.034 [-0.003, +0.072], 0.083 | 0 / 0 / 0 |
| D large vs A | +0.063 [+0.004, +0.126], 0.055 (12/4) | +0.040 [+0.006, +0.076], 0.027 | 0 / 0 / 0 |
| E tuned vs A | +0.031 [+0.007, +0.062], 0.029 (6/0) | +0.021 [+0.007, +0.039], 0.002 | 0 / 0 / 0 |
| **F base vs E** | +0.035 [-0.023, +0.093], 0.269 (11/6) | +0.021 [-0.019, +0.061], 0.314 | 0 / 0 / 0 |
| **F large vs E** | +0.049 [-0.002, +0.102], 0.079 (10/3) | +0.032 [-0.005, +0.070], 0.095 | 0 / 0 / 0 |
| C0 legacy filter vs A | n/a | n/a | hand-off Hit@6 -0.043 [-0.100, 0] (0/3) |

Guidebook only: base MRR gain was +0.116 [+0.030, +0.212] before tuning and +0.066 [0.000, +0.141]
after; skipping heading-only chunks alone gave +0.066 [+0.020, +0.126]. Legal only: base +0.007,
large +0.052 [-0.027, +0.133] after tuning. Keyword labels agree on direction with smaller effects
(tuned large vs E: nDCG +0.034 [+0.014, +0.056], MRR +0.021 [-0.025, +0.072]).

About 40 paired tests were run; at alpha 0.05 roughly two false positives are expected.

## Per-query outcomes (tuned + large vs tuned, first relevant chunk)

| Outcome | Queries |
|---|---|
| Correct chunk never reached the reranker | 5 (all guidebook; scope resolver dropped every candidate, see below) |
| Relevant chunk retrieved but outside the scored pool | 0 |
| Fixed (moved into the top 6) | 0 |
| Harmed (pushed out of the top 6) | 0 |
| Improved inside the top 6 | 10: tools_block 2->1, memory_block 2->1, workflow_retrieval_step 2->1, pattern_plan_execute 2->1, legal_article12 2->1, legal_admin_vs_constitutional 2->1, legal_civil_damages_paraphrase 6->1, legal_double_jeopardy_paraphrase 4->2, legal_marriage_age_paraphrase 2->1, legal_property_types 2->1 |
| Worsened inside the top 6 | 3: mcp_tools 1->2, legal_us_congress 1->2, legal_equal_pay 1->3 |
| Unchanged | 52 |

## Sweeps (tuned first stage, ALL)

| Knob | Result |
|---|---|
| Depth 4 / 8 / 15 / 30 (no reranker) | R@20 0.688 / 0.839 / 0.886 / 0.871; context Hit@6 0.843 / 0.800 / 0.729 / 0.686. More depth feeds more distractors into context assembly. |
| Reranker gain by depth (large, MRR) | +0.067 [+0.015, +0.125] at 4; +0.049 at 8; +0.045 at 15; +0.048 at 30 |
| Reranker pool at depth 30 (base MRR) | 20: 0.864, 40: 0.854, all: 0.838. Large: 0.873 / 0.865 / 0.856. Scoring more candidates hurts. |
| Score filter off / 0.25 / 0.45 | Identical ranking; hand-off shrinks 5.5 -> 3.7 -> 3.3 chunks; no hand-off Hit@6 loss once RERANK_MIN_KEEP=3 is honored |
| RRF k 10 / 30 / 60 | No measurable change |
| BM25 k1/b, stemming, metadata fields | Component-level best (k1 2.0, b 0.5, section fields) +0.017 mean BM25 nDCG@10; neutral to negative in the full pipeline |
| Sub-query merge by RRF | Neutral (nDCG -0.002) |
| Depth mode "max" (router depth) | Context Hit@6 -0.071 [-0.143, -0.014] |
| Skip chunks <= 5 words | MRR +0.031 [+0.007, +0.062], nDCG +0.021 [+0.007, +0.039]; the only first-stage change kept |

## Findings outside the reranker (measured here; findings 1-2 fixed behind flags in [DOWNSTREAM_EVIDENCE_LOSS.md](DOWNSTREAM_EVIDENCE_LOSS.md))

1. **Document-scope drop.** Queries that mention "the guidebook"/"the doc" with no active document get
   `CURRENT_DOCUMENT` scope with no id (`backend/rag/scope_resolver.py` step 5), and scope enforcement in
   `_stage_retrieve` removes every candidate. 5/33 guidebook queries return zero evidence. The frontend
   sends `active_document_id` only when a document filter is selected (`frontend/lib/api-client.ts`).
2. **Context assembly discards evidence.** Hand-off Hit@6 0.929 becomes context Hit@6 0.800 (legal
   1.000 -> 0.784). `governing_clause_selector.select(..., candidate_pool=candidate_chunks)` +
   `order_for_context` replace the reranked list with policy-clause picks from the whole pool; for 8/37
   legal queries the final context contains no relevant chunk while retrieval had one at rank 1. This is
   why no reranker configuration changes context Hit@6.
3. **Short chunks.** 34% of guidebook chunks and 13% of legal chunks have <= 5 words (headings, page
   furniture). Skipping them recovers about half of the reranker's pre-tuning MRR gain on the guidebook.
4. **Intermittent native crash.** The evaluation process segfaulted three times inside torch forward
   passes while embedding and reranking ran concurrently (resumed each time). Not reproduced in the app.

## Verdict

**Reranker conditionally useful.** It improves where the first relevant chunk sits inside the top 6
(MRR +0.049 for large, CI touching zero after tuning) and helps most on paraphrased questions
(legal_civil_damages 6->1), but at standard depth it never changed whether relevant evidence was handed
on or reached the LLM, and the downstream governing-clause step overrides its order. On CPU the cost is
about +2.1 s (base) / +7.0 s (large) per query for that gain.

## Reproduce

```bash
python scripts/eval_retrieval_backend.py run --corpus guidebook=storage/eval_corpora/guidebook/bm25/corpus.json --corpus legal=storage/eval_corpora/legal/bm25/corpus.json --queries guidebook=data/eval/retrieval/guidebook_labels.json --queries legal=data/eval/retrieval/legal_labels.json --suite headline --score-device cuda --out logs/retrieval_eval/current
python scripts/eval_retrieval_backend.py run ... --suite tuned --suite tuned_sweep --tuned-config logs/retrieval_eval/tuned/tuned_first_stage.json --out logs/retrieval_eval/tuned
python scripts/eval_retrieval_backend.py latency --results logs/retrieval_eval/pooling --corpus ... --queries ... --device cpu
python scripts/eval_retrieval_backend.py summarize --results logs/retrieval_eval/final --labels guidebook=data/eval/retrieval/guidebook_labels.json --labels legal=data/eval/retrieval/legal_labels.json
```
