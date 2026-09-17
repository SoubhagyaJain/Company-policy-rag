# Phase 4 quality A/B log

Each Phase 4 change runs through the retrieval harness before it ships. The harness goes from retrieval to the final prompt context, with no LLM unless noted, on three labelled corpora: guidebook (33 labelled queries), legal (37) and handbook (21).

Every run uses the same config: the current defaults (`min_chunk_words=5`, scope `resolve`, context `rank_anchor` k=2, no reranker). Raw results are in `logs/retrieval_eval/phase4/`, which is git-ignored.

```bash
python scripts/eval_retrieval_backend.py run --corpus guidebook=storage/eval_corpora/guidebook/bm25/corpus.json --corpus legal=storage/eval_corpora/legal/bm25/corpus.json --corpus handbook=data/eval/retrieval/handbook_corpus.json --queries guidebook=data/eval/retrieval/guidebook_labels.json --queries legal=data/eval/retrieval/legal_labels.json --queries handbook=data/eval/retrieval/handbook_labels.json --configs <config.json> --out logs/retrieval_eval/phase4
python scripts/eval_retrieval_backend.py summarize --results logs/retrieval_eval/phase4 --labels guidebook=data/eval/retrieval/guidebook_labels.json --labels legal=data/eval/retrieval/legal_labels.json --labels handbook=data/eval/retrieval/handbook_labels.json --compare <candidate>:<baseline>
```

## 4.3 Remove corpus-specific query heuristics (shipped, 2026-09-17)

### What was removed

- **`query_rewrite.py`:** two phrase → vocabulary tables. Their terms were appended to *every* retrieval query that contained a trigger phrase.
  - 11 policy entries written for a company-rules PDF that is not in any eval set: "sister" → "private electrical work", "benefits" → "health insurance … dental vision … 30 days", and so on.
  - 13 guidebook entries.
  - Also removed: `detect_corpus` (no readers), and the guidebook-only patterns in `is_comprehensive_list`.
- **`multi_query.py`:** seven guidebook sub-query tables in the deterministic fallback. The fallback keeps the query itself, its separate question parts and any "including X, Y" topics.
- **`verifier.py`:** the hard-coded "$5,000 / furniture" check. `$5,000` is still caught by the generic numeric check. An invented *non-numeric* entity ("furniture") is no longer caught; the test is kept as a strict `xfail` recording that gap.
- **`citations.py`:** the `kickoff` (CrewAI) cue for code evidence.

### Why

The guidebook expansions were keyed to the literal wording of the guidebook eval questions:
- 18 of 35 guidebook questions trigger one.
- 0 of 37 legal questions and 0 of 24 handbook questions do.

The appended text is the answer's vocabulary. "…building block…" appends `Role-playing Focus Tasks Tools Cooperation Guardrails Planning Memory six AI agents`, and "where does the guidebook point readers for full code examples" appends `code is available Check this code dailydoseofds link repository`. That is eval leakage, not retrieval quality, and it can only help a user who phrases a question like the eval set against that one PDF.

### Results

Current defaults, heuristic multi-query (as in the harness):

| Corpus | Config | Context Hit@6 | Context coverage | nDCG@10 (ctx) | Lost all evidence | Retrieval p50 ms |
|---|---|---|---|---|---|---|
| guidebook | before | 0.970 | 0.739 | 0.54 | 1 | 107 |
| guidebook | after | 0.939 | 0.662 | 0.49 | 1 | 94 |
| legal | before / after | 1.000 | 0.738 | 0.65 | 0 | 142 / 124 |
| handbook | before / after | 1.000 | 1.000 | 0.90 | 0 | 87 / 87 |

- **Guidebook, after − before:** context coverage −0.077 [−0.127, −0.032] (p = 0.003); context nDCG@10 −0.052 (p = 0.018); Context Hit@6 −0.030 (the `code_available_links` query).
- **Legal and handbook:** identical on every query.

### Attribution (guidebook only)

| Variant | Context coverage |
|---|---|
| before (expansions + tables) | 0.739 |
| old query rewriter only (expansions, no tables) | 0.745 |
| old sub-query tables only (tables, no expansions) | 0.662 |
| after (neither) | 0.662 |
| LLM multi-query on, before | 0.716 |
| LLM multi-query on, after | 0.639 |

The whole guidebook change comes from the appended expansion terms. The sub-query tables were neutral, both with the heuristic and with the production LLM decomposition.

**Decision.** The guidebook numbers above the "after" row were inflated by question-specific vocabulary, so the new guidebook baseline is coverage 0.662 / Context Hit@6 0.939. Recovering that gap has to come from generic mechanisms that are measured on the same harness:
- contextual chunk headers (7.2)
- embeddings (7.3)
- same-section expansion (7.1)

### Side finding for Phase 5.2

LLM multi-query did not raise guidebook coverage over the deterministic fallback, in either variant:
- with the old heuristics: 0.716 vs 0.739
- after the removal: 0.639 vs 0.662

It added about 20 s of LLM calls over 35 queries. `policy_reliability.expand_policy_queries` has the same kind of hand-written policy vocabulary and is ablated in 5.2.

## 4.4 Metadata accuracy: clause ids (shipped, 2026-09-17)

`clause_id` was taken from any line that starts with a number. List items, page headers ("98 | AI Agents") and quantities ("20 days") therefore became clause ids. Now `clause_id` comes from:
1. the parsed `section_number`,
2. otherwise a dotted number ("22.3"),
3. otherwise an explicit "Section N".

Nothing in retrieval reads `clause_id` or `parent_section`, and BM25 indexes `section_path, section_title, section_number, source_file, category`, so there is no retrieval A/B to run. Fixture test: `tests/unit/test_policy_reliability.py::test_bare_leading_numbers_are_not_clause_ids`.

Document-level `key_entities` / `topic_tags` are still copied onto every chunk. Only the metadata filter extractor reads them, and it is off by default. Revisit only if filtering is re-enabled.

## 5.2 (part 1) Remove the bag-of-words policy sub-queries (shipped, 2026-09-17)

`expand_policy_queries` added up to two sub-queries to *every* non-fast-path question on every corpus. Each was a sorted bag of "important concepts", e.g. "What role do Tools play in AI agents?" → `agents play role tools`, `role tools`. None of the 96 eval questions match a policy topic profile, so all it did was add BM25 and dense lookups.

Harness run on HEAD after 4.3/4.4, candidate minus baseline, 91 labelled queries:

| Metric | Delta | 95% CI | W/L/T |
|---|---|---|---|
| Context Hit@6 | +0.000 | [0.000, 0.000] | 0/0/91 |
| Context coverage | +0.000 | [−0.012, +0.012] | 4/4/83 |
| Context nDCG@10 | +0.004 | [−0.006, +0.013] | 6/3/82 |
| Ranking MRR | +0.011 | [0.000, +0.027] | 2/0/89 |

**Latency.** Retrieval p50 fell from 113 to 98 ms overall:

| Corpus | Before (ms) | After (ms) |
|---|---|---|
| guidebook | 94 | 86 |
| legal | 124 | 120 |
| handbook | 87 | 79 |

The call was neutral, so it was removed. The policy topic profiles (`_TOPIC_PROFILES`: private work, after-hours calls, smoking in vehicles, and so on) still drive the governing-clause selector and `is_policy_question`. They target a company-rules PDF that is not in the repository, so they cannot be A/B'd here; the corpus for `data/eval/policy_reliability_dataset.json` is missing.

## 4.2 Context assembly: keep the ranked hand-off (shipped, 2026-09-17)

`rank_anchor` (the previous default) builds the context in this order:
1. the governing-clause selector's primary rule,
2. the top 2 ranked chunks,
3. more selector picks.

The selector draws from the whole candidate pool and scores mostly lexically, so its picks pushed ranked chunks 3–6 out of the prompt. New modes in `merge_governing_context`:

| Mode | Context |
|---|---|
| `rank` | The ranked hand-off, unchanged. The selector only feeds the policy decision block. |
| `rank_rescue` | Ranked order. The selector's top 2 picks take the last slots when ranking left them out. |
| `rank_policy` | `rank_rescue` for workplace-policy questions (`is_policy_question`), `rank` otherwise. |

Harness results across 91 labelled queries (guidebook 33, legal 37, handbook 21):

| Mode | Final context Hit@6 / MRR / nDCG@10 | Relevant discarded | Lost all evidence | Coverage |
|---|---|---|---|---|
| rank_anchor k=2 (old default) | 0.98 / 0.74 / 0.65 | 41 | 1 | 0.771 |
| rank_anchor k=6 (primary + ranked) | 0.99 / 0.74 / 0.68 | 10 | 0 | 0.806 |
| rank | 0.99 / 0.93 / 0.79 | 4 | 0 | 0.816 |
| rank_rescue | 0.98 / 0.93 / 0.78 | 17 | 1 | 0.799 |
| **rank_policy (new default)** | **0.99 / 0.93 / 0.79** | **4** | **0** | **0.816** |

`rank_policy` compared with `rank_anchor` k=2 (paired):

| Scope | Context MRR | Context nDCG@10 | Context coverage |
|---|---|---|---|
| All 91 queries | +0.192 [+0.127, +0.255], p < 0.001 | +0.140 [+0.103, +0.178], p < 0.001 | +0.045 [+0.013, +0.076], p = 0.007 |
| Guidebook | | | +0.057 (0.667 → 0.724) |
| Legal | | | +0.059 |

On guidebook, `rank_policy` recovers most of the coverage that the 4.3 removal took away, without the leaked vocabulary.

**`rank_policy` vs `rank`.** They were identical on every query. The rescue fired on 10 policy questions (8 on the handbook), but the selector's picks were already in the ranked top 6. Always-on rescue (`rank_rescue`) cost coverage: −0.017 [−0.041, +0.006]. `rank_policy` keeps the governing-clause safety net for the case the selector exists for, a governing rule ranked below an unrelated one. A unit test covers that case: `tests/test_downstream_evidence_loss.py::test_rank_rescue_keeps_the_governing_clause_ranked_below_an_unrelated_rule`.

Latency for the `rank_rescue` and `rank_policy` runs (132–135 ms p50) is not comparable with the other rows. Those runs overlapped with a live answer evaluation on the same machine; the assembly change itself adds no model calls.

### Answer-level check (handbook, qwen2.5:7b, n = 24)

`rank` compared with `rank_anchor` k=2 (paired). `rank_policy` builds the same handbook contexts as `rank`.

| Metric | Delta | 95% CI | p | W/L/T |
|---|---|---|---|---|
| keyword recall | 0.000 | [0, 0] | 1.00 | 0/0/21 |
| tagged citation | 0.000 | [0, 0] | 1.00 | 0/0/21 |
| abstention correct | 0.000 | [0, 0] | 1.00 | 0/0/24 |
| citation precision | −0.125 | [−0.275, 0.000] | 0.25 | 0/3/17 |
| latency | −1191 ms | [−2050, −321] | 0.01 | 17/7/0 |

The three citation misses are the model citing the wrong number, not missing evidence. In each case the relevant chunk was `[Source 1]`, and qwen tagged `[Source 2]` or `[Source 3]`. `hb_access_review` even named the wrong section.

The same case (`hb_contractor_remote`) missed under `rank_anchor` in the baseline run. Citation precision for the unchanged `rank_anchor` config ranged from 0.925 to 1.000 across two runs, so this is within run-to-run noise. It is the failure 4.5 (citation support check) targets.

**Shipped:** `CONTEXT_ASSEMBLY_MODE=rank_policy` is the default. The CI retrieval gate baseline was rewritten: handbook context MRR is 1.000 (was 0.881), so a regression back to anchor-style assembly now fails the gate.

## 4.5 Citations: stop header echo with an inline-citation rule (shipped, 2026-09-17)

**Finding.** qwen2.5:7b copied the prompt's source header into the answer in 31 of 96 stored answers, e.g. `[Source 2] File: sample_employee_handbook.md | Section: 1. Parental and Maternity Leave | Page: 1 | Evidence Type: TEXT`. For many answers that line was the only citation, and several of the wrong citations in `ANSWER_EVAL_BASELINE.md` were such lines.

**Evaluated and not shipped:**
- **Lexical citation repair.** A sentence → cited-chunk support score that re-attributes a tag to the chunk that supports it. Run offline over 141 stored answers, it made 0 repairs at safe thresholds. On the real misattributions the model paraphrases: the tagged and correct chunks differ by ≤ 0.3 support (e.g. 0.2 vs 0.5), so any threshold low enough to repair them would also move correct tags. The module was deleted.
- **Stripping echoed header lines.** Also run offline. Citation precision rose (0.875 → 0.964 on the `rank` run), but tagged answers fell from 20 to 14, because the header line was the only tag.

**Live A/B.** Handbook, qwen2.5:7b, n = 24, current defaults:

| Variant | tagged citation | citation precision | clean format (no header echo) | keyword recall | latency p50 |
|---|---|---|---|---|---|
| control | 0.952 | 0.900 | 0.500 | 1.000 | 6180 ms |
| inline rule | 1.000 | 0.905 | **1.000** | 1.000 | 4722 ms |
| inline rule + `<source id=…>` blocks | 1.000 | 0.952 | 1.000 | 0.976* | 4547 ms |

\* One miss was a label bug: the keyword `annual` did not match "reviewed annually". The label now accepts `annual|annually|every year|yearly`.

**Inline rule vs control:**

| Metric | Delta | 95% CI | p | W/L/T |
|---|---|---|---|---|
| clean format | +0.500 | [0.292, 0.708] | 0.001 | 12/0/12 |
| latency | −1473 ms | [−2411, −736] | < 0.001 | 22/2/0 |
| citation precision | 0.000 | | | 1/1/18 |

The XML variant was not distinguishable from the inline rule at n = 24. The smaller change shipped: the header format is kept, and the prompt rule changes from "Cite supporting blocks with their exact [Source N] tags" to:

- Put the tag of the supporting source (`[Source N]` or `[Visual Source N]`, N = that source's number) right after each sentence it supports.
- Never copy source headers or metadata (file names, sections, pages, evidence types) into the answer.

`clean_format` is now an answer-eval metric. The remaining citation errors are the model choosing the wrong number for a correct sentence; `hb_contractor_remote` cites `[Source 2]` in every variant.

## 5.1 Conversation interpreter LLM pass off by default (shipped, 2026-09-17)

The pipeline runs `ConversationInterpreter` on every turn. It always makes a deterministic interpretation, which covers follow-up fragments, pronouns, "going back to", source/simplify reuse and ambiguity. With `ENABLE_CONVERSATION_INTERPRETER=true` (the previous default) it then also made one qwen2.5:7b call on every turn with history (512 max tokens).

**Run.** `scripts/benchmark_conversation.py`: 12 multi-turn cases, production BM25, top-3. The "improved" system's interpreter was given the local model.

| Interpreter | LLM calls | Unusable output | Policy accuracy | Hit@3 | MRR | Query term coverage | Logic p50 |
|---|---|---|---|---|---|---|---|
| deterministic only | 0 | – | 1.000 | 1.000 | 0.955 | 1.000 | 0.2 ms |
| + LLM pass (before) | 11 | 11 / 11 | 1.000 | 1.000 | 0.955 | 1.000 | 8561 ms |
| + LLM pass, parser fixed | 11 | 1 / 11 | 1.000 | 1.000 | 0.955 | 1.000 | 8275 ms |

**Before the fix, every output failed validation.** The prompt shows the schema as lists of allowed values, and the model answered `"intent": ["factual"]`. So the ~8.5 s call was always discarded. Parsing now unwraps one-item lists for scalar fields. Once parsed, the model largely restated the deterministic baseline, down to its rationale string, and changed no retrieval decision.

**Decision.** `ENABLE_CONVERSATION_INTERPRETER` defaults to `false`. The deterministic interpreter still runs on every turn; the flag only controls the extra LLM pass. Each follow-up turn saves roughly 8 s of generation on the RTX 4050. The benchmark is at ceiling (12 cases), so this is a latency decision with no measured quality cost, not proof the LLM pass can never help.
