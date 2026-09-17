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
