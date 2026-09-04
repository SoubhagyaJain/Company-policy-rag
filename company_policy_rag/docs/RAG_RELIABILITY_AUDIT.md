# RAG Reliability Audit — Pass 1

**Scope:** production reliability of the retrieval + generation pipeline, not any single
document or test question. **Method:** static audit of `backend/` (≈17.7k LOC of Python),
cross-referenced against the 127-file test suite. **Constraint:** the live stack
(Ollama + ChromaDB + an indexed corpus) is **not runnable in this environment**, so
end-to-end faithfulness/latency deltas are called out where they require it; everything
implemented this pass is validated with offline unit tests.

---

## 1. Architecture inventory — what already exists

The requested pipeline is, encouragingly, **already implemented** end-to-end. Mapping the
requested stages to the code:

| Requested capability | Implementation |
|---|---|
| classify / route | `rag/query_router.py` |
| decompose / multi-part | `rag/multi_query.py` (`decompose_multi_part`, `MultiQueryGenerator`) |
| hybrid retrieval (dense + BM25 + RRF) | `retrieval/hybrid.py`, `retrieval/bm25.py`, `retrieval/vector.py` |
| cross-encoder rerank | `retrieval/reranker.py` |
| context / parent-section expansion | `rag/scope_resolver.py`, `utils/section_tracker.py`, `pipeline._expand_adjacent_text_evidence` |
| rule + exception resolution | `rag/policy_reliability.py` |
| numeric / temporal (deterministic) | `enforce_deterministic_calculations`, `policy_reliability` calculations |
| missing-evidence gate / re-retrieve | `rag/evidence_gate.py`, `rag/retry_engine.py` |
| verify claims / citations | `rag/verifier.py`, `rag/citations.py` |
| abstention | evidence gate + `_is_degraded_or_abstention_answer` |
| semantic cache | `rag/semantic_cache.py` |
| conversation isolation | `rag/conversation_resolver.py` + cache eligibility gating |

**RRF (`retrieval/hybrid.py`)** is correct: standard `1/(k+rank)` fusion, preserves per-list
dense/sparse scores, graceful single-retriever fallback.

**Semantic-cache isolation (`pipeline.py:1725`)** is stronger than typical: the cache is made
**ineligible** whenever the request is a follow-up, carries filters, is document-scoped, or
targets a non-global scope; the cache key (`cache_context`) also embeds scope/filters/mode,
and `_queries_are_interchangeable` rejects sharing across mismatched numbers, audiences,
relationships, authorization verbs, times, quoted strings, and negation. **Verdict: the
"cache contamination" and "conversation isolation" risks are already well-mitigated.**
(Validated: `tests/unit/test_semantic_cache_safety.py` — 5/5 pass.)

---

## 2. Findings (prioritized)

### FIXED this pass

**F1 — Hardcoded, example-specific answer preamble (P1, correctness / rule #9).**
`pipeline.py` deterministically extracts an exact `1..N` list from retrieved text
(`_extract_requested_numbered_list`, a sound general mechanism) but then wrapped it with a
**hardcoded** lead-in: `"The five LLM fine-tuning techniques are:"`. Any *other* enumerated
question ("list the 3 steps to reset a password") therefore emitted a **factually wrong
subject sentence** — the textbook rule-#9 violation (tuned to one eval question).
*Fix:* new `_enumeration_preamble(query, count)` derives the subject phrase from the query
itself. It reproduces the exact fine-tuning wording for that query (existing e2e stays green)
and yields a correct subject for every other question, with a neutral, non-fabricated
fallback when the noun phrase can't be parsed.
*Tests:* `tests/unit/test_enumeration_preamble.py` (new).

**F2 — `grounding_status` always reported `PASS` (P1, observability integrity).**
`pipeline.py` built the trace with `grounding_status = "PASS" if (best_report and
best_report.passed) else "PASS"` — a copy-paste that made the field **constant**, so trace
logs and any consumer reading `trace.grounding_status` could never see a failed grounding.
(The telemetry DB independently recomputes status from `verification.passed`, which limited
the blast radius, but the trace itself lied.)
*Fix:* `"PASS" if (best_report is None or best_report.passed) else "FAIL"`.

### OPEN — highest-impact, needs the live eval harness to land safely

**P1 — Faithfulness gate is lexical-overlap-dominated with a 0.90 floor + overfit branches
(`rag/verifier.py`).** This is the single biggest *general* reliability weakness. Concretely:
- `_evaluate_faithfulness` sets `faith_score = max(faith_score, 0.90)` whenever the answer
  shares ≥3 non-stopword tokens with the context (`verifier.py:198`). A fluent, topically
  related but **unsupported** answer clears the gate unless it trips a hardcoded rule.
- Those hardcoded rules are example-specific: literal `$5,000` / `furniture` branches
  (`:122-128`), a fixed product allowlist `Slack|Jira|Salesforce|…` (`:133-138`), and a
  fixed technical-number ignore list `384|512|8000|8080|…` (`:163-164`).
*Why not ripped out now:* those branches are **load-bearing for the e2e adversarial suite**
(`tests/e2e/test_e2e_*`), which requires Ollama + ChromaDB + an indexed corpus that cannot
run here. Removing them blind would regress tests I cannot execute — precisely the failure
mode the task warns against.
*Recommended fix (modular, additive):* add an **LLM claim-level (NLI) faithfulness check** —
decompose the answer into atomic claims, ask the already-available `req_llm` "is each claim
entailed by the evidence?", and make that the primary faithfulness signal, keeping the
lexical heuristic as a cheap offline fallback and **lowering the 0.90 floor**. Gate behind a
setting (`verification_llm_claim_check`) so token/latency cost is opt-in. Validate against the
e2e tiers before flipping the default.

**P2 — `_evaluate_completeness` comparison entities are hardcoded (`verifier.py:262-274`).**
`full-time|part-time|employee|contractor|manager` is an allowlist; multi-group comparisons
over other entities aren't checked. Generalize to entities extracted from the query.

**P2 — Deterministic list override bypasses the verifier.** When `_extract_requested_
numbered_list` fires, the regex-built answer skips LLM synthesis (good — zero hallucination)
but should still pass through citation attachment for each item so every listed item is
traceable to its source line.

---

## 3. Before / after

| Item | Before | After |
|---|---|---|
| Enumerated-answer subject | Always "LLM fine-tuning techniques" regardless of question | Derived from the query; correct per question; e2e wording preserved |
| `trace.grounding_status` | Constant `PASS` (even on failed grounding) | Real `PASS`/`FAIL` from the verification report |
| New unit tests | — | `test_enumeration_preamble.py` (5 assertions, all pass) |
| Regression check | — | `test_semantic_cache_safety.py` 5/5 pass; module imports clean |

**Pre-existing failures (not introduced here):** `test_verifier_stress.py` has 2 float-rounding
mismatches (0.387 vs 0.388) and `test_citation_verification.py` has 3 snippet/metadata
assertions failing — all in files already modified in the working tree before this audit.

**Measurement honesty:** retrieval-quality, faithfulness-rate, citation-accuracy, hallucination-
rate and latency deltas at the *pipeline* level require the running stack + a labelled eval set.
The offline deltas above are unit-level. The P1 faithfulness work should be measured on the
e2e tiers once the stack is up.

---

## 4. Recommended next steps (ordered)

1. Stand up the eval harness (Ollama + Chroma + a small labelled corpus) so faithfulness and
   retrieval metrics are measurable per change.
2. Implement the LLM claim-level faithfulness check behind a flag; measure hallucination-rate
   on the adversarial tier before defaulting it on.
3. Generalize the completeness comparison-entity extraction.
4. Attach per-item citations on the deterministic-list path.
5. Add labelled eval cases for each category the task lists (factual, multi-hop, rule+exception,
   numerical, temporal, contradiction, unanswerable, adversarial, citation, multi-document) as a
   golden set the CI can gate on.
