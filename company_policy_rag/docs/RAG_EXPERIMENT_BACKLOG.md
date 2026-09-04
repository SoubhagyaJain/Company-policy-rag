# RAG Experiment Backlog

## Operating rules

- Do not start quality tuning until the active-backend baseline and P0 correctness fixes exist.
- Use the frozen corpus/dataset/index/model/hardware manifest from `RAG_EVALUATION_PLAN.md`.
- Change one independent variable unless a factorial design is explicitly declared.
- Primary metric, minimum material effect, latency/memory guardrails, and sample size are registered before the run.
- **BASELINE REQUIRED** means the initial active-path run supplies the value and reviewers then set the bound.
- Rollback always means restoring the prior versioned config/index/model/cache generation; source documents remain untouched.

## Portfolio

| ID | Area | Priority | Dependency | Primary metric |
|---|---|---:|---|---|
| EXP-001 | chunk size/overlap | P1 | active evaluator | answer-span coverage + nDCG@10 |
| EXP-002 | structural chunking | P1 | labeled structure fixtures | context precision/recall |
| EXP-003 | selective OCR/layout | P2 | parse quality reports | evidence coverage by layout class |
| EXP-004 | BM25 tokenizer | P1 | retrieval lineage | MRR@10 on identifier/code queries |
| EXP-005 | dense query/document encoding | P1 | model revision pin | dense Recall@10 |
| EXP-006 | fusion policy | P1 | dense/BM25 ablation | fused nDCG@10 |
| EXP-007 | candidate depth | P1 | stage lineage | relevant survival / latency |
| EXP-008 | reranker span/cutoff | P1 | corrected score policy | reranked nDCG/context precision |
| EXP-009 | parent/adjacent expansion | P1 | evidence IDs | citation recall / context tokens |
| EXP-010 | context packing order | P1 | citation index fix | answer correctness |
| EXP-011 | query rewrite gating | P1 | route labels | retrieval gain / added latency |
| EXP-012 | multi-query generalization | P2 | hard-coded query removal | structural-query recall |
| EXP-013 | response-mode budgets | P2 | latency baseline | quality/TTFT frontier |
| EXP-014 | evidence gate/abstention | P1 | citation labels | risk-weighted assertion/refusal |
| EXP-015 | claim-level citation validation | P1 | claim dataset | citation F1 / faithfulness |
| EXP-016 | semantic-cache threshold | P1 | cache consistency fix | safe hit rate / false-hit rate |
| EXP-017 | model concurrency leases | P1 | cancellation/admission | p95 TTFT / throughput / memory |
| EXP-018 | cancellation propagation | P0 | instrumentation | cancel-stop latency/wasted tokens |
| EXP-019 | embedding model choice | P2 | earlier retrieval tuning | quality/latency/memory frontier |
| EXP-020 | generation model/prompt | P2 | retrieval/citation stable | correctness/faithfulness/TTFT |

## EXP-001 — Chunk size and overlap

- **Hypothesis:** a tokenizer-aware size/overlap grid improves answer-span coverage and retrieval ranking over the current hard-coded 512/64 policy.
- **Current limitation:** active construction conflicts with central 480/64 settings and uses approximate character/token conversion.
- **Change:** index identical corpus at sizes 256/384/512/768 tokens and overlaps 10%/15%/20%; no other policy change.
- **Success:** improve span coverage and nDCG@10 materially while keeping duplication, index size, ingestion time, and query p95 inside pre-registered bounds (**BASELINE REQUIRED**).
- **Failure:** quality is flat/worse, large duplication/storage increase, or critical classes regress.
- **Test scenario:** direct, cross-section, numeric, table, code, and long-section queries; report by document type.
- **Likely failure cause:** chunk length is not the bottleneck; labels/parser/query encoding dominate.
- **Stop condition:** any index inconsistency or two consecutive grid regions show no practical effect.
- **Rollback:** activate prior chunk-policy/index generation.

## EXP-002 — Structure-aware policy variants

- **Hypothesis:** heading/table/code-aware boundaries outperform generic semantic/recursive fallback on structured documents.
- **Current limitation:** strategy selection is heuristic and markdown-table-specific; no active benchmark.
- **Change:** compare current adaptive, recursive-only, heading-first, and document-class-specific policies at matched token budgets.
- **Success:** higher context precision/recall and citation attribution for relevant classes with no overall hard-class regression.
- **Failure:** fragmented prose, missing tables, excessive tiny chunks, or worse latency/storage.
- **Test scenario:** annotated PDF/DOCX/Markdown headings, lists, tables, code blocks, page continuations.
- **Likely failure cause:** parser structure labels are inaccurate or adaptive classifier chooses wrong content type.
- **Stop condition:** parser structure accuracy is below an agreed prerequisite; fix parsing first.
- **Rollback:** prior adaptive policy generation.

## EXP-003 — Selective OCR/layout extraction

- **Hypothesis:** OCR only on detected textless/low-text pages recovers answer evidence without unacceptable compute cost.
- **Current limitation:** PDF text extraction has no OCR; ingestion vision is cache-only.
- **Change:** compare no OCR, textless-only OCR, and low-confidence page OCR on representative scans/layouts.
- **Success:** improve page parse/evidence coverage and question recall; ingestion latency and memory stay within approved bounds.
- **Failure:** hallucinated OCR, layout-order corruption, excessive latency/VRAM, or no query-quality gain.
- **Test scenario:** scans, rotated pages, two columns, tables, stamps, normal text PDFs as negative controls.
- **Likely failure cause:** detector threshold or OCR model/layout reconstruction quality.
- **Stop condition:** corpus audit shows negligible answer-bearing scanned pages or OCR error exceeds benefit.
- **Rollback:** disable OCR inference while retaining textless detection/status.

## EXP-004 — Domain-aware BM25 tokenization

- **Hypothesis:** preserving version strings, hyphenated identifiers, RFC/code tokens, and function-like terms improves lexical retrieval.
- **Current limitation:** regex `[a-z0-9]+` splits domain identifiers and supports no phrase/field weighting.
- **Change:** compare current tokenizer with conservative domain tokenizer, optional stemming, and weighted metadata fields.
- **Success:** improve BM25 MRR@10 and hybrid nDCG on identifier/code/numeric classes without broad-query regression.
- **Failure:** stemming collisions, common metadata dominance, larger/slower index.
- **Test scenario:** `Qwen2.5-VL`, `RFC 7231`, policy IDs, dates, `randomNum(max,min)`, hyphenated terms.
- **Likely failure cause:** dense retrieval already covers cases or weighting overfits examples.
- **Stop condition:** no practical hybrid gain despite lexical-only improvement.
- **Rollback:** prior lexical snapshot/tokenizer version.

## EXP-005 — Query/document embedding treatment

- **Hypothesis:** model-recommended query/document prefixes or a retrieval-appropriate embedding revision improves dense recall.
- **Current limitation:** the same normalized `SentenceTransformer.encode` path is used for queries and documents; silent pseudo-embedding fallback exists.
- **Change:** compare current encoding, recommended asymmetric prefixes, and one pinned compatible model at the same vector dimension/top-K where possible.
- **Success:** improve dense Recall@10 and hybrid downstream score within latency/memory guardrails.
- **Failure:** corpus-class regression, model unavailable offline, index/memory cost, or no downstream gain.
- **Test scenario:** all retrieval classes with dense-only and hybrid ablations.
- **Likely failure cause:** chunking/labels/reranking dominate or new model mismatches domain.
- **Stop condition:** embedding load/resource target fails or hybrid gain is immaterial.
- **Rollback:** prior embedding/index generation; never use pseudo embeddings in a scored run.

## EXP-006 — Fusion policy

- **Hypothesis:** tuned RRF K or modest class-aware weighting improves fused ranking over fixed K=60.
- **Current limitation:** one fixed unweighted RRF policy is used for all query types.
- **Change:** compare K values and pre-registered weighted RRF variants; retain full rank contributions.
- **Success:** improve fused nDCG/MRR and relevant survival with no scope/correctness regression.
- **Failure:** overfits route classes, hurts rare lexical queries, or becomes unstable across splits.
- **Test scenario:** dense-dominant, lexical-dominant, mixed, numeric/identifier, ambiguous queries.
- **Likely failure cause:** route confidence is uncalibrated or candidate pools are too shallow.
- **Stop condition:** validation gain disappears on held-out split.
- **Rollback:** fixed RRF K=60 fingerprint.

## EXP-007 — Retrieval and rerank candidate depth

- **Hypothesis:** reranking more than the current `max(top_n*2, 8)` pool recovers evidence that fusion ranks slightly lower.
- **Current limitation:** relevant chunks can be irreversibly truncated before cross-encoder scoring.
- **Change:** candidate depths 8/12/20/30/50 at fixed final context N and batched inference.
- **Success:** improve relevant survival/reranked nDCG/citation recall for acceptable p95 and memory increase (**BASELINE REQUIRED**).
- **Failure:** latency dominates, no relevant recovery, or noisy deeper pool reduces precision.
- **Test scenario:** hard-negative, cross-document, broad structural, and multi-evidence questions.
- **Likely failure cause:** initial retrieval miss, not depth; or reranker text span/model weakness.
- **Stop condition:** marginal quality gain falls below registered minimum for two successive depths.
- **Rollback:** prior depth/top-N config.

## EXP-008 — Reranker text span and cutoff

- **Hypothesis:** scoring title/section plus a query-relevant or longer span and using top-N/calibrated cutoff outperforms first-350-character plus raw-logit ratio.
- **Current limitation:** late answer evidence is invisible; ratio math is unsafe for negative logits; configured batch size is unused.
- **Change:** factorial experiment on 350/700/token-bounded/query-window spans and top-N/absolute/calibrated cutoff; keep one primary interaction pre-registered.
- **Success:** higher nDCG/context precision and stage survival within p95/memory bound.
- **Failure:** longer spans dilute relevance or exceed compute; calibration does not transfer.
- **Test scenario:** answer spans early/middle/late, negative logits, long chunks, metadata-heavy headings.
- **Likely failure cause:** cross-encoder mismatch or poor candidate content.
- **Stop condition:** no held-out gain or p95 violates hard guardrail.
- **Rollback:** previous reranker config, but replace mathematically invalid ratio with safe top-N baseline.

## EXP-009 — Parent and adjacent-page expansion

- **Hypothesis:** selective structural/page expansion improves multi-sentence completeness and citation recall.
- **Current limitation:** expansion exists but prior evidence continuity is inactive and scope/provenance impact is not measured.
- **Change:** compare none, adjacent-page, parent, and relevance-gated combinations under equal context budgets.
- **Success:** improve required-fact/citation recall without lower precision, scope violations, or token inflation beyond bound.
- **Failure:** irrelevant context, wrong-document expansion, lost final evidence due to budget.
- **Test scenario:** page continuations, definitions with exceptions, tables with captions, cross-section synthesis.
- **Likely failure cause:** parent IDs/continuation cues are inaccurate.
- **Stop condition:** any scope violation; fix ownership checks before resuming.
- **Rollback:** disable expansion flags.

## EXP-010 — Context packing order

- **Hypothesis:** ordering by evidence utility and grouping complementary chunks improves answer correctness over raw rank order.
- **Current limitation:** current complementary packing is deterministic but not evaluated, and citation fallback ranks can diverge from final order.
- **Change:** compare rerank order, document/section grouping, diversity-aware, and required-fact-coverage packing at equal tokens.
- **Success:** higher correctness/completeness/citation F1 with no faithfulness or latency regression.
- **Failure:** model ignores late/high-value evidence, grouping amplifies one document, or citations mis-map.
- **Test scenario:** conflicting docs, multi-part questions, redundant chunks, answer evidence across sections.
- **Likely failure cause:** generation position bias or inaccurate utility/diversity features.
- **Stop condition:** citation index invariants fail or critical correctness regresses.
- **Rollback:** prior deterministic packer/final-source mapping.

## EXP-011 — Query rewrite gating

- **Hypothesis:** using rewrites only when a calibrated classifier predicts benefit reduces latency and avoids query drift.
- **Current limitation:** rewrite/expansion policy is heuristic and can add model work before retrieval/TTFT.
- **Change:** compare always-off, current, confidence-gated, and retrieval-feedback-gated rewrite; keep original query in the union.
- **Success:** equal/higher retrieval/answer metrics with lower p95/TTFT, or material quality gain within latency bound.
- **Failure:** missed follow-ups/acronyms, semantic drift, added model variability.
- **Test scenario:** standalone direct, ambiguous, acronym, follow-up, topic shift, already-specific queries.
- **Likely failure cause:** gating labels/classifier not representative.
- **Stop condition:** any critical class loses required evidence beyond tolerance.
- **Rollback:** current rewrite policy fingerprint.

## EXP-012 — Generalized multi-query planning

- **Hypothesis:** corpus/domain-neutral structural expansions outperform current guidebook-specific hard-coded subqueries.
- **Current limitation:** some multi-query templates encode one document's concepts.
- **Change:** compare current templates, neutral deterministic facets, and bounded model-generated subqueries, all deduplicated with original query.
- **Success:** improve structural/synthesis recall and answer completeness on held-out policy and guidebook sets.
- **Failure:** query explosion, drift, duplicate retrieval, latency increase, or non-guidebook regression.
- **Test scenario:** broad summaries, section navigation, cross-document comparison, unrelated policy domains.
- **Likely failure cause:** route-to-facet mapping is weak or labels too narrow.
- **Stop condition:** added queries do not add unique relevant evidence at acceptable cost.
- **Rollback:** disable multi-query or use neutral deterministic set.

## EXP-013 — Response-mode budgets

- **Hypothesis:** compact/standard/detailed candidate, context, and output budgets can be placed on a better quality-latency frontier.
- **Current limitation:** current 4/3, 8/6, 15/10 candidate/context choices and token budgets are hand-set.
- **Change:** small grid around each mode; keep prompt/model fixed.
- **Success:** user-intended completeness per mode with improved TTFT/total latency and no citation/faithfulness regression.
- **Failure:** modes become indistinguishable, compact omits required facts, detailed accumulates distractors.
- **Test scenario:** simple direct, list, synthesis, long explanation across all modes.
- **Likely failure cause:** output prompt rather than evidence budget drives behavior.
- **Stop condition:** no Pareto improvement over current point.
- **Rollback:** current response-mode table.

## EXP-014 — Evidence gate and abstention calibration

- **Hypothesis:** class-aware evidence thresholds reduce unsupported answers while maintaining useful answer rate.
- **Current limitation:** evidence/verification scores are heuristic and uncalibrated; strict/balanced behavior lacks an active baseline.
- **Change:** calibrate on answerable/no-answer sets; compare global versus numeric/quote/policy-obligation-specific thresholds.
- **Success:** lower risk-weighted false assertion with acceptable false-abstention; hard citation/scope gates remain perfect.
- **Failure:** excessive refusals, confident unsupported assertions, class transfer failure.
- **Test scenario:** supported, partial, conflicting, nonexistent, numeric, quote, and ambiguous questions.
- **Likely failure cause:** evidence score lacks predictive features or labels disagree.
- **Stop condition:** calibration curve is not stable on held-out data.
- **Rollback:** conservative explicit abstention policy; label score `uncalibrated`.

## EXP-015 — Claim-level citation validation

- **Hypothesis:** validating atomic claims against final evidence improves citation F1/faithfulness beyond tag presence and lexical overlap.
- **Current limitation:** current verifier can accept invalid indices and weak word overlap; citation fallback is not claim-aware.
- **Change:** compare deterministic claim/numeric/quote checks, calibrated NLI/judge, and hybrid validation; human labels are reference.
- **Success:** higher human-agreement citation precision/recall and fewer unsupported claims at acceptable latency.
- **Failure:** claim splitting errors, judge bias, latency, or over-refusal.
- **Test scenario:** multi-claim sentences, numbers/units, exceptions, quotes, code, paraphrases, unsupported extras.
- **Likely failure cause:** evidence spans too coarse or validation model domain mismatch.
- **Stop condition:** validator agreement with humans fails registered minimum or adds unacceptable latency.
- **Rollback:** strict deterministic index/numeric/quote validation plus explicit unavailable status.

## EXP-016 — Semantic-cache threshold and guards

- **Hypothesis:** the current 0.95 similarity plus critical-token guards can be tuned to increase safe hits without semantic collisions.
- **Current limitation:** no active false-hit baseline; policy fingerprint is incomplete and context lineage on hits is limited.
- **Change:** after consistency fixes, offline replay thresholds 0.90–0.99 and guard ablations; validate cited chunk existence/scope.
- **Success:** greater safe hit rate/latency savings with zero critical false hits and bounded general false-hit rate (**BASELINE REQUIRED**).
- **Failure:** audience/negation/number/version collisions, stale citations, or quality drift.
- **Test scenario:** minimal pairs differing in numbers, negation, department, document, response mode, policy version.
- **Likely failure cause:** embedding similarity is not a safe answer-equivalence signal.
- **Stop condition:** any high-risk false hit; raise threshold/disable affected class.
- **Rollback:** disable semantic cache or restore 0.95 guarded version.

## EXP-017 — Model concurrency and leases

- **Hypothesis:** a bounded concurrency/queue policy improves p95 and prevents memory collapse versus unbounded request threads.
- **Current limitation:** no text-model admission control or residency budget.
- **Change:** concurrency 1/2/3/4 with bounded queues/timeouts; profile one approved model and representative mixed load.
- **Success:** best throughput with p95 TTFT/total and RAM/VRAM inside approved SLO; predictable overload responses.
- **Failure:** queue amplification, OOM/model thrash, unfair long requests, reduced throughput.
- **Test scenario:** compact/standard/detailed mix, cache hits/misses, uploads/rerank/vision contention.
- **Likely failure cause:** hardware/model server serializes internally or model switching dominates.
- **Stop condition:** memory safety threshold or error-rate bound breached.
- **Rollback:** concurrency one, bounded queue, optional vision disabled.

## EXP-018 — Cooperative cancellation

- **Hypothesis:** propagating disconnect cancellation to Ollama reduces wasted generation and recovers leases quickly without corrupting successful streams.
- **Current limitation:** SSE stops while daemon worker/model generation continues.
- **Change:** pass token through chat service/pipeline/client; stop reading/close generation; compare current and cooperative paths.
- **Success:** cancel-stop p95 and wasted tokens/duration meet target **BASELINE REQUIRED**; no thread/lease growth and no completed-answer cache write.
- **Failure:** model continues remotely, resource leaks, race creates false success/partial cache.
- **Test scenario:** cancel before retrieval, during queue, first token, mid-generation, and after completion under load.
- **Likely failure cause:** Ollama client/server does not honor connection close or worker boundary masks token.
- **Stop condition:** cancellation introduces successful-request corruption; keep admission bounds and redesign integration.
- **Rollback:** prior stream transport with concurrency one/short output cap; retain cancellation waste metric.

## EXP-019 — Embedding model frontier

- **Hypothesis:** a pinned alternative embedding model yields a better quality/latency/memory point after pipeline defects are fixed.
- **Current limitation:** configured model names conflict and current quality is unmeasured; premature model changes would confound root causes.
- **Change:** compare current BGE-small with at most two locally feasible candidates using their prescribed encoding and separate indexes.
- **Success:** meaningful hybrid/downstream quality gain or equivalent quality with lower resource/latency.
- **Failure:** offline unavailability, excessive index/model cost, regression in numeric/code/policy classes.
- **Test scenario:** full retrieval suite, cold/warm embedding, ingestion throughput, memory profile.
- **Likely failure cause:** retrieval bottleneck is lexical/chunking/reranking rather than embedding model.
- **Stop condition:** no Pareto improvement.
- **Rollback:** current pinned BGE-small generation.

## EXP-020 — Generation model and grounded prompt

- **Hypothesis:** after evidence quality stabilizes, an explicit untrusted-evidence/claim-citation prompt or model variant improves faithfulness without unacceptable TTFT.
- **Current limitation:** model/prompt effects are confounded by retrieval and citation defects; model lifecycle cost is unmeasured.
- **Change:** compare current prompt, hardened evidence prompt, and at most one locally feasible generation model at matched evidence/parameters.
- **Success:** higher correctness/faithfulness/citation F1 and prompt-injection resistance inside latency/memory guardrails.
- **Failure:** prompt length/latency, excessive abstention, citation-format regression, model memory contention.
- **Test scenario:** core answers, no-answer, conflicting evidence, source injection, numbers/quotes/code, streaming.
- **Likely failure cause:** unsupported answers originate in missing evidence or validator, not generator.
- **Stop condition:** retrieval/citation stage is not stable, or no Pareto improvement.
- **Rollback:** prior prompt/model fingerprint and unload challenger.

## Promotion checklist

An experiment is promotable only when:

- raw results, manifest, code/config diff, and confidence intervals are retained;
- all hard safety/correctness gates pass;
- primary metric meets the pre-registered material effect on held-out data;
- no critical query/document class regresses beyond tolerance;
- warm/cold latency, throughput, and RAM/VRAM guardrails pass on named hardware;
- index/cache migration and rollback are rehearsed;
- the winning fingerprint is added to active settings and its regression case to CI.

Experiments that fail remain valuable records. Do not silently retune the hypothesis after results; register a follow-up experiment with the observed likely cause.
