# Conversation RAG benchmark

Generated: `2026-09-08T03:36:45.187338+00:00`

## Method

This benchmark changes only the conversation layer. The before and after systems share the same fictional handbook, production BM25 implementation, top-3 retrieval, answer prompt, and answer model. The before path uses the legacy query rewriter; the after path uses the production `ConversationInterpreter` and its retrieval policy.

Answers and grounding judgments were generated locally with `qwen2.5:7b` at temperature 0 and seed 42.
Run environment: `Windows-10-10.0.26200-SP0`, Python `3.11.9`; Intel Core i5-13420H, NVIDIA RTX 4050 Laptop GPU 6 GB; Ollama reported 39% CPU / 61% GPU model placement.

## Results

| Metric | Before | After |
|---|---:|---:|
| Retrieval hit@3 | 90.9% | 100.0% |
| Mean reciprocal rank | 86.4% | 95.5% |
| Retrieval-policy accuracy | 75.0% | 100.0% |
| Standalone-query term coverage | 86.4% | 100.0% |
| Citation correctness | 81.2% | 100.0% |
| Answers with only correct citations | 72.7% | 100.0% |
| Unsupported-claim rate | 7.9% | 7.1% |
| Logic + retrieval p50 | 0.23 ms | 0.43 ms |
| End-to-end p50 | 5957.29 ms | 4250.44 ms |

Retrieval accuracy is hit@3 over cases that require evidence. Citation correctness is the share of emitted citation tags that resolve to an expected source section. Unsupported-claim rate is the LLM-judged share of factual claims that the supplied sources do not entail. Latency values are from this machine and one run.

## Case-level evidence

| Case | Behavior | Before action / hit | After action / hit |
|---|---|---|---|
| `maternity_probation_pronoun` | pronoun_resolution | retrieve / hit | retrieve / hit |
| `travel_international_fragment` | incomplete_followup | retrieve / hit | retrieve / hit |
| `remote_eligibility_fragment` | incomplete_followup | retrieve / hit | retrieve / hit |
| `remote_contractors_followup` | nested_followup | retrieve / hit | retrieve / hit |
| `leave_to_vpn_topic_switch` | topic_switch | retrieve / hit | retrieve / hit |
| `return_to_leave_probation` | topic_return | retrieve / hit | retrieve / hit |
| `reuse_point_two` | evidence_reuse | retrieve / miss | reuse_previous / hit |
| `reuse_show_source` | evidence_reuse | retrieve / hit | reuse_previous / hit |
| `ambiguous_plural_reference` | ambiguity_clarification | retrieve / n/a | ask_clarification / n/a |
| `travel_comparison_multi_part` | comparison_multi_part | retrieve / hit | decompose / hit |
| `annual_leave_how_much` | incomplete_followup | retrieve / hit | retrieve / hit |
| `vpn_why_verification` | reason_followup | retrieve / hit | retrieve / hit |

## Reproduce

```bash
python scripts/benchmark_conversation.py --with-generation
```

The machine-readable output includes rewritten queries, retrieved section IDs, generated answers, citation mappings, judge counts, and per-case timings in `data/eval/conversation_benchmark_results.json`.

## Limits

- The corpus and cases are fictional and intentionally small; these scores do not predict every production corpus.
- Hallucination rate is an unsupported-claim proxy judged by the same local model family, not a human adjudication.
- Latency is hardware-dependent and represents one local run; logic/retrieval timings are more stable than generation timings.
