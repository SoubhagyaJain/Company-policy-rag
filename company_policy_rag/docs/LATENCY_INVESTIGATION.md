# RAG latency investigation and benchmark

The reported 82.86 s total, 4.72 s retrieval and 78 s generation are an observation, not a trace in this repository. The difference (0.14 s) does not establish what happened inside generation. `llm_synthesis` measures wall time around prompt construction, inference, and postprocessing. It is not the model's decode time. The answer may involve retries; inspect all suffixed stage timings. Ollama's load, prefill and decode durations are model-reported nested durations, not additional stages to add to synthesis wall time. The captured LLM usage currently describes the selected attempt, not the sum across all retries or the verifier's LLM calls.

The backend creates a prompt from the source-fidelity template, response-depth instructions, optional policy/fidelity material, formatted evidence and the resolved question. `context_tokens` is an estimate of evidence, while Ollama's `prompt_tokens` includes the full templated prompt. The server may truncate a prompt exceeding its context window. Generation can use a bounded thinking call and follow-up, with counts combined. `load_ms` is Ollama's reported load duration, not proof of a cold load. `first_text_ms` starts when the streaming iterator is consumed and ends at the first visible content delta; it includes any hidden thinking. The API's TTFT is from request start to first callback/event, and does not measure delivery to the browser. Non-streamed calls cannot measure first-token latency. High-risk requests with retries are buffered for verification; ordinary live streams can show text before the verifier finishes, without retrying an already shown answer.

Cross-encoder calls run synchronously, including first model initialization, but the default `ENABLE_RERANKER` is false; reused evidence also bypasses them. A missing local model or prediction error falls back to candidate scores. The thinking event called evidence verification is an instantaneous milestone, not the verifier. Evidence gating occurs earlier; actual answer verification is synchronous after generation under `verification`, and can include a separate LLM judge for high-risk answers. Timing 0.00 ms does not prove execution. `stage_status` distinguishes disabled, reuse, no candidates, fallback, and executed stages. An LLM judge being requested does not prove it succeeded. Stages absent from the trace did not reach instrumentation (e.g. bypass/cache/cancellation); do not infer zero cost.

The live snapshot on 2026-11-05 showed qwen3.5:9b at 45% CPU / 55% GPU in `ollama ps`, RTX 4050 laptop GPU 6,141 MiB total and about 5,000 MiB used. This is a point-in-time observation, not evidence of the configuration at the time of the 82.86 s report. CPU offload is a plausible decode bottleneck, not a measured attribution.

## Reproduction

Use the project's virtual environment, start Ollama with the configured models, and run these commands from `company_policy_rag`. Keep background GPU activity, model residency, cache state, and settings stable. This harness uses the production backend pipeline with a separate indexed snapshot and disables retrieval/semantic caches and vision. Each configuration uses standard response instructions, a fixed 15-candidate retrieval limit and fixed configured context/output budgets, varying only the maximum evidence count. The existing output-fit guard can reduce the actual output cap as the prompt grows; inspect `llm_usage.num_predict`. Actual final counts can be lower due to corpus size, filters, expansion, or the token budget. The shipped handbook snapshot contains only seven chunks, so top-10 is a cap, not ten guaranteed chunks. For a meaningful top-10 experiment use a larger labelled corpus in the same format. `--warmup` performs one unrecorded query per process; record cold starts separately. Run multiple independent repetitions in separate output directories and alternate their order to reduce warm-cache and time-order bias.

```powershell
foreach ($k in 3,5,10) {
  .\.venv\Scripts\python.exe scripts/eval_answers_backend.py run --corpus handbook=data/eval/retrieval/handbook_corpus.json --queries handbook=data/eval/retrieval/handbook_labels.json --config "top$k" --context-chunks $k --out logs/latency_chunks --warmup
}
.\.venv\Scripts\python.exe scripts/eval_answers_backend.py summarize --results logs/latency_chunks --compare top5:top3,top10:top3 --spot-check logs/latency_chunks/review.csv
```

Add `--stream` to measure the first pipeline callback; streaming changes retry and buffering decisions, so compare like with like. The default is the non-streaming query path. Environment JSON records model, settings and before-run `ollama /api/ps`, plus a GPU memory snapshot. New rows include per-stage wall times, stage statuses, model token counts, estimated context, and model durations. `ollama_ps_after` records server-reported loaded model and VRAM where supported; memory and placement can change during a request, so snapshots do not measure peak RAM/VRAM. Collect OS/GPU peak memory and server logs separately when needed. Benchmark data may contain answers and evidence identifiers.

Keyword recall, labelled citation precision, citation-in-context and abstention accuracy are deterministic proxies, not grounded-answer accuracy or claim-level citation correctness. Use the generated human review CSV and inspect each claim against its cited source; the optional `--judge-model` adds a fallible faithfulness estimate, costs additional time, and should be checked against human judgments. Report missing citation denominators and sample counts, and do not treat unavailable metrics as zero. The harness indexes a fixed corpus and its timing excludes indexing and the optional judge. Its calls do not include browser/network SSE overhead. `--force` appends duplicate rows; use a fresh output directory for repetitions.

## Candidate improvements, after measurement

| Priority | Candidate | Expected impact | Cost | Correctness risk |
|---|---|---|---|---|
| 1 | Investigate CPU/GPU split, competing VRAM users and server placement using per-token decode rate and memory snapshots | High if decode is slow from offload | Low to medium | Low if model and parameters stay fixed; validate placement |
| 2 | Compare evidence caps with matched prompts and human-checked claims | Medium if prefill/context dominates | Low | Medium to high: missing evidence and citations |
| 3 | Review output length and thinking budget against accuracy and TTFT | High if decode dominates | Medium | High: shorter or less reasoned answers can omit facts |
| 4 | Measure warm residency and model-load overhead before changing keep-alive | High for cold requests only | Low | Low correctness, potentially high memory cost |
| 5 | Optimize reranker only when enabled and its measured cost matters | Low when disabled | Medium | Medium if scoring or candidate pool changes |

Do not change model, retrieval, verification, or generation settings until the paired measurements and correctness review support the change.

## Local smoke measurements (2026-11-05)

Two labelled questions, one sample each, qwen3.5:9b Q4_K_M, non-streaming, model already resident. Top-3 had a warmup; the following two configurations started warm but had no separate warmup. Results are illustrative only, not a statistically reliable comparison. The source snapshot contains seven chunks.

| Cap | Actual chunks | Latencies (owner, leave) | Keyword recall | Mean labelled citation precision |
|---|---|---|---|---|
| 3 | 3, 3 | 32.09 s, 26.97 s | 100%, 100% | 100% |
| 5 | 5, 5 | 36.32 s, 28.98 s | 100%, 100% | 75% |
| 10 | 7, 7 | 29.15 s, 25.13 s | 100%, 100% | 100% |

All six had the required labelled chunk in context and all cited chunks came from that context. The top-5 owner answer also cited the security section for an ancillary claim; the evaluation marks that chunk irrelevant to the question, which is not the same as proving the ancillary claim false. No independent human claim-level accuracy assessment was performed. These two answerable questions cannot assess abstention accuracy, corpus-wide groundedness, or reliable latency percentiles. Prompt counts ranged from 2,203 to 3,787. Reported decode throughput ranged from 9.46 to 10.71 tokens/s; decode durations were 20.72–28.97 s, prefill 0.89–3.47 s. In the two top-3 traces reranking was disabled and answer verification executed in 0.92–1.03 ms. These times apply to these traces, not the historical 82.86 s report. The reported prompt and decode durations may combine a bounded thinking turn and a follow-up.

At a separate snapshot while these runs were in progress, physical memory was 15.71 GiB total and 2.12 GiB free. This is not a request peak. The raw local results and environment snapshots are in `logs/latency_chunks/`; generated files may be ignored by Git.

A separate warm streaming top-3 owner query reached the first pipeline callback at 17,991 ms from query start and the first visible LLM delta at 15,580 ms from iterator start; retrieval took 2,400 ms. Ollama reported 1,372 ms prefill, 24,421 ms decoding, 265 completion tokens and 10.85 tokens/s. Total query wall time was 30,678 ms. This measures in-process callbacks, not browser delivery. The model's bounded thinking turn may delay visible text; these timings alone do not apportion the gap precisely. The raw sample is in `logs/latency_stream/`.
