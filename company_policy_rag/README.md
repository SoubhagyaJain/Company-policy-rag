# Grounded Policy RAG — application

**The full project documentation lives in the [repository README](../README.md).** That is the one to read: architecture, how to run it, measured results, design decisions, configuration, API reference and troubleshooting.

This file is a pointer so the application directory is not a dead end.

## This directory

```text
backend/    FastAPI app + the RAG core (production code)
frontend/   Next.js UI
scripts/    evaluation harnesses, CI gates, benchmarks, fine-tuning tools
tests/      unit · integration · e2e · adversarial
docs/       A/B log, evaluation baselines, audits, roadmap
data/eval/  labelled corpora, query labels, committed gate baselines
src/        shared Settings object + the first-generation pipeline (legacy)
```

## Run it

```bash
docker compose up --build      # whole stack, no .env and no host Ollama needed
```

From source, with Ollama running on the host:

```bash
python -m venv .venv && .venv/Scripts/Activate.ps1   # macOS/Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn backend.api.main:app --reload --port 8000
cd frontend && npm ci && npm run dev
```

Step-by-step instructions with a checkpoint after each step, plus the failures worth knowing about in advance, are in the [repository README](../README.md#-run-it-from-source-step-by-step).

## Checks

```bash
python scripts/run_core_tests.py                            # 430 backend tests, ~30 s
python scripts/benchmark_conversation.py --assert-minimums  # conversation gate
python scripts/ci_retrieval_smoke.py                        # retrieval gate (labelled fixture)
python scripts/production_retrieval_smoke.py --assert-minimums   # retrieval gate (self-contained)
cd frontend && npm test                                     # 216 frontend tests
```

Quality changes are decided by measurement; the evidence for each default is in [`docs/PHASE4_AB_LOG.md`](docs/PHASE4_AB_LOG.md) and [`docs/ANSWER_EVAL_BASELINE.md`](docs/ANSWER_EVAL_BASELINE.md).
