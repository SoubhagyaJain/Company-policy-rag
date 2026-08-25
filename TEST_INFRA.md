# E2E Test Infra: Qwen 2.5 Coder 7B Fine-Tuning & Deployment Pipeline

## Test Philosophy
- Requirement-driven, opaque-box testing derived directly from `ORIGINAL_REQUEST.md`.
- Systematic 4-tier methodology:
  - **Tier 1 (Feature Coverage)**: >=5 test cases per feature covering happy-path and baseline execution.
  - **Tier 2 (Boundary & Corner Cases)**: >=5 test cases per feature covering edge cases, empty/malformed inputs, extreme hyperparameter values, missing files, out-of-bounds metrics.
  - **Tier 3 (Cross-Feature Combinations)**: Pairwise interaction testing (e.g. Alpaca dataset -> QLoRA -> Merge -> Q4_K_M GGUF -> Modelfile; ShareGPT -> LoRA -> FP16 GGUF -> Ollama API registration).
  - **Tier 4 (Real-World Application Scenarios)**: End-to-end integration workflows (complete data pipeline -> fine-tuning smoke-run -> export -> Ollama registration -> live RAG query execution & stream validation).

---

## Feature Inventory & Test Matrix
| # | Feature | Source (Requirement) | Tier 1 (Count) | Tier 2 (Count) | Tier 3 (Pairwise) | Tier 4 (E2E) |
|---|---------|---------------------|:--------------:|:--------------:|:-----------------:|:------------:|
| 1 | F1.1 Multi-format Dataset Loader | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 2 | F1.2 Validation Split & Hygiene | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 3 | F1.3 LoRA/QLoRA PEFT Config | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 4 | F1.4 Training & Metrics Logging | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 5 | F2.1 LoRA Adapter Merge | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 6 | F2.2 GGUF Quantization & Export | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 7 | F2.3 Ollama Modelfile Generation | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 8 | F2.4 Ollama Storage Registration | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 9 | F3.1 Environment & Config Defaults | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 10 | F3.2 Backend Dynamic Model Integration | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 11 | F3.3 Frontend Model Defaults | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |

---

## Test Architecture
- **Test Runner**: Pytest via `company_policy_rag\.venv\Scripts\pytest.exe company_policy_rag/tests/`
- **Unit Test Directory**: `company_policy_rag/tests/unit/`
- **E2E Test Directory**: `company_policy_rag/tests/e2e/`
- **Test Data Fixtures**: `company_policy_rag/tests/fixtures/` (sample Alpaca JSON, ShareGPT JSON, JSONL pairs, corrupt JSON, extreme hyperparameter configs).
- **Pass/Fail Semantics**: All test suites must return exit code 0 with 100% assertions passing.

---

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Expected Outcome |
|---|----------|--------------------|------------------|
| 1 | Full Pipeline Ingestion & Validation | F1.1, F1.2, F1.3 | Successfully loads and validates multiple format datasets and formats ChatML tokens |
| 2 | Smoke Fine-Tuning Run & Perplexity Log | F1.3, F1.4 | Runs training on miniature dataset, logs eval loss and calculates bounded perplexity |
| 3 | Standalone LoRA Weight Merge & Export | F2.1, F2.2 | Merges adapter weights into full model directory and validates safetensors structure |
| 4 | GGUF Quantization & Modelfile Generation | F2.2, F2.3 | Generates valid Modelfile with stop tokens and export bundle |
| 5 | Live Ollama Model Registration & Tag Probe | F2.4, F3.1, F3.2 | Programmatically registers model and verifies `ollama list` contains model tag |
| 6 | End-to-End RAG Chat Query Execution | F3.1, F3.2, F3.3 | RAG backend responds to user queries using the fine-tuned default model with streaming |

---

## Coverage Thresholds
- Tier 1: ≥ 5 test cases per feature (11 features × 5 = 55 test cases)
- Tier 2: ≥ 5 test cases per feature (11 features × 5 = 55 test cases)
- Tier 3: Pairwise coverage across major data and export pipelines (≥ 11 test cases)
- Tier 4: ≥ 6 comprehensive real-world scenarios
- **Total Minimum Target: ≥ 127 automated test cases**
