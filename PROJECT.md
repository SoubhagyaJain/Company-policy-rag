# Project: Qwen 2.5 Coder 7B Fine-Tuning, GGUF Export, Ollama Registration & System-Wide Integration

## Architecture
The system integrates an end-to-end parameter-efficient fine-tuning (PEFT/LoRA/QLoRA) and deployment pipeline into the existing FastAPI + ChromaDB + Ollama + Next.js RAG platform:

1. **Fine-Tuning Engine (`company_policy_rag/src/finetuning/`)**:
   - `dataset_loader.py`: Multi-format dataset ingestion (Alpaca, ShareGPT, JSONL prompt-response pairs), normalization to ChatML schema, and deterministic validation splitting.
   - `trainer.py`: LoRA/QLoRA training orchestrator using Hugging Face `transformers`, `peft` (`LoraConfig`), and `trl` (`SFTTrainer`) with 4-bit NF4 quantization, completion-only loss masking, and step-level loss/perplexity metrics logging.
   - `merger.py`: High-performance LoRA adapter weight merger (`peft` `merge_and_unload`) exporting 16-bit standalone weights and tokenizer configurations.
   - `gguf_exporter.py`: GGUF conversion and quantization engine supporting `Q4_K_M`, `Q8_0`, and `FP16` with multi-tier fallback for local/Windows environments.
   - `modelfile_generator.py`: Automated Ollama `Modelfile` generator configured with ChatML template, stop tokens (`<|im_end|>`, `<|endoftext|>`), parameters (`num_ctx 8192`, `temperature 0.1`), and enterprise policy system prompt.
   - `ollama_registrar.py`: Direct local Ollama storage registration via CLI (`ollama create`) and REST API (`POST /api/create`).

2. **CLI & Workflow Entrypoints (`company_policy_rag/scripts/`)**:
   - `finetune_qwen_coder.py`: CLI for dataset validation, fine-tuning configuration, metrics logging, and adapter export.
   - `export_and_register_ollama.py`: CLI for merging LoRA weights, GGUF export, Modelfile generation, and registering the model into Ollama.
   - `run_finetune_pipeline.py`: Unified end-to-end pipeline runner executing training -> merge -> GGUF -> Modelfile -> Ollama registration in a single command.

3. **System-Wide Default Configuration (`company_policy_rag/`)**:
   - `src/config.py`: Centralized Pydantic settings with `llm_model: str = "qwen2.5-coder-7b-policy"` (alias `OLLAMA_LLM_MODEL`), `metadata_extractor_model`, and `eval_llm_model`.
   - `backend/api/routes/models.py`, `backend/models/api_dto.py`, `backend/dependencies.py`: API default active model and dynamic Ollama model routing.
   - `.env`, `.env.example`, `docker-compose.yml`: Environment variables configuring default model `qwen2.5-coder-7b-policy`.
   - `frontend/components/ChatWindow.tsx`, `frontend/lib/api-client.ts`: UI model selection defaults.

4. **Automated Verification Suite (`company_policy_rag/tests/`)**:
   - `tests/unit/test_finetuning_dataset.py`: Unit tests for multi-format dataset loading, normalization, and validation splitting.
   - `tests/unit/test_finetuning_trainer.py`: Unit tests for LoRA trainer, loss masking, metrics logging, and smoke-run execution.
   - `tests/unit/test_export_merge_modelfile.py`: Unit tests for adapter merge, Modelfile generation, stop tokens, and GGUF exporter.
   - `tests/unit/test_ollama_registration.py`: Unit tests for Ollama registration utility.
   - `tests/e2e/test_finetuned_rag_e2e.py`: End-to-end integration test verifying live RAG chat generation, routing, self-reflection, and non-regression with the fine-tuned model.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | F1.1 Multi-format Dataset Loader | Support Alpaca, ShareGPT, and JSONL prompt-response pairs with auto-detection | M1 | ORIGINAL_REQUEST §R1 |
| 2 | F1.2 Validation Split & Hygiene | Deterministic train/validation splitting with seed control, format validation, length checks | M1 | ORIGINAL_REQUEST §R1 |
| 3 | F1.3 LoRA/QLoRA PEFT Architecture | 4-bit NF4 QLoRA, 8-bit LoRA, FP16 LoRA targeting all 7 linear projections (`q/k/v/o/gate/up/down_proj`) | M1 | ORIGINAL_REQUEST §R1 |
| 4 | F1.4 Training Execution & Metrics | Completion-only loss masking, step logging, eval loss, stable perplexity calculation, artifacts export | M1 | ORIGINAL_REQUEST §R1 |
| 5 | F2.1 LoRA Adapter Weight Merging | Merge LoRA adapter weights with base Qwen 2.5 Coder 7B into standalone FP16 weights | M2 | ORIGINAL_REQUEST §R2 |
| 6 | F2.2 GGUF Conversion & Quantization | Export model to GGUF format supporting Q4_K_M, Q8_0, and FP16 with local fallback ladder | M2 | ORIGINAL_REQUEST §R2 |
| 7 | F2.3 Ollama Modelfile Generation | Generate optimized Modelfile with ChatML template, stop tokens, system prompt, and parameters | M2 | ORIGINAL_REQUEST §R2 |
| 8 | F2.4 Ollama Registration Utility | Register fine-tuned GGUF directly into local Ollama storage via CLI/API | M2 | ORIGINAL_REQUEST §R2 |
| 9 | F3.1 Environment & Config Defaults | Update `.env`, `.env.example`, `docker-compose.yml`, `config.py` default to `qwen2.5-coder-7b-policy` | M3 | ORIGINAL_REQUEST §R3 |
| 10 | F3.2 Backend Dynamic Model Integration | Update `backend/api/routes/models.py`, `api_dto.py`, `dependencies.py` for fine-tuned default model | M3 | ORIGINAL_REQUEST §R3 |
| 11 | F3.3 Frontend Model Defaults | Update `ChatWindow.tsx`, `api-client.ts` to default model selection to fine-tuned Qwen 2.5 Coder | M3 | ORIGINAL_REQUEST §R3 |
| 12 | F4.1 Dataset Validation Test Suite | Unit tests for Alpaca, ShareGPT, JSONL loader, splitting, schema validation | M4 / E2E | ORIGINAL_REQUEST §R4 |
| 13 | F4.2 Smoke-Test Training Execution | Automated test verifying training execution, metrics computation, and adapter output | M4 / E2E | ORIGINAL_REQUEST §R4 |
| 14 | F4.3 Merge, GGUF & Modelfile Tests | Automated tests verifying weight merger, GGUF export validation, Modelfile syntax & stop tokens | M4 / E2E | ORIGINAL_REQUEST §R4 |
| 15 | F4.4 Ollama Registration Verification | Automated test verifying Ollama model creation and local tag registration | M4 / E2E | ORIGINAL_REQUEST §R4 |
| 16 | F4.5 End-to-End RAG API Validation | E2E test verifying RAG query execution, streaming, routing, verification with fine-tuned model | M4 / E2E | ORIGINAL_REQUEST §R4 |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Testing Track | Requirement-driven opaque-box test suite (Tiers 1-4) & `TEST_READY.md` | none | IN_PROGRESS |
| 1 | M1: Fine-Tuning Pipeline | Dataset loader, LoRA/QLoRA trainer, metrics logger, CLI `finetune_qwen_coder.py` | none | IN_PROGRESS |
| 2 | M2: Model Merging, GGUF Export & Ollama Registration | Weight merger, GGUF quantization exporter, Modelfile generator, Ollama registrar, CLI `export_and_register_ollama.py` | M1 | PLANNED |
| 3 | M3: System Integration & Defaults | Configuration files (.env, config.py, docker-compose.yml), backend routes, frontend defaults | M2 | PLANNED |
| 4 | M4: Automated Verification & Final E2E Validation | Pass 100% E2E test suite (Tiers 1-4), smoke-test validation, and adversarial coverage hardening (Tier 5) | M3, E2E | PLANNED |

---

## Interface Contracts

### 1. `DatasetLoader` (`company_policy_rag/src/finetuning/dataset_loader.py`)
- `load_dataset_from_file(file_path: str, val_split: float = 0.1, seed: int = 42) -> Tuple[Dataset, Dataset]`
- `normalize_record(record: dict) -> List[Dict[str, str]]`: Returns `[{"role": "system"|"user"|"assistant", "content": "..."}]`.
- Supported formats: Alpaca (`instruction`, `input`, `output`), ShareGPT (`conversations`/`messages`), JSONL (`prompt`/`response` or `messages`).

### 2. `QwenLoRATrainer` (`company_policy_rag/src/finetuning/trainer.py`)
- `train_lora(config: FineTuneConfig) -> TrainingOutput`
- `FineTuneConfig`: `model_name_or_path: str`, `dataset_path: str`, `output_dir: str`, `lora_r: int = 16`, `lora_alpha: int = 32`, `lora_dropout: float = 0.05`, `use_qlora: bool = True`, `batch_size: int = 2`, `gradient_accumulation_steps: int = 4`, `learning_rate: float = 2e-4`, `num_train_epochs: int = 3`, `max_seq_length: int = 2048`, `val_split: float = 0.1`, `smoke_test: bool = False`.
- Output: Adapter directory containing `adapter_model.safetensors`, `adapter_config.json`, `training_history.json`, `metrics_summary.json`.

### 3. `ModelMerger` (`company_policy_rag/src/finetuning/merger.py`)
- `merge_lora_weights(base_model_path: str, adapter_path: str, output_dir: str, device: str = "cpu") -> str`
- Output: Standalone merged HuggingFace model directory with full model weights and tokenizer files.

### 4. `GGUFExporter` & `ModelfileGenerator` (`company_policy_rag/src/finetuning/`)
- `convert_to_gguf(model_dir: str, output_file: str, quantization: str = "Q4_K_M") -> str`
- `generate_modelfile(gguf_path: str, output_path: str, system_prompt: Optional[str] = None, num_ctx: int = 8192, temperature: float = 0.1) -> str`
- Stop tokens: `<|im_end|>`, `<|endoftext|>`.

### 5. `OllamaRegistrar` (`company_policy_rag/src/finetuning/ollama_registrar.py`)
- `register_model_in_ollama(model_name: str, modelfile_path: str, ollama_url: str = "http://localhost:11434") -> bool`

---

## Code Layout
```
company_policy_rag/
├── src/
│   ├── config.py                               # System settings (default LLM model: qwen2.5-coder-7b-policy)
│   ├── ollama_client.py                        # Ollama client & model probe/preload utilities
│   ├── generation.py                           # LLM generation with ChatML formatting & verification
│   └── finetuning/                             # [NEW] Fine-tuning & deployment package
│       ├── __init__.py
│       ├── dataset_loader.py                   # Multi-format dataset ingestion & validation splitting
│       ├── trainer.py                          # LoRA/QLoRA trainer & perplexity metrics logger
│       ├── merger.py                           # LoRA adapter weight merger
│       ├── gguf_exporter.py                    # GGUF converter & quantization utility
│       ├── modelfile_generator.py              # Ollama Modelfile generator
│       └── ollama_registrar.py                 # Ollama model registration utility
├── scripts/
│   ├── finetune_qwen_coder.py                  # CLI entrypoint for fine-tuning
│   ├── export_and_register_ollama.py           # CLI entrypoint for merge, GGUF export, & Ollama register
│   └── run_finetune_pipeline.py                # Unified end-to-end pipeline runner
├── backend/
│   ├── api/routes/models.py                    # Dynamic model selection endpoints
│   ├── models/api_dto.py                       # API data transfer objects & model defaults
│   └── dependencies.py                         # FastAPI dependency injection
├── frontend/
│   ├── components/ChatWindow.tsx               # UI default model selector
│   └── lib/api-client.ts                       # Frontend API client
├── tests/
│   ├── unit/
│   │   ├── test_finetuning_dataset.py          # Unit tests for dataset loader & splitting
│   │   ├── test_finetuning_trainer.py          # Unit tests for trainer & metrics logging
│   │   ├── test_export_merge_modelfile.py      # Unit tests for merger, GGUF exporter, Modelfile
│   │   └── test_ollama_registration.py         # Unit tests for Ollama registrar
│   └── e2e/
│       └── test_finetuned_rag_e2e.py           # End-to-end integration test
├── .env.example
├── docker-compose.yml
└── pyproject.toml
```
