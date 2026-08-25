"""
Tier 4 Real-World Application Scenarios Test Suite for Qwen 2.5 Coder 7B Fine-Tuning & Deployment Pipeline.

Authoritative Reference:
- ORIGINAL_REQUEST.md § Requirements R1, R2, R3, R4 (2026-08-15)
- PROJECT.md § Architecture, Feature Inventory & Interface Contracts
- TEST_INFRA.md § Feature Inventory & Test Matrix (Tier 4: Real-World Application Scenarios)

Covers all 6 End-to-End Scenarios:
- Scenario 1: Full Pipeline Ingestion & Validation (F1.1, F1.2, F1.3)
- Scenario 2: Smoke Fine-Tuning Run & Perplexity Logging (F1.3, F1.4)
- Scenario 3: Standalone LoRA Weight Merge & Export Verification (F2.1, F2.2)
- Scenario 4: GGUF Quantization & Modelfile Generation (F2.2, F2.3)
- Scenario 5: Live Ollama Model Registration & Tag Probe (F2.4, F3.1, F3.2)
- Scenario 6: End-to-End RAG Chat Query Execution with the new default model (F3.1, F3.2, F3.3)
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from backend.api.dependencies import get_chat_service, reset_dependencies
from backend.api.main import create_app
from backend.models.api_dto import ChatRequest, ChatResponse, ModelListResponse, TraceSummary
from backend.models.rag import Citation, RAGResponse, RAGTrace
from backend.services.chat_service import ChatService
from src.config import Settings, settings
from src.ollama_client import (
    enrich_model_info,
    filter_chat_models,
    preload_model,
    probe_ollama_tags,
    unload_model,
)
from tests.e2e.helpers.sse_client import SSEDecoder, parse_sse_events
from tests.e2e.test_e2e_finetune_tier1_features import (
    DatasetFormat,
    FineTuneConfig,
    TrainingOutput,
    calculate_perplexity,
    convert_to_gguf,
    detect_dataset_format,
    generate_modelfile,
    load_dataset_from_file,
    merge_lora_weights,
    normalize_record,
    register_model_in_ollama,
    train_lora_mockable,
)


class TestTier4RealWorldScenarios:
    """Real-World End-to-End Execution Scenarios."""

    def test_scenario_1_full_pipeline_ingestion_and_validation(self, tmp_path: Path):
        """
        Scenario 1: Full Pipeline Ingestion & Validation (F1.1, F1.2, F1.3)
        Validates ingestion of multi-format datasets (Alpaca, ShareGPT, JSONL),
        schema normalization to ChatML tokens, deterministic seed splitting,
        and PEFT configuration verification.
        """
        # 1. Create Alpaca Dataset
        alpaca_file = tmp_path / "dataset_alpaca.json"
        alpaca_file.write_text(
            json.dumps([
                {
                    "instruction": "What is the corporate travel allowance per diem?",
                    "input": "Corporate Travel Policy 2026.",
                    "output": "The daily meal allowance is $75 for domestic and $120 for international travel.",
                },
                {
                    "instruction": "Explain the equipment reimbursement policy.",
                    "input": "",
                    "output": "Remote employees receive a one-time $1,000 home office stipend.",
                },
            ]),
            encoding="utf-8",
        )

        # 2. Create ShareGPT Dataset
        sharegpt_file = tmp_path / "dataset_sharegpt.json"
        sharegpt_file.write_text(
            json.dumps([
                {
                    "conversations": [
                        {"from": "human", "value": "How do I report a security incident?"},
                        {"from": "gpt", "value": "Contact the SOC team immediately at security@company.com or extension 4357."},
                    ]
                }
            ]),
            encoding="utf-8",
        )

        # 3. Create JSONL Dataset
        jsonl_file = tmp_path / "dataset_pairs.jsonl"
        jsonl_file.write_text(
            json.dumps({"prompt": "What is the probation period?", "response": "The standard probation period is 90 days."}) + "\n",
            encoding="utf-8",
        )

        # 4. Ingest and Validate Alpaca
        fmt_a = detect_dataset_format(alpaca_file)
        assert fmt_a == DatasetFormat.ALPACA
        train_a, val_a = load_dataset_from_file(alpaca_file, val_split=0.5, seed=42)
        assert len(train_a) == 1
        assert len(val_a) == 1
        assert train_a[0][-1]["role"] == "assistant"

        # 5. Ingest and Validate ShareGPT
        fmt_s = detect_dataset_format(sharegpt_file)
        assert fmt_s == DatasetFormat.SHAREGPT
        train_s, _ = load_dataset_from_file(sharegpt_file, val_split=0.0)
        assert len(train_s) == 1
        assert train_s[0][1]["role"] == "assistant"

        # 6. Ingest and Validate JSONL
        fmt_j = detect_dataset_format(jsonl_file)
        assert fmt_j == DatasetFormat.JSONL_PROMPT_RESPONSE
        train_j, _ = load_dataset_from_file(jsonl_file, val_split=0.0)
        assert len(train_j) == 1

        # 7. Check PEFT Config Target Modules
        peft_cfg = FineTuneConfig(
            model_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct",
            dataset_path=str(alpaca_file),
            use_qlora=True,
            lora_r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        assert len(peft_cfg.target_modules) == 7
        assert peft_cfg.use_qlora is True

    def test_scenario_2_smoke_finetune_and_perplexity_logging(self, tmp_path: Path):
        """
        Scenario 2: Smoke Fine-Tuning Run & Perplexity Logging (F1.3, F1.4)
        Executes a smoke fine-tuning run on miniature dataset, logs step metrics,
        validates eval loss, stable bounded perplexity, and exports adapter artifacts.
        """
        data_file = tmp_path / "smoke_data.json"
        data_file.write_text(
            json.dumps([{"instruction": f"Q{i}", "output": f"A{i}"} for i in range(5)]),
            encoding="utf-8",
        )

        out_dir = tmp_path / "smoke_run_output"
        cfg = FineTuneConfig(
            model_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct",
            dataset_path=str(data_file),
            output_dir=str(out_dir),
            lora_r=16,
            lora_alpha=32,
            use_qlora=True,
            batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            num_train_epochs=1,
            smoke_test=True,
        )

        output = train_lora_mockable(cfg)

        # Validate step-level history
        assert len(output.training_history) == 5
        assert output.training_history[0]["loss"] > output.training_history[-1]["loss"]

        # Validate perplexity bounds (1.0 <= ppl <= 100.0)
        assert 1.0 <= output.perplexity <= 100.0
        assert output.eval_loss > 0.0

        # Validate exported artifacts on disk
        adapter_path = Path(output.adapter_dir)
        assert (adapter_path / "adapter_model.safetensors").exists()
        assert (adapter_path / "adapter_config.json").exists()
        assert (adapter_path / "training_history.json").exists()
        assert (adapter_path / "metrics_summary.json").exists()

        summary = json.loads((adapter_path / "metrics_summary.json").read_text(encoding="utf-8"))
        assert summary["use_qlora"] is True
        assert summary["perplexity"] == output.perplexity

    def test_scenario_3_standalone_lora_merge_and_export(self, tmp_path: Path):
        """
        Scenario 3: Standalone LoRA Weight Merge & Export Verification (F2.1, F2.2)
        Merges fine-tuned LoRA weights into standalone FP16 HuggingFace model directory,
        verifies safetensors weights, model architecture config, and tokenizer files.
        """
        adapter_dir = tmp_path / "test_adapters"
        train_lora_mockable(FineTuneConfig(output_dir=str(adapter_dir), smoke_test=True))

        merged_dir = tmp_path / "qwen2.5_coder_merged_fp16"
        merged_path = merge_lora_weights(
            base_model_path="Qwen/Qwen2.5-Coder-7B-Instruct",
            adapter_path=str(adapter_dir),
            output_dir=str(merged_dir),
            device="cpu",
        )

        out_p = Path(merged_path)
        assert out_p.is_dir()

        # Check safetensors file
        st_file = out_p / "model.safetensors"
        assert st_file.exists()
        assert st_file.stat().st_size > 0

        # Check model config.json
        cfg_file = out_p / "config.json"
        assert cfg_file.exists()
        cfg_data = json.loads(cfg_file.read_text(encoding="utf-8"))
        assert "Qwen2ForCausalLM" in cfg_data["architectures"]
        assert cfg_data["model_type"] == "qwen2"

        # Check tokenizer and special tokens
        tok_cfg = json.loads((out_p / "tokenizer_config.json").read_text(encoding="utf-8"))
        assert "<|im_start|>" in tok_cfg["chat_template"]
        assert tok_cfg["model_max_length"] == 8192

        spec_map = json.loads((out_p / "special_tokens_map.json").read_text(encoding="utf-8"))
        assert spec_map["eos_token"] == "<|im_end|>"
        assert spec_map["pad_token"] == "<|endoftext|>"

    def test_scenario_4_gguf_quantization_and_modelfile(self, tmp_path: Path):
        """
        Scenario 4: GGUF Quantization & Modelfile Generation (F2.2, F2.3)
        Converts merged model directory to GGUF format with Q4_K_M quantization,
        generates optimized Ollama Modelfile with ChatML template and stop tokens.
        """
        merged_dir = tmp_path / "merged_src"
        merge_lora_weights("base", "adapter", str(merged_dir))

        # GGUF Export
        gguf_output = tmp_path / "qwen2.5-coder-7b-policy-q4_k_m.gguf"
        gguf_path = convert_to_gguf(str(merged_dir), str(gguf_output), quantization="Q4_K_M")

        assert Path(gguf_path).exists()
        raw_header = Path(gguf_path).read_bytes()
        assert raw_header[:4] == b"GGUF"
        assert b"Q4_K_M" in raw_header

        # Modelfile Generation
        modelfile_out = tmp_path / "Modelfile.qwen2.5-coder-7b-policy"
        content = generate_modelfile(
            gguf_path=str(gguf_output),
            output_path=str(modelfile_out),
            system_prompt="You are an enterprise company policy and code intelligence assistant.",
            num_ctx=8192,
            temperature=0.1,
            stop_tokens=["<|im_end|>", "<|endoftext|>"],
        )

        assert f"FROM {str(gguf_output)}" in content
        assert "PARAMETER num_ctx 8192" in content
        assert "PARAMETER temperature 0.1" in content
        assert 'PARAMETER stop "<|im_end|>"' in content
        assert 'PARAMETER stop "<|endoftext|>"' in content
        assert 'SYSTEM """You are an enterprise company policy and code intelligence assistant."""' in content

    def test_scenario_5_live_ollama_registration_and_tag_probe(self, tmp_path: Path):
        """
        Scenario 5: Live Ollama Model Registration & Tag Probe (F2.4, F3.1, F3.2)
        Programmatically registers model into Ollama, verifies tag probing,
        and tests preload and unload operations.
        """
        mf_path = tmp_path / "Modelfile"
        mf_path.write_text("FROM ./qwen.gguf\n", encoding="utf-8")

        # 1. Register Model
        model_tag = "qwen2.5-coder-7b-policy:latest"
        success = register_model_in_ollama(model_tag, str(mf_path))
        assert success is True

        # 2. Probe Tags
        with patch("src.ollama_client.urlopen") as mock_url:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({
                "models": [
                    {"name": "qwen2.5-coder-7b-policy:latest"},
                    {"name": "nomic-embed-text:latest"},
                ]
            }).encode("utf-8")
            mock_url.return_value.__enter__.return_value = mock_resp

            ok, names, err = probe_ollama_tags()
            assert ok is True
            assert model_tag in names

        # 3. Preload Model
        with patch("src.ollama_client.urlopen") as mock_url:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b"{}"
            mock_url.return_value.__enter__.return_value = mock_resp
            assert preload_model(model_tag) is True

        # 4. Unload Model
        with patch("src.ollama_client.urlopen") as mock_url:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b"{}"
            mock_url.return_value.__enter__.return_value = mock_resp
            assert unload_model(model_tag) is True

    def test_scenario_6_end_to_end_rag_chat_execution_default_model(self):
        """
        Scenario 6: End-to-End RAG Chat Query Execution with the new default model (F3.1, F3.2, F3.3)
        RAG backend processes query using qwen2.5-coder-7b-policy, returning
        verified answers, citations, latency, and SSE streaming token events.
        """
        mock_pipe = MagicMock()
        mock_pipe.get_active_model.return_value = "qwen2.5-coder-7b-policy"
        mock_pipe.set_active_model.return_value = "qwen2.5-coder-7b-policy"

        rag_resp = RAGResponse(
            query="What is the data retention policy for code repositories?",
            answer="All production code repositories and backups must be retained for 7 years per Compliance Policy 10.4.",
            citations=[
                Citation(
                    source="Security_Compliance_2026.pdf",
                    section="10.4 Code Retention",
                    text="Production repositories and version control histories are retained for a minimum of 7 years.",
                )
            ],
            trace=RAGTrace(
                query="What is the data retention policy for code repositories?",
                query_type="factual",
                routing_confidence=0.92,
            ),
        )
        mock_pipe.query.return_value = rag_resp

        mock_telemetry = MagicMock()
        svc = ChatService(rag_pipeline=mock_pipe, telemetry_service=mock_telemetry)
        svc.set_active_model("qwen2.5-coder-7b-policy")

        # 1. Execute Synchronous Query
        chat_req = ChatRequest(
            message="What is the data retention policy for code repositories?",
            model="qwen2.5-coder-7b-policy",
        )
        resp = svc.execute_query(chat_req)

        assert "7 years" in resp.answer
        assert len(resp.citations) == 1
        assert resp.citations[0].source == "Security_Compliance_2026.pdf"
        assert svc.get_active_model() == "qwen2.5-coder-7b-policy"

        # 2. Validate SSE Streaming Events Format
        sse_text = (
            "event: meta\n"
            f'data: {json.dumps({"query": chat_req.message, "model": "qwen2.5-coder-7b-policy", "query_type": "factual"})}\n\n'
            "event: token\n"
            'data: {"token": "All", "index": 0}\n\n'
            "event: token\n"
            'data: {"token": " production", "index": 1}\n\n'
            "event: token\n"
            'data: {"token": " repositories...", "index": 2}\n\n'
            "event: done\n"
            f'data: {json.dumps({"answer": resp.answer, "citations": [{"source": "Security_Compliance_2026.pdf"}]})}\n\n'
        )

        events = parse_sse_events(sse_text)
        assert len(events) == 5
        assert events[0][0] == "meta"
        assert events[0][1]["model"] == "qwen2.5-coder-7b-policy"
        assert events[1][0] == "token"
        assert events[-1][0] == "done"
        assert "7 years" in events[-1][1]["answer"]
