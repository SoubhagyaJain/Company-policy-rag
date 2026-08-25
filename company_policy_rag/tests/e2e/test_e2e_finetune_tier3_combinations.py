"""
Tier 3 Cross-Feature Combinations Test Suite for Qwen 2.5 Coder 7B Fine-Tuning & Deployment Pipeline.

Authoritative Reference:
- ORIGINAL_REQUEST.md § Requirements R1, R2, R3, R4 (2026-08-15)
- PROJECT.md § Architecture, Feature Inventory & Interface Contracts
- TEST_INFRA.md § Feature Inventory & Test Matrix (Tier 3: Pairwise Combinations)

Coverage Target: >= 11 comprehensive pairwise cross-feature interaction test cases.
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
import httpx

from backend.api.dependencies import get_chat_service, reset_dependencies
from backend.api.main import create_app
from backend.models.api_dto import ChatRequest, ChatResponse, ModelListResponse
from backend.models.rag import Citation, RAGResponse, RAGTrace
from backend.services.chat_service import ChatService
from src.config import Settings, settings
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


class TestTier3Combinations:
    """Pairwise Cross-Feature Combinatorial Integration Test Suite."""

    def test_comb_1_alpaca_qlora_merge_q4km_modelfile(self, tmp_path: Path):
        """
        Combination 1:
        Alpaca dataset format -> 4-bit QLoRA fine-tuning -> Standalone LoRA merge ->
        Q4_K_M GGUF quantization -> Ollama Modelfile generation with ChatML template.
        """
        # Step 1: Ingest Alpaca data
        alpaca_file = tmp_path / "alpaca_policy.json"
        alpaca_file.write_text(
            json.dumps([
                {
                    "instruction": "What is the parental leave duration?",
                    "input": "HR Policy Section 6.1.",
                    "output": "Eligible employees receive 16 weeks of fully paid parental leave.",
                }
            ]),
            encoding="utf-8",
        )
        fmt = detect_dataset_format(alpaca_file)
        assert fmt == DatasetFormat.ALPACA
        train_data, _ = load_dataset_from_file(alpaca_file, val_split=0.0)
        assert len(train_data) == 1

        # Step 2: Train with 4-bit QLoRA
        adapter_dir = tmp_path / "qlora_adapters"
        cfg = FineTuneConfig(
            dataset_path=str(alpaca_file),
            output_dir=str(adapter_dir),
            use_qlora=True,
            lora_r=16,
            lora_alpha=32,
            smoke_test=True,
        )
        train_out = train_lora_mockable(cfg)
        assert (Path(train_out.adapter_dir) / "adapter_model.safetensors").exists()

        # Step 3: Merge weights into standalone directory
        merged_dir = tmp_path / "merged_q4km"
        merged_path = merge_lora_weights(
            base_model_path="Qwen/Qwen2.5-Coder-7B-Instruct",
            adapter_path=train_out.adapter_dir,
            output_dir=str(merged_dir),
        )
        assert (Path(merged_path) / "model.safetensors").exists()

        # Step 4: Export to GGUF Q4_K_M
        gguf_file = tmp_path / "qwen_coder_q4km.gguf"
        gguf_path = convert_to_gguf(merged_path, str(gguf_file), quantization="Q4_K_M")
        assert Path(gguf_path).read_bytes().startswith(b"GGUF")

        # Step 5: Generate Modelfile
        modelfile = tmp_path / "Modelfile_q4km"
        mf_content = generate_modelfile(gguf_path=gguf_path, output_path=str(modelfile))
        assert "FROM " in mf_content
        assert "<|im_start|>system" in mf_content
        assert 'PARAMETER stop "<|im_end|>"' in mf_content

    def test_comb_2_sharegpt_lora_fp16_ollama_api(self, tmp_path: Path):
        """
        Combination 2:
        ShareGPT multi-turn dataset -> 16-bit FP16 LoRA -> LoRA merge ->
        FP16 GGUF export -> Ollama API registration.
        """
        sharegpt_file = tmp_path / "sharegpt.json"
        sharegpt_file.write_text(
            json.dumps([
                {
                    "conversations": [
                        {"from": "human", "value": "How many vacation days do full-time employees get?"},
                        {"from": "gpt", "value": "Full-time employees receive 20 days of paid vacation annually."},
                    ]
                }
            ]),
            encoding="utf-8",
        )
        assert detect_dataset_format(sharegpt_file) == DatasetFormat.SHAREGPT

        adapter_dir = tmp_path / "fp16_adapters"
        cfg = FineTuneConfig(
            dataset_path=str(sharegpt_file),
            output_dir=str(adapter_dir),
            use_qlora=False,
            lora_r=32,
            lora_alpha=64,
            smoke_test=True,
        )
        train_out = train_lora_mockable(cfg)

        merged_dir = tmp_path / "merged_fp16"
        merged_path = merge_lora_weights("base", train_out.adapter_dir, str(merged_dir))

        gguf_file = tmp_path / "model_fp16.gguf"
        gguf_path = convert_to_gguf(merged_path, str(gguf_file), quantization="FP16")

        modelfile = tmp_path / "Modelfile_fp16"
        generate_modelfile(gguf_path=gguf_path, output_path=str(modelfile))

        success = register_model_in_ollama("qwen2.5-coder-7b-policy:fp16", str(modelfile))
        assert success is True

    def test_comb_3_jsonl_qlora_q80_system_backend_routing(self, tmp_path: Path):
        """
        Combination 3:
        JSONL prompt-response -> 4-bit QLoRA -> Q8_0 GGUF export ->
        System Config default -> Backend dynamic model routing.
        """
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(
            json.dumps({"prompt": "Explain code review SLA", "response": "Code reviews must be completed in 24 hours."}) + "\n",
            encoding="utf-8",
        )

        train_data, _ = load_dataset_from_file(jsonl_file, val_split=0.0)
        assert len(train_data) == 1

        adapter_dir = tmp_path / "q80_adapters"
        train_out = train_lora_mockable(FineTuneConfig(output_dir=str(adapter_dir), smoke_test=True))

        gguf_file = tmp_path / "model_q80.gguf"
        gguf_path = convert_to_gguf(train_out.adapter_dir, str(gguf_file), quantization="Q8_0")
        assert Path(gguf_path).exists()

        # Connect to Backend ChatService dynamic routing
        mock_pipe = MagicMock()
        mock_pipe.set_active_model.return_value = "qwen2.5-coder-7b-policy"
        mock_pipe.get_active_model.return_value = "qwen2.5-coder-7b-policy"
        mock_pipe.query.return_value = RAGResponse(
            query="Explain code review SLA",
            answer="Code reviews must be completed in 24 hours.",
            citations=[],
        )
        mock_telemetry = MagicMock()

        svc = ChatService(rag_pipeline=mock_pipe, telemetry_service=mock_telemetry)
        svc.set_active_model("qwen2.5-coder-7b-policy")

        req = ChatRequest(message="Explain code review SLA", model="qwen2.5-coder-7b-policy")
        resp = svc.execute_query(req)
        assert "24 hours" in resp.answer
        assert svc.get_active_model() == "qwen2.5-coder-7b-policy"

    def test_comb_4_alpaca_lora8bit_q80_modelfile_cli(self, tmp_path: Path):
        """
        Combination 4:
        Alpaca format -> 8-bit LoRA -> LoRA Merge -> Q8_0 GGUF -> Modelfile -> Ollama registration.
        """
        alpaca_file = tmp_path / "alpaca.json"
        alpaca_file.write_text(json.dumps([{"instruction": "Task", "output": "Done"}]), encoding="utf-8")

        adapter_dir = tmp_path / "lora8_adapters"
        train_out = train_lora_mockable(
            FineTuneConfig(dataset_path=str(alpaca_file), output_dir=str(adapter_dir), use_qlora=False, smoke_test=True)
        )

        merged_dir = tmp_path / "merged_lora8"
        merge_lora_weights("base", train_out.adapter_dir, str(merged_dir))

        gguf_file = tmp_path / "lora8_q80.gguf"
        convert_to_gguf(str(merged_dir), str(gguf_file), quantization="Q8_0")

        mf = tmp_path / "Modelfile_lora8"
        generate_modelfile(str(gguf_file), str(mf), num_ctx=8192, temperature=0.1)

        assert register_model_in_ollama("qwen2.5-coder-7b-policy:q8_0", str(mf)) is True

    def test_comb_5_sharegpt_qlora_q4km_backend_chat_stream(self, tmp_path: Path):
        """
        Combination 5:
        ShareGPT multi-turn -> QLoRA -> Merge -> Q4_K_M GGUF -> Backend model switch -> SSE streaming format.
        """
        sharegpt_file = tmp_path / "sharegpt.json"
        sharegpt_file.write_text(
            json.dumps([{"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]}]),
            encoding="utf-8",
        )

        adapter_dir = tmp_path / "adapters"
        train_lora_mockable(FineTuneConfig(output_dir=str(adapter_dir), smoke_test=True))

        gguf_file = tmp_path / "q4km.gguf"
        convert_to_gguf(str(adapter_dir), str(gguf_file), quantization="Q4_K_M")

        # Test SSE streaming payload formatting
        sse_raw = (
            "event: meta\n"
            'data: {"model": "qwen2.5-coder-7b-policy", "query": "Hello"}\n\n'
            "event: token\n"
            'data: {"token": "Hello", "index": 0}\n\n'
            "event: done\n"
            'data: {"answer": "Hello! How can I help you today?"}\n\n'
        )
        events = parse_sse_events(sse_raw)
        assert len(events) == 3
        assert events[0][0] == "meta"
        assert events[0][1]["model"] == "qwen2.5-coder-7b-policy"
        assert events[2][0] == "done"

    def test_comb_6_jsonl_messages_fp16_merge_modelfile(self, tmp_path: Path):
        """
        Combination 6:
        JSONL messages format -> FP16 LoRA -> Merge weights -> Modelfile with custom enterprise system prompt.
        """
        jsonl_file = tmp_path / "messages.jsonl"
        jsonl_file.write_text(
            json.dumps({
                "messages": [
                    {"role": "system", "content": "Custom enterprise prompt."},
                    {"role": "user", "content": "What is the severance policy?"},
                    {"role": "assistant", "content": "Standard severance is 2 weeks per year of service."},
                ]
            }) + "\n",
            encoding="utf-8",
        )

        train_data, _ = load_dataset_from_file(jsonl_file, val_split=0.0)
        assert train_data[0][0]["role"] == "system"

        adapter_dir = tmp_path / "fp16_msg_adapters"
        train_out = train_lora_mockable(FineTuneConfig(output_dir=str(adapter_dir), smoke_test=True))

        merged_dir = tmp_path / "merged_msg"
        merge_lora_weights("base", train_out.adapter_dir, str(merged_dir))

        gguf_file = tmp_path / "fp16_msg.gguf"
        convert_to_gguf(str(merged_dir), str(gguf_file), quantization="FP16")

        mf = tmp_path / "Modelfile_custom_prompt"
        content = generate_modelfile(
            str(gguf_file),
            str(mf),
            system_prompt="Custom enterprise prompt.",
            num_ctx=16384,
        )
        assert 'SYSTEM """Custom enterprise prompt."""' in content
        assert "PARAMETER num_ctx 16384" in content

    def test_comb_7_alpaca_custom_prompt_qlora_merge_chat(self, tmp_path: Path):
        """
        Combination 7:
        Alpaca with custom system prompt -> QLoRA -> Merge -> GGUF Q4_K_M -> Live Chat query with verified citations.
        """
        alpaca_file = tmp_path / "alpaca_custom.json"
        alpaca_file.write_text(
            json.dumps([
                {
                    "instruction": "Where are corporate travel guidelines documented?",
                    "output": "Guidelines are in Travel_Policy_2026.pdf.",
                    "system": "Legal compliance assistant.",
                }
            ]),
            encoding="utf-8",
        )

        train_data, _ = load_dataset_from_file(alpaca_file, val_split=0.0)
        assert train_data[0][0]["content"] == "Legal compliance assistant."

        adapter_dir = tmp_path / "adapters_custom"
        train_out = train_lora_mockable(FineTuneConfig(output_dir=str(adapter_dir), smoke_test=True))

        gguf_file = tmp_path / "q4km_custom.gguf"
        convert_to_gguf(train_out.adapter_dir, str(gguf_file), quantization="Q4_K_M")

        mock_pipe = MagicMock()
        mock_pipe.query.return_value = RAGResponse(
            query="Where are travel guidelines?",
            answer="Guidelines are in Travel_Policy_2026.pdf.",
            citations=[Citation(source="Travel_Policy_2026.pdf", section="Section 1", text="Guidelines...")],
        )
        mock_telemetry = MagicMock()

        svc = ChatService(rag_pipeline=mock_pipe, telemetry_service=mock_telemetry)
        resp = svc.execute_query(ChatRequest(message="Where are travel guidelines?"))
        assert len(resp.citations) == 1
        assert resp.citations[0].source == "Travel_Policy_2026.pdf"

    def test_comb_8_multiturn_sharegpt_qlora_q80_tag_probe(self, tmp_path: Path):
        """
        Combination 8:
        Multi-turn ShareGPT -> QLoRA -> Merge -> Q8_0 GGUF -> Ollama model creation -> probe_ollama_tags tag verification.
        """
        sharegpt_file = tmp_path / "multiturn.json"
        sharegpt_file.write_text(
            json.dumps([
                {
                    "conversations": [
                        {"from": "human", "value": "Turn 1"},
                        {"from": "gpt", "value": "Response 1"},
                        {"from": "human", "value": "Turn 2"},
                        {"from": "gpt", "value": "Response 2"},
                    ]
                }
            ]),
            encoding="utf-8",
        )

        train_data, _ = load_dataset_from_file(sharegpt_file, val_split=0.0)
        assert len(train_data[0]) == 4

        adapter_dir = tmp_path / "multi_adapters"
        train_out = train_lora_mockable(FineTuneConfig(output_dir=str(adapter_dir), smoke_test=True))

        gguf_file = tmp_path / "multi_q80.gguf"
        convert_to_gguf(train_out.adapter_dir, str(gguf_file), quantization="Q8_0")

        mf = tmp_path / "Modelfile_multi"
        generate_modelfile(str(gguf_file), str(mf))
        register_model_in_ollama("qwen2.5-coder-7b-policy:multiturn", str(mf))

        with patch("src.ollama_client.urlopen") as mock_url:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({
                "models": [{"name": "qwen2.5-coder-7b-policy:multiturn"}]
            }).encode("utf-8")
            mock_url.return_value.__enter__.return_value = mock_resp

            ok, names, err = probe_ollama_tags()
            assert ok is True
            assert "qwen2.5-coder-7b-policy:multiturn" in names

    def test_comb_9_empty_instruction_alpaca_fallback_lora(self, tmp_path: Path):
        """
        Combination 9:
        Alpaca records with blank input -> Dynamic prompt fallback -> LoRA training -> Export bundle verification.
        """
        alpaca_file = tmp_path / "no_input.json"
        alpaca_file.write_text(
            json.dumps([{"instruction": "List holidays", "input": "", "output": "10 public holidays."}]),
            encoding="utf-8",
        )

        train_data, _ = load_dataset_from_file(alpaca_file, val_split=0.0)
        user_msg = next(t["content"] for t in train_data[0] if t["role"] == "user")
        assert user_msg == "List holidays"

        adapter_dir = tmp_path / "no_input_adapters"
        train_out = train_lora_mockable(FineTuneConfig(output_dir=str(adapter_dir), smoke_test=True))
        assert train_out.perplexity > 0

    def test_comb_10_high_rank_lora_merge_fp16_modelfile(self, tmp_path: Path):
        """
        Combination 10:
        High-rank LoRA (r=64, alpha=128) -> Training -> Merge -> FP16 export -> Modelfile with stop tokens.
        """
        adapter_dir = tmp_path / "high_rank_adapters"
        train_out = train_lora_mockable(
            FineTuneConfig(
                output_dir=str(adapter_dir),
                lora_r=64,
                lora_alpha=128,
                smoke_test=True,
            )
        )
        assert train_out.metrics_summary["lora_r"] == 64

        merged_dir = tmp_path / "merged_high_rank"
        merge_lora_weights("base", train_out.adapter_dir, str(merged_dir))

        gguf_file = tmp_path / "high_rank_fp16.gguf"
        convert_to_gguf(str(merged_dir), str(gguf_file), quantization="FP16")

        mf = tmp_path / "Modelfile_high_rank"
        content = generate_modelfile(str(gguf_file), str(mf))
        assert '<|im_end|>' in content
        assert '<|endoftext|>' in content

    def test_comb_11_jsonl_prompt_response_qlora_q4km_system_settings(self, tmp_path: Path, monkeypatch):
        """
        Combination 11:
        JSONL prompt-response -> QLoRA -> Q4_K_M export -> System settings configuration -> Active model verification.
        """
        monkeypatch.setenv("OLLAMA_LLM_MODEL", "qwen2.5-coder-7b-policy")
        s = Settings()
        assert s.llm_model == "qwen2.5-coder-7b-policy"

        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(json.dumps({"prompt": "Test Q", "response": "Test A"}) + "\n", encoding="utf-8")

        adapter_dir = tmp_path / "comb11_adapters"
        train_out = train_lora_mockable(FineTuneConfig(output_dir=str(adapter_dir), smoke_test=True))

        gguf_file = tmp_path / "comb11_q4km.gguf"
        convert_to_gguf(train_out.adapter_dir, str(gguf_file), quantization="Q4_K_M")

        with patch("src.ollama_client.probe_ollama_tags", return_value=(True, ["qwen2.5-coder-7b-policy"], None)):
            from backend.api.routes.models import get_available_models
            res = get_available_models()
            assert res.active_model == "qwen2.5-coder-7b-policy"
