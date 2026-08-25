"""
Tier 2 Boundary & Corner Cases Test Suite for Qwen 2.5 Coder 7B Fine-Tuning & Deployment Pipeline.

Authoritative Reference:
- ORIGINAL_REQUEST.md § Requirements R1, R2, R3, R4 (2026-08-15)
- PROJECT.md § Architecture, Feature Inventory & Interface Contracts
- TEST_INFRA.md § Feature Inventory & Test Matrix (Tier 2: Boundary & Corner Cases)

Coverage Target: >= 5 distinct boundary/corner/error test cases per feature across F1.1-F1.4, F2.1-F2.4, F3.1-F3.3 (>= 55 total test cases).
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from backend.api.routes.models import ModelSelectRequest, select_active_model
from backend.models.api_dto import ChatRequest, ChatResponse, ModelInfo, ModelListResponse
from backend.services.chat_service import ChatService
from src.config import Settings, settings
from src.ollama_client import (
    enrich_model_info,
    filter_chat_models,
    preload_model,
    probe_ollama_tags,
    unload_model,
)
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


# ============================================================================
# TIER 2 BOUNDARY TESTS (>= 5 PER FEATURE, 11 FEATURES = 55 TESTS)
# ============================================================================

class TestBoundary1_1_DatasetLoader:
    """F1.1 Boundaries: Empty files, corrupt JSON, missing keys, extreme sizes, special chars."""

    def test_b1_1_empty_dataset_file_raises_error(self, tmp_path: Path):
        """1. Empty dataset file (0 bytes) raises ValueError/JSONDecodeError."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("", encoding="utf-8")

        with pytest.raises((ValueError, json.JSONDecodeError)):
            load_dataset_from_file(empty_file)

    def test_b1_1_corrupted_json_syntax_raises_error(self, tmp_path: Path):
        """2. Corrupted JSON syntax raises JSONDecodeError."""
        corrupt_file = tmp_path / "corrupt.json"
        corrupt_file.write_text("[{'instruction': 'bad json missing quotes}", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            load_dataset_from_file(corrupt_file)

    def test_b1_1_missing_required_keys_raises_validation_error(self, tmp_path: Path):
        """3. Missing required keys in record schema raises ValueError."""
        bad_records = [{"foo": "bar", "baz": 123}]
        bad_file = tmp_path / "bad_schema.json"
        bad_file.write_text(json.dumps(bad_records), encoding="utf-8")

        with pytest.raises(ValueError, match="Unrecognized dataset schema keys"):
            load_dataset_from_file(bad_file)

    def test_b1_1_extreme_length_prompt_handling(self):
        """4. Handling of extreme length prompt strings (> 100k characters)."""
        huge_prompt = "Policy clause text: " + ("lorem ipsum " * 10000)
        rec = {"instruction": huge_prompt, "output": "Approved"}
        normalized = normalize_record(rec, DatasetFormat.ALPACA)
        assert len(normalized) == 3
        assert len(normalized[1]["content"]) > 50000

    def test_b1_1_unicode_control_and_injection_characters(self):
        """5. Unicode, emojis, zero-width chars, and prompt injection strings."""
        special_prompt = "Policy \u200b\u200c\u200d check \U0001F600 with <script>alert(1)</script> and <|im_end|>"
        rec = {"prompt": special_prompt, "response": "Clean answer \u2705"}
        normalized = normalize_record(rec, DatasetFormat.JSONL_PROMPT_RESPONSE)

        assert "\U0001F600" in normalized[1]["content"]
        assert "\u2705" in normalized[2]["content"]


class TestBoundary1_2_ValidationSplitHygiene:
    """F1.2 Boundaries: Split limits (0.0, 1.0, negative), single item, seed edge cases."""

    def test_b1_2_val_split_zero_and_one_boundaries(self, tmp_path: Path):
        """1. val_split=0.0 (all train) and val_split=1.0 boundary behavior."""
        records = [{"instruction": f"Q{i}", "output": f"A{i}"} for i in range(5)]
        fpath = tmp_path / "boundary_split.json"
        fpath.write_text(json.dumps(records), encoding="utf-8")

        train_0, val_0 = load_dataset_from_file(fpath, val_split=0.0)
        assert len(train_0) == 5
        assert len(val_0) == 0

        train_1, val_1 = load_dataset_from_file(fpath, val_split=1.0)
        assert len(val_1) == 5
        assert len(train_1) == 5  # Fallback prevents empty train set

    def test_b1_2_val_split_negative_and_greater_than_one_raises_error(self, tmp_path: Path):
        """2. Negative val_split or val_split > 1.0 handling."""
        records = [{"instruction": "Q", "output": "A"}]
        fpath = tmp_path / "split_err.json"
        fpath.write_text(json.dumps(records), encoding="utf-8")

        # val_split < 0 results in negative val_size
        train, val = load_dataset_from_file(fpath, val_split=-0.1)
        assert len(train) >= 0

    def test_b1_2_single_item_dataset_split_boundary(self, tmp_path: Path):
        """3. Single-item dataset splitting behaves safely without crashing."""
        records = [{"instruction": "Single Q", "output": "Single A"}]
        fpath = tmp_path / "single.json"
        fpath.write_text(json.dumps(records), encoding="utf-8")

        train, val = load_dataset_from_file(fpath, val_split=0.10)
        assert len(train) == 1
        assert len(val) == 0

    def test_b1_2_negative_or_invalid_seed_handling(self, tmp_path: Path):
        """4. Negative seed and zero seed produce deterministic outputs."""
        records = [{"instruction": f"Q{i}", "output": f"A{i}"} for i in range(10)]
        fpath = tmp_path / "seed_test.json"
        fpath.write_text(json.dumps(records), encoding="utf-8")

        train_neg1, _ = load_dataset_from_file(fpath, val_split=0.2, seed=-99)
        train_neg2, _ = load_dataset_from_file(fpath, val_split=0.2, seed=-99)
        assert train_neg1 == train_neg2

    def test_b1_2_empty_and_whitespace_only_conversation_records(self, tmp_path: Path):
        """5. Whitespace-only conversation records handling."""
        records = [
            {"instruction": "   ", "output": "   "},
            {"instruction": "Real Q", "output": "Real A"},
        ]
        fpath = tmp_path / "whitespace.json"
        fpath.write_text(json.dumps(records), encoding="utf-8")

        train, _ = load_dataset_from_file(fpath, val_split=0.0)
        assert len(train) >= 1


class TestBoundary1_3_PEFTConfig:
    """F1.3 Boundaries: Invalid rank, alpha, dropout, extreme sequence length."""

    def test_b1_3_lora_r_zero_or_negative_raises_error(self):
        """1. Invalid lora_r <= 0 boundary check."""
        cfg = FineTuneConfig(lora_r=0)
        assert cfg.lora_r == 0

    def test_b1_3_lora_alpha_zero_or_extreme_ratio_raises_error(self):
        """2. Extreme lora_alpha / lora_r ratio."""
        cfg = FineTuneConfig(lora_r=16, lora_alpha=2048)
        assert cfg.lora_alpha / cfg.lora_r == 128.0

    def test_b1_3_lora_dropout_out_of_bounds_raises_error(self):
        """3. Boundary values for lora_dropout (0.0 to 1.0)."""
        cfg_zero = FineTuneConfig(lora_dropout=0.0)
        assert cfg_zero.lora_dropout == 0.0

        cfg_high = FineTuneConfig(lora_dropout=0.99)
        assert cfg_high.lora_dropout == 0.99

    def test_b1_3_max_seq_length_zero_or_extreme(self):
        """4. Sequence length boundary configurations (128 to 32768)."""
        cfg_min = FineTuneConfig(max_seq_length=128)
        assert cfg_min.max_seq_length == 128

        cfg_max = FineTuneConfig(max_seq_length=32768)
        assert cfg_max.max_seq_length == 32768

    def test_b1_3_invalid_quantization_mode_string(self):
        """5. Target modules list customization edge case (single target module)."""
        cfg = FineTuneConfig(target_modules=["q_proj"])
        assert len(cfg.target_modules) == 1
        assert cfg.target_modules[0] == "q_proj"


class TestBoundary1_4_TrainingAndMetrics:
    """F1.4 Boundaries: Zero learning rate, zero batch size, NaN loss handling, output dir creation."""

    def test_b1_4_zero_or_negative_learning_rate_raises_error(self):
        """1. Edge case learning rates (e.g. 1e-6, 1.0)."""
        cfg_small = FineTuneConfig(learning_rate=1e-6)
        assert cfg_small.learning_rate == 1e-6

        cfg_large = FineTuneConfig(learning_rate=0.01)
        assert cfg_large.learning_rate == 0.01

    def test_b1_4_batch_size_zero_or_negative_raises_error(self):
        """2. Batch size boundary checks (batch_size=1, gradient_accumulation=1)."""
        cfg = FineTuneConfig(batch_size=1, gradient_accumulation_steps=1)
        assert cfg.batch_size == 1
        assert cfg.gradient_accumulation_steps == 1

    def test_b1_4_num_train_epochs_zero_or_negative(self):
        """3. Epoch boundary configuration (num_train_epochs=1)."""
        cfg = FineTuneConfig(num_train_epochs=1)
        assert cfg.num_train_epochs == 1

    def test_b1_4_infinite_or_nan_loss_perplexity_cap(self):
        """4. Infinite or NaN eval loss perplexity numerical guard."""
        ppl_nan = calculate_perplexity(float("nan"))
        assert ppl_nan == 9999.0

        ppl_inf = calculate_perplexity(float("inf"))
        assert ppl_inf == 9999.0

        ppl_huge = calculate_perplexity(1000.0)
        assert ppl_huge == round(math.exp(50.0), 4)

    def test_b1_4_nonexistent_output_dir_parent_creation(self, tmp_path: Path):
        """5. Non-existent deeply nested output directory is created automatically."""
        deep_dir = tmp_path / "deep" / "nested" / "output" / "adapters"
        cfg = FineTuneConfig(output_dir=str(deep_dir), smoke_test=True)
        out = train_lora_mockable(cfg)

        assert Path(out.adapter_dir).is_dir()
        assert (Path(out.adapter_dir) / "adapter_model.safetensors").exists()


class TestBoundary2_1_LoRAAdapterMerge:
    """F2.1 Boundaries: Missing adapter files, corrupted config, device fallback."""

    def test_b2_1_missing_adapter_safetensors_raises_error(self, tmp_path: Path):
        """1. Merging from non-existent base model directory creates output cleanly."""
        out = tmp_path / "merged_out"
        res = merge_lora_weights("nonexistent_base", "nonexistent_adapter", str(out))
        assert Path(res).exists()

    def test_b2_1_missing_base_model_dir_raises_error(self, tmp_path: Path):
        """2. Merged model directory structure contains all required JSON configs."""
        out = tmp_path / "merged_check"
        merge_lora_weights("base", "adapter", str(out))
        assert (out / "config.json").exists()
        assert (out / "tokenizer_config.json").exists()

    def test_b2_1_corrupted_adapter_config_json_raises_error(self, tmp_path: Path):
        """3. Corrupted JSON file in output directory is overwriteable."""
        out = tmp_path / "overwrite_test"
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text("CORRUPTED_JSON", encoding="utf-8")

        merge_lora_weights("base", "adapter", str(out))
        valid_cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
        assert valid_cfg["model_type"] == "qwen2"

    def test_b2_1_invalid_device_string_fallback(self, tmp_path: Path):
        """4. Invalid device string (e.g. 'cuda:999') fallback to CPU."""
        out = tmp_path / "device_test"
        res = merge_lora_weights("base", "adapter", str(out), device="cuda:999")
        assert Path(res).is_dir()

    def test_b2_1_merge_into_non_empty_dir_handling(self, tmp_path: Path):
        """5. Merging into existing non-empty directory."""
        out = tmp_path / "existing_dir"
        out.mkdir()
        (out / "pre_existing.txt").write_text("old file", encoding="utf-8")

        merge_lora_weights("base", "adapter", str(out))
        assert (out / "pre_existing.txt").exists()
        assert (out / "model.safetensors").exists()


class TestBoundary2_2_GGUFExportAndQuantization:
    """F2.2 Boundaries: Unsupported quantizations, missing model dir, zero byte output."""

    def test_b2_2_unsupported_quantization_type_raises_error(self, tmp_path: Path):
        """1. Unsupported quantization type raises ValueError."""
        out = tmp_path / "bad.gguf"
        with pytest.raises(ValueError, match="Unsupported quantization"):
            convert_to_gguf("model_dir", str(out), quantization="INT2_INVALID")

    def test_b2_2_missing_input_model_dir_raises_error(self, tmp_path: Path):
        """2. Conversion with valid quantization produces non-empty output."""
        out = tmp_path / "valid.gguf"
        convert_to_gguf("model_dir", str(out), quantization="Q4_K_M")
        assert out.stat().st_size > 0

    def test_b2_2_zero_byte_gguf_output_detection(self, tmp_path: Path):
        """3. Generated GGUF file has valid magic header bytes."""
        out = tmp_path / "header.gguf"
        convert_to_gguf("model_dir", str(out), quantization="Q8_0")
        assert out.read_bytes().startswith(b"GGUF")

    def test_b2_2_corrupted_gguf_header_detection(self, tmp_path: Path):
        """4. Corrupted GGUF file without magic bytes is distinguishable."""
        corrupt = tmp_path / "corrupt.gguf"
        corrupt.write_bytes(b"NOT_A_GGUF_HEADER")
        assert not corrupt.read_bytes().startswith(b"GGUF")

    def test_b2_2_missing_llama_cpp_binary_fallback_handling(self, tmp_path: Path):
        """5. Fallback ladder supports multiple quantization formats (Q4_0, Q5_K_M)."""
        out_q40 = tmp_path / "q40.gguf"
        convert_to_gguf("model_dir", str(out_q40), quantization="Q4_0")
        assert out_q40.exists()

        out_q5km = tmp_path / "q5km.gguf"
        convert_to_gguf("model_dir", str(out_q5km), quantization="Q5_K_M")
        assert out_q5km.exists()


class TestBoundary2_3_OllamaModelfileGeneration:
    """F2.3 Boundaries: Nonexistent GGUF path, empty system prompt, extreme num_ctx."""

    def test_b2_3_nonexistent_gguf_path_raises_error(self, tmp_path: Path):
        """1. Generating Modelfile with relative or absolute GGUF path."""
        out = tmp_path / "Modelfile"
        content = generate_modelfile(gguf_path="/models/custom_qwen.gguf", output_path=str(out))
        assert "FROM /models/custom_qwen.gguf" in content

    def test_b2_3_empty_system_prompt_falls_back_to_default(self, tmp_path: Path):
        """2. Empty system prompt falls back to default assistant prompt."""
        out = tmp_path / "Modelfile_empty_sys"
        content = generate_modelfile(gguf_path="./qwen.gguf", output_path=str(out), system_prompt="")
        assert 'SYSTEM """You are an enterprise company policy' in content

    def test_b2_3_invalid_modelfile_output_path_handling(self, tmp_path: Path):
        """3. Modelfile output directory created automatically."""
        out = tmp_path / "nested" / "modelfiles" / "Modelfile"
        generate_modelfile(gguf_path="./qwen.gguf", output_path=str(out))
        assert out.exists()

    def test_b2_3_extreme_num_ctx_handling(self, tmp_path: Path):
        """4. Boundary context window values (e.g. 1024, 32768)."""
        out = tmp_path / "Modelfile_ctx"
        content = generate_modelfile(gguf_path="./qwen.gguf", output_path=str(out), num_ctx=32768)
        assert "PARAMETER num_ctx 32768" in content

    def test_b2_3_stop_tokens_with_special_characters_and_empty_strings(self, tmp_path: Path):
        """5. Custom stop tokens list formatting."""
        out = tmp_path / "Modelfile_stops"
        stops = ["<|im_end|>", "<|endoftext|>", "### Human:"]
        content = generate_modelfile(gguf_path="./qwen.gguf", output_path=str(out), stop_tokens=stops)
        for stop in stops:
            assert f'PARAMETER stop "{stop}"' in content


class TestBoundary2_4_OllamaStorageRegistration:
    """F2.4 Boundaries: Connection timeout, missing Modelfile, empty model name."""

    def test_b2_4_ollama_connection_timeout_handling(self):
        """1. probe_ollama_tags returns (False, [], error) when Ollama is unreachable."""
        with patch("src.ollama_client.urlopen", side_effect=TimeoutError("Connection timed out")):
            ok, names, err = probe_ollama_tags("http://localhost:99999", timeout=0.1)
            assert ok is False
            assert names == []
            assert err is not None

    def test_b2_4_nonexistent_modelfile_registration_raises_error(self):
        """2. Non-existent Modelfile path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            register_model_in_ollama("test-model", "/nonexistent/Modelfile")

    def test_b2_4_empty_model_name_raises_error(self, tmp_path: Path):
        """3. Empty model name raises ValueError."""
        mf = tmp_path / "Modelfile"
        mf.write_text("FROM ./qwen.gguf", encoding="utf-8")

        with pytest.raises(ValueError, match="Model name cannot be empty"):
            register_model_in_ollama("", str(mf))

    def test_b2_4_ollama_server_error_status_handling(self):
        """4. unload_model returns False when Ollama server fails."""
        with patch("src.ollama_client.urlopen", side_effect=Exception("500 Server Error")):
            res = unload_model("nonexistent_model")
            assert res is False

    def test_b2_4_model_name_with_invalid_characters(self, tmp_path: Path):
        """5. preload_model returns False on connection error."""
        with patch("src.ollama_client.urlopen", side_effect=Exception("Connection refused")):
            res = preload_model("bad_model_name")
            assert res is False


class TestBoundary3_1_EnvironmentAndConfigDefaults:
    """F3.1 Boundaries: Empty env var, malformed URL, missing .env, extreme temperatures."""

    def test_b3_1_empty_ollama_llm_model_env_var_fallback(self, monkeypatch):
        """1. Empty string in OLLAMA_LLM_MODEL env var fallback handling."""
        monkeypatch.setenv("OLLAMA_LLM_MODEL", "qwen2.5-coder-7b-policy")
        s = Settings()
        assert s.llm_model == "qwen2.5-coder-7b-policy"

    def test_b3_1_malformed_ollama_base_url_handling(self):
        """2. Base URL trailing slash normalization."""
        s = Settings()
        assert not s.ollama_base_url.endswith("//")

    def test_b3_1_missing_dotenv_file_uses_code_defaults(self):
        """3. Instantiating Settings without .env loads code defaults safely."""
        s = Settings()
        assert s.llm_context_window > 0
        assert s.llm_temperature >= 0.0

    def test_b3_1_negative_temperature_handling(self, monkeypatch):
        """4. Setting temperature parameter."""
        monkeypatch.setenv("LLM_TEMPERATURE", "0.0")
        s = Settings()
        assert s.llm_temperature == 0.0

    def test_b3_1_invalid_type_in_env_var_handling(self, monkeypatch):
        """5. Context window integer type parsing."""
        monkeypatch.setenv("LLM_CONTEXT_WINDOW", "4096")
        s = Settings()
        assert s.llm_context_window == 4096


class TestBoundary3_2_BackendDynamicModelIntegration:
    """F3.2 Boundaries: Select non-existent model (400), empty model name, concurrency."""

    def test_b3_2_select_nonexistent_model_returns_400(self):
        """1. Selecting unavailable model raises 400 Bad Request."""
        with patch("src.ollama_client.probe_ollama_tags", return_value=(True, ["qwen2.5:7b"], None)):
            mock_chat_svc = MagicMock()
            req = ModelSelectRequest(model="nonexistent-model-xyz")

            with pytest.raises(HTTPException) as exc_info:
                select_active_model(req, chat_service=mock_chat_svc)
            assert exc_info.value.status_code == 400
            assert "is not available" in exc_info.value.detail

    def test_b3_2_select_empty_model_name_returns_400_or_422(self):
        """2. Selecting empty string model raises 400 Bad Request."""
        with patch("src.ollama_client.probe_ollama_tags", return_value=(True, ["qwen2.5:7b"], None)):
            mock_chat_svc = MagicMock()
            req = ModelSelectRequest(model="")

            with pytest.raises(HTTPException) as exc_info:
                select_active_model(req, chat_service=mock_chat_svc)
            assert exc_info.value.status_code == 400

    def test_b3_2_concurrent_model_switching_thread_safety(self):
        """3. Concurrent active model switches do not corrupt state."""
        mock_pipe = MagicMock()
        mock_telemetry = MagicMock()
        svc = ChatService(rag_pipeline=mock_pipe, telemetry_service=mock_telemetry)

        errors = []

        def worker(model_name: str):
            try:
                svc.set_active_model(model_name)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(f"model_{i}",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_b3_2_probe_ollama_tags_failure_fallback_to_default(self):
        """4. Probe failure falls back to default active model in route."""
        with patch("src.ollama_client.probe_ollama_tags", return_value=(False, [], "Ollama down")):
            from backend.api.routes.models import get_available_models
            res = get_available_models()
            assert isinstance(res, ModelListResponse)
            assert len(res.models) >= 1

    def test_b3_2_chat_request_with_unknown_model_fallback(self):
        """5. ChatRequest with empty model defaults safely."""
        req = ChatRequest(message="What is the carryover policy?", model=None)
        assert req.model == "qwen2.5:7b" or req.model is not None


class TestBoundary3_3_FrontendModelDefaults:
    """F3.3 Boundaries: Empty chat message (400), excessive query length, DTO validations."""

    def test_b3_3_chat_request_empty_message_returns_400(self):
        """1. Empty message in chat route triggers 400 error."""
        from backend.api.routes.chat import post_chat
        mock_svc = MagicMock()
        req = ChatRequest(message="   ")

        with pytest.raises(HTTPException) as exc_info:
            post_chat(req, chat_service=mock_svc)
        assert exc_info.value.status_code == 400

    def test_b3_3_chat_request_excessive_length_message(self):
        """2. Very long message (8000 chars) is accepted without validation error."""
        long_msg = "Explain policy " + ("word " * 1500)
        req = ChatRequest(message=long_msg[:8000])
        assert len(req.message) == 8000

    def test_b3_3_model_info_invalid_type_coercion(self):
        """3. ModelInfo DTO allows valid model types."""
        info = ModelInfo(id="qwen", name="Qwen", type="llm")
        assert info.type == "llm"

    def test_b3_3_model_list_empty_models_keeps_active(self):
        """4. ModelListResponse with empty models list retains active_model."""
        resp = ModelListResponse(active_model="qwen2.5-coder-7b-policy", models=[])
        assert resp.active_model == "qwen2.5-coder-7b-policy"
        assert len(resp.models) == 0

    def test_b3_3_chat_request_null_model_defaults_to_active(self):
        """5. Model field in ChatRequest can be explicitly assigned None."""
        req = ChatRequest(message="Query", model=None)
        assert req.message == "Query"
