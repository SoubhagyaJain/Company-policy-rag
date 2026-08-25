"""Adversarial stress test harness for LoRA/QLoRA trainer and CLI."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import AutoTokenizer

from company_policy_rag.scripts.finetune_qwen_coder import (
    build_arg_parser,
    create_fine_tune_config,
    load_config_file,
    main as cli_main,
    parse_cli_args,
    parse_target_modules,
    resolve_dataset_path,
)
from company_policy_rag.src.finetuning.trainer import (
    DEFAULT_CHATML_TEMPLATE,
    FineTuneConfig,
    TrainingMetricsCallback,
    calculate_perplexity,
    formatting_prompts_func,
    get_completion_data_collator,
    setup_model,
    setup_peft_config,
    setup_tokenizer,
    train_lora,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Adversarial Perplexity Calculation Stress Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPerplexityAdversarial:
    """Stress-test numerical stability and edge cases of calculate_perplexity."""

    @pytest.mark.parametrize(
        "loss_input,expected_type,expected_value",
        [
            # Standard & boundary values
            (0.0, float, 1.0),
            (1.0, float, math.e),
            (2.0, float, math.exp(2.0)),
            (10.0, float, math.exp(10.0)),
            (50.0, float, math.exp(50.0)),
            (100.0, float, math.exp(100.0)),
            # Extreme floats & overflow boundary
            (100.0001, float, float("inf")),
            (150.0, float, float("inf")),
            (709.78, float, float("inf")),
            (800.0, float, float("inf")),
            (1e9, float, float("inf")),
            (float("inf"), float, float("inf")),
            # Underflow / Small values
            (1e-12, float, math.exp(1e-12)),
            (-0.0, float, 1.0),
            # Invalid / Negative values (cross entropy loss cannot be negative)
            (-0.0001, float, float("nan")),
            (-1.0, float, float("nan")),
            (-100.0, float, float("nan")),
            (float("-inf"), float, float("nan")),
            # NaN / None inputs
            (float("nan"), float, float("nan")),
            (None, float, float("nan")),
        ],
    )
    def test_perplexity_values(self, loss_input, expected_type, expected_value):
        result = calculate_perplexity(loss_input)
        assert isinstance(result, expected_type)
        if math.isnan(expected_value):
            assert math.isnan(result), f"Expected NaN for input {loss_input}, got {result}"
        elif math.isinf(expected_value):
            assert math.isinf(result) and result > 0, f"Expected +inf for input {loss_input}, got {result}"
        else:
            assert abs(result - expected_value) < 1e-5, f"Expected {expected_value} for {loss_input}, got {result}"

    def test_perplexity_type_robustness(self):
        """Test behavior when non-float types are erroneously passed."""
        # Ints should work transparently
        assert calculate_perplexity(0) == 1.0
        assert calculate_perplexity(2) == math.exp(2)
        assert calculate_perplexity(200) == float("inf")
        assert math.isnan(calculate_perplexity(-5))


# ══════════════════════════════════════════════════════════════════════════════
# 2. FineTuneConfig Validation & Boundary Stress Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFineTuneConfigAdversarial:
    """Stress-test configuration validation against invalid inputs and boundary conditions."""

    def test_missing_dataset_path_non_smoke(self):
        config = FineTuneConfig(dataset_path="", smoke_test=False)
        with pytest.raises(ValueError, match="`dataset_path` must be specified"):
            config.validate()

    def test_missing_dataset_path_smoke_test_allowed(self):
        config = FineTuneConfig(dataset_path="", smoke_test=True)
        # Should not raise
        config.validate()

    @pytest.mark.parametrize("invalid_quant", ["16bit", "int4", "fp16", "bf16", "", "4", "8", None, 123])
    def test_invalid_quantization_values(self, invalid_quant):
        config = FineTuneConfig(dataset_path="data/test.jsonl", quantization=invalid_quant)
        with pytest.raises(ValueError, match="Invalid quantization mode"):
            config.validate()

    @pytest.mark.parametrize("valid_quant", ["4bit", "8bit", "none"])
    def test_valid_quantization_values(self, valid_quant):
        config = FineTuneConfig(dataset_path="data/test.jsonl", quantization=valid_quant)
        config.validate()

    @pytest.mark.parametrize("invalid_split", [-0.1, -1.0, 1.0, 1.1, 2.0, 100.0])
    def test_invalid_val_split_ranges(self, invalid_split):
        config = FineTuneConfig(dataset_path="data/test.jsonl", val_split=invalid_split)
        with pytest.raises(ValueError, match="val_split"):
            config.validate()

    @pytest.mark.parametrize("valid_split", [0.0, 0.01, 0.1, 0.5, 0.999])
    def test_valid_val_split_ranges(self, valid_split):
        config = FineTuneConfig(dataset_path="data/test.jsonl", val_split=valid_split)
        config.validate()

    @pytest.mark.parametrize("invalid_batch_size", [0, -1, -8])
    def test_invalid_batch_sizes(self, invalid_batch_size):
        config = FineTuneConfig(dataset_path="data/test.jsonl", per_device_train_batch_size=invalid_batch_size)
        with pytest.raises(ValueError, match="per_device_train_batch_size"):
            config.validate()

    @pytest.mark.parametrize("invalid_lr", [0.0, -0.0, -1e-4, -1.0])
    def test_invalid_learning_rates(self, invalid_lr):
        config = FineTuneConfig(dataset_path="data/test.jsonl", learning_rate=invalid_lr)
        with pytest.raises(ValueError, match="learning_rate"):
            config.validate()

    @pytest.mark.parametrize("invalid_lora_r", [0, -1, -16])
    def test_invalid_lora_r(self, invalid_lora_r):
        config = FineTuneConfig(dataset_path="data/test.jsonl", lora_r=invalid_lora_r)
        with pytest.raises(ValueError, match="lora_r"):
            config.validate()

    def test_empty_target_modules(self):
        config = FineTuneConfig(dataset_path="data/test.jsonl", target_modules=[])
        with pytest.raises(ValueError, match="target_modules"):
            config.validate()

    def test_from_dict_unknown_keys_filtering(self):
        malformed_dict = {
            "dataset_path": "data/test.jsonl",
            "lora_r": 32,
            "__injected_malicious_key": True,
            "extra_unsupported_param": [1, 2, 3],
        }
        config = FineTuneConfig.from_dict(malformed_dict)
        assert config.dataset_path == "data/test.jsonl"
        assert config.lora_r == 32
        assert not hasattr(config, "__injected_malicious_key")
        assert not hasattr(config, "extra_unsupported_param")

    def test_from_yaml_empty_file(self, tmp_path: Path):
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("", encoding="utf-8")
        config = FineTuneConfig.from_yaml(empty_yaml)
        assert config.model_name_or_path == "Qwen/Qwen2.5-Coder-7B-Instruct"


# ══════════════════════════════════════════════════════════════════════════════
# 3. CLI Argument Parsing, Overrides & Flag Stress Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCLIAdversarial:
    """Stress-test CLI two-phase parsing, YAML overriding, and error traps."""

    def test_cli_yaml_override_all_key_hyperparameters(self, tmp_path: Path):
        yaml_data = {
            "dataset_path": "base_data.jsonl",
            "model_name_or_path": "base_model",
            "learning_rate": 0.001,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "quantization": "8bit",
            "num_train_epochs": 1.0,
            "per_device_train_batch_size": 4,
            "val_split": 0.2,
            "output_dir": "./base_out",
        }
        config_path = tmp_path / "base_config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(yaml_data, f)

        # Override via CLI
        argv = [
            "--config", str(config_path),
            "--dataset_path", "override_data.jsonl",
            "--learning_rate", "5e-5",
            "--lora_r", "64",
            "--lora_alpha", "128",
            "--quantization", "4bit",
            "--num_train_epochs", "5.0",
            "--per_device_train_batch_size", "1",
            "--val_split", "0.05",
            "--output_dir", "./override_out",
            "--smoke_test",
        ]
        args = parse_cli_args(argv)
        config = create_fine_tune_config(args)

        assert config.dataset_path == "override_data.jsonl"
        assert config.learning_rate == 5e-5
        assert config.lora_r == 64
        assert config.lora_alpha == 128
        assert config.quantization == "4bit"
        assert config.num_train_epochs == 5.0
        assert config.per_device_train_batch_size == 1
        assert config.val_split == 0.05
        assert config.output_dir == "./override_out"
        assert config.smoke_test is True

    @pytest.mark.parametrize(
        "input_modules,expected_list",
        [
            ("q_proj,v_proj", ["q_proj", "v_proj"]),
            ("  q_proj  ,  k_proj ,   v_proj  ", ["q_proj", "k_proj", "v_proj"]),
            ("gate_proj", ["gate_proj"]),
            (["q_proj", "up_proj"], ["q_proj", "up_proj"]),
            ("", []),  # Empty string returns empty list (which FineTuneConfig rejects during validation)
            (None, ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        ],
    )
    def test_parse_target_modules_variations(self, input_modules, expected_list):
        assert parse_target_modules(input_modules) == expected_list

    def test_cli_invalid_config_path_raises(self):
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config_file("totally_non_existent_config_path_xyz.yaml")

    def test_cli_dry_run_exits_cleanly(self, tmp_path: Path):
        data_file = tmp_path / "valid.jsonl"
        with open(data_file, "w", encoding="utf-8") as f:
            f.write('{"instruction": "What is security?", "output": "Security is important."}\n')

        argv = ["--dataset_path", str(data_file), "--dry-run", "--val_split", "0.0"]
        exit_code = cli_main(argv)
        assert exit_code == 0

    def test_cli_missing_dataset_returns_error_code(self):
        argv = ["--dataset_path", "non_existent_file_12345.jsonl", "--dry-run"]
        exit_code = cli_main(argv)
        assert exit_code == 1


# ══════════════════════════════════════════════════════════════════════════════
# 4. TrainingMetricsCallback Resilience & Serialization Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMetricsCallbackAdversarial:
    """Stress-test metrics logging callback against IO errors and serialization edge cases."""

    def test_callback_handles_file_write_permission_error_gracefully(self, tmp_path: Path):
        cb = TrainingMetricsCallback(output_dir=tmp_path)
        state = MagicMock(global_step=1, epoch=0.1, best_global_step=None)

        # Mock open to raise PermissionError during on_log
        with patch("builtins.open", side_effect=PermissionError("Permission Denied")):
            # Must not raise / must not crash training
            cb.on_log(args=None, state=state, control=None, logs={"loss": 2.0})

        # History should still be updated in-memory
        assert len(cb.history) == 1
        assert cb.history[0]["step"] == 1
        assert cb.history[0]["loss"] == 2.0

    def test_callback_handles_on_train_end_write_error_gracefully(self, tmp_path: Path):
        cb = TrainingMetricsCallback(output_dir=tmp_path)
        state = MagicMock(global_step=10, epoch=1.0, best_global_step=10)
        cb.history = [{"step": 10, "loss": 1.0, "eval_loss": 0.8}]

        with patch("builtins.open", side_effect=OSError("Disk Full")):
            # Must not crash
            cb.on_train_end(args=None, state=state, control=None)

        assert cb.summary["best_eval_loss"] == 0.8

    def test_callback_serialization_with_inf_and_nan_losses(self, tmp_path: Path):
        cb = TrainingMetricsCallback(output_dir=tmp_path)
        state = MagicMock(global_step=1, epoch=0.1, best_global_step=None)

        logs_nan = {"loss": float("nan"), "eval_loss": 150.0}
        cb.on_log(args=None, state=state, control=None, logs=logs_nan)

        history_file = tmp_path / "training_history.json"
        assert history_file.is_file()

        with open(history_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data) == 1
        # eval_loss was 150.0 -> eval_perplexity should be inf
        assert data[0]["eval_perplexity"] == float("inf")

    def test_callback_empty_history_on_train_end(self, tmp_path: Path):
        cb = TrainingMetricsCallback(output_dir=tmp_path)
        state = MagicMock(global_step=0, epoch=0.0, best_global_step=None)

        # on_train_end called with 0 logs
        cb.on_train_end(args=None, state=state, control=None)

        assert cb.summary["best_eval_loss"] is None
        assert math.isnan(cb.summary["best_eval_perplexity"])
        assert cb.summary["final_train_loss"] is None
        assert cb.summary["final_eval_loss"] is None
        assert cb.summary["history_entries"] == 0

        summary_file = tmp_path / "metrics_summary.json"
        assert summary_file.is_file()
        with open(summary_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["best_eval_loss"] is None
        assert data["best_global_step"] is None


# ══════════════════════════════════════════════════════════════════════════════
# 5. Tokenizer Alignment & Data Collator Prompt Masking Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTokenizerAndCollatorMasking:
    """Verify tokenizer padding side and prompt token loss masking with -100."""

    def test_tokenizer_padding_side_strictly_right(self):
        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<|endoftext|>"
        mock_tok.pad_token_id = None
        mock_tok.eos_token_id = 151643
        mock_tok.chat_template = None

        with patch("company_policy_rag.src.finetuning.trainer.AutoTokenizer.from_pretrained", return_value=mock_tok):
            config = FineTuneConfig()
            tok = setup_tokenizer(config)
            assert tok.padding_side == "right", "Causal LM training requires right-padding"

    def test_completion_data_collator_masks_prompt_with_minus_100(self):
        """Empirically verify that DataCollatorForCompletionOnlyLM masks prompt tokens with -100."""
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception:
            pytest.skip("gpt2 tokenizer not downloadable in offline environment")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        response_template = "### Response:\n"
        collator = get_completion_data_collator(tokenizer, response_template=response_template)

        full_text = "### Instruction:\nWhat is policy X?\n### Response:\nPolicy X requires badge at all times."
        encoded = tokenizer(full_text)
        input_ids = encoded["input_ids"]

        batch = collator([{"input_ids": input_ids, "attention_mask": encoded["attention_mask"]}])

        labels = batch["labels"][0].tolist()
        response_text_idx = full_text.index(response_template) + len(response_template)
        response_text = full_text[response_text_idx:]
        response_tokens = tokenizer.encode(response_text, add_special_tokens=False)

        num_response_tokens = len(response_tokens)
        prompt_labels = labels[:-num_response_tokens]
        response_labels = labels[-num_response_tokens:]

        assert all(label == -100 for label in prompt_labels), f"Expected all prompt tokens to be -100, got {prompt_labels}"
        assert any(label != -100 for label in response_labels), f"Expected response tokens to have valid IDs, got {response_labels}"

    def test_multi_turn_collator_masking_behavior(self):
        """Verify collator behavior on multi-turn dialogue containing multiple response templates."""
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception:
            pytest.skip("gpt2 tokenizer not downloadable in offline environment")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        response_template = "\n### Assistant:\n"
        collator = get_completion_data_collator(tokenizer, response_template=response_template)

        multi_turn_text = (
            "### User:\nHello\n### Assistant:\nHi there!\n"
            "### User:\nWhat is policy 101?\n### Assistant:\nPolicy 101 covers vacation time."
        )
        encoded = tokenizer(multi_turn_text)
        batch = collator([{"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}])
        labels = batch["labels"][0].tolist()

        # Both user instructions must be masked with -100, and assistant responses unmasked
        assert -100 in labels, "Prompt tokens must be masked with -100"
        # There should be non-masked tokens corresponding to completions
        unmasked_tokens = [tok for tok in labels if tok != -100]
        assert len(unmasked_tokens) > 0, "Assistant response tokens must have valid token IDs in labels"


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLI Aliases & Rapid Logging Stress Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCLIAliasesAndStress:
    """Stress-test CLI flags, aliases, and high-frequency logging."""

    def test_cli_argument_aliases(self):
        argv = [
            "--train_batch_size", "8",
            "--eval_batch_size", "4",
            "--gradient_accum", "16",
            "--lr", "3e-4",
            "--smoke_test",
        ]
        args = parse_cli_args(argv)
        assert args.per_device_train_batch_size == 8
        assert args.per_device_eval_batch_size == 4
        assert args.gradient_accumulation_steps == 16
        assert args.learning_rate == 3e-4
        assert args.smoke_test is True

    def test_high_frequency_metrics_logging(self, tmp_path: Path):
        cb = TrainingMetricsCallback(output_dir=tmp_path)
        for step in range(1, 101):
            state = MagicMock(global_step=step, epoch=step * 0.1, best_global_step=None)
            loss_val = max(0.1, 2.5 - step * 0.02)
            eval_loss = max(0.15, 2.0 - step * 0.015) if step % 10 == 0 else None
            logs = {"loss": loss_val}
            if eval_loss is not None:
                logs["eval_loss"] = eval_loss
            cb.on_log(args=None, state=state, control=None, logs=logs)

        assert len(cb.history) == 100
        history_file = tmp_path / "training_history.json"
        assert history_file.is_file()
        with open(history_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 100
        assert data[-1]["step"] == 100


# ══════════════════════════════════════════════════════════════════════════════
# 7. End-to-End Stress Run
# ══════════════════════════════════════════════════════════════════════════════

def test_trainer_end_to_end_adversarial_workflow(tmp_path: Path):
    """Test full trainer orchestrator under smoke_test with synthetic dataset."""
    dataset_file = tmp_path / "smoke_dataset.jsonl"
    with open(dataset_file, "w", encoding="utf-8") as f:
        for i in range(5):
            f.write(json.dumps({"instruction": f"Prompt {i}", "output": f"Answer {i}"}) + "\n")

    output_dir = tmp_path / "output_test"

    config = FineTuneConfig(
        dataset_path=str(dataset_file),
        output_dir=str(output_dir),
        val_split=0.2,
        smoke_test=True,
        use_cpu=True,
        quantization="none",
        learning_rate=1e-4,
        num_train_epochs=1.0,
    )

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "<|endoftext|>"
    mock_tokenizer.eos_token = "<|endoftext|>"
    mock_tokenizer.pad_token_id = 151643
    mock_tokenizer.eos_token_id = 151643
    mock_tokenizer.chat_template = DEFAULT_CHATML_TEMPLATE

    mock_peft_model = MagicMock()
    mock_peft_model.get_nb_trainable_parameters.return_value = (1000, 50000)
    mock_peft_model.parameters.return_value = [torch.zeros(10, requires_grad=True)]

    mock_train_result = MagicMock()
    mock_train_result.training_loss = 0.42
    mock_train_result.metrics = {"train_runtime": 2.5}

    mock_trainer = MagicMock()
    mock_trainer.train.return_value = mock_train_result
    mock_trainer.evaluate.return_value = {"eval_loss": 0.38}
    mock_trainer.model = mock_peft_model

    with patch("company_policy_rag.src.finetuning.trainer.setup_tokenizer", return_value=mock_tokenizer), \
         patch("company_policy_rag.src.finetuning.trainer.setup_model", return_value=mock_peft_model), \
         patch("company_policy_rag.src.finetuning.trainer.get_peft_model", return_value=mock_peft_model), \
         patch("company_policy_rag.src.finetuning.trainer.SFTTrainer", return_value=mock_trainer):

        results = train_lora(config)

        assert results["status"] == "success"
        assert results["train_loss"] == 0.42
        assert results["eval_loss"] == 0.38
        assert abs(results["eval_perplexity"] - math.exp(0.38)) < 1e-4
        assert results["trainable_percent"] == 2.0
        assert (output_dir / "training_config.json").is_file()
        assert (output_dir / "chat_template.json").is_file()
