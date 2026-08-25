"""Unit tests for LoRA/QLoRA trainer architecture, metrics, and PEFT configuration."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from company_policy_rag.scripts.finetune_qwen_coder import (
    build_arg_parser,
    create_fine_tune_config,
    load_config_file,
    main as cli_main,
    parse_cli_args,
    parse_target_modules,
)
from company_policy_rag.src.finetuning.trainer import (
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


# ── 1. FineTuneConfig Tests ──────────────────────────────────────────────────

def test_finetune_config_defaults():
    config = FineTuneConfig()
    assert config.model_name_or_path == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert config.quantization == "4bit"
    assert config.lora_r == 16
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.05
    assert config.learning_rate == 2e-4
    assert config.val_split == 0.1
    assert config.seed == 42
    assert "q_proj" in config.target_modules
    assert "down_proj" in config.target_modules
    assert len(config.target_modules) == 7


def test_finetune_config_custom_values():
    config = FineTuneConfig(
        dataset_path="data/custom.jsonl",
        lora_r=32,
        lora_alpha=64,
        learning_rate=1e-4,
        quantization="8bit",
        target_modules=["q_proj", "v_proj"],
    )
    d = config.to_dict()
    assert d["dataset_path"] == "data/custom.jsonl"
    assert d["lora_r"] == 32
    assert d["lora_alpha"] == 64
    assert d["learning_rate"] == 1e-4
    assert d["quantization"] == "8bit"
    assert d["target_modules"] == ["q_proj", "v_proj"]


def test_finetune_config_from_dict_and_yaml(tmp_path: Path):
    data = {
        "dataset_path": "data/test.jsonl",
        "lora_r": 8,
        "quantization": "none",
        "unknown_junk_param": 12345,
    }
    config = FineTuneConfig.from_dict(data)
    assert config.dataset_path == "data/test.jsonl"
    assert config.lora_r == 8
    assert config.quantization == "none"

    yaml_file = tmp_path / "test_config.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        f.write("dataset_path: data/yaml_test.jsonl\nlora_r: 64\nquantization: 4bit\n")

    yaml_config = FineTuneConfig.from_yaml(yaml_file)
    assert yaml_config.dataset_path == "data/yaml_test.jsonl"
    assert yaml_config.lora_r == 64
    assert yaml_config.quantization == "4bit"


def test_finetune_config_validation_invalid_quant():
    config = FineTuneConfig(dataset_path="data/test.jsonl", quantization="16bit")
    with pytest.raises(ValueError, match="Invalid quantization mode"):
        config.validate()


def test_finetune_config_validation_missing_dataset():
    config = FineTuneConfig(dataset_path="")
    with pytest.raises(ValueError, match="`dataset_path` must be specified"):
        config.validate()


def test_finetune_config_validation_invalid_split():
    config = FineTuneConfig(dataset_path="data/test.jsonl", val_split=1.5)
    with pytest.raises(ValueError, match="val_split"):
        config.validate()


def test_finetune_config_validation_invalid_lr():
    config = FineTuneConfig(dataset_path="data/test.jsonl", learning_rate=-0.01)
    with pytest.raises(ValueError, match="learning_rate"):
        config.validate()


# ── 2. Perplexity Calculation Tests ──────────────────────────────────────────

def test_calculate_perplexity_normal():
    assert calculate_perplexity(0.0) == 1.0
    assert abs(calculate_perplexity(1.0) - math.e) < 1e-6
    assert abs(calculate_perplexity(2.5) - math.exp(2.5)) < 1e-4


def test_calculate_perplexity_overflow_guard():
    assert calculate_perplexity(150.0) == float("inf")
    assert calculate_perplexity(1000.0) == float("inf")


def test_calculate_perplexity_edge_cases():
    assert math.isnan(calculate_perplexity(None))
    assert math.isnan(calculate_perplexity(float("nan")))
    assert math.isnan(calculate_perplexity(-1.0))


# ── 3. TrainingMetricsCallback Tests ─────────────────────────────────────────

def test_training_metrics_callback_on_log(tmp_path: Path):
    cb = TrainingMetricsCallback(output_dir=tmp_path)

    state = MagicMock()
    state.global_step = 10
    state.epoch = 0.5

    logs = {"loss": 1.5, "learning_rate": 2e-4}
    cb.on_log(args=None, state=state, control=None, logs=logs)

    assert len(cb.history) == 1
    assert cb.history[0]["step"] == 10
    assert cb.history[0]["loss"] == 1.5
    assert "train_perplexity" in cb.history[0]
    assert abs(cb.history[0]["train_perplexity"] - math.exp(1.5)) < 1e-4

    history_file = tmp_path / "training_history.json"
    assert history_file.is_file()
    with open(history_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 1


def test_training_metrics_callback_on_train_end(tmp_path: Path):
    cb = TrainingMetricsCallback(output_dir=tmp_path)

    state = MagicMock()
    state.global_step = 50
    state.epoch = 3.0
    state.best_global_step = 40

    cb.history = [
        {"step": 10, "loss": 2.0, "eval_loss": 1.8},
        {"step": 20, "loss": 1.5, "eval_loss": 1.2},
        {"step": 30, "loss": 1.0, "eval_loss": 0.9},
        {"step": 40, "loss": 0.8, "eval_loss": 0.75},
        {"step": 50, "loss": 0.7, "eval_loss": 0.80},
    ]

    cb.on_train_end(args=None, state=state, control=None)

    assert cb.summary["best_eval_loss"] == 0.75
    assert cb.summary["final_eval_loss"] == 0.80
    assert cb.summary["final_train_loss"] == 0.7
    assert cb.summary["total_steps"] == 50

    summary_file = tmp_path / "metrics_summary.json"
    assert summary_file.is_file()
    with open(summary_file, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    assert summary_data["best_eval_loss"] == 0.75


# ── 4. Tokenizer & PEFT Setup Tests ──────────────────────────────────────────

def test_setup_tokenizer_right_padding():
    mock_tok = MagicMock()
    mock_tok.pad_token = None
    mock_tok.eos_token = "<|endoftext|>"
    mock_tok.pad_token_id = None
    mock_tok.eos_token_id = 151643
    mock_tok.chat_template = None

    with patch("company_policy_rag.src.finetuning.trainer.AutoTokenizer.from_pretrained", return_value=mock_tok):
        config = FineTuneConfig()
        tokenizer = setup_tokenizer(config)
        assert tokenizer.padding_side == "right"
        assert tokenizer.pad_token == "<|endoftext|>"
        assert tokenizer.pad_token_id == 151643
        assert tokenizer.chat_template is not None


def test_setup_peft_config_target_modules():
    config = FineTuneConfig(lora_r=32, lora_alpha=64, lora_dropout=0.1)
    peft_config = setup_peft_config(config)
    assert peft_config.r == 32
    assert peft_config.lora_alpha == 64
    assert peft_config.lora_dropout == 0.1
    assert set(peft_config.target_modules) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }


def test_setup_model_quantization_cpu_guard():
    config = FineTuneConfig(quantization="4bit", smoke_test=False, use_cpu=False)
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
            setup_model(config, MagicMock())


def test_setup_model_smoke_test_cpu_downgrade():
    config = FineTuneConfig(quantization="4bit", smoke_test=True, use_cpu=False)
    mock_model = MagicMock()
    with patch("torch.cuda.is_available", return_value=False), \
         patch("company_policy_rag.src.finetuning.trainer.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mock_from_pretrained:
        model = setup_model(config, MagicMock())
        assert model == mock_model
        # Ensure quantization_config is None on CPU fallback
        kwargs = mock_from_pretrained.call_args.kwargs
        assert "quantization_config" not in kwargs or kwargs["quantization_config"] is None


# ── 5. Completion Collator & Prompt Formatting Tests ─────────────────────────

def test_formatting_prompts_func():
    mock_tok = MagicMock()
    mock_tok.apply_chat_template.return_value = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>\n"

    example = {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
    }
    result = formatting_prompts_func(example, mock_tok)
    assert "<|im_start|>user" in result
    mock_tok.apply_chat_template.assert_called_once_with(example["messages"], tokenize=False, add_generation_prompt=False)


def test_get_completion_data_collator():
    mock_tok = MagicMock()
    collator = get_completion_data_collator(mock_tok)
    assert collator.response_template == "<|im_start|>assistant\n"


# ── 6. CLI and Config Parsing Tests ──────────────────────────────────────────

def test_parse_target_modules():
    assert parse_target_modules("q_proj,v_proj") == ["q_proj", "v_proj"]
    assert parse_target_modules(["gate_proj", "up_proj"]) == ["gate_proj", "up_proj"]
    assert len(parse_target_modules(None)) == 7


def test_cli_parsing_and_yaml_override(tmp_path: Path):
    yaml_content = {
        "dataset_path": "data/sample.jsonl",
        "lora_r": 16,
        "learning_rate": 0.0002,
        "num_train_epochs": 3.0,
    }
    yaml_file = tmp_path / "config.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        json.dump(yaml_content, f)

    # CLI overrides learning_rate
    argv = ["--config", str(yaml_file), "--learning_rate", "0.0001", "--smoke_test"]
    args = parse_cli_args(argv)
    config = create_fine_tune_config(args)

    assert config.dataset_path == "data/sample.jsonl"
    assert config.lora_r == 16
    assert config.learning_rate == 0.0001
    assert config.smoke_test is True


def test_cli_dry_run_execution(tmp_path: Path):
    data_file = tmp_path / "sample.jsonl"
    with open(data_file, "w", encoding="utf-8") as f:
        f.write('{"instruction": "What is RAG?", "output": "Retrieval-Augmented Generation"}\n')

    argv = ["--dataset_path", str(data_file), "--dry-run", "--val_split", "0.0"]
    ret = cli_main(argv)
    assert ret == 0


# ── 7. End-to-End Mock Training Execution ────────────────────────────────────

def test_train_lora_end_to_end_mock(tmp_path: Path):
    data_file = tmp_path / "sample.jsonl"
    with open(data_file, "w", encoding="utf-8") as f:
        f.write('{"instruction": "Q1", "output": "A1"}\n')
        f.write('{"instruction": "Q2", "output": "A2"}\n')

    out_dir = tmp_path / "out"

    config = FineTuneConfig(
        dataset_path=str(data_file),
        output_dir=str(out_dir),
        val_split=0.0,
        smoke_test=True,
        use_cpu=True,
    )

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "<|endoftext|>"
    mock_tokenizer.eos_token = "<|endoftext|>"
    mock_tokenizer.pad_token_id = 151643
    mock_tokenizer.eos_token_id = 151643
    mock_tokenizer.chat_template = "template"

    mock_model = MagicMock()
    mock_model.parameters.return_value = [torch.zeros(10, requires_grad=True)]
    mock_peft_model = MagicMock()
    mock_peft_model.get_nb_trainable_parameters.return_value = (10, 100)
    mock_peft_model.parameters.return_value = [torch.zeros(10, requires_grad=True)]

    mock_train_result = MagicMock()
    mock_train_result.training_loss = 0.5
    mock_train_result.metrics = {"train_runtime": 1.23}

    mock_trainer = MagicMock()
    mock_trainer.train.return_value = mock_train_result
    mock_trainer.evaluate.return_value = {"eval_loss": 0.6}
    mock_trainer.model = mock_peft_model

    with patch("company_policy_rag.src.finetuning.trainer.setup_tokenizer", return_value=mock_tokenizer), \
         patch("company_policy_rag.src.finetuning.trainer.setup_model", return_value=mock_model), \
         patch("company_policy_rag.src.finetuning.trainer.get_peft_model", return_value=mock_peft_model), \
         patch("company_policy_rag.src.finetuning.trainer.SFTTrainer", return_value=mock_trainer):

        result = train_lora(config)

        assert result["status"] == "success"
        assert result["output_dir"] == str(out_dir)
        assert result["train_loss"] == 0.5
        assert result["trainable_params"] == 10
        assert result["all_params"] == 100
        assert (out_dir / "training_config.json").is_file()
        assert (out_dir / "chat_template.json").is_file()
