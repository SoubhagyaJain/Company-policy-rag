"""
Adversarial Stress Test Suite for Milestone 4 (CLI Scripts, Execution Flags & Config Boundaries).

Empirical challenger test harness targeting:
1. company_policy_rag/scripts/run_finetune_pipeline.py:
   - Invalid stage arguments, unknown flags, invalid choices
   - Non-existent dataset paths
   - Conflicting skip flags (--skip-train with missing adapter)
   - Custom/nested output directories
   - Dry-run mode resilience across flag combinations
   - Config file loading (valid, missing, corrupt)

2. company_policy_rag/scripts/export_and_register_ollama.py:
   - Invalid quantization choices
   - Malformed template strings and Modelfile boundary generation
   - Invalid base model paths and missing adapter paths
   - Custom output directories and flag combinations

3. company_policy_rag/scripts/finetune_qwen_coder.py:
   - Malformed YAML/JSON configuration files
   - Out-of-bounds learning rates, val_split, batch size, lora_r
   - Non-existent / corrupted dataset formats and empty datasets
   - Target module parsing boundaries
   - Perplexity calculation boundary stress

Authoritative Reference:
- ORIGINAL_REQUEST.md (§ R1, R2, R3, R4)
- PROJECT.md (§ Architecture, Feature Inventory, Interface Contracts)
- TEST_INFRA.md (§ Tier 2 & Tier 5 Adversarial Coverage Hardening)
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root and company_policy_rag root are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
for p in [str(PROJECT_ROOT), str(WORKSPACE_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

# ── Dynamic Import with Robust Multi-Tier Fallback ──────────────────────────
try:
    from company_policy_rag.scripts.run_finetune_pipeline import (
        build_arg_parser as build_pipeline_parser,
        load_config_file as load_pipeline_config,
        main as pipeline_main,
        parse_pipeline_args,
    )
    from company_policy_rag.scripts.export_and_register_ollama import (
        build_arg_parser as build_export_parser,
        main as export_main,
    )
    from company_policy_rag.scripts.finetune_qwen_coder import (
        build_arg_parser as build_finetune_parser,
        create_fine_tune_config,
        load_config_file as load_finetune_config,
        main as finetune_main,
        parse_cli_args as parse_finetune_args,
        parse_target_modules,
        resolve_dataset_path,
    )
    from company_policy_rag.src.finetuning.dataset_loader import (
        DatasetEmptyError,
        DatasetFormatValidationError,
        DatasetValidationError,
        compute_dataset_statistics,
        detect_format,
        load_dataset_from_file,
        normalize_record,
        sanitize_messages,
        split_dataset,
    )
    from company_policy_rag.src.finetuning.gguf_exporter import (
        QUANT_CANONICAL_MAP,
        SUPPORTED_QUANTIZATIONS,
        GGUFExporter,
        convert_to_gguf,
        normalize_quantization,
        validate_gguf_file,
    )
    from company_policy_rag.src.finetuning.merger import (
        ModelMerger,
        merge_lora_weights,
        resolve_device,
        resolve_torch_dtype,
    )
    from company_policy_rag.src.finetuning.modelfile_generator import (
        DEFAULT_ENTERPRISE_SYSTEM_PROMPT,
        ModelfileGenerator,
        generate_modelfile,
        normalize_gguf_path,
        parse_modelfile,
    )
    from company_policy_rag.src.finetuning.ollama_registrar import (
        OllamaRegistrar,
        find_ollama_binary,
        get_model_details,
        probe_ollama_tags,
        register_model_api,
        register_model_cli,
        register_model_in_ollama,
        verify_model_registered,
    )
    from company_policy_rag.src.finetuning.trainer import (
        FineTuneConfig,
        TrainingMetricsCallback,
        calculate_perplexity,
    )
except ImportError:
    from scripts.run_finetune_pipeline import (
        build_arg_parser as build_pipeline_parser,
        load_config_file as load_pipeline_config,
        main as pipeline_main,
        parse_pipeline_args,
    )
    from scripts.export_and_register_ollama import (
        build_arg_parser as build_export_parser,
        main as export_main,
    )
    from scripts.finetune_qwen_coder import (
        build_arg_parser as build_finetune_parser,
        create_fine_tune_config,
        load_config_file as load_finetune_config,
        main as finetune_main,
        parse_cli_args as parse_finetune_args,
        parse_target_modules,
        resolve_dataset_path,
    )
    from src.finetuning.dataset_loader import (
        DatasetEmptyError,
        DatasetFormatValidationError,
        DatasetValidationError,
        compute_dataset_statistics,
        detect_format,
        load_dataset_from_file,
        normalize_record,
        sanitize_messages,
        split_dataset,
    )
    from src.finetuning.gguf_exporter import (
        QUANT_CANONICAL_MAP,
        SUPPORTED_QUANTIZATIONS,
        GGUFExporter,
        convert_to_gguf,
        normalize_quantization,
        validate_gguf_file,
    )
    from src.finetuning.merger import (
        ModelMerger,
        merge_lora_weights,
        resolve_device,
        resolve_torch_dtype,
    )
    from src.finetuning.modelfile_generator import (
        DEFAULT_ENTERPRISE_SYSTEM_PROMPT,
        ModelfileGenerator,
        generate_modelfile,
        normalize_gguf_path,
        parse_modelfile,
    )
    from src.finetuning.ollama_registrar import (
        OllamaRegistrar,
        find_ollama_binary,
        get_model_details,
        probe_ollama_tags,
        register_model_api,
        register_model_cli,
        register_model_in_ollama,
        verify_model_registered,
    )
    from src.finetuning.trainer import (
        FineTuneConfig,
        TrainingMetricsCallback,
        calculate_perplexity,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 1. Pipeline CLI Stress Tests (`run_finetune_pipeline.py`)
# ═════════════════════════════════════════════════════════════════════════════

class TestRunFinetunePipelineStress:
    """Stress testing run_finetune_pipeline CLI for invalid args, missing paths, and skip combinations."""

    def test_pipeline_invalid_flag_raises_system_exit(self) -> None:
        """Unknown CLI flag triggers argparse error and SystemExit (code 2)."""
        parser = build_pipeline_parser()
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["--non_existent_unrecognized_flag"])
        assert excinfo.value.code == 2

    def test_pipeline_invalid_quantization_choice(self) -> None:
        """Invalid training quantization mode (e.g. 16bit, fp4) triggers SystemExit."""
        parser = build_pipeline_parser()
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["--quantization", "16bit"])
        assert excinfo.value.code == 2

    def test_pipeline_invalid_gguf_quantization_choice(self) -> None:
        """Invalid GGUF quantization mode (e.g. Q4_K_INVALID) triggers SystemExit."""
        parser = build_pipeline_parser()
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["--gguf_quantization", "Q4_K_INVALID"])
        assert excinfo.value.code == 2

    def test_pipeline_invalid_dataset_format_choice(self) -> None:
        """Invalid dataset format triggers SystemExit."""
        parser = build_pipeline_parser()
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["--dataset_format", "invalid_format_xyz"])
        assert excinfo.value.code == 2

    def test_pipeline_invalid_merge_device_choice(self) -> None:
        """Invalid merge device choice triggers SystemExit."""
        parser = build_pipeline_parser()
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["--merge_device", "tpu"])
        assert excinfo.value.code == 2

    def test_pipeline_invalid_merge_dtype_choice(self) -> None:
        """Invalid merge dtype choice triggers SystemExit."""
        parser = build_pipeline_parser()
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["--merge_dtype", "int8"])
        assert excinfo.value.code == 2

    def test_pipeline_nonexistent_dataset_returns_error(self, tmp_path: Path) -> None:
        """Passing non-existent dataset path to pipeline execution without dry-run returns 1."""
        fake_path = tmp_path / "does_not_exist_data.jsonl"
        rc = pipeline_main([
            "--dataset_path", str(fake_path),
            "--output_dir", str(tmp_path / "out"),
        ])
        assert rc == 1

    def test_pipeline_skip_train_with_missing_adapter(self, tmp_path: Path) -> None:
        """Pipeline with --skip-train handles missing adapter directory using robust fallback synthesizer."""
        out_dir = tmp_path / "skip_train_out"
        rc = pipeline_main([
            "--output_dir", str(out_dir),
            "--skip-train",
            "--skip-register",  # skip live ollama daemon registration
        ])
        assert rc == 0
        assert (out_dir / "merged" / "config.json").exists()
        assert (out_dir / "Modelfile").exists()

    def test_pipeline_all_stages_skipped(self, tmp_path: Path) -> None:
        """Pipeline with all skip flags enabled completes cleanly without errors."""
        out_dir = tmp_path / "all_skipped_out"
        rc = pipeline_main([
            "--output_dir", str(out_dir),
            "--skip-train",
            "--skip-merge",
            "--skip-quant",
            "--skip-register",
            "--skip-verify",
        ])
        assert rc == 0
        assert (out_dir / "Modelfile").exists()

    def test_pipeline_custom_deeply_nested_output_dir(self, tmp_path: Path) -> None:
        """Pipeline creates deeply nested output directory structure seamlessly."""
        deep_dir = tmp_path / "nested" / "sub1" / "sub2" / "deep_pipeline_out"
        rc = pipeline_main([
            "--output_dir", str(deep_dir),
            "--skip-train",
            "--skip-register",
        ])
        assert rc == 0
        assert deep_dir.exists()
        assert (deep_dir / "Modelfile").exists()

    def test_pipeline_dry_run_with_multiple_flags(self, tmp_path: Path) -> None:
        """Dry-run mode returns 0 across varied combinations of hyperparameter flags."""
        rc = pipeline_main([
            "--dry-run",
            "--model_name_or_path", "Qwen/Qwen2.5-Coder-7B-Instruct",
            "--quantization", "8bit",
            "--lora_r", "32",
            "--lora_alpha", "64",
            "--learning_rate", "1e-4",
            "--num_train_epochs", "5.0",
            "--gguf_quantization", "Q8_0",
            "--ollama_model_name", "qwen-custom-tag",
            "--num_ctx", "16384",
            "--temperature", "0.05",
        ])
        assert rc == 0

    def test_pipeline_config_file_nonexistent_raises(self) -> None:
        """load_config_file with non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_pipeline_config("non_existent_config_file.yaml")

    def test_pipeline_config_file_valid_yaml(self, tmp_path: Path) -> None:
        """load_config_file correctly loads and parses valid YAML configuration."""
        yaml_path = tmp_path / "valid_pipeline_config.yaml"
        yaml_path.write_text("model_name_or_path: custom/model\ngguf_quantization: Q8_0\n", encoding="utf-8")
        cfg = load_pipeline_config(yaml_path)
        assert cfg["model_name_or_path"] == "custom/model"
        assert cfg["gguf_quantization"] == "Q8_0"


# ═════════════════════════════════════════════════════════════════════════════
# 2. Export & Register CLI Stress Tests (`export_and_register_ollama.py`)
# ═════════════════════════════════════════════════════════════════════════════

class TestExportAndRegisterStress:
    """Stress testing export_and_register_ollama CLI for invalid quantizations, malformed Modelfile and paths."""

    def test_export_invalid_quantization_flag(self) -> None:
        """Invalid quantization argument to export CLI raises SystemExit code 2."""
        parser = build_export_parser()
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["--quantization", "UNSUPPORTED_QUANT_123"])
        assert excinfo.value.code == 2

    def test_export_invalid_device_choice(self) -> None:
        """Invalid device choice to export CLI raises SystemExit code 2."""
        parser = build_export_parser()
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["--device", "npu"])
        assert excinfo.value.code == 2

    def test_export_invalid_dtype_choice(self) -> None:
        """Invalid dtype choice to export CLI raises SystemExit code 2."""
        parser = build_export_parser()
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["--dtype", "int16"])
        assert excinfo.value.code == 2

    def test_export_main_empty_base_model_returns_error(self, tmp_path: Path) -> None:
        """Empty base_model_path returns exit code 1."""
        rc = export_main([
            "--base_model_path", "",
            "--adapter_path", str(tmp_path / "adapter"),
        ])
        assert rc == 1

    def test_export_main_skip_merge_and_skip_quant(self, tmp_path: Path) -> None:
        """Export CLI with --skip-merge and --skip-quant generates Modelfile and skips execution stages."""
        out_dir = tmp_path / "export_skip"
        rc = export_main([
            "--output_dir", str(out_dir),
            "--skip-merge",
            "--skip-quant",
            "--skip-register",
        ])
        assert rc == 0
        assert (out_dir / "Modelfile").exists()

    def test_modelfile_generation_empty_gguf_path_error(self) -> None:
        """generate_modelfile with empty gguf_path raises ValueError."""
        with pytest.raises(ValueError, match="gguf_path cannot be empty"):
            generate_modelfile(gguf_path="")

    def test_modelfile_generation_empty_system_prompt_fallback(self) -> None:
        """generate_modelfile with empty or whitespace-only system prompt falls back to default enterprise prompt."""
        content = generate_modelfile(gguf_path="model.gguf", system_prompt="   ")
        assert DEFAULT_ENTERPRISE_SYSTEM_PROMPT in content

    def test_modelfile_generation_extreme_context_and_temperature(self, tmp_path: Path) -> None:
        """generate_modelfile correctly handles boundary context windows and temperatures."""
        out_file = tmp_path / "Modelfile_Boundaries"
        content = generate_modelfile(
            gguf_path="model.gguf",
            output_path=out_file,
            num_ctx=131072,
            temperature=0.0,
            parameters={"top_k": 1, "top_p": 0.1, "repeat_penalty": 1.5},
        )
        assert "PARAMETER num_ctx 131072" in content
        assert "PARAMETER temperature 0.0" in content
        assert "PARAMETER top_k 1" in content
        assert "PARAMETER top_p 0.1" in content
        assert "PARAMETER repeat_penalty 1.5" in content

    def test_modelfile_generation_special_chars_in_system_prompt(self) -> None:
        """generate_modelfile preserves quotes, backslashes, markdown, and unicode in system prompts."""
        complex_prompt = (
            'Special system prompt with "double quotes", \'single quotes\',\n'
            'code blocks: ```python\nprint("hello")\n```,\n'
            'and Unicode: 🚀 🤖 \u2705 \u274c.'
        )
        content = generate_modelfile(gguf_path="model.gguf", system_prompt=complex_prompt)
        assert complex_prompt in content

    def test_normalize_quantization_all_canonical_keys(self) -> None:
        """normalize_quantization maps all supported aliases correctly and rejects invalid ones."""
        for alias, canonical in QUANT_CANONICAL_MAP.items():
            assert normalize_quantization(alias) == canonical
            assert normalize_quantization(alias.lower()) == canonical

        with pytest.raises(ValueError, match="Invalid quantization type"):
            normalize_quantization("")

        with pytest.raises(ValueError, match="Unsupported quantization"):
            normalize_quantization("NON_EXISTENT_QUANT")


# ═════════════════════════════════════════════════════════════════════════════
# 3. Fine-Tune CLI Stress Tests (`finetune_qwen_coder.py`)
# ═════════════════════════════════════════════════════════════════════════════

class TestFinetuneQwenCoderStress:
    """Stress testing finetune_qwen_coder CLI for config parsing, out-of-bounds hyperparams, and corrupt data."""

    def test_finetune_invalid_dataset_format_flag(self) -> None:
        """Invalid dataset format choice raises SystemExit code 2."""
        parser = build_finetune_parser()
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["--dataset_format", "invalid_choice"])
        assert excinfo.value.code == 2

    def test_finetune_invalid_quantization_flag(self) -> None:
        """Invalid quantization choice raises SystemExit code 2."""
        parser = build_finetune_parser()
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["--quantization", "fp16_invalid"])
        assert excinfo.value.code == 2

    def test_finetune_invalid_lora_bias_flag(self) -> None:
        """Invalid lora bias choice raises SystemExit code 2."""
        parser = build_finetune_parser()
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["--lora_bias", "invalid_bias"])
        assert excinfo.value.code == 2

    def test_finetune_main_corrupt_yaml_config_returns_error(self, tmp_path: Path) -> None:
        """Corrupt YAML config file passed to CLI returns exit code 1."""
        corrupt_yaml = tmp_path / "corrupt_config.yaml"
        corrupt_yaml.write_text("learning_rate: [unclosed list\n  - item1\n", encoding="utf-8")
        rc = finetune_main(["--config", str(corrupt_yaml)])
        assert rc == 1

    def test_finetune_main_corrupt_json_config_returns_error(self, tmp_path: Path) -> None:
        """Corrupt JSON config file passed to CLI returns exit code 1."""
        corrupt_json = tmp_path / "corrupt_config.json"
        corrupt_json.write_text("{\"learning_rate\": 0.0001, invalid_json_syntax}", encoding="utf-8")
        rc = finetune_main(["--config", str(corrupt_json)])
        assert rc == 1

    def test_finetune_main_missing_config_returns_error(self) -> None:
        """Non-existent config file path returns exit code 1."""
        rc = finetune_main(["--config", "non_existent_config_12345.yaml"])
        assert rc == 1

    def test_finetune_config_validation_learning_rate_zero_or_negative(self) -> None:
        """Learning rate <= 0 raises ValueError."""
        c1 = FineTuneConfig(dataset_path="data/test.jsonl", learning_rate=0.0)
        with pytest.raises(ValueError, match="`learning_rate` must be > 0"):
            c1.validate()

        c2 = FineTuneConfig(dataset_path="data/test.jsonl", learning_rate=-1e-4)
        with pytest.raises(ValueError, match="`learning_rate` must be > 0"):
            c2.validate()

    def test_finetune_config_validation_val_split_boundaries(self) -> None:
        """val_split < 0.0 or val_split >= 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="`val_split` must be between 0.0 and 1.0"):
            FineTuneConfig(dataset_path="data/test.jsonl", val_split=-0.1).validate()

        with pytest.raises(ValueError, match="`val_split` must be between 0.0 and 1.0"):
            FineTuneConfig(dataset_path="data/test.jsonl", val_split=1.0).validate()

        with pytest.raises(ValueError, match="`val_split` must be between 0.0 and 1.0"):
            FineTuneConfig(dataset_path="data/test.jsonl", val_split=1.5).validate()

    def test_finetune_config_validation_batch_size_zero_or_negative(self) -> None:
        """Batch size <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="`per_device_train_batch_size` must be > 0"):
            FineTuneConfig(dataset_path="data/test.jsonl", per_device_train_batch_size=0).validate()

        with pytest.raises(ValueError, match="`per_device_train_batch_size` must be > 0"):
            FineTuneConfig(dataset_path="data/test.jsonl", per_device_train_batch_size=-4).validate()

    def test_finetune_config_validation_lora_r_zero_or_negative(self) -> None:
        """LoRA rank <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="`lora_r` must be > 0"):
            FineTuneConfig(dataset_path="data/test.jsonl", lora_r=0).validate()

        with pytest.raises(ValueError, match="`lora_r` must be > 0"):
            FineTuneConfig(dataset_path="data/test.jsonl", lora_r=-8).validate()

    def test_finetune_config_validation_empty_target_modules(self) -> None:
        """Empty target modules list raises ValueError."""
        with pytest.raises(ValueError, match="`target_modules` must not be empty"):
            FineTuneConfig(dataset_path="data/test.jsonl", target_modules=[]).validate()

    def test_parse_target_modules_boundaries(self) -> None:
        """parse_target_modules handles lists, comma-separated strings, empty values, and non-string inputs."""
        assert parse_target_modules(["q_proj", "k_proj"]) == ["q_proj", "k_proj"]
        assert parse_target_modules("q_proj, v_proj , gate_proj ") == ["q_proj", "v_proj", "gate_proj"]
        # Empty string falls back to default 7 linear projections
        assert len(parse_target_modules("")) == 7
        assert len(parse_target_modules(None)) == 7

    def test_finetune_main_missing_dataset_path_without_smoke_test(self) -> None:
        """finetune_main without dataset_path and without smoke_test returns 1."""
        rc = finetune_main(["--dataset_path", ""])
        assert rc == 1

    def test_finetune_main_nonexistent_dataset_path(self, tmp_path: Path) -> None:
        """finetune_main with non-existent dataset path returns 1."""
        rc = finetune_main(["--dataset_path", str(tmp_path / "missing.jsonl")])
        assert rc == 1

    def test_finetune_main_empty_dataset_file(self, tmp_path: Path) -> None:
        """finetune_main with empty dataset file (0 bytes) returns 1."""
        empty_file = tmp_path / "empty_dataset.jsonl"
        empty_file.write_text("", encoding="utf-8")
        rc = finetune_main(["--dataset_path", str(empty_file)])
        assert rc == 1

    def test_finetune_main_dry_run_with_valid_fixture(self) -> None:
        """finetune_main with --dry-run and valid fixture dataset returns 0."""
        sample_dataset = FIXTURES_DIR / "alpaca_sample.json"
        rc = finetune_main([
            "--dry-run",
            "--dataset_path", str(sample_dataset),
            "--learning_rate", "1e-4",
            "--lora_r", "16",
        ])
        assert rc == 0


# ═════════════════════════════════════════════════════════════════════════════
# 4. Metrics & Perplexity Boundary Stress Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestPerplexityStress:
    """Stress testing calculate_perplexity for overflow, underflow, NaN, and extreme values."""

    def test_calculate_perplexity_nominal(self) -> None:
        """Nominal loss values compute exact exp(loss)."""
        assert math.isclose(calculate_perplexity(0.0), 1.0, rel_tol=1e-5)
        assert math.isclose(calculate_perplexity(1.0), math.e, rel_tol=1e-5)
        assert math.isclose(calculate_perplexity(2.5), math.exp(2.5), rel_tol=1e-5)

    def test_calculate_perplexity_none_and_nan(self) -> None:
        """None, NaN, or negative loss safely return float('nan')."""
        assert math.isnan(calculate_perplexity(None))
        assert math.isnan(calculate_perplexity(float("nan")))
        assert math.isnan(calculate_perplexity(-0.5))

    def test_calculate_perplexity_overflow_protection(self) -> None:
        """Large loss values (> 100 or huge float) return float('inf') without raising OverflowError."""
        assert calculate_perplexity(100.1) == float("inf")
        assert calculate_perplexity(500.0) == float("inf")
        assert calculate_perplexity(1e10) == float("inf")


# ═════════════════════════════════════════════════════════════════════════════
# 5. Dataset Loader Boundary Stress Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestDatasetLoaderStress:
    """Stress testing dataset normalization, sanitization, and splitting boundaries."""

    def test_detect_format_empty_records_raises(self) -> None:
        """detect_format on empty buffer raises DatasetEmptyError."""
        with pytest.raises(DatasetEmptyError):
            detect_format([])

    def test_detect_format_unrecognized_schema_raises(self) -> None:
        """detect_format on unrecognized schema dictionary raises DatasetFormatValidationError."""
        with pytest.raises(DatasetFormatValidationError):
            detect_format([{"unrecognized_col_a": 1, "unrecognized_col_b": 2}])

    def test_sanitize_messages_empty_or_malformed(self) -> None:
        """sanitize_messages returns empty list for empty/malformed inputs."""
        assert sanitize_messages([]) == []
        assert sanitize_messages([{"role": "user", "content": ""}]) == []  # empty content
        assert sanitize_messages([{"role": "user", "content": "hello"}]) == []  # missing assistant turn
        assert sanitize_messages([{"role": "assistant", "content": "hi"}]) == []  # missing user turn

    def test_sanitize_messages_consecutive_identical_roles_merged(self) -> None:
        """Consecutive user turns are merged into a single turn."""
        messages = [
            {"role": "user", "content": "Part 1 of query"},
            {"role": "user", "content": "Part 2 of query"},
            {"role": "assistant", "content": "Final Answer"},
        ]
        sanitized = sanitize_messages(messages)
        assert len(sanitized) == 2
        assert "Part 1 of query\n\nPart 2 of query" in sanitized[0]["content"]
        assert sanitized[1]["role"] == "assistant"

    def test_split_dataset_small_n_boundaries(self) -> None:
        """split_dataset handles N=0 (error), N=1 (train only, val=None), and small N splits."""
        from datasets import Dataset

        with pytest.raises(DatasetEmptyError):
            split_dataset(Dataset.from_list([]))

        ds1 = Dataset.from_list([{"messages": [{"role": "user", "content": "hi"}]}])
        tr1, val1 = split_dataset(ds1, val_split=0.2)
        assert len(tr1) == 1
        assert val1 is None

        ds3 = Dataset.from_list([{"item": i} for i in range(3)])
        tr3, val3 = split_dataset(ds3, val_split=0.33)
        assert len(tr3) >= 1
        assert len(val3) >= 1
        assert len(tr3) + len(val3) == 3
