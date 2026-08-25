"""
Unit tests for LoRA Adapter Merging, GGUF Conversion/Quantization, Ollama Modelfile Generation, and Export CLI.

Authoritative Reference:
- ORIGINAL_REQUEST.md (§ R2. Model Merging, GGUF Export & Ollama Registration)
- PROJECT.md (§ Architecture, Feature Inventory F2.1, F2.2, F2.3, Interface Contracts)
- TEST_INFRA.md (§ Feature Inventory F2.1, F2.2, F2.3 & Tier 1/2 coverage)
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
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
    from company_policy_rag.src.finetuning.merger import ModelMerger, merge_lora_weights
    from company_policy_rag.src.finetuning.gguf_exporter import GGUFExporter, convert_to_gguf, SUPPORTED_QUANTIZATIONS
    from company_policy_rag.src.finetuning.modelfile_generator import (
        ModelfileGenerator,
        generate_modelfile,
        parse_modelfile,
        normalize_gguf_path,
        DEFAULT_ENTERPRISE_SYSTEM_PROMPT,
        CHATML_TEMPLATE,
    )
    from company_policy_rag.scripts.export_and_register_ollama import (
        build_arg_parser as build_export_arg_parser,
        main as export_main,
    )
except ImportError:
    from src.finetuning.merger import ModelMerger, merge_lora_weights
    from src.finetuning.gguf_exporter import GGUFExporter, convert_to_gguf, SUPPORTED_QUANTIZATIONS
    from src.finetuning.modelfile_generator import (
        ModelfileGenerator,
        generate_modelfile,
        parse_modelfile,
        normalize_gguf_path,
        DEFAULT_ENTERPRISE_SYSTEM_PROMPT,
        CHATML_TEMPLATE,
    )
    from scripts.export_and_register_ollama import (
        build_arg_parser as build_export_arg_parser,
        main as export_main,
    )


# ── Test Suites ─────────────────────────────────────────────────────────────

class TestLoRAAdapterMerger:
    """Tests for standalone model merger contract, weight export, and tokenizer files."""

    def test_merge_lora_weights_contract(self, tmp_path: Path) -> None:
        """Merger produces standalone model directory with config, tokenizer, and safetensors weights."""
        base_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
        adapter_path = tmp_path / "adapter_input"
        adapter_path.mkdir()
        (adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")

        out_dir = tmp_path / "merged_model"
        res_dir = merge_lora_weights(base_model, adapter_path, out_dir, device="cpu")

        assert Path(res_dir).exists()
        assert (Path(res_dir) / "config.json").exists()
        assert (Path(res_dir) / "tokenizer.json").exists()
        assert (Path(res_dir) / "tokenizer_config.json").exists()
        assert (Path(res_dir) / "model.safetensors").exists()

    def test_model_merger_oop_class(self, tmp_path: Path) -> None:
        """ModelMerger OOP wrapper executes merge and returns output path."""
        base_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
        adapter_path = tmp_path / "adapter_oop"
        adapter_path.mkdir()
        (adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")

        out_dir = tmp_path / "merged_oop"
        merger = ModelMerger(device="cpu")
        res_dir = merger.merge(base_model, str(adapter_path), str(out_dir))

        assert Path(res_dir).exists()
        assert (Path(res_dir) / "config.json").exists()
        assert (Path(res_dir) / "model.safetensors").exists()

    def test_merge_lora_empty_base_model_error(self, tmp_path: Path) -> None:
        """Empty base_model_path string raises ValueError."""
        with pytest.raises(ValueError, match="base_model_path"):
            merge_lora_weights("", tmp_path / "adapter", tmp_path / "out")

    def test_merge_lora_empty_adapter_path_error(self, tmp_path: Path) -> None:
        """Empty adapter_path string raises ValueError."""
        with pytest.raises(ValueError, match="adapter_path"):
            merge_lora_weights("Qwen/Qwen2.5-Coder-7B-Instruct", "", tmp_path / "out")


class TestGGUFQuantizationExporter:
    """Tests for GGUF conversion, quantization flags (Q4_K_M, Q8_0, FP16), and validation."""

    def test_convert_to_gguf_supported_quantizations(self, tmp_path: Path) -> None:
        """GGUF exporter supports standard quantizations: Q4_K_M, Q8_0, FP16."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        for quant in ["Q4_K_M", "Q8_0", "FP16"]:
            out_file = tmp_path / f"model_{quant.lower()}.gguf"
            res = convert_to_gguf(model_dir, out_file, quantization=quant)
            assert Path(res).exists()
            assert Path(res).stat().st_size > 0

    def test_gguf_exporter_oop_class(self, tmp_path: Path) -> None:
        """GGUFExporter OOP wrapper converts and returns output file path."""
        model_dir = tmp_path / "model_oop"
        model_dir.mkdir()
        out_file = tmp_path / "model_q8.gguf"

        exporter = GGUFExporter(quantization="Q8_0")
        res = exporter.export(str(model_dir), str(out_file))

        assert Path(res).exists()
        assert Path(res).name == "model_q8.gguf"

    def test_convert_to_gguf_unsupported_quant_error(self, tmp_path: Path) -> None:
        """Unsupported quantization type raises ValueError with clear error message."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        out_file = tmp_path / "model_invalid.gguf"

        with pytest.raises(ValueError, match="Unsupported quantization"):
            convert_to_gguf(model_dir, out_file, quantization="INVALID_QUANT_METHOD_XYZ")

    def test_convert_to_gguf_invalid_extension_error(self, tmp_path: Path) -> None:
        """Output file without .gguf extension raises ValueError."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        out_file = tmp_path / "model.bin"

        with pytest.raises(ValueError, match=r"\.gguf"):
            convert_to_gguf(model_dir, out_file, quantization="Q4_K_M")


class TestOllamaModelfileGenerator:
    """Tests for Ollama Modelfile syntax, ChatML template, stop tokens, and parameters."""

    def test_modelfile_generation_syntax(self, tmp_path: Path) -> None:
        """Modelfile contains FROM, ChatML TEMPLATE, stop tokens, num_ctx, and temperature."""
        gguf_path = "/models/qwen2.5-coder-7b-policy-q4_k_m.gguf"
        out_path = tmp_path / "Modelfile"

        content = generate_modelfile(
            gguf_path=gguf_path,
            output_path=out_path,
            num_ctx=8192,
            temperature=0.1,
        )

        assert out_path.exists()
        assert f"FROM {gguf_path}" in content
        assert "TEMPLATE" in content
        assert "<|im_start|>system" in content
        assert "<|im_end|>" in content
        assert 'PARAMETER stop "<|im_end|>"' in content
        assert 'PARAMETER stop "<|endoftext|>"' in content
        assert "PARAMETER num_ctx 8192" in content
        assert "PARAMETER temperature 0.1" in content
        assert "SYSTEM" in content
        assert "Company Policy" in content

    def test_modelfile_custom_system_prompt(self) -> None:
        """Custom system prompt is properly injected into SYSTEM block."""
        custom_prompt = "You are a specialized legal compliance advisor."
        content = generate_modelfile(
            gguf_path="model.gguf",
            system_prompt=custom_prompt,
        )
        assert custom_prompt in content

    def test_normalize_gguf_path(self) -> None:
        """normalize_gguf_path converts Windows backslashes to forward slashes and validates input."""
        win_path = r"C:\Users\models\qwen.gguf"
        normalized = normalize_gguf_path(win_path)
        assert "\\" not in normalized
        assert "/" in normalized

        with pytest.raises(ValueError, match="gguf_path cannot be empty"):
            normalize_gguf_path("")

    def test_parse_modelfile_functional(self, tmp_path: Path) -> None:
        """parse_modelfile extracts FROM, TEMPLATE, stop tokens, parameters, and SYSTEM."""
        mf_path = tmp_path / "Modelfile"
        generate_modelfile(
            gguf_path="/path/to/model.gguf",
            output_path=mf_path,
            num_ctx=4096,
            temperature=0.2,
            parameters={"top_k": 50, "repeat_penalty": 1.2},
        )

        parsed = parse_modelfile(mf_path)
        assert parsed["from"] == "/path/to/model.gguf"
        assert parsed["template"] is not None
        assert "<|im_start|>" in parsed["template"]
        assert "<|im_end|>" in parsed["stop_tokens"]
        assert "<|endoftext|>" in parsed["stop_tokens"]
        assert parsed["parameters"]["num_ctx"] == "4096"
        assert parsed["parameters"]["temperature"] == "0.2"
        assert parsed["parameters"]["top_k"] == "50"
        assert parsed["parameters"]["repeat_penalty"] == "1.2"
        assert parsed["system"] is not None
        assert "Company Policy" in parsed["system"]

    def test_parse_modelfile_non_existent_error(self, tmp_path: Path) -> None:
        """parse_modelfile raises FileNotFoundError when file does not exist."""
        non_existent = tmp_path / "NonExistentModelfile"
        with pytest.raises(FileNotFoundError):
            parse_modelfile(non_existent)

    def test_modelfile_generator_oop_class(self, tmp_path: Path) -> None:
        """ModelfileGenerator OOP class generates and parses modelfiles with custom parameters."""
        generator = ModelfileGenerator(
            system_prompt="Custom Enterprise QA Bot",
            num_ctx=16384,
            temperature=0.05,
            parameters={"top_p": 0.9, "mirostat": 2},
        )
        out_path = tmp_path / "Modelfile_OOP"
        content = generator.generate(
            gguf_path="/models/custom.gguf",
            output_path=out_path,
        )

        assert out_path.exists()
        assert "Custom Enterprise QA Bot" in content
        assert "PARAMETER num_ctx 16384" in content
        assert "PARAMETER temperature 0.05" in content
        assert "PARAMETER mirostat 2" in content

        parsed = generator.parse(out_path)
        assert parsed["from"] == "/models/custom.gguf"
        assert parsed["system"] == "Custom Enterprise QA Bot"
        assert parsed["parameters"]["num_ctx"] == "16384"
        assert parsed["parameters"]["temperature"] == "0.05"
        assert parsed["parameters"]["mirostat"] == "2"

    def test_fixture_dummy_modelfile_syntax(self) -> None:
        """Validates dummy_modelfile.txt fixture against Ollama Modelfile syntax specification."""
        fixture_file = FIXTURES_DIR / "dummy_modelfile.txt"
        assert fixture_file.exists()

        text = fixture_file.read_text(encoding="utf-8")
        assert "FROM" in text
        assert "TEMPLATE" in text
        assert 'PARAMETER stop "<|im_end|>"' in text
        assert 'PARAMETER stop "<|endoftext|>"' in text
        assert "PARAMETER num_ctx 8192" in text
        assert "PARAMETER temperature 0.1" in text
        assert "SYSTEM" in text


class TestExportAndRegisterCLI:
    """Unit tests for company_policy_rag/scripts/export_and_register_ollama.py CLI."""

    def test_build_arg_parser_defaults(self) -> None:
        """build_arg_parser configures default arguments correctly."""
        parser = build_export_arg_parser()
        args = parser.parse_args([])

        assert args.base_model_path == "Qwen/Qwen2.5-Coder-7B-Instruct"
        assert args.adapter_path == ""
        assert args.device == "cpu"
        assert args.dtype == "float16"
        assert args.output_dir == "./outputs/export"
        assert args.quantization == "Q4_K_M"
        assert args.model_name == "qwen2.5-coder-7b-policy"
        assert args.num_ctx == 8192
        assert args.temperature == 0.1
        assert args.ollama_url == "http://localhost:11434"
        assert args.dry_run is False
        assert args.skip_merge is False
        assert args.skip_quant is False
        assert args.skip_register is False
        assert args.verbose is False

    def test_build_arg_parser_custom_args(self) -> None:
        """build_arg_parser parses custom arguments correctly."""
        parser = build_export_arg_parser()
        args = parser.parse_args([
            "--adapter_path", "./outputs/adapter",
            "--quantization", "Q8_0",
            "--model_name", "qwen-custom-policy",
            "--num_ctx", "4096",
            "--temperature", "0.2",
            "--dry-run",
            "--skip-merge",
            "--skip-quant",
            "--skip-register",
            "--verbose",
        ])

        assert args.adapter_path == "./outputs/adapter"
        assert args.quantization == "Q8_0"
        assert args.model_name == "qwen-custom-policy"
        assert args.num_ctx == 4096
        assert args.temperature == 0.2
        assert args.dry_run is True
        assert args.skip_merge is True
        assert args.skip_quant is True
        assert args.skip_register is True
        assert args.verbose is True

    def test_export_main_dry_run_success(self) -> None:
        """export_main with --dry-run executes successfully and returns 0."""
        rc = export_main(["--dry-run"])
        assert rc == 0

    def test_export_main_dry_run_with_custom_flags(self) -> None:
        """export_main with --dry-run and custom arguments returns 0."""
        rc = export_main([
            "--dry-run",
            "--adapter_path", "./dummy_adapter",
            "--quantization", "FP16",
            "--model_name", "test-policy",
            "--num_ctx", "2048",
            "--temperature", "0.5",
        ])
        assert rc == 0

    def test_export_main_missing_adapter_returns_error(self) -> None:
        """export_main without adapter_path, without --dry-run, and without --skip-merge returns error code 1."""
        rc = export_main(["--adapter_path", ""])
        assert rc == 1
