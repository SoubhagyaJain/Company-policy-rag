"""Adversarial and Empirical Verification Test Suite for Milestone 2:
LoRA Adapter Merger, GGUF Exporter, Quantization Normalization, Device Fallback, and Header Validation.

Authoritative Reference:
- ORIGINAL_REQUEST.md (§ R2. Model Merging, GGUF Export & Ollama Registration)
- PROJECT.md (§ Architecture, Feature Inventory F2.1, F2.2, Interface Contracts)
- TEST_INFRA.md (§ Feature Inventory F2.1, F2.2 & Tier 1/2 coverage)
"""

from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest
import torch

from company_policy_rag.src.finetuning.merger import (
    ModelMerger,
    MergeConfig,
    MergeOutput,
    merge_lora_weights,
    resolve_device,
    resolve_torch_dtype,
    cleanup_memory,
    synthesize_fallback_merged_artifacts,
    DEFAULT_CHATML_TEMPLATE,
)
from company_policy_rag.src.finetuning.gguf_exporter import (
    GGUFExporter,
    GGUFValidationResult,
    GGUFExportError,
    GGUF_MAGIC,
    GGUF_VERSION,
    SUPPORTED_QUANTIZATIONS,
    QUANT_CANONICAL_MAP,
    convert_to_gguf,
    normalize_quantization,
    validate_gguf_file,
    _write_simulated_gguf,
)


# ==============================================================================
# 1. ADVERSARIAL STRESS TESTS: MODEL MERGER (`merger.py`)
# ==============================================================================

class TestLoRAMergerAdversarial:
    """Stress-test LoRA weight merger on boundary conditions, empty inputs, device fallback, and dtypes."""

    def test_empty_base_model_path_raises_value_error(self, tmp_path: Path) -> None:
        """Empty base model strings or whitespace-only paths must raise ValueError."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        out_dir = tmp_path / "out"

        with pytest.raises(ValueError, match="base_model_path cannot be empty"):
            merge_lora_weights(base_model_path="", adapter_path=adapter_dir, output_dir=out_dir)

        with pytest.raises(ValueError, match="base_model_path cannot be empty"):
            merge_lora_weights(base_model_path="   ", adapter_path=adapter_dir, output_dir=out_dir)

    def test_empty_adapter_path_raises_value_error(self, tmp_path: Path) -> None:
        """Empty adapter strings or whitespace-only paths must raise ValueError."""
        out_dir = tmp_path / "out"

        with pytest.raises(ValueError, match="adapter_path cannot be empty"):
            merge_lora_weights(base_model_path="Qwen/Qwen2.5-Coder-7B-Instruct", adapter_path="", output_dir=out_dir)

        with pytest.raises(ValueError, match="adapter_path cannot be empty"):
            merge_lora_weights(base_model_path="Qwen/Qwen2.5-Coder-7B-Instruct", adapter_path="   ", output_dir=out_dir)

    def test_empty_output_dir_raises_value_error(self, tmp_path: Path) -> None:
        """Empty output_dir string or whitespace-only path must raise ValueError."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        with pytest.raises(ValueError, match="output_dir cannot be empty"):
            merge_lora_weights(base_model_path="Qwen/Qwen2.5-Coder-7B-Instruct", adapter_path=adapter_dir, output_dir="")

        with pytest.raises(ValueError, match="output_dir cannot be empty"):
            merge_lora_weights(base_model_path="Qwen/Qwen2.5-Coder-7B-Instruct", adapter_path=adapter_dir, output_dir="   ")

    def test_merge_config_dataclass_validation(self) -> None:
        """MergeConfig.validate() enforces non-empty fields."""
        cfg_valid = MergeConfig(base_model_path="Qwen/Qwen2.5", adapter_path="./ad", output_dir="./out")
        cfg_valid.validate()  # should not raise

        with pytest.raises(ValueError, match="base_model_path"):
            MergeConfig(base_model_path="", adapter_path="./ad", output_dir="./out").validate()

        with pytest.raises(ValueError, match="adapter_path"):
            MergeConfig(base_model_path="Qwen/Qwen2.5", adapter_path="", output_dir="./out").validate()

        with pytest.raises(ValueError, match="output_dir"):
            MergeConfig(base_model_path="Qwen/Qwen2.5", adapter_path="./ad", output_dir="").validate()

    @pytest.mark.parametrize("invalid_device", [
        "cuda:999",
        "cuda:9999",
        "invalid_device",
        "tpu",
        "mps",
        "foo_bar",
        "cuda:invalid",
        "cuda:-1",
        "UNKNOWN",
    ])
    def test_out_of_range_and_invalid_device_fallback_to_cpu(self, invalid_device: str) -> None:
        """Out-of-range, non-existent, or malformed device strings must gracefully fall back to CPU."""
        dev, device_map = resolve_device(invalid_device)
        assert dev == "cpu"
        assert device_map is None

    def test_valid_device_resolutions(self) -> None:
        """Standard device strings resolve safely."""
        dev_cpu, map_cpu = resolve_device("cpu")
        assert dev_cpu == "cpu"
        assert map_cpu is None

        dev_upper, map_upper = resolve_device("CPU")
        assert dev_upper == "cpu"
        assert map_upper is None

        dev_auto, map_auto = resolve_device("auto")
        if torch.cuda.is_available():
            assert dev_auto == "cuda"
            assert map_auto == "auto"
        else:
            assert dev_auto == "cpu"
            assert map_auto is None

    @pytest.mark.parametrize("dtype_input,expected_dtype", [
        ("float16", torch.float16),
        ("fp16", torch.float16),
        ("f16", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("bf16", torch.bfloat16),
        ("bf", torch.bfloat16),
        ("float32", torch.float32),
        ("fp32", torch.float32),
        ("f32", torch.float32),
        ("auto", torch.float16),
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
    ])
    def test_resolve_torch_dtype_supported(self, dtype_input: Any, expected_dtype: torch.dtype) -> None:
        """resolve_torch_dtype correctly maps string representations and torch.dtype objects."""
        assert resolve_torch_dtype(dtype_input) == expected_dtype

    def test_resolve_torch_dtype_unsupported_raises_value_error(self) -> None:
        """Unsupported precision strings raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            resolve_torch_dtype("int8")

        with pytest.raises(ValueError, match="Unsupported dtype"):
            resolve_torch_dtype("float64")

        with pytest.raises(ValueError, match="Unsupported dtype"):
            resolve_torch_dtype("INVALID_DTYPE")

    def test_merge_lora_weights_execution_contract(self, tmp_path: Path) -> None:
        """merge_lora_weights creates complete standalone model directory with configs and weights."""
        adapter_path = tmp_path / "mock_adapter"
        adapter_path.mkdir()
        (adapter_path / "adapter_config.json").write_text('{"base_model_name_or_path": "Qwen/Qwen2.5-Coder-7B-Instruct"}', encoding="utf-8")
        out_dir = tmp_path / "merged_output"

        result_path = merge_lora_weights(
            base_model_path="Qwen/Qwen2.5-Coder-7B-Instruct",
            adapter_path=adapter_path,
            output_dir=out_dir,
            device="cpu",
            dtype="float16",
        )

        assert Path(result_path).exists()
        assert Path(result_path) == out_dir

        # Verify all mandatory HuggingFace and ChatML artifacts
        assert (out_dir / "config.json").is_file()
        assert (out_dir / "tokenizer_config.json").is_file()
        assert (out_dir / "special_tokens_map.json").is_file()
        assert (out_dir / "tokenizer.json").is_file()
        assert (out_dir / "model.safetensors").is_file()
        assert (out_dir / "model.safetensors").stat().st_size > 0

        # Verify ChatML tokens in tokenizer config
        tok_cfg = (out_dir / "tokenizer_config.json").read_text(encoding="utf-8")
        assert "<|im_start|>" in tok_cfg
        assert "<|im_end|>" in tok_cfg
        assert "<|endoftext|>" in tok_cfg

    def test_model_merger_oop_class(self, tmp_path: Path) -> None:
        """ModelMerger OOP wrapper functions identically to functional merge_lora_weights."""
        merger = ModelMerger(device="cpu", dtype="bfloat16")
        adapter_dir = tmp_path / "adapter_oop"
        adapter_dir.mkdir()
        out_dir = tmp_path / "out_oop"

        res = merger.merge(
            base_model_path="Qwen/Qwen2.5-Coder-7B-Instruct",
            adapter_path=adapter_dir,
            output_dir=out_dir,
        )
        assert Path(res).exists()
        assert (Path(res) / "config.json").exists()

        # Test alias merge_and_unload
        out_dir_2 = tmp_path / "out_oop_2"
        res_2 = merger.merge_and_unload(
            base_model_path="Qwen/Qwen2.5-Coder-7B-Instruct",
            adapter_path=adapter_dir,
            output_dir=out_dir_2,
        )
        assert Path(res_2).exists()


# ==============================================================================
# 2. ADVERSARIAL STRESS TESTS: GGUF EXPORTER (`gguf_exporter.py`)
# ==============================================================================

class TestGGUFExporterAdversarial:
    """Stress-test GGUF exporter on quantization normalization, extension enforcement, and header validation."""

    @pytest.mark.parametrize("invalid_quant", [
        "INVALID_QUANT",
        "Q99",
        "Q4_K_M_XYZ",
        "INT8",
        "FP32_EXTRA",
        "Q3_K_S_UNSUPPORTED",
        "",
        "   ",
    ])
    def test_unsupported_quantization_raises_value_error(self, invalid_quant: str) -> None:
        """Unsupported or empty quantization format strings must raise ValueError."""
        with pytest.raises(ValueError):
            normalize_quantization(invalid_quant)

    @pytest.mark.parametrize("non_string_quant", [None, 123, 4.5, [], {}])
    def test_non_string_quantization_raises_value_error(self, non_string_quant: Any) -> None:
        """Non-string quantization arguments must raise ValueError."""
        with pytest.raises(ValueError):
            normalize_quantization(non_string_quant)

    @pytest.mark.parametrize("raw_input,canonical_output", [
        ("Q4_K_M", "Q4_K_M"),
        ("q4_k_m", "Q4_K_M"),
        ("Q4_KM", "Q4_K_M"),
        ("q4_km", "Q4_K_M"),
        ("Q8_0", "Q8_0"),
        ("q8_0", "Q8_0"),
        ("Q80", "Q8_0"),
        ("q80", "Q8_0"),
        ("FP16", "FP16"),
        ("fp16", "FP16"),
        ("F16", "FP16"),
        ("f16", "FP16"),
        ("Q4_0", "Q4_0"),
        ("q4_0", "Q4_0"),
        ("Q40", "Q4_0"),
        ("q40", "Q4_0"),
        ("Q5_K_M", "Q5_K_M"),
        ("q5_k_m", "Q5_K_M"),
        ("Q5_KM", "Q5_K_M"),
        ("q5_km", "Q5_K_M"),
    ])
    def test_quantization_normalization_canonical_map(self, raw_input: str, canonical_output: str) -> None:
        """normalize_quantization correctly normalizes all case/underscore variants."""
        assert normalize_quantization(raw_input) == canonical_output

    @pytest.mark.parametrize("invalid_filename", [
        "model.bin",
        "model.safetensors",
        "model.onnx",
        "model.pt",
        "model.pth",
        "model.txt",
        "model",
        "model.gguf.tmp",
        "gguf",
    ])
    def test_non_gguf_extension_raises_value_error(self, tmp_path: Path, invalid_filename: str) -> None:
        """convert_to_gguf must reject any output path not ending in '.gguf'."""
        model_dir = tmp_path / "model_dir"
        model_dir.mkdir()
        out_file = tmp_path / invalid_filename

        with pytest.raises(ValueError, match=r"\.gguf"):
            convert_to_gguf(
                model_dir=model_dir,
                output_file=out_file,
                quantization="Q4_K_M",
            )

    def test_validate_gguf_nonexistent_file(self, tmp_path: Path) -> None:
        """Validating a non-existent file path returns is_valid=False with descriptive error."""
        missing = tmp_path / "does_not_exist.gguf"
        report = validate_gguf_file(missing)

        assert report["is_valid"] is False
        assert report["file_size_bytes"] == 0
        assert "does not exist" in report["error_message"].lower()

    def test_validate_gguf_zero_byte_file(self, tmp_path: Path) -> None:
        """Validating a 0-byte file returns is_valid=False and size warning."""
        zero_file = tmp_path / "zero.gguf"
        zero_file.write_bytes(b"")

        report = validate_gguf_file(zero_file)
        assert report["is_valid"] is False
        assert report["file_size_bytes"] == 0
        assert "too small" in report["error_message"].lower()

    @pytest.mark.parametrize("byte_len", [1, 5, 12, 23])
    def test_validate_gguf_truncated_header(self, tmp_path: Path, byte_len: int) -> None:
        """Validating a file smaller than standard 24-byte GGUF header returns is_valid=False."""
        trunc_file = tmp_path / f"trunc_{byte_len}.gguf"
        trunc_file.write_bytes(b"GGUF" + b"\x00" * (byte_len - 4) if byte_len >= 4 else b"A" * byte_len)

        report = validate_gguf_file(trunc_file)
        assert report["is_valid"] is False
        assert report["file_size_bytes"] == byte_len
        assert "too small" in report["error_message"].lower()

    def test_validate_gguf_corrupted_magic_bytes(self, tmp_path: Path) -> None:
        """Validating a file with corrupt magic bytes (not 'GGUF') returns is_valid=False."""
        corrupt_file = tmp_path / "corrupt_magic.gguf"
        # 24 bytes total: 4 magic + 4 version + 8 tensor_count + 8 kv_count
        corrupt_bytes = b"GGML" + struct.pack("<IQQ", 3, 1, 2)
        corrupt_file.write_bytes(corrupt_bytes)

        report = validate_gguf_file(corrupt_file)
        assert report["is_valid"] is False
        assert report["magic"] == "GGML"
        assert "Invalid magic bytes" in report["error_message"]

    @pytest.mark.parametrize("quant", ["Q4_K_M", "Q8_0", "FP16", "Q4_0", "Q5_K_M"])
    def test_validate_gguf_valid_synthetic_files(self, tmp_path: Path, quant: str) -> None:
        """Synthetic GGUF binaries produced by convert_to_gguf pass full GGUF v3 header validation."""
        model_dir = tmp_path / f"model_{quant.lower()}"
        model_dir.mkdir()
        out_file = tmp_path / f"model_{quant.lower()}.gguf"

        res_path = convert_to_gguf(
            model_dir=model_dir,
            output_file=out_file,
            quantization=quant,
            allow_simulation=True,
        )

        assert Path(res_path).exists()
        assert Path(res_path) == out_file

        report = validate_gguf_file(out_file)
        assert report["is_valid"] is True
        assert report["magic"] == "GGUF"
        assert report["version"] == 3
        assert report["tensor_count"] >= 1
        assert report["kv_count"] >= 1
        assert report["error_message"] is None
        assert report["file_size_bytes"] > 24

    def test_gguf_exporter_class_oop_contract(self, tmp_path: Path) -> None:
        """GGUFExporter class wrapper correctly exports and validates GGUF files."""
        exporter = GGUFExporter(quantization="Q8_0", allow_simulation=True)
        model_dir = tmp_path / "hf_model"
        model_dir.mkdir()
        out_gguf = tmp_path / "exported_q8.gguf"

        res = exporter.export(model_dir=model_dir, output_file=out_gguf)
        assert Path(res).exists()

        val_report = exporter.validate(res)
        assert val_report["is_valid"] is True
        assert val_report["magic"] == "GGUF"
        assert val_report["version"] == 3
