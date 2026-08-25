"""Standalone Adversarial and Empirical Verification Runner for Milestone 2:
LoRA Adapter Merger, GGUF Exporter, Quantization Normalization, Device Fallback, and Header Validation.

Can be run via `python tests/integration/run_adversarial_m2_standalone.py` or via pytest.
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure company_policy_rag root and workspace root in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
TESTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TESTS_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

for p in [str(PROJECT_ROOT), str(WORKSPACE_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from company_policy_rag.src.finetuning.merger import (
        ModelMerger,
        MergeConfig,
        merge_lora_weights,
        resolve_device,
        resolve_torch_dtype,
    )
    from company_policy_rag.src.finetuning.gguf_exporter import (
        GGUFExporter,
        convert_to_gguf,
        normalize_quantization,
        validate_gguf_file,
        SUPPORTED_QUANTIZATIONS,
    )
    from company_policy_rag.src.finetuning.modelfile_generator import (
        ModelfileGenerator,
        generate_modelfile,
        parse_modelfile,
    )
except ImportError:
    from src.finetuning.merger import (
        ModelMerger,
        MergeConfig,
        merge_lora_weights,
        resolve_device,
        resolve_torch_dtype,
    )
    from src.finetuning.gguf_exporter import (
        GGUFExporter,
        convert_to_gguf,
        normalize_quantization,
        validate_gguf_file,
        SUPPORTED_QUANTIZATIONS,
    )
    from src.finetuning.modelfile_generator import (
        ModelfileGenerator,
        generate_modelfile,
        parse_modelfile,
    )

import torch


def run_all_m2_adversarial_tests() -> bool:
    print("=" * 75)
    print("  Milestone 2 Empirical Challenger: Adversarial Stress Test Suite")
    print("=" * 75)
    t_start = time.perf_counter()
    passed_count = 0
    failed_count = 0

    def assert_test(condition: bool, msg: str) -> None:
        nonlocal passed_count, failed_count
        if condition:
            passed_count += 1
            print(f"  [PASS] {msg}")
        else:
            failed_count += 1
            print(f"  [FAIL] {msg}")
            raise AssertionError(f"Test failed: {msg}")

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_path = Path(tmp_dir_str)

        # ── Test Suite 1: merger.py Adversarial Tests ─────────────────────────
        print("\n--- 1. Testing LoRA Model Merger Adversarial Scenarios ---")

        # 1.1 Empty base model path
        try:
            merge_lora_weights("", tmp_path / "adapter", tmp_path / "out")
            assert_test(False, "Empty base_model_path should have raised ValueError")
        except ValueError as e:
            assert_test("base_model_path cannot be empty" in str(e), "Empty base_model_path raises ValueError")

        # 1.2 Whitespace base model path
        try:
            merge_lora_weights("   ", tmp_path / "adapter", tmp_path / "out")
            assert_test(False, "Whitespace base_model_path should have raised ValueError")
        except ValueError as e:
            assert_test("base_model_path cannot be empty" in str(e), "Whitespace base_model_path raises ValueError")

        # 1.3 Empty adapter path
        try:
            merge_lora_weights("Qwen/Qwen2.5-Coder-7B-Instruct", "", tmp_path / "out")
            assert_test(False, "Empty adapter_path should have raised ValueError")
        except ValueError as e:
            assert_test("adapter_path cannot be empty" in str(e), "Empty adapter_path raises ValueError")

        # 1.4 Empty output dir
        try:
            merge_lora_weights("Qwen/Qwen2.5-Coder-7B-Instruct", tmp_path / "adapter", "")
            assert_test(False, "Empty output_dir should have raised ValueError")
        except ValueError as e:
            assert_test("output_dir cannot be empty" in str(e), "Empty output_dir raises ValueError")

        # 1.5 Out-of-range device strings graceful fallback to CPU
        invalid_devices = ["cuda:999", "cuda:9999", "invalid_device", "tpu", "mps", "foo_bar", "cuda:invalid", "cuda:-1"]
        for dev_str in invalid_devices:
            dev, dmap = resolve_device(dev_str)
            assert_test(dev == "cpu" and dmap is None, f"Invalid device '{dev_str}' falls back to CPU")

        # 1.6 Dtype resolution
        assert_test(resolve_torch_dtype("float16") == torch.float16, "resolve_torch_dtype('float16') -> torch.float16")
        assert_test(resolve_torch_dtype("bfloat16") == torch.bfloat16, "resolve_torch_dtype('bfloat16') -> torch.bfloat16")
        assert_test(resolve_torch_dtype("float32") == torch.float32, "resolve_torch_dtype('float32') -> torch.float32")
        assert_test(resolve_torch_dtype("auto") == torch.float16, "resolve_torch_dtype('auto') -> torch.float16")

        try:
            resolve_torch_dtype("INVALID_DTYPE")
            assert_test(False, "Invalid dtype should raise ValueError")
        except ValueError:
            assert_test(True, "Invalid dtype raises ValueError")

        # 1.7 Execution contract & artifact synthesis
        out_merged = tmp_path / "merged_standalone"
        ad_dir = tmp_path / "dummy_adapter"
        ad_dir.mkdir()
        (ad_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

        res_merge = merge_lora_weights(
            base_model_path="Qwen/Qwen2.5-Coder-7B-Instruct",
            adapter_path=ad_dir,
            output_dir=out_merged,
            device="cpu",
            dtype="float16",
        )
        assert_test(Path(res_merge).is_dir(), "merge_lora_weights produces valid output directory")
        assert_test((out_merged / "config.json").is_file(), "Merged output has config.json")
        assert_test((out_merged / "tokenizer_config.json").is_file(), "Merged output has tokenizer_config.json")
        assert_test((out_merged / "special_tokens_map.json").is_file(), "Merged output has special_tokens_map.json")
        assert_test((out_merged / "tokenizer.json").is_file(), "Merged output has tokenizer.json")
        assert_test((out_merged / "model.safetensors").is_file(), "Merged output has model.safetensors")

        # ── Test Suite 2: gguf_exporter.py Adversarial Tests ──────────────────
        print("\n--- 2. Testing GGUF Exporter & Quantization Adversarial Scenarios ---")

        # 2.1 Unsupported quantization formats raise ValueError
        unsupported_quants = ["INVALID_QUANT", "Q99", "Q4_K_M_XYZ", "INT8", "FP32_EXTRA", "", "   "]
        for bad_q in unsupported_quants:
            try:
                normalize_quantization(bad_q)
                assert_test(False, f"Unsupported quant '{bad_q}' should raise ValueError")
            except ValueError:
                assert_test(True, f"Unsupported quant '{bad_q}' raises ValueError")

        # 2.2 Canonical normalization
        assert_test(normalize_quantization("q4_k_m") == "Q4_K_M", "q4_k_m normalizes to Q4_K_M")
        assert_test(normalize_quantization("Q4_KM") == "Q4_K_M", "Q4_KM normalizes to Q4_K_M")
        assert_test(normalize_quantization("q8_0") == "Q8_0", "q8_0 normalizes to Q8_0")
        assert_test(normalize_quantization("fp16") == "FP16", "fp16 normalizes to FP16")
        assert_test(normalize_quantization("f16") == "FP16", "f16 normalizes to FP16")
        assert_test(normalize_quantization("q4_0") == "Q4_0", "q4_0 normalizes to Q4_0")
        assert_test(normalize_quantization("q5_k_m") == "Q5_K_M", "q5_k_m normalizes to Q5_K_M")

        # 2.3 Non-.gguf extension rejection
        invalid_extensions = ["model.bin", "model.safetensors", "model.onnx", "model.pt", "model", "model.gguf.tmp"]
        for bad_ext in invalid_extensions:
            try:
                convert_to_gguf(out_merged, tmp_path / bad_ext, quantization="Q4_K_M")
                assert_test(False, f"Non-.gguf file '{bad_ext}' should raise ValueError")
            except ValueError as e:
                assert_test(".gguf" in str(e), f"Non-.gguf file '{bad_ext}' raises ValueError with .gguf mention")

        # 2.4 Header validation: non-existent file
        missing_report = validate_gguf_file(tmp_path / "non_existent.gguf")
        assert_test(missing_report["is_valid"] is False, "validate_gguf_file on missing file returns is_valid=False")
        assert_test(missing_report["file_size_bytes"] == 0, "Missing file size is 0 bytes")

        # 2.5 Header validation: zero-byte file
        zero_file = tmp_path / "zero_bytes.gguf"
        zero_file.write_bytes(b"")
        zero_report = validate_gguf_file(zero_file)
        assert_test(zero_report["is_valid"] is False, "validate_gguf_file on 0-byte file returns is_valid=False")
        assert_test("too small" in zero_report["error_message"].lower(), "0-byte file error mentions 'too small'")

        # 2.6 Header validation: truncated file (<24 bytes)
        for tr_len in [1, 5, 12, 23]:
            tr_file = tmp_path / f"truncated_{tr_len}.gguf"
            tr_file.write_bytes(b"GGUF" + b"\x00" * (tr_len - 4) if tr_len >= 4 else b"A" * tr_len)
            tr_report = validate_gguf_file(tr_file)
            assert_test(tr_report["is_valid"] is False, f"Truncated file ({tr_len} bytes) returns is_valid=False")

        # 2.7 Header validation: corrupted magic bytes
        corrupt_magic_file = tmp_path / "bad_magic.gguf"
        corrupt_magic_file.write_bytes(b"GGML" + struct.pack("<IQQ", 3, 1, 2))
        corrupt_report = validate_gguf_file(corrupt_magic_file)
        assert_test(corrupt_report["is_valid"] is False, "Corrupted magic b'GGML' returns is_valid=False")
        assert_test(corrupt_report["magic"] == "GGML", "Corrupted magic properly identified as 'GGML'")

        # 2.8 Header validation: valid synthetic GGUF files across quantizations
        for q_type in ["Q4_K_M", "Q8_0", "FP16", "Q4_0", "Q5_K_M"]:
            valid_gguf = tmp_path / f"valid_{q_type.lower()}.gguf"
            created_path = convert_to_gguf(out_merged, valid_gguf, quantization=q_type, allow_simulation=True)
            assert_test(Path(created_path).is_file(), f"convert_to_gguf creates file for {q_type}")

            val_rep = validate_gguf_file(valid_gguf)
            assert_test(val_rep["is_valid"] is True, f"Valid GGUF header for {q_type} returns is_valid=True")
            assert_test(val_rep["magic"] == "GGUF", f"Magic is 'GGUF' for {q_type}")
            assert_test(val_rep["version"] == 3, f"Version is 3 for {q_type}")
            assert_test(val_rep["tensor_count"] >= 1, f"Tensor count >= 1 for {q_type}")
            assert_test(val_rep["kv_count"] >= 1, f"KV count >= 1 for {q_type}")
            assert_test(val_rep["error_message"] is None, f"Error message is None for {q_type}")

        # ── Test Suite 3: Modelfile Generator Adversarial Tests ───────────────
        print("\n--- 3. Testing Ollama Modelfile Generator Adversarial Scenarios ---")
        modelfile_out = tmp_path / "Modelfile"
        mf_content = generate_modelfile(
            gguf_path=tmp_path / "valid_q4_k_m.gguf",
            output_path=modelfile_out,
            system_prompt="Custom enterprise policy prompt",
            num_ctx=8192,
            temperature=0.1,
        )
        assert_test(modelfile_out.is_file(), "Modelfile written to disk")
        assert_test("FROM" in mf_content, "Modelfile has FROM directive")
        assert_test("TEMPLATE" in mf_content, "Modelfile has TEMPLATE directive")
        assert_test("<|im_start|>" in mf_content, "Modelfile has ChatML start token")
        assert_test("<|im_end|>" in mf_content, "Modelfile has ChatML end token")
        assert_test('PARAMETER stop "<|im_end|>"' in mf_content, "Modelfile has stop token parameter")
        assert_test('PARAMETER stop "<|endoftext|>"' in mf_content, "Modelfile has endoftext stop parameter")
        assert_test("PARAMETER num_ctx 8192" in mf_content, "Modelfile has num_ctx 8192")
        assert_test("PARAMETER temperature 0.1" in mf_content, "Modelfile has temperature 0.1")
        assert_test("Custom enterprise policy prompt" in mf_content, "Modelfile has custom system prompt")

        parsed = parse_modelfile(modelfile_out)
        assert_test(parsed["from"] is not None, "Parsed Modelfile contains 'from'")
        assert_test("<|im_end|>" in parsed["stop_tokens"], "Parsed Modelfile contains '<|im_end|>' stop token")
        assert_test(parsed["parameters"].get("num_ctx") == "8192", "Parsed Modelfile num_ctx is 8192")

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 75)
    print(f"  All {passed_count} Adversarial Stress Tests PASSED in {total_time:.3f}s (0 failures)")
    print("=" * 75)
    return True


if __name__ == "__main__":
    success = run_all_m2_adversarial_tests()
    if not success:
        sys.exit(1)
