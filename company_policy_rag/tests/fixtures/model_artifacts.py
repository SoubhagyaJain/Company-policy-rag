"""Synthetic fixtures only. These are never production model artifacts."""

import json
import struct
from pathlib import Path
from company_policy_rag.src.finetuning.merger import DEFAULT_CHATML_TEMPLATE

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3


def synthesize_fallback_merged_artifacts(output_dir: Path, dtype_str: str = "float16") -> None:
    """Synthesize valid mock metadata, ChatML tokens, config.json, and safetensors for offline/test environments."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. config.json
    config_file = output_dir / "config.json"
    config_data = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": 3584,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "vocab_size": 152064,
        "torch_dtype": "bfloat16" if "bf16" in dtype_str.lower() else "float16",
        "transformers_version": "4.45.0",
    }
    config_file.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

    # 2. tokenizer_config.json
    tok_config_file = output_dir / "tokenizer_config.json"
    tok_config_data = {
        "chat_template": DEFAULT_CHATML_TEMPLATE,
        "model_max_length": 8192,
        "tokenizer_class": "Qwen2TokenizerFast",
        "clean_up_tokenization_spaces": False,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
    }
    tok_config_file.write_text(json.dumps(tok_config_data, indent=2), encoding="utf-8")

    # 3. special_tokens_map.json
    spec_tokens_file = output_dir / "special_tokens_map.json"
    spec_tokens_data = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
    }
    spec_tokens_file.write_text(json.dumps(spec_tokens_data, indent=2), encoding="utf-8")

    # 4. tokenizer.json
    tok_file = output_dir / "tokenizer.json"
    if not tok_file.exists() or tok_file.stat().st_size == 0:
        tok_file.write_text(
            json.dumps({"version": "1.0", "truncation": None, "padding": None}, indent=2), encoding="utf-8"
        )

    # 5. model.safetensors
    st_file = output_dir / "model.safetensors"
    if not st_file.exists() or st_file.stat().st_size == 0:
        st_file.write_bytes(b"MOCK_FP16_MERGED_WEIGHTS_SAFETENSORS")


def _write_simulated_gguf(
    output_path: Path,
    quantization: str,
    architecture: str = "qwen2",
) -> None:
    """Generate a structurally valid GGUF binary containing header metadata and quantization tag."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # GGUF Header: Magic (4B) + Version 3 (4B) + Tensor Count (8B) + KV Count (8B)
    magic = GGUF_MAGIC
    version = struct.pack("<I", GGUF_VERSION)
    tensor_count = struct.pack("<Q", 1)  # 1 tensor
    kv_count = struct.pack("<Q", 2)  # 2 metadata KV pairs

    header = magic + version + tensor_count + kv_count

    # KV pair 1: "general.architecture" -> "qwen2"
    key1 = "general.architecture".encode("utf-8")
    val1 = architecture.encode("utf-8")
    kv1 = (
        struct.pack("<Q", len(key1))
        + key1
        + struct.pack("<I", 8)  # GGUF_TYPE_STRING = 8
        + struct.pack("<Q", len(val1))
        + val1
    )

    # KV pair 2: "general.quantization_type" -> quantization format string (e.g. "Q4_K_M")
    key2 = "general.quantization_type".encode("utf-8")
    val2 = quantization.encode("utf-8")
    kv2 = (
        struct.pack("<Q", len(key2))
        + key2
        + struct.pack("<I", 8)  # GGUF_TYPE_STRING = 8
        + struct.pack("<Q", len(val2))
        + val2
    )

    # Tensor Info 1: "token_embd.weight"
    t_name = "token_embd.weight".encode("utf-8")
    t_info = (
        struct.pack("<Q", len(t_name))
        + t_name
        + struct.pack("<I", 1)  # n_dims = 1
        + struct.pack("<Q", 128)  # dim[0] = 128
        + struct.pack("<I", 1)  # type = GGML_TYPE_F16 (1)
        + struct.pack("<Q", 0)  # offset = 0
    )

    # 32-byte alignment padding + dummy tensor payload with explicit quantization marker
    payload = quantization.encode("utf-8") + b"\x00" * 256

    full_binary = header + kv1 + kv2 + t_info + payload
    output_path.write_bytes(full_binary)
