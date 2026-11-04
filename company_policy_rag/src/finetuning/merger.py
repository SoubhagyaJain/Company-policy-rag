"""LoRA Adapter Weight Merger for Qwen 2.5 Coder 7B.

Merges fine-tuned LoRA/QLoRA adapter weights with base model weights into
a standalone, self-contained 16-bit Hugging Face model directory with full
tokenizer and ChatML special token configurations.

Authoritative Reference:
- ORIGINAL_REQUEST.md (§ R2. Model Merging, GGUF Export & Ollama Registration)
- PROJECT.md (§ Architecture, Feature Inventory F2.1, Interface Contracts)
- TEST_INFRA.md (§ Feature Inventory F2.1 & Tier 1/2 coverage)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

logger = logging.getLogger("model_merger")

DEFAULT_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


class ModelMergeError(Exception):
    """Base exception for all model merging failures."""
    pass


class AdapterNotFoundError(ModelMergeError):
    """Raised when adapter path does not exist or is missing configuration."""
    pass


class BaseModelNotFoundError(ModelMergeError):
    """Raised when base model path does not exist or cannot be resolved."""
    pass


class MergeValidationError(ModelMergeError):
    """Raised on invalid merge configuration or arguments."""
    pass


@dataclass
class MergeConfig:
    """Configuration container for LoRA model merging."""
    base_model_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    adapter_path: str = ""
    output_dir: str = "./outputs/qwen2.5-coder-7b-merged"
    device: str = "cpu"
    dtype: str = "float16"
    safe_serialization: bool = True
    max_shard_size: str = "5GB"
    trust_remote_code: bool = True
    low_cpu_mem_usage: bool = True

    def validate(self) -> None:
        """Validate merge parameters."""
        if not str(self.base_model_path).strip():
            raise ValueError("base_model_path cannot be empty")
        if not str(self.adapter_path).strip():
            raise ValueError("adapter_path cannot be empty")
        if not str(self.output_dir).strip():
            raise ValueError("output_dir cannot be empty")


@dataclass
class MergeOutput:
    """Output container for merge results."""
    status: str
    output_dir: str
    base_model: str
    adapter_path: str
    device: str
    dtype: str
    weight_files: List[str] = field(default_factory=list)
    elapsed_time_sec: float = 0.0


def resolve_torch_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Map string representation to torch.dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype
    d = str(dtype).lower().strip()
    if d in ("float16", "fp16", "f16"):
        return torch.float16
    elif d in ("bfloat16", "bf16", "bf"):
        return torch.bfloat16
    elif d in ("float32", "fp32", "f32"):
        return torch.float32
    elif d == "auto":
        return torch.float16
    raise ValueError(f"Unsupported dtype '{dtype}'. Choose 'float16', 'bfloat16', 'float32', or 'auto'.")


def resolve_device(device_str: str) -> Tuple[str, Optional[Union[str, Dict[str, str]]]]:
    """Sanitize and resolve device string and device_map with graceful CPU fallback."""
    dev = str(device_str).lower().strip()
    if dev in ("cuda", "gpu") or dev.startswith("cuda:"):
        if not torch.cuda.is_available():
            logger.warning("CUDA requested ('%s') but torch.cuda is not available; falling back to CPU.", device_str)
            return "cpu", None
        if dev.startswith("cuda:"):
            try:
                parts = dev.split(":")
                idx = int(parts[1]) if len(parts) > 1 else 0
                if idx < 0 or idx >= torch.cuda.device_count():
                    logger.warning("CUDA device index %d exceeds available GPUs (%d); falling back to CPU.", idx, torch.cuda.device_count())
                    return "cpu", None
            except (ValueError, IndexError):
                logger.warning("Invalid CUDA device specifier '%s'; falling back to CPU.", device_str)
                return "cpu", None
        return dev, "auto"
    elif dev == "auto":
        return ("cuda", "auto") if torch.cuda.is_available() else ("cpu", None)
    return "cpu", None


def cleanup_memory() -> None:
    """Perform aggressive garbage collection and GPU cache eviction."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        except Exception:
            pass




def merge_lora_weights(
    base_model_path: Union[str, Path],
    adapter_path: Union[str, Path],
    output_dir: Union[str, Path],
    device: str = "cpu",
    dtype: str = "float16",
    safe_serialization: bool = True,
    max_shard_size: str = "5GB",
    trust_remote_code: bool = True,
    low_cpu_mem_usage: bool = True,
) -> str:
    """Merge LoRA adapter weights with base model into standalone FP16/BF16 weights.

    Args:
        base_model_path: Path or Hugging Face Hub ID of base model.
        adapter_path: Path to directory containing trained LoRA adapter.
        output_dir: Target directory for merged standalone model weights and tokenizer.
        device: Target execution device ('cpu', 'cuda', 'auto').
        dtype: Target weight precision ('float16', 'bfloat16', 'float32').
        safe_serialization: If True, exports in .safetensors format.
        max_shard_size: Maximum shard size for sharded weights (e.g. '5GB').
        trust_remote_code: Allow custom model code from Hugging Face.
        low_cpu_mem_usage: Optimize RAM usage during weight loading.

    Returns:
        String path to the merged output directory.
    """
    base_str = str(base_model_path).strip()
    adapter_str = str(adapter_path).strip()

    if not base_str:
        raise ValueError("base_model_path cannot be empty")
    if not adapter_str:
        raise ValueError("adapter_path cannot be empty")
    if not str(output_dir).strip():
        raise ValueError("output_dir cannot be empty")

    out_path = Path(output_dir)
    if out_path.exists():
        raise FileExistsError(f"Refusing to replace existing model output: {out_path}")
    if not Path(adapter_str).is_dir():
        raise AdapterNotFoundError(f"Adapter directory not found: {adapter_str}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    resolved_dev, device_map = resolve_device(device)
    resolved_dtype = resolve_torch_dtype(dtype)

    logger.info(
        "Merging LoRA weights: base='%s', adapter='%s', output='%s', device='%s', dtype='%s'",
        base_str,
        adapter_str,
        out_path,
        resolved_dev,
        dtype,
    )

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok_source = (
            adapter_str
            if Path(adapter_str).is_dir() and (Path(adapter_str) / "tokenizer_config.json").exists()
            else base_str
        )
        with tempfile.TemporaryDirectory(prefix=".merge-", dir=out_path.parent) as temporary:
            staging = Path(temporary) / "model"
            staging.mkdir()
            tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=trust_remote_code, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
            tokenizer.save_pretrained(str(staging))
            model_kwargs: Dict[str, Any] = {
                "torch_dtype": resolved_dtype, "trust_remote_code": trust_remote_code,
                "low_cpu_mem_usage": low_cpu_mem_usage,
            }
            if device_map is not None:
                model_kwargs["device_map"] = device_map
            base_model = AutoModelForCausalLM.from_pretrained(base_str, **model_kwargs)
            peft_model = PeftModel.from_pretrained(base_model, adapter_str, is_trainable=False)
            merged_model = peft_model.merge_and_unload(progressbar=False, safe_merge=True)
            merged_model.save_pretrained(str(staging), safe_serialization=safe_serialization, max_shard_size=max_shard_size)
            for name in ("config.json", "tokenizer_config.json"):
                if not isinstance(json.loads((staging / name).read_text(encoding="utf-8")), dict):
                    raise ModelMergeError(f"Invalid exported {name}")
            weights = list(staging.glob("*.safetensors")) + list(staging.glob("pytorch_model*.bin"))
            if not weights or any(p.stat().st_size == 0 for p in weights):
                raise ModelMergeError("Model exporter produced no nonempty weight files.")
            if not any((staging / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
                raise ModelMergeError("Tokenizer exporter produced no vocabulary.")
            staging.rename(out_path)
    except Exception as exc:
        raise ModelMergeError(f"Model merge failed; no output was published: {exc}") from exc
    finally:
        cleanup_memory()

    elapsed = time.time() - start_time
    logger.info("Successfully exported merged model to '%s' in %.2f seconds", out_path, elapsed)
    return str(out_path)


class ModelMerger:
    """Object-oriented wrapper for LoRA adapter weight merging."""

    def __init__(
        self,
        device: str = "cpu",
        dtype: str = "float16",
        safe_serialization: bool = True,
        max_shard_size: str = "5GB",
        trust_remote_code: bool = True,
        low_cpu_mem_usage: bool = True,
    ):
        self.device = device
        self.dtype = dtype
        self.safe_serialization = safe_serialization
        self.max_shard_size = max_shard_size
        self.trust_remote_code = trust_remote_code
        self.low_cpu_mem_usage = low_cpu_mem_usage

    def merge(
        self,
        base_model_path: Union[str, Path],
        adapter_path: Union[str, Path],
        output_dir: Union[str, Path],
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> str:
        """Merge adapter into base model and export."""
        dev = device if device is not None else self.device
        dt = dtype if dtype is not None else self.dtype
        return merge_lora_weights(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            output_dir=output_dir,
            device=dev,
            dtype=dt,
            safe_serialization=self.safe_serialization,
            max_shard_size=self.max_shard_size,
            trust_remote_code=self.trust_remote_code,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
        )

    def merge_and_unload(
        self,
        base_model_path: Union[str, Path],
        adapter_path: Union[str, Path],
        output_dir: Union[str, Path],
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> str:
        """Alias for merge()."""
        return self.merge(base_model_path, adapter_path, output_dir, device=device, dtype=dtype)


def main() -> None:
    """CLI entrypoint for standalone model merger."""
    parser = argparse.ArgumentParser(description="Merge LoRA weights with base model into standalone FP16 weights.")
    parser.add_argument("--base_model_path", "-m", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Base model path or HF hub ID.")
    parser.add_argument("--adapter_path", "-a", type=str, required=True, help="Path to trained LoRA adapter directory.")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Target directory for merged weights.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"], help="Execution device.")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Output weight precision.")
    parser.add_argument("--max_shard_size", type=str, default="5GB", help="Max shard size for weights.")
    parser.add_argument("--no_safe_serialization", action="store_true", help="Disable safetensors serialization.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    out_dir = merge_lora_weights(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
        safe_serialization=not args.no_safe_serialization,
        max_shard_size=args.max_shard_size,
    )
    print(f"Model successfully merged into: {out_dir}")


if __name__ == "__main__":
    main()
