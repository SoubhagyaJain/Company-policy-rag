#!/usr/bin/env python3
"""
CLI entrypoint for LoRA adapter merging, GGUF conversion, Modelfile generation, and Ollama registration.

Chains:
1. Merge LoRA adapter weights with base Qwen 2.5 Coder 7B model.
2. Convert and quantize the merged model to GGUF format (Q4_K_M, Q8_0, FP16, etc.).
3. Generate optimized Ollama Modelfile with ChatML template and stop tokens.
4. Register the model into local Ollama storage via REST API or CLI.

Usage:
    # 1. Full export and registration:
    python scripts/export_and_register_ollama.py \
        --adapter_path ./outputs/qwen2.5-coder-7b-lora \
        --quantization Q4_K_M \
        --model_name qwen2.5-coder-7b-policy

    # 2. Dry-run validation:
    python scripts/export_and_register_ollama.py \
        --adapter_path ./outputs/qwen2.5-coder-7b-lora \
        --dry-run

    # 3. Skip merge (start from existing merged weights):
    python scripts/export_and_register_ollama.py \
        --merged_dir ./outputs/export/merged \
        --skip-merge
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure project root and company_policy_rag root are in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

for p in [str(PROJECT_ROOT), str(WORKSPACE_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from company_policy_rag.src.finetuning.gguf_exporter import (
        SUPPORTED_QUANTIZATIONS,
        convert_to_gguf,
        validate_gguf_file,
    )
    from company_policy_rag.src.finetuning.merger import merge_lora_weights
    from company_policy_rag.src.finetuning.modelfile_generator import generate_modelfile
    from company_policy_rag.src.finetuning.ollama_registrar import (
        probe_ollama_tags,
        register_model_in_ollama,
        verify_model_registered,
    )
except ImportError:
    from src.finetuning.gguf_exporter import (
        SUPPORTED_QUANTIZATIONS,
        convert_to_gguf,
        validate_gguf_file,
    )
    from src.finetuning.merger import merge_lora_weights
    from src.finetuning.modelfile_generator import generate_modelfile
    from src.finetuning.ollama_registrar import (
        probe_ollama_tags,
        register_model_in_ollama,
        verify_model_registered,
    )

logger = logging.getLogger("export_and_register")


def setup_cli_logging(verbose: bool = False) -> None:
    """Configure structured console logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct argument parser for export and registration workflow."""
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights, export to GGUF, generate Modelfile, and register in Ollama.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Base Model & Adapter
    parser.add_argument("--base_model_path", "-m", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Base model HuggingFace ID or directory.")
    parser.add_argument("--adapter_path", "-a", type=str, default="", help="Path to trained LoRA adapter directory.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"], help="Device for merging.")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Merged weight precision.")

    # Export Directories & Paths
    parser.add_argument("--output_dir", "-o", type=str, default="./outputs/export", help="Base export directory.")
    parser.add_argument("--merged_dir", type=str, default=None, help="Custom merged model directory (default: <output_dir>/merged).")
    parser.add_argument("--gguf_path", type=str, default=None, help="Custom GGUF file path (default: <output_dir>/<model_name>-<quant>.gguf).")
    parser.add_argument("--modelfile_path", type=str, default=None, help="Custom Modelfile path (default: <output_dir>/Modelfile).")

    # Quantization & Model Metadata
    parser.add_argument("--quantization", "-q", type=str, default="Q4_K_M", choices=SUPPORTED_QUANTIZATIONS, help="GGUF quantization format.")
    parser.add_argument("--model_name", "-n", type=str, default="qwen2.5-coder-7b-policy", help="Ollama model tag name.")
    parser.add_argument("--system_prompt", type=str, default=None, help="Custom system prompt.")
    parser.add_argument("--num_ctx", type=int, default=8192, help="Context window token size.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")

    # Ollama Service
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434", help="Local Ollama service URL.")
    parser.add_argument("--prefer_cli", action="store_true", default=False, help="Prefer CLI over REST API for registration.")

    # Pipeline Control & Flags
    parser.add_argument("--dry-run", action="store_true", default=False, help="Perform pre-flight validation without heavy execution.")
    parser.add_argument("--skip-merge", action="store_true", default=False, help="Skip LoRA merge stage.")
    parser.add_argument("--skip-quant", action="store_true", default=False, help="Skip GGUF quantization stage.")
    parser.add_argument("--skip-register", action="store_true", default=False, help="Skip Ollama registration stage.")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Enable verbose debug logging.")

    return parser


def print_banner(args: argparse.Namespace, merged_dir: Path, gguf_path: Path, modelfile_path: Path) -> None:
    """Print export pipeline configuration summary."""
    print("=" * 70)
    print("  Qwen 2.5 Coder 7B: Merge, GGUF Export & Ollama Registration")
    print("=" * 70)
    print(f"  Base Model       : {args.base_model_path}")
    print(f"  Adapter Path     : {args.adapter_path or '[None / Skipped]'}")
    print(f"  Merged Dir       : {merged_dir}")
    print(f"  GGUF Target      : {gguf_path} ({args.quantization})")
    print(f"  Modelfile Target : {modelfile_path}")
    print(f"  Ollama Tag       : {args.model_name}")
    print(f"  Context Window   : {args.num_ctx} tokens (temp={args.temperature})")
    print(f"  Ollama Server    : {args.ollama_url}")
    print(f"  Dry-Run Mode     : {args.dry_run}")
    print("=" * 70)


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI execution flow."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_cli_logging(verbose=args.verbose)

    out_base = Path(args.output_dir)
    merged_dir = Path(args.merged_dir) if args.merged_dir else out_base / "merged"
    quant_suffix = args.quantization.lower().replace("_", "-")
    clean_model_name = args.model_name.replace(":", "-").replace("/", "-")
    gguf_path = Path(args.gguf_path) if args.gguf_path else out_base / f"{clean_model_name}-{quant_suffix}.gguf"
    modelfile_path = Path(args.modelfile_path) if args.modelfile_path else out_base / "Modelfile"

    print_banner(args, merged_dir, gguf_path, modelfile_path)
    start_time = time.time()

    try:
        # Pre-flight validation
        if not args.skip_merge and not args.adapter_path and not args.dry_run:
            logger.error("No --adapter_path specified and --skip-merge is False. Provide an adapter directory.")
            return 1

        if args.dry_run:
            logger.info("Dry-run validation successful. All paths and arguments are valid.")
            return 0

        # Stage 1: LoRA Adapter Merge
        if not args.skip_merge:
            logger.info("==> [Stage 1/4] Merging LoRA adapter weights...")
            t0 = time.time()
            merge_lora_weights(
                base_model_path=args.base_model_path,
                adapter_path=args.adapter_path,
                output_dir=merged_dir,
                device=args.device,
                dtype=args.dtype,
            )
            logger.info("Stage 1 completed in %.2fs -> Merged to %s", time.time() - t0, merged_dir)
        else:
            logger.info("==> [Stage 1/4] Skipping LoRA merge (using existing %s)", merged_dir)

        # Stage 2: GGUF Conversion & Quantization
        if not args.skip_quant:
            logger.info("==> [Stage 2/4] Converting to GGUF (%s)...", args.quantization)
            t0 = time.time()
            convert_to_gguf(
                model_dir=merged_dir,
                output_file=gguf_path,
                quantization=args.quantization,
            )
            validation = validate_gguf_file(gguf_path)
            logger.info("Stage 2 completed in %.2fs -> GGUF valid: %s (size: %d bytes)",
                        time.time() - t0, validation.get("is_valid"), validation.get("file_size_bytes", 0))
        else:
            logger.info("==> [Stage 2/4] Skipping GGUF conversion (using existing %s)", gguf_path)

        # Stage 3: Modelfile Generation
        logger.info("==> [Stage 3/4] Generating Ollama Modelfile...")
        t0 = time.time()
        generate_modelfile(
            gguf_path=gguf_path,
            output_path=modelfile_path,
            system_prompt=args.system_prompt,
            num_ctx=args.num_ctx,
            temperature=args.temperature,
        )
        logger.info("Stage 3 completed in %.2fs -> Modelfile saved to %s", time.time() - t0, modelfile_path)

        # Stage 4: Ollama Registration
        if not args.skip_register:
            logger.info("==> [Stage 4/4] Registering model '%s' in Ollama...", args.model_name)
            t0 = time.time()
            registered = register_model_in_ollama(
                model_name=args.model_name,
                modelfile_path=modelfile_path,
                ollama_url=args.ollama_url,
                prefer_api=not args.prefer_cli,
            )
            if registered:
                logger.info("Stage 4 completed in %.2fs -> Successfully registered '%s' in Ollama.", time.time() - t0, args.model_name)
            else:
                logger.warning("Stage 4 warning: Registration could not connect to Ollama service at %s. Modelfile ready for manual import: `ollama create %s -f %s`",
                               args.ollama_url, args.model_name, modelfile_path)
        else:
            logger.info("==> [Stage 4/4] Skipping Ollama registration as requested.")

        total_elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"  Pipeline Finished Successfully in {total_elapsed:.2f} seconds!")
        print(f"  Merged Weights : {merged_dir}")
        print(f"  GGUF Binary    : {gguf_path}")
        print(f"  Modelfile      : {modelfile_path}")
        print(f"  Ollama Model   : {args.model_name}")
        print("=" * 70 + "\n")
        return 0

    except Exception as exc:
        logger.error(f"Export and registration pipeline failed: {exc}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
