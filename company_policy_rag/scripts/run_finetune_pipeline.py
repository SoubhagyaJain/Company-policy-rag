#!/usr/bin/env python3
"""
Unified End-to-End Fine-Tuning and Deployment Pipeline for Qwen 2.5 Coder 7B.

Orchestrates the entire lifecycle in a single workflow:
1. Dataset Ingestion, Validation & Profiling (Alpaca, ShareGPT, JSONL)
2. LoRA / QLoRA Training (4-bit NF4, loss masking, perplexity logging)
3. Standalone Weight Merging (merge_and_unload, safetensors, ChatML tokens)
4. GGUF Export & Quantization (Q4_K_M, Q8_0, FP16)
5. Ollama Modelfile Generation (ChatML template, stop tokens, system prompt)
6. Direct Ollama Storage Registration (REST API / CLI)
7. Tag Verification & Service Probe

Usage:
    # 1. Full end-to-end pipeline execution:
    python scripts/run_finetune_pipeline.py \
        --dataset_path data/sample_finetune/sharegpt_sample.jsonl \
        --output_dir ./outputs/pipeline_run

    # 2. Lightweight CI / Smoke test mode:
    python scripts/run_finetune_pipeline.py \
        --dataset_path data/sample_finetune/sharegpt_sample.jsonl \
        --smoke-test --dry-run

    # 3. Resume from existing adapter (skip training):
    python scripts/run_finetune_pipeline.py \
        --output_dir ./outputs/pipeline_run \
        --skip-train
"""

from __future__ import annotations

import argparse
import json
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
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from company_policy_rag.src.finetuning.dataset_loader import (
        DatasetEmptyError,
        DatasetFormatValidationError,
        DatasetValidationError,
        compute_dataset_statistics,
        load_dataset_from_file,
    )
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
    from company_policy_rag.src.finetuning.trainer import (
        FineTuneConfig,
        train_lora,
    )
except ImportError:
    from src.finetuning.dataset_loader import (
        DatasetEmptyError,
        DatasetFormatValidationError,
        DatasetValidationError,
        compute_dataset_statistics,
        load_dataset_from_file,
    )
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
    from src.finetuning.trainer import (
        FineTuneConfig,
        train_lora,
    )

logger = logging.getLogger("finetune_pipeline")


def setup_cli_logging(verbose: bool = False) -> None:
    """Configure structured console logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def load_config_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)
    if not path.is_file():
        alt = PROJECT_ROOT / config_path
        if alt.is_file():
            path = alt
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() in (".yaml", ".yml"):
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML is required for .yaml configs.")
            return yaml.safe_load(f) or {}
        return json.load(f) or {}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build unified pipeline argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified Training, Merging, GGUF Export & Ollama Registration Pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General & Mode
    parser.add_argument("--config", "-c", type=str, default=None, help="Pipeline config YAML/JSON.")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Perform full validation without training or disk writes.")
    parser.add_argument("--smoke-test", action="store_true", default=False, help="Execute lightweight smoke-test mode.")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Verbose debug logging.")

    # Master Output Dir
    parser.add_argument("--output_dir", "-o", type=str, default="./outputs/pipeline_run", help="Master pipeline output directory.")

    # Stage 1: Dataset
    parser.add_argument("--dataset_path", "-d", type=str, default="", help="Path to instruction dataset.")
    parser.add_argument("--val_dataset_path", type=str, default=None, help="Path to validation dataset.")
    parser.add_argument("--dataset_format", type=str, default="auto", choices=["auto", "alpaca", "sharegpt", "prompt_response"])
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_samples", type=int, default=None, help="Max dataset samples.")

    # Stage 2: Fine-Tuning
    parser.add_argument("--model_name_or_path", "-m", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Base model ID or path.")
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "8bit", "none"], help="Training quantization mode.")
    parser.add_argument("--lora_r", "-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", "-a", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--learning_rate", "--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--num_train_epochs", "-e", type=float, default=3.0, help="Training epochs.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps (-1 for epochs).")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation.")
    parser.add_argument("--use_cpu", action="store_true", default=False, help="Force CPU training.")

    # Stage 3: Merge
    parser.add_argument("--merge_device", type=str, default="cpu", choices=["cpu", "cuda", "auto"], help="Merge device.")
    parser.add_argument("--merge_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Merged weight dtype.")

    # Stage 4: GGUF Quantization
    parser.add_argument("--gguf_quantization", "-gq", type=str, default="Q4_K_M", choices=SUPPORTED_QUANTIZATIONS, help="GGUF quantization.")

    # Stage 5: Modelfile
    parser.add_argument("--system_prompt", type=str, default=None, help="Custom enterprise system prompt.")
    parser.add_argument("--num_ctx", type=int, default=8192, help="Ollama context window.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")

    # Stage 6: Ollama Registration
    parser.add_argument("--ollama_model_name", "-n", type=str, default="qwen2.5-coder-7b-policy", help="Registered model tag.")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434", help="Ollama URL.")
    parser.add_argument("--prefer_cli", action="store_true", default=False, help="Prefer CLI over REST API.")

    # Granular Stage Skipping
    parser.add_argument("--skip-train", action="store_true", default=False, help="Skip training stage.")
    parser.add_argument("--skip-merge", action="store_true", default=False, help="Skip model merge stage.")
    parser.add_argument("--skip-quant", action="store_true", default=False, help="Skip GGUF quantization stage.")
    parser.add_argument("--skip-register", action="store_true", default=False, help="Skip Ollama registration stage.")
    parser.add_argument("--skip-verify", action="store_true", default=False, help="Skip post-registration verification.")

    return parser


def parse_pipeline_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Two-phase parsing supporting --config with CLI overrides."""
    if argv is None:
        argv = sys.argv[1:]

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", "-c", type=str, default=None)
    known, _ = config_parser.parse_known_args(argv)

    parser = build_arg_parser()
    if known.config:
        cfg = load_config_file(known.config)
        parser.set_defaults(**cfg)

    return parser.parse_args(argv)


def print_pipeline_header(args: argparse.Namespace) -> None:
    """Print ASCII pipeline banner."""
    print("=" * 76)
    print("   QWEN 2.5 CODER 7B: UNIFIED FINE-TUNING & DEPLOYMENT PIPELINE")
    print("=" * 76)
    print(f"  Base Model       : {args.model_name_or_path}")
    print(f"  Dataset          : {args.dataset_path or '[None / Skipped]'}")
    print(f"  Output Root      : {args.output_dir}")
    print(f"  GGUF Quant       : {args.gguf_quantization}")
    print(f"  Ollama Target    : {args.ollama_model_name} ({args.ollama_url})")
    print(f"  Execution Mode   : Smoke={args.smoke_test}, DryRun={args.dry_run}")
    print("=" * 76)


def print_pipeline_summary(stage_timings: Dict[str, float], artifacts: Dict[str, Any]) -> None:
    """Print structured summary report."""
    print("\n" + "=" * 76)
    print("   PIPELINE EXECUTION COMPLETE — STAGE SUMMARY REPORT")
    print("=" * 76)
    for stage, elapsed in stage_timings.items():
        print(f"  - {stage:<28} : {elapsed:6.2f}s")
    total = sum(stage_timings.values())
    print("-" * 76)
    print(f"  TOTAL ELAPSED TIME           : {total:6.2f}s")
    print("=" * 76)
    print("  PRODUCED ARTIFACTS:")
    for name, val in artifacts.items():
        print(f"  - {name:<26} : {val}")
    print("=" * 76 + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    """Main pipeline execution entrypoint."""
    args = parse_pipeline_args(argv)
    setup_cli_logging(verbose=args.verbose)
    print_pipeline_header(args)

    out_root = Path(args.output_dir)
    adapter_dir = out_root / "adapter"
    merged_dir = out_root / "merged"
    quant_tag = args.gguf_quantization.lower().replace("_", "-")
    gguf_path = out_root / f"{args.ollama_model_name}-{quant_tag}.gguf"
    modelfile_path = out_root / "Modelfile"

    stage_timings: Dict[str, float] = {}
    artifacts: Dict[str, Any] = {}

    try:
        # Pre-flight Validation / Dry-Run
        if args.dataset_path and Path(args.dataset_path).is_file():
            logger.info("Validating dataset at: %s", args.dataset_path)
            train_ds, val_ds = load_dataset_from_file(
                file_path=args.dataset_path,
                val_split=args.val_split,
                seed=args.seed,
                max_samples=args.max_samples,
            )
            stats = compute_dataset_statistics(train_ds)
            logger.info("Dataset validated: %d train samples, %d val samples, %d total turns",
                        len(train_ds), len(val_ds) if val_ds else 0, stats["total_turns"])

        if args.dry_run:
            logger.info("Dry-run validation successful. Exiting without execution.")
            return 0

        # Stage 1: Fine-Tuning
        if not args.skip_train:
            logger.info("==> [1/6] Running LoRA Fine-Tuning Stage...")
            t0 = time.time()
            ft_config = FineTuneConfig(
                model_name_or_path=args.model_name_or_path,
                dataset_path=args.dataset_path,
                output_dir=str(adapter_dir),
                val_dataset_path=args.val_dataset_path,
                dataset_format=args.dataset_format,
                val_split=args.val_split,
                seed=args.seed,
                quantization=args.quantization,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                learning_rate=args.learning_rate,
                num_train_epochs=args.num_train_epochs,
                max_steps=args.max_steps,
                per_device_train_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                smoke_test=args.smoke_test,
                use_cpu=args.use_cpu,
                max_samples=args.max_samples,
            )
            train_res = train_lora(ft_config)
            stage_timings["1. LoRA Training"] = time.time() - t0
            artifacts["Adapter Directory"] = str(adapter_dir)
            artifacts["Final Eval Loss"] = train_res.get("eval_loss")
            artifacts["Final Perplexity"] = train_res.get("eval_perplexity")
        else:
            logger.info("==> [1/6] Skipping training stage.")

        # Stage 2: Model Merging
        if not args.skip_merge:
            logger.info("==> [2/6] Running LoRA Adapter Merging Stage...")
            t0 = time.time()
            merge_lora_weights(
                base_model_path=args.model_name_or_path,
                adapter_path=str(adapter_dir),
                output_dir=str(merged_dir),
                device=args.merge_device,
                dtype=args.merge_dtype,
            )
            stage_timings["2. LoRA Merge"] = time.time() - t0
            artifacts["Merged Weights Dir"] = str(merged_dir)
        else:
            logger.info("==> [2/6] Skipping weight merging stage.")

        # Stage 3: GGUF Conversion & Quantization
        if not args.skip_quant:
            logger.info("==> [3/6] Running GGUF Conversion Stage (%s)...", args.gguf_quantization)
            t0 = time.time()
            convert_to_gguf(
                model_dir=str(merged_dir),
                output_file=str(gguf_path),
                quantization=args.gguf_quantization,
            )
            stage_timings["3. GGUF Export"] = time.time() - t0
            artifacts["GGUF Binary"] = str(gguf_path)
        else:
            logger.info("==> [3/6] Skipping GGUF conversion stage.")

        # Stage 4: Modelfile Generation
        logger.info("==> [4/6] Generating Ollama Modelfile Stage...")
        t0 = time.time()
        generate_modelfile(
            gguf_path=str(gguf_path),
            output_path=str(modelfile_path),
            system_prompt=args.system_prompt,
            num_ctx=args.num_ctx,
            temperature=args.temperature,
        )
        stage_timings["4. Modelfile Gen"] = time.time() - t0
        artifacts["Ollama Modelfile"] = str(modelfile_path)

        # Stage 5: Ollama Registration
        if not args.skip_register:
            logger.info("==> [5/6] Registering Model in Ollama ('%s')...", args.ollama_model_name)
            t0 = time.time()
            reg_ok = register_model_in_ollama(
                model_name=args.ollama_model_name,
                modelfile_path=str(modelfile_path),
                ollama_url=args.ollama_url,
                prefer_api=not args.prefer_cli,
            )
            stage_timings["5. Ollama Registration"] = time.time() - t0
            artifacts["Registration Status"] = "SUCCESS" if reg_ok else "OFFLINE / UNREACHABLE"
        else:
            logger.info("==> [5/6] Skipping Ollama registration stage.")

        # Stage 6: Tag Verification
        if not args.skip_verify and not args.skip_register:
            logger.info("==> [6/6] Verifying Registered Model in Ollama Manifest...")
            t0 = time.time()
            verified = verify_model_registered(args.ollama_model_name, ollama_url=args.ollama_url)
            stage_timings["6. Verification Probe"] = time.time() - t0
            artifacts["Manifest Tag Verified"] = verified
        else:
            logger.info("==> [6/6] Skipping verification probe.")

        print_pipeline_summary(stage_timings, artifacts)
        return 0

    except Exception as exc:
        logger.error(f"Unified pipeline failed: {exc}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
