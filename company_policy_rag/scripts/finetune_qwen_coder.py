#!/usr/bin/env python3
"""
CLI entrypoint for LoRA / QLoRA Fine-Tuning of Qwen 2.5 Coder 7B.

Supports YAML/JSON configuration files via `--config` with full CLI argument
overrides, dataset pre-flight validation, `--dry-run`, and structured metrics reporting.

Usage:
    # 1. Using YAML configuration file:
    python scripts/finetune_qwen_coder.py --config configs/train_qwen_lora_default.yaml

    # 2. YAML config with CLI overrides:
    python scripts/finetune_qwen_coder.py --config configs/train_qwen_lora_default.yaml \
        --dataset_path data/sample_finetune/sharegpt_sample.jsonl \
        --learning_rate 1e-4 --num_train_epochs 2

    # 3. Dry-run validation only:
    python scripts/finetune_qwen_coder.py --config configs/train_smoke_test.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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
    from src.finetuning.trainer import (
        FineTuneConfig,
        train_lora,
    )

logger = logging.getLogger("finetune_cli")


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
    """Load configuration from a YAML or JSON file."""
    path = Path(config_path)
    if not path.is_file():
        # Try relative to PROJECT_ROOT or current dir
        alt_path = PROJECT_ROOT / config_path
        if alt_path.is_file():
            path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() in (".yaml", ".yml"):
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML is required to read .yaml config files. Install with `pip install pyyaml`.")
            data = yaml.safe_load(f)
        elif path.suffix.lower() == ".json":
            data = json.load(f)
        else:
            content = f.read()
            if YAML_AVAILABLE:
                try:
                    data = yaml.safe_load(content)
                except Exception:
                    data = json.loads(content)
            else:
                data = json.loads(content)

    return data or {}


def parse_target_modules(val: Any) -> List[str]:
    """Normalize target modules from list or comma-separated string."""
    if isinstance(val, list):
        return [str(m).strip() for m in val if str(m).strip()]
    if isinstance(val, str):
        return [m.strip() for m in val.split(",") if m.strip()]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the comprehensive CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen 2.5 Coder 7B with LoRA/QLoRA on Custom Instruction Datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config & Mode
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to YAML or JSON config file.")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Validate config and dataset without running training.")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Enable verbose debug output.")

    # Model & Tokenizer
    parser.add_argument("--model_name_or_path", "-m", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="HuggingFace model ID or local path.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--trust_remote_code", action="store_true", default=True, help="Trust remote code.")

    # Dataset & Hygiene
    parser.add_argument("--dataset_path", "-d", type=str, default="", help="Path to training dataset (.json or .jsonl).")
    parser.add_argument("--val_dataset_path", type=str, default=None, help="Path to separate validation dataset.")
    parser.add_argument("--dataset_format", type=str, default="auto", choices=["auto", "alpaca", "sharegpt", "prompt_response"], help="Dataset format.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of dataset samples.")
    parser.add_argument("--skip_malformed", action="store_true", default=True, help="Skip malformed dataset lines.")

    # PEFT / LoRA Hyperparameters
    parser.add_argument("--quantization", "-q", type=str, default="4bit", choices=["4bit", "8bit", "none"], help="Quantization mode.")
    parser.add_argument("--lora_r", "-r", type=int, default=16, help="LoRA rank dimension.")
    parser.add_argument("--lora_alpha", "-a", type=int, default=32, help="LoRA alpha scaling factor.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate.")
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"], help="LoRA bias mode.")
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", help="Comma-separated target projection modules.")

    # Training Hyperparameters
    parser.add_argument("--output_dir", "-o", type=str, default="./outputs/qwen2.5-coder-7b-lora", help="Output directory for checkpoints and adapter.")
    parser.add_argument("--num_train_epochs", "-e", type=float, default=3.0, help="Number of training epochs.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum training steps (overrides epochs if > 0).")
    parser.add_argument("--per_device_train_batch_size", "--train_batch_size", type=int, default=2, help="Training batch size per device.")
    parser.add_argument("--per_device_eval_batch_size", "--eval_batch_size", type=int, default=2, help="Evaluation batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", "--gradient_accum", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", "--lr", type=float, default=2e-4, help="Peak learning rate.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate schedule.")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup steps ratio.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing.")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit", help="Optimizer algorithm.")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16 mixed precision.")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use float16 mixed precision.")

    # Logging & Checkpoints
    parser.add_argument("--logging_steps", type=int, default=10, help="Metrics logging interval.")
    parser.add_argument("--eval_steps", type=int, default=50, help="Validation evaluation interval.")
    parser.add_argument("--save_steps", type=int, default=50, help="Checkpoint save interval.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Maximum checkpoints to keep.")

    # Testing / CI
    parser.add_argument("--smoke_test", action="store_true", default=False, help="Run in fast smoke test mode.")
    parser.add_argument("--use_cpu", action="store_true", default=False, help="Force CPU execution.")

    return parser


def parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Two-phase parsing supporting --config with CLI overrides."""
    if argv is None:
        argv = sys.argv[1:]

    # Phase 1: Check for --config
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", "-c", type=str, default=None)
    known, _ = config_parser.parse_known_args(argv)

    # Main parser
    parser = build_arg_parser()

    # Phase 2: If config provided, load and set defaults
    if known.config:
        file_config = load_config_file(known.config)
        # Normalize target_modules if it is a list in YAML
        if "target_modules" in file_config and isinstance(file_config["target_modules"], list):
            file_config["target_modules"] = ",".join(file_config["target_modules"])
        parser.set_defaults(**file_config)

    # Phase 3: Parse full args (CLI overrides defaults)
    args = parser.parse_args(argv)
    return args


def create_fine_tune_config(args: argparse.Namespace) -> FineTuneConfig:
    """Construct and validate FineTuneConfig dataclass from parsed arguments."""
    target_modules = parse_target_modules(args.target_modules)

    config = FineTuneConfig(
        model_name_or_path=args.model_name_or_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        val_dataset_path=args.val_dataset_path,
        dataset_format=args.dataset_format,
        val_split=args.val_split,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        quantization=args.quantization,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_bias=args.lora_bias,
        target_modules=target_modules,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        smoke_test=args.smoke_test,
        use_cpu=args.use_cpu,
        skip_malformed=args.skip_malformed,
        max_samples=args.max_samples,
    )
    return config


def print_banner(config: FineTuneConfig) -> None:
    """Print configuration summary banner."""
    print("=" * 70)
    print("  Qwen 2.5 Coder 7B LoRA / QLoRA Fine-Tuning Pipeline")
    print("=" * 70)
    print(f"  Base Model       : {config.model_name_or_path}")
    print(f"  Dataset Path     : {config.dataset_path}")
    print(f"  Dataset Format   : {config.dataset_format}")
    print(f"  Validation Split : {config.val_split} (seed={config.seed})")
    print(f"  Quantization     : {config.quantization}")
    print(f"  LoRA Config      : rank={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    print(f"  Target Modules   : {', '.join(config.target_modules)}")
    print(f"  Batch Size       : train={config.per_device_train_batch_size}, accum={config.gradient_accumulation_steps}")
    print(f"  Learning Rate    : {config.learning_rate} ({config.lr_scheduler_type})")
    print(f"  Epochs / Steps   : epochs={config.num_train_epochs}, max_steps={config.max_steps}")
    print(f"  Output Directory : {config.output_dir}")
    print(f"  Smoke Test Mode  : {config.smoke_test} (use_cpu={config.use_cpu})")
    print("=" * 70)


def print_results(results: Dict[str, Any]) -> None:
    """Print training results summary table."""
    print("\n" + "=" * 70)
    print("  Training Execution Complete — Results Summary")
    print("=" * 70)
    print(f"  Output Directory   : {results.get('output_dir', 'N/A')}")
    print(f"  Total Steps        : {results.get('global_step', results.get('total_steps', 'N/A'))}")
    
    train_loss = results.get("train_loss")
    if isinstance(train_loss, (int, float)):
        print(f"  Final Train Loss   : {train_loss:.4f}")
    else:
        print(f"  Final Train Loss   : {train_loss}")

    eval_loss = results.get("eval_loss")
    if isinstance(eval_loss, (int, float)):
        print(f"  Final Eval Loss    : {eval_loss:.4f}")
    else:
        print(f"  Final Eval Loss    : {eval_loss}")

    eval_ppl = results.get("eval_perplexity")
    if isinstance(eval_ppl, (int, float)):
        print(f"  Eval Perplexity    : {eval_ppl:.4f}")
    else:
        print(f"  Eval Perplexity    : {eval_ppl}")

    runtime = results.get("train_runtime")
    if isinstance(runtime, (int, float)):
        print(f"  Runtime (seconds)  : {runtime:.2f}")
    else:
        print(f"  Runtime (seconds)  : {runtime}")
    print("=" * 70 + "\n")


def resolve_dataset_path(path_str: str) -> Path:
    """Resolve dataset path supporting cwd, relative, and workspace paths."""
    p = Path(path_str)
    if p.is_file():
        return p
    alt = PROJECT_ROOT / path_str
    if alt.is_file():
        return alt
    alt2 = WORKSPACE_ROOT / path_str
    if alt2.is_file():
        return alt2
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI execution function."""
    args = parse_cli_args(argv)
    setup_cli_logging(verbose=args.verbose)

    try:
        config = create_fine_tune_config(args)
        print_banner(config)

        # Pre-flight dataset validation
        if config.dataset_path:
            dataset_path = resolve_dataset_path(config.dataset_path)
            if not dataset_path.is_file():
                logger.error(f"Dataset file not found: {config.dataset_path} (resolved: {dataset_path})")
                return 1

            config.dataset_path = str(dataset_path)

            if config.val_dataset_path:
                val_path = resolve_dataset_path(config.val_dataset_path)
                if not val_path.is_file():
                    logger.error(f"Validation dataset file not found: {config.val_dataset_path}")
                    return 1
                config.val_dataset_path = str(val_path)

            logger.info(f"Validating dataset at: {dataset_path}")
            train_ds, val_ds = load_dataset_from_file(
                file_path=str(dataset_path),
                val_split=config.val_split,
                seed=config.seed,
                val_dataset_path=config.val_dataset_path,
                dataset_format=config.dataset_format if config.dataset_format != "auto" else None,
                skip_malformed=config.skip_malformed,
                max_samples=config.max_samples,
            )
            val_len = len(val_ds) if val_ds is not None else 0
            logger.info(f"Dataset successfully validated. Train samples: {len(train_ds)}, Val samples: {val_len}")

            # Compute and log profiling statistics
            stats = compute_dataset_statistics(train_ds, max_seq_length=config.max_seq_length)
            logger.info(
                "Dataset profiling: total_turns=%d, mean_turns=%.1f, len_median=%.1f, len_max=%d",
                stats["total_turns"],
                stats["mean_turns_per_sample"],
                stats["length_stats"]["median"],
                stats["length_stats"]["max"],
            )

        elif not config.smoke_test:
            logger.error("No --dataset_path provided. Specify a valid dataset file path.")
            return 1

        if args.dry_run:
            logger.info("Dry-run requested: configuration and dataset validated successfully. Exiting without training.")
            return 0

        # Run training
        logger.info("Initiating LoRA / QLoRA training run...")
        results = train_lora(config)
        print_results(results)
        return 0

    except DatasetFormatValidationError as e:
        logger.error(f"Dataset Schema Validation Error: {e}")
        return 1
    except DatasetEmptyError as e:
        logger.error(f"Dataset Empty Error: {e}")
        return 1
    except FileNotFoundError as e:
        logger.error(f"File Not Found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
