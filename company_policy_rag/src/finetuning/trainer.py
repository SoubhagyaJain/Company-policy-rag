"""LoRA / QLoRA Fine-Tuning Orchestrator for Qwen 2.5 Coder 7B."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import yaml
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

# Import dataset loader
try:
    from company_policy_rag.src.finetuning.dataset_loader import load_dataset_from_file
except ImportError:
    try:
        from src.finetuning.dataset_loader import load_dataset_from_file
    except ImportError:
        from .dataset_loader import load_dataset_from_file

logger = logging.getLogger(__name__)

# Default ChatML template for Qwen 2.5
DEFAULT_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


# ── 1. Configuration Dataclass ──────────────────────────────────────────────

@dataclass
class FineTuneConfig:
    """Configuration container for LoRA/QLoRA fine-tuning."""

    model_name_or_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    dataset_path: str = ""
    output_dir: str = "./outputs/qwen2.5-coder-7b-lora"
    val_dataset_path: Optional[str] = None
    dataset_format: str = "auto"
    val_split: float = 0.1
    seed: int = 42
    max_seq_length: int = 2048
    quantization: str = "4bit"  # "4bit", "8bit", "none"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    max_steps: int = -1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    bf16: bool = True
    fp16: bool = False
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 2
    trust_remote_code: bool = True
    report_to: str = "none"
    smoke_test: bool = False
    use_cpu: bool = False
    skip_malformed: bool = True
    max_samples: Optional[int] = None

    def validate(self) -> None:
        """Validate configuration settings and sanity-check values."""
        if not self.dataset_path and not self.smoke_test:
            raise ValueError("`dataset_path` must be specified.")
        if self.quantization not in ("4bit", "8bit", "none"):
            raise ValueError(
                f"Invalid quantization mode: '{self.quantization}'. Choose '4bit', '8bit', or 'none'."
            )
        if self.val_split < 0.0 or self.val_split >= 1.0:
            raise ValueError(f"`val_split` must be between 0.0 and 1.0, got {self.val_split}")
        if self.per_device_train_batch_size <= 0:
            raise ValueError(f"`per_device_train_batch_size` must be > 0, got {self.per_device_train_batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"`learning_rate` must be > 0, got {self.learning_rate}")
        if self.lora_r <= 0:
            raise ValueError(f"`lora_r` must be > 0, got {self.lora_r}")
        if not self.target_modules:
            raise ValueError("`target_modules` must not be empty.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FineTuneConfig":
        """Instantiate config from dictionary with unknown key filtering."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "FineTuneConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)


@dataclass
class TrainingOutput:
    """Output container for training results."""
    status: str
    output_dir: str
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    eval_perplexity: Optional[float] = None
    train_runtime: Optional[float] = None
    trainable_params: int = 0
    all_params: int = 0
    trainable_percent: float = 0.0
    metrics_summary: Dict[str, Any] = field(default_factory=dict)


# ── 2. Perplexity Calculation Guard ─────────────────────────────────────────

def calculate_perplexity(eval_loss: Optional[float]) -> float:
    """Safely calculate perplexity from cross-entropy loss with overflow protection.

    Args:
        eval_loss: Cross-entropy evaluation loss.

    Returns:
        exp(eval_loss), float('inf') if eval_loss > 100 or on overflow, float('nan') if None/NaN/<0.
    """
    if eval_loss is None:
        return float("nan")
    try:
        if math.isnan(eval_loss) or eval_loss < 0:
            return float("nan")
        if eval_loss > 100.0:
            return float("inf")
        return math.exp(eval_loss)
    except (OverflowError, ValueError):
        return float("inf")


# ── 3. Metrics Tracking Callback ────────────────────────────────────────────

class TrainingMetricsCallback(TrainerCallback):
    """Callback for step-level metrics logging and real-time history/summary export."""

    def __init__(self, output_dir: Union[Path, str]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {}

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        if not logs:
            return
        entry: Dict[str, Any] = {
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch is not None else None,
            **logs,
        }
        if "loss" in logs and isinstance(logs["loss"], (int, float)):
            entry["train_perplexity"] = calculate_perplexity(logs["loss"])
        if "eval_loss" in logs and isinstance(logs["eval_loss"], (int, float)):
            entry["eval_perplexity"] = calculate_perplexity(logs["eval_loss"])

        self.history.append(entry)
        try:
            with open(self.output_dir / "training_history.json", "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.warning("Failed to write training_history.json: %s", e)

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        eval_losses = [h["eval_loss"] for h in self.history if "eval_loss" in h and isinstance(h["eval_loss"], (int, float))]
        train_losses = [h["loss"] for h in self.history if "loss" in h and isinstance(h["loss"], (int, float))]

        best_eval_loss = min(eval_losses) if eval_losses else None
        final_eval_loss = eval_losses[-1] if eval_losses else None
        final_train_loss = train_losses[-1] if train_losses else None

        self.summary = {
            "best_global_step": getattr(state, "best_global_step", state.global_step),
            "best_eval_loss": best_eval_loss,
            "best_eval_perplexity": calculate_perplexity(best_eval_loss),
            "final_train_loss": final_train_loss,
            "final_eval_loss": final_eval_loss,
            "final_eval_perplexity": calculate_perplexity(final_eval_loss),
            "total_steps": state.global_step,
            "total_epochs": state.epoch,
            "history_entries": len(self.history),
        }
        try:
            with open(self.output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
                json.dump(self.summary, f, indent=2)
        except Exception as e:
            logger.warning("Failed to write metrics_summary.json: %s", e)


MetricsLoggingCallback = TrainingMetricsCallback


# ── 4. Model & Tokenizer Initializers ───────────────────────────────────────

def setup_tokenizer(config: FineTuneConfig) -> PreTrainedTokenizerBase:
    """Load and configure tokenizer for Qwen 2.5 Coder ChatML."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"

    # Ensure ChatML template exists
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = DEFAULT_CHATML_TEMPLATE

    return tokenizer


get_tokenizer = setup_tokenizer


def setup_model(
    config: FineTuneConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> PreTrainedModel:
    """Load base causal language model according to quantization and device profile."""
    quant_mode = config.quantization
    use_cpu = config.use_cpu or not torch.cuda.is_available()

    if use_cpu:
        if quant_mode in ("4bit", "8bit"):
            if not config.smoke_test and not config.use_cpu:
                raise RuntimeError(
                    f"BitsAndBytes {quant_mode} quantization requires a CUDA GPU. "
                    f"Set quantization='none' or use_cpu=True for CPU execution."
                )
            logger.warning("No CUDA GPU detected or use_cpu=True; falling back to unquantized CPU mode.")
            quant_mode = "none"

    bnb_config = None
    torch_dtype = torch.float32

    if not use_cpu:
        if config.bf16 and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif config.fp16:
            torch_dtype = torch.float16

        if quant_mode == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
        elif quant_mode == "8bit":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )

    device_map = None if use_cpu else "auto"

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
        "torch_dtype": torch_dtype,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        **model_kwargs,
    )

    if quant_mode in ("4bit", "8bit") and not use_cpu:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )

    return model


def setup_peft_config(config: FineTuneConfig) -> LoraConfig:
    """Construct LoraConfig targeting all 7 linear projection layers."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        target_modules=config.target_modules,
    )


def get_lora_model(model: PreTrainedModel, peft_config: LoraConfig) -> PeftModel:
    """Wrap base model with PEFT LoRA adapters."""
    return get_peft_model(model, peft_config)


# ── 5. Data Collation & ChatML Formatting ───────────────────────────────────

def get_completion_data_collator(
    tokenizer: PreTrainedTokenizerBase,
    response_template: str = "<|im_start|>assistant\n",
) -> DataCollatorForCompletionOnlyLM:
    """Create completion-only data collator for masking prompt token loss."""
    return DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )


create_completion_collator = get_completion_data_collator


def formatting_prompts_func(example: Dict[str, Any], tokenizer: PreTrainedTokenizerBase) -> str:
    """Format single record containing messages into ChatML string."""
    messages = example.get("messages", [])
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


# ── 6. Main Orchestrator: train_lora ─────────────────────────────────────────

def train_lora(config: FineTuneConfig) -> Dict[str, Any]:
    """Execute end-to-end LoRA/QLoRA fine-tuning for Qwen 2.5 Coder.

    Args:
        config: FineTuneConfig specification.

    Returns:
        Dictionary summarizing execution status, paths, parameters, and final metrics.
    """
    config.validate()
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save training config snapshot
    with open(output_path / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    logger.info("Loading datasets from %s...", config.dataset_path)
    train_dataset, eval_dataset = load_dataset_from_file(
        file_path=config.dataset_path,
        val_split=config.val_split,
        seed=config.seed,
        val_dataset_path=config.val_dataset_path,
        dataset_format=config.dataset_format,
        skip_malformed=config.skip_malformed,
        max_samples=config.max_samples,
    )

    logger.info("Initializing tokenizer...")
    tokenizer = setup_tokenizer(config)

    logger.info("Initializing base model (quantization=%s)...", config.quantization)
    model = setup_model(config, tokenizer)

    logger.info("Configuring PEFT LoRA adapters...")
    peft_config = setup_peft_config(config)
    model = get_lora_model(model, peft_config)

    if hasattr(model, "get_nb_trainable_parameters"):
        trainable_params, all_params = model.get_nb_trainable_parameters()
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())

    trainable_percent = (100.0 * trainable_params / all_params) if all_params > 0 else 0.0
    logger.info(
        "Trainable parameters: %s / %s (%.4f%%)",
        f"{trainable_params:,}", f"{all_params:,}", trainable_percent
    )

    data_collator = get_completion_data_collator(tokenizer)
    metrics_callback = TrainingMetricsCallback(output_path)

    has_eval = eval_dataset is not None and len(eval_dataset) > 0
    eval_strategy = "steps" if has_eval else "no"

    is_cuda = torch.cuda.is_available() and not config.use_cpu
    use_bf16 = config.bf16 and is_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = config.fp16 and is_cuda and not torch.cuda.is_bf16_supported()
    optim_choice = config.optim if is_cuda else "adamw_torch"

    training_args = TrainingArguments(
        output_dir=str(output_path),
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        gradient_checkpointing=config.gradient_checkpointing and is_cuda,
        optim=optim_choice,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=config.logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=config.eval_steps if has_eval else None,
        save_strategy="steps" if config.save_steps > 0 else "no",
        save_steps=config.save_steps if config.save_steps > 0 else None,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=has_eval and config.save_steps > 0,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False,
        report_to=config.report_to,
        seed=config.seed,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if has_eval else None,
        peft_config=peft_config,
        max_seq_length=config.max_seq_length,
        dataset_text_field=None,
        formatting_func=lambda ex: formatting_prompts_func(ex, tokenizer),
        data_collator=data_collator,
        callbacks=[metrics_callback],
    )

    logger.info("Beginning training execution...")
    train_result = trainer.train()

    eval_result = None
    if has_eval:
        logger.info("Running final evaluation pass...")
        eval_result = trainer.evaluate()

    logger.info("Saving final adapter and tokenizer bundle to %s...", output_path)
    trainer.model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Save chat template explicitly as chat_template.json
    try:
        chat_tmpl = getattr(tokenizer, "chat_template", DEFAULT_CHATML_TEMPLATE)
        with open(output_path / "chat_template.json", "w", encoding="utf-8") as f:
            json.dump({"chat_template": chat_tmpl}, f, indent=2)
    except Exception as e:
        logger.warning("Failed to write chat_template.json: %s", e)

    train_loss = getattr(train_result, "training_loss", None)
    eval_loss = eval_result.get("eval_loss") if eval_result else None
    eval_ppl = calculate_perplexity(eval_loss) if eval_loss is not None else None
    train_runtime = train_result.metrics.get("train_runtime") if hasattr(train_result, "metrics") else None

    return {
        "status": "success",
        "output_dir": str(output_path),
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "eval_perplexity": eval_ppl,
        "train_runtime": train_runtime,
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percent": trainable_percent,
        "metrics_summary": metrics_callback.summary,
    }
