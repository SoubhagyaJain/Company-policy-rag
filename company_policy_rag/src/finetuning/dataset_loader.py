"""Dataset Ingestion, Hygiene, Normalization, and Splitting for Qwen 2.5 Coder Fine-Tuning."""

from __future__ import annotations

import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import Dataset

logger = logging.getLogger(__name__)

# Standard role mapping dictionary for ShareGPT / ChatML normalization
ROLE_MAPPINGS: Dict[str, str] = {
    "human": "user",
    "user": "user",
    "user_msg": "user",
    "client": "user",
    "customer": "user",
    "prompter": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "chatgpt": "assistant",
    "system_response": "assistant",
    "system": "system",
    "sys": "system",
    "instruction": "system",
    "system_prompt": "system",
}

VALID_ROLES = {"system", "user", "assistant"}


class DatasetLoaderError(Exception):
    """Base exception for all dataset loader errors."""
    pass


class DatasetFormatValidationError(DatasetLoaderError):
    """Raised when format detection fails or unexpected keys/schemas are encountered."""
    pass


class DatasetValidationError(DatasetLoaderError):
    """Raised when record contents fail schema or turn validation."""
    pass


class DatasetEmptyError(DatasetLoaderError):
    """Raised when dataset is empty or all records are filtered out."""
    pass


def detect_format(sample_records: List[Dict[str, Any]]) -> str:
    """Detect dataset format by scoring top-level dictionary keys across sample records.

    Supported formats:
        - 'sharegpt': records with 'conversations' or 'messages' lists.
        - 'alpaca': records with 'instruction' and ('output' or 'response').
        - 'prompt_response': records with 'prompt'/'response', 'query'/'answer', etc.

    Args:
        sample_records: A list of sample dictionary records.

    Returns:
        Best matching format string ('alpaca', 'sharegpt', 'prompt_response').

    Raises:
        DatasetEmptyError: If sample_records is empty.
        DatasetFormatValidationError: If no supported format matches.
    """
    if not sample_records:
        raise DatasetEmptyError("Cannot detect format: sample record buffer is empty.")

    votes = {"sharegpt": 0, "alpaca": 0, "prompt_response": 0}

    for rec in sample_records:
        if not isinstance(rec, dict):
            continue

        # 1. ShareGPT check
        if "conversations" in rec and isinstance(rec["conversations"], list):
            votes["sharegpt"] += 1
            continue
        if "messages" in rec and isinstance(rec["messages"], list) and len(rec["messages"]) > 0:
            first_msg = rec["messages"][0]
            if isinstance(first_msg, dict) and any(
                k in first_msg for k in ("from", "value", "role", "content", "speaker", "text")
            ):
                votes["sharegpt"] += 1
                continue

        # 2. Alpaca check
        if "instruction" in rec and ("output" in rec or "response" in rec):
            votes["alpaca"] += 1
            continue

        # 3. Prompt-Response check
        if ("prompt" in rec and "response" in rec) or \
           ("query" in rec and "answer" in rec) or \
           ("question" in rec and ("response" in rec or "answer" in rec)) or \
           ("input" in rec and "output" in rec and "instruction" not in rec):
            votes["prompt_response"] += 1
            continue

    best_format, max_votes = max(votes.items(), key=lambda x: x[1])

    if max_votes == 0:
        sample_keys = [list(r.keys()) for r in sample_records[:3] if isinstance(r, dict)]
        raise DatasetFormatValidationError(
            f"Unable to auto-detect dataset format from sample records. "
            f"Sample keys found: {sample_keys}. "
            f"Supported formats: 'alpaca', 'sharegpt', 'prompt_response'. "
            f"Please specify --dataset_format explicitly."
        )

    return best_format


def sanitize_messages(
    messages: List[Dict[str, str]],
    default_system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Sanitize conversational turns:
    1. Trim whitespace, drop turns with empty content.
    2. Normalize roles to 'system', 'user', 'assistant'.
    3. Ensure a single consolidated system message is at index 0.
    4. Merge consecutive turns with identical roles.
    5. Enforce trailing assistant turn (prune trailing non-assistant turns).
    6. Verify at least one user and one assistant turn exist.

    Args:
        messages: List of message dictionaries with 'role' and 'content'.
        default_system_prompt: Optional system prompt to inject if no system turn exists.

    Returns:
        Sanitized list of message dicts, or [] if invalid.
    """
    if not messages:
        return []

    # Step 1 & 2: Clean content and map roles
    cleaned: List[Dict[str, str]] = []
    system_parts: List[str] = []

    for turn in messages:
        if not isinstance(turn, dict):
            continue
        raw_role = str(turn.get("role", "")).strip().lower()
        role = ROLE_MAPPINGS.get(raw_role, raw_role)
        if role not in VALID_ROLES:
            continue

        content = str(turn.get("content", "")).strip()
        if not content:
            continue

        if role == "system":
            system_parts.append(content)
        else:
            cleaned.append({"role": role, "content": content})

    # Step 3: Handle system message
    system_content: Optional[str] = None
    if system_parts:
        system_content = "\n\n".join(system_parts)
    elif default_system_prompt and default_system_prompt.strip():
        system_content = default_system_prompt.strip()

    # Step 4: Merge consecutive turns with identical roles
    merged: List[Dict[str, str]] = []
    for turn in cleaned:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1]["content"] = merged[-1]["content"] + "\n\n" + turn["content"]
        else:
            merged.append({"role": turn["role"], "content": turn["content"]})

    # Step 5: Enforce trailing assistant turn
    while merged and merged[-1]["role"] != "assistant":
        merged.pop()

    # Step 6: Verify viability: must have at least 1 user turn and 1 assistant turn
    has_user = any(t["role"] == "user" for t in merged)
    has_assistant = any(t["role"] == "assistant" for t in merged)
    if not has_user or not has_assistant:
        return []

    # Assemble final list with system prompt at index 0
    final_messages: List[Dict[str, str]] = []
    if system_content:
        final_messages.append({"role": "system", "content": system_content})
    final_messages.extend(merged)

    return final_messages


def normalize_record(
    record: Dict[str, Any],
    format_name: Optional[str] = None,
    default_system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Normalize a raw record into standard ChatML messages.

    Args:
        record: Raw data dictionary.
        format_name: Format identifier ('alpaca', 'sharegpt', 'prompt_response', or None/'auto').
        default_system_prompt: Default system prompt to prepend if none present.

    Returns:
        List of normalized messages: [{'role': ..., 'content': ...}].
    """
    if not isinstance(record, dict):
        return []

    resolved_format = format_name
    if not resolved_format or resolved_format == "auto":
        resolved_format = detect_format([record])

    if resolved_format == "alpaca":
        instruction = str(record.get("instruction") or "").strip()
        raw_input = record.get("input")
        input_text = str(raw_input).strip() if raw_input is not None else ""
        raw_output = record.get("output") if record.get("output") is not None else record.get("response")
        output_text = str(raw_output).strip() if raw_output is not None else ""

        if not instruction or not output_text:
            return []

        user_content = instruction + (f"\n\nContext:\n{input_text}" if input_text else "")
        raw_msgs = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output_text},
        ]
        return sanitize_messages(raw_msgs, default_system_prompt=default_system_prompt)

    elif resolved_format == "sharegpt":
        turns = record.get("conversations") or record.get("messages") or []
        if not isinstance(turns, list):
            return []

        raw_msgs = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role_raw = turn.get("role") or turn.get("from") or turn.get("speaker") or ""
            role = ROLE_MAPPINGS.get(str(role_raw).strip().lower(), "user")
            content = turn.get("content") or turn.get("value") or turn.get("text") or ""
            raw_msgs.append({"role": role, "content": str(content)})

        return sanitize_messages(raw_msgs, default_system_prompt=default_system_prompt)

    elif resolved_format == "prompt_response":
        prompt = str(
            record.get("prompt")
            or record.get("query")
            or record.get("question")
            or record.get("instruction")
            or record.get("input")
            or ""
        ).strip()
        response = str(
            record.get("response")
            or record.get("answer")
            or record.get("output")
            or record.get("completion")
            or ""
        ).strip()

        if not prompt or not response:
            return []

        raw_msgs = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        return sanitize_messages(raw_msgs, default_system_prompt=default_system_prompt)

    else:
        raise DatasetFormatValidationError(f"Unsupported dataset format: '{resolved_format}'")


def split_dataset(
    dataset: Dataset,
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple[Dataset, Optional[Dataset]]:
    """Deterministically split a Hugging Face Dataset into train and validation sets.

    Handles small dataset edge cases:
        - N = 0: raises DatasetEmptyError.
        - N = 1: returns (dataset, None) with warning.
        - 2 <= N < 10: guarantees at least 1 train and 1 val sample when val_split > 0.

    Args:
        dataset: Hugging Face Dataset to split.
        val_split: Fraction of samples to allocate to validation.
        seed: Random seed for deterministic shuffling.

    Returns:
        Tuple of (train_dataset, val_dataset). val_dataset is None if val_split <= 0 or N == 1.
    """
    n_samples = len(dataset)
    if n_samples == 0:
        raise DatasetEmptyError("Dataset is empty; cannot split.")

    if val_split <= 0.0 or val_split is None:
        return dataset, None

    if n_samples == 1:
        logger.warning("Dataset contains only 1 sample; returning as train with val=None.")
        return dataset, None

    indices = list(range(n_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_count = max(1, min(int(round(n_samples * val_split)), n_samples - 1))
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)

    return train_dataset, val_dataset


def load_dataset_from_file(
    file_path: Union[str, Path],
    val_split: float = 0.1,
    seed: int = 42,
    val_dataset_path: Optional[Union[str, Path]] = None,
    dataset_format: Optional[str] = "auto",
    default_system_prompt: Optional[str] = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    skip_malformed: bool = True,
    max_samples: Optional[int] = None,
) -> Tuple[Dataset, Optional[Dataset]]:
    """Load, auto-detect, normalize, sanitize, and split dataset from file.

    Args:
        file_path: Path to training data (.json or .jsonl).
        val_split: Validation split fraction.
        seed: Random seed for splitting.
        val_dataset_path: Optional separate validation file path.
        dataset_format: Dataset format identifier ('auto', 'alpaca', 'sharegpt', 'prompt_response').
        default_system_prompt: Default system prompt for normalization.
        skip_malformed: If True, skips invalid JSON lines; if False, raises DatasetValidationError.
        max_samples: Optional maximum number of samples to load.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    train_raw_records = _read_raw_file(file_path, skip_malformed=skip_malformed, max_samples=max_samples)

    if not train_raw_records:
        raise DatasetEmptyError(f"No valid records found in {file_path}")

    # Determine format
    resolved_format = dataset_format
    if not resolved_format or resolved_format == "auto":
        resolved_format = detect_format(train_raw_records[:50])

    logger.info("Using dataset format: %s for %s", resolved_format, file_path)

    # Normalize train records
    train_normalized = []
    for rec in train_raw_records:
        msgs = normalize_record(rec, format_name=resolved_format, default_system_prompt=default_system_prompt)
        if msgs:
            train_normalized.append({"messages": msgs})

    if not train_normalized:
        raise DatasetEmptyError(f"All records in {file_path} failed validation or sanitization.")

    train_ds = Dataset.from_list(train_normalized)

    # If separate validation path provided
    if val_dataset_path:
        val_raw_records = _read_raw_file(val_dataset_path, skip_malformed=skip_malformed, max_samples=max_samples)
        if not val_raw_records:
            logger.warning("Separate validation file %s was empty. Returning val=None.", val_dataset_path)
            return train_ds, None

        val_normalized = []
        for rec in val_raw_records:
            msgs = normalize_record(rec, format_name=resolved_format, default_system_prompt=default_system_prompt)
            if msgs:
                val_normalized.append({"messages": msgs})

        val_ds = Dataset.from_list(val_normalized) if val_normalized else None
        return train_ds, val_ds

    # Otherwise, deterministic split
    return split_dataset(train_ds, val_split=val_split, seed=seed)


def _read_raw_file(
    file_path: Union[str, Path],
    skip_malformed: bool = True,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Read raw records from JSON or JSONL file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    records: List[Dict[str, Any]] = []

    # Read content
    with open(path, "r", encoding="utf-8") as f:
        # Check if entire file is a JSON array
        first_char = ""
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_char = ch
                break
        f.seek(0)

        if first_char == "[":
            try:
                data = json.load(f)
                if isinstance(data, list):
                    records = [r for r in data if isinstance(r, dict)]
            except json.JSONDecodeError as err:
                if not skip_malformed:
                    raise DatasetValidationError(f"Invalid JSON array in {path}: {err}") from err
                logger.warning("Failed to parse JSON array in %s: %s", path, err)
        else:
            # Line by line JSONL
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                        if max_samples is not None and len(records) >= max_samples:
                            break
                except json.JSONDecodeError as err:
                    if not skip_malformed:
                        raise DatasetValidationError(f"Malformed JSON on line {line_idx} in {path}: {err}") from err
                    logger.warning("Skipping malformed JSON line %d in %s: %s", line_idx, path, err)

    if max_samples is not None and len(records) > max_samples:
        records = records[:max_samples]

    return records


def compute_dataset_statistics(
    dataset: Dataset,
    tokenizer: Optional[Any] = None,
    max_seq_length: int = 2048,
) -> Dict[str, Any]:
    """Compute comprehensive token length distribution and turn statistics."""
    num_samples = len(dataset)
    if num_samples == 0:
        return {
            "num_samples": 0,
            "total_turns": 0,
            "mean_turns_per_sample": 0.0,
            "role_counts": {"system": 0, "user": 0, "assistant": 0},
            "length_stats": {"min": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0},
            "truncated_samples_count": 0,
            "truncated_samples_percent": 0.0,
        }

    total_turns = 0
    role_counts = {"system": 0, "user": 0, "assistant": 0}
    lengths: List[int] = []
    truncated_count = 0

    for sample in dataset:
        messages = sample.get("messages", [])
        total_turns += len(messages)
        for msg in messages:
            role = msg.get("role", "")
            if role in role_counts:
                role_counts[role] += 1

        if tokenizer is not None:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            tok_len = len(tokenizer.encode(text))
        else:
            # Approximate token count by character length / 4
            text = " ".join(m.get("content", "") for m in messages)
            tok_len = max(1, len(text) // 4)

        lengths.append(tok_len)
        if tok_len > max_seq_length:
            truncated_count = truncated_count + 1

    lengths.sort()
    n = len(lengths)

    def percentile(p: float) -> float:
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(lengths[int(k)])
        return float(lengths[f] * (c - k) + lengths[c] * (k - f))

    stats = {
        "min": lengths[0] if lengths else 0,
        "mean": sum(lengths) / n if n else 0.0,
        "median": percentile(0.5) if n else 0.0,
        "p90": percentile(0.90) if n else 0.0,
        "p95": percentile(0.95) if n else 0.0,
        "p99": percentile(0.99) if n else 0.0,
        "max": lengths[-1] if lengths else 0,
    }

    return {
        "num_samples": num_samples,
        "total_turns": total_turns,
        "mean_turns_per_sample": total_turns / num_samples if num_samples else 0.0,
        "role_counts": role_counts,
        "length_stats": stats,
        "truncated_samples_count": truncated_count,
        "truncated_samples_percent": (truncated_count / num_samples) * 100.0 if num_samples else 0.0,
    }


# Aliases for package compatibility
profile_dataset_tokens = compute_dataset_statistics
validate_dataset_records = detect_format
