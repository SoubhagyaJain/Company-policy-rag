"""Unit tests for dataset ingestion, format auto-detection, hygiene, normalization, and splitting."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import pytest
from datasets import Dataset

from company_policy_rag.src.finetuning.dataset_loader import (
    DatasetEmptyError,
    DatasetFormatValidationError,
    DatasetValidationError,
    compute_dataset_statistics,
    detect_format,
    load_dataset_from_file,
    normalize_record,
    profile_dataset_tokens,
    sanitize_messages,
    split_dataset,
)


# ── 1. Format Auto-Detection Tests ──────────────────────────────────────────

def test_detect_format_alpaca():
    records = [
        {"instruction": "Calculate sum", "input": "2 + 2", "output": "4"},
        {"instruction": "Explain quantum computing", "output": "Quantum computing uses qubits."},
    ]
    assert detect_format(records) == "alpaca"


def test_detect_format_sharegpt_conversations():
    records = [
        {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi there!"},
            ]
        }
    ]
    assert detect_format(records) == "sharegpt"


def test_detect_format_sharegpt_messages():
    records = [
        {
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
            ]
        }
    ]
    assert detect_format(records) == "sharegpt"


def test_detect_format_prompt_response():
    records1 = [{"prompt": "What is RAG?", "response": "Retrieval-Augmented Generation"}]
    records2 = [{"query": "What is BM25?", "answer": "A lexical ranking function"}]
    records3 = [{"question": "How to write code?", "response": "Use an IDE."}]
    
    assert detect_format(records1) == "prompt_response"
    assert detect_format(records2) == "prompt_response"
    assert detect_format(records3) == "prompt_response"


def test_detect_format_ambiguous_error():
    records = [{"unknown_key_1": "val1", "unknown_key_2": "val2"}]
    with pytest.raises(DatasetFormatValidationError):
        detect_format(records)


def test_detect_format_empty_error():
    with pytest.raises(DatasetEmptyError):
        detect_format([])


# ── 2. Alpaca Normalization Tests ────────────────────────────────────────────

def test_alpaca_with_input():
    record = {
        "instruction": "Summarize the text.",
        "input": "The quick brown fox jumps over the lazy dog.",
        "output": "A fox jumps over a dog.",
    }
    messages = normalize_record(record, format_name="alpaca")
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert "Summarize the text." in messages[0]["content"]
    assert "Context:\nThe quick brown fox" in messages[0]["content"]
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "A fox jumps over a dog."


def test_alpaca_without_input():
    record = {
        "instruction": "What is 10 + 20?",
        "input": "",
        "output": "30",
    }
    messages = normalize_record(record, format_name="alpaca")
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is 10 + 20?"
    assert "Context:" not in messages[0]["content"]
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "30"


def test_alpaca_custom_system_prompt():
    record = {
        "instruction": "List 3 colors.",
        "output": "Red, Blue, Green.",
    }
    messages = normalize_record(record, format_name="alpaca", default_system_prompt="You are a design bot.")
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a design bot."
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


def test_alpaca_missing_fields_returns_empty():
    record = {"instruction": "Hello"}  # missing output
    assert normalize_record(record, format_name="alpaca") == []


# ── 3. ShareGPT Normalization Tests ──────────────────────────────────────────

def test_sharegpt_from_value_mapping():
    record = {
        "conversations": [
            {"from": "system", "value": "System instruction."},
            {"from": "human", "value": "User query."},
            {"from": "gpt", "value": "Assistant answer."},
        ]
    }
    messages = normalize_record(record, format_name="sharegpt")
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "System instruction."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "User query."
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Assistant answer."


def test_sharegpt_speaker_text_mapping():
    record = {
        "conversations": [
            {"speaker": "user", "text": "Can you help me?"},
            {"speaker": "bot", "text": "Yes, I can!"},
        ]
    }
    messages = normalize_record(record, format_name="sharegpt")
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_sharegpt_multi_turn():
    record = {
        "conversations": [
            {"from": "human", "value": "Turn 1 Q"},
            {"from": "gpt", "value": "Turn 1 A"},
            {"from": "human", "value": "Turn 2 Q"},
            {"from": "gpt", "value": "Turn 2 A"},
        ]
    }
    messages = normalize_record(record, format_name="sharegpt")
    assert len(messages) == 4
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]


def test_sharegpt_consecutive_merge():
    record = {
        "conversations": [
            {"from": "human", "value": "Part 1 of query."},
            {"from": "human", "value": "Part 2 of query."},
            {"from": "gpt", "value": "Response."},
        ]
    }
    messages = normalize_record(record, format_name="sharegpt")
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Part 1 of query.\n\nPart 2 of query."
    assert messages[1]["role"] == "assistant"


def test_sharegpt_trailing_user_pruning():
    record = {
        "conversations": [
            {"from": "human", "value": "Query 1"},
            {"from": "gpt", "value": "Answer 1"},
            {"from": "human", "value": "Dangling user question without answer"},
        ]
    }
    messages = normalize_record(record, format_name="sharegpt")
    assert len(messages) == 2
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == "Answer 1"


def test_sharegpt_system_prompt_placement():
    record = {
        "conversations": [
            {"from": "human", "value": "Hello"},
            {"from": "system", "value": "Mid-dialogue system note"},
            {"from": "gpt", "value": "Hi"},
        ]
    }
    messages = normalize_record(record, format_name="sharegpt")
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Mid-dialogue system note"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


# ── 4. Prompt-Response Normalization Tests ───────────────────────────────────

def test_prompt_response_standard():
    record = {"prompt": "What is cosine similarity?", "response": "A measure of vector similarity."}
    messages = normalize_record(record, format_name="prompt_response")
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is cosine similarity?"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "A measure of vector similarity."


def test_prompt_response_query_answer():
    record = {"query": "What is ChromaDB?", "answer": "An open source embedding database."}
    messages = normalize_record(record, format_name="prompt_response")
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


# ── 5. Sanitization & Hygiene Tests ──────────────────────────────────────────

def test_whitespace_trimming():
    raw_msgs = [
        {"role": "user", "content": "   Trimmed query   \n"},
        {"role": "assistant", "content": "  Trimmed response  "},
    ]
    sanitized = sanitize_messages(raw_msgs)
    assert sanitized[0]["content"] == "Trimmed query"
    assert sanitized[1]["content"] == "Trimmed response"


def test_empty_turn_removal():
    raw_msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "   "},
        {"role": "assistant", "content": "Actual response"},
    ]
    sanitized = sanitize_messages(raw_msgs)
    assert len(sanitized) == 2
    assert sanitized[1]["content"] == "Actual response"


def test_invalid_conversation_discard():
    raw_msgs = [{"role": "user", "content": "Only user message without assistant"}]
    assert sanitize_messages(raw_msgs) == []


def test_skip_malformed_jsonl_true(tmp_path: Path):
    file_path = tmp_path / "corrupt.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write('{"instruction": "Valid 1", "output": "Ans 1"}\n')
        f.write('CORRUPTED NOT JSON LINE\n')
        f.write('{"instruction": "Valid 2", "output": "Ans 2"}\n')

    train_ds, val_ds = load_dataset_from_file(file_path, val_split=0.0, skip_malformed=True)
    assert len(train_ds) == 2


def test_skip_malformed_jsonl_false(tmp_path: Path):
    file_path = tmp_path / "corrupt.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write('{"instruction": "Valid 1", "output": "Ans 1"}\n')
        f.write('CORRUPTED NOT JSON LINE\n')

    with pytest.raises(DatasetValidationError):
        load_dataset_from_file(file_path, val_split=0.0, skip_malformed=False)


# ── 6. Splitting & Determinism Tests ─────────────────────────────────────────

def test_split_determinism_same_seed():
    raw_data = [{"messages": [{"role": "user", "content": f"Q{i}"}, {"role": "assistant", "content": f"A{i}"}]} for i in range(20)]
    ds = Dataset.from_list(raw_data)

    train1, val1 = split_dataset(ds, val_split=0.2, seed=42)
    train2, val2 = split_dataset(ds, val_split=0.2, seed=42)

    assert [x["messages"][0]["content"] for x in train1] == [x["messages"][0]["content"] for x in train2]
    assert [x["messages"][0]["content"] for x in val1] == [x["messages"][0]["content"] for x in val2]


def test_split_different_seeds():
    raw_data = [{"messages": [{"role": "user", "content": f"Q{i}"}, {"role": "assistant", "content": f"A{i}"}]} for i in range(20)]
    ds = Dataset.from_list(raw_data)

    train1, val1 = split_dataset(ds, val_split=0.2, seed=42)
    train2, val2 = split_dataset(ds, val_split=0.2, seed=999)

    assert [x["messages"][0]["content"] for x in train1] != [x["messages"][0]["content"] for x in train2]


def test_split_val_split_zero():
    raw_data = [{"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}]
    ds = Dataset.from_list(raw_data)
    train, val = split_dataset(ds, val_split=0.0)
    assert len(train) == 1
    assert val is None


def test_split_small_dataset_n1():
    raw_data = [{"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}]
    ds = Dataset.from_list(raw_data)
    train, val = split_dataset(ds, val_split=0.2)
    assert len(train) == 1
    assert val is None


def test_split_small_dataset_n2_n5():
    raw_data = [{"messages": [{"role": "user", "content": f"Q{i}"}, {"role": "assistant", "content": f"A{i}"}]} for i in range(5)]
    ds = Dataset.from_list(raw_data)
    train, val = split_dataset(ds, val_split=0.2, seed=42)
    assert len(train) >= 1
    assert len(val) >= 1
    assert len(train) + len(val) == 5


def test_split_separate_val_file(tmp_path: Path):
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        f.write('{"instruction": "Train Q", "output": "Train A"}\n')
    with open(val_file, "w", encoding="utf-8") as f:
        f.write('{"instruction": "Val Q", "output": "Val A"}\n')

    train_ds, val_ds = load_dataset_from_file(train_file, val_dataset_path=val_file)
    assert len(train_ds) == 1
    assert len(val_ds) == 1
    assert "Train Q" in train_ds[0]["messages"][0]["content"]
    assert "Val Q" in val_ds[0]["messages"][0]["content"]


# ── 7. Statistics & Profiling Tests ──────────────────────────────────────────

def test_compute_dataset_statistics():
    raw_data = [
        {
            "messages": [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "Short query"},
                {"role": "assistant", "content": "Short answer"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "A much longer user query with more tokens and information"},
                {"role": "assistant", "content": "Detailed long response explanation"},
            ]
        },
    ]
    ds = Dataset.from_list(raw_data)
    stats = compute_dataset_statistics(ds, max_seq_length=50)

    assert stats["num_samples"] == 2
    assert stats["total_turns"] == 5
    assert stats["role_counts"]["system"] == 1
    assert stats["role_counts"]["user"] == 2
    assert stats["role_counts"]["assistant"] == 2
    assert "mean" in stats["length_stats"]
    assert "median" in stats["length_stats"]


# ── 8. Packaged Sample Files Loading Test ────────────────────────────────────

def test_load_packaged_sample_files():
    project_root = Path(__file__).resolve().parent.parent.parent
    sample_dir = project_root / "data" / "sample_finetune"

    alpaca_file = sample_dir / "alpaca_sample.jsonl"
    sharegpt_file = sample_dir / "sharegpt_sample.jsonl"
    prompt_resp_file = sample_dir / "prompt_response_sample.jsonl"

    assert alpaca_file.is_file(), f"Missing {alpaca_file}"
    assert sharegpt_file.is_file(), f"Missing {sharegpt_file}"
    assert prompt_resp_file.is_file(), f"Missing {prompt_resp_file}"

    # Load Alpaca
    train_alpaca, _ = load_dataset_from_file(alpaca_file, val_split=0.0)
    assert len(train_alpaca) >= 10

    # Load ShareGPT
    train_sharegpt, _ = load_dataset_from_file(sharegpt_file, val_split=0.0)
    assert len(train_sharegpt) >= 6

    # Load Prompt-Response
    train_pr, _ = load_dataset_from_file(prompt_resp_file, val_split=0.0)
    assert len(train_pr) >= 8
