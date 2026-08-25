"""Adversarial Empirical Stress Test Suite for Dataset Ingestion, Normalization, Hygiene, and Splitting.

This test harness executes edge-case mining, boundary condition probing, format auto-detection stress tests,
role-merging verification, deterministic seed repeatability, and failure mode analysis on `dataset_loader.py`.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest
from datasets import Dataset

from company_policy_rag.src.finetuning.dataset_loader import (
    DatasetEmptyError,
    DatasetFormatValidationError,
    DatasetLoaderError,
    DatasetValidationError,
    compute_dataset_statistics,
    detect_format,
    load_dataset_from_file,
    normalize_record,
    sanitize_messages,
    split_dataset,
)


# ==============================================================================
# 1. FILE SYSTEM & CORRUPTION ADVERSARIAL STRESS TESTS
# ==============================================================================

class TestFileCorruptionAndEmptyHandling:
    """Stress-test file reading against corrupt, empty, and deceptive files."""

    def test_completely_empty_file_raises_dataset_empty_error(self, tmp_path: Path):
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("", encoding="utf-8")

        with pytest.raises(DatasetEmptyError, match="No valid records found"):
            load_dataset_from_file(empty_file)

    def test_whitespace_and_blank_lines_only_file(self, tmp_path: Path):
        blank_file = tmp_path / "blank.jsonl"
        blank_file.write_text("   \n\n\t  \r\n   \n", encoding="utf-8")

        with pytest.raises(DatasetEmptyError, match="No valid records found"):
            load_dataset_from_file(blank_file)

    def test_non_existent_file_raises_file_not_found(self, tmp_path: Path):
        missing_file = tmp_path / "does_not_exist.jsonl"
        with pytest.raises(FileNotFoundError):
            load_dataset_from_file(missing_file)

    def test_corrupt_json_lines_with_skip_malformed_true(self, tmp_path: Path):
        corrupt_file = tmp_path / "mixed_corrupt.jsonl"
        content = (
            '{"instruction": "Q1", "output": "A1"}\n'
            '{"broken_json": true, \n'
            'NOT EVEN JSON LINE\n'
            '{"instruction": "Q2", "output": "A2"}\n'
            '{truncated_json\n'
            '{"instruction": "Q3", "output": "A3"}\n'
        )
        corrupt_file.write_text(content, encoding="utf-8")

        train_ds, val_ds = load_dataset_from_file(corrupt_file, val_split=0.0, skip_malformed=True)
        assert len(train_ds) == 3
        user_queries = [s["messages"][1]["content"] if s["messages"][0]["role"] == "system" else s["messages"][0]["content"] for s in train_ds]
        assert "Q1" in user_queries[0]
        assert "Q2" in user_queries[1]
        assert "Q3" in user_queries[2]

    def test_corrupt_json_lines_with_skip_malformed_false(self, tmp_path: Path):
        corrupt_file = tmp_path / "mixed_corrupt.jsonl"
        content = (
            '{"instruction": "Q1", "output": "A1"}\n'
            '{"broken_json": true, \n'
            '{"instruction": "Q2", "output": "A2"}\n'
        )
        corrupt_file.write_text(content, encoding="utf-8")

        with pytest.raises(DatasetValidationError, match="Malformed JSON on line 2"):
            load_dataset_from_file(corrupt_file, val_split=0.0, skip_malformed=False)

    def test_json_array_empty_list(self, tmp_path: Path):
        json_array_file = tmp_path / "empty_array.json"
        json_array_file.write_text("[]", encoding="utf-8")

        with pytest.raises(DatasetEmptyError, match="No valid records found"):
            load_dataset_from_file(json_array_file)

    def test_json_array_with_non_dict_elements(self, tmp_path: Path):
        mixed_array_file = tmp_path / "mixed_array.json"
        content = json.dumps([
            123,
            "just a string",
            None,
            [],
            {"instruction": "Valid array item", "output": "Valid array answer"},
            False,
        ])
        mixed_array_file.write_text(content, encoding="utf-8")

        train_ds, val_ds = load_dataset_from_file(mixed_array_file, val_split=0.0)
        assert len(train_ds) == 1
        assert "Valid array item" in str(train_ds[0]["messages"])

    def test_json_array_corrupted_syntax_with_skip_malformed_true_vs_false(self, tmp_path: Path):
        broken_array_file = tmp_path / "broken_array.json"
        broken_array_file.write_text('[{"instruction": "Q1", "output": "A1"}, ', encoding="utf-8")

        # skip_malformed=False should raise DatasetValidationError
        with pytest.raises(DatasetValidationError, match="Invalid JSON array"):
            load_dataset_from_file(broken_array_file, skip_malformed=False)

        # skip_malformed=True cannot salvage a broken JSON array file and raises DatasetEmptyError
        with pytest.raises(DatasetEmptyError):
            load_dataset_from_file(broken_array_file, skip_malformed=True)

    def test_utf8_bom_handling_failure_mode(self, tmp_path: Path):
        """Empirically document that UTF-8 BOM in JSON array triggers line parser fallback and drops records."""
        bom_json = tmp_path / "bom_dataset.json"
        # Write bytes with UTF-8 BOM prefix \xef\xbb\xbf
        bom_json.write_bytes(b"\xef\xbb\xbf" + json.dumps([{"instruction": "Q_BOM", "output": "A_BOM"}]).encode("utf-8"))

        # Under skip_malformed=True, JSONDecodeError on BOM line causes DatasetEmptyError
        with pytest.raises(DatasetEmptyError):
            load_dataset_from_file(bom_json, val_split=0.0, skip_malformed=True)


# ==============================================================================
# 2. SCHEMA ABNORMALITY & WEAKNESS TESTS
# ==============================================================================

class TestSchemaAbnormalities:
    """Stress-test normalization across pathological and atypical schemas."""

    def test_alpaca_null_fields(self):
        # instruction is None -> should return empty
        record1 = {"instruction": None, "input": "abc", "output": "def"}
        assert normalize_record(record1, format_name="alpaca") == []

        # output is None -> should return empty
        record2 = {"instruction": "abc", "input": "def", "output": None}
        assert normalize_record(record2, format_name="alpaca") == []

        # input is None -> should treat input as empty and keep instruction
        record3 = {"instruction": "Valid instruction", "input": None, "output": "Valid answer"}
        msgs = normalize_record(record3, format_name="alpaca", default_system_prompt=None)
        assert len(msgs) == 2
        assert msgs[0]["content"] == "Valid instruction"
        assert "Context:" not in msgs[0]["content"]
        assert msgs[1]["content"] == "Valid answer"

    def test_non_string_primitive_types_are_coerced(self):
        record = {
            "instruction": 12345,
            "input": 67890,
            "output": True,
        }
        msgs = normalize_record(record, format_name="alpaca", default_system_prompt=None)
        assert len(msgs) == 2
        assert "12345" in msgs[0]["content"]
        assert "67890" in msgs[0]["content"]
        assert msgs[1]["content"] == "True"

    def test_extra_unknown_keys_are_ignored(self):
        record = {
            "instruction": "Explain LoRA",
            "output": "Low-Rank Adaptation",
            "extra_metadata": {"author": "Tester", "score": 9.9},
            "random_flag": True,
            "unrelated_id": 9999,
        }
        msgs = normalize_record(record, format_name="alpaca", default_system_prompt=None)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Explain LoRA"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "Low-Rank Adaptation"

    def test_sharegpt_non_dict_turns_skipped(self):
        record = {
            "conversations": [
                "not a dict turn",
                None,
                123,
                {"role": "user", "content": "Valid user question"},
                {"role": "assistant", "content": "Valid assistant response"},
            ]
        }
        msgs = normalize_record(record, format_name="sharegpt", default_system_prompt=None)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Valid user question"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "Valid assistant response"

    def test_sharegpt_unknown_role_fallback_behavior(self):
        """Empirically test how unknown roles in ShareGPT format are handled.
        
        Currently, normalize_record uses ROLE_MAPPINGS.get(role_raw, 'user'),
        which coerces unmapped roles to 'user'.
        """
        record = {
            "conversations": [
                {"role": "user", "content": "Initial prompt"},
                {"role": "unknown_tool_output", "content": "Data payload"},
                {"role": "assistant", "content": "Final answer"},
            ]
        }
        msgs = normalize_record(record, format_name="sharegpt", default_system_prompt=None)
        assert len(msgs) == 2
        # Consecutive user turns get merged:
        assert msgs[0]["role"] == "user"
        assert "Initial prompt\n\nData payload" in msgs[0]["content"]
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "Final answer"

    def test_prompt_response_alternative_key_pairs(self):
        pairs = [
            ({"prompt": "P1", "response": "R1"}, "P1", "R1"),
            ({"query": "Q2", "answer": "A2"}, "Q2", "A2"),
            ({"question": "Q3", "response": "R3"}, "Q3", "R3"),
            ({"question": "Q4", "answer": "A4"}, "Q4", "A4"),
            ({"instruction": "I5", "response": "R5"}, "I5", "R5"),
            ({"input": "In6", "output": "Out6"}, "In6", "Out6"),
        ]
        for rec, expected_p, expected_r in pairs:
            msgs = normalize_record(rec, format_name="prompt_response", default_system_prompt=None)
            assert len(msgs) == 2
            assert msgs[0]["role"] == "user"
            assert msgs[0]["content"] == expected_p
            assert msgs[1]["role"] == "assistant"
            assert msgs[1]["content"] == expected_r


# ==============================================================================
# 3. CONVERSATIONAL ROLE SEQUENCING & SANITIZATION STRESS TESTS
# ==============================================================================

class TestRoleSequencingAndHygiene:
    """Stress-test role transitions, merging, consolidation, and pruning."""

    def test_consecutive_user_turns_merged_with_double_newline(self):
        raw_msgs = [
            {"role": "user", "content": "Question part 1"},
            {"role": "user", "content": "Question part 2"},
            {"role": "user", "content": "Question part 3"},
            {"role": "assistant", "content": "Consolidated answer"},
        ]
        sanitized = sanitize_messages(raw_msgs)
        assert len(sanitized) == 2
        assert sanitized[0]["role"] == "user"
        assert sanitized[0]["content"] == "Question part 1\n\nQuestion part 2\n\nQuestion part 3"
        assert sanitized[1]["role"] == "assistant"
        assert sanitized[1]["content"] == "Consolidated answer"

    def test_consecutive_assistant_turns_merged_with_double_newline(self):
        raw_msgs = [
            {"role": "user", "content": "Tell me a story"},
            {"role": "assistant", "content": "Once upon a time."},
            {"role": "assistant", "content": "There lived a coder."},
            {"role": "assistant", "content": "The end."},
        ]
        sanitized = sanitize_messages(raw_msgs)
        assert len(sanitized) == 2
        assert sanitized[0]["role"] == "user"
        assert sanitized[1]["role"] == "assistant"
        assert sanitized[1]["content"] == "Once upon a time.\n\nThere lived a coder.\n\nThe end."

    def test_multiple_interleaved_system_prompts_consolidated_at_index_zero(self):
        raw_msgs = [
            {"role": "system", "content": "System Rule 1: Be concise."},
            {"role": "user", "content": "Hello!"},
            {"role": "system", "content": "System Rule 2: Use Markdown."},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "system", "content": "System Rule 3: Do not hallucinate."},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I am functioning normally."},
        ]
        sanitized = sanitize_messages(raw_msgs)
        assert sanitized[0]["role"] == "system"
        assert sanitized[0]["content"] == "System Rule 1: Be concise.\n\nSystem Rule 2: Use Markdown.\n\nSystem Rule 3: Do not hallucinate."
        assert [m["role"] for m in sanitized[1:]] == ["user", "assistant", "user", "assistant"]

    def test_trailing_non_assistant_turns_are_pruned(self):
        raw_msgs = [
            {"role": "user", "content": "Turn 1 Q"},
            {"role": "assistant", "content": "Turn 1 A"},
            {"role": "user", "content": "Turn 2 Q (unanswered)"},
        ]
        sanitized = sanitize_messages(raw_msgs)
        assert len(sanitized) == 2
        assert sanitized[-1]["role"] == "assistant"
        assert sanitized[-1]["content"] == "Turn 1 A"

    def test_trailing_multiple_non_assistant_turns_pruned(self):
        raw_msgs = [
            {"role": "user", "content": "Valid Q"},
            {"role": "assistant", "content": "Valid A"},
            {"role": "user", "content": "Dangling Q1"},
            {"role": "user", "content": "Dangling Q2"},
        ]
        sanitized = sanitize_messages(raw_msgs)
        assert len(sanitized) == 2
        assert sanitized[-1]["role"] == "assistant"
        assert sanitized[-1]["content"] == "Valid A"

    def test_single_turn_only_assistant_is_discarded(self):
        raw_msgs = [{"role": "assistant", "content": "Only assistant text without user"}]
        assert sanitize_messages(raw_msgs) == []

    def test_single_turn_only_user_is_discarded(self):
        raw_msgs = [{"role": "user", "content": "Only user text without assistant"}]
        assert sanitize_messages(raw_msgs) == []

    def test_only_system_turns_are_discarded(self):
        raw_msgs = [
            {"role": "system", "content": "Sys 1"},
            {"role": "system", "content": "Sys 2"},
        ]
        assert sanitize_messages(raw_msgs) == []

    def test_role_synonym_mappings(self):
        synonyms_user = ["human", "user_msg", "client", "customer", "prompter"]
        synonyms_assistant = ["gpt", "bot", "model", "chatgpt", "system_response"]

        for u_syn, a_syn in zip(synonyms_user, synonyms_assistant):
            record = {
                "conversations": [
                    {"speaker": u_syn, "text": f"Question from {u_syn}"},
                    {"speaker": a_syn, "text": f"Answer from {a_syn}"},
                ]
            }
            msgs = normalize_record(record, format_name="sharegpt", default_system_prompt=None)
            assert len(msgs) == 2
            assert msgs[0]["role"] == "user"
            assert msgs[0]["content"] == f"Question from {u_syn}"
            assert msgs[1]["role"] == "assistant"
            assert msgs[1]["content"] == f"Answer from {a_syn}"


# ==============================================================================
# 4. SPLITTING BOUNDARY CONDITIONS & DETERMINISTIC SEED VERIFICATION
# ==============================================================================

class TestSplittingBoundaryConditionsAndSeeding:
    """Stress-test splitting boundaries (N=0,1,2,5,9,10,100) and random seed determinism."""

    def _make_dataset(self, n: int) -> Dataset:
        data = [{"messages": [{"role": "user", "content": f"Q_{i}"}, {"role": "assistant", "content": f"A_{i}"}]} for i in range(n)]
        return Dataset.from_list(data)

    def test_n0_raises_empty_error(self):
        ds = self._make_dataset(0)
        with pytest.raises(DatasetEmptyError, match="Dataset is empty"):
            split_dataset(ds, val_split=0.1)

    def test_n1_always_yields_val_none(self):
        ds = self._make_dataset(1)
        for split in [0.0, 0.1, 0.5, 0.99]:
            train_ds, val_ds = split_dataset(ds, val_split=split)
            assert len(train_ds) == 1
            assert val_ds is None

    def test_n2_allocates_at_least_1_train_and_1_val(self):
        ds = self._make_dataset(2)
        train_ds, val_ds = split_dataset(ds, val_split=0.1, seed=42)
        assert len(train_ds) == 1
        assert len(val_ds) == 1

    def test_n5_allocates_at_least_1_train_and_1_val(self):
        ds = self._make_dataset(5)
        train_ds, val_ds = split_dataset(ds, val_split=0.1, seed=42)
        assert len(train_ds) == 4
        assert len(val_ds) == 1
        assert len(train_ds) + len(val_ds) == 5

    def test_n9_allocates_correct_proportion(self):
        ds = self._make_dataset(9)
        train_ds, val_ds = split_dataset(ds, val_split=0.2, seed=42)
        assert len(val_ds) == 2
        assert len(train_ds) == 7

    def test_n10_allocates_correct_proportion(self):
        ds = self._make_dataset(10)
        train_ds, val_ds = split_dataset(ds, val_split=0.1, seed=42)
        assert len(val_ds) == 1
        assert len(train_ds) == 9

    def test_n100_split_proportions(self):
        ds = self._make_dataset(100)
        train_ds, val_ds = split_dataset(ds, val_split=0.15, seed=42)
        assert len(val_ds) == 15
        assert len(train_ds) == 85

    def test_val_split_zero_and_negative(self):
        ds = self._make_dataset(20)
        for split in [0.0, -0.1, -1.0]:
            train_ds, val_ds = split_dataset(ds, val_split=split)
            assert len(train_ds) == 20
            assert val_ds is None

    def test_extreme_val_split_guarantees_at_least_one_train_sample(self):
        ds = self._make_dataset(10)
        train_ds, val_ds = split_dataset(ds, val_split=0.99, seed=42)
        assert len(train_ds) >= 1
        assert len(val_ds) == 9
        assert len(train_ds) + len(val_ds) == 10

    def test_deterministic_seed_repeatability_100_runs(self):
        ds = self._make_dataset(50)
        reference_train, reference_val = split_dataset(ds, val_split=0.2, seed=1337)
        ref_train_contents = [x["messages"][0]["content"] for x in reference_train]
        ref_val_contents = [x["messages"][0]["content"] for x in reference_val]

        for _ in range(100):
            t_ds, v_ds = split_dataset(ds, val_split=0.2, seed=1337)
            assert [x["messages"][0]["content"] for x in t_ds] == ref_train_contents
            assert [x["messages"][0]["content"] for x in v_ds] == ref_val_contents

    def test_disjointness_of_train_and_val_partitions(self):
        ds = self._make_dataset(60)
        train_ds, val_ds = split_dataset(ds, val_split=0.25, seed=77)
        train_queries = set(x["messages"][0]["content"] for x in train_ds)
        val_queries = set(x["messages"][0]["content"] for x in val_ds)

        assert len(train_queries.intersection(val_queries)) == 0
        assert len(train_queries.union(val_queries)) == 60


# ==============================================================================
# 5. FORMAT AUTO-DETECTION DECEPTIVE & ADVERSARIAL CASES
# ==============================================================================

class TestFormatAutoDetectionRobustness:
    """Stress-test format detector against deceptive keys, hybrid keys, and ambiguous buffers."""

    def test_majority_voting_resolution(self):
        mixed_buffer = [
            {"conversations": [{"from": "human", "value": "h1"}, {"from": "gpt", "value": "g1"}]},
            {"conversations": [{"from": "human", "value": "h2"}, {"from": "gpt", "value": "g2"}]},
            {"conversations": [{"from": "human", "value": "h3"}, {"from": "gpt", "value": "g3"}]},
            {"conversations": [{"from": "human", "value": "h4"}, {"from": "gpt", "value": "g4"}]},
            {"instruction": "i1", "output": "o1"},
            {"instruction": "i2", "output": "o2"},
        ]
        assert detect_format(mixed_buffer) == "sharegpt"

    def test_deceptive_keys_alpaca_with_conversations(self):
        record = {
            "conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}],
            "instruction": "Deceptive instruction",
            "output": "Deceptive output",
        }
        detected = detect_format([record])
        assert detected == "sharegpt"

    def test_empty_conversations_list_detected_properly(self):
        record = {"conversations": []}
        assert detect_format([record]) == "sharegpt"

    def test_ambiguous_non_format_keys_raise_validation_error(self):
        ambiguous_records = [
            {"user_name": "alice", "score": 100},
            {"timestamp": "2026-08-15", "payload": "data"},
        ]
        with pytest.raises(DatasetFormatValidationError, match="Unable to auto-detect dataset format"):
            detect_format(ambiguous_records)


# ==============================================================================
# 6. TOKEN PROFILING & STATISTICS ADVERSARIAL TESTS
# ==============================================================================

class TestDatasetProfilingAndStatistics:
    """Stress-test token statistics, empty datasets, and truncation metrics."""

    def test_zero_samples_statistics(self):
        ds = Dataset.from_list([])
        stats = compute_dataset_statistics(ds)
        assert stats["num_samples"] == 0
        assert stats["total_turns"] == 0
        assert stats["truncated_samples_count"] == 0
        assert stats["length_stats"]["mean"] == 0.0

    def test_truncation_detection(self):
        long_content = "Word " * 600
        raw_data = [
            {"messages": [{"role": "user", "content": "Short"}, {"role": "assistant", "content": "Short"}]},
            {"messages": [{"role": "user", "content": long_content}, {"role": "assistant", "content": "OK"}]},
        ]
        ds = Dataset.from_list(raw_data)
        stats = compute_dataset_statistics(ds, max_seq_length=500)

        assert stats["num_samples"] == 2
        assert stats["truncated_samples_count"] == 1
        assert stats["truncated_samples_percent"] == 50.0
        assert stats["length_stats"]["max"] >= 500


# ==============================================================================
# 7. MAX SAMPLES TRUNCATION PROBING
# ==============================================================================

class TestMaxSamplesTruncation:
    """Verify max_samples behaves strictly as a pre-filter ceiling."""

    def test_max_samples_limits_raw_records(self, tmp_path: Path):
        file_path = tmp_path / "large_dataset.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for i in range(50):
                f.write(json.dumps({"instruction": f"Q{i}", "output": f"A{i}"}) + "\n")

        train_ds, _ = load_dataset_from_file(file_path, val_split=0.0, max_samples=7)
        assert len(train_ds) == 7
