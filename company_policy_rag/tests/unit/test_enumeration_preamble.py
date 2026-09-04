"""
Regression tests for the deterministic enumerated-answer lead-in.

Root cause fixed: the pipeline hardcoded the preamble
    "The five LLM fine-tuning techniques are:"
for EVERY deterministically extracted numbered list, so any other
"list the N X" question emitted a factually wrong subject sentence
(e.g. a PTO-steps question would be labelled "LLM fine-tuning techniques").

`_enumeration_preamble` now derives the subject from the query itself:
- it reproduces the exact fine-tuning phrasing for the fine-tuning query
  (so the existing e2e expectation stays green), and
- it produces a correct, non-fabricated subject for every other question,
  falling back to a neutral lead-in when the noun phrase cannot be parsed.
"""

import pytest

from backend.rag.pipeline import _enumeration_preamble


class TestEnumerationPreamble:
    def test_fine_tuning_phrasing_preserved(self):
        # The one query the old hardcode was written for must still match exactly.
        assert (
            _enumeration_preamble("What are the five LLM fine-tuning techniques?", 5)
            == "The five LLM fine-tuning techniques are:"
        )

    @pytest.mark.parametrize(
        "query,count,expected",
        [
            ("List the 3 steps to reset a password", 3, "The 3 steps are:"),
            ("what are the four types of leave", 4, "The four types are:"),
            ("name the 5 methods", 5, "The 5 methods are:"),
            ("what are the two ways to submit expenses", 2, "The two ways are:"),
        ],
    )
    def test_generic_subject_is_derived_from_query(self, query, count, expected):
        assert _enumeration_preamble(query, count) == expected

    def test_no_false_fine_tuning_label_for_unrelated_lists(self):
        # The core bug: a non-fine-tuning enumeration must never be labelled
        # "LLM fine-tuning techniques".
        for query in (
            "List the 3 steps to reset a password",
            "what are the four types of leave",
            "name the 5 approval methods",
        ):
            out = _enumeration_preamble(query, 3).lower()
            assert "fine-tuning" not in out
            assert "llm" not in out

    def test_neutral_fallback_when_subject_unparseable(self):
        # No enumerated noun phrase -> a neutral, non-fabricated lead-in.
        out = _enumeration_preamble("summarize the policy", 2)
        assert out == "Based on the retrieved document, the 2 items are:"
        assert "fine-tuning" not in out.lower()
