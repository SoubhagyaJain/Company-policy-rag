"""Unit tests for evaluation framework (no Ollama required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from src.evaluation import (
    RELEVANCY_PROMPT,
    CaseResult,
    GoldenCase,
    _aggregate_for_cases,
    compute_retrieval_metrics,
    evaluate_case,
    filter_cases_by_corpus,
    is_chunk_relevant,
    load_golden_dataset,
)


def _node(text: str, **meta: str | int) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text=text, metadata=dict(meta)), score=0.8)


class TestRelevanceMatching:
    def test_section_path_match(self) -> None:
        node = _node(
            "Employees accrue 15 days of vacation annually.",
            section_path="II. GENERAL > 5.2 Vacation Benefits",
            section_title="Vacation Benefits",
        )
        assert is_chunk_relevant(node, ["vacation", "leave"])

    def test_no_match(self) -> None:
        node = _node("Dress code requires business casual.", section_title="Dress Code")
        assert not is_chunk_relevant(node, ["harassment", "discrimination"])

    def test_empty_sections_not_relevant(self) -> None:
        node = _node("Some text about pets.")
        assert not is_chunk_relevant(node, [])


class TestRetrievalMetrics:
    def test_hit_rate_and_precision(self) -> None:
        nodes = [
            _node("Vacation policy details", section_title="Vacation"),
            _node("Unrelated dress code", section_title="Dress Code"),
        ]
        hit, prec, rec, rel, _ = compute_retrieval_metrics(nodes, ["vacation"])
        assert hit == 1.0
        assert prec == 0.5
        assert rel == 1

    def test_empty_retrieval(self) -> None:
        hit, prec, rec, _, _ = compute_retrieval_metrics([], ["vacation"])
        assert hit == 0.0
        assert prec == 0.0


class TestGoldenDataset:
    def test_load_dataset(self) -> None:
        cases = load_golden_dataset()
        assert len(cases) >= 50
        assert all(isinstance(c, GoldenCase) for c in cases)
        categories = {c.category for c in cases}
        assert "factual" in categories
        assert "edge_case" in categories
        corpora = {c.corpus for c in cases}
        assert "policy" in corpora
        assert "guidebook" in corpora

    def test_filter_by_corpus(self) -> None:
        cases = load_golden_dataset()
        policy_only = filter_cases_by_corpus(cases, "policy")
        guidebook_only = filter_cases_by_corpus(cases, "guidebook")
        assert len(policy_only) >= 20
        assert len(guidebook_only) >= 20
        assert all(c.corpus == "policy" for c in policy_only)

    def test_dataset_file_exists(self) -> None:
        path = Path(__file__).resolve().parent.parent / "data" / "eval" / "golden_dataset.json"
        assert path.exists()

    def test_ci_smoke_dataset_exists(self) -> None:
        root = Path(__file__).resolve().parent.parent
        smoke = root / "data" / "eval" / "golden_subset_ci_smoke.json"
        baseline = root / "data" / "eval" / "ci_smoke_baseline.json"
        assert smoke.exists()
        assert baseline.exists()
        cases = load_golden_dataset(smoke)
        assert len(cases) == 8

    def test_guidebook_dataset_canonical_path(self) -> None:
        root = Path(__file__).resolve().parent.parent
        path = root / "data" / "eval" / "golden_dataset_guidebook.json"
        assert path.exists()
        cases = load_golden_dataset(path)
        assert len(cases) == 35
        assert all(c.corpus == "guidebook" for c in cases)
        query_types = {c.query_type for c in cases}
        assert "code" in query_types
        assert "edge_case" in query_types


def test_relevancy_prompt_includes_context() -> None:
    assert "CONTEXT:" in RELEVANCY_PROMPT
    assert "{context}" in RELEVANCY_PROMPT
    assert "semantic-mapping" in RELEVANCY_PROMPT.lower()


class TestRetrievalOnly:
    def test_evaluate_case_skips_generation_and_judge(self) -> None:
        case = GoldenCase(
            id="sick_leave",
            category="factual",
            question="How many sick days?",
            expected_answer="Should describe sick leave.",
            relevant_sections=["sick", "leave"],
            corpus="policy",
            query_type="factual",
        )
        nodes = [_node("Sick leave accrual details", section_title="Sick Leave")]
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = nodes
        mock_index = MagicMock()

        with patch("src.evaluation.create_retriever", return_value=mock_retriever):
            result = evaluate_case(
                case,
                mock_index,
                use_llm_judge=False,
                retrieval_only=True,
            )

        assert result.hit_rate == 1.0
        assert result.context_precision == 1.0
        assert result.generated_answer == ""
        assert result.faithfulness is None
        assert result.answer_relevancy is None
        assert result.guard_modified is False
        mock_retriever.retrieve.assert_called_once_with(case.question)


class TestPhase3Aggregates:
    def test_code_validation_and_fallback_rates(self) -> None:
        results = [
            CaseResult(
                id="a",
                category="factual",
                corpus="guidebook",
                query_type="code",
                question="q1",
                hit_rate=1.0,
                context_precision=1.0,
                context_recall=1.0,
                faithfulness=1.0,
                answer_relevancy=0.9,
                generated_answer="ok",
                retrieved_count=3,
                relevant_retrieved_count=2,
                code_validation_triggered=True,
                code_validation_passed=True,
                fallback_reason="none",
            ),
            CaseResult(
                id="b",
                category="factual",
                corpus="guidebook",
                query_type="code",
                question="q2",
                hit_rate=1.0,
                context_precision=1.0,
                context_recall=1.0,
                faithfulness=0.8,
                answer_relevancy=0.7,
                generated_answer="fallback",
                retrieved_count=3,
                relevant_retrieved_count=2,
                code_validation_triggered=True,
                code_validation_passed=False,
                fallback_reason="code_validation",
            ),
            CaseResult(
                id="c",
                category="factual",
                corpus="policy",
                query_type="factual",
                question="q3",
                hit_rate=1.0,
                context_precision=1.0,
                context_recall=1.0,
                faithfulness=1.0,
                answer_relevancy=1.0,
                generated_answer="plain",
                retrieved_count=2,
                relevant_retrieved_count=1,
            ),
        ]
        agg = _aggregate_for_cases(results)
        assert agg["code_validation_pass_rate"] == 0.5
        assert agg["low_confidence_fallback_rate"] == pytest.approx(1 / 3)
