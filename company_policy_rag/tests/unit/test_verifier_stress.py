"""
Adversarial Stress Test Suite for SelfReflectionVerifier and RetryEngine.
Covers:
1. Hallucination detection: ungrounded monetary figures, unmentioned software, unsupported entities.
2. Missing query constraints: omitted deadlines, omitted comparison contrasts, partial entity coverage.
3. Invalid and missing citations: missing [Source N], out-of-bounds indices [Source 99], [Source 0].
4. Mathematical verification of composite score: 0.35*Faith + 0.30*Comp + 0.20*Cit + 0.15*Coh.
5. Boundary and edge cases: empty strings, unanswerable queries, empty context pools, custom validator fallbacks.
6. RetryEngine integration: parameter adjustments, prompt directives, and 2-retry hard cap.
"""
from __future__ import annotations

import math
import pytest
from pydantic import BaseModel

from backend.models.chunk import Chunk, ChunkMetadata
from backend.models.rag import Citation, RetrievalStrategy, ScoredChunk, VerificationReport
from backend.rag.retry_engine import RetryEngine
from backend.rag.verifier import SelfReflectionVerifier


def _create_sample_chunks() -> list[ScoredChunk]:
    """Helper to create realistic scored context chunks for testing."""
    c1 = Chunk(
        id="chunk_pto_01",
        text="Section 3.1 Paid Time Off: Full-time employees accrue 15 days of PTO annually. Part-time employees receive 7 days.",
        metadata=ChunkMetadata(
            document_id="doc_pto",
            source_file="PTO_Policy_2026.pdf",
            file_path="data/policies/PTO_Policy_2026.pdf",
            file_hash="hash_pto_01",
            document_type="company_policy",
        ),
    )
    c2 = Chunk(
        id="chunk_travel_02",
        text="Section 4.2 Travel Expense Reimbursement: Daily meal per diem is capped at $75 per day. Mileage is reimbursed at $0.65 per mile. Expense reports must be submitted within 30 days.",
        metadata=ChunkMetadata(
            document_id="doc_travel",
            source_file="Travel_Policy_2026.pdf",
            file_path="data/policies/Travel_Policy_2026.pdf",
            file_hash="hash_travel_02",
            document_type="company_policy",
        ),
    )
    c3 = Chunk(
        id="chunk_sec_03",
        text="Section 1.5 Remote Security: Remote employees must connect through GlobalProtect VPN with MFA enabled. Unauthorized hardware or personal USB drives are strictly prohibited.",
        metadata=ChunkMetadata(
            document_id="doc_sec",
            source_file="Security_Policy_2026.pdf",
            file_path="data/policies/Security_Policy_2026.pdf",
            file_hash="hash_sec_03",
            document_type="company_policy",
        ),
    )
    return [
        ScoredChunk(chunk=c1, score=0.95),
        ScoredChunk(chunk=c2, score=0.88),
        ScoredChunk(chunk=c3, score=0.82),
    ]


class TestHallucinationDetection:
    """Stress tests verifying that ungrounded facts, false numbers, and unmentioned entities are flagged."""

    def test_ungrounded_large_monetary_figures(self) -> None:
        """Adversarial hallucination: $50,000 or $5,000 allowance not present in context."""
        verifier = SelfReflectionVerifier(threshold=0.70)
        chunks = _create_sample_chunks()
        citations = [Citation(source_index=2, chunk_id="chunk_travel_02", document_id="doc_travel", source_file="Travel_Policy_2026.pdf", snippet=chunks[1].chunk.text)]

        query = "What is the travel per diem allowance?"
        hallucinated_answer = "The travel per diem allowance is $50,000 per day for luxury hotels. [Source 2]"

        report = verifier.verify(query, hallucinated_answer, chunks, citations)

        assert report.passed is False
        assert report.faithfulness < 0.65
        assert report.faithfulness <= 0.40
        assert len(report.unsupported_claims) > 0
        assert any("$50,000" in claim or "Unverified numerical" in claim for claim in report.unsupported_claims)

    def test_ungrounded_five_thousand_monetary_figure(self) -> None:
        """Adversarial hallucination: $5,000 or $5000 explicitly checked in verifier rules."""
        verifier = SelfReflectionVerifier(threshold=0.70)
        chunks = _create_sample_chunks()
        citations = [Citation(source_index=2, chunk_id="chunk_travel_02", document_id="doc_travel", source_file="Travel_Policy_2026.pdf", snippet=chunks[1].chunk.text)]

        query = "What is the home office budget?"
        hallucinated_answer = "Employees receive $5,000 for home office setup expenses. [Source 2]"

        report = verifier.verify(query, hallucinated_answer, chunks, citations)

        assert report.passed is False
        assert report.faithfulness < 0.65
        assert report.faithfulness <= 0.35
        assert len(report.unsupported_claims) > 0

    def test_unmentioned_software_and_tools(self) -> None:
        """Adversarial hallucination: Answer fabricates third-party software stack not mentioned in policy."""
        verifier = SelfReflectionVerifier(threshold=0.70)
        # Context is purely PTO policy
        chunks = [_create_sample_chunks()[0]]
        citations = [Citation(source_index=1, chunk_id="chunk_pto_01", document_id="doc_pto", source_file="PTO_Policy_2026.pdf", snippet=chunks[0].chunk.text)]

        query = "How do I request PTO?"
        hallucinated_answer = "PTO requests must be submitted through Slack, Jira Service Desk, Salesforce, Docker, and Kubernetes workflows. [Source 1]"

        report = verifier.verify(query, hallucinated_answer, chunks, citations)

        assert report.passed is False
        assert report.faithfulness < 0.65
        assert report.faithfulness <= 0.35

    def test_unsupported_furniture_category(self) -> None:
        """Adversarial hallucination: Answer claims furniture purchases when context does not mention furniture."""
        verifier = SelfReflectionVerifier(threshold=0.70)
        chunks = [_create_sample_chunks()[0]]  # Only PTO
        citations = [Citation(source_index=1, chunk_id="chunk_pto_01", document_id="doc_pto", source_file="PTO_Policy_2026.pdf", snippet=chunks[0].chunk.text)]

        query = "What equipment can I expense?"
        hallucinated_answer = "You may expense unauthorized furniture and luxury chairs under the policy. [Source 1]"

        report = verifier.verify(query, hallucinated_answer, chunks, citations)

        assert report.passed is False
        assert report.faithfulness <= 0.35
        assert len(report.unsupported_claims) > 0


class TestMissingQueryConstraints:
    """Stress tests verifying that multi-part queries with missing constraints drop completeness below 0.65."""

    def test_multi_part_query_missing_deadline(self) -> None:
        """Query requests eligibility AND submission deadline; answer completely ignores deadline."""
        verifier = SelfReflectionVerifier(threshold=0.70)
        chunks = _create_sample_chunks()
        citations = [Citation(source_index=2, chunk_id="chunk_travel_02", document_id="doc_travel", source_file="Travel_Policy_2026.pdf", snippet=chunks[1].chunk.text)]

        query = "What is the per diem rate and what are the submission deadlines for travel expenses?"
        # Answer covers per diem ($75) but omits deadline (30 days)
        partial_answer = "The per diem rate is $75 per day for daily meals. [Source 2]"

        report = verifier.verify(query, partial_answer, chunks, citations)

        assert report.passed is False
        assert report.completeness < 0.65
        assert report.completeness <= 0.45
        assert len(report.missing_aspects) > 0
        assert any("deadline" in m.lower() for m in report.missing_aspects)

    def test_comparison_query_missing_contrast_distinction(self) -> None:
        """Comparison query without comparative/differential terms in answer."""
        verifier = SelfReflectionVerifier(threshold=0.70)
        chunks = _create_sample_chunks()
        citations = [Citation(source_index=1, chunk_id="chunk_pto_01", document_id="doc_pto", source_file="PTO_Policy_2026.pdf", snippet=chunks[0].chunk.text)]

        query = "Compare PTO accrual between full-time and part-time employees."
        # Answer lists full-time PTO only without comparative structure
        one_sided_answer = "Full-time employees accrue 15 days of PTO annually according to policy. [Source 1]"

        report = verifier.verify(query, one_sided_answer, chunks, citations)

        assert report.passed is False
        assert report.completeness < 0.65
        assert any("Comparative distinction" in m for m in report.missing_aspects)

    def test_multi_topic_query_insufficient_aspect_coverage(self) -> None:
        """Query with 4 distinct subjects; answer covers only 1."""
        verifier = SelfReflectionVerifier(threshold=0.70)
        chunks = _create_sample_chunks()
        citations = [Citation(source_index=1, chunk_id="chunk_pto_01", document_id="doc_pto", source_file="PTO_Policy_2026.pdf", snippet=chunks[0].chunk.text)]

        query = "What are the rules regarding bereavement leave, parental benefits, jury duty compensation, and sabbatical eligibility?"
        sparse_answer = "The policy provides bereavement leave for eligible employees. [Source 1]"

        report = verifier.verify(query, sparse_answer, chunks, citations)

        assert report.passed is False
        assert report.completeness < 0.65


class TestCitationEvaluation:
    """Stress tests evaluating missing, malformed, and out-of-bounds bracketed citations."""

    def test_missing_citations_entirely(self) -> None:
        """Answer contains factual content but zero citations and empty citation list."""
        verifier = SelfReflectionVerifier(threshold=0.70)
        chunks = _create_sample_chunks()

        query = "What is the annual PTO accrual for full-time employees?"
        uncited_answer = "Full-time employees accrue 15 days of PTO annually."

        report = verifier.verify(query, uncited_answer, chunks, citations=[])

        assert report.passed is False
        assert report.citation_coverage == 0.20
        assert report.citation_coverage < 0.50

    def test_out_of_bounds_citation_index(self) -> None:
        """Answer cites [Source 99] when only 3 context chunks are provided."""
        verifier = SelfReflectionVerifier(threshold=0.70)
        chunks = _create_sample_chunks()  # Total 3 chunks (valid 1..3)

        query = "What is the annual PTO accrual?"
        bad_citation_answer = "Full-time employees accrue 15 days of PTO annually. [Source 99]"

        report = verifier.verify(query, bad_citation_answer, chunks, citations=[])

        assert report.passed is False
        assert report.citation_coverage == 0.15
        assert report.citation_coverage < 0.50

    def test_zero_index_citation(self) -> None:
        """Answer cites [Source 0] which is invalid 1-indexed citation."""
        verifier = SelfReflectionVerifier(threshold=0.70)
        chunks = _create_sample_chunks()

        query = "What is the annual PTO accrual?"
        zero_cit_answer = "Full-time employees accrue 15 days of PTO annually. [Source 0]"

        report = verifier.verify(query, zero_cit_answer, chunks, citations=[])

        assert report.passed is False
        assert report.citation_coverage == 0.15

    def test_valid_citation_scoring(self) -> None:
        """Answer with valid [Source 1] matching context pool."""
        verifier = SelfReflectionVerifier(threshold=0.70)
        chunks = _create_sample_chunks()
        citation = Citation(source_index=1, chunk_id="chunk_pto_01", document_id="doc_pto", source_file="PTO_Policy_2026.pdf", snippet=chunks[0].chunk.text)

        query = "What is the annual PTO accrual for full-time employees?"
        valid_answer = "Full-time employees accrue 15 days of PTO annually. [Source 1]"

        report = verifier.verify(query, valid_answer, chunks, citations=[citation])

        assert report.citation_coverage >= 0.95


class TestCompositeScoreCalculation:
    """Stress tests validating the composite score mathematical formula and bounded pass gates."""

    @pytest.mark.parametrize(
        "faith,comp,cit,coh,expected_composite",
        [
            (1.0, 1.0, 1.0, 1.0, 1.000),
            (0.0, 0.0, 0.0, 0.0, 0.000),
            (0.80, 0.70, 0.90, 0.85, 0.798),
            (0.65, 0.50, 0.50, 0.70, 0.583),
            (0.90, 0.85, 0.95, 0.95, 0.902),
            (0.35, 0.40, 0.20, 0.70, 0.388),
            (0.70, 0.70, 0.70, 0.70, 0.700),
        ],
    )
    def test_composite_formula_precision(
        self,
        faith: float,
        comp: float,
        cit: float,
        coh: float,
        expected_composite: float,
    ) -> None:
        """Validates exact formula: 0.35*Faith + 0.30*Comp + 0.20*Cit + 0.15*Coh."""
        verifier = SelfReflectionVerifier(threshold=0.70)

        # Use custom validator hook to inject precise component scores
        def validator(q: str, a: str, c: list[ScoredChunk]):
            return faith, comp, cit, coh, []

        report = verifier.verify("query", "answer text.", _create_sample_chunks(), [], custom_validator=validator)

        expected = round(0.35 * faith + 0.30 * comp + 0.20 * cit + 0.15 * coh, 3)
        assert report.composite_score == expected
        assert math.isclose(report.composite_score, expected_composite, abs_tol=1e-3)

    def test_bounded_pass_gate_fails_when_faithfulness_below_subgate(self) -> None:
        """
        Adversarial case: High completeness, citation, and coherence push composite >= 0.70,
        but faith=0.35 is below 0.65. Verification MUST fail.
        """
        verifier = SelfReflectionVerifier(threshold=0.70)

        # Composite: 0.35*(0.35) + 0.30*(0.95) + 0.20*(0.95) + 0.15*(0.95) = 0.1225 + 0.285 + 0.190 + 0.1425 = 0.740
        def validator(q: str, a: str, c: list[ScoredChunk]):
            return 0.35, 0.95, 0.95, 0.95, []

        report = verifier.verify("query", "answer.", _create_sample_chunks(), [], custom_validator=validator)

        assert report.composite_score == 0.740
        assert report.composite_score >= 0.70
        # Must fail because faith < 0.65
        assert report.passed is False
        assert report.critique is not None
        assert "claims not grounded" in report.critique


class TestBoundaryAndEdgeCases:
    """Stress tests boundary cases: empty answer, unanswerable queries, empty context chunks, error recovery."""

    def test_empty_and_whitespace_answers(self) -> None:
        """Empty string, whitespace, and newline inputs return zero scores and passed=False."""
        verifier = SelfReflectionVerifier()
        chunks = _create_sample_chunks()

        for bad_ans in ["", "   ", "\n\t\n", None]:
            report = verifier.verify("What is the PTO policy?", bad_ans or "", chunks, [])
            assert report.passed is False
            assert report.composite_score == 0.0
            assert report.faithfulness == 0.0
            assert report.completeness == 0.0
            assert report.citation_coverage == 0.0
            assert report.coherence == 0.0
            assert "Empty answer" in report.critique

    def test_unanswerable_query_response(self) -> None:
        """Standard unanswerable response passes verification with full marks."""
        verifier = SelfReflectionVerifier()
        chunks = []  # No context chunks found

        unanswerable_ans = "I am unable to answer based on the provided documents."
        report = verifier.verify("What is the quantum computing policy?", unanswerable_ans, chunks, [])

        assert report.passed is True
        assert report.faithfulness == 1.0
        assert report.completeness == 1.0
        assert report.citation_coverage == 1.0
        assert report.coherence >= 0.90
        assert report.composite_score >= 0.90

    def test_empty_context_chunks_with_fabricated_claims(self) -> None:
        """Empty context chunks with answer asserting claims and citations."""
        verifier = SelfReflectionVerifier()
        empty_chunks = []
        citations = [Citation(source_index=1, chunk_id="chunk_fake", document_id="doc_fake", source_file="fake.pdf", snippet="Fake snippet")]

        answer = "Employees receive 100 days of vacation annually. [Source 1]"
        report = verifier.verify("What is vacation policy?", answer, empty_chunks, citations)

        assert report.passed is False
        assert report.faithfulness <= 0.20
        assert len(report.unsupported_claims) > 0

    def test_short_answer_and_punctuation_coherence(self) -> None:
        """Answers under 5 words or missing punctuation have reduced coherence."""
        verifier = SelfReflectionVerifier()

        # Under 5 words
        short_report = verifier.verify("What is PTO?", "15 days. [Source 1]", _create_sample_chunks(), [])
        assert short_report.coherence == 0.50

        # No punctuation ending
        unpunct_report = verifier.verify(
            "What is PTO?",
            "Full-time employees receive fifteen days of paid time off per year",
            _create_sample_chunks(),
            [],
        )
        assert unpunct_report.coherence == 0.70

    def test_custom_validator_exception_fallback(self) -> None:
        """Custom validator raising exception does not crash verifier; falls back to heuristics."""
        verifier = SelfReflectionVerifier()
        chunks = _create_sample_chunks()
        citation = Citation(source_index=1, chunk_id="chunk_pto_01", document_id="doc_pto", source_file="PTO_Policy_2026.pdf", snippet=chunks[0].chunk.text)

        def buggy_validator(q, a, c):
            raise RuntimeError("Unexpected LLM validator network timeout")

        report = verifier.verify(
            "What is the PTO accrual for full-time employees?",
            "Full-time employees accrue 15 days of PTO annually. [Source 1]",
            chunks,
            [citation],
            custom_validator=buggy_validator,
        )

        assert report is not None
        assert report.passed is True
        assert report.faithfulness >= 0.70


class TestRetryEngineIntegration:
    """Stress tests for RetryEngine autonomous recovery and hard cap enforcement."""

    def test_retry_engine_faithfulness_adjustment(self) -> None:
        """Low faithfulness triggers temperature decrease and min_score_ratio increase."""
        engine = RetryEngine(max_retries=2)
        report = VerificationReport(
            faithfulness=0.40,
            completeness=0.85,
            citation_coverage=0.90,
            coherence=0.95,
            composite_score=0.71,
            passed=False,
            unsupported_claims=["$50,000 allowance"],
        )
        strategy = RetrievalStrategy(min_score_ratio=0.40, temperature=0.10)

        assert engine.should_retry(0, report) is True
        new_strat, prompt = engine.prepare_retry(0, report, strategy)

        assert new_strat.min_score_ratio == 0.50
        assert new_strat.temperature == 0.05
        assert "Strictly adhere to the retrieved facts" in prompt
        assert "$50,000 allowance" in prompt

    def test_retry_engine_completeness_adjustment(self) -> None:
        """Low completeness triggers top_k expansion, multi-query, and parent expansion."""
        engine = RetryEngine(max_retries=2)
        report = VerificationReport(
            faithfulness=0.90,
            completeness=0.40,
            citation_coverage=0.90,
            coherence=0.95,
            composite_score=0.74,
            passed=False,
            missing_aspects=["Application submission deadline"],
        )
        strategy = RetrievalStrategy(dense_top_k=15, bm25_top_k=15, rerank_top_n=6, enable_multi_query=False)

        assert engine.should_retry(0, report) is True
        new_strat, prompt = engine.prepare_retry(0, report, strategy)

        assert new_strat.dense_top_k == 25
        assert new_strat.bm25_top_k == 25
        assert new_strat.rerank_top_n == 9
        assert new_strat.enable_multi_query is True
        assert new_strat.enable_parent_expansion is True
        assert "Application submission deadline" in prompt

    def test_retry_engine_two_retry_hard_cap(self) -> None:
        """Strict enforcement of 2-retry hard cap: attempt 0 (allowed), attempt 1 (allowed), attempt 2 (rejected)."""
        engine = RetryEngine(max_retries=2)
        failing_report = VerificationReport(
            faithfulness=0.20,
            completeness=0.20,
            citation_coverage=0.20,
            coherence=0.50,
            composite_score=0.245,
            passed=False,
        )

        assert engine.should_retry(0, failing_report) is True
        assert engine.should_retry(1, failing_report) is True
        assert engine.should_retry(2, failing_report) is False

        with pytest.raises(ValueError, match="Max retries"):
            engine.prepare_retry(2, failing_report, RetrievalStrategy())
