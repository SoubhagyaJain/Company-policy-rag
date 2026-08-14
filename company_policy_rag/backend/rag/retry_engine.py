"""
Autonomous Retry Engine for self-reflection verification failure recovery.
Dynamically adjusts retrieval parameters and synthesizes prompt guidance (hard capped at 2 retries).
"""
from __future__ import annotations

from typing import Tuple

from backend.models.rag import RetrievalStrategy, VerificationReport
from backend.utils.logging import logger
from src.config import settings


class RetryEngine:
    """
    Autonomous retry engine orchestrating retrieval parameter adjustments and prompt directives
    based on verification evaluation failures across 4 dimensions:
    - Faithfulness: adjusts reranking threshold (min_score_ratio), lowers temperature, instructs strict grounding.
    - Completeness: increases dense_top_k and bm25_top_k (+10), enables multi_query and parent_expansion, targets missing aspects.
    - Citation Coverage: requires explicit bracketed [Source N] citations for all claims.
    - Coherence: enforces structure, formatting, and clarity.
    
    Hard Cap: strictly limits retries to MAX_RETRIES (default 2), allowing at most 3 total attempts
    (attempt 0 = initial, attempt 1 = 1st retry, attempt 2 = 2nd retry).
    """

    MAX_RETRIES: int = 2

    def __init__(self, max_retries: int | None = None) -> None:
        if max_retries is not None:
            self.max_retries = max_retries
        else:
            self.max_retries = getattr(settings, "verification_max_retries", 2)

    def should_retry(self, attempt: int, report: VerificationReport) -> bool:
        """
        Determine if another retry cycle should be executed.
        
        Returns True ONLY if:
        1. Current attempt index < max_retries (attempt 0 or 1 for max_retries=2)
        2. Verification report did NOT pass (report.passed is False)
        """
        return attempt < self.max_retries and not report.passed

    def prepare_retry(
        self,
        attempt: int,
        report: VerificationReport,
        strategy: RetrievalStrategy,
        query: str = "",
    ) -> Tuple[RetrievalStrategy, str]:
        """
        Adjust retrieval strategy parameters and generate prompt refinement guidance
        tailored to the specific verification failure causes.
        
        Args:
            attempt: Current 0-based attempt index (0 = initial failed attempt, 1 = 1st retry failed)
            report: VerificationReport from SelfReflectionVerifier
            strategy: Current RetrievalStrategy
            query: Original user query string
            
        Returns:
            Tuple of (adjusted_strategy, prompt_refinement_instructions)
            
        Raises:
            ValueError if attempt >= max_retries
        """
        if attempt >= self.max_retries:
            raise ValueError(f"Max retries ({self.max_retries}) exceeded.")

        new_strategy = strategy.model_copy(deep=True)
        instructions: list[str] = []

        # 1. Faithfulness Failure Recovery
        if report.faithfulness < 0.65 or report.unsupported_claims:
            new_strategy.min_score_ratio = min(0.60, round(new_strategy.min_score_ratio + 0.10, 2))
            new_strategy.temperature = max(0.0, round(new_strategy.temperature - 0.05, 2))
            instructions.append("Strictly adhere to the retrieved facts. Remove unverified claims.")
            if report.unsupported_claims:
                instructions.append(f"Remove unsupported claims: {', '.join(report.unsupported_claims)}.")

        # 2. Completeness Failure Recovery
        if report.completeness < 0.65 or report.missing_aspects:
            new_strategy.dense_top_k = strategy.dense_top_k + 10
            new_strategy.bm25_top_k = strategy.bm25_top_k + 10
            new_strategy.rerank_top_n = min(15, strategy.rerank_top_n + 3)
            new_strategy.enable_multi_query = True
            new_strategy.enable_parent_expansion = True
            if report.missing_aspects:
                instructions.append(f"Specifically address: {', '.join(report.missing_aspects)}.")

        # 3. Citation Coverage Failure Recovery
        if report.citation_coverage < 0.50:
            instructions.append("Attach explicit [Source N] citation brackets for each substantive factual statement.")

        # 4. Coherence Failure Recovery
        if report.coherence < 0.70:
            instructions.append("Ensure clear logical flow, structured formatting, and complete sentences.")

        if not instructions:
            instructions.append("Strictly adhere to the retrieved facts and ensure all questions are answered with [Source N] citations.")

        prompt_refinement = " ".join(instructions)
        logger.info(
            "Retry prepared for attempt %d/%d: dense_top_k=%d, bm25_top_k=%d, min_score_ratio=%.2f. Refinement: %s",
            attempt + 1,
            self.max_retries,
            new_strategy.dense_top_k,
            new_strategy.bm25_top_k,
            new_strategy.min_score_ratio,
            prompt_refinement,
        )
        return new_strategy, prompt_refinement
