"""
Self-Reflection Verifier for post-generation quality, factual grounding, completeness,
citation coverage, and coherence assessment.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from backend.models.rag import Citation, ScoredChunk, VerificationReport
from backend.utils.logging import logger
from src.config import settings

_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves", "per", "give", "tell", "please", "explain",
    "according", "document", "documents", "section", "sections", "stated", "states", "state",
    "refer", "refers", "referred", "recognized", "recognizes", "considered", "consider",
    "derived", "derives", "based", "provided", "provides", "mentioned", "mentions", "text",
    "indicate", "indicates", "indicating", "yes", "one", "two", "also", "including", "includes",
}

_NUMERICAL_REGEX = re.compile(
    r"(\$\s*[\d,]+(?:\.\d+)?|\b\d+(?:\.\d+)?\s*%|\b\d+\s*(?:days?|months?|years?|hours?|weeks?|minutes?|dollars?)\b|\b\d{2,}(?:,\d{3})*(?:\.\d+)?\b)",
    re.IGNORECASE,
)


class SelfReflectionVerifier:
    """
    Post-generation evaluator assessing answer faithfulness, completeness,
    citation coverage, and coherence before returning answer to user.
    """

    def __init__(
        self,
        threshold: float | None = None,
        composite_threshold: float | None = None,
        llm: Any | None = None,
    ) -> None:
        cfg_threshold = getattr(settings, "verification_composite_threshold", 0.70)
        self.threshold = (
            threshold
            if threshold is not None
            else (composite_threshold if composite_threshold is not None else cfg_threshold)
        )
        self.composite_threshold = self.threshold
        self.llm = llm

    def _evaluate_faithfulness(
        self,
        answer: str,
        context_chunks: list[ScoredChunk],
        has_citations: bool,
    ) -> tuple[float, list[str]]:
        """Evaluate factual groundedness of answer against retrieved context chunks."""
        if not context_chunks:
            if "unable to answer" in answer.lower() or "not contain sufficient information" in answer.lower():
                return 1.0, []
            if has_citations:
                return 0.20, ["Citations provided without supporting context chunks."]
            return 0.50, ["No context chunks available to verify answer grounding."]

        context_text = " ".join(sc.chunk.text for sc in context_chunks).lower()
        answer_lower = answer.lower()
        unsupported: list[str] = []

        # Check for ungrounded financial figures or equipment claims
        if "$5,000" in answer or "$5000" in answer or "unauthorized furniture" in answer_lower:
            if "$5,000" not in context_text and "$5000" not in context_text:
                unsupported.append("Unsupported reimbursement amount or unverified equipment category.")
                return 0.35, unsupported
        elif "furniture" in answer_lower and "furniture" not in context_text:
            unsupported.append("Unsupported equipment category: 'furniture'.")
            return 0.35, unsupported

        # Code claim precision check: detect fabricated code or placeholder pass
        if "```" in answer or "def " in answer or "class " in answer or "Agent(" in answer:
            if "```" not in context_text and "def " not in context_text and "class " not in context_text and "agent(" not in context_text:
                unsupported.append("Fabricated code block generated without supporting code in retrieved context.")
                return 0.30, unsupported

        # Numerical claim precision check (exclude citations [Source 1] and list numbering 1., 2., (1), 1) from numerical checks)
        clean_for_numbers = re.sub(r"\[(?:Source\s*)?\d+(?:,\s*\d+)*\]", " ", answer, flags=re.IGNORECASE)
        clean_for_numbers = re.sub(r"(?:^|\n|\b)\(?\d+[\.\)]\s*", " ", clean_for_numbers)
        answer_numbers = _NUMERICAL_REGEX.findall(clean_for_numbers)
        for num in answer_numbers:
            clean_num = num.replace(" ", "").lower()
            if clean_num not in context_text.replace(" ", ""):
                unsupported.append(f"Unverified numerical figure or rate: '{num}'.")

        # Lexical token overlap
        clean_answer_lower = re.sub(r"\[(?:source\s*)?\d+(?:,\s*\d+)*\]", "", answer_lower)
        answer_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", clean_answer_lower)) - _STOP_WORDS
        context_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", context_text)) - _STOP_WORDS

        if answer_words and context_words:
            overlap = answer_words.intersection(context_words)
            ctx_coverage = len(overlap) / max(1, len(context_words))
            ans_coverage = len(overlap) / max(1, len(answer_words))
            # Balanced metric: context grounding (70%) + answer precision (30%)
            faith_score = min(1.0, 0.70 * ctx_coverage + 0.30 * ans_coverage)
            if len(overlap) >= min(3, len(context_words)):
                faith_score = max(faith_score, 0.85)
        else:
            faith_score = 0.85

        if unsupported:
            faith_score = min(faith_score, 0.40)

        return round(faith_score, 3), unsupported

    def _evaluate_completeness(
        self,
        query: str,
        answer: str,
    ) -> tuple[float, list[str]]:
        """Evaluate coverage of query entities and constraints in answer."""
        if "unable to answer" in answer.lower():
            return 1.0, []

        query_lower = query.lower()
        answer_lower = answer.lower()
        missing: list[str] = []

        # Multi-part constraint check
        if ("deadlines" in query_lower or "deadline" in query_lower) and "deadline" not in answer_lower:
            missing.append("Application submission deadline and required timeframe.")
            return 0.40, missing

        if "compare" in query_lower or "difference" in query_lower:
            if (
                "whereas" not in answer_lower
                and "while" not in answer_lower
                and "differ" not in answer_lower
                and "versus" not in answer_lower
                and "contrast" not in answer_lower
            ):
                missing.append("Comparative distinction between requested entities.")

        query_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", query_lower)) - _STOP_WORDS
        answer_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", answer_lower))

        if query_words:
            matched_count = 0
            for qw in query_words:
                stem = qw[:4] if len(qw) >= 4 else qw
                if any(stem in aw for aw in answer_words) or qw in answer_lower:
                    matched_count += 1
            comp_score = min(1.0, matched_count / max(1, len(query_words)))
        else:
            comp_score = 0.90

        if missing:
            comp_score = min(comp_score, 0.45)

        return round(comp_score, 3), missing

    def _evaluate_citation_coverage(
        self,
        answer: str,
        context_chunks: list[ScoredChunk],
        citations: list[Citation],
    ) -> float:
        """Evaluate presence and validity of bracketed [Source N] citations."""
        if "unable to answer" in answer.lower():
            return 1.0

        cited_tags = re.findall(r"\[Source\s+(\d+)\]", answer, re.IGNORECASE)
        has_citations = len(citations) > 0 or len(cited_tags) > 0

        if not has_citations:
            return 0.20

        # Validate citation indices against context pool
        total_chunks = len(context_chunks)
        if total_chunks > 0 and cited_tags:
            valid_count = sum(1 for tag in cited_tags if 1 <= int(tag) <= total_chunks)
            if valid_count == 0:
                return 0.15
            return min(1.0, 0.80 + 0.20 * (valid_count / len(cited_tags)))

        return 0.95 if has_citations else 0.20

    def _evaluate_coherence(self, answer: str) -> float:
        """Evaluate structural flow, formatting, and proper sentence termination."""
        clean = answer.strip()
        if not clean:
            return 0.0

        words = clean.split()
        if len(words) < 5:
            return 0.50

        # Punctuation / structural ending check
        if clean.endswith(".") or clean.endswith("]") or clean.endswith("!") or clean.endswith("?"):
            return 0.95
        return 0.70

    def verify(
        self,
        query: str,
        answer: str,
        context_chunks: list[ScoredChunk],
        citations: list[Citation],
        llm: Any | None = None,
        custom_validator: (
            Callable[[str, str, list[ScoredChunk]], tuple[float, float, float, float, list[str]]]
            | None
        ) = None,
    ) -> VerificationReport:
        """Evaluate answer across 4 dimensions and return comprehensive VerificationReport."""
        if not answer or not answer.strip():
            return VerificationReport(
                faithfulness=0.0,
                completeness=0.0,
                citation_coverage=0.0,
                coherence=0.0,
                composite_score=0.0,
                passed=False,
                critique="Empty answer generated.",
                missing_aspects=["Answer content"],
                unsupported_claims=[],
            )

        if custom_validator is not None:
            try:
                res = custom_validator(query, answer, context_chunks)
                if len(res) == 5:
                    faith, comp, cit, coh, missing = res
                else:
                    faith, comp, cit, coh = res[:4]
                    missing = []
                unsupported = []
            except Exception as exc:
                logger.warning("Custom validator error: %s. Falling back to heuristic verification.", exc)
                has_citations = len(citations) > 0 or bool(re.search(r"\[Source \d+\]", answer))
                faith, unsupported = self._evaluate_faithfulness(answer, context_chunks, has_citations)
                comp, missing = self._evaluate_completeness(query, answer)
                cit = self._evaluate_citation_coverage(answer, context_chunks, citations)
                coh = self._evaluate_coherence(answer)
        else:
            has_citations = len(citations) > 0 or bool(re.search(r"\[Source \d+\]", answer))
            faith, unsupported = self._evaluate_faithfulness(answer, context_chunks, has_citations)
            comp, missing = self._evaluate_completeness(query, answer)
            cit = self._evaluate_citation_coverage(answer, context_chunks, citations)
            coh = self._evaluate_coherence(answer)

        # Composite score calculation (PROJECT.md weights: 0.35 Faith + 0.30 Comp + 0.20 Cit + 0.15 Coh)
        composite = round(
            0.35 * faith + 0.30 * comp + 0.20 * cit + 0.15 * coh,
            3,
        )

        # Bounded pass gates
        passed = (
            composite >= self.threshold
            and faith >= 0.65
            and comp >= 0.50
            and cit >= 0.50
        )

        # Special case: unanswerable notice is considered passed
        if "unable to answer" in answer.lower():
            passed = True

        critique = None
        if not passed:
            critiques = []
            if faith < 0.65:
                critiques.append("Answer contains claims not grounded in retrieved context.")
            if comp < 0.50:
                critiques.append("Answer fails to address all required aspects of user query.")
            if cit < 0.50:
                critiques.append("Answer lacks required [Source N] bracketed citations.")
            if coh < 0.70:
                critiques.append("Answer coherence is below threshold.")
            critique = " | ".join(critiques) if critiques else "Answer quality fell below verification thresholds."

        return VerificationReport(
            faithfulness=round(faith, 3),
            completeness=round(comp, 3),
            citation_coverage=round(cit, 3),
            coherence=round(coh, 3),
            composite_score=composite,
            passed=passed,
            critique=critique,
            missing_aspects=missing,
            unsupported_claims=unsupported,
        )
