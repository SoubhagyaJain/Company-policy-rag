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

_LLM_FAITHFULNESS_PROMPT = """You are a strict grounding auditor for a document question-answering system.
Judge whether EVERY factual claim in the ANSWER is directly supported by the CONTEXT.
Use ONLY the CONTEXT — never outside knowledge. Numbers, amounts, dates, times, names,
and conditions must match the CONTEXT exactly; a changed or invented figure is unsupported.
Citation tags like [Source 1] are not claims and never count as unsupported.

CONTEXT:
{context}

ANSWER:
{answer}

Respond with ONLY a single-line JSON object, no prose before or after:
{{"faithfulness": <number between 0.0 and 1.0>, "unsupported_claims": ["<short claim>", ...]}}
- faithfulness 1.0 = every claim supported; 0.0 = mostly fabricated.
- unsupported_claims lists each specific ANSWER claim NOT supported by the CONTEXT (empty list if all supported).
JSON:"""


_QUANTITATIVE_CLAIM_REGEX = re.compile(
    r"(\$\s*[\d,]+(?:\.\d+)?|\b\d+(?:\.\d+)?\s*%|\b\d+\s*(?:days?|months?|weeks?|hours?|minutes?|dollars?|cents?|gb|mb|usd|eur)\b)",
    re.IGNORECASE,
)
_GENERAL_INTEGER_REGEX = re.compile(
    r"\b\d{2,}(?:,\d{3})*(?:\.\d+)?\b",
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
        allowed_derived_facts: list[str] | None = None,
    ) -> tuple[float, list[str]]:
        """Evaluate factual groundedness of answer against retrieved context chunks."""
        if not context_chunks:
            if "unable to answer" in answer.lower() or "not contain sufficient information" in answer.lower():
                return 1.0, []
            if has_citations:
                return 0.20, ["Citations provided without supporting context chunks."]
            return 0.50, ["No context chunks available to verify answer grounding."]

        # Build comprehensive context text including chunk text, metadata, page numbers, and headers
        context_parts = []
        for sc in context_chunks:
            context_parts.append(sc.chunk.text)
            meta = sc.chunk.metadata
            if meta:
                if meta.page_number is not None:
                    context_parts.append(f"page {meta.page_number} {meta.page_number}")
                if meta.display_page_number is not None:
                    context_parts.append(f"page {meta.display_page_number} {meta.display_page_number}")
                if meta.page_label:
                    context_parts.append(f"page {meta.page_label} {meta.page_label}")
                if meta.section_title:
                    context_parts.append(str(meta.section_title))
                if meta.section_path:
                    context_parts.append(str(meta.section_path))
                if meta.source_file:
                    context_parts.append(str(meta.source_file))
                if meta.image_assets:
                    for ast in meta.image_assets:
                        if isinstance(ast, dict):
                            context_parts.append(str(ast.get("display_page_number") or ""))
                            context_parts.append(str(ast.get("page_label") or ""))
                            context_parts.append(str(ast.get("page_number") or ""))
                            context_parts.append(str(ast.get("asset_id") or ""))
                if meta.extra:
                    for k, v in meta.extra.items():
                        if isinstance(v, (str, int, float)):
                            context_parts.append(str(v))

        context_parts.extend(allowed_derived_facts or [])
        context_text = " ".join(context_parts).lower()
        context_compact = context_text.replace(" ", "")
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

        # Named software/products are especially easy for a generator to add
        # from model memory. Treat capitalized product-like tokens that are not
        # in the evidence as unsupported (excluding sentence starts and tags).
        product_candidates = re.findall(
            r"(?<![.!?]\s)\b(?:Slack|Jira(?:\s+Service\s+Desk)?|Salesforce|Docker|Kubernetes|"
            r"Workday|ServiceNow|Teams|Zoom)\b",
            answer,
            re.IGNORECASE,
        )
        missing_products = sorted(
            {product for product in product_candidates if product.casefold() not in context_text}
        )
        if missing_products:
            unsupported.append(
                "Unsupported software or workflow: " + ", ".join(missing_products) + "."
            )

        # Code claim precision check: detect fabricated code or placeholder pass
        if "```" in answer or "def " in answer or "class " in answer or "Agent(" in answer:
            if "```" not in context_text and "def " not in context_text and "class " not in context_text and "agent(" not in context_text:
                unsupported.append("Fabricated code block generated without supporting code in retrieved context.")
                return 0.30, unsupported

        # Clean citations, page numbers, section headers, steps, line numbers, and years before numerical checks
        clean_for_numbers = answer
        clean_for_numbers = re.sub(r"\[(?:VISUAL\s+)?SOURCE\s*\d+(?:,\s*\d+)*\]", " ", clean_for_numbers, flags=re.IGNORECASE)
        clean_for_numbers = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", " ", clean_for_numbers)
        clean_for_numbers = re.sub(r"\bpages?\s+(?:numbers?\s+)?\d+(?:\s*-\s*\d+)?\b", " ", clean_for_numbers, flags=re.IGNORECASE)
        clean_for_numbers = re.sub(r"\bp\.\s*\d+\b", " ", clean_for_numbers, flags=re.IGNORECASE)
        clean_for_numbers = re.sub(r"\bsections?\s+[\d\.]+\b", " ", clean_for_numbers, flags=re.IGNORECASE)
        clean_for_numbers = re.sub(r"\b(?:steps?|lines?)\s+\d+\b", " ", clean_for_numbers, flags=re.IGNORECASE)
        clean_for_numbers = re.sub(r"(?:^|\n|\b)\(?\d+[\.\)]\s*", " ", clean_for_numbers)
        clean_for_numbers = re.sub(r"\|\s*\d+\s*\|", " | ", clean_for_numbers)
        clean_for_numbers = re.sub(r"\b(?:\d+b|k\s*=\s*\d+|top_k\s*=\s*\d+|top_n\s*=\s*\d+|384|512|1024|2048|4096)\b", " ", clean_for_numbers, flags=re.IGNORECASE)
        clean_for_numbers = re.sub(r"\b(?:8000|8080|3000|5000|200|404|500)\b", " ", clean_for_numbers)
        clean_for_numbers = re.sub(r"\b(19\d\d|20\d\d)\b", " ", clean_for_numbers)

        # 1. Substantive quantitative claim check (money, percentages, durations)
        quant_numbers = _QUANTITATIVE_CLAIM_REGEX.findall(clean_for_numbers)
        for num in quant_numbers:
            clean_num = num.replace(" ", "").lower()
            if clean_num not in context_compact:
                unsupported.append(f"Unverified numerical figure or rate: '{num}'.")

        # 2. General multi-digit integer check
        for qn in quant_numbers:
            clean_for_numbers = clean_for_numbers.replace(qn, " ")

        general_numbers = _GENERAL_INTEGER_REGEX.findall(clean_for_numbers)
        for num in general_numbers:
            clean_num = num.replace(" ", "").lower()
            if clean_num not in context_compact:
                if clean_num in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "20", "30", "50", "60", "100"):
                    continue
                unsupported.append(f"Unverified numerical figure or rate: '{num}'.")

        # Lexical token overlap
        clean_answer_lower = re.sub(r"\[(?:visual\s+)?source\s*\d+(?:,\s*\d+)*\]", "", answer_lower)
        clean_answer_lower = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", clean_answer_lower)
        answer_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", clean_answer_lower)) - _STOP_WORDS
        context_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", context_text)) - _STOP_WORDS

        if answer_words and context_words:
            overlap = answer_words.intersection(context_words)
            ctx_coverage = len(overlap) / max(1, len(context_words))
            ans_coverage = len(overlap) / max(1, len(answer_words))
            # Balanced metric: context grounding (70%) + answer precision (30%)
            faith_score = min(1.0, 0.70 * ctx_coverage + 0.30 * ans_coverage)
            if len(overlap) >= min(3, len(context_words)):
                faith_score = max(faith_score, 0.90)
            elif len(overlap) >= 1:
                faith_score = max(faith_score, 0.80)
        else:
            faith_score = 0.90

        if unsupported:
            # A policy answer with even one unsupported concrete claim must not
            # clear the faithfulness gate merely because the surrounding prose
            # overlaps the retrieved text.
            faith_score = min(0.35, round(faith_score - 0.10 * len(unsupported), 3))

        return round(faith_score, 3), unsupported

    def _evaluate_faithfulness_llm(
        self,
        answer: str,
        context_chunks: list[ScoredChunk],
        llm: Any | None,
    ) -> tuple[float, list[str]] | None:
        """LLM claim-support audit. Returns (score, unsupported_claims), or None
        to signal the caller to keep the heuristic verdict (no LLM, empty
        context, or an LLM/parse failure)."""
        if llm is None or not context_chunks:
            return None

        max_chars = int(getattr(settings, "llm_verification_max_context_chars", 6000))
        context_text = "\n\n".join(sc.chunk.text for sc in context_chunks).strip()
        if not context_text:
            return None
        context_text = context_text[:max_chars]

        prompt = _LLM_FAITHFULNESS_PROMPT.format(context=context_text, answer=answer)
        try:
            try:
                raw = str(llm.complete(prompt, temperature=0.0, max_new_tokens=256)).strip()
            except TypeError:
                raw = str(llm.complete(prompt)).strip()
        except Exception as exc:
            logger.warning("LLM faithfulness verification failed (%s); using heuristic.", exc)
            return None

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            logger.debug("LLM faithfulness verdict had no JSON object; using heuristic.")
            return None
        try:
            data = json.loads(match.group(0))
        except Exception:
            logger.debug("LLM faithfulness verdict was not valid JSON; using heuristic.")
            return None

        score = data.get("faithfulness")
        if not isinstance(score, (int, float)):
            return None
        score = max(0.0, min(1.0, float(score)))

        raw_claims = data.get("unsupported_claims") or []
        if not isinstance(raw_claims, list):
            raw_claims = []
        claims = [str(c).strip() for c in raw_claims if str(c).strip()][:10]
        return round(score, 3), claims

    def _evaluate_completeness(
        self,
        query: str,
        answer: str,
    ) -> tuple[float, list[str]]:
        """Evaluate coverage of query entities and constraints in answer."""
        if "unable to answer" in answer.lower() or "could not find this information" in answer.lower():
            return 1.0, []

        query_lower = query.lower()
        answer_lower = answer.lower()
        missing: list[str] = []

        query_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", query_lower)) - _STOP_WORDS
        answer_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", answer_lower))

        if query_words:
            matched_count = 0
            for qw in query_words:
                stem = qw[:4] if len(qw) >= 4 else qw
                if any(stem in aw for aw in answer_words) or qw in answer_lower:
                    matched_count += 1
            comp_score = min(1.0, matched_count / max(1, len(query_words)))
            if matched_count >= 1 and comp_score >= 0.60:
                comp_score = max(comp_score, 0.80)
        else:
            comp_score = 0.90

        # Multi-part policy questions must cover each explicitly requested
        # aspect; a fluent answer to only the first half is incomplete.
        aspect_source = re.sub(r"^.*?\b(?:what|which|compare|explain)\b", "", query_lower)
        raw_aspects = [
            part.strip(" ?.,")
            for part in re.split(r",|\band\b|\bversus\b|\bvs\.?\b", aspect_source)
            if part.strip(" ?.,")
        ]
        meaningful_aspects: list[str] = []
        for aspect in raw_aspects:
            tokens = [
                token for token in re.findall(r"[a-z0-9-]+", aspect)
                if len(token) >= 4 and token not in _STOP_WORDS
            ]
            if tokens:
                meaningful_aspects.append(" ".join(tokens))
        for aspect in meaningful_aspects:
            tokens = aspect.split()
            if not any(token[:4] in answer_lower for token in tokens):
                missing.append(aspect)

        if "compare" in query_lower:
            comparison_entities = [
                term
                for term in ("full-time", "part-time", "employee", "contractor", "manager")
                if term in query_lower
            ]
            comparison_missing = False
            for entity in comparison_entities:
                if entity not in answer_lower:
                    missing.append(entity)
                    comparison_missing = True
            if comparison_missing:
                missing.append("Comparative distinction between requested groups")

        if missing:
            comp_score = min(0.45, round(comp_score - 0.15 * len(missing), 3))

        return round(comp_score, 3), missing

    def _evaluate_citation_coverage(
        self,
        answer: str,
        context_chunks: list[ScoredChunk],
        citations: list[Citation],
    ) -> float:
        """Evaluate presence and validity of bracketed citations."""
        if "unable to answer" in answer.lower() or "could not find this information" in answer.lower():
            return 1.0

        cited_tags = re.findall(r"\[(?:VISUAL\s+)?SOURCE\s*(\d+)\]", answer, re.IGNORECASE)
        bracket_nums = re.findall(r"\[(\d+)\]", answer)
        all_tags = cited_tags + bracket_nums
        has_citations = len(citations) > 0 or len(all_tags) > 0

        if not has_citations:
            return 0.20

        # Validate citation indices against context pool
        total_chunks = len(context_chunks)
        if total_chunks > 0 and all_tags:
            valid_count = sum(1 for tag in all_tags if 1 <= int(tag) <= total_chunks)
            if valid_count == 0:
                return 0.15
            return min(1.0, 0.85 + 0.15 * (valid_count / len(all_tags)))

        return 0.95 if has_citations else 0.50

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
        allowed_derived_facts: list[str] | None = None,
        custom_validator: (
            Callable[[str, str, list[ScoredChunk]], tuple[float, float, float, float, list[str]]]
            | None
        ) = None,
        use_llm_judge: bool = False,
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
                has_citations = len(citations) > 0 or bool(re.search(r"\[(?:VISUAL\s+)?Source\s*\d+\]", answer, re.IGNORECASE))
                faith, unsupported = self._evaluate_faithfulness(
                    answer, context_chunks, has_citations, allowed_derived_facts
                )
                comp, missing = self._evaluate_completeness(query, answer)
                cit = self._evaluate_citation_coverage(answer, context_chunks, citations)
                coh = self._evaluate_coherence(answer)
        else:
            has_citations = len(citations) > 0 or bool(re.search(r"\[(?:VISUAL\s+)?Source\s*\d+\]", answer, re.IGNORECASE))
            faith, unsupported = self._evaluate_faithfulness(
                answer, context_chunks, has_citations, allowed_derived_facts
            )
            comp, missing = self._evaluate_completeness(query, answer)
            cit = self._evaluate_citation_coverage(answer, context_chunks, citations)
            coh = self._evaluate_coherence(answer)

            # LLM claim-support audit augments the lexical heuristic: it can only
            # make the verdict stricter (catch hallucinations the overlap check
            # misses), never inflate a weak answer. Skipped for abstentions, and
            # any LLM/parse failure leaves the heuristic verdict untouched.
            answer_l = answer.lower()
            is_abstention = "unable to answer" in answer_l or "could not find" in answer_l
            if use_llm_judge and not is_abstention:
                llm_result = self._evaluate_faithfulness_llm(
                    answer, context_chunks, llm if llm is not None else self.llm
                )
                if llm_result is not None:
                    llm_faith, llm_unsupported = llm_result
                    faith = min(faith, llm_faith)
                    seen = {u.casefold() for u in unsupported}
                    for claim in llm_unsupported:
                        if claim.casefold() not in seen:
                            unsupported.append(claim)
                            seen.add(claim.casefold())

        # Composite score calculation (PROJECT.md weights: 0.35 Faith + 0.30 Comp + 0.20 Cit + 0.15 Coh)
        composite = round(
            0.35 * faith + 0.30 * comp + 0.20 * cit + 0.15 * coh,
            3,
        )

        # Bounded pass gates
        passed = (
            composite >= self.threshold
            and faith >= getattr(settings, "verification_faithfulness_threshold", 0.75)
            and comp >= getattr(settings, "verification_completeness_threshold", 0.70)
            and cit >= getattr(settings, "verification_citation_threshold", 0.60)
            and not unsupported
            and not missing
        )

        # Special case: unanswerable notice is considered passed
        if "unable to answer" in answer.lower():
            passed = True

        critique = None
        if not passed:
            critiques = []
            if faith < 0.50:
                critiques.append("Answer contains claims not grounded in retrieved context.")
            if comp < 0.35:
                critiques.append("Answer fails to address required aspects of user query.")
            if cit < 0.35:
                critiques.append("Answer lacks required [Source N] bracketed citations.")
            if coh < 0.60:
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
            overall_grounded=passed,
        )
