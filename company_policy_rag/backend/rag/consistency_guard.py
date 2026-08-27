from __future__ import annotations

from typing import Any

from backend.models.rag import Citation, EvidenceStatus, ScoredChunk
from backend.utils.logging import logger


class ConversationConsistencyGuard:
    """
    Monotonic evidence consistency and downgrade protection engine.

    Prevents contradictory downgrades (e.g. Turn 1: Direct evidence -> Turn 2: 'I could not find this information')
    by enforcing evidence continuity and status monotonicity across multi-turn conversations.
    """

    @staticmethod
    def enforce_downgrade_protection(
        previous_status: EvidenceStatus | str | None,
        previous_chunks: list[ScoredChunk] | None,
        previous_citations: list[Citation] | None,
        current_status: EvidenceStatus | str,
        current_chunks: list[ScoredChunk],
        is_followup: bool,
    ) -> tuple[EvidenceStatus, list[ScoredChunk], list[Citation], bool]:
        """
        Enforce monotonic evidence status and preserve valid prior evidence when current retrieval is weak.

        Returns:
            tuple[EvidenceStatus, list[ScoredChunk], list[Citation], bool]:
            - effective_evidence_status
            - merged_context_chunks
            - preserved_citations
            - continuity_applied
        """
        prev_chunks = list(previous_chunks or [])
        prev_cits = list(previous_citations or [])
        curr_chunks = list(current_chunks or [])

        # Parse enums
        curr_enum = (
            current_status
            if isinstance(current_status, EvidenceStatus)
            else EvidenceStatus(str(current_status).upper())
            if str(current_status).upper() in EvidenceStatus.__members__
            else EvidenceStatus.MISSING
        )

        if not is_followup or previous_status is None:
            return curr_enum, curr_chunks, [], False

        prev_enum = (
            previous_status
            if isinstance(previous_status, EvidenceStatus)
            else EvidenceStatus(str(previous_status).upper())
            if str(previous_status).upper() in EvidenceStatus.__members__
            else EvidenceStatus.MISSING
        )

        has_prev_evidence = bool(prev_chunks)
        has_curr_evidence = bool(curr_chunks)

        # Merge candidate pools with deduplication by chunk ID
        dedup_map: dict[str, ScoredChunk] = {}
        for sc in curr_chunks:
            dedup_map[sc.chunk.id] = sc
        for sc in prev_chunks:
            if sc.chunk.id not in dedup_map:
                dedup_map[sc.chunk.id] = sc
        merged_chunks = list(dedup_map.values())

        continuity_applied = bool(
            has_prev_evidence
            and (
                len(merged_chunks) > len(curr_chunks)
                or not has_curr_evidence
                or is_followup
            )
        )

        # Monotonic State Machine Calculation
        if prev_enum == EvidenceStatus.DIRECT:
            effective_status = EvidenceStatus.DIRECT if has_prev_evidence else curr_enum
        elif prev_enum == EvidenceStatus.PARTIAL:
            if curr_enum == EvidenceStatus.DIRECT:
                effective_status = EvidenceStatus.DIRECT
            else:
                effective_status = EvidenceStatus.PARTIAL if has_prev_evidence else curr_enum
        elif prev_enum == EvidenceStatus.RELATED:
            if curr_enum in (EvidenceStatus.DIRECT, EvidenceStatus.PARTIAL):
                effective_status = curr_enum
            else:
                effective_status = EvidenceStatus.RELATED if has_prev_evidence else curr_enum
        else:
            effective_status = curr_enum

        logger.info(
            "[CONSISTENCY_GUARD] prev_status=%s curr_status=%s effective_status=%s "
            "prev_chunks=%d curr_chunks=%d merged_chunks=%d continuity_applied=%s",
            prev_enum.value,
            curr_enum.value,
            effective_status.value,
            len(prev_chunks),
            len(curr_chunks),
            len(merged_chunks),
            continuity_applied,
        )

        return effective_status, merged_chunks, prev_cits, continuity_applied

    @staticmethod
    def format_monotonic_prompt_directive(
        is_followup: bool,
        retained_prior_evidence: bool,
        additional_evidence_found: bool,
        previous_topic: str | None = None,
    ) -> str:
        """
        Generate prompt grounding directives to prevent false absence claims.
        """
        if not is_followup or not retained_prior_evidence:
            return ""

        if not additional_evidence_found:
            return (
                "CONVERSATION CONTINUITY DIRECTIVE:\n"
                f"- This is a follow-up query regarding '{previous_topic or 'the active subject'}'.\n"
                "- Continue answering and expanding from the previously verified evidence provided below.\n"
                "- DO NOT state that the information cannot be found or that the document does not contain it.\n"
                "- If no new additional sections were retrieved beyond prior context, clearly explain from the existing evidence."
            )
        return (
            "CONVERSATION CONTINUITY DIRECTIVE:\n"
            f"- This is a follow-up query expanding on '{previous_topic or 'the active subject'}'.\n"
            "- Integrate previously verified facts with newly retrieved evidence into a comprehensive explanation."
        )
