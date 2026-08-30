from __future__ import annotations

import re
from dataclasses import dataclass, field
from backend.models.logical_document import detect_continuation_signals
from backend.models.rag import EvidenceStatus, QueryCategory, ScoredChunk
from backend.utils.logging import logger

_CODE_DETECTION_REGEX = re.compile(
    r"(?:```(?:python|bash|sh|json|yaml|yml|javascript|ts|js)?\s*[\s\S]*?```|"
    r"(?:def\s+\w+\s*\(|class\s+\w+|import\s+\w+|from\s+\w+\s+import|"
    r"Agent\s*\(|Task\s*\(|Crew\s*\(|LLM\s*\(|SequentialProcess|"
    r"\w+\s*=\s*(?:Agent|Task|Crew|crew\.kickoff|kickoff)\s*\(|"
    r"crew\.kickoff|result\s*=\s*|\.kickoff\s*\())",
    re.MULTILINE,
)

_WORKFLOW_DIAGRAM_REGEX = re.compile(
    r"(?:workflow|architecture|diagram|flowchart|flow|pipeline|system\s+overview|process\s+flow|interaction\s+diagram)",
    re.IGNORECASE,
)

_IMPLEMENTATION_INTENTS = {
    "implementation",
    "code",
    "procedural",
    "build",
    "make",
    "develop",
    "construct",
}

_VISUAL_INTENTS = {
    "architecture",
    "diagram",
    "workflow",
    "flowchart",
}

_NUMBER_WORDS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _referenced_enumeration_is_complete(candidate_chunks: list[ScoredChunk]) -> bool:
    """Return true when continuation text contains every item promised by a visual cue.

    Some PDFs render a summary infographic on one page and repeat its labels as
    searchable numbered text on the following pages. In that case the text is
    direct evidence and a query-time vision call is unnecessary.
    """
    ordered = sorted(
        candidate_chunks,
        key=lambda sc: (
            sc.chunk.metadata.document_id,
            sc.chunk.metadata.page_number or 0,
            sc.chunk.id,
        ),
    )
    combined = "\n".join(sc.chunk.text for sc in ordered)
    cue_match = re.search(
        r"\b(?P<count>\d+|two|three|four|five|six|seven|eight|nine|ten)\b"
        r".{0,80}\b(?:techniques?|methods?|steps?|types?|ways?|items?)\b"
        r".{0,120}\b(?:depicted|illustrated|shown|listed|presented)\b",
        combined,
        re.IGNORECASE | re.DOTALL,
    )
    if not cue_match:
        return False

    raw_count = cue_match.group("count").lower()
    expected = int(raw_count) if raw_count.isdigit() else _NUMBER_WORDS.get(raw_count, 0)
    if expected < 2:
        return False

    numbered_items = {
        int(value)
        for value in re.findall(r"(?m)^\s*(\d{1,2})\s*[.)]\s*\S", combined)
    }
    return all(item in numbered_items for item in range(1, expected + 1))


@dataclass
class EvidenceSufficiencyResult:
    is_sufficient: bool
    missing_evidence_types: list[str] = field(default_factory=list)
    detected_continuation_cues: list[str] = field(default_factory=list)
    pages_to_inspect: list[int] = field(default_factory=list)
    anchor_chunk: ScoredChunk | None = None
    visual_asset_available: bool = False
    vision_understanding_available: bool = False
    evidence_status: EvidenceStatus = EvidenceStatus.DIRECT
    reason: str = "Evidence is sufficient for query intent."


def compute_monotonic_evidence_status(
    previous_status: EvidenceStatus | str | None,
    current_status: EvidenceStatus | str,
    has_prev_evidence: bool = False,
    has_curr_evidence: bool = False,
    is_followup: bool = False,
) -> EvidenceStatus:
    """
    Enforce evidence status monotonicity for multi-turn conversational follow-ups.
    Ensures DIRECT or PARTIAL evidence from a prior turn is NEVER downgraded to MISSING
    merely because a follow-up query retrieval returned weak or zero new chunks.
    """
    curr_enum = (
        current_status
        if isinstance(current_status, EvidenceStatus)
        else EvidenceStatus(str(current_status).upper())
        if str(current_status).upper() in EvidenceStatus.__members__
        else EvidenceStatus.MISSING
    )

    if not is_followup or previous_status is None:
        return curr_enum

    prev_enum = (
        previous_status
        if isinstance(previous_status, EvidenceStatus)
        else EvidenceStatus(str(previous_status).upper())
        if str(previous_status).upper() in EvidenceStatus.__members__
        else EvidenceStatus.MISSING
    )

    # If neither previous nor current evidence exists, status is MISSING
    if not has_prev_evidence and not has_curr_evidence and curr_enum == EvidenceStatus.MISSING:
        return EvidenceStatus.MISSING

    # Rule: DIRECT + relevant follow-up evidence -> DIRECT
    if prev_enum == EvidenceStatus.DIRECT:
        if curr_enum in (EvidenceStatus.DIRECT, EvidenceStatus.PARTIAL, EvidenceStatus.RELATED):
            return EvidenceStatus.DIRECT
        return EvidenceStatus.DIRECT if has_prev_evidence else curr_enum

    # Rule: PARTIAL + additional evidence -> DIRECT or PARTIAL
    if prev_enum == EvidenceStatus.PARTIAL:
        if curr_enum == EvidenceStatus.DIRECT:
            return EvidenceStatus.DIRECT
        if has_prev_evidence:
            return EvidenceStatus.PARTIAL
        return curr_enum

    # Rule: RELATED + additional evidence -> RELATED or PARTIAL
    if prev_enum == EvidenceStatus.RELATED:
        if curr_enum in (EvidenceStatus.DIRECT, EvidenceStatus.PARTIAL):
            return curr_enum
        return EvidenceStatus.RELATED if has_prev_evidence else curr_enum

    return curr_enum


class EvidenceSufficiencyGate:
    """
    Pre-generation evaluation gate verifying that retrieved evidence is sufficient
    to answer the user query faithfully according to query intent before invoking LLM synthesis.
    Distinguishes TEXT_EVIDENCE, VISUAL_ASSET_AVAILABLE, and VISION_UNDERSTANDING_AVAILABLE.
    """

    def __init__(self, max_continuation_depth: int = 3) -> None:
        self.max_continuation_depth = max_continuation_depth

    def evaluate(
        self,
        query: str,
        intent: str | QueryCategory,
        candidate_chunks: list[ScoredChunk],
        previous_status: EvidenceStatus | str | None = None,
        previous_chunks: list[ScoredChunk] | None = None,
        is_followup: bool = False,
    ) -> EvidenceSufficiencyResult:
        intent_str = intent.value.lower() if isinstance(intent, QueryCategory) else str(intent).lower()
        query_lower = query.lower()

        # Check if query specifically demands code or implementation
        requires_code = (
            intent_str in _IMPLEMENTATION_INTENTS
            or any(k in query_lower for k in ("how can i make", "how to make", "how to build", "show me the code", "give me the code", "code for", "implementation of", "task defined", "how is the", "agent defined"))
        )

        requires_diagram = (
            intent_str in _VISUAL_INTENTS
            or bool(_WORKFLOW_DIAGRAM_REGEX.search(query_lower))
            or any(k in query_lower for k in ("diagram", "architecture", "flowchart", "workflow", "system overview", "how does this work", "explain the content creation workflow"))
        )

        requires_table = (
            intent_str == "table"
            or any(k in query_lower for k in ("table", "matrix", "benchmark results", "values shown in table"))
        )

        if not candidate_chunks:
            # Check if previous evidence exists to maintain monotonicity
            if is_followup and previous_chunks and previous_status:
                monotonic_st = compute_monotonic_evidence_status(
                    previous_status=previous_status,
                    current_status=EvidenceStatus.MISSING,
                    has_prev_evidence=bool(previous_chunks),
                    has_curr_evidence=False,
                    is_followup=True,
                )
                if monotonic_st != EvidenceStatus.MISSING:
                    return EvidenceSufficiencyResult(
                        is_sufficient=True,
                        detected_continuation_cues=[],
                        pages_to_inspect=[],
                        anchor_chunk=previous_chunks[0] if previous_chunks else None,
                        visual_asset_available=any(c.chunk.metadata.image_assets for c in previous_chunks),
                        vision_understanding_available=any("diagram" in str(c.chunk.metadata.content_type).lower() for c in previous_chunks),
                        evidence_status=monotonic_st,
                        reason="Retained previous conversation evidence for follow-up query.",
                    )

            return EvidenceSufficiencyResult(
                is_sufficient=False,
                missing_evidence_types=["all_context"],
                evidence_status=EvidenceStatus.MISSING,
                reason="No context chunks were retrieved.",
            )

        # 1. Inspect existing chunks for code, tables, diagrams, and continuation cues
        has_concrete_code = False
        has_diagram_understanding = False
        has_extracted_visual_understanding = False
        has_visual_asset = False
        has_table = False
        all_cues: list[str] = []
        anchor_candidate: ScoredChunk | None = None
        pages_to_check: set[int] = set()

        for sc in candidate_chunks:
            text = sc.chunk.text
            meta = sc.chunk.metadata
            c_type = str(getattr(meta, "content_type", "")).lower()
            extra = getattr(meta, "extra", {}) or {}

            # Check if chunk has visual asset attached
            if meta.image_assets or meta.visual_asset_ids or extra.get("image_url") or extra.get("image_hash") or extra.get("is_visual_extraction"):
                has_visual_asset = True

            if (
                extra.get("is_visual_extraction")
                or extra.get("visual_type") in ("diagram_architecture", "workflow", "figure", "table_data", "code_screenshot")
            ):
                has_extracted_visual_understanding = True

            # Code detection
            if (
                "code" in c_type
                or extra.get("content_type") == "code"
                or bool(_CODE_DETECTION_REGEX.search(text))
                or bool(extra.get("raw_code"))
            ):
                has_concrete_code = True

            # Diagram understanding detection
            if (
                "diagram" in c_type
                or extra.get("visual_type") in ("diagram_architecture", "workflow", "figure")
                or extra.get("is_visual_extraction")
                or bool(_WORKFLOW_DIAGRAM_REGEX.search(text))
                or any("diagram" in str(x).lower() or "workflow" in str(x).lower() for x in (meta.image_assets or []))
            ):
                has_diagram_understanding = True
            elif "workflow" in text.lower() and len(text.strip()) > 30:
                has_diagram_understanding = True


            # Table detection
            if "table" in c_type or extra.get("visual_type") == "table_data" or "|---" in text:
                has_table = True

            # Detect continuation cues in text
            cues = detect_continuation_signals(text)
            if cues:
                all_cues.extend(cues)
                if anchor_candidate is None:
                    anchor_candidate = sc
                p_num = getattr(meta, "page_number", None)
                if p_num is not None and isinstance(p_num, int):
                    pages_to_check.add(p_num)
                    if p_num > 1:
                        pages_to_check.add(p_num - 1)
                    for offset in range(1, self.max_continuation_depth + 1):
                        pages_to_check.add(p_num + offset)

        if anchor_candidate is None and candidate_chunks:
            anchor_candidate = candidate_chunks[0]
            p_num = getattr(anchor_candidate.chunk.metadata, "page_number", None)
            if p_num is not None and isinstance(p_num, int):
                pages_to_check.add(p_num)
                if p_num > 1:
                    pages_to_check.add(p_num - 1)
                for offset in range(1, self.max_continuation_depth + 1):
                    pages_to_check.add(p_num + offset)

        missing_types: list[str] = []

        # 2. Check implementation requirement
        if requires_code and not has_concrete_code:
            missing_types.append("code_implementation")

        # 3. Check diagram requirement
        if requires_diagram and not has_diagram_understanding:
            missing_types.append("architecture_diagram")

        # 4. Check table requirement
        if requires_table and not has_table:
            missing_types.append("table_data")

        # A sentence such as "five techniques are depicted below" is a pointer,
        # not the five techniques themselves. Force visual extraction instead of
        # allowing the generator to fill the absent labels from model memory.
        visual_reference_cues = [
            cue
            for cue in all_cues
            if re.search(
                r"\b(?:depicted|illustrated|shown|figure|diagram|chart|table|visual)\b",
                cue,
                re.IGNORECASE,
            )
        ]
        text_resolves_visual_reference = _referenced_enumeration_is_complete(candidate_chunks)
        if (
            visual_reference_cues
            and not has_extracted_visual_understanding
            and not text_resolves_visual_reference
        ):
            missing_types.append("referenced_visual_content")

        # Determine 4-tier evidence status: DIRECT, PARTIAL, RELATED, MISSING
        all_text = "\n".join(sc.chunk.text for sc in candidate_chunks)
        if requires_code:
            if has_concrete_code:
                has_full_def = bool(re.search(r"(?:def\s+\w+|class\s+\w+|Agent\s*\([^)]+\)|Crew\s*\([^)]+\))", all_text))
                raw_status = EvidenceStatus.DIRECT if has_full_def else EvidenceStatus.PARTIAL
            elif candidate_chunks:
                raw_status = EvidenceStatus.RELATED
            else:
                raw_status = EvidenceStatus.MISSING
        else:
            if candidate_chunks:
                raw_status = EvidenceStatus.DIRECT if (not missing_types) else EvidenceStatus.PARTIAL
            else:
                raw_status = EvidenceStatus.MISSING

        # Apply monotonicity constraint for follow-up conversations
        final_evidence_status = compute_monotonic_evidence_status(
            previous_status=previous_status,
            current_status=raw_status,
            has_prev_evidence=bool(previous_chunks),
            has_curr_evidence=bool(candidate_chunks),
            is_followup=is_followup,
        )

        if missing_types:
            reason = (
                f"Missing required evidence types ({', '.join(missing_types)}) for intent '{intent_str}'. "
                f"Continuation cues detected: {all_cues or 'none'}."
            )
            logger.info("Evidence Sufficiency Gate: %s -> Inspecting pages %s", reason, sorted(pages_to_check))
            return EvidenceSufficiencyResult(
                is_sufficient=False,
                missing_evidence_types=missing_types,
                detected_continuation_cues=all_cues,
                pages_to_inspect=sorted(pages_to_check),
                anchor_chunk=anchor_candidate,
                visual_asset_available=has_visual_asset,
                vision_understanding_available=has_diagram_understanding,
                evidence_status=final_evidence_status,
                reason=reason,
            )

        return EvidenceSufficiencyResult(
            is_sufficient=True,
            detected_continuation_cues=all_cues,
            pages_to_inspect=[],
            anchor_chunk=anchor_candidate,
            visual_asset_available=has_visual_asset,
            vision_understanding_available=has_diagram_understanding,
            evidence_status=final_evidence_status,
            reason="Evidence is sufficient.",
        )
