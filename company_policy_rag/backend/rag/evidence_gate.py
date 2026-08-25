from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from backend.models.chunk import ContentType
from backend.models.logical_document import detect_continuation_signals
from backend.models.rag import QueryCategory, ScoredChunk
from backend.utils.logging import logger

_CODE_DETECTION_REGEX = re.compile(
    r"(?:```(?:python|bash|sh|json|yaml|yml|javascript|ts|js)?\s*[\s\S]*?```|"
    r"(?:def\s+\w+\s*\(|class\s+\w+|import\s+\w+|from\s+\w+\s+import|"
    r"Agent\s*\(|Task\s*\(|Crew\s*\(|LLM\s*\(|SequentialProcess|"
    r"\w+\s*=\s*(?:Agent|Task|Crew)\s*\())",
    re.MULTILINE,
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


@dataclass
class EvidenceSufficiencyResult:
    is_sufficient: bool
    missing_evidence_types: list[str] = field(default_factory=list)
    detected_continuation_cues: list[str] = field(default_factory=list)
    pages_to_inspect: list[int] = field(default_factory=list)
    anchor_chunk: ScoredChunk | None = None
    reason: str = "Evidence is sufficient for query intent."


class EvidenceSufficiencyGate:
    """
    Pre-generation evaluation gate verifying that retrieved evidence is sufficient
    to answer the user query faithfully according to query intent before invoking LLM synthesis.
    """

    def __init__(self, max_continuation_depth: int = 2) -> None:
        self.max_continuation_depth = max_continuation_depth

    def evaluate(
        self,
        query: str,
        intent: str | QueryCategory,
        candidate_chunks: list[ScoredChunk],
    ) -> EvidenceSufficiencyResult:
        intent_str = intent.value.lower() if isinstance(intent, QueryCategory) else str(intent).lower()
        query_lower = query.lower()

        # Check if query specifically demands code or implementation
        requires_code = (
            intent_str in _IMPLEMENTATION_INTENTS
            or any(k in query_lower for k in ("how can i make", "how to make", "how to build", "show me the code", "give me the code", "code for", "implementation of", "task defined", "how is the", "agent defined"))
        )

        requires_diagram = (
            intent_str in ("architecture", "diagram", "workflow")
            or any(k in query_lower for k in ("diagram", "architecture", "flowchart", "workflow", "system overview"))
        )

        requires_table = (
            intent_str == "table"
            or any(k in query_lower for k in ("table", "matrix", "benchmark results", "values shown in table"))
        )

        if not candidate_chunks:
            return EvidenceSufficiencyResult(
                is_sufficient=False,
                missing_evidence_types=["all_context"],
                reason="No context chunks were retrieved.",
            )

        # 1. Inspect existing chunks for code, tables, diagrams, and continuation cues
        has_concrete_code = False
        has_diagram = False
        has_table = False
        all_cues: list[str] = []
        anchor_candidate: ScoredChunk | None = None
        pages_to_check: set[int] = set()

        for sc in candidate_chunks:
            text = sc.chunk.text
            meta = sc.chunk.metadata
            c_type = str(getattr(meta, "content_type", "")).lower()
            extra = getattr(meta, "extra", {}) or {}

            # Code detection
            if (
                "code" in c_type
                or extra.get("content_type") == "code"
                or bool(_CODE_DETECTION_REGEX.search(text))
                or getattr(meta, "has_code", False)
            ):
                has_concrete_code = True

            # Diagram detection
            if "diagram" in c_type or extra.get("visual_type") == "diagram_architecture":
                has_diagram = True

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
                    for offset in range(1, self.max_continuation_depth + 1):
                        pages_to_check.add(p_num + offset)

        if anchor_candidate is None and candidate_chunks:
            anchor_candidate = candidate_chunks[0]
            p_num = getattr(anchor_candidate.chunk.metadata, "page_number", None)
            if p_num is not None and isinstance(p_num, int):
                pages_to_check.add(p_num)
                for offset in range(1, self.max_continuation_depth + 1):
                    pages_to_check.add(p_num + offset)

        missing_types: list[str] = []

        # 2. Check implementation requirement
        if requires_code and not has_concrete_code:
            missing_types.append("code_implementation")

        # 3. Check diagram requirement
        if requires_diagram and not has_diagram:
            missing_types.append("architecture_diagram")

        # 4. Check table requirement
        if requires_table and not has_table:
            missing_types.append("table_data")

        if missing_types:
            reason = (
                f"Missing required evidence types ({', '.join(missing_types)}) for intent '{intent_str}'. "
                f"Continuation cues detected: {all_cues or 'none'}."
            )
            logger.info("Evidence Sufficiency Gate FAIL: %s -> Triggering expansion for pages %s", reason, sorted(pages_to_check))
            return EvidenceSufficiencyResult(
                is_sufficient=False,
                missing_evidence_types=missing_types,
                detected_continuation_cues=all_cues,
                pages_to_inspect=sorted(pages_to_check),
                anchor_chunk=anchor_candidate,
                reason=reason,
            )

        return EvidenceSufficiencyResult(
            is_sufficient=True,
            detected_continuation_cues=all_cues,
            pages_to_inspect=[],
            anchor_chunk=anchor_candidate,
            reason="Evidence is sufficient.",
        )
