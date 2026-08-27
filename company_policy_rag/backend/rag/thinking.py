from __future__ import annotations

from datetime import datetime, timezone
import re
import threading
import time
from typing import Any
import uuid

from backend.models.rag import (
    ReasoningSummary,
    ThinkingDetailLevel,
    ThinkingEvent,
    ThinkingStage,
    ThinkingStatus,
)

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc


# Stages visible in COMPACT mode
COMPACT_STAGES = {
    ThinkingStage.RECEIVED,
    ThinkingStage.CONVERSATION_CONTEXT,
    ThinkingStage.RETRIEVAL,
    ThinkingStage.EVIDENCE_VERIFICATION,
    ThinkingStage.ANSWER_PLANNING,
    ThinkingStage.COMPLETED,
    ThinkingStage.DEGRADED,
}

# Whitelist of allowed metadata keys in details dictionary to ensure safety
ALLOWED_DETAIL_KEYS = {
    "candidate_count",
    "context_count",
    "rerank_count",
    "source_count",
    "citation_count",
    "reused_count",
    "active_topic",
    "active_entities",
    "is_follow_up",
    "topic_shift",
    "answer_mode",
    "evidence_status",
    "intent",
    "confidence",
    "pages",
    "document_id",
    "document_name",
    "visual_type",
    "degraded_stage",
    "degraded_reason",
    "fallback_action",
    "warning_reason",
    "duration_ms",
    "cache_hit",
}

# Patterns indicating potential leakage of internal reasoning or raw prompts
SENSITIVE_PATTERNS = [
    re.compile(r"(?:system\s*prompt|system:\s*|retrieved\s*context:|grounded_system_prompt)", re.IGNORECASE),
    re.compile(r"(?:chain-of-thought|raw\s*reasoning|internal\s*thought|let's\s*think\s*step)", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|secret|password|bearer\s+[a-zA-Z0-9_\-\.]+)", re.IGNORECASE),
    re.compile(r"(?:vector[_-]?id|embedding\s*vector|\[\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+)", re.IGNORECASE),
]


def _sanitize_details(details: dict[str, Any] | None) -> dict[str, Any]:
    """
    Sanitize the details dictionary to strictly exclude internal prompts,
    vector IDs, raw embeddings, CoT, or secrets.
    """
    if not details:
        return {}

    sanitized: dict[str, Any] = {}
    for k, v in details.items():
        if k not in ALLOWED_DETAIL_KEYS:
            continue

        # Check string values for sensitive patterns
        if isinstance(v, str):
            is_sensitive = False
            for pat in SENSITIVE_PATTERNS:
                if pat.search(v):
                    is_sensitive = True
                    break
            if not is_sensitive:
                sanitized[k] = v
        elif isinstance(v, (int, float, bool)):
            sanitized[k] = v
        elif isinstance(v, list):
            # Allow safe lists of strings/ints (e.g. pages, active entities)
            safe_list = []
            for item in v:
                if isinstance(item, (int, float, bool)):
                    safe_list.append(item)
                elif isinstance(item, str):
                    if not any(pat.search(item) for pat in SENSITIVE_PATTERNS):
                        safe_list.append(item)
            sanitized[k] = safe_list
        elif isinstance(v, dict):
            # Recurse shallowly
            sanitized[k] = _sanitize_details(v)

    return sanitized


class ThinkingStateMachine:
    """
    Deterministic state machine managing the lifecycle of Safe Thinking Events.
    Generates human-readable milestone summaries from actual pipeline state with
    zero raw chain-of-thought, zero internal prompt leakage, zero added LLM calls,
    and built-in duration tracking and graceful degradation handling.
    """

    def __init__(
        self,
        query_id: str,
        detail_level: ThinkingDetailLevel | str = ThinkingDetailLevel.STANDARD,
    ) -> None:
        self.query_id = query_id
        if isinstance(detail_level, str):
            try:
                self.detail_level = ThinkingDetailLevel(detail_level.lower())
            except ValueError:
                self.detail_level = ThinkingDetailLevel.STANDARD
        else:
            self.detail_level = detail_level

        self.events: list[ThinkingEvent] = []
        self.stage_starts: dict[ThinkingStage, float] = {}
        self.stage_durations_ms: dict[str, float] = {}
        self.degraded_stages: list[str] = []
        self._lock = threading.RLock()
        self._total_start = time.perf_counter()
        self._is_completed = False

    def is_visible(self, stage: ThinkingStage) -> bool:
        """Determine whether an event at this stage should be emitted given the current detail level."""
        if self.detail_level == ThinkingDetailLevel.OFF:
            return False
        if self.detail_level == ThinkingDetailLevel.COMPACT:
            return stage in COMPACT_STAGES
        return True

    def _generate_default_title(self, stage: ThinkingStage, status: ThinkingStatus) -> str:
        """Generate a deterministic, safe title for the given stage and status."""
        if status == ThinkingStatus.WARNING:
            return f"Notice: {stage.value.replace('_', ' ').title()}"
        if status == ThinkingStatus.FAILED:
            return f"Failed: {stage.value.replace('_', ' ').title()}"

        titles: dict[ThinkingStage, str] = {
            ThinkingStage.RECEIVED: "Understanding your question",
            ThinkingStage.CONVERSATION_CONTEXT: "Resolving conversation context",
            ThinkingStage.FOLLOW_UP_RESOLUTION: "Analyzing follow-up intent",
            ThinkingStage.QUERY_ANALYSIS: "Analyzing query intent",
            ThinkingStage.QUERY_REWRITE: "Optimizing search query",
            ThinkingStage.RETRIEVAL: "Searching relevant sources",
            ThinkingStage.RERANKING: "Evaluating relevance",
            ThinkingStage.EVIDENCE_ANALYSIS: "Analyzing retrieved evidence",
            ThinkingStage.EVIDENCE_REUSE: "Reusing verified evidence",
            ThinkingStage.PAGE_EXPANSION: "Checking nearby pages",
            ThinkingStage.VISUAL_ANALYSIS: "Checking visual content",
            ThinkingStage.EVIDENCE_VERIFICATION: "Verifying evidence",
            ThinkingStage.ANSWER_PLANNING: "Preparing response",
            ThinkingStage.ANSWER_GENERATION: "Generating grounded response",
            ThinkingStage.CITATION_BUILDING: "Building verified citations",
            ThinkingStage.COMPLETED: "Response completed",
            ThinkingStage.DEGRADED: "Operating in degraded mode",
        }
        return titles.get(stage, stage.value.replace("_", " ").title())

    def _generate_default_summary(
        self,
        stage: ThinkingStage,
        status: ThinkingStatus,
        details: dict[str, Any] | None = None,
    ) -> str:
        """Generate a safe, deterministic milestone summary purely from pipeline variables."""
        d = details or {}

        if stage == ThinkingStage.RECEIVED:
            return "Received and analyzed query for document grounding."

        elif stage == ThinkingStage.CONVERSATION_CONTEXT:
            is_followup = d.get("is_follow_up", False)
            active_topic = d.get("active_topic")
            if is_followup and active_topic:
                return f"Detected a follow-up question and linked it to the previously discussed {active_topic}."
            if is_followup:
                return "Detected a follow-up question and maintaining conversation context."
            return "Evaluating conversation context across previous dialogue turns."

        elif stage == ThinkingStage.FOLLOW_UP_RESOLUTION:
            is_followup = d.get("is_follow_up", False)
            mode = d.get("answer_mode", "DIRECT")
            if is_followup:
                return f"Resolved follow-up intent for answer mode '{mode}' with target expansion."
            return "Query is a standalone request."

        elif stage == ThinkingStage.QUERY_ANALYSIS:
            intent = d.get("intent", "factual")
            conf = d.get("confidence")
            if conf is not None and isinstance(conf, (int, float)):
                return f"Identified intent as {intent} (confidence: {float(conf):.2f})."
            return f"Identified query intent as {intent}."

        elif stage == ThinkingStage.QUERY_REWRITE:
            return "Expanded query terms and sub-queries for comprehensive document coverage."

        elif stage == ThinkingStage.RETRIEVAL:
            cand_cnt = d.get("candidate_count")
            if cand_cnt is not None:
                return f"Retrieved {cand_cnt} candidate chunks across document sections using hybrid search."
            return "Combined semantic and keyword search to locate related sections."

        elif stage == ThinkingStage.RERANKING:
            rerank_cnt = d.get("rerank_count")
            if rerank_cnt is not None:
                return f"Selected top {rerank_cnt} most relevant passages using cross-encoder scoring."
            return "Scored and prioritized retrieved passages based on relevance."

        elif stage == ThinkingStage.EVIDENCE_ANALYSIS:
            ev_st = d.get("evidence_status", "DIRECT")
            return f"Evaluated evidence sufficiency: {ev_st}."

        elif stage == ThinkingStage.EVIDENCE_REUSE:
            reused_cnt = d.get("reused_count")
            if reused_cnt is not None:
                return f"Reused {reused_cnt} previously verified evidence sources while incorporating new context."
            return "Using previously verified sources while searching for additional supporting context."

        elif stage == ThinkingStage.PAGE_EXPANSION:
            pages = d.get("pages")
            if pages:
                return f"Expanded search to adjacent pages ({pages}) to capture surrounding context."
            return "Inspecting adjacent document pages and sections for continuity."

        elif stage == ThinkingStage.VISUAL_ANALYSIS:
            v_type = d.get("visual_type")
            if v_type:
                return f"Inspecting relevant {v_type} assets for code or structural diagrams."
            return "Inspecting relevant code or diagram assets."

        elif stage == ThinkingStage.EVIDENCE_VERIFICATION:
            ev_st = str(d.get("evidence_status", "DIRECT")).upper()
            if ev_st == "DIRECT":
                return "Verified direct implementation and factual evidence."
            if ev_st == "PARTIAL":
                return "Found partial implementation evidence and supporting context."
            if ev_st == "RELATED":
                return "Found related conceptual and architectural evidence."
            return "Completed verification of evidence sufficiency."

        elif stage == ThinkingStage.ANSWER_PLANNING:
            mode = d.get("answer_mode", "grounded")
            return f"Structuring grounded answer following {mode} mode guidelines."

        elif stage == ThinkingStage.ANSWER_GENERATION:
            return "Synthesizing response directly from verified document context and citations."

        elif stage == ThinkingStage.CITATION_BUILDING:
            cit_cnt = d.get("citation_count")
            if cit_cnt is not None:
                return f"Generated {cit_cnt} verified source citations mapped to document excerpts."
            return "Building citations to link claims directly to source documents."

        elif stage == ThinkingStage.COMPLETED:
            return "Completed answer generation and evidence verification."

        elif stage == ThinkingStage.DEGRADED:
            reason = d.get("degraded_reason") or d.get("fallback_action")
            if reason:
                return f"Degraded gracefully: {reason}."
            return "Pipeline encountered an issue and degraded gracefully."

        return f"Stage {stage.value} completed."

    def start_stage(
        self,
        stage: ThinkingStage,
        title: str | None = None,
        summary: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> ThinkingEvent | None:
        """Mark a stage as running, start its duration timer, and return the emitted event."""
        with self._lock:
            self.stage_starts[stage] = time.perf_counter()
            clean_details = _sanitize_details(details)
            final_title = title or self._generate_default_title(stage, ThinkingStatus.RUNNING)
            final_summary = summary or self._generate_default_summary(stage, ThinkingStatus.RUNNING, clean_details)

            event = ThinkingEvent(
                id=f"thk_{uuid.uuid4().hex[:8]}",
                query_id=self.query_id,
                stage=stage,
                status=ThinkingStatus.RUNNING,
                title=final_title,
                summary=final_summary,
                details={} if self.detail_level == ThinkingDetailLevel.COMPACT else clean_details,
                started_at=datetime.now(UTC).isoformat(),
            )
            self.events.append(event)

            if not self.is_visible(stage):
                return None
            return event

    def complete_stage(
        self,
        stage: ThinkingStage,
        title: str | None = None,
        summary: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> ThinkingEvent | None:
        """Mark a stage as completed, record its duration, and return the emitted event."""
        with self._lock:
            t_start = self.stage_starts.pop(stage, time.perf_counter())
            duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
            self.stage_durations_ms[stage.value] = duration_ms

            clean_details = _sanitize_details(details)
            clean_details["duration_ms"] = duration_ms

            final_title = title or self._generate_default_title(stage, ThinkingStatus.COMPLETED)
            final_summary = summary or self._generate_default_summary(stage, ThinkingStatus.COMPLETED, clean_details)

            event = ThinkingEvent(
                id=f"thk_{uuid.uuid4().hex[:8]}",
                query_id=self.query_id,
                stage=stage,
                status=ThinkingStatus.COMPLETED,
                title=final_title,
                summary=final_summary,
                details={} if self.detail_level == ThinkingDetailLevel.COMPACT else clean_details,
                started_at=datetime.now(UTC).isoformat(),
                completed_at=datetime.now(UTC).isoformat(),
                duration_ms=duration_ms,
            )
            self.events.append(event)

            if not self.is_visible(stage):
                return None
            return event

    def degrade_stage(
        self,
        stage: ThinkingStage,
        reason: str,
        fallback_action: str = "",
        details: dict[str, Any] | None = None,
    ) -> ThinkingEvent | None:
        """Record a graceful degradation event when a primary retrieval, reranker, or vision step fails."""
        with self._lock:
            t_start = self.stage_starts.pop(stage, time.perf_counter())
            duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
            self.stage_durations_ms[f"{stage.value}_degraded"] = duration_ms

            if stage.value not in self.degraded_stages:
                self.degraded_stages.append(stage.value)

            clean_details = _sanitize_details(details)
            clean_details["degraded_stage"] = stage.value
            clean_details["degraded_reason"] = reason
            clean_details["duration_ms"] = duration_ms
            if fallback_action:
                clean_details["fallback_action"] = fallback_action

            title = f"{stage.value.replace('_', ' ').title()} Unavailable"
            summary = reason if not fallback_action else f"{reason}. {fallback_action}"

            event = ThinkingEvent(
                id=f"thk_{uuid.uuid4().hex[:8]}",
                query_id=self.query_id,
                stage=ThinkingStage.DEGRADED,
                status=ThinkingStatus.WARNING,
                title=title,
                summary=summary,
                details={} if self.detail_level == ThinkingDetailLevel.COMPACT else clean_details,
                started_at=datetime.now(UTC).isoformat(),
                completed_at=datetime.now(UTC).isoformat(),
                duration_ms=duration_ms,
            )
            self.events.append(event)

            if not self.is_visible(ThinkingStage.DEGRADED):
                return None
            return event

    def warn_stage(
        self,
        stage: ThinkingStage,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> ThinkingEvent | None:
        """Emit a warning event for a stage (e.g. ambiguity detected or partial match)."""
        with self._lock:
            clean_details = _sanitize_details(details)
            clean_details["warning_reason"] = reason

            event = ThinkingEvent(
                id=f"thk_{uuid.uuid4().hex[:8]}",
                query_id=self.query_id,
                stage=stage,
                status=ThinkingStatus.WARNING,
                title=self._generate_default_title(stage, ThinkingStatus.WARNING),
                summary=reason,
                details={} if self.detail_level == ThinkingDetailLevel.COMPACT else clean_details,
                started_at=datetime.now(UTC).isoformat(),
                completed_at=datetime.now(UTC).isoformat(),
            )
            self.events.append(event)

            if not self.is_visible(stage):
                return None
            return event

    def skip_stage(
        self,
        stage: ThinkingStage,
        reason: str = "",
        details: dict[str, Any] | None = None,
    ) -> ThinkingEvent | None:
        """Record that a stage was skipped (e.g., visual analysis when no image exists)."""
        with self._lock:
            clean_details = _sanitize_details(details)
            event = ThinkingEvent(
                id=f"thk_{uuid.uuid4().hex[:8]}",
                query_id=self.query_id,
                stage=stage,
                status=ThinkingStatus.SKIPPED,
                title=f"Skipped {stage.value.replace('_', ' ').title()}",
                summary=reason or f"{stage.value.replace('_', ' ').title()} was not needed for this query.",
                details={} if self.detail_level == ThinkingDetailLevel.COMPACT else clean_details,
                started_at=datetime.now(UTC).isoformat(),
                completed_at=datetime.now(UTC).isoformat(),
                duration_ms=0.0,
            )
            self.events.append(event)

            if not self.is_visible(stage):
                return None
            return event

    def record_stage(
        self,
        stage: ThinkingStage,
        status: ThinkingStatus = ThinkingStatus.COMPLETED,
        title: str | None = None,
        summary: str | None = None,
        details: dict[str, Any] | None = None,
        duration_ms: float = 0.0,
    ) -> ThinkingEvent | None:
        """Directly record an event with explicit status and duration."""
        with self._lock:
            clean_details = _sanitize_details(details)
            if duration_ms > 0:
                clean_details["duration_ms"] = duration_ms
                self.stage_durations_ms[stage.value] = duration_ms

            final_title = title or self._generate_default_title(stage, status)
            final_summary = summary or self._generate_default_summary(stage, status, clean_details)

            now_iso = datetime.now(UTC).isoformat()
            event = ThinkingEvent(
                id=f"thk_{uuid.uuid4().hex[:8]}",
                query_id=self.query_id,
                stage=stage,
                status=status,
                title=final_title,
                summary=final_summary,
                details={} if self.detail_level == ThinkingDetailLevel.COMPACT else clean_details,
                started_at=now_iso,
                completed_at=now_iso if status != ThinkingStatus.RUNNING else None,
                duration_ms=duration_ms,
            )
            self.events.append(event)

            if not self.is_visible(stage):
                return None
            return event

    def get_reasoning_summary(
        self,
        intent: str = "factual",
        answer_mode: str = "DIRECT",
        is_follow_up: bool = False,
        used_conversation_context: bool = False,
        reused_previous_evidence: bool = False,
        retrieved_new_evidence: bool = True,
        used_visual_evidence: bool = False,
        evidence_status: str = "DIRECT",
        sources_used: list[str] | None = None,
    ) -> ReasoningSummary:
        """Construct the safe structured telemetry summary of reasoning for RAGTrace."""
        total_duration = round((time.perf_counter() - self._total_start) * 1000, 2)
        return ReasoningSummary(
            intent=intent,
            answer_mode=answer_mode,
            is_follow_up=is_follow_up,
            used_conversation_context=used_conversation_context,
            reused_previous_evidence=reused_previous_evidence,
            retrieved_new_evidence=retrieved_new_evidence,
            used_visual_evidence=used_visual_evidence,
            evidence_status=evidence_status,
            sources_used=sources_used or [],
            degraded_stages=list(self.degraded_stages),
            total_duration_ms=total_duration,
        )

    def get_all_events(self) -> list[ThinkingEvent]:
        """Return all recorded thinking events."""
        with self._lock:
            return list(self.events)

    def get_visible_events(self) -> list[ThinkingEvent]:
        """Return recorded thinking events filtered by detail_level visibility."""
        with self._lock:
            if self.detail_level == ThinkingDetailLevel.OFF:
                return []
            return [ev for ev in self.events if self.is_visible(ev.stage)]
