from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Any

from cachetools import TTLCache
from pydantic import BaseModel, Field, field_validator

from backend.models.rag import Citation, EvidenceStatus, QueryCategory, ScoredChunk


class AnswerMode(str, Enum):
    DIRECT = "DIRECT"
    SUMMARY = "SUMMARY"
    EXPLANATION = "EXPLANATION"
    DETAILED = "DETAILED"
    STEP_BY_STEP = "STEP_BY_STEP"
    CODE_EXPLANATION = "CODE_EXPLANATION"
    COMPARISON = "COMPARISON"
    CONTINUATION = "CONTINUATION"
    CONTINUE = "CONTINUE"  # Backward-compatible alias for CONTINUATION
    EXPAND = "EXPAND"  # Backward-compatible alias for DETAILED / expansion


class ExpansionPlan(BaseModel):
    """Deterministic expansion plan for detailed or expanded follow-up turns."""

    restate_subject: str = "minimal"  # minimal | full | omit
    preserve_prior_facts: bool = True
    retrieve_additional_context: bool = True
    inspect_adjacent_evidence: bool = True
    explain_components: bool = True
    explain_execution_flow: bool = True
    explain_code_line_by_line: bool = True
    target_detail_level: str = "detailed"


class FollowUpResolution(BaseModel):
    """Structured resolution model representing conversation follow-up analysis."""

    is_follow_up: bool
    confidence: float = 1.0
    resolved_query: str
    primary_subject: str | None = None
    referenced_answer_id: str | None = None
    answer_mode: AnswerMode = AnswerMode.DIRECT
    expansion_requested: bool = False
    requested_detail_level: str = "standard"  # compact | standard | detailed
    preserve_previous_evidence: bool = True
    evidence_continuity_ids: list[str] = Field(default_factory=list)
    ambiguity_detected: bool = False
    rationale: str = ""


class ConversationEvidenceContext(BaseModel):
    """Encapsulates verified evidence context across turns for continuous grounding."""

    conversation_id: str
    turn_id: str
    query: str
    normalized_subjects: list[str] = Field(default_factory=list)
    verified_chunk_ids: list[str] = Field(default_factory=list)
    verified_citations: list[Citation] = Field(default_factory=list)
    evidence_status: EvidenceStatus = EvidenceStatus.DIRECT
    visual_asset_ids: list[str] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    answer_mode: AnswerMode = AnswerMode.DIRECT
    timestamp: float = Field(default_factory=time.time)

    @field_validator("evidence_status", mode="before")
    @classmethod
    def normalize_evidence_status(cls, v: Any) -> EvidenceStatus:
        if isinstance(v, EvidenceStatus):
            return v
        raw = str(getattr(v, "value", v) or "").strip().upper()
        if raw in ("DIRECT", "SUFFICIENT", "SUFFICIENT_CONTEXT"):
            return EvidenceStatus.DIRECT
        elif raw in ("PARTIAL", "PARTIAL_CONTEXT"):
            return EvidenceStatus.PARTIAL
        elif raw in ("RELATED", "RELATED_CONTEXT"):
            return EvidenceStatus.RELATED
        elif raw in ("MISSING", "INSUFFICIENT", "NO_EVIDENCE"):
            return EvidenceStatus.MISSING
        try:
            return EvidenceStatus(raw)
        except Exception:
            return EvidenceStatus.DIRECT

    @field_validator("answer_mode", mode="before")
    @classmethod
    def normalize_answer_mode(cls, v: Any) -> AnswerMode:
        if isinstance(v, AnswerMode):
            return v
        raw = str(getattr(v, "value", v) or "").strip().upper()
        try:
            return AnswerMode(raw)
        except Exception:
            return AnswerMode.DIRECT

    @property
    def session_id(self) -> str:
        return self.conversation_id


class ConversationTurn(BaseModel):
    """Encapsulates a single turn in a multi-turn conversation for auditing and replay."""

    turn_id: str
    timestamp: float = Field(default_factory=time.time)
    user_query: str
    resolved_query: str = ""
    is_followup: bool = False
    topic_shift: bool = False
    intent: str = "factual"
    answer_mode: str = "DIRECT"
    evidence_status: str = "DIRECT"
    active_topic: str | None = None
    active_entities: list[str] = Field(default_factory=list)
    retrieved_chunks: list[ScoredChunk] = Field(default_factory=list)
    visual_evidence: list[ScoredChunk] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    answer: str = ""
    resolution: FollowUpResolution | None = None
    expansion_plan: ExpansionPlan | None = None
    evidence_context: ConversationEvidenceContext | None = None

    @field_validator("resolved_query", mode="before")
    @classmethod
    def normalize_resolved_query(cls, v: Any, info: Any) -> str:
        if v is not None and str(v).strip():
            return str(v).strip()
        if hasattr(info, "data") and isinstance(info.data, dict):
            return str(info.data.get("user_query") or "").strip()
        return ""



class ConversationRAGState(BaseModel):
    """
    State tracking conversation context, grounded evidence, active topic,
    and history across multiple conversation turns. Isolated per conversation_id.
    """

    conversation_id: str
    last_user_query: str | None = None
    last_resolved_query: str | None = None
    active_topic: str | None = None
    active_entities: list[str] = Field(default_factory=list)
    active_document_ids: list[str] = Field(default_factory=list)
    previous_intent: QueryCategory | str | None = None
    previous_answer_mode: AnswerMode | str | None = None
    previous_evidence_status: EvidenceStatus | str | None = None
    previous_retrieved_chunks: list[ScoredChunk] = Field(default_factory=list)
    previous_visual_evidence: list[ScoredChunk] = Field(default_factory=list)
    previous_citations: list[Citation] = Field(default_factory=list)
    last_answer: str | None = None
    turns: list[ConversationTurn] = Field(default_factory=list)
    evidence_contexts: list[ConversationEvidenceContext] = Field(default_factory=list)
    last_resolution: FollowUpResolution | None = None
    last_expansion_plan: ExpansionPlan | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def to_history_messages(self, max_turns: int = 6) -> list[dict[str, Any]]:
        """Extract user and assistant message history for prompt formatting."""
        recent_turns = self.turns[-max_turns:]
        messages: list[dict[str, Any]] = []
        for turn in recent_turns:
            if turn.user_query:
                messages.append({"role": "user", "content": turn.user_query})
            if turn.answer:
                messages.append({"role": "assistant", "content": turn.answer})
        return messages


class ConversationStateManager:
    """
    Thread-safe, isolated in-memory cache of ConversationRAGState objects.
    Ensures zero cross-talk or evidence leakage between distinct conversation IDs.
    """

    def __init__(self, maxsize: int = 1000, ttl: int = 86400) -> None:
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._lock = threading.RLock()

    def get_state(self, conversation_id: str) -> ConversationRAGState:
        """Retrieve existing state or create a fresh empty state for the given conversation_id."""
        with self._lock:
            if conversation_id not in self._cache:
                self._cache[conversation_id] = ConversationRAGState(conversation_id=conversation_id)
            return self._cache[conversation_id].model_copy(deep=True)

    def save_state(self, state: ConversationRAGState) -> None:
        """Persist updated conversation state into thread-safe cache."""
        with self._lock:
            state.updated_at = time.time()
            self._cache[state.conversation_id] = state.model_copy(deep=True)

    def update_state(self, conversation_id: str, state: ConversationRAGState) -> None:
        """Update conversation state for the given conversation_id with deep copy isolation."""
        with self._lock:
            state.conversation_id = conversation_id
            state.updated_at = time.time()
            self._cache[conversation_id] = state.model_copy(deep=True)

    def delete_state(self, conversation_id: str) -> bool:
        """Evict a specific conversation from state cache. Returns True if existed, False otherwise."""
        with self._lock:
            existed = conversation_id in self._cache
            self._cache.pop(conversation_id, None)
            return existed

    def clear_all(self) -> None:
        """Purge all conversation states from memory."""
        with self._lock:
            self._cache.clear()

    def exists(self, conversation_id: str) -> bool:
        """Check if conversation_id exists in active cache."""
        with self._lock:
            return conversation_id in self._cache

