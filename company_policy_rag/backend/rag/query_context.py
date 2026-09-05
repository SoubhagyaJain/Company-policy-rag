"""Mutable state container threaded through the RAG query pipeline stages.

``QueryContext`` collects the state that used to live as ~80 local variables inside
``RAGPipeline._query_internal`` (a single ~1,365-line method). Extracting each pipeline
stage into its own method requires an explicit place to carry state between stages;
this dataclass is that place.

Design notes:
- ``from __future__ import annotations`` keeps every annotation a string, so importing
  this module never evaluates the domain types below — no import cost, no circular-import
  risk with ``pipeline.py`` or the rag submodules. The ``TYPE_CHECKING`` block exists only
  so editors/readers can see the real types.
- Every field has a default so a stage can populate ``ctx`` incrementally. This mirrors the
  original method, where locals were assigned progressively as execution advanced.
- This is a behavior-preserving refactor: field names match the original locals 1:1 so the
  extraction is a mechanical move, not a redesign.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from backend.models.conversation import ConversationRAGState
    from backend.models.rag import Citation, RAGTrace, ScoredChunk, VerificationReport
    from backend.rag.thinking import ThinkingStateMachine


@dataclass
class QueryContext:
    """Carries per-query state across the extracted pipeline stages."""

    # ── Request inputs (set once at construction) ──────────────────────────
    user_query: str = ""
    filters: dict[str, Any] | None = None
    chat_mode: str = "documents"
    history: list[dict[str, Any]] | None = None
    model: str | None = None
    active_document_id: str | None = None
    active_document_name: str | None = None
    selected_document_ids: list[str] | None = None
    document_scope: str | None = None
    conversation_state: "ConversationRAGState | None" = None
    response_mode: str = "standard"
    thinking_detail_level: Any = None
    thinking_sm: "ThinkingStateMachine | None" = None
    stream_callback: Callable[[str], None] | None = None

    # ── Derived setup ──────────────────────────────────────────────────────
    total_start: float = 0.0
    stage_timings: dict[str, float] = field(default_factory=dict)
    response_mode_config: Any = None
    req_llm: Any = None
    selected_model: str = ""

    # ── Classification & conversation resolution ───────────────────────────
    classification: Any = None
    strategy: Any = None
    fidelity_mode: str = "explain"
    conv_res: Any = None
    is_history_followup: bool = False
    effective_search_query: str = ""

    # ── Semantic cache eligibility ─────────────────────────────────────────
    cache_context: str = ""
    cache_eligible: bool = False

    # ── Scope, rewrite & filters ───────────────────────────────────────────
    scope_decision: Any = None
    known_docs: dict[str, str] = field(default_factory=dict)
    rewrite_res: Any = None
    inferred_filters: dict[str, Any] = field(default_factory=dict)
    applied_filters: dict[str, Any] = field(default_factory=dict)
    filter_relaxed: bool = False

    # ── Planning (multi-part, fast path, retry budget) ─────────────────────
    question_parts: list[str] = field(default_factory=list)
    is_fast_path: bool = False
    enable_verification: bool = True
    max_retries: int = 0
    current_strategy: Any = None
    # High-risk (policy / numeric) answers are verified before the user sees
    # them: their tokens are buffered instead of streamed live, and they keep a
    # retry budget even on the streaming path. ``stream_live`` is the emit gate —
    # True only when incremental token emission to the client is permitted.
    is_high_risk: bool = False
    stream_live: bool = False

    # ── Per-attempt working state ──────────────────────────────────────────
    attempt: int = 0
    prompt_refinement: str = ""
    retry_reasons: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    candidate_chunks: list["ScoredChunk"] = field(default_factory=list)
    reranked_chunks: list["ScoredChunk"] = field(default_factory=list)
    expanded_chunks: list["ScoredChunk"] = field(default_factory=list)
    answer_text: str = ""
    citations: list["Citation"] = field(default_factory=list)
    report: "VerificationReport | None" = None
    formatted_context: str = ""
    context_tokens: int = 0
    cross_document_count: int = 0
    telemetry_extra: dict[str, Any] = field(default_factory=dict)
    policy_selection: Any = None
    continuity_applied: bool = False
    raw_new_chunk_count: int = 0
    prev_all: list["ScoredChunk"] = field(default_factory=list)

    # ── Best-of-retries accumulators ───────────────────────────────────────
    best_answer: str = ""
    best_citations: list["Citation"] = field(default_factory=list)
    best_context_chunks: list["ScoredChunk"] = field(default_factory=list)
    best_context_tokens: int = 0
    best_candidate_chunks: list["ScoredChunk"] = field(default_factory=list)
    best_reranked_chunks: list["ScoredChunk"] = field(default_factory=list)
    best_report: "VerificationReport | None" = None
    best_policy_selection: Any = None
    best_score: float = -1.0
