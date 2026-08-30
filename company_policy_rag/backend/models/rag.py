from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import uuid
from typing import Any

from pydantic import BaseModel, Field

from backend.models.chunk import Chunk

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc


class ThinkingStage(str, Enum):
    RECEIVED = "received"
    CONVERSATION_CONTEXT = "conversation_context"
    FOLLOW_UP_RESOLUTION = "follow_up_resolution"
    QUERY_ANALYSIS = "query_analysis"
    QUERY_REWRITE = "query_rewrite"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    EVIDENCE_ANALYSIS = "evidence_analysis"
    EVIDENCE_REUSE = "evidence_reuse"
    PAGE_EXPANSION = "page_expansion"
    VISUAL_ANALYSIS = "visual_analysis"
    EVIDENCE_VERIFICATION = "evidence_verification"
    ANSWER_PLANNING = "answer_planning"
    ANSWER_GENERATION = "answer_generation"
    CITATION_BUILDING = "citation_building"
    COMPLETED = "completed"
    DEGRADED = "degraded"


class ThinkingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    WARNING = "warning"
    FAILED = "failed"


class ThinkingDetailLevel(str, Enum):
    OFF = "off"
    COMPACT = "compact"
    STANDARD = "standard"
    DETAILED = "detailed"


class ThinkingEvent(BaseModel):
    id: str = Field(default_factory=lambda: f"thk_{uuid.uuid4().hex[:8]}")
    query_id: str
    stage: ThinkingStage
    status: ThinkingStatus = ThinkingStatus.RUNNING
    title: str
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None
    duration_ms: float = 0.0


class ReasoningSummary(BaseModel):
    intent: str
    answer_mode: str
    is_follow_up: bool
    used_conversation_context: bool
    reused_previous_evidence: bool
    retrieved_new_evidence: bool
    used_visual_evidence: bool
    evidence_status: str
    sources_used: list[str] = Field(default_factory=list)
    degraded_stages: list[str] = Field(default_factory=list)
    total_duration_ms: float = 0.0


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float = Field(..., description="Relevance score (RRF score, cosine similarity, or cross-encoder logit)")
    rerank_score: float | None = Field(default=None, description="Raw cross-encoder logit score if reranked")
    sparse_score: float | None = Field(default=None, description="Raw BM25 score if sparse retrieved")
    dense_score: float | None = Field(default=None, description="Raw dense vector similarity score")
    rank: int | None = Field(default=None, description="Position rank in final retrieved list")


class Citation(BaseModel):
    source_index: int = Field(..., description="1-based source index [Source N]")
    chunk_id: str
    document_id: str
    source_file: str
    document_name: str | None = None
    page_number: int | None = None
    internal_page_index: int | None = None
    display_page_number: str | int | None = None
    page_label: str | None = None
    section_title: str | None = None
    section_path: str | None = None
    snippet: str = Field(..., description="Relevant text snippet cited")
    relevance_score: float = Field(default=0.0)
    selection_reason: str = Field(default="cited_in_answer", description="cited_in_answer | score_threshold_fallback")
    evidence_type: str = Field(default="TEXT", description="TEXT | CODE | DIAGRAM_ARCHITECTURE | TABLE_DATA | IMAGE | FIGURE")
    visual_asset_id: str | None = None
    visual_status: str | None = None
    image_url: str | None = None
    image_assets: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def display_page(self) -> str:
        """Preferred display page string: display_page_number -> page_label -> physical page_number."""
        if self.display_page_number is not None and str(self.display_page_number).strip():
            return str(self.display_page_number).strip()
        if self.page_label and str(self.page_label).strip():
            return str(self.page_label).strip()
        if self.page_number is not None:
            return str(self.page_number)
        return ""



class QueryRewriteResult(BaseModel):
    original_query: str
    rewritten_query: str
    sub_queries: list[str] = Field(default_factory=list)
    expanded_terms: list[str] = Field(default_factory=list)
    is_comprehensive_list: bool = False
    inferred_corpus: str | None = None


class EvidenceStatus(str, Enum):
    DIRECT = "DIRECT"
    PARTIAL = "PARTIAL"
    RELATED = "RELATED"
    MISSING = "MISSING"


class QueryCategory(str, Enum):
    FACTUAL = "factual"
    COMPARISON = "comparison"
    ENUMERATION = "enumeration"
    PROCEDURAL = "procedural"
    CONVERSATIONAL = "conversational"
    IMPLEMENTATION = "implementation"
    CODE = "code"
    EXPLANATION = "explanation"
    ARCHITECTURE = "architecture"


class RetrievalStrategy(BaseModel):
    name: str = "balanced_hybrid"
    dense_top_k: int = 25
    bm25_top_k: int = 25
    rrf_k: int = 60
    rerank_top_n: int = 6
    min_score_ratio: float = 0.40
    enable_multi_query: bool = False
    enable_parent_expansion: bool = True
    temperature: float = 0.1


class QueryClassification(BaseModel):
    category: QueryCategory = QueryCategory.FACTUAL
    confidence: float = 1.0
    strategy: RetrievalStrategy = Field(default_factory=RetrievalStrategy)
    reasoning: str = ""


class VerificationReport(BaseModel):
    faithfulness: float = 1.0
    completeness: float = 1.0
    citation_coverage: float = 1.0
    coherence: float = 1.0
    composite_score: float = 1.0
    passed: bool = True
    critique: str | None = None
    missing_aspects: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    retry_count: int = 0


class RAGTrace(BaseModel):
    query: str
    rewritten_query: str | None = None
    sub_queries: list[str] = Field(default_factory=list)
    query_type: str | None = None
    routing_confidence: float | None = None
    retrieval_strategy: str | None = None
    query_scope: str | None = None
    active_document_id: str | None = None
    active_document_name: str | None = None
    allowed_document_ids: list[str] = Field(default_factory=list)
    cross_document_chunks_rejected: int = 0
    final_context_documents: list[str] = Field(default_factory=list)
    inferred_filters: dict[str, Any] = Field(default_factory=dict)
    applied_filters: dict[str, Any] = Field(default_factory=dict)
    filter_relaxed: bool = False
    retrieved_candidate_count: int = 0
    post_rerank_count: int = 0
    final_context_count: int = 0
    response_mode: str = "standard"
    retrieval_top_k: int = 0
    rerank_top_k: int = 0
    context_tokens: int = 0
    generation_max_tokens: int = 0
    execution_time_ms: float = 0.0
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    fallback_reason: str = "none"
    faithfulness_checked: bool = False
    faithfulness_passed: bool = True
    verification_report: dict[str, Any] | None = None
    verification_score: float | None = None
    retry_count: int = 0
    retry_reasons: list[str] = Field(default_factory=list)
    cache_hit: bool = False
    cache_similarity: float | None = None
    # Conversation-Aware Observability fields
    conversation_id: str | None = None
    is_followup: bool = False
    topic_shift: bool = False
    follow_up_confidence: float = 0.0
    active_topic: str | None = None
    active_entities: list[str] = Field(default_factory=list)
    answer_mode: str | None = None
    previous_evidence_status: str | None = None
    evidence_continuity_applied: bool = False
    merged_chunk_count: int = 0
    previous_chunk_count: int = 0
    new_chunk_count: int = 0
    # High-Observability fields (Phase 14)
    anchor_section: str | None = None
    page_identity: str | None = None
    text_candidates: int = 0
    visual_candidates: int = 0
    final_text_evidence: int = 0
    final_visual_evidence: int = 0
    visual_asset_status: str | None = None
    vision_status: str | None = None
    evidence_status: str | None = None
    grounding_status: str | None = None
    evidence_text_count: int = 0
    evidence_code_count: int = 0
    evidence_diagram_count: int = 0
    evidence_table_count: int = 0
    section_expansion: bool = False
    adjacent_page_check: bool = False
    vision_fallback: bool = False
    vision_model: str | None = None
    vision_cache_status: str | None = None
    evidence_sufficiency_passed: bool = True
    generation_model: str | None = None
    grounding_validation_passed: bool = True
    # Safe Thinking & Telemetry Extensions (Milestone 2 & Milestone 3)
    reasoning_summary: ReasoningSummary | dict[str, Any] | None = None
    thinking_events: list[ThinkingEvent | dict[str, Any]] = Field(default_factory=list)

    def to_safe_dict(self) -> dict[str, Any]:
        """
        Return a strictly sanitized dictionary representation of RAGTrace with
        zero raw chain-of-thought, zero system prompts, zero vector IDs, and zero embeddings.
        """
        raw = self.model_dump()
        # Ensure reasoning_summary is a safe dict
        if self.reasoning_summary is not None:
            if hasattr(self.reasoning_summary, "model_dump"):
                raw["reasoning_summary"] = self.reasoning_summary.model_dump()
            elif isinstance(self.reasoning_summary, dict):
                raw["reasoning_summary"] = dict(self.reasoning_summary)
        # Ensure thinking_events are safe dicts
        raw["thinking_events"] = [
            e.model_dump() if hasattr(e, "model_dump") else e for e in self.thinking_events
        ]
        # Guarantee removal of any accidental private keys
        for forbidden in ("system_prompt", "raw_cot", "embeddings", "vector_ids", "vector_id", "prompt_text", "api_key", "secret"):
            raw.pop(forbidden, None)
        return raw


class RAGResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"resp_{uuid.uuid4().hex[:12]}")
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    context_chunks: list[ScoredChunk] = Field(default_factory=list)
    trace: RAGTrace
    model: str = Field(default="qwen2.5:7b")
    token_usage: dict[str, int] = Field(default_factory=dict)
