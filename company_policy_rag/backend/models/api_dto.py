from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from backend.models.rag import Citation, RAGTrace


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=8000, description="User chat query (1-8000 chars)")
    session_id: str | None = Field(default=None, min_length=1, max_length=128, description="Session ID for conversation history")
    model: str | None = Field(default="qwen2.5:7b", min_length=1, max_length=128, description="Selected LLM model (optional, defaults to active)")
    grounding_mode: str | None = Field(default="balanced", description="balanced | strict")
    corpus_scope: str | None = Field(default="all", description="all | policy | guidebook")
    chat_mode: str | None = Field(default="direct", description="direct | agent")
    active_document_id: str | None = Field(default=None, max_length=128, description="ID of currently active / uploaded document")
    active_document_name: str | None = Field(default=None, max_length=512, description="Filename / title of active document")
    selected_document_ids: list[str] | None = Field(default=None, description="Allowed document IDs for multi-doc comparison")
    document_scope: str | None = Field(default=None, description="Scope override: current_document | selected_documents | global")
    filters: dict[str, Any] | None = Field(default=None, description="Metadata filters")
    inferred_filters: dict[str, Any] | None = Field(default=None, description="Inferred metadata filters")
    enable_verification: bool | None = Field(default=None, description="Enable answer verification override")
    enable_routing: bool | None = Field(default=None, description="Enable query routing override")
    thinking_detail_level: str | None = Field(default="standard", description="off | compact | standard | detailed")
    stream: bool = Field(default=False, description="Enable SSE streaming mode")


class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"resp_{uuid.uuid4().hex[:12]}")
    message_id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    session_id: str | None = None
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    latency_ms: float = 0.0
    metrics: dict[str, Any] = Field(default_factory=dict)
    trace: RAGTrace | None = None
    low_confidence: bool = False
    grounding_mode: str = "balanced"
    model: str = "qwen2.5:7b"
    document_scope: str | None = None
    active_document_id: str | None = None
    active_document_name: str | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    query_type: str | None = None
    routing_confidence: float | None = None
    inferred_filters: dict[str, Any] = Field(default_factory=dict)
    verification: dict[str, Any] | None = None
    reasoning_summary: dict[str, Any] | None = None
    thinking_events: list[dict[str, Any]] = Field(default_factory=list)


class IngestionStatus(str, Enum):
    UPLOADING = "UPLOADING"
    TEXT_INDEXING = "TEXT_INDEXING"
    READY = "READY"
    VISION_PROCESSING = "VISION_PROCESSING"
    READY_WITH_VISION = "READY_WITH_VISION"
    PARTIALLY_INDEXED = "PARTIALLY_INDEXED"
    FAILED = "FAILED"


class IngestionStage(str, Enum):
    UPLOAD = "UPLOAD"
    TEXT_EXTRACTION = "TEXT_EXTRACTION"
    SECTION_DETECTION = "SECTION_DETECTION"
    CHUNKING = "CHUNKING"
    EMBEDDINGS = "EMBEDDINGS"
    VECTOR_INDEX = "VECTOR_INDEX"
    BM25_INDEX = "BM25_INDEX"
    FINALIZING = "FINALIZING"
    READY = "READY"
    FAILED = "FAILED"


class StageProgress(BaseModel):
    stage: str
    status: str = "PENDING"  # PENDING | IN_PROGRESS | COMPLETED | FAILED | SKIPPED
    message: str | None = None
    duration_ms: float = 0.0
    started_at: str | None = None
    completed_at: str | None = None


class IngestionStatusResponse(BaseModel):
    document_id: str
    job_id: str = Field(default_factory=lambda: f"job_{uuid.uuid4().hex[:10]}")
    filename: str
    status: str = "READY"  # TEXT_INDEXING | READY | VISION_PROCESSING | READY_WITH_VISION | PARTIALLY_INDEXED | FAILED | indexed
    progress: int = 100  # 0 to 100
    current_stage: str = "READY"
    text_ready: bool = True
    pages_processed: int = 0
    pages_total: int = 0
    sections_detected: int = 0
    chunks_created: int = 0
    chunks_indexed: int = 0
    vision_status: str = "NONE"  # NONE | PENDING | PROCESSING | COMPLETED | PARTIAL | SKIPPED | DEGRADED
    vision_pages_processed: int = 0
    vision_pages_total: int = 0
    error: str | None = None
    failed_stage: str | None = None
    stages: list[StageProgress] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    duration_ms: float = 0.0
    can_retry: bool = False


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    chunks_indexed: int
    chunk_strategy: str
    status: str = "READY"
    progress: int = 100
    current_stage: str = "READY"
    text_ready: bool = True
    vision_status: str = "NONE"
    vision_pages_processed: int = 0
    vision_pages_total: int = 0
    category: str = "general"
    department: str | None = None
    effective_date: str | None = None
    policy_id: str | None = None
    topic_tags: list[str] = Field(default_factory=list)
    key_entities: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    error: str | None = None
    failed_stage: str | None = None


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    chunk_count: int
    category: str = "general"
    department: str | None = None
    effective_date: str | None = None
    policy_id: str | None = None
    topic_tags: list[str] = Field(default_factory=list)
    key_entities: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = "indexed"
    progress: int = 100
    current_stage: str = "READY"
    text_ready: bool = True
    vision_status: str = "NONE"
    vision_pages_processed: int = 0
    vision_pages_total: int = 0
    error: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary] = Field(default_factory=list)
    total_count: int = 0


class DocumentDetailResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    chunk_count: int
    category: str = "general"
    department: str | None = None
    effective_date: str | None = None
    policy_id: str | None = None
    topic_tags: list[str] = Field(default_factory=list)
    key_entities: list[str] = Field(default_factory=list)
    created_at: str
    status: str = "indexed"
    progress: int = 100
    current_stage: str = "READY"
    text_ready: bool = True
    vision_status: str = "NONE"
    vision_pages_processed: int = 0
    vision_pages_total: int = 0
    error: str | None = None
    failed_stage: str | None = None
    chunks: list[dict[str, Any]] = Field(default_factory=list)


class TraceSummary(BaseModel):
    trace_id: str = Field(default_factory=lambda: f"tr_{uuid.uuid4().hex[:12]}")
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    query: str
    rewritten_query: str | None = None
    sub_queries: list[str] = Field(default_factory=list)
    query_type: str | None = None
    routing_confidence: float | None = None
    retrieval_strategy: str | None = None
    inferred_filters: dict[str, Any] = Field(default_factory=dict)
    applied_filters: dict[str, Any] = Field(default_factory=dict)
    filter_relaxed: bool = False
    candidate_count: int = 0
    post_rerank_count: int = 0
    final_context_count: int = 0
    execution_time_ms: float = 0.0
    ttft_ms: float | None = None
    stage_timings: dict[str, float] = Field(default_factory=dict)
    similarity_scores: list[float] = Field(default_factory=list)
    rerank_scores: list[float] = Field(default_factory=list)
    bm25_scores: list[float] = Field(default_factory=list)
    rrf_scores: list[float] = Field(default_factory=list)
    sources_used: list[str] = Field(default_factory=list)
    token_usage: dict[str, int] = Field(default_factory=dict)
    faithfulness_passed: bool = True
    verification: dict[str, Any] | None = None
    verification_score: float | None = None
    retry_count: int = 0
    retry_reasons: list[str] = Field(default_factory=list)


class ObservabilityMetrics(BaseModel):
    total_queries: int = 0
    avg_latency_ms: float = 0.0
    avg_ttft_ms: float = 0.0
    p95_latency_ms: float = 0.0
    token_usage: dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    score_distributions: dict[str, float] = Field(default_factory=lambda: {"similarity_avg": 0.0, "rerank_avg": 0.0})
    active_documents: int = 0
    indexed_chunks: int = 0
    recent_traces: list[TraceSummary] = Field(default_factory=list)


class TraceDetailResponse(BaseModel):
    trace: TraceSummary
    full_rag_trace: RAGTrace | None = None
    chunks_detail: list[dict[str, Any]] = Field(default_factory=list)


class HealthStatus(BaseModel):
    status: str = "ok"
    redis: bool = False
    vector_db: bool = True
    bm25_index: bool = True
    models_loaded: bool = True
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    index_ready: bool = True
    chunk_count: int = 0
    collection: str | None = "company_policy_chunks"


class ModelInfo(BaseModel):
    id: str
    name: str
    type: str = Field(..., description="llm | embedding | reranker")
    loaded: bool = True
    is_active: bool = False


class ModelListResponse(BaseModel):
    active_model: str = "qwen2.5:7b"
    vision_model: str = "qwen2.5vl:7b"
    vision_enabled: bool = True
    models: list[ModelInfo] = Field(default_factory=list)
