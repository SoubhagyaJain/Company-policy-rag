from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SubsystemStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"


class SubsystemHealth(BaseModel):
    api: SubsystemStatus = SubsystemStatus.HEALTHY
    ollama: SubsystemStatus = SubsystemStatus.HEALTHY
    vector_db: SubsystemStatus = SubsystemStatus.HEALTHY
    bm25: SubsystemStatus = SubsystemStatus.HEALTHY
    embedding_model: SubsystemStatus = SubsystemStatus.HEALTHY
    text_model: SubsystemStatus = SubsystemStatus.HEALTHY
    vision_model: SubsystemStatus = SubsystemStatus.HEALTHY
    semantic_cache: SubsystemStatus = SubsystemStatus.HEALTHY
    vision_cache: SubsystemStatus = SubsystemStatus.HEALTHY
    memory: SubsystemStatus = SubsystemStatus.HEALTHY
    uptime_seconds: float = 0.0
    error_rate: float = 0.0
    active_model_text: str = "qwen2.5:7b"
    active_model_vision: str = "qwen2.5vl:7b"
    details: dict[str, str] = Field(default_factory=dict)


class QueryMetrics(BaseModel):
    total_queries: int = 0
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    avg_latency_ms: float | None = None
    avg_ttft_ms: float | None = None
    avg_tokens_per_second: float | None = None
    avg_prompt_tokens: float | None = None
    avg_completion_tokens: float | None = None
    error_rate: float = 0.0
    requests_per_minute: float = 0.0


class LatencyBreakdown(BaseModel):
    request_received_ms: float | None = None
    query_classification_ms: float | None = None
    conversation_memory_ms: float | None = None
    query_rewrite_ms: float | None = None
    embedding_ms: float | None = None
    bm25_ms: float | None = None
    vector_search_ms: float | None = None
    hybrid_fusion_ms: float | None = None
    reranking_ms: float | None = None
    section_expansion_ms: float | None = None
    visual_detection_ms: float | None = None
    vision_extraction_ms: float | None = None
    context_build_ms: float | None = None
    qwen_prefill_ms: float | None = None
    ttft_ms: float | None = None
    generation_ms: float | None = None
    streaming_ms: float | None = None
    response_serialization_ms: float | None = None
    total_latency_ms: float = 0.0


class EvidenceContentType(str, Enum):
    TEXT = "text"
    CODE = "code"
    DIAGRAM = "diagram"
    TABLE = "table"


class EvidenceItem(BaseModel):
    chunk_id: str
    document_id: str | None = None
    source_file: str | None = None
    page_number: int | None = None
    page_label: str | None = None
    section_title: str | None = None
    section_id: str | None = None
    content_type: EvidenceContentType = EvidenceContentType.TEXT
    snippet: str = ""
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    selected: bool = True
    image_url: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class RetrievalQualityMetrics(BaseModel):
    retrieval_hit_rate: float = 0.0
    avg_candidate_count: float = 0.0
    avg_rerank_score: float = 0.0
    avg_final_chunk_count: float = 0.0
    evidence_sufficiency_rate: float = 0.0
    # Proxies vs Evaluation Distinction
    measured_metrics: dict[str, Any] = Field(default_factory=dict)
    proxy_metrics: dict[str, Any] = Field(default_factory=dict)
    evaluation_metrics: dict[str, Any] = Field(default_factory=dict)  # Only populated if ground truth dataset is tested


class GroundingStatus(str, Enum):
    GROUNDED = "grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    UNSUPPORTED = "unsupported"
    CONVERSATIONAL_BYPASS = "conversational_bypass"
    NOT_APPLICABLE = "not_applicable"


class GroundingTelemetry(BaseModel):
    supported_claims_pct: float | None = None
    unsupported_claims_pct: float | None = None
    inferred_claims_pct: float | None = None
    citation_count: int = 0
    citation_coverage_pct: float | None = None
    grounding_status: GroundingStatus = GroundingStatus.NOT_APPLICABLE
    supported_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    inferred_claims: list[str] = Field(default_factory=list)


class VisionFailureRecord(BaseModel):
    id: str = Field(default_factory=lambda: f"vfail_{uuid.uuid4().hex[:8]}")
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    document_id: str | None = None
    source_file: str | None = None
    page_number: int | None = None
    visual_type: str = "unknown"
    error_type: str = "TIMEOUT"  # TIMEOUT | CIRCUIT_OPEN | MODEL_ERROR | PARSE_ERROR
    duration_ms: float = 0.0
    request_id: str | None = None
    message: str = ""


class VisionTelemetry(BaseModel):
    model_name: str = "qwen2.5vl:7b"
    visual_pages_detected: int = 0
    code_screenshots: int = 0
    diagrams: int = 0
    tables: int = 0
    requests_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0
    avg_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    cache_hit_rate: float | None = None
    negative_cache_hit_rate: float | None = None
    circuit_breaker_state: str = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
    recent_failures: list[VisionFailureRecord] = Field(default_factory=list)


class TextModelTelemetry(BaseModel):
    model_name: str = "qwen2.5:7b"
    requests_count: int = 0
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    avg_ttft_ms: float | None = None
    avg_tokens_per_second: float | None = None
    avg_prompt_tokens: float | None = None
    avg_completion_tokens: float | None = None
    total_tokens: int = 0
    errors_count: int = 0


class ModelTelemetrySummary(BaseModel):
    text_model: TextModelTelemetry = Field(default_factory=TextModelTelemetry)
    vision_model: VisionTelemetry = Field(default_factory=VisionTelemetry)


class TokenTelemetry(BaseModel):
    avg_system_prompt_tokens: float = 0.0
    avg_memory_tokens: float = 0.0
    avg_user_query_tokens: float = 0.0
    avg_rag_context_tokens: float = 0.0
    avg_prompt_tokens: float = 0.0
    p95_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0
    p95_completion_tokens: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0


class MemoryResolutionEvent(BaseModel):
    id: str = Field(default_factory=lambda: f"mres_{uuid.uuid4().hex[:8]}")
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    session_id: str | None = None
    user_query: str = ""
    resolved_query: str = ""
    referent_found: str | None = None
    resolution_status: str = "SUCCESS"  # SUCCESS | NO_REFERENT | FAILED
    latency_ms: float = 0.0


class MemoryTelemetry(BaseModel):
    active_sessions: int = 0
    messages_today: int = 0
    memory_hit_rate: float | None = None
    reference_resolution_success_rate: float | None = None
    summary_updates: int = 0
    avg_memory_latency_ms: float | None = None
    avg_recent_turn_tokens: float = 0.0
    avg_summary_tokens: float = 0.0
    avg_memory_retrieval_tokens: float = 0.0
    recent_resolutions: list[MemoryResolutionEvent] = Field(default_factory=list)


class IngestionStageTelemetry(BaseModel):
    stage: str
    status: str
    duration_ms: float = 0.0
    message: str | None = None


class DocumentIngestionTrace(BaseModel):
    document_id: str
    filename: str
    category: str = "general"
    file_size_bytes: int = 0
    status: str = "READY"
    current_stage: str = "READY"
    pages_count: int = 0
    chunks_count: int = 0
    sections_count: int = 0
    visual_assets_count: int = 0
    vision_success_count: int = 0
    vision_failed_count: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    total_duration_ms: float = 0.0
    error: str | None = None
    stages: list[IngestionStageTelemetry] = Field(default_factory=list)


class IngestionTelemetry(BaseModel):
    documents_processed: int = 0
    ready_count: int = 0
    processing_count: int = 0
    failed_count: int = 0
    pages_processed: int = 0
    chunks_indexed: int = 0
    embeddings_generated: int = 0
    vector_index_ready: bool = True
    bm25_index_ready: bool = True
    visual_assets_total: int = 0
    vision_success_count: int = 0
    vision_failed_count: int = 0
    recent_ingestions: list[DocumentIngestionTrace] = Field(default_factory=list)


class CacheTypeMetrics(BaseModel):
    name: str
    hits: int = 0
    misses: int = 0
    hit_rate: float | None = None
    avg_hit_latency_ms: float | None = None
    avg_miss_latency_ms: float | None = None
    evictions: int = 0
    size_entries: int = 0
    size_bytes: int = 0


class CacheTelemetry(BaseModel):
    semantic_cache: CacheTypeMetrics = Field(
        default_factory=lambda: CacheTypeMetrics(name="Semantic Cache")
    )
    embedding_cache: CacheTypeMetrics = Field(
        default_factory=lambda: CacheTypeMetrics(name="Embedding Cache")
    )
    retrieval_cache: CacheTypeMetrics = Field(
        default_factory=lambda: CacheTypeMetrics(name="Retrieval Cache")
    )
    vision_cache: CacheTypeMetrics = Field(
        default_factory=lambda: CacheTypeMetrics(name="Vision Cache")
    )
    negative_vision_cache: CacheTypeMetrics = Field(
        default_factory=lambda: CacheTypeMetrics(name="Negative Vision Cache")
    )


class SeverityLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorIncident(BaseModel):
    incident_id: str = Field(default_factory=lambda: f"err_{uuid.uuid4().hex[:10]}")
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    request_id: str | None = None
    document_id: str | None = None
    conversation_id: str | None = None
    component: str  # API | Retrieval | Embedding | Ollama | Vision | Memory | Cache | Ingestion | Frontend | Streaming
    severity: SeverityLevel = SeverityLevel.ERROR
    message: str
    duration_ms: float | None = None
    retry_count: int = 0
    stack_trace: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class AlertStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertItem(BaseModel):
    alert_id: str
    rule_name: str
    severity: AlertStatus = AlertStatus.HEALTHY
    current_value: float | str
    threshold_value: float | str
    message: str
    triggered_at: str | None = None
    active: bool = False


class TimeSeriesPoint(BaseModel):
    timestamp: str
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    requests_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    avg_chunks: float = 0.0
    avg_rerank_score: float = 0.0
    vision_requests: int = 0
    vision_timeouts: int = 0
    vision_cache_hits: int = 0
    errors_count: int = 0


class QueryTraceRecord(BaseModel):
    trace_id: str = Field(default_factory=lambda: f"tr_{uuid.uuid4().hex[:12]}")
    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")
    conversation_id: str | None = None
    document_id: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    # Query Info
    original_query: str
    resolved_query: str | None = None
    rewritten_query: str | None = None
    sub_queries: list[str] = Field(default_factory=list)
    query_type: str = "factual"
    routing_confidence: float | None = None
    retrieval_strategy: str = "balanced_hybrid"

    # Conversational Bypass Semantics
    retrieval_required: bool = True
    conversational_bypass: bool = False
    evidence_required: bool = True

    # Retrieval counts
    candidate_count: int = 0
    post_rerank_count: int = 0
    final_chunk_count: int = 0

    # Section Expansion & Visuals
    anchor_section: str | None = None
    section_id: str | None = None
    target_pages: list[int] = Field(default_factory=list)
    section_expansion_used: bool = False
    vision_used: bool = False
    vision_model: str | None = None
    vision_cache_status: str | None = None

    # Evidence items
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    evidence_text_count: int = 0
    evidence_code_count: int = 0
    evidence_diagram_count: int = 0
    evidence_table_count: int = 0

    # Grounding & Verification
    grounding: GroundingTelemetry = Field(default_factory=GroundingTelemetry)
    faithfulness_passed: bool = True
    verification_score: float | None = None
    retry_count: int = 0
    retry_reasons: list[str] = Field(default_factory=list)

    # Cache
    cache_hit: bool = False
    cache_similarity: float | None = None
    cache_type: str | None = None

    # Timing & Tokens
    execution_time_ms: float = 0.0
    ttft_ms: float | None = None
    stage_timings: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    tokens_per_second: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    token_breakdown: dict[str, int] = Field(default_factory=dict)

    # Models & Scopes
    generation_model: str = "qwen2.5:7b"
    query_scope: str = "global"
    active_document_name: str | None = None
    sources_used: list[str] = Field(default_factory=list)
    error: str | None = None

    # Safe Context (sensitive data protected/masked)
    safe_context_preview: str | None = None


class ObservabilitySummary(BaseModel):
    time_range: str = "24h"
    health: SubsystemHealth = Field(default_factory=SubsystemHealth)
    query_metrics: QueryMetrics = Field(default_factory=QueryMetrics)
    latency_breakdown: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    retrieval_quality: RetrievalQualityMetrics = Field(default_factory=RetrievalQualityMetrics)
    grounding: GroundingTelemetry = Field(default_factory=GroundingTelemetry)
    models: ModelTelemetrySummary = Field(default_factory=ModelTelemetrySummary)
    tokens: TokenTelemetry = Field(default_factory=TokenTelemetry)
    memory: MemoryTelemetry = Field(default_factory=MemoryTelemetry)
    ingestion: IngestionTelemetry = Field(default_factory=IngestionTelemetry)
    caches: CacheTelemetry = Field(default_factory=CacheTelemetry)
    alerts: list[AlertItem] = Field(default_factory=list)
    recent_traces: list[QueryTraceRecord] = Field(default_factory=list)
    recent_incidents: list[ErrorIncident] = Field(default_factory=list)
    time_series: list[TimeSeriesPoint] = Field(default_factory=list)
