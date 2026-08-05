from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.models.rag import Citation, RAGTrace, ScoredChunk


class ChatRequest(BaseModel):
    message: str = Field(..., description="User chat query (1-8000 chars)")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation history")
    model: Optional[str] = Field(default="qwen2.5:7b", description="Selected LLM model")
    grounding_mode: Optional[str] = Field(default="balanced", description="balanced | strict")
    corpus_scope: Optional[str] = Field(default="all", description="all | policy | guidebook")
    chat_mode: Optional[str] = Field(default="direct", description="direct | agent")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")
    stream: bool = Field(default=False, description="Enable SSE streaming mode")


class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"resp_{uuid.uuid4().hex[:12]}")
    message_id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    session_id: Optional[str] = None
    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    latency_ms: float = 0.0
    metrics: Dict[str, Any] = Field(default_factory=dict)
    trace: Optional[RAGTrace] = None
    low_confidence: bool = False
    grounding_mode: str = "balanced"
    model: str = "qwen2.5:7b"
    token_usage: Dict[str, int] = Field(default_factory=dict)


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    chunks_indexed: int
    chunk_strategy: str
    status: str = "indexed"
    category: str = "general"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    chunk_count: int
    category: str = "general"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "indexed"


class DocumentListResponse(BaseModel):
    documents: List[DocumentSummary] = Field(default_factory=list)
    total_count: int = 0


class DocumentDetailResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    chunk_count: int
    category: str = "general"
    created_at: str
    status: str = "indexed"
    chunks: List[Dict[str, Any]] = Field(default_factory=list)


class TraceSummary(BaseModel):
    trace_id: str = Field(default_factory=lambda: f"tr_{uuid.uuid4().hex[:12]}")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    query: str
    rewritten_query: Optional[str] = None
    sub_queries: List[str] = Field(default_factory=list)
    candidate_count: int = 0
    post_rerank_count: int = 0
    final_context_count: int = 0
    execution_time_ms: float = 0.0
    ttft_ms: Optional[float] = None
    stage_timings: Dict[str, float] = Field(default_factory=dict)
    similarity_scores: List[float] = Field(default_factory=list)
    rerank_scores: List[float] = Field(default_factory=list)
    bm25_scores: List[float] = Field(default_factory=list)
    rrf_scores: List[float] = Field(default_factory=list)
    sources_used: List[str] = Field(default_factory=list)
    token_usage: Dict[str, int] = Field(default_factory=dict)
    faithfulness_passed: bool = True


class ObservabilityMetrics(BaseModel):
    total_queries: int = 0
    avg_latency_ms: float = 0.0
    avg_ttft_ms: float = 0.0
    p95_latency_ms: float = 0.0
    token_usage: Dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    score_distributions: Dict[str, float] = Field(default_factory=lambda: {"similarity_avg": 0.0, "rerank_avg": 0.0})
    active_documents: int = 0
    indexed_chunks: int = 0
    recent_traces: List[TraceSummary] = Field(default_factory=list)


class TraceDetailResponse(BaseModel):
    trace: TraceSummary
    full_rag_trace: Optional[RAGTrace] = None
    chunks_detail: List[Dict[str, Any]] = Field(default_factory=list)


class HealthStatus(BaseModel):
    status: str = "ok"
    redis: bool = False
    vector_db: bool = True
    bm25_index: bool = True
    models_loaded: bool = True
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    index_ready: bool = True
    chunk_count: int = 0
    collection: Optional[str] = "company_policy_chunks"


class ModelInfo(BaseModel):
    id: str
    name: str
    type: str = Field(..., description="llm | embedding | reranker")
    loaded: bool = True
    is_active: bool = False


class ModelListResponse(BaseModel):
    active_model: str = "qwen2.5:7b"
    models: List[ModelInfo] = Field(default_factory=list)
