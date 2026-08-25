from __future__ import annotations

from enum import Enum
import uuid
from typing import Any

from pydantic import BaseModel, Field

from backend.models.chunk import Chunk


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
    page_number: int | None = None
    internal_page_index: int | None = None
    page_label: str | None = None
    section_title: str | None = None
    section_path: str | None = None
    snippet: str = Field(..., description="Relevant text snippet cited")
    relevance_score: float = Field(default=0.0)
    selection_reason: str = Field(default="cited_in_answer", description="cited_in_answer | score_threshold_fallback")
    image_url: str | None = None
    image_assets: list[dict[str, Any]] = Field(default_factory=list)


class QueryRewriteResult(BaseModel):
    original_query: str
    rewritten_query: str
    sub_queries: list[str] = Field(default_factory=list)
    expanded_terms: list[str] = Field(default_factory=list)
    is_comprehensive_list: bool = False
    inferred_corpus: str | None = None


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
    # High-Observability fields (Phase 14)
    anchor_section: str | None = None
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


class RAGResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"resp_{uuid.uuid4().hex[:12]}")
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    context_chunks: list[ScoredChunk] = Field(default_factory=list)
    trace: RAGTrace
    model: str = Field(default="qwen2.5:7b")
    token_usage: dict[str, int] = Field(default_factory=dict)
