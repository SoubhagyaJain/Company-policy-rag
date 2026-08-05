from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.models.chunk import Chunk


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float = Field(..., description="Relevance score (RRF score, cosine similarity, or cross-encoder logit)")
    rerank_score: Optional[float] = Field(default=None, description="Raw cross-encoder logit score if reranked")
    sparse_score: Optional[float] = Field(default=None, description="Raw BM25 score if sparse retrieved")
    dense_score: Optional[float] = Field(default=None, description="Raw dense vector similarity score")
    rank: Optional[int] = Field(default=None, description="Position rank in final retrieved list")


class Citation(BaseModel):
    source_index: int = Field(..., description="1-based source index [Source N]")
    chunk_id: str
    document_id: str
    source_file: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    section_path: Optional[str] = None
    snippet: str = Field(..., description="Relevant text snippet cited")
    relevance_score: float = Field(default=0.0)
    selection_reason: str = Field(default="cited_in_answer", description="cited_in_answer | score_threshold_fallback")


class QueryRewriteResult(BaseModel):
    original_query: str
    rewritten_query: str
    sub_queries: List[str] = Field(default_factory=list)
    expanded_terms: List[str] = Field(default_factory=list)
    is_comprehensive_list: bool = False
    inferred_corpus: Optional[str] = None


class RAGTrace(BaseModel):
    query: str
    rewritten_query: Optional[str] = None
    sub_queries: List[str] = Field(default_factory=list)
    retrieved_candidate_count: int = 0
    post_rerank_count: int = 0
    final_context_count: int = 0
    execution_time_ms: float = 0.0
    stage_timings_ms: Dict[str, float] = Field(default_factory=dict)
    fallback_reason: str = "none"
    faithfulness_checked: bool = False
    faithfulness_passed: bool = True


class RAGResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"resp_{uuid.uuid4().hex[:12]}")
    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    context_chunks: List[ScoredChunk] = Field(default_factory=list)
    trace: RAGTrace
    model: str = Field(default="qwen2.5:7b")
    token_usage: Dict[str, int] = Field(default_factory=dict)
