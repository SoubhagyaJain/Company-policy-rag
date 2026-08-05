from __future__ import annotations

from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from backend.api.dependencies import get_document_service, get_telemetry_service
from backend.models.api_dto import ObservabilityMetrics, TraceDetailResponse, TraceSummary
from backend.services.document_service import DocumentService
from backend.services.telemetry_service import TelemetryService

router = APIRouter(tags=["Admin Observability"])


@router.get("/api/admin/observability", response_model=ObservabilityMetrics)
def get_observability_metrics(
    limit: int = 20,
    telemetry_service: TelemetryService = Depends(get_telemetry_service),
    doc_service: DocumentService = Depends(get_document_service),
) -> ObservabilityMetrics:
    """
    Get aggregated system observability metrics (total queries, avg latency, TTFT,
    token usage, score distributions, and recent trace summaries).
    """
    docs_res = doc_service.list_documents()
    active_docs_count = docs_res.total_count
    indexed_chunks_count = doc_service.vector_store.count()

    return telemetry_service.get_metrics(
        recent_limit=limit,
        active_documents=active_docs_count,
        indexed_chunks=indexed_chunks_count,
    )


@router.get("/api/admin/traces")
def get_query_traces(
    limit: int = 50,
    offset: int = 0,
    telemetry_service: TelemetryService = Depends(get_telemetry_service),
) -> Dict[str, Any]:
    """Retrieve list of execution traces for debugging RAG pipeline operations."""
    traces = telemetry_service.get_recent_traces(limit=limit, offset=offset)
    metrics = telemetry_service.get_metrics()
    return {
        "traces": [t.model_dump() for t in traces],
        "total_count": metrics.total_queries,
    }


@router.get("/api/admin/traces/{trace_id}", response_model=TraceDetailResponse)
def get_trace_detail_by_id(
    trace_id: str,
    telemetry_service: TelemetryService = Depends(get_telemetry_service),
) -> TraceDetailResponse:
    """Retrieve full trace details for a specific query execution trace ID."""
    trace = telemetry_service.get_trace_by_id(trace_id)
    if not trace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution trace with ID '{trace_id}' not found.",
        )
    return TraceDetailResponse(trace=trace)
