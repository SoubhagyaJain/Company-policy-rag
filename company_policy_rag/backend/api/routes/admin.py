from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.api.dependencies import get_document_service, get_telemetry_service
from backend.models.api_dto import (
    ObservabilityMetrics,
    TraceDetailResponse,
)
from backend.models.telemetry_models import (
    ErrorIncident,
    ObservabilitySummary,
    QueryTraceRecord,
    SubsystemHealth,
)
from backend.services.document_service import DocumentService
from backend.services.telemetry_service import TelemetryService

router = APIRouter(tags=["Admin Observability"])


@router.get("/api/admin/observability")
@router.get("/api/admin/observability/summary")
def get_observability_summary_endpoint(
    time_range: str = Query(default="24h", description="5m | 15m | 1h | 6h | 24h | 7d"),
    document_id: str | None = Query(default=None),
    conversation_id: str | None = Query(default=None),
    intent: str | None = Query(default=None),
    model: str | None = Query(default=None),
    status: str | None = Query(default=None),
    grounding: str | None = Query(default=None),
    vision: str | None = Query(default=None),
    cache: str | None = Query(default=None),
    has_error: bool | None = Query(default=None),
    telemetry_service: TelemetryService = Depends(get_telemetry_service),
    doc_service: DocumentService = Depends(get_document_service),
) -> ObservabilitySummary:
    """
    Unified Production RAG Observability Summary.
    Provides sub-system health, query performance, latency breakdown, retrieval quality,
    grounding telemetry, vision metrics, token tracking, memory resolution, ingestion status,
    cache metrics, threshold alerts, time-series, and recent query traces.
    """
    docs_res = doc_service.list_documents()
    active_docs_count = docs_res.total_count
    indexed_chunks_count = doc_service.vector_store.count()

    return telemetry_service.get_observability_summary(
        time_range=time_range,
        document_id=document_id,
        conversation_id=conversation_id,
        intent=intent,
        model=model,
        status=status,
        grounding=grounding,
        vision=vision,
        cache=cache,
        has_error=has_error,
        vector_db_ready=True,
        bm25_ready=True,
        doc_count=active_docs_count,
        chunk_count=indexed_chunks_count,
    )


@router.get("/api/admin/observability/health", response_model=SubsystemHealth)
def get_subsystem_health_endpoint(
    telemetry_service: TelemetryService = Depends(get_telemetry_service),
    doc_service: DocumentService = Depends(get_document_service),
) -> SubsystemHealth:
    """Independent health indicator for all 10 system components."""
    docs_res = doc_service.list_documents()
    return telemetry_service.probe_global_health(
        vector_db_ready=True,
        bm25_ready=True,
        doc_count=docs_res.total_count,
        chunk_count=doc_service.vector_store.count(),
    )


@router.get("/api/admin/observability/queries")
def get_filtered_query_traces(
    time_range: str = Query(default="24h"),
    document_id: str | None = None,
    conversation_id: str | None = None,
    intent: str | None = None,
    model: str | None = None,
    status: str | None = None,
    grounding: str | None = None,
    vision: str | None = None,
    cache: str | None = None,
    has_error: bool | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    telemetry_service: TelemetryService = Depends(get_telemetry_service),
) -> dict[str, Any]:
    """Retrieve filtered & paginated query trace records with real metadata."""
    traces, total_count = telemetry_service.db.get_filtered_traces(
        time_range=time_range,
        document_id=document_id,
        conversation_id=conversation_id,
        intent=intent,
        model=model,
        status=status,
        grounding=grounding,
        vision=vision,
        cache=cache,
        has_error=has_error,
        limit=limit,
        offset=offset,
    )
    return {
        "traces": [t.model_dump() for t in traces],
        "total_count": total_count,
        "limit": limit,
        "offset": offset,
    }


@router.get("/api/admin/observability/queries/{identifier}", response_model=QueryTraceRecord)
def get_query_trace_detail(
    identifier: str,
    telemetry_service: TelemetryService = Depends(get_telemetry_service),
) -> QueryTraceRecord:
    """Retrieve full execution and evidence trace for a specific request_id or trace_id."""
    trace = telemetry_service.db.get_trace_by_id_or_request_id(identifier)
    if not trace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query trace with identifier '{identifier}' not found.",
        )
    return trace


@router.get("/api/admin/observability/errors", response_model=list[ErrorIncident])
def get_error_incidents(
    time_range: str = Query(default="24h"),
    component: str | None = None,
    severity: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    telemetry_service: TelemetryService = Depends(get_telemetry_service),
) -> list[ErrorIncident]:
    """Retrieve structured error and failure incidents across all pipeline components."""
    return telemetry_service.db.get_recent_incidents(
        time_range=time_range,
        component=component,
        severity=severity,
        limit=limit,
    )


@router.post("/api/admin/observability/clear")
def clear_observability_telemetry(
    telemetry_service: TelemetryService = Depends(get_telemetry_service),
):
    """Purge in-memory and persistent telemetry databases."""
    telemetry_service.clear()
    return {"status": "success", "message": "Telemetry database cleared successfully."}


# ── Backward Compatibility Endpoints ─────────────────────────


@router.get("/api/admin/traces")
def get_legacy_query_traces(
    limit: int = 50,
    offset: int = 0,
    telemetry_service: TelemetryService = Depends(get_telemetry_service),
) -> dict[str, Any]:
    """Legacy query traces endpoint."""
    traces = telemetry_service.get_recent_traces(limit=limit, offset=offset)
    metrics = telemetry_service.get_metrics()
    return {
        "traces": [t.model_dump() for t in traces],
        "total_count": metrics.total_queries,
    }


@router.get("/api/admin/traces/{trace_id}", response_model=TraceDetailResponse)
def get_legacy_trace_detail_by_id(
    trace_id: str,
    telemetry_service: TelemetryService = Depends(get_telemetry_service),
) -> TraceDetailResponse:
    """Legacy trace detail endpoint."""
    trace = telemetry_service.get_trace_by_id(trace_id)
    if not trace:
        # Fallback check from DB
        db_trace = telemetry_service.db.get_trace_by_id_or_request_id(trace_id)
        if db_trace:
            legacy = TraceSummary(
                trace_id=db_trace.trace_id,
                timestamp=db_trace.timestamp,
                query=db_trace.original_query,
                rewritten_query=db_trace.rewritten_query,
                sub_queries=db_trace.sub_queries,
                query_type=db_trace.query_type,
                routing_confidence=db_trace.routing_confidence,
                retrieval_strategy=db_trace.retrieval_strategy,
                candidate_count=db_trace.candidate_count,
                post_rerank_count=db_trace.post_rerank_count,
                final_context_count=db_trace.final_chunk_count,
                execution_time_ms=db_trace.execution_time_ms,
                ttft_ms=db_trace.ttft_ms,
                sources_used=db_trace.sources_used,
                token_usage={"prompt_tokens": db_trace.prompt_tokens, "completion_tokens": db_trace.completion_tokens},
                faithfulness_passed=db_trace.faithfulness_passed,
                verification_score=db_trace.verification_score,
                retry_count=db_trace.retry_count,
                retry_reasons=db_trace.retry_reasons,
            )
            return TraceDetailResponse(trace=legacy)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution trace with ID '{trace_id}' not found.",
        )
    return TraceDetailResponse(trace=trace)
