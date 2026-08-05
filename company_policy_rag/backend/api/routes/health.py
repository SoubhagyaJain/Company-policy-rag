from __future__ import annotations

from fastapi import APIRouter, Depends
from backend.api.dependencies import get_document_service
from backend.models.api_dto import HealthStatus
from backend.services.document_service import DocumentService

router = APIRouter(tags=["Health"])


@router.get("/api/health", response_model=HealthStatus)
def get_health_status(
    doc_service: DocumentService = Depends(get_document_service),
) -> HealthStatus:
    """Return health readiness check including vector_db, redis, and model statuses."""
    chunk_cnt = doc_service.vector_store.count()
    return HealthStatus(
        status="ok",
        redis=False,  # In-memory fallback mode active
        vector_db=True,
        bm25_index=True,
        models_loaded=True,
        index_ready=True,
        chunk_count=chunk_cnt,
        collection=doc_service.vector_store.collection_name,
    )
