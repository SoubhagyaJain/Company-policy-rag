from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from backend.api.dependencies import get_document_service
from backend.models.api_dto import (
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentUploadResponse,
)
from backend.services.document_service import MAX_FILE_SIZE_BYTES, DocumentService

router = APIRouter(tags=["Documents"])


@router.post(
    "/api/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document_file(
    file: UploadFile = File(...),
    category: str = Form("general"),
    chunk_strategy: str | None = Form(None),
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    """
    Upload and index document file up to 100MB supporting PDF, DOCX, TXT, MD, HTML, CSV, JSON.
    Applies adaptive chunking strategy and indexes into Vector Store and BM25 search.
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename cannot be empty.")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds maximum allowed limit of 100MB.",
        )

    try:
        res = doc_service.upload_document(
            filename=file.filename,
            content_bytes=content,
            category=category,
            chunk_strategy=chunk_strategy,
        )
        return res
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document upload: {exc!s}",
        )


@router.get("/api/documents", response_model=DocumentListResponse)
def list_indexed_documents(
    category: str | None = None,
    limit: int = 100,
    offset: int = 0,
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    """Retrieve list of indexed documents with summary metadata."""
    return doc_service.list_documents(category=category, limit=limit, offset=offset)


@router.get("/api/documents/{doc_id}", response_model=DocumentDetailResponse)
def get_document_details(
    doc_id: str,
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentDetailResponse:
    """Get document detail and chunk index summary."""
    detail = doc_service.get_document_detail(doc_id)
    if not detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found.",
        )
    return detail


@router.delete("/api/documents/{doc_id}")
def delete_document_by_id(
    doc_id: str,
    doc_service: DocumentService = Depends(get_document_service),
):
    """Delete document, vector embeddings, and BM25 postings."""
    res = doc_service.delete_document(doc_id)
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found.",
        )
    return res
