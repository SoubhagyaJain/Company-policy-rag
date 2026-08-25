from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from backend.api.dependencies import get_document_service
from backend.models.api_dto import (
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentUploadResponse,
    IngestionStatusResponse,
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
    Applies adaptive chunking strategy, batched embeddings, and indexes into Vector Store and BM25 search.
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
    """Retrieve list of indexed documents with summary metadata and real-time status."""
    return doc_service.list_documents(category=category, limit=limit, offset=offset)


@router.get("/api/documents/{doc_id}/status", response_model=IngestionStatusResponse)
def get_document_ingestion_status(
    doc_id: str,
    doc_service: DocumentService = Depends(get_document_service),
) -> IngestionStatusResponse:
    """Get real-time ingestion status, stage breakdown, and progress for a document."""
    status_res = doc_service.get_ingestion_status(doc_id)
    if not status_res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found.",
        )
    return status_res


@router.post("/api/documents/{doc_id}/retry", response_model=IngestionStatusResponse)
def retry_document_indexing(
    doc_id: str,
    doc_service: DocumentService = Depends(get_document_service),
) -> IngestionStatusResponse:
    """Retry indexing for a previously uploaded document without re-uploading the file."""
    try:
        return doc_service.retry_document(doc_id)
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retry document indexing: {exc!s}",
        )


@router.get("/api/documents/{doc_id}/assets")
def list_document_image_assets(
    doc_id: str,
    doc_service: DocumentService = Depends(get_document_service),
):
    """List all extracted standalone original visual assets for a document."""
    assets = doc_service.image_asset_manager.list_assets(doc_id)
    return {
        "document_id": doc_id,
        "total_assets": len(assets),
        "assets": [
            {
                "asset_id": a.asset_id,
                "internal_page_index": a.internal_page_index,
                "page_number": a.page_number,
                "page_label": a.page_label,
                "image_hash": a.image_hash,
                "content_type": a.content_type,
                "width": a.width,
                "height": a.height,
                "file_size_bytes": a.file_size_bytes,
                "asset_url": a.asset_url,
                "vision_status": a.vision_status,
                "vision_description": a.vision_description,
                "error": a.error,
                "created_at": a.created_at,
            }
            for a in assets
        ],
    }


@router.get("/api/documents/{doc_id}/images/{image_hash}")
def get_document_image_file(
    doc_id: str,
    image_hash: str,
    doc_service: DocumentService = Depends(get_document_service),
):
    """Serve standalone original high-resolution visual asset file."""
    asset = doc_service.image_asset_manager.get_asset(doc_id, image_hash)
    if not asset or not Path(asset.file_path).is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image asset with hash '{image_hash}' not found for document '{doc_id}'.",
        )
    media_type = "image/png" if asset.file_path.endswith(".png") else "image/jpeg"
    return FileResponse(asset.file_path, media_type=media_type)


@router.get("/api/documents/{doc_id}/pages/{page_number}/image")
def get_document_page_image_file(
    doc_id: str,
    page_number: int,
    doc_service: DocumentService = Depends(get_document_service),
):
    """Serve standalone original image for a specific physical page number."""
    asset = doc_service.image_asset_manager.get_page_asset(doc_id, page_number)
    if not asset or not Path(asset.file_path).is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No image asset found for page {page_number} in document '{doc_id}'.",
        )
    media_type = "image/png" if asset.file_path.endswith(".png") else "image/jpeg"
    return FileResponse(asset.file_path, media_type=media_type)


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
    """Delete document, vector embeddings, BM25 postings, and image assets."""
    res = doc_service.delete_document(doc_id)
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found.",
        )
    return res
