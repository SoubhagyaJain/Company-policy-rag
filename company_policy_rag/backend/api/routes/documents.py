from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from backend.api.dependencies import (
    get_conversation_state_manager,
    get_document_service,
    get_semantic_cache_manager,
)
from backend.models.api_dto import (
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentUploadResponse,
    IngestionStatusResponse,
)
from backend.models.conversation import ConversationStateManager
from backend.rag.semantic_cache import SemanticCacheManager
from backend.services.document_service import MAX_FILE_SIZE_BYTES, DocumentService, DuplicateDocumentError
from backend.utils.logging import logger

router = APIRouter(tags=["Documents"])
_UPLOAD_READ_CHUNK_BYTES = 1024 * 1024


async def _read_upload_with_limit(file: UploadFile) -> bytes:
    """Read an upload incrementally so oversized bodies are rejected early."""
    content = bytearray()
    while True:
        block = await file.read(_UPLOAD_READ_CHUNK_BYTES)
        if not block:
            break
        content.extend(block)
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="File size exceeds maximum allowed limit of 100MB.",
            )
    return bytes(content)


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

    content = await _read_upload_with_limit(file)

    try:
        res = await run_in_threadpool(
            doc_service.upload_document,
            file.filename,
            content,
            category,
            chunk_strategy,
        )
        return res
    except DuplicateDocumentError as duplicate_err:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "DUPLICATE_DOCUMENT",
                "message": str(duplicate_err),
                "existing_document_id": duplicate_err.document_id,
                "existing_filename": duplicate_err.filename,
                "file_hash": duplicate_err.file_hash,
            },
        )
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as exc:
        logger.exception("Document upload failed for filename=%s: %s", Path(file.filename).name, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process document upload.",
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


@router.get("/api/documents/duplicates")
def list_duplicate_documents(
    doc_service: DocumentService = Depends(get_document_service),
):
    """Preview byte-identical documents without changing storage."""
    return doc_service.deduplicate_documents(dry_run=True)


@router.post("/api/documents/deduplicate")
def deduplicate_documents(
    dry_run: bool = True,
    doc_service: DocumentService = Depends(get_document_service),
):
    """Preview or remove byte-identical duplicate documents across all storage layers."""
    return doc_service.deduplicate_documents(dry_run=dry_run)


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


@router.get("/api/documents/{doc_id}/visual-assets/{asset_id}")
def get_document_visual_asset_file(
    doc_id: str,
    asset_id: str,
    doc_service: DocumentService = Depends(get_document_service),
):
    """Serve standalone original high-resolution visual asset file by asset_id or image_hash."""
    asset = doc_service.image_asset_manager.get_asset(doc_id, asset_id)
    if not asset or not Path(asset.file_path).is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Visual asset '{asset_id}' not found for document '{doc_id}'.",
        )
    media_type = "image/png" if asset.file_path.endswith(".png") else "image/jpeg"
    return FileResponse(asset.file_path, media_type=media_type)


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


@router.get("/api/documents/{doc_id}/pages/{page_identifier}/image")
def get_document_page_image_file(
    doc_id: str,
    page_identifier: str,
    doc_service: DocumentService = Depends(get_document_service),
):
    """Serve standalone original image for a specific page (supports physical page, internal index, or printed label)."""
    parsed_id: int | str = int(page_identifier) if page_identifier.isdigit() else page_identifier
    asset = doc_service.image_asset_manager.get_page_asset(doc_id, parsed_id)
    if not asset or not Path(asset.file_path).is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No image asset found for page '{page_identifier}' in document '{doc_id}'.",
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
    semantic_cache: SemanticCacheManager = Depends(get_semantic_cache_manager),
    conversation_state: ConversationStateManager = Depends(get_conversation_state_manager),
):
    """Fully delete a document: its vector embeddings, BM25 postings, docstore
    chunks, image assets, and stored file — plus every piece of derived
    conversational state so the deleted content can never resurface. Cached Q&A
    answers (the semantic cache) and follow-up conversation context can cite a
    document, so both are invalidated; they cannot be scoped to a single document,
    so they are cleared wholesale (they simply rebuild on the next query)."""
    res = doc_service.delete_document(doc_id)
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found.",
        )

    caches_invalidated = {"semantic_cache": False, "conversation_state": False}
    try:
        semantic_cache.clear()
        caches_invalidated["semantic_cache"] = True
    except Exception as exc:  # best-effort: the document itself is already gone
        logger.warning("Semantic cache invalidation failed after deleting %s: %s", doc_id, exc)
    try:
        conversation_state.clear_all()
        caches_invalidated["conversation_state"] = True
    except Exception as exc:
        logger.warning("Conversation state clear failed after deleting %s: %s", doc_id, exc)

    res["caches_invalidated"] = caches_invalidated
    return res
