from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.embeddings.embeddings import EmbeddingService
from backend.embeddings.vector_store import ChromaVectorStore
from backend.ingestion.chunkers.adaptive_chunker import AdaptiveChunker
from backend.ingestion.loaders.loader_factory import load_document
from backend.models.api_dto import (
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentSummary,
    DocumentUploadResponse,
)
from backend.models.chunk import Chunk
from backend.retrieval.bm25 import BM25SearchIndex
from backend.utils.logging import logger

MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100MB limit


class DocumentService:
    """
    Service managing multi-format document ingestion (PDF, DOCX, TXT, MD, HTML, CSV, JSON),
    100MB file uploads, adaptive chunking, vector & BM25 indexing, docstore persistence,
    document listing, details, and deletion.
    """

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        bm25_index: Optional[BM25SearchIndex] = None,
        embedding_service: Optional[EmbeddingService] = None,
        docstore: Optional[Dict[str, Chunk]] = None,
        storage_dir: str = "app/storage/uploads",
    ) -> None:
        self.vector_store = vector_store or ChromaVectorStore()
        self.bm25_index = bm25_index or BM25SearchIndex()
        self.embedding_service = embedding_service or EmbeddingService()
        self.docstore = docstore if docstore is not None else {}
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory document metadata registry
        self._documents: Dict[str, Dict[str, Any]] = {}

    def upload_document(
        self,
        filename: str,
        content_bytes: bytes,
        category: str = "general",
        chunk_strategy: Optional[str] = None,
    ) -> DocumentUploadResponse:
        """Process file upload, parse, chunk adaptively, embed, index, and record in docstore."""
        file_size = len(content_bytes)
        if file_size > MAX_FILE_SIZE_BYTES:
            raise ValueError(f"File size ({file_size} bytes) exceeds maximum limit of 100MB.")

        if not filename or not content_bytes:
            raise ValueError("File content or filename cannot be empty.")

        document_id = f"doc_{uuid.uuid4().hex[:12]}"
        ext = Path(filename).suffix.lower()
        file_type = ext.lstrip(".") or "txt"

        # Write to temporary file for loader processing
        temp_path = self.storage_dir / f"{document_id}_{filename}"
        try:
            temp_path.write_bytes(content_bytes)

            base_metadata = {
                "document_id": document_id,
                "category": category,
                "source_file": filename,
                "file_path": str(temp_path),
            }

            # 1. Multi-format Load
            raw_docs = load_document(temp_path, base_metadata=base_metadata)
            if not raw_docs:
                raise ValueError(f"Could not extract content from file '{filename}'.")

            # 2. Adaptive Chunking
            chunker = AdaptiveChunker(chunk_size=512, chunk_overlap=64, override_strategy=chunk_strategy)
            chunks = chunker.chunk(raw_docs)

            # Ensure document_id and source_file metadata are set on chunks
            used_strategy = "auto"
            for idx, c in enumerate(chunks):
                c.metadata.document_id = document_id
                c.metadata.source_file = filename
                c.metadata.category = category
                used_strategy = c.metadata.chunk_strategy or used_strategy

            # 3. Embedding Generation
            texts = [c.text for c in chunks]
            embeddings = self.embedding_service.embed_chunks(texts)
            for c, emb in zip(chunks, embeddings):
                c.embedding = emb

            # 4. Vector & BM25 Indexing
            self.vector_store.add_chunks(chunks)
            existing_chunks = list(self.bm25_index.entries) + chunks
            self.bm25_index.build_index(existing_chunks)

            # 5. Docstore Persistence
            for c in chunks:
                self.docstore[c.id] = c

            # 6. Document Registry
            created_at = datetime.now(timezone.utc).isoformat()
            doc_record = {
                "document_id": document_id,
                "filename": filename,
                "file_type": file_type,
                "file_size_bytes": file_size,
                "chunk_count": len(chunks),
                "chunk_strategy": used_strategy,
                "category": category,
                "created_at": created_at,
                "status": "indexed",
                "chunks": [
                    {
                        "chunk_id": c.id,
                        "chunk_index": c.metadata.chunk_index,
                        "text_snippet": c.text[:100] + ("..." if len(c.text) > 100 else ""),
                        "token_count": len(c.text.split()),
                        "page_number": c.metadata.page_number,
                        "section_title": c.metadata.section_title,
                    }
                    for c in chunks
                ],
            }
            self._documents[document_id] = doc_record

            logger.info(
                "Document uploaded & indexed: id=%s, file=%s, chunks=%d, strategy=%s",
                document_id,
                filename,
                len(chunks),
                used_strategy,
            )

            return DocumentUploadResponse(
                document_id=document_id,
                filename=filename,
                file_type=file_type,
                file_size_bytes=file_size,
                chunks_indexed=len(chunks),
                chunk_strategy=used_strategy,
                category=category,
                status="indexed",
                created_at=created_at,
            )
        finally:
            # Clean up temp upload file if created
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

    def list_documents(
        self,
        category: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> DocumentListResponse:
        """Return list of all registered documents."""
        records = list(self._documents.values())

        if category:
            records = [r for r in records if r.get("category") == category]

        paginated = records[offset : offset + limit]

        summaries = [
            DocumentSummary(
                document_id=r["document_id"],
                filename=r["filename"],
                file_type=r["file_type"],
                file_size_bytes=r["file_size_bytes"],
                chunk_count=r["chunk_count"],
                category=r["category"],
                created_at=r["created_at"],
                status=r["status"],
            )
            for r in paginated
        ]

        return DocumentListResponse(documents=summaries, total_count=len(records))

    def get_document_detail(self, document_id: str) -> Optional[DocumentDetailResponse]:
        """Get full details and chunk metadata for a document."""
        r = self._documents.get(document_id)
        if not r:
            return None

        return DocumentDetailResponse(
            document_id=r["document_id"],
            filename=r["filename"],
            file_type=r["file_type"],
            file_size_bytes=r["file_size_bytes"],
            chunk_count=r["chunk_count"],
            category=r["category"],
            created_at=r["created_at"],
            status=r["status"],
            chunks=r.get("chunks", []),
        )

    def delete_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Remove document and purge all associated vector embeddings and BM25 postings."""
        doc_record = self._documents.get(document_id)
        if not doc_record:
            return None

        filename = doc_record["filename"]

        # 1. Purge from Vector Store
        self.vector_store.delete_by_document_id(document_id)

        # 2. Purge from BM25 Index
        self.bm25_index.remove_by_document_id(document_id)

        # 3. Purge from Docstore
        chunk_ids_to_del = [
            cid for cid, chunk in list(self.docstore.items())
            if chunk.metadata.document_id == document_id
        ]
        for cid in chunk_ids_to_del:
            self.docstore.pop(cid, None)

        # 4. Purge from Registry
        del self._documents[document_id]

        logger.info("Deleted document id=%s, filename=%s (%d chunks purged)", document_id, filename, len(chunk_ids_to_del))

        return {
            "status": "deleted",
            "document_id": document_id,
            "filename": filename,
            "deleted_chunks": len(chunk_ids_to_del),
        }
