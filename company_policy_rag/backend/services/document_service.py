from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.embeddings.embeddings import EmbeddingService
from backend.embeddings.vector_store import ChromaVectorStore, unpack_chroma_metadata
from backend.ingestion.chunkers.adaptive_chunker import AdaptiveChunker
from backend.ingestion.loaders.loader_factory import load_document
from backend.ingestion.metadata_extractor import DocumentMetadataExtractor
from backend.models.api_dto import (
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentSummary,
    DocumentUploadResponse,
    IngestionStage,
    IngestionStatus,
    IngestionStatusResponse,
    StageProgress,
)
from backend.models.chunk import Chunk
from backend.models.telemetry_models import SeverityLevel
from backend.retrieval.bm25 import BM25SearchIndex
from backend.utils.logging import logger
from backend.vision.image_asset_manager import ImageAssetManager
from backend.vision.vision_cache import VisionCacheManager
from src.config import settings

MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100MB limit
EMBEDDING_BATCH_SIZE = 32


class DuplicateDocumentError(ValueError):
    """Raised when uploaded bytes already belong to an indexed document."""

    def __init__(self, document_id: str, filename: str, file_hash: str) -> None:
        self.document_id = document_id
        self.filename = filename
        self.file_hash = file_hash
        super().__init__(f"This file is already indexed as '{filename}' ({document_id}).")


class DocumentService:
    """
    Service managing multi-format document ingestion (PDF, DOCX, TXT, MD, HTML, CSV, JSON),
    100MB file uploads, adaptive chunking, vector & BM25 indexing, docstore persistence,
    document listing, details, status tracking, retry support, and deletion.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        bm25_index: BM25SearchIndex | None = None,
        embedding_service: EmbeddingService | None = None,
        docstore: dict[str, Chunk] | None = None,
        image_asset_manager: ImageAssetManager | None = None,
        vision_cache_manager: VisionCacheManager | None = None,
        storage_dir: str = "app/storage/uploads",
    ) -> None:
        self.vector_store = vector_store or ChromaVectorStore(
            collection_name=settings.chroma_collection_name,
            persist_dir=settings.chroma_persist_dir,
        )
        self.bm25_index = bm25_index or BM25SearchIndex()
        self.embedding_service = embedding_service or EmbeddingService()
        self.docstore = docstore if docstore is not None else {}
        self.image_asset_manager = image_asset_manager or ImageAssetManager()
        self.vision_cache_manager = vision_cache_manager or VisionCacheManager()
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._hash_catalog_path = self.storage_dir.parent / "content_hashes.json"

        # In-memory document metadata & job registries
        self._documents: dict[str, dict[str, Any]] = {}
        self._ingestion_jobs: dict[str, IngestionStatusResponse] = {}
        self._stored_files: dict[str, Path] = {}
        self._pending_hashes: dict[str, str] = {}
        self._hash_catalog = self._load_hash_catalog()
        self._lock = threading.Lock()
        # Single-worker queue for the heavy parse→chunk→embed→index pipeline, so
        # large uploads run off the request thread (no proxy/browser timeout) and
        # ingestions are serialized (BM25/vector writes never race each other).
        self._ingestion_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
        self._restore_document_registry()

    def _load_hash_catalog(self) -> dict[str, dict[str, Any]]:
        if not self._hash_catalog_path.is_file():
            return {}
        try:
            payload = json.loads(self._hash_catalog_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _metadata_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return []

    @staticmethod
    def _restored_document_identity(meta: dict[str, Any]) -> tuple[str, str | None]:
        """Return a safe ID and an optional legacy source-file deletion scope."""
        raw_document_id = str(meta.get("document_id") or "").strip()
        if re.fullmatch(r"doc_[0-9a-f]{12}", raw_document_id):
            return raw_document_id, None

        source_file = str(meta.get("source_file") or "").strip()
        stable_seed = str(meta.get("file_hash") or source_file or raw_document_id).strip()
        if not stable_seed:
            return "", None

        digest = hashlib.sha256(stable_seed.encode("utf-8")).hexdigest()[:12]
        return f"doc_{digest}", source_file or None

    def _save_hash_catalog(self) -> None:
        temp_path = self._hash_catalog_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(self._hash_catalog, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self._hash_catalog_path)

    def _hash_stored_file(self, document_id: str, file_path: Path) -> str:
        stat = file_path.stat()
        cached = self._hash_catalog.get(document_id, {})
        if (
            cached.get("bytes") == stat.st_size
            and cached.get("modified_ns") == stat.st_mtime_ns
            and len(str(cached.get("sha256", ""))) == 64
        ):
            return str(cached["sha256"])

        hasher = hashlib.sha256()
        with file_path.open("rb") as file_handle:
            while block := file_handle.read(1024 * 1024):
                hasher.update(block)
        digest = hasher.hexdigest()
        self._hash_catalog[document_id] = {
            "sha256": digest,
            "bytes": stat.st_size,
            "modified_ns": stat.st_mtime_ns,
        }
        return digest

    def _restore_document_registry(self) -> None:
        """Rebuild the UI document registry from the persistent vector index."""
        aggregated: dict[str, dict[str, Any]] = {}
        page_numbers: dict[str, set[str]] = {}
        collection = getattr(self.vector_store, "_collection", None)
        if collection is not None:
            try:
                result = collection.get(include=["documents", "metadatas"])
                result_ids = result.get("ids", []) or []
                result_documents = result.get("documents", []) or []
                result_metadatas = result.get("metadatas", []) or []
                for chunk_id, document_text, raw_meta in zip(
                    result_ids,
                    result_documents,
                    result_metadatas,
                ):
                    meta = raw_meta or {}
                    document_id, legacy_source_file = self._restored_document_identity(meta)
                    if not document_id:
                        continue
                    restored_chunk = Chunk(
                        id=str(chunk_id),
                        text=str(document_text or ""),
                        metadata=unpack_chroma_metadata(meta),
                    )
                    # Legacy indexes sometimes persisted None or another non-canonical
                    # value as document_id. Keep all in-memory indexes aligned with the
                    # canonical ID exposed through the documents API.
                    restored_chunk.metadata.document_id = document_id
                    self.docstore[restored_chunk.id] = restored_chunk
                    record = aggregated.setdefault(
                        document_id,
                        {
                            "document_id": document_id,
                            "filename": str(meta.get("source_file") or f"{document_id}.pdf"),
                            "file_type": str(meta.get("document_type") or "pdf").lower(),
                            "file_size_bytes": 0,
                            "file_hash": str(meta.get("file_hash") or ""),
                            "pages_count": 0,
                            "storage_state": "INDEX_ONLY",
                            "chunk_count": 0,
                            "category": str(meta.get("category") or meta.get("extra_category") or "General"),
                            "department": meta.get("department"),
                            "effective_date": meta.get("effective_date"),
                            "policy_id": meta.get("policy_id"),
                            "key_entities": self._metadata_list(meta.get("key_entities")),
                            "topic_tags": self._metadata_list(meta.get("topic_tags")),
                            "created_at": datetime.now(UTC).isoformat(),
                            "status": "READY",
                            "progress": 100,
                            "current_stage": "READY",
                            "text_ready": True,
                            "vision_status": "READY_ON_DEMAND",
                            "vision_pages_processed": 0,
                            "vision_pages_total": 0,
                            "chunks": [],
                            "_legacy_source_file": legacy_source_file,
                        },
                    )
                    record["chunk_count"] += 1
                    page_identity = meta.get("page_label") or meta.get("page_number") or meta.get("internal_page_index")
                    if page_identity is not None:
                        page_numbers.setdefault(document_id, set()).add(str(page_identity))

                # BM25's local pickle may be absent even while Chroma is healthy.
                # Rebuild it from the authoritative persisted chunks so lexical
                # retrieval and adjacent-page expansion survive process restarts.
                if self.docstore and not self.bm25_index.entries:
                    self.bm25_index.build_index(list(self.docstore.values()))
            except Exception as exc:
                logger.warning("Could not restore document registry from vector index: %s", exc)

        for file_path in self.storage_dir.glob("doc_*_*"):
            if not file_path.is_file() or len(file_path.name) < 18:
                continue
            document_id = file_path.name[:16]
            if not document_id.startswith("doc_") or file_path.name[16] != "_":
                continue
            filename = file_path.name[17:]
            stat = file_path.stat()
            record = aggregated.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "filename": filename,
                    "file_type": file_path.suffix.lstrip(".").lower() or "unknown",
                    "file_size_bytes": stat.st_size,
                    "file_hash": "",
                    "pages_count": 0,
                    "storage_state": "FILE_ONLY",
                    "chunk_count": 0,
                    "category": "General",
                    "department": None,
                    "effective_date": None,
                    "policy_id": None,
                    "key_entities": [],
                    "topic_tags": [],
                    "created_at": datetime.fromtimestamp(stat.st_ctime, UTC).isoformat(),
                    "status": "READY",
                    "progress": 100,
                    "current_stage": "READY",
                    "text_ready": True,
                    "vision_status": "READY_ON_DEMAND",
                    "vision_pages_processed": 0,
                    "vision_pages_total": 0,
                    "chunks": [],
                },
            )
            record["filename"] = filename
            record["file_type"] = file_path.suffix.lstrip(".").lower() or record["file_type"]
            record["file_size_bytes"] = stat.st_size
            record["created_at"] = datetime.fromtimestamp(stat.st_ctime, UTC).isoformat()
            record["file_hash"] = self._hash_stored_file(document_id, file_path)
            record["pages_count"] = len(page_numbers.get(document_id, set()))
            record["storage_state"] = "HEALTHY" if record["chunk_count"] > 0 else "FILE_ONLY"
            self._stored_files[document_id] = file_path

        self._documents.update(aggregated)
        self._save_hash_catalog()
        if aggregated:
            logger.info("Restored %d documents into the document registry.", len(aggregated))

    @staticmethod
    def _canonical_duplicate_record(records: list[dict[str, Any]]) -> dict[str, Any]:
        """Prefer the newest fully indexed copy, then the newest stored copy."""
        return max(
            records,
            key=lambda record: (
                int(record.get("chunk_count", 0) > 0),
                str(record.get("created_at") or ""),
                str(record.get("document_id") or ""),
            ),
        )

    def get_duplicate_groups(self) -> list[dict[str, Any]]:
        """Return byte-identical stored files grouped by their full SHA-256 digest."""
        with self._lock:
            records = [dict(record) for record in self._documents.values()]

        by_hash: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            file_hash = str(record.get("file_hash") or "")
            document_id = str(record.get("document_id") or "")
            file_path = self._stored_files.get(document_id)
            if len(file_hash) == 64 and file_path is not None and file_path.is_file():
                by_hash.setdefault(file_hash, []).append(record)

        groups: list[dict[str, Any]] = []
        for file_hash, group_records in by_hash.items():
            if len(group_records) < 2:
                continue
            canonical = self._canonical_duplicate_record(group_records)
            ordered = sorted(
                group_records,
                key=lambda record: (record["document_id"] != canonical["document_id"], record["created_at"]),
            )
            groups.append(
                {
                    "file_hash": file_hash,
                    "filename": canonical["filename"],
                    "keep_document_id": canonical["document_id"],
                    "duplicate_count": len(group_records) - 1,
                    "documents": [
                        {
                            "document_id": record["document_id"],
                            "filename": record["filename"],
                            "chunk_count": record.get("chunk_count", 0),
                            "created_at": record.get("created_at"),
                            "keep": record["document_id"] == canonical["document_id"],
                        }
                        for record in ordered
                    ],
                }
            )
        return sorted(groups, key=lambda group: str(group["filename"]).casefold())

    def deduplicate_documents(self, dry_run: bool = True) -> dict[str, Any]:
        """Preview or remove only exact-content duplicate documents."""
        groups = self.get_duplicate_groups()
        remove_ids = [
            document["document_id"]
            for group in groups
            for document in group["documents"]
            if not document["keep"]
        ]
        removed: list[dict[str, Any]] = []
        if not dry_run:
            for document_id in remove_ids:
                result = self.delete_document(document_id)
                if result:
                    removed.append(result)
        return {
            "dry_run": dry_run,
            "duplicate_groups": len(groups),
            "duplicates_found": len(remove_ids),
            "duplicates_removed": len(removed),
            "groups": groups,
            "removed": removed,
        }

    def get_ingestion_status(self, document_id: str) -> IngestionStatusResponse | None:
        """Return real-time structured ingestion status for a document."""
        with self._lock:
            if document_id in self._ingestion_jobs:
                job = self._ingestion_jobs[document_id]
                # Check for stuck job detection
                if job.status == IngestionStatus.TEXT_INDEXING.value or job.status == "PROCESSING":
                    try:
                        updated_dt = datetime.fromisoformat(job.updated_at)
                        # A single stage on a large file (parsing a 100MB PDF, a big
                        # embedding batch) can legitimately run for minutes without a
                        # heartbeat, so only treat a job as stuck after a generous
                        # window to avoid false failures on big documents.
                        if (datetime.now(UTC) - updated_dt).total_seconds() > 900:
                            job.status = IngestionStatus.FAILED.value
                            job.error = "Ingestion job became unresponsive (heartbeat timeout)."
                            job.can_retry = True
                    except Exception:
                        pass
                return job

            # Fallback check from document registry
            doc = self._documents.get(document_id)
            if doc:
                return IngestionStatusResponse(
                    document_id=document_id,
                    job_id=f"job_{document_id[:8]}",
                    filename=doc.get("filename", ""),
                    status=doc.get("status", "READY"),
                    progress=doc.get("progress", 100),
                    current_stage=doc.get("current_stage", "READY"),
                    text_ready=doc.get("text_ready", True),
                    pages_processed=doc.get("pages_count", 1),
                    pages_total=doc.get("pages_count", 1),
                    chunks_created=doc.get("chunk_count", 0),
                    chunks_indexed=doc.get("chunk_count", 0),
                    vision_status=doc.get("vision_status", "NONE"),
                    created_at=doc.get("created_at", datetime.now(UTC).isoformat()),
                    updated_at=datetime.now(UTC).isoformat(),
                    can_retry=False,
                )
        return None

    def _update_job_stage(
        self,
        document_id: str,
        stage: IngestionStage,
        status: str,  # PENDING | IN_PROGRESS | COMPLETED | FAILED
        progress: int,
        message: str | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        with self._lock:
            job = self._ingestion_jobs.get(document_id)
            if not job:
                return

            job.current_stage = stage.value
            job.progress = progress
            job.updated_at = datetime.now(UTC).isoformat()

            # Find or append stage progress entry
            found = False
            for sp in job.stages:
                if sp.stage == stage.value:
                    sp.status = status
                    sp.message = message
                    sp.duration_ms = duration_ms
                    if status == "COMPLETED":
                        sp.completed_at = datetime.now(UTC).isoformat()
                    found = True
                    break

            if not found:
                job.stages.append(
                    StageProgress(
                        stage=stage.value,
                        status=status,
                        message=message,
                        duration_ms=duration_ms,
                        started_at=datetime.now(UTC).isoformat(),
                        completed_at=datetime.now(UTC).isoformat() if status == "COMPLETED" else None,
                    )
                )

    def _execute_ingestion_stages(
        self,
        document_id: str,
        filename: str,
        file_path: Path,
        category: str = "general",
        chunk_strategy: str | None = None,
    ) -> DocumentUploadResponse:
        """
        Execute core text ingestion pipeline with structured logging, batched embeddings,
        per-stage timing, forward-progress chunk verification, and strict failure isolation.
        """
        t_global_start = time.perf_counter()
        file_size = file_path.stat().st_size if file_path.exists() else 0
        ext = file_path.suffix.lower()
        file_type = ext.lstrip(".") or "txt"

        logger.info("[INGESTION] document_id=%s file=%s size=%d bytes", document_id, filename, file_size)

        # STAGE 1: FILE SAVED
        t0 = time.perf_counter()
        self._update_job_stage(
            document_id=document_id,
            stage=IngestionStage.UPLOAD,
            status="COMPLETED",
            progress=10,
            message="File validated and saved to disk.",
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        logger.info("[INGESTION] document_id=%s STAGE 1: FILE SAVED status=SUCCESS", document_id)

        try:
            # STAGE 2: TEXT EXTRACTION
            t_stage = time.perf_counter()
            self._update_job_stage(
                document_id=document_id,
                stage=IngestionStage.TEXT_EXTRACTION,
                status="IN_PROGRESS",
                progress=15,
                message="Extracting document pages and text...",
            )

            base_metadata = {
                "document_id": document_id,
                "category": category,
                "source_file": filename,
                "file_path": str(file_path),
            }

            raw_docs = load_document(file_path, base_metadata=base_metadata)
            if not raw_docs:
                # A scanned PDF has no text layer at all, so vision extraction is the only
                # way to read one. Name that cause instead of a generic parse failure.
                if ext == ".pdf" and not getattr(settings, "vision_enabled", True):
                    logger.warning(
                        "No text layer in '%s' (scanned or image-only PDF) and vision extraction "
                        "is disabled via VISION_ENABLED=false. Indexing 0 chunks; this document "
                        "will not be retrievable until vision is enabled and it is re-uploaded.",
                        filename,
                    )
                else:
                    logger.warning(f"Could not extract text content from file '{filename}'. Proceeding with 0 chunks.")

            pages_count = len(raw_docs)
            t_extract = round((time.perf_counter() - t_stage) * 1000, 2)
            self._update_job_stage(
                document_id=document_id,
                stage=IngestionStage.TEXT_EXTRACTION,
                status="COMPLETED",
                progress=25,
                message=(
                    f"Extracted {pages_count} pages successfully."
                    if pages_count
                    else "No readable text found. This file indexes 0 chunks and will not be retrievable."
                ),
                duration_ms=t_extract,
            )
            logger.info(
                "[INGESTION] document_id=%s STAGE 2: TEXT EXTRACTION pages=%d duration=%.2fms status=SUCCESS",
                document_id,
                pages_count,
                t_extract,
            )

            # STAGE 3: SECTION DETECTION & METADATA EXTRACTION
            t_stage = time.perf_counter()
            self._update_job_stage(
                document_id=document_id,
                stage=IngestionStage.SECTION_DETECTION,
                status="IN_PROGRESS",
                progress=30,
                message="Detecting logical sections and metadata...",
            )

            full_text = "\n\n".join(doc.content for doc in raw_docs)
            extractor = DocumentMetadataExtractor()
            first_meta = raw_docs[0].metadata if raw_docs else None
            extracted_meta = extractor.extract(full_text, doc_metadata=first_meta)

            sections_found = sum(1 for doc in raw_docs if doc.metadata.section_title)
            for doc in raw_docs:
                doc.metadata.document_id = document_id
                doc.metadata.department = extracted_meta.department
                doc.metadata.effective_date = extracted_meta.effective_date
                doc.metadata.policy_id = extracted_meta.policy_id
                doc.metadata.key_entities = list(extracted_meta.key_entities)
                doc.metadata.topic_tags = list(extracted_meta.topic_tags)

            t_sections = round((time.perf_counter() - t_stage) * 1000, 2)
            self._update_job_stage(
                document_id=document_id,
                stage=IngestionStage.SECTION_DETECTION,
                status="COMPLETED",
                progress=35,
                message=f"Detected {sections_found} sections.",
                duration_ms=t_sections,
            )
            logger.info(
                "[INGESTION] document_id=%s STAGE 3: SECTION DETECTION sections=%d duration=%.2fms status=SUCCESS",
                document_id,
                sections_found,
                t_sections,
            )

            # STAGE 4: CHUNKING
            t_stage = time.perf_counter()
            self._update_job_stage(
                document_id=document_id,
                stage=IngestionStage.CHUNKING,
                status="IN_PROGRESS",
                progress=40,
                message="Applying adaptive chunking strategy...",
            )

            chunker = AdaptiveChunker(chunk_size=512, chunk_overlap=64, override_strategy=chunk_strategy)
            chunks = chunker.chunk(raw_docs)

            if not chunks:
                logger.warning("Chunking produced 0 chunks for document.")

            used_strategy = "auto"
            for idx, c in enumerate(chunks):
                c.metadata.document_id = document_id
                c.metadata.source_file = filename
                c.metadata.category = category
                c.metadata.department = extracted_meta.department
                c.metadata.effective_date = extracted_meta.effective_date
                c.metadata.policy_id = extracted_meta.policy_id
                if not c.metadata.key_entities:
                    c.metadata.key_entities = list(extracted_meta.key_entities)
                if not c.metadata.topic_tags:
                    c.metadata.topic_tags = list(extracted_meta.topic_tags)
                used_strategy = c.metadata.chunk_strategy or used_strategy

            t_chunk = round((time.perf_counter() - t_stage) * 1000, 2)
            self._update_job_stage(
                document_id=document_id,
                stage=IngestionStage.CHUNKING,
                status="COMPLETED",
                progress=50,
                message=f"Created {len(chunks)} chunks via {used_strategy} strategy.",
                duration_ms=t_chunk,
            )
            logger.info(
                "[INGESTION] document_id=%s STAGE 4: CHUNKING chunks_created=%d strategy=%s duration=%.2fms status=SUCCESS",
                document_id,
                len(chunks),
                used_strategy,
                t_chunk,
            )

            # STAGE 5: EMBEDDING GENERATION (Batched with progress updates)
            t_stage = time.perf_counter()
            self._update_job_stage(
                document_id=document_id,
                stage=IngestionStage.EMBEDDINGS,
                status="IN_PROGRESS",
                progress=52,
                message=f"Generating embeddings for {len(chunks)} chunks...",
            )

            texts = [c.text for c in chunks]
            total_chunks = len(texts)
            all_embeddings: list[list[float]] = []

            # Batch processing to prevent CPU/GPU blocking
            num_batches = (total_chunks + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
            for b_idx in range(num_batches):
                b_start = b_idx * EMBEDDING_BATCH_SIZE
                b_end = min(b_start + EMBEDDING_BATCH_SIZE, total_chunks)
                batch_texts = texts[b_start:b_end]

                t_batch_start = time.perf_counter()
                batch_embs = self.embedding_service.embed_chunks(batch_texts)
                all_embeddings.extend(batch_embs)
                b_duration = round((time.perf_counter() - t_batch_start) * 1000, 2)

                # Progress scales from 50% to 75%
                batch_progress = int(50 + 25 * (b_end / total_chunks))
                self._update_job_stage(
                    document_id=document_id,
                    stage=IngestionStage.EMBEDDINGS,
                    status="IN_PROGRESS",
                    progress=batch_progress,
                    message=f"Generating embeddings ({b_end}/{total_chunks} chunks)...",
                )
                logger.info(
                    "[INGESTION] document_id=%s Embedding batch: %d/%d (chunks %d-%d), duration=%.2fms",
                    document_id,
                    b_idx + 1,
                    num_batches,
                    b_start + 1,
                    b_end,
                    b_duration,
                )

            for c, emb in zip(chunks, all_embeddings):
                c.embedding = emb

            t_embed = round((time.perf_counter() - t_stage) * 1000, 2)
            self._update_job_stage(
                document_id=document_id,
                stage=IngestionStage.EMBEDDINGS,
                status="COMPLETED",
                progress=75,
                message=f"Generated {len(all_embeddings)} embeddings.",
                duration_ms=t_embed,
            )
            logger.info(
                "[INGESTION] document_id=%s STAGE 5: EMBEDDINGS processed=%d/%d duration=%.2fms status=SUCCESS",
                document_id,
                len(all_embeddings),
                total_chunks,
                t_embed,
            )

            # STAGE 6: VECTOR INDEXING
            t_stage = time.perf_counter()
            self._update_job_stage(
                document_id=document_id,
                stage=IngestionStage.VECTOR_INDEX,
                status="IN_PROGRESS",
                progress=80,
                message="Indexing chunks in ChromaDB vector store...",
            )

            self.vector_store.add_chunks(chunks)
            t_vec = round((time.perf_counter() - t_stage) * 1000, 2)
            self._update_job_stage(
                document_id=document_id,
                stage=IngestionStage.VECTOR_INDEX,
                status="COMPLETED",
                progress=85,
                message=f"Indexed {len(chunks)} vectors in ChromaDB.",
                duration_ms=t_vec,
            )
            logger.info(
                "[INGESTION] document_id=%s STAGE 6: VECTOR INDEX duration=%.2fms status=SUCCESS",
                document_id,
                t_vec,
            )

            # STAGE 7: BM25 INDEXING
            t_stage = time.perf_counter()
            self._update_job_stage(
                document_id=document_id,
                stage=IngestionStage.BM25_INDEX,
                status="IN_PROGRESS",
                progress=90,
                message="Building BM25 sparse search index...",
            )

            existing_chunks = list(self.bm25_index.entries) + chunks
            self.bm25_index.build_index(existing_chunks)
            self.bm25_index.save()
            t_bm25 = round((time.perf_counter() - t_stage) * 1000, 2)
            self._update_job_stage(
                document_id=document_id,
                stage=IngestionStage.BM25_INDEX,
                status="COMPLETED",
                progress=95,
                message="BM25 index updated.",
                duration_ms=t_bm25,
            )
            logger.info(
                "[INGESTION] document_id=%s STAGE 7: BM25 INDEX duration=%.2fms status=SUCCESS",
                document_id,
                t_bm25,
            )

            # STAGE 8: FINALIZATION & DOCSTORE PERSISTENCE
            t_stage = time.perf_counter()
            self._update_job_stage(
                document_id=document_id,
                stage=IngestionStage.FINALIZING,
                status="IN_PROGRESS",
                progress=98,
                message="Finalizing document registration...",
            )

            for c in chunks:
                self.docstore[c.id] = c

            created_at = datetime.now(UTC).isoformat()
            extracted_assets = self.image_asset_manager.list_assets(document_id)
            doc_record = {
                "document_id": document_id,
                "filename": filename,
                "file_type": file_type,
                "file_size_bytes": file_size,
                "file_hash": self._hash_catalog.get(document_id, {}).get("sha256", ""),
                "pages_count": pages_count,
                "storage_state": "HEALTHY",
                "chunk_count": len(chunks),
                "chunk_strategy": used_strategy,
                "category": category,
                "department": extracted_meta.department,
                "effective_date": extracted_meta.effective_date,
                "policy_id": extracted_meta.policy_id,
                "key_entities": list(extracted_meta.key_entities),
                "topic_tags": list(extracted_meta.topic_tags),
                "created_at": created_at,
                "status": "READY",
                "progress": 100,
                "current_stage": "READY",
                "text_ready": True,
                "vision_status": "READY_ON_DEMAND",
                "vision_pages_processed": 0,
                "vision_pages_total": 0,
                "visual_assets_count": len(extracted_assets),
                "image_assets": [
                    {
                        "asset_id": a.asset_id,
                        "page_number": a.page_number,
                        "page_label": a.page_label,
                        "image_hash": a.image_hash,
                        "content_type": a.content_type,
                        "width": a.width,
                        "height": a.height,
                        "asset_url": a.asset_url,
                        "vision_status": a.vision_status,
                    }
                    for a in extracted_assets
                ],
                "chunks": [
                    {
                        "chunk_id": c.id,
                        "chunk_index": c.metadata.chunk_index,
                        "text_snippet": c.text[:100] + ("..." if len(c.text) > 100 else ""),
                        "token_count": len(c.text.split()),
                        "page_number": c.metadata.page_number,
                        "internal_page_index": c.metadata.internal_page_index,
                        "page_label": c.metadata.page_label,
                        "section_title": c.metadata.section_title,
                        "image_assets": c.metadata.image_assets,
                    }
                    for c in chunks
                ],
            }

            with self._lock:
                self._documents[document_id] = doc_record

            t_final = round((time.perf_counter() - t_stage) * 1000, 2)
            t_total = round((time.perf_counter() - t_global_start) * 1000, 2)

            self._update_job_stage(
                document_id=document_id,
                stage=IngestionStage.FINALIZING,
                status="COMPLETED",
                progress=100,
                message="Document READY for search & RAG.",
                duration_ms=t_final,
            )

            with self._lock:
                if document_id in self._ingestion_jobs:
                    j = self._ingestion_jobs[document_id]
                    j.status = "READY"
                    j.progress = 100
                    j.current_stage = "READY"
                    j.text_ready = True
                    j.pages_processed = pages_count
                    j.pages_total = pages_count
                    j.chunks_created = len(chunks)
                    j.chunks_indexed = len(chunks)
                    j.duration_ms = t_total
                    j.can_retry = False
                    stages_dump = [s.model_dump() for s in j.stages]

            try:
                from backend.api.dependencies import get_telemetry_service
                get_telemetry_service().record_ingestion_trace(
                    document_id=document_id,
                    filename=filename,
                    category=category,
                    file_size_bytes=file_size,
                    status="READY",
                    current_stage="READY",
                    pages_count=pages_count,
                    chunks_count=len(chunks),
                    visual_assets_count=len(raw_docs),
                    vision_success_count=len(raw_docs),
                    vision_failed_count=0,
                    total_duration_ms=t_total,
                    error=None,
                    stages=stages_dump,
                )
            except Exception:
                pass

            logger.info(
                "[INGESTION] document_id=%s STAGE 8: FINALIZATION status=SUCCESS DOCUMENT STATUS=READY PROGRESS=100 (total=%.2fms)",
                document_id,
                t_total,
            )

            return DocumentUploadResponse(
                document_id=document_id,
                filename=filename,
                file_type=file_type,
                file_size_bytes=file_size,
                file_hash=doc_record["file_hash"],
                pages_count=pages_count,
                chunks_indexed=len(chunks),
                chunk_strategy=used_strategy,
                status="READY",
                progress=100,
                current_stage="READY",
                text_ready=True,
                vision_status="READY_ON_DEMAND",
                category=category,
                department=extracted_meta.department,
                effective_date=extracted_meta.effective_date,
                policy_id=extracted_meta.policy_id,
                key_entities=list(extracted_meta.key_entities),
                topic_tags=list(extracted_meta.topic_tags),
                created_at=created_at,
            )

        except Exception as exc:
            logger.exception("[INGESTION] FAILED for document_id=%s file=%s: %s", document_id, filename, exc)
            with self._lock:
                if document_id in self._ingestion_jobs:
                    j = self._ingestion_jobs[document_id]
                    j.status = "FAILED"
                    j.error = str(exc)
                    j.failed_stage = j.current_stage
                    j.can_retry = True
                    stages_dump = [s.model_dump() for s in j.stages]
                else:
                    stages_dump = []

            try:
                from backend.api.dependencies import get_telemetry_service
                get_telemetry_service().record_ingestion_trace(
                    document_id=document_id,
                    filename=filename,
                    category=category,
                    file_size_bytes=file_size,
                    status="FAILED",
                    current_stage="FAILED",
                    pages_count=0,
                    chunks_count=0,
                    visual_assets_count=0,
                    vision_success_count=0,
                    vision_failed_count=0,
                    total_duration_ms=round((time.perf_counter() - t_global_start) * 1000, 2),
                    error=str(exc),
                    stages=stages_dump,
                )
                get_telemetry_service().record_error(
                    component="Ingestion",
                    severity=SeverityLevel.ERROR,
                    message=f"Document ingestion failed for '{filename}': {exc!s}",
                    document_id=document_id,
                    stack_trace=str(exc),
                )
            except Exception:
                pass
            raise

    def upload_document(
        self,
        filename: str,
        content_bytes: bytes,
        category: str = "general",
        chunk_strategy: str | None = None,
    ) -> DocumentUploadResponse:
        """Process file upload, store persistently, and run core text ingestion."""
        file_size = len(content_bytes)
        if file_size > MAX_FILE_SIZE_BYTES:
            raise ValueError(f"File size ({file_size} bytes) exceeds maximum limit of 100MB.")

        if not filename or not content_bytes:
            raise ValueError("File content or filename cannot be empty.")

        content_hash = hashlib.sha256(content_bytes).hexdigest()
        document_id = f"doc_{uuid.uuid4().hex[:12]}"
        safe_filename = Path(filename).name
        target_path = self.storage_dir / f"{document_id}_{safe_filename}"

        with self._lock:
            for record in self._documents.values():
                existing_id = str(record.get("document_id") or "")
                existing_path = self._stored_files.get(existing_id)
                if (
                    record.get("file_hash") == content_hash
                    and existing_path is not None
                    and existing_path.is_file()
                ):
                    raise DuplicateDocumentError(existing_id, str(record.get("filename") or filename), content_hash)
            pending_id = self._pending_hashes.get(content_hash)
            if pending_id:
                raise DuplicateDocumentError(pending_id, safe_filename, content_hash)
            self._pending_hashes[content_hash] = document_id

        try:
            target_path.write_bytes(content_bytes)
            self._hash_stored_file(document_id, target_path)
            self._save_hash_catalog()

            with self._lock:
                self._stored_files[document_id] = target_path
                self._ingestion_jobs[document_id] = IngestionStatusResponse(
                    document_id=document_id,
                    job_id=f"job_{uuid.uuid4().hex[:10]}",
                    filename=filename,
                    status="TEXT_INDEXING",
                    progress=5,
                    current_stage="QUEUED",
                    text_ready=False,
                    created_at=datetime.now(UTC).isoformat(),
                    updated_at=datetime.now(UTC).isoformat(),
                )
        except Exception:
            with self._lock:
                self._pending_hashes.pop(content_hash, None)
            raise

        # Run the heavy pipeline (parse → chunk → embed → index) off the request
        # thread on the single-worker queue. A 100MB file can take minutes to
        # embed on CPU; blocking the HTTP response would trip the proxy/browser
        # timeout and freeze the UI. The client polls GET
        # /api/documents/{id}/status — which reflects live per-stage progress —
        # until the job reaches READY or FAILED.
        def _run_ingestion() -> None:
            try:
                self._execute_ingestion_stages(
                    document_id=document_id,
                    filename=filename,
                    file_path=target_path,
                    category=category,
                    chunk_strategy=chunk_strategy,
                )
            except Exception:
                # _execute_ingestion_stages already records the FAILED job state;
                # swallow here so the shared worker survives for the next upload.
                logger.exception("[INGESTION] background job crashed for document_id=%s", document_id)
            finally:
                with self._lock:
                    self._pending_hashes.pop(content_hash, None)

        self._ingestion_executor.submit(_run_ingestion)

        file_type = Path(filename).suffix.lstrip(".").lower() or "unknown"
        return DocumentUploadResponse(
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            file_size_bytes=file_size,
            file_hash=self._hash_catalog.get(document_id, {}).get("sha256", ""),
            pages_count=0,
            chunks_indexed=0,
            chunk_strategy=chunk_strategy or "auto",
            status="TEXT_INDEXING",
            progress=5,
            current_stage="QUEUED",
            text_ready=False,
            vision_status="PENDING",
            category=category,
            created_at=datetime.now(UTC).isoformat(),
        )

    def retry_document(self, document_id: str) -> IngestionStatusResponse:
        """Retry indexing for a previously uploaded document without re-uploading the file."""
        with self._lock:
            file_path = self._stored_files.get(document_id)
            if not file_path or not file_path.is_file():
                # Search disk for stored file matching document_id
                matches = list(self.storage_dir.glob(f"{document_id}_*"))
                if matches:
                    file_path = matches[0]
                    self._stored_files[document_id] = file_path

            if not file_path or not file_path.is_file():
                raise ValueError(f"No stored document file found for ID '{document_id}' to retry.")

            filename = file_path.name.replace(f"{document_id}_", "")
            job = IngestionStatusResponse(
                document_id=document_id,
                job_id=f"job_{uuid.uuid4().hex[:10]}",
                filename=filename,
                status="TEXT_INDEXING",
                progress=10,
                current_stage="UPLOAD",
                text_ready=False,
                created_at=datetime.now(UTC).isoformat(),
                updated_at=datetime.now(UTC).isoformat(),
            )
            self._ingestion_jobs[document_id] = job

        # Re-run ingestion on the background queue (same reasoning as upload):
        # a large document must not block the retry request. The client polls
        # /status for progress until READY/FAILED.
        retry_path = file_path

        def _run_retry() -> None:
            try:
                self._execute_ingestion_stages(
                    document_id=document_id,
                    filename=filename,
                    file_path=retry_path,
                )
            except Exception:
                logger.exception("[INGESTION] background retry crashed for document_id=%s", document_id)

        self._ingestion_executor.submit(_run_retry)

        return job

    def list_documents(
        self,
        category: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> DocumentListResponse:
        """Return list of all registered documents."""
        with self._lock:
            records = list(self._documents.values())

        duplicate_groups = self.get_duplicate_groups()

        if category:
            records = [r for r in records if r.get("category") == category]

        records.sort(key=lambda record: (str(record.get("filename", "")).casefold(), str(record.get("created_at", ""))))
        paginated = records[offset : offset + limit]

        summaries = [
            DocumentSummary(
                document_id=r["document_id"],
                filename=r["filename"],
                file_type=r["file_type"],
                file_size_bytes=r["file_size_bytes"],
                file_hash=r.get("file_hash", ""),
                pages_count=r.get("pages_count", 0),
                storage_state=r.get("storage_state", "HEALTHY"),
                chunk_count=r["chunk_count"],
                category=r["category"],
                department=r.get("department"),
                effective_date=r.get("effective_date"),
                policy_id=r.get("policy_id"),
                topic_tags=r.get("topic_tags", []),
                key_entities=r.get("key_entities", []),
                created_at=r["created_at"],
                status=r.get("status", "READY"),
                progress=r.get("progress", 100),
                current_stage=r.get("current_stage", "READY"),
                text_ready=r.get("text_ready", True),
                vision_status=r.get("vision_status", "NONE"),
                vision_pages_processed=r.get("vision_pages_processed", 0),
                vision_pages_total=r.get("vision_pages_total", 0),
                error=r.get("error"),
            )
            for r in paginated
        ]

        return DocumentListResponse(
            documents=summaries,
            total_count=len(records),
            duplicate_groups=len(duplicate_groups),
            duplicate_documents=sum(group["duplicate_count"] for group in duplicate_groups),
        )

    def get_document_detail(self, document_id: str) -> DocumentDetailResponse | None:
        """Get full details and chunk metadata for a document."""
        with self._lock:
            r = self._documents.get(document_id)
        if not r:
            return None

        return DocumentDetailResponse(
            document_id=r["document_id"],
            filename=r["filename"],
            file_type=r["file_type"],
            file_size_bytes=r["file_size_bytes"],
            file_hash=r.get("file_hash", ""),
            pages_count=r.get("pages_count", 0),
            storage_state=r.get("storage_state", "HEALTHY"),
            chunk_count=r["chunk_count"],
            category=r["category"],
            department=r.get("department"),
            effective_date=r.get("effective_date"),
            policy_id=r.get("policy_id"),
            topic_tags=r.get("topic_tags", []),
            key_entities=r.get("key_entities", []),
            created_at=r["created_at"],
            status=r.get("status", "READY"),
            progress=r.get("progress", 100),
            current_stage=r.get("current_stage", "READY"),
            text_ready=r.get("text_ready", True),
            vision_status=r.get("vision_status", "NONE"),
            vision_pages_processed=r.get("vision_pages_processed", 0),
            vision_pages_total=r.get("vision_pages_total", 0),
            error=r.get("error"),
            failed_stage=r.get("failed_stage"),
            chunks=r.get("chunks", []),
        )

    def delete_document(self, document_id: str) -> dict[str, Any] | None:
        """Remove document and purge all associated vector embeddings and BM25 postings."""
        with self._lock:
            doc_record = self._documents.get(document_id)
            if not doc_record:
                return None
            filename = doc_record["filename"]

        # 1. Purge from Vector Store. Legacy chunks without a usable document_id
        # can only be scoped reliably by their persisted source filename.
        legacy_source_file = doc_record.get("_legacy_source_file")
        if legacy_source_file:
            self.vector_store.delete_by_source(str(legacy_source_file))
        else:
            self.vector_store.delete_by_document_id(document_id)

        # 2. Purge from BM25 Index
        self.bm25_index.remove_by_document_id(document_id)
        self.bm25_index.save()

        # 3. Purge from Docstore
        chunk_ids_to_del = [
            cid for cid, chunk in list(self.docstore.items())
            if chunk.metadata.document_id == document_id
        ]
        for cid in chunk_ids_to_del:
            self.docstore.pop(cid, None)

        # 4. Purge derived image assets and visual response cache.
        deleted_assets = self.image_asset_manager.delete_document_assets(document_id)
        deleted_vision_cache = self.vision_cache_manager.delete_by_document_id(document_id)

        # 5. Purge from Registry, job store, hash catalog, and every exact source-file match.
        with self._lock:
            self._documents.pop(document_id, None)
            self._ingestion_jobs.pop(document_id, None)
            self._stored_files.pop(document_id, None)
            self._hash_catalog.pop(document_id, None)
            self._save_hash_catalog()

        deleted_source_files = 0
        storage_root = self.storage_dir.resolve()
        for file_path in self.storage_dir.glob(f"{document_id}_*"):
            if file_path.is_file() and file_path.resolve().parent == storage_root:
                file_path.unlink()
                deleted_source_files += 1

        logger.info("Deleted document id=%s, filename=%s (%d chunks purged)", document_id, filename, len(chunk_ids_to_del))

        return {
            "status": "deleted",
            "document_id": document_id,
            "filename": filename,
            "deleted_chunks": len(chunk_ids_to_del),
            "deleted_source_files": deleted_source_files,
            "deleted_assets": deleted_assets,
            "deleted_vision_cache_entries": deleted_vision_cache,
        }
