from __future__ import annotations

import threading
from typing import Optional

from backend.embeddings.embeddings import EmbeddingService
from backend.embeddings.vector_store import ChromaVectorStore
from backend.rag.pipeline import RAGPipeline
from backend.retrieval.bm25 import BM25SearchIndex
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.vector import DenseVectorRetriever
from backend.services.chat_service import ChatService
from backend.services.document_service import DocumentService
from backend.services.telemetry_service import TelemetryService

_lock = threading.RLock()

_telemetry_service: Optional[TelemetryService] = None
_document_service: Optional[DocumentService] = None
_rag_pipeline: Optional[RAGPipeline] = None
_chat_service: Optional[ChatService] = None


def get_telemetry_service() -> TelemetryService:
    global _telemetry_service
    if _telemetry_service is None:
        with _lock:
            if _telemetry_service is None:
                _telemetry_service = TelemetryService()
    return _telemetry_service


def get_document_service() -> DocumentService:
    global _document_service
    if _document_service is None:
        with _lock:
            if _document_service is None:
                _document_service = DocumentService()
    return _document_service


def get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        with _lock:
            if _rag_pipeline is None:
                doc_service = get_document_service()
                vector_store = doc_service.vector_store
                bm25_index = doc_service.bm25_index
                embedding_service = doc_service.embedding_service

                dense_retriever = DenseVectorRetriever(
                    vector_store=vector_store,
                    embedding_service=embedding_service,
                )
                hybrid_retriever = HybridRetriever(
                    dense_retriever=dense_retriever,
                    bm25_index=bm25_index,
                )
                _rag_pipeline = RAGPipeline(
                    hybrid_retriever=hybrid_retriever,
                    docstore=doc_service.docstore,
                )
    return _rag_pipeline


def get_chat_service() -> ChatService:
    global _chat_service
    if _chat_service is None:
        with _lock:
            if _chat_service is None:
                pipeline = get_rag_pipeline()
                telemetry = get_telemetry_service()
                _chat_service = ChatService(
                    rag_pipeline=pipeline,
                    telemetry_service=telemetry,
                )
    return _chat_service


def reset_dependencies() -> None:
    """Reset singletons (useful for test isolation)."""
    global _telemetry_service, _document_service, _rag_pipeline, _chat_service
    with _lock:
        _telemetry_service = None
        _document_service = None
        _rag_pipeline = None
        _chat_service = None
