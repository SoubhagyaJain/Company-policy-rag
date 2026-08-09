from __future__ import annotations

import os
import threading

from dotenv import load_dotenv

try:
    from llama_index.llms.ollama import Ollama
except Exception:
    Ollama = None

from backend.rag.pipeline import RAGPipeline
from backend.rag.semantic_cache import SemanticCacheManager
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.reranker import CrossEncoderReranker
from backend.retrieval.vector import DenseVectorRetriever
from backend.services.chat_service import ChatService
from backend.services.document_service import DocumentService
from backend.services.telemetry_service import TelemetryService

load_dotenv()

_lock = threading.RLock()

_telemetry_service: TelemetryService | None = None
_document_service: DocumentService | None = None
_semantic_cache_manager: SemanticCacheManager | None = None
_rag_pipeline: RAGPipeline | None = None
_chat_service: ChatService | None = None


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


def get_semantic_cache_manager() -> SemanticCacheManager:
    global _semantic_cache_manager
    if _semantic_cache_manager is None:
        with _lock:
            if _semantic_cache_manager is None:
                doc_service = get_document_service()
                _semantic_cache_manager = SemanticCacheManager(
                    vector_store=doc_service.vector_store,
                    embedding_service=doc_service.embedding_service,
                )
    return _semantic_cache_manager


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
                # Initialize Reranker
                reranker_device = os.getenv("RERANKER_DEVICE", "cpu")
                reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")
                reranker_top_n = int(os.getenv("RERANKER_TOP_N", "5"))
                reranker_min_ratio = float(os.getenv("RERANK_MIN_SCORE_RATIO", "0.40"))
                
                reranker = CrossEncoderReranker(
                    model_name=reranker_model,
                    top_n=reranker_top_n,
                    device=reranker_device,
                    min_ratio=reranker_min_ratio
                )

                hybrid_retriever = HybridRetriever(
                    dense_retriever=dense_retriever,
                    bm25_index=bm25_index,
                    reranker=reranker,
                )
                
                # Initialize LLM
                ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                ollama_model = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:14b-instruct")
                temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
                request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "300.0"))
                
                llm = None
                global Ollama
                if Ollama is None:
                    try:
                        from llama_index.llms.ollama import Ollama
                    except Exception:
                        Ollama = None

                if Ollama is not None:
                    try:
                        llm = Ollama(
                            base_url=ollama_url,
                            model=ollama_model,
                            temperature=temperature,
                            request_timeout=request_timeout,
                        )
                    except Exception:
                        llm = None
                
                cache_manager = get_semantic_cache_manager()
                _rag_pipeline = RAGPipeline(
                    hybrid_retriever=hybrid_retriever,
                    reranker=reranker,
                    docstore=doc_service.docstore,
                    llm=llm,
                    semantic_cache=cache_manager,
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
    global _telemetry_service, _document_service, _rag_pipeline, _chat_service, _semantic_cache_manager
    with _lock:
        if _semantic_cache_manager is not None:
            try:
                _semantic_cache_manager.clear()
            except Exception:
                pass
        _telemetry_service = None
        _document_service = None
        _rag_pipeline = None
        _chat_service = None
        _semantic_cache_manager = None


