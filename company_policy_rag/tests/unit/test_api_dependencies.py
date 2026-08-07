from __future__ import annotations

import pytest

from backend.api.dependencies import (
    get_chat_service,
    get_document_service,
    get_rag_pipeline,
    get_telemetry_service,
    reset_dependencies,
)
from backend.rag.pipeline import RAGPipeline
from backend.services.chat_service import ChatService
from backend.services.document_service import DocumentService
from backend.services.telemetry_service import TelemetryService


@pytest.fixture(autouse=True)
def cleanup_deps():
    reset_dependencies()
    yield
    reset_dependencies()


def test_get_telemetry_service():
    service = get_telemetry_service()
    assert isinstance(service, TelemetryService)
    assert get_telemetry_service() is service


def test_get_document_service():
    service = get_document_service()
    assert isinstance(service, DocumentService)
    assert get_document_service() is service


def test_get_rag_pipeline_instantiation():
    pipeline = get_rag_pipeline()
    assert isinstance(pipeline, RAGPipeline)
    assert pipeline.hybrid_retriever is not None
    assert pipeline.hybrid_retriever.dense_retriever is not None
    assert pipeline.hybrid_retriever.bm25_index is not None
    assert get_rag_pipeline() is pipeline


def test_get_chat_service_instantiation():
    service = get_chat_service()
    assert isinstance(service, ChatService)
    assert service.pipeline is not None
    assert service.telemetry_service is not None
    assert get_chat_service() is service


def test_reset_dependencies():
    pipeline1 = get_rag_pipeline()
    chat_service1 = get_chat_service()
    reset_dependencies()
    pipeline2 = get_rag_pipeline()
    chat_service2 = get_chat_service()
    assert pipeline1 is not pipeline2
    assert chat_service1 is not chat_service2
