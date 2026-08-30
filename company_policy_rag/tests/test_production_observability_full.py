"""
Full Production Observability & Telemetry Test Suite.

Validates end-to-end telemetry persistence, aggregations, 10-subsystem health probes,
16-stage latency breakdowns, multi-model separation, multi-tier caches, grounding claims,
conversational bypass semantics, and REST API endpoints.
"""

from __future__ import annotations

import os
import tempfile
import time
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.models.telemetry_models import (
    ObservabilitySummary,
    QueryTraceRecord,
    SubsystemHealth,
    SubsystemStatus,
    QueryMetrics,
    LatencyBreakdown,
    RetrievalQualityMetrics,
    GroundingTelemetry,
    ModelTelemetrySummary,
    TextModelTelemetry,
    VisionTelemetry,
    TokenTelemetry,
    MemoryTelemetry,
    IngestionTelemetry,
    CacheTelemetry,
    CacheTypeMetrics,
    AlertItem,
    ErrorIncident,
    EvidenceItem,
    EvidenceContentType,
)
from backend.services.telemetry_db import TelemetryDB
from backend.services.telemetry_service import TelemetryService


@pytest.fixture
def temp_telemetry_db():
    """Create a temporary SQLite database for telemetry testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    db = TelemetryDB(db_path=db_path)
    yield db
    
    # Cleanup
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
        wal_file = db_path + "-wal"
        if os.path.exists(wal_file):
            os.remove(wal_file)
        shm_file = db_path + "-shm"
        if os.path.exists(shm_file):
            os.remove(shm_file)
    except Exception:
        pass


def test_telemetry_models_validation():
    """Verify strict type validation on canonical Pydantic models."""
    record = QueryTraceRecord(
        trace_id="tr_model_test_01",
        request_id="req_model_test_01",
        timestamp=datetime.now(timezone.utc).isoformat(),
        original_query="What is the remote work policy?",
        query_type="factual",
        routing_confidence=0.96,
        retrieval_strategy="balanced_hybrid",
        retrieval_required=True,
        conversational_bypass=False,
        evidence_required=True,
        candidate_count=4,
        top_rerank_score=0.92,
        rerank_latency_ms=35.0,
        total_latency_ms=480.0,
        ttft_ms=180.0,
        prompt_tokens=240,
        completion_tokens=110,
        model="qwen2.5:7b",
        vision_used=False,
        evidence_items=[
            EvidenceItem(
                chunk_id="chunk_01",
                document_id="doc_hr",
                source_file="HR_Policy.pdf",
                page_number=12,
                content_type=EvidenceContentType.TEXT,
                rerank_score=0.92,
                snippet="Remote work policy guidelines and stipend allowance.",
            )
        ],
    )

    assert record.trace_id == "tr_model_test_01"
    assert record.evidence_items[0].content_type == EvidenceContentType.TEXT
    assert record.evidence_items[0].rerank_score == 0.92


def test_empty_summary_reports_unmeasured_values_without_placeholders(temp_telemetry_db):
    """An empty telemetry store must not manufacture healthy-looking performance data."""
    service = TelemetryService(db_path=str(temp_telemetry_db.db_path))

    summary = service.get_observability_summary(time_range="1h")

    assert summary.query_metrics.total_queries == 0
    assert summary.latency_breakdown.query_classification_ms is None
    assert summary.latency_breakdown.generation_ms is None
    assert summary.grounding.grounding_status.value == "not_applicable"
    assert summary.grounding.supported_claims_pct is None
    assert summary.models.vision_model.visual_pages_detected == 0
    assert summary.models.vision_model.cache_hit_rate is None
    assert summary.memory.memory_hit_rate is None
    assert summary.memory.avg_memory_latency_ms is None
    assert summary.tokens.avg_system_prompt_tokens == 0
    assert summary.tokens.p95_prompt_tokens == 0


def test_clear_discards_pending_write_behind_records(temp_telemetry_db):
    """Clearing telemetry is durable even when records are still queued for SQLite."""
    db = temp_telemetry_db
    db.record_query_trace(
        QueryTraceRecord(
            trace_id="tr_pending_clear",
            request_id="req_pending_clear",
            original_query="This queued trace must be cleared",
        )
    )

    db.clear()
    time.sleep(0.2)

    assert db.get_trace_by_id_or_request_id("tr_pending_clear") is None
    assert db.compute_aggregates(time_range="1h")["total_queries"] == 0


def test_telemetry_db_aggregations_and_time_ranges(temp_telemetry_db):
    """Verify that TelemetryDB correctly aggregates metrics over time ranges."""
    db = temp_telemetry_db
    service = TelemetryService(db_path=str(db.db_path))

    # Insert 5 query records with known latencies & tokens
    for i in range(1, 6):
        record = QueryTraceRecord(
            trace_id=f"tr_agg_{i}",
            request_id=f"req_agg_{i}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            original_query=f"Query {i}",
            query_type="factual" if i % 2 == 0 else "procedural",
            routing_confidence=0.90 + (i * 0.01),
            retrieval_strategy="balanced_hybrid",
            retrieval_required=True,
            conversational_bypass=False,
            evidence_required=True,
            candidate_count=i * 2,
            top_rerank_score=0.85 + (i * 0.02),
            rerank_latency_ms=20.0 * i,
            execution_time_ms=100.0 * i,  # 100, 200, 300, 400, 500
            ttft_ms=50.0 * i,           # 50, 100, 150, 200, 250
            prompt_tokens=100 * i,
            completion_tokens=50 * i,
            model="qwen2.5:7b",
            faithfulness_passed=(i != 3), # 1 failed, 4 passed
            verification_score=0.90,
        )
        db.record_query_trace(record)

    # Sleep slightly to allow background writer thread to commit
    time.sleep(0.3)

    summary = service.get_observability_summary(time_range="1h")
    assert summary.query_metrics.total_queries == 5
    assert summary.query_metrics.avg_latency_ms == 300.0
    assert summary.query_metrics.p50_latency_ms == 300.0
    assert summary.query_metrics.avg_ttft_ms == 150.0
    assert summary.tokens.total_tokens == 1500 + 750


def test_conversational_bypass_telemetry(temp_telemetry_db):
    """Verify that conversational bypass records retrieval_required=False and 0 chunks."""
    db = temp_telemetry_db

    bypass_record = QueryTraceRecord(
        trace_id="tr_conv_01",
        request_id="req_conv_01",
        timestamp=datetime.now(timezone.utc).isoformat(),
        original_query="Hello! How are you?",
        query_type="conversational",
        routing_confidence=0.99,
        retrieval_strategy="conversational_bypass",
        retrieval_required=False,
        conversational_bypass=True,
        evidence_required=False,
        candidate_count=0,
        top_rerank_score=0.0,
        rerank_latency_ms=0.0,
        execution_time_ms=65.0,
        ttft_ms=30.0,
        prompt_tokens=40,
        completion_tokens=25,
        model="qwen2.5:7b",
        faithfulness_passed=True,
        verification_score=None,
        anchor_section="Conversational Bypass",
    )
    db.record_query_trace(bypass_record)
    time.sleep(0.3)

    fetched = db.get_trace_by_id_or_request_id("tr_conv_01")
    assert fetched is not None
    assert fetched.conversational_bypass is True
    assert fetched.retrieval_required is False
    assert fetched.evidence_required is False
    assert fetched.candidate_count == 0
    assert fetched.verification_score is None


def test_multi_model_monitoring_separation(temp_telemetry_db):
    """Verify that Text (qwen2.5:7b) and Vision (Qwen3-VL-2B-Instruct) metrics are kept separated."""
    db = temp_telemetry_db
    service = TelemetryService(db_path=str(db.db_path))

    # Record 1 text inference
    db.record_query_trace(
        QueryTraceRecord(
            trace_id="tr_text_only",
            request_id="req_text_only",
            timestamp=datetime.now(timezone.utc).isoformat(),
            original_query="What is HR vacation policy?",
            candidate_count=3,
            top_rerank_score=0.92,
            rerank_latency_ms=25.0,
            execution_time_ms=450.0,
            ttft_ms=160.0,
            prompt_tokens=200,
            completion_tokens=80,
            model="qwen2.5:7b",
            vision_used=False,
        )
    )

    # Record 1 vision event
    db.record_vision_event(
        document_id="doc_arch_guide",
        page_number=82,
        visual_type="DIAGRAM_ARCHITECTURE",
        status="SUCCESS",
        duration_ms=1200.0,
    )

    time.sleep(0.3)
    summary = service.get_observability_summary(time_range="1h")

    assert summary.models.text_model.model_name == "qwen2.5:7b"
    assert summary.models.vision_model.model_name == "Qwen3-VL-2B-Instruct"
    assert summary.models.vision_model.requests_count == 1
    assert summary.models.vision_model.visual_pages_detected == 1


def test_rest_api_endpoints():
    """Verify all FastAPI REST endpoints for production observability."""
    app = create_app()
    client = TestClient(app)

    # 1. Health endpoint
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert "status" in data_health
    assert "vector_db" in data_health

    # 2. Subsystem health breakdown endpoint
    res_subsystems = client.get("/api/admin/observability/health")
    assert res_subsystems.status_code == 200
    data_subsystems = res_subsystems.json()
    assert "api" in data_subsystems
    assert "vector_db" in data_subsystems
    assert "text_model" in data_subsystems
    assert "vision_model" in data_subsystems

    # 3. Observability summary endpoint
    res_summary = client.get("/api/admin/observability/summary?time_range=1h")
    assert res_summary.status_code == 200
    data_summary = res_summary.json()
    assert "query_metrics" in data_summary
    assert "latency_breakdown" in data_summary
    assert "retrieval_quality" in data_summary
    assert "grounding" in data_summary
    assert "models" in data_summary
    assert "caches" in data_summary
    assert "health" in data_summary

    # 4. Query traces list endpoint
    res_queries = client.get("/api/admin/observability/queries?limit=10")
    assert res_queries.status_code == 200
    data_queries = res_queries.json()
    assert "traces" in data_queries
    assert isinstance(data_queries["traces"], list)

    # 5. Error incidents endpoint
    res_errors = client.get("/api/admin/observability/errors?limit=10")
    assert res_errors.status_code == 200
    data_errors = res_errors.json()
    assert isinstance(data_errors, list)

    # 6. Legacy / current observability endpoint
    res_obs = client.get("/api/admin/observability")
    assert res_obs.status_code == 200
    data_obs = res_obs.json()
    assert "query_metrics" in data_obs or "total_queries" in data_obs
    assert "health" in data_obs
