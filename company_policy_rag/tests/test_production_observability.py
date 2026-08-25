from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.models.telemetry_models import (
    ErrorIncident,
    ObservabilitySummary,
    QueryTraceRecord,
    SeverityLevel,
    SubsystemStatus,
)
from backend.services.telemetry_db import TelemetryDB
from backend.services.telemetry_service import TelemetryService


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_telemetry.sqlite3"


@pytest.fixture
def telemetry_service(test_db_path: Path) -> TelemetryService:
    return TelemetryService(db_path=test_db_path)


def test_telemetry_db_crud(test_db_path: Path):
    db = TelemetryDB(db_path=test_db_path)

    # 1. Record Trace
    trace = QueryTraceRecord(
        trace_id="tr_test_1",
        request_id="req_test_1",
        conversation_id="sess_1",
        document_id="doc_1",
        original_query="What is the remote work policy?",
        query_type="factual",
        retrieval_required=True,
        conversational_bypass=False,
        evidence_required=True,
        candidate_count=5,
        post_rerank_count=3,
        final_chunk_count=3,
        execution_time_ms=850.5,
        ttft_ms=320.0,
        prompt_tokens=150,
        completion_tokens=85,
        total_tokens=235,
        generation_model="qwen2.5:7b",
    )
    db.record_query_trace(trace)
    import time
    time.sleep(0.1)  # Allow background writer to flush

    # 2. Retrieve Trace
    fetched = db.get_trace_by_id_or_request_id("req_test_1")
    assert fetched is not None
    assert fetched.trace_id == "tr_test_1"
    assert fetched.original_query == "What is the remote work policy?"
    assert fetched.execution_time_ms == 850.5

    # 3. Filtered Query
    traces, total = db.get_filtered_traces(time_range="24h", intent="factual")
    assert total >= 1
    assert any(t.trace_id == "tr_test_1" for t in traces)

    # 4. Record Error Incident
    incident = ErrorIncident(
        incident_id="err_test_1",
        request_id="req_test_1",
        component="Vision",
        severity=SeverityLevel.ERROR,
        message="Vision model timeout after 35s",
        duration_ms=35000.0,
    )
    db.record_error_incident(incident)
    time.sleep(0.1)

    incidents = db.get_recent_incidents(time_range="24h", component="vision")
    assert len(incidents) >= 1
    assert incidents[0].incident_id == "err_test_1"


def test_observability_summary_endpoint():
    app = create_app()
    client = TestClient(app)

    response = client.get("/api/admin/observability?time_range=24h")
    assert response.status_code == 200
    data = response.json()

    assert "health" in data
    assert "query_metrics" in data
    assert "latency_breakdown" in data
    assert "retrieval_quality" in data
    assert "grounding" in data
    assert "models" in data
    assert "tokens" in data
    assert "memory" in data
    assert "ingestion" in data
    assert "caches" in data
    assert "alerts" in data
    assert "recent_traces" in data

    # Verify Health Subsystems
    health = data["health"]
    assert "api" in health
    assert "ollama" in health
    assert "vector_db" in health
    assert "bm25" in health
    assert "text_model" in health
    assert "vision_model" in health
    assert "semantic_cache" in health
    assert "vision_cache" in health
    assert "memory" in health


def test_subsystem_health_endpoint():
    app = create_app()
    client = TestClient(app)

    response = client.get("/api/admin/observability/health")
    assert response.status_code == 200
    data = response.json()
    assert data["api"] == "healthy"
    assert data["vector_db"] == "healthy"
    assert "active_model_text" in data
    assert "active_model_vision" in data


def test_conversational_bypass_semantics():
    app = create_app()
    client = TestClient(app)

    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 200
    data = response.json()

    assert "Hello" in data["answer"]
    trace = data["trace"]
    assert trace["retrieval_strategy"] == "conversational_bypass"
    assert trace["retrieved_candidate_count"] == 0
    assert trace["faithfulness_checked"] is False
    assert trace["verification_score"] is None
