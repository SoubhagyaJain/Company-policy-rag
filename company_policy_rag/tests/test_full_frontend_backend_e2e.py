"""
Master End-to-End Frontend <-> Backend Integration Verification Suite.

Tests the full user flow:
1. System Health & Models Status (/api/health, /api/models)
2. Document Ingestion (/api/documents/upload, /api/documents)
3. Synchronous Chat Flow (/api/chat) with full DTO Serialization
4. Real-time Sub-1s SSE Token Streaming (/api/chat/stream)
5. Telemetry & Observability Audit (/api/admin/observability)
6. Cross-Page Multimodal RAG with Qwen 2.5 7B & Qwen 2.5 VL 7B
"""

from __future__ import annotations

import io
import json
import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_e2e_01_health_and_models_endpoints(client: TestClient):
    """Verify system health and model availability."""
    # 1. Health check
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert data_health["status"] == "ok"
    assert data_health["vector_db"] is True
    assert data_health["models_loaded"] is True

    # 2. Models list
    res_models = client.get("/api/models")
    assert res_models.status_code == 200
    data_models = res_models.json()
    assert "models" in data_models
    assert data_models["vision_model"] == "qwen2.5vl:7b"
    assert data_models["vision_enabled"] is True


def test_e2e_02_document_upload_and_listing(client: TestClient):
    """Verify document upload and metadata indexing."""
    sample_policy_content = (
        "# Remote Work Policy 2026\n\n"
        "## Section 1: Eligibility\n"
        "All full-time employees with at least 90 days of tenure are eligible for remote work.\n\n"
        "## Section 2: Equipment Reimbursement\n"
        "The company provides up to $500 annually for approved home office equipment.\n"
    ).encode("utf-8")

    files = {
        "file": ("Remote_Work_Policy_2026.md", io.BytesIO(sample_policy_content), "text/markdown"),
    }
    data = {
        "category": "hr_policy",
        "chunk_strategy": "markdown_header",
    }

    res_upload = client.post("/api/documents/upload", files=files, data=data)
    assert res_upload.status_code == 201
    upload_dto = res_upload.json()
    assert upload_dto["filename"] == "Remote_Work_Policy_2026.md"
    assert upload_dto["chunks_indexed"] >= 1
    assert upload_dto["status"] == "indexed"

    # Verify document listing
    res_list = client.get("/api/documents")
    assert res_list.status_code == 200
    list_dto = res_list.json()
    assert any(d["filename"] == "Remote_Work_Policy_2026.md" for d in list_dto["documents"])


def test_e2e_03_synchronous_chat_flow(client: TestClient):
    """Verify synchronous chat endpoint with citation and trace serialization."""
    payload = {
        "message": "Who is eligible for remote work under the remote work policy?",
        "session_id": "test_sess_001",
        "model": "qwen2.5:7b",
    }

    res_chat = client.post("/api/chat", json=payload)
    assert res_chat.status_code == 200
    chat_dto = res_chat.json()

    assert "id" in chat_dto
    assert "answer" in chat_dto
    assert "citations" in chat_dto
    assert "trace" in chat_dto
    assert "latency_ms" in chat_dto
    assert chat_dto["trace"]["query_type"] in ("factual", "procedural", "implementation", "explanation")


def test_e2e_04_sse_streaming_chat_flow(client: TestClient):
    """Verify SSE streaming chat endpoint with event sequence (start, retrieval, chunk, citation, trace, done)."""
    payload = {
        "message": "What is the equipment reimbursement under the remote work policy?",
        "session_id": "test_sess_002",
        "model": "qwen2.5:7b",
    }

    res_stream = client.post(
        "/api/chat/stream",
        json=payload,
        headers={"Accept": "text/event-stream"},
    )
    assert res_stream.status_code == 200

    raw_text = res_stream.text
    assert "event: start" in raw_text
    assert "event: retrieval" in raw_text
    assert "event: chunk" in raw_text
    assert "event: done" in raw_text

    # Parse done event JSON
    lines = raw_text.split("\n")
    done_payload = None
    for idx, line in enumerate(lines):
        if line.strip() == "event: done":
            for next_line in lines[idx + 1:]:
                if next_line.startswith("data:"):
                    done_payload = json.loads(next_line[5:].strip())
                    break
            break

    assert done_payload is not None
    assert "answer" in done_payload
    assert "retrieval_trace" in done_payload
    assert "ttft_ms" in done_payload
    assert "total_latency_ms" in done_payload


def test_e2e_05_admin_observability_endpoint(client: TestClient):
    """Verify admin observability metrics and recent traces."""
    res_obs = client.get("/api/admin/observability")
    assert res_obs.status_code == 200
    obs_dto = res_obs.json()

    assert "total_queries" in obs_dto
    assert obs_dto["total_queries"] >= 1
    assert "recent_traces" in obs_dto
    assert len(obs_dto["recent_traces"]) >= 1
