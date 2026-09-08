from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from backend.api.dependencies import reset_dependencies
from backend.api.main import _SuccessfulPollingAccessFilter, app


@pytest.fixture(autouse=True)
def cleanup_deps():
    reset_dependencies()
    yield
    reset_dependencies()


def test_get_admin_observability():
    client = TestClient(app)
    response = client.get("/api/admin/observability")
    assert response.status_code == 200
    data = response.json()
    assert "total_queries" in data
    assert "avg_latency_ms" in data
    assert "token_usage" in data
    assert "recent_traces" in data


def test_successful_observability_poll_access_logs_are_quiet_but_errors_remain():
    access_filter = _SuccessfulPollingAccessFilter()

    def record(path: str, status_code: int) -> logging.LogRecord:
        return logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:1234", "GET", path, "1.1", status_code),
            exc_info=None,
        )

    assert access_filter.filter(record("/api/admin/observability?time_range=24h", 200)) is False
    assert access_filter.filter(record("/api/admin/observability?time_range=24h", 500)) is True
    assert access_filter.filter(record("/api/chat", 200)) is True


def test_get_admin_traces():
    client = TestClient(app)
    # Execute a query first to generate a trace
    client.post("/api/chat", json={"message": "What is the parental leave policy?"})

    response = client.get("/api/admin/traces")
    assert response.status_code == 200
    data = response.json()
    assert "traces" in data
    assert "total_count" in data
    assert len(data["traces"]) >= 1


def test_get_admin_trace_detail_success_and_404():
    client = TestClient(app)

    # 1. Execute query to record trace
    chat_resp = client.post("/api/chat", json={"message": "What is the code of conduct?"})
    assert chat_resp.status_code == 200

    # 2. Get trace list to retrieve trace_id
    traces_resp = client.get("/api/admin/traces")
    assert traces_resp.status_code == 200
    traces = traces_resp.json()["traces"]
    assert len(traces) > 0
    trace_id = traces[0]["trace_id"]

    # 3. Retrieve trace detail
    detail_resp = client.get(f"/api/admin/traces/{trace_id}")
    assert detail_resp.status_code == 200
    detail_data = detail_resp.json()
    assert detail_data["trace"]["trace_id"] == trace_id

    # 4. Invalid trace_id 404 check
    invalid_resp = client.get("/api/admin/traces/tr_nonexistent999")
    assert invalid_resp.status_code == 404
