from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.dependencies import get_chat_service, reset_dependencies
from backend.api.main import app


@pytest.fixture(autouse=True)
def cleanup_deps():
    reset_dependencies()
    yield
    reset_dependencies()


def test_post_chat_success():
    client = TestClient(app)
    payload = {
        "message": "What is the employee annual leave policy?",
        "session_id": "test_session_123",
        "grounding_mode": "balanced",
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "id" in data
    assert "message_id" in data
    assert data["session_id"] == "test_session_123"
    assert data["query"] == "What is the employee annual leave policy?"
    assert "answer" in data
    assert isinstance(data["citations"], list)
    assert "latency_ms" in data
    assert "metrics" in data


def test_post_chat_empty_message():
    client = TestClient(app)
    payload = {"message": "   "}
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_post_chat_auto_generates_session_id():
    client = TestClient(app)
    payload = {"message": "Tell me about remote work policies."}
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"].startswith("sess_")


def test_post_chat_with_category_filter():
    client = TestClient(app)
    payload = {
        "message": "What are the security guidelines?",
        "filters": {"category": "policy"},
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What are the security guidelines?"
