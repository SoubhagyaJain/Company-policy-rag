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


def test_post_chat_multi_turn_conversational_memory():
    client = TestClient(app)
    session_id = "mem_test_session_1"

    # Turn 1
    p1 = {
        "message": "What is the employee annual leave policy?",
        "session_id": session_id,
    }
    r1 = client.post("/api/chat", json=p1)
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["session_id"] == session_id

    # Turn 2 (Follow-up with pronoun)
    p2 = {
        "message": "Does it apply to part-time employees?",
        "session_id": session_id,
    }
    r2 = client.post("/api/chat", json=p2)
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["session_id"] == session_id

    # Verify ChatService internal session tracking
    chat_svc = get_chat_service()
    history = chat_svc._sessions.get(session_id, [])
    assert len(history) == 4  # User1, Assistant1, User2, Assistant2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "What is the employee annual leave policy?"
    assert history[2]["role"] == "user"
    assert history[2]["content"] == "Does it apply to part-time employees?"


def test_post_chat_model_parameter():
    client = TestClient(app)
    payload = {
        "message": "What is the resignation notice period?",
        "model": "llama3.1:8b",
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "llama3.1:8b"


def test_post_chat_semantic_cache_hit_and_miss():
    client = TestClient(app)
    payload = {
        "message": "What is the policy for parental leave?",
        "session_id": "cache_test_sess_1",
    }
    # First request: Cache Miss
    r1 = client.post("/api/chat", json=payload)
    assert r1.status_code == 200
    d1 = r1.json()
    assert "trace" in d1
    assert d1["trace"]["cache_hit"] is False

    # Second request: Cache Hit (if response had citations)
    r2 = client.post("/api/chat", json=payload)
    assert r2.status_code == 200
    d2 = r2.json()
    assert "trace" in d2
    if len(d1.get("citations", [])) > 0:
        assert d2["trace"]["cache_hit"] is True
        assert d2["answer"] == d1["answer"]


