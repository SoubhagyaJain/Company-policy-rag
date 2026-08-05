from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.dependencies import reset_dependencies
from backend.api.main import app


@pytest.fixture(autouse=True)
def cleanup_deps():
    reset_dependencies()
    yield
    reset_dependencies()


def test_get_health():
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["vector_db"] is True
    assert data["bm25_index"] is True
    assert data["models_loaded"] is True
    assert "timestamp" in data


def test_get_models():
    client = TestClient(app)
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "active_model" in data
    assert "models" in data
    assert len(data["models"]) >= 1


def test_select_active_model_success():
    client = TestClient(app)
    response = client.post("/api/models/select", json={"model": "qwen2.5:7b"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "switched"
    assert data["active_model"] == "qwen2.5:7b"


def test_select_active_model_invalid():
    client = TestClient(app)
    response = client.post("/api/models/select", json={"model": "non_existent_model_xyz"})
    assert response.status_code == 400
    assert "detail" in response.json()
