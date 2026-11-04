from fastapi.testclient import TestClient

from backend.api.main import create_app
from src.config import settings


def test_untrusted_origin_cannot_mutate_even_with_simple_upload_request():
    client = TestClient(create_app(), client=("127.0.0.1", 50000))
    response = client.post("/api/chat", headers={"Origin": "https://untrusted.example"}, json={"message": "hi"})
    assert response.status_code == 403


def test_remote_admin_is_denied_without_operator_key(monkeypatch):
    monkeypatch.setattr(settings, "admin_token", "")
    client = TestClient(create_app(), client=("203.0.113.10", 50000))
    assert client.get("/api/admin/storage").status_code == 403


def test_operator_key_is_required_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "admin_token", "test-secret")
    app = create_app()
    from backend.api.dependencies import get_storage_service
    from types import SimpleNamespace

    app.dependency_overrides[get_storage_service] = lambda: SimpleNamespace(live=lambda: {"status": "ok"})
    client = TestClient(app, client=("127.0.0.1", 50000))
    assert client.get("/api/admin/storage/live").status_code == 403
    assert client.get("/api/admin/storage/live", headers={"Authorization": "Bearer wrong"}).status_code == 403
    assert client.get("/api/admin/storage/live", headers={"Authorization": "Bearer test-secret"}).status_code == 200


def test_unknown_host_is_rejected():
    client = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert client.get("/", headers={"Host": "untrusted.example"}).status_code == 400
