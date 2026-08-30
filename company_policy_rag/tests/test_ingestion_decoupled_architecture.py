from backend.vision.vision_service import VisionCircuitBreaker
from backend.vision.vision_cache import VisionCacheManager

def test_fast_document_upload_and_ready_status(isolated_document_client):
    """Verify document uploads fast, marks READY immediately, and text RAG is active."""
    doc_content = b"""# Engineering Remote Work Policy

1. Overview
All engineering personnel are eligible for hybrid remote work up to 3 days per week.

2. Equipment & Security
Company laptops must run endpoint protection and VPN at all times.
"""
    response = isolated_document_client.post(
        "/api/documents/upload",
        files={"file": ("Engineering_Remote_Policy.md", doc_content, "text/markdown")},
        data={"category": "Engineering"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["document_id"].startswith("doc_")
    assert data["status"] == "READY"
    assert data["progress"] == 100
    assert data["text_ready"] is True
    assert data["chunks_indexed"] > 0

    # Verify status endpoint
    doc_id = data["document_id"]
    status_res = isolated_document_client.get(f"/api/documents/{doc_id}/status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["document_id"] == doc_id
    assert status_data["status"] == "READY"
    assert status_data["progress"] == 100
    assert status_data["text_ready"] is True
    assert len(status_data["stages"]) > 0

def test_document_retry_endpoint(isolated_document_client):
    """Verify POST /api/documents/{doc_id}/retry successfully re-indexes the document."""
    doc_content = b"""# Code Review Standard 2026

All pull requests require 2 passing approvals and CI green.
"""
    response = isolated_document_client.post(
        "/api/documents/upload",
        files={"file": ("Code_Review_Standard.txt", doc_content, "text/plain")},
        data={"category": "Engineering"},
    )
    assert response.status_code == 201
    doc_id = response.json()["document_id"]

    # Call retry endpoint
    retry_res = isolated_document_client.post(f"/api/documents/{doc_id}/retry")
    assert retry_res.status_code == 200
    retry_data = retry_res.json()
    assert retry_data["document_id"] == doc_id
    assert retry_data["status"] == "READY"
    assert retry_data["progress"] == 100

def test_vision_circuit_breaker_and_failure_isolation():
    """Verify circuit breaker trips on consecutive failures and prevents system freeze."""
    cb = VisionCircuitBreaker(failure_threshold=3, recovery_cooldown=2.0)
    assert cb.allow_request() is True

    # Record 3 failures
    cb.record_failure()
    assert cb.allow_request() is True
    cb.record_failure()
    assert cb.allow_request() is True
    cb.record_failure()
    # Now circuit breaker should be open
    assert cb.allow_request() is False

    # After recovery cooldown, half-open allows a probe
    import time
    time.sleep(2.1)
    assert cb.allow_request() is True

    # On success, reset
    cb.record_success()
    assert cb.allow_request() is True
    assert cb.failure_count == 0

def test_vision_negative_cache():
    """Verify failed image extractions are negatively cached to prevent infinite timeout loops."""
    cache = VisionCacheManager()
    dummy_bytes = b"bad_broken_image_bytes_12345"
    img_hash = VisionCacheManager.compute_image_hash(dummy_bytes)

    # Initially not failed
    is_failed, _ = cache.is_failed(img_hash)
    assert is_failed is False

    # Mark failed
    cache.mark_failed(img_hash, "Ollama connection timeout after 35s")
    is_failed, reason = cache.is_failed(img_hash)
    assert is_failed is True
    assert "timeout" in reason
