from __future__ import annotations

import io
def test_upload_document_text_file(isolated_document_client):
    client = isolated_document_client
    file_content = b"Section 1: Work Hours\nStandard working hours are 9:00 AM to 5:00 PM Monday through Friday."
    files = {"file": ("work_hours_policy.txt", io.BytesIO(file_content), "text/plain")}
    data = {"category": "policy", "chunk_strategy": "recursive"}

    response = client.post("/api/documents/upload", files=files, data=data)
    assert response.status_code == 201

    res = response.json()
    assert "document_id" in res
    assert res["filename"] == "work_hours_policy.txt"
    assert res["chunks_indexed"] > 0
    assert res["status"] == "READY"
    assert res["category"] == "policy"

    doc_id = res["document_id"]

    # Test GET /api/documents
    list_resp = client.get("/api/documents")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total_count"] >= 1
    doc_ids = [d["document_id"] for d in list_data["documents"]]
    assert doc_id in doc_ids

    # Test GET /api/documents/{doc_id}
    detail_resp = client.get(f"/api/documents/{doc_id}")
    assert detail_resp.status_code == 200
    detail_data = detail_resp.json()
    assert detail_data["document_id"] == doc_id
    assert detail_data["filename"] == "work_hours_policy.txt"
    assert len(detail_data["chunks"]) > 0

    # Test DELETE /api/documents/{doc_id}
    del_resp = client.delete(f"/api/documents/{doc_id}")
    assert del_resp.status_code == 200
    del_data = del_resp.json()
    assert del_data["status"] == "deleted"
    assert del_data["document_id"] == doc_id

    # Verify document no longer exists
    get_again = client.get(f"/api/documents/{doc_id}")
    assert get_again.status_code == 404


def test_upload_markdown_file_adaptive(isolated_document_client):
    client = isolated_document_client
    md_content = b"# HR Policy\n\n## Leave Policy\nEmployees are entitled to 20 days paid leave per annum."
    files = {"file": ("hr_policy.md", io.BytesIO(md_content), "text/markdown")}

    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 201
    res = response.json()
    assert res["filename"] == "hr_policy.md"
    assert res["chunks_indexed"] >= 1


def test_upload_oversized_file_rejected(isolated_document_client):
    client = isolated_document_client
    # Create fake large file buffer > 100MB header simulation or content check
    # Instead of allocating 101MB RAM in test, test file content size validation logic directly or using small buffer mock if needed
    # Test client sending 100MB+ header or content
    large_content = b"0" * (100 * 1024 * 1024 + 1)
    files = {"file": ("huge_file.txt", io.BytesIO(large_content), "text/plain")}

    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 413
    assert "100MB" in response.json()["detail"]


def test_get_nonexistent_document(isolated_document_client):
    client = isolated_document_client
    response = client.get("/api/documents/non_existent_doc_id")
    assert response.status_code == 404


def test_delete_nonexistent_document(isolated_document_client):
    client = isolated_document_client
    response = client.delete("/api/documents/non_existent_doc_id")
    assert response.status_code == 404
