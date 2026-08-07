from __future__ import annotations

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytest
from fastapi.testclient import TestClient

from backend.api.dependencies import reset_dependencies
from backend.api.main import app
from backend.embeddings.vector_store import ChromaVectorStore


@pytest.fixture(autouse=True)
def setup_ephemeral_chroma(monkeypatch, tmp_path):
    """Ensure tests run against isolated ephemeral vector store to avoid SQLite file locks."""
    import chromadb
    def _ephemeral_init(self):
        try:
            client = chromadb.Client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            self._collection = None

    monkeypatch.setattr(ChromaVectorStore, "_init_chroma", _ephemeral_init)
    reset_dependencies()
    yield
    reset_dependencies()


# ============================================================================
# 1. Malformed & Edge-Case Payloads (/api/chat and /api/chat/stream)
# ============================================================================

def test_chat_empty_and_whitespace_payload():
    client = TestClient(app)
    # Synchronous endpoint empty string
    res = client.post("/api/chat", json={"message": ""})
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()

    # Synchronous endpoint whitespace only
    res_space = client.post("/api/chat", json={"message": "   \n\t  "})
    assert res_space.status_code == 400
    assert "empty" in res_space.json()["detail"].lower()

    # Streaming endpoint empty string
    res_stream_empty = client.post("/api/chat/stream", json={"message": ""})
    assert res_stream_empty.status_code == 400
    assert "empty" in res_stream_empty.json()["detail"].lower()

    # Streaming endpoint whitespace
    res_stream_space = client.post("/api/chat/stream", json={"message": "  \n "})
    assert res_stream_space.status_code == 400
    assert "empty" in res_stream_space.json()["detail"].lower()


def test_chat_huge_payload():
    client = TestClient(app)
    huge_message = "What is the policy for " + "travel " * 1000
    res = client.post("/api/chat", json={"message": huge_message})
    assert res.status_code == 200
    data = res.json()
    assert "id" in data
    assert "answer" in data


def test_chat_invalid_filters_and_fields():
    client = TestClient(app)
    # Filter with dict
    res = client.post(
        "/api/chat",
        json={"message": "Remote work policy", "filters": {"category": "hr"}},
    )
    assert res.status_code == 200

    # None filters
    res_none_filter = client.post(
        "/api/chat",
        json={"message": "Expense policy", "filters": None},
    )
    assert res_none_filter.status_code == 200


# ============================================================================
# 2. SSE Streaming Resilience, Sequence Ordering & Sub-1s TTFT
# ============================================================================

def test_sse_stream_chunk_sequence_and_reassembly():
    client = TestClient(app)
    payload = {"message": "How do I claim expense reimbursement?", "session_id": "adversarial_stream_sess"}

    t0 = time.perf_counter()
    with client.stream("POST", "/api/chat/stream", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        events = []
        chunks_collected = []
        current_event = None
        current_data = []
        first_chunk_ttft = None

        for line in response.iter_lines():
            if not line:
                if current_event and current_data:
                    data_str = "\n".join(current_data)
                    data_obj = json.loads(data_str)
                    events.append((current_event, data_obj))
                    if current_event == "chunk":
                        if first_chunk_ttft is None:
                            first_chunk_ttft = (time.perf_counter() - t0) * 1000
                        chunks_collected.append(data_obj["content"])
                    current_event = None
                    current_data = []
                continue

            line_str = line if isinstance(line, str) else line.decode("utf-8")
            if line_str.startswith("event:"):
                current_event = line_str[6:].strip()
            elif line_str.startswith("data:"):
                current_data.append(line_str[5:].strip())

        event_names = [e[0] for e in events]
        assert event_names[0] == "start", "First SSE event must be 'start'"
        assert "chunk" in event_names
        assert "citation" in event_names
        assert "trace" in event_names
        assert event_names[-1] == "done", "Last SSE event must be 'done'"

        # Sub-1s TTFT check
        assert first_chunk_ttft is not None
        assert first_chunk_ttft < 1000.0, f"TTFT latency high: {first_chunk_ttft:.2f}ms"

        # Check reassembled string content
        reassembled_text = "".join(chunks_collected)
        assert len(reassembled_text.strip()) > 0


def test_sse_stream_abrupt_client_disconnect():
    """Simulate client dropping connection after start event without breaking server."""
    client = TestClient(app)
    payload = {"message": "Tell me about parental leave benefits"}

    with client.stream("POST", "/api/chat/stream", json=payload) as response:
        assert response.status_code == 200
        # Read only until start event then break
        for line in response.iter_lines():
            line_str = line if isinstance(line, str) else line.decode("utf-8")
            if "event: start" in line_str:
                break
    
    # Server should still be healthy
    health_res = client.get("/api/health")
    assert health_res.status_code == 200


# ============================================================================
# 3. Concurrent Document Upload & Deletion Resilience
# ============================================================================

def test_document_upload_exceeds_max_size(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr("backend.api.routes.documents.MAX_FILE_SIZE_BYTES", 100)
    monkeypatch.setattr("backend.services.document_service.MAX_FILE_SIZE_BYTES", 100)

    oversized_data = b"x" * 150
    file_obj = ("too_large.txt", io.BytesIO(oversized_data), "text/plain")

    res = client.post("/api/documents/upload", files={"file": file_obj})
    assert res.status_code == 413
    assert "exceeds" in res.json()["detail"].lower()


def test_document_upload_path_traversal_and_unicode_filename():
    client = TestClient(app)
    content = b"Policy Document Content for Path Traversal Test."
    file_obj = ("../../etc/passwd_policy.txt", io.BytesIO(content), "text/plain")

    res = client.post("/api/documents/upload", files={"file": file_obj})
    assert res.status_code in (201, 200)
    data = res.json()
    assert data["status"] == "indexed"
    doc_id = data["document_id"]

    # Verify retrieval
    doc_res = client.get(f"/api/documents/{doc_id}")
    assert doc_res.status_code == 200

    # Cleanup
    del_res = client.delete(f"/api/documents/{doc_id}")
    assert del_res.status_code == 200


def test_delete_non_existent_document():
    client = TestClient(app)
    res = client.delete("/api/documents/doc_nonexistent_999999")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_concurrent_document_uploads_and_deletions():
    """Stress test concurrent upload and deletion requests."""
    client = TestClient(app)

    def upload_doc(index: int):
        filename = f"concurrent_doc_{index}.txt"
        content = f"This is concurrent policy document #{index} containing safety and security guidance.".encode()
        file_tuple = (filename, io.BytesIO(content), "text/plain")
        r = client.post("/api/documents/upload", files={"file": file_tuple})
        return r.status_code, r.json() if r.status_code == 201 else None

    # Run 5 concurrent uploads
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(upload_doc, i) for i in range(5)]
        results = [f.result() for f in as_completed(futures)]

    uploaded_ids = []
    for status_code, data in results:
        assert status_code == 201
        assert data["status"] == "indexed"
        uploaded_ids.append(data["document_id"])

    # Verify all 5 appear in document list
    list_res = client.get("/api/documents")
    assert list_res.status_code == 200
    all_docs = list_res.json()["documents"]
    doc_ids_in_list = {d["document_id"] for d in all_docs}
    for uid in uploaded_ids:
        assert uid in doc_ids_in_list

    # Delete all 5 uploaded documents concurrently
    def delete_doc(doc_id: str):
        r = client.delete(f"/api/documents/{doc_id}")
        return r.status_code

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(delete_doc, uid) for uid in uploaded_ids]
        del_statuses = [f.result() for f in as_completed(futures)]

    for st in del_statuses:
        assert st == 200

    # Confirm purged
    for uid in uploaded_ids:
        get_res = client.get(f"/api/documents/{uid}")
        assert get_res.status_code == 404
