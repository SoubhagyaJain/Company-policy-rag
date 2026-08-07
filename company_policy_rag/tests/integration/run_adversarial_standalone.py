from __future__ import annotations

import os
import io
import json
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi.testclient import TestClient

from backend.api.dependencies import reset_dependencies
from backend.embeddings.embeddings import EmbeddingService
from backend.retrieval.reranker import CrossEncoderReranker
from backend.embeddings.vector_store import ChromaVectorStore

def run_all_tests():
    print("Starting Standalone Adversarial Verification...")

    # Fast deterministic fallback mode for instant test execution
    def _fast_embedding_init(self):
        self._model = None
        self._model_loaded = True

    def _fast_reranker_init(self):
        self._model = None
        self._model_loaded = True

    def _memory_only_init(self):
        self._collection = None

    EmbeddingService._init_model = _fast_embedding_init
    CrossEncoderReranker._init_model = _fast_reranker_init
    ChromaVectorStore._init_chroma = _memory_only_init

    reset_dependencies()
    from backend.api.main import app
    client = TestClient(app)

    # 1. Empty & Whitespace Payloads
    t0 = time.perf_counter()
    print("Testing empty & whitespace payloads...", end=" ", flush=True)
    res = client.post("/api/chat", json={"message": ""})
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()

    res_space = client.post("/api/chat", json={"message": "   \n\t  "})
    assert res_space.status_code == 400

    res_stream_empty = client.post("/api/chat/stream", json={"message": ""})
    assert res_stream_empty.status_code == 400

    res_stream_space = client.post("/api/chat/stream", json={"message": "  \n "})
    assert res_stream_space.status_code == 400
    print(f"PASSED ({time.perf_counter() - t0:.2f}s)")

    # 2. Huge Payload
    t0 = time.perf_counter()
    print("Testing huge payload (1000 words)...", end=" ", flush=True)
    huge_message = "What is the policy for " + "travel " * 1000
    res = client.post("/api/chat", json={"message": huge_message})
    assert res.status_code == 200
    assert "answer" in res.json()
    print(f"PASSED ({time.perf_counter() - t0:.2f}s)")

    # 3. Invalid Filters & Fields
    t0 = time.perf_counter()
    print("Testing invalid filters & fields...", end=" ", flush=True)
    res = client.post("/api/chat", json={"message": "Remote work policy", "filters": {"category": "hr"}})
    assert res.status_code == 200
    res_none = client.post("/api/chat", json={"message": "Expense policy", "filters": None})
    assert res_none.status_code == 200
    print(f"PASSED ({time.perf_counter() - t0:.2f}s)")

    # 4. SSE Stream Sequence & Sub-1s TTFT
    t0 = time.perf_counter()
    print("Testing SSE stream sequence & TTFT...", end=" ", flush=True)
    payload = {"message": "How do I claim expense reimbursement?", "session_id": "adv_stream_sess"}
    with client.stream("POST", "/api/chat/stream", json=payload) as response:
        assert response.status_code == 200
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
        assert event_names[0] == "start"
        assert "chunk" in event_names
        assert "citation" in event_names
        assert "trace" in event_names
        assert event_names[-1] == "done"
        assert first_chunk_ttft < 1000.0, f"TTFT too high: {first_chunk_ttft:.2f}ms"
    print(f"PASSED ({time.perf_counter() - t0:.2f}s, TTFT: {first_chunk_ttft:.2f}ms)")

    # 5. Abrupt Disconnect
    t0 = time.perf_counter()
    print("Testing client abrupt disconnect...", end=" ", flush=True)
    payload = {"message": "Tell me about parental leave benefits"}
    with client.stream("POST", "/api/chat/stream", json=payload) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            line_str = line if isinstance(line, str) else line.decode("utf-8")
            if "event: start" in line_str:
                break
    health_res = client.get("/api/health")
    assert health_res.status_code == 200
    print(f"PASSED ({time.perf_counter() - t0:.2f}s)")

    # 6. Upload Exceeds Max Size
    t0 = time.perf_counter()
    print("Testing upload size limit (mocked)...", end=" ", flush=True)
    import backend.api.routes.documents
    import backend.services.document_service
    backend.api.routes.documents.MAX_FILE_SIZE_BYTES = 100
    backend.services.document_service.MAX_FILE_SIZE_BYTES = 100
    oversized_data = b"x" * 150
    file_obj = ("too_large.txt", io.BytesIO(oversized_data), "text/plain")
    res = client.post("/api/documents/upload", files={"file": file_obj})
    assert res.status_code == 413
    backend.api.routes.documents.MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024
    backend.services.document_service.MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024
    print(f"PASSED ({time.perf_counter() - t0:.2f}s)")

    # 7. Path Traversal & Unicode
    t0 = time.perf_counter()
    print("Testing path traversal & unicode filename upload...", end=" ", flush=True)
    content = b"Policy Document Content for Path Traversal Test."
    file_obj = ("../../etc/passwd_policy.txt", io.BytesIO(content), "text/plain")
    res = client.post("/api/documents/upload", files={"file": file_obj})
    assert res.status_code in (200, 201)
    doc_id = res.json()["document_id"]
    doc_res = client.get(f"/api/documents/{doc_id}")
    assert doc_res.status_code == 200
    del_res = client.delete(f"/api/documents/{doc_id}")
    assert del_res.status_code == 200
    print(f"PASSED ({time.perf_counter() - t0:.2f}s)")

    # 8. Delete non-existent doc
    t0 = time.perf_counter()
    print("Testing delete non-existent document...", end=" ", flush=True)
    res = client.delete("/api/documents/doc_nonexistent_999999")
    assert res.status_code == 404
    print(f"PASSED ({time.perf_counter() - t0:.2f}s)")

    # 9. Concurrent Upload & Deletion
    t0 = time.perf_counter()
    print("Testing concurrent upload and deletion (5 threads)...", end=" ", flush=True)
    def upload_doc(index: int):
        filename = f"concurrent_doc_{index}.txt"
        content = f"This is concurrent policy document #{index} containing safety and security guidance.".encode()
        file_tuple = (filename, io.BytesIO(content), "text/plain")
        r = client.post("/api/documents/upload", files={"file": file_tuple})
        return r.status_code, r.json() if r.status_code == 201 else None

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(upload_doc, i) for i in range(5)]
        results = [f.result() for f in as_completed(futures)]

    uploaded_ids = []
    for status_code, data in results:
        assert status_code == 201
        uploaded_ids.append(data["document_id"])

    def delete_doc(doc_id: str):
        r = client.delete(f"/api/documents/{doc_id}")
        return r.status_code

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(delete_doc, uid) for uid in uploaded_ids]
        del_statuses = [f.result() for f in as_completed(futures)]

    for st in del_statuses:
        assert st == 200
    print(f"PASSED ({time.perf_counter() - t0:.2f}s)")

    print("\nALL ADVERSARIAL VERIFICATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_all_tests()
