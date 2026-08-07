import pytest
from fastapi.testclient import TestClient
from backend.api.main import app
from backend.api.dependencies import get_telemetry_service, get_document_service, reset_dependencies
from backend.services.telemetry_service import TelemetryService
from backend.services.document_service import DocumentService
from backend.models.api_dto import TraceSummary
from backend.models.rag import RAGResponse, ScoredChunk
from backend.models.chunk import Chunk, ChunkMetadata


@pytest.fixture(autouse=True)
def setup_teardown():
    reset_dependencies()
    yield
    reset_dependencies()


class TestTelemetryRingBufferStress:
    """Stress testing telemetry circular buffer limits, wrap-around, and trace eviction."""

    def test_circular_buffer_overflow_1500_traces(self):
        """Record 1500 traces into a 1000-max trace buffer and verify eviction and bounds."""
        telemetry = TelemetryService(max_traces=1000)

        # Record 1500 traces
        for i in range(1500):
            trace = TraceSummary(
                trace_id=f"trace_{i:04d}",
                query=f"Query {i}",
                candidate_count=5,
                post_rerank_count=3,
                final_context_count=2,
                execution_time_ms=10.0 + (i % 50),
                ttft_ms=5.0 + (i % 20),
                similarity_scores=[0.8, 0.75],
                rerank_scores=[0.9, 0.85],
                bm25_scores=[12.5, 10.0],
                rrf_scores=[0.03, 0.02],
                sources_used=["policy.pdf"],
                token_usage={"prompt_tokens": 100, "completion_tokens": 50},
            )
            telemetry.record_trace(trace)

        # Check internal deque length stays strictly at 1000
        assert len(telemetry._traces) == 1000

        # Check trace eviction: trace_0000 to trace_0499 must be evicted (return None)
        assert telemetry.get_trace_by_id("trace_0000") is None
        assert telemetry.get_trace_by_id("trace_0499") is None

        # Check trace retention: trace_0500 to trace_1499 must exist
        assert telemetry.get_trace_by_id("trace_0500") is not None
        assert telemetry.get_trace_by_id("trace_1499") is not None

        # Check total queries recorded metric vs buffer count
        metrics = telemetry.get_metrics(recent_limit=20)
        assert metrics.total_queries == 1500
        assert len(metrics.recent_traces) == 20
        # Newest trace should be first in recent_traces
        assert metrics.recent_traces[0].trace_id == "trace_1499"

        # Verify get_recent_traces pagination
        recent_50 = telemetry.get_recent_traces(limit=50, offset=0)
        assert len(recent_50) == 50
        assert recent_50[0].trace_id == "trace_1499"
        assert recent_50[49].trace_id == "trace_1450"

    def test_admin_trace_detail_endpoint_evicted_vs_valid(self):
        """API test for GET /api/admin/traces/{trace_id} on evicted vs valid trace."""
        client = TestClient(app)
        telemetry = get_telemetry_service()
        telemetry.clear()

        # Fill 1050 traces
        for i in range(1050):
            trace = TraceSummary(
                trace_id=f"t_{i}",
                query=f"Q {i}",
                execution_time_ms=15.0,
                token_usage={"prompt_tokens": 10, "completion_tokens": 10},
            )
            telemetry.record_trace(trace)

        # Evicted trace t_0 -> 404 Not Found
        res_evicted = client.get("/api/admin/traces/t_0")
        assert res_evicted.status_code == 404
        assert "not found" in res_evicted.json()["detail"].lower()

        # Valid active trace t_1049 -> 200 OK
        res_valid = client.get("/api/admin/traces/t_1049")
        assert res_valid.status_code == 200
        data = res_valid.json()
        assert data["trace"]["trace_id"] == "t_1049"

    def test_admin_traces_list_endpoint(self):
        """API test for GET /api/admin/traces endpoint."""
        client = TestClient(app)
        res = client.get("/api/admin/traces?limit=10&offset=0")
        assert res.status_code == 200
        body = res.json()
        assert "traces" in body
        assert "total_count" in body


class TestDocumentDeletionEdgeCases:
    """Boundary testing document deletion in DocumentService and API routes."""

    def test_delete_non_existent_document_id(self):
        """Deleting a document ID that does not exist returns 404."""
        client = TestClient(app)
        res = client.delete("/api/documents/doc_nonexistent_999999")
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()

    def test_double_deletion_same_document(self):
        """Deleting an existing document twice returns 200 then 404."""
        client = TestClient(app)
        # Upload a document first
        upload_res = client.post(
            "/api/documents/upload",
            files={"file": ("delete_test.txt", b"Content for deletion test", "text/plain")},
            data={"category": "test"},
        )
        assert upload_res.status_code == 200
        doc_id = upload_res.json()["document_id"]

        # First delete -> 200 OK
        del1 = client.delete(f"/api/documents/{doc_id}")
        assert del1.status_code == 200
        assert del1.json()["status"] == "deleted"

        # Second delete -> 404 Not Found
        del2 = client.delete(f"/api/documents/{doc_id}")
        assert del2.status_code == 404

    def test_delete_when_store_empty(self):
        """Calling delete on service when document store is completely empty."""
        doc_service = DocumentService()
        result = doc_service.delete_document("doc_empty_store_123")
        assert result is None

    def test_delete_document_with_special_characters(self):
        """Deleting a document with special characters in filename."""
        client = TestClient(app)
        special_filename = "policy #1 (draft & final) [v2.0].txt"
        upload_res = client.post(
            "/api/documents/upload",
            files={"file": (special_filename, b"Special char content test", "text/plain")},
            data={"category": "special"},
        )
        assert upload_res.status_code == 200
        doc_id = upload_res.json()["document_id"]

        del_res = client.delete(f"/api/documents/{doc_id}")
        assert del_res.status_code == 200
        assert del_res.json()["status"] == "deleted"

    def test_same_filename_collateral_deletion_bug_investigation(self):
        """
        Adversarial test: Upload 2 documents with the SAME filename but DIFFERENT doc_ids.
        Deleting doc_1 should purge doc_1. Check what happens to doc_2's chunks in vector & BM25 store.
        """
        doc_service = DocumentService()
        res1 = doc_service.upload_document("same_name.txt", b"First copy of document content", category="cat1")
        res2 = doc_service.upload_document("same_name.txt", b"Second copy of document content", category="cat2")

        doc_id_1 = res1.document_id
        doc_id_2 = res2.document_id

        assert doc_id_1 != doc_id_2
        assert len(doc_service.list_documents().documents) == 2

        # Delete doc_1
        doc_service.delete_document(doc_id_1)

        # Check doc registry
        docs_left = doc_service.list_documents().documents
        assert len(docs_left) == 1
        assert docs_left[0].document_id == doc_id_2

        # Verify whether doc_2's chunks remain in vector store and BM25 index
        # (This tests whether delete_by_source(filename) collaterally wiped out doc_2)
        bm25_chunks_remaining = [c for c in doc_service.bm25_index.entries if c.metadata.document_id == doc_id_2]
        vector_count_remaining = doc_service.vector_store.count()

        # Record finding: If delete_by_source wiped out doc_2's chunks, bm25_chunks_remaining will be 0!
        return {
            "doc_2_bm25_chunks": len(bm25_chunks_remaining),
            "total_vector_count": vector_count_remaining,
        }


class TestDocumentUploadEdgeCases:
    """Boundary testing document upload size, formats, and corrupt files."""

    def test_upload_empty_file(self):
        """Uploading empty file (0 bytes) returns 400 Bad Request."""
        client = TestClient(app)
        res = client.post(
            "/api/documents/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert res.status_code == 400
        assert "empty" in res.json()["detail"].lower()

    def test_upload_corrupt_pdf_file(self):
        """Uploading corrupt PDF bytes."""
        client = TestClient(app)
        corrupt_pdf_bytes = b"%PDF-1.4 corrupt junk noise \x00\xff\xfe\xfd data"
        res = client.post(
            "/api/documents/upload",
            files={"file": ("corrupt.pdf", corrupt_pdf_bytes, "application/pdf")},
        )
        # Loader should either extract text safely or raise 400 Bad Request
        assert res.status_code in (400, 422, 200)

    def test_upload_corrupt_json_file(self):
        """Uploading malformed JSON file."""
        client = TestClient(app)
        res = client.post(
            "/api/documents/upload",
            files={"file": ("corrupt.json", b"{ invalid json content: [", "application/json")},
        )
        assert res.status_code in (400, 422, 200)

    def test_upload_unsupported_extension(self):
        """Uploading file with unsupported extension."""
        client = TestClient(app)
        res = client.post(
            "/api/documents/upload",
            files={"file": ("test.xyz", b"Some random text data inside xyz file", "application/octet-stream")},
        )
        # Should either fallback to text loader or return 400
        assert res.status_code in (200, 400)


class TestCORSAndMiddlewareBehavior:
    """Testing CORS middleware under standard and non-standard headers."""

    def test_cors_options_preflight_allowed_origin(self):
        """Preflight OPTIONS request from http://localhost:3000."""
        client = TestClient(app)
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type, Authorization, X-Custom-Header",
        }
        res = client.options("/api/chat", headers=headers)
        assert res.status_code == 200
        assert res.headers.get("access-control-allow-origin") == "http://localhost:3000"
        assert "access-control-allow-credentials" in res.headers

    def test_non_standard_request_headers(self):
        """API endpoint handles non-standard and custom headers gracefully."""
        client = TestClient(app)
        custom_headers = {
            "Origin": "http://localhost:3000",
            "X-Forwarded-For": "203.0.113.195",
            "X-Request-ID": "req-adversarial-12345",
            "User-Agent": "EmpiricalChallenger/2.0",
            "Accept-Language": "en-US,en;q=0.9",
        }
        res = client.get("/api/health", headers=custom_headers)
        assert res.status_code == 200
        assert res.json()["status"] == "ok"
