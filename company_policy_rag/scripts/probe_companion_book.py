import os
import tempfile
import time
import sys
from pathlib import Path

os.environ["VISION_ENABLED"] = "false"
from fastapi.testclient import TestClient
from backend.api.main import app
from backend.api.dependencies import get_document_service
from backend.services.document_service import DocumentService
from backend.vision.image_asset_manager import ImageAssetManager
from backend.vision.vision_cache import VisionCacheManager
from backend.ingestion.loaders.loader_factory import _default_factory
from backend.ingestion.loaders.pdf import PDFLoader


class Embeddings:
    def embed_chunks(self, texts):
        return [[1.0, 0.5] for _ in texts]


class NoImages:
    def extract_page_images(self, **kwargs):
        return []


with tempfile.TemporaryDirectory(prefix="rag-book-test-", ignore_cleanup_errors=True) as root:
    root = Path(root)
    for loader in _default_factory.loaders:
        if isinstance(loader, PDFLoader):
            loader.image_asset_manager = NoImages()
    service = DocumentService(
        storage_dir=str(root / "uploads"),
        embedding_service=Embeddings(),
        image_asset_manager=ImageAssetManager(root / "images"),
        vision_cache_manager=VisionCacheManager(root / "vision"),
        catalog_enabled=False,
    )
    app.dependency_overrides[get_document_service] = lambda: service
    client = TestClient(app)
    pdf = Path(sys.argv[1])
    with pdf.open("rb") as f:
        res = client.post("/api/documents/upload", files={"file": (pdf.name, f, "application/pdf")})
    print("UPLOAD", res.status_code, res.text[:1000], flush=True)
    doc_id = res.json()["document_id"]
    service._ingestion_executor.submit(lambda: None).result(timeout=600)
    status = client.get(f"/api/documents/{doc_id}/status")
    print("STATUS", status.status_code, status.text[:1800], flush=True)
    chunks = list(service.docstore.values())
    print(
        "COUNTS",
        len(chunks),
        service.vector_store.count(),
        sum(bool(c.metadata.extra.get("companion_navigation")) for c in chunks),
        flush=True,
    )
    from backend.retrieval.bm25 import BM25SearchIndex, DEFAULT_METADATA_FIELDS
    import statistics

    baseline = BM25SearchIndex(
        storage_dir=str(root / "baseline"),
        metadata_fields=tuple(x for x in DEFAULT_METADATA_FIELDS if x != "companion_navigation"),
    )
    baseline.build_index(chunks)
    queries = [
        ("iron law of ML systems", 45),
        ("data selection fundamentals", 440),
        ("dynamic batching queueing", 730),
        ("engineering leadership responsibility", 825),
        ("arithmetic intensity compute bandwidth", 548),
    ]
    for label, index in [("BEFORE", baseline), ("AFTER", service.bm25_index)]:
        hits = []
        reciprocal = []
        timings = []
        for query, page in queries:
            start = time.perf_counter()
            ranked = index.search(query, top_k=10)
            timings.append((time.perf_counter() - start) * 1000)
            rank = next((i for i, x in enumerate(ranked, 1) if x.chunk.metadata.page_number == page), 0)
            hits.append(int(0 < rank <= 5))
            reciprocal.append(1 / rank if rank else 0)
            print("RANK", label, query, page, rank, flush=True)
        print(
            "METRIC",
            label,
            "page_hit_at_5",
            sum(hits) / len(hits),
            "mrr_at_10",
            sum(reciprocal) / len(reciprocal),
            "median_ms",
            round(statistics.median(timings), 2),
            flush=True,
        )
    dup = client.post("/api/documents/upload", files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")})
    print("DUPLICATE", dup.status_code, flush=True)
    deletion = service.delete_document(doc_id)
    print("DELETED", deletion["status"], service.vector_store.count(), len(service.bm25_index.entries), flush=True)
    service.close()
    app.dependency_overrides.clear()
