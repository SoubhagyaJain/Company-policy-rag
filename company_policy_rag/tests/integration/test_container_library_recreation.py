"""Opt-in real container recreation check using a disposable library only.

Build docker/Dockerfile.backend, then set RAG_CONTAINER_TEST_IMAGE to that image.
No running application container, existing volume, or Ollama service is used.
"""
import json
import os
import subprocess

import pytest


IMAGE = os.environ.get("RAG_CONTAINER_TEST_IMAGE")
pytestmark = pytest.mark.skipif(not IMAGE, reason="set RAG_CONTAINER_TEST_IMAGE to a built backend image")

PROBE = r'''
import json, os
from pathlib import Path
from backend.services.document_service import DocumentService
from backend.vision.image_asset_manager import ImageAssetManager
from backend.vision.vision_cache import VisionCacheManager
from src.config import settings
class Embeddings:
    def embed_chunks(self, texts): return [[1.0, 0.5] for text in texts]
root = Path(settings.app_storage_dir)
service = DocumentService(embedding_service=Embeddings(),
    image_asset_manager=ImageAssetManager(root / "images"),
    vision_cache_manager=VisionCacheManager(root / "vision_cache"))
manifest = root / "recreation-test.json"
try:
    if not manifest.exists():
        uploaded = service.upload_document("policy.txt", b"Employees receive twenty days of annual paid leave.")
        service._ingestion_executor.submit(lambda: None).result(timeout=30)
        assert service.get_ingestion_status(uploaded.document_id).status == "READY"
        expected = {"document_id": uploaded.document_id, "chunks": sorted(service.docstore)}
        manifest.write_text(json.dumps(expected))
    else:
        expected = json.loads(manifest.read_text())
        assert [d.document_id for d in service.list_documents().documents] == [expected["document_id"]]
        assert sorted(service.docstore) == expected["chunks"]
        assert sorted(c.id for c in service.bm25_index.entries) == expected["chunks"]
        assert service.vector_store.count() == len(expected["chunks"])
        assert service._stored_files[expected["document_id"]].read_bytes() == b"Employees receive twenty days of annual paid leave."
    print("RECREATION_RESULT=" + json.dumps(expected))
finally:
    service._ingestion_executor.shutdown(wait=True)
'''


def test_library_survives_two_distinct_containers(tmp_path):
    results = []
    for _ in range(2):
        completed = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", "--user", "0:0",
             "--mount", f"type=bind,source={tmp_path},target=/app/app/storage",
             "-e", "APP_STORAGE_DIR=/app/app/storage", "-e", "REDIS_ENABLED=false",
             "--entrypoint", "python", IMAGE, "-c", PROBE],
            capture_output=True, text=True, check=True, timeout=120,
        )
        result = next(line for line in completed.stdout.splitlines() if line.startswith("RECREATION_RESULT="))
        results.append(json.loads(result.split("=", 1)[1]))
    assert results[0] == results[1]
