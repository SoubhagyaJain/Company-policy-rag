"""Project-local pytest bootstrap for both top-level and package-style imports."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

for path in (WORKSPACE_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


class _DeterministicTestEmbeddings:
    """Small local embeddings that keep API tests out of production model/storage state."""

    def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        return [[float((index % 7) + 1) / 7.0 for index in range(384)] for _ in texts]


@pytest.fixture
def isolated_document_client(tmp_path: Path):
    """FastAPI client with every document persistence layer rooted in pytest temp storage."""
    from fastapi.testclient import TestClient

    from backend.api.dependencies import get_document_service, reset_dependencies
    from backend.api.main import app
    from backend.embeddings.vector_store import ChromaVectorStore
    from backend.retrieval.bm25 import BM25SearchIndex
    from backend.services.document_service import DocumentService
    from backend.vision.image_asset_manager import ImageAssetManager
    from backend.vision.vision_cache import VisionCacheManager

    reset_dependencies()
    service = DocumentService(
        vector_store=ChromaVectorStore(persist_dir=str(tmp_path / "chroma")),
        bm25_index=BM25SearchIndex(storage_dir=str(tmp_path / "bm25")),
        embedding_service=_DeterministicTestEmbeddings(),  # type: ignore[arg-type]
        image_asset_manager=ImageAssetManager(tmp_path / "images"),
        vision_cache_manager=VisionCacheManager(tmp_path / "vision_cache"),
        storage_dir=str(tmp_path / "uploads"),
    )
    app.dependency_overrides[get_document_service] = lambda: service
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_document_service, None)
        reset_dependencies()
