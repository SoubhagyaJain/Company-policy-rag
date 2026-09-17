"""Cached retrieval candidates must never outlive the corpus or settings they came from."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

from backend.embeddings.vector_store import ChromaVectorStore
from backend.models.chunk import Chunk, ChunkMetadata
from backend.models.rag import RetrievalStrategy, ScoredChunk
from backend.rag.pipeline import RAGPipeline
from backend.retrieval.retrieval_cache import RetrievalCache, get_retrieval_cache
from backend.services.document_service import DocumentService


def _chunk(cid: str, document_id: str = "doc_aaaaaaaaaaaa") -> Chunk:
    return Chunk(
        id=cid,
        text=f"text for {cid}",
        metadata=ChunkMetadata(document_id=document_id, source_file="a.txt"),
        embedding=[0.1] * 8,
    )


def test_cache_misses_when_version_changes() -> None:
    cache = RetrievalCache()
    hits = [ScoredChunk(chunk=_chunk("c1"), score=1.0)]
    cache.set("q", hits, top_k=8, version="v1")

    assert cache.get("q", top_k=8, version="v1") == hits
    assert cache.get("q", top_k=8, version="v2") is None


def test_cached_list_is_a_copy() -> None:
    cache = RetrievalCache()
    hits = [ScoredChunk(chunk=_chunk("c1"), score=1.0)]
    cache.set("q", hits, version="v")
    got = cache.get("q", version="v")
    got.append(ScoredChunk(chunk=_chunk("c2"), score=0.5))
    assert len(cache.get("q", version="v")) == 1


def test_pipeline_cache_version_tracks_corpus_changes(tmp_path: Path) -> None:
    store = ChromaVectorStore(collection_name="cache_version", persist_dir=str(tmp_path / "chroma"))
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.hybrid_retriever = SimpleNamespace(
        dense_retriever=SimpleNamespace(vector_store=store),
        bm25_index=SimpleNamespace(entries=[]),
        min_chunk_words=5,
    )
    strategy = RetrievalStrategy()

    store.add_chunks([_chunk("c1")])
    before = pipeline._retrieval_cache_version(strategy)
    store.add_chunks([_chunk("c2")])
    after_add = pipeline._retrieval_cache_version(strategy)
    store.delete_by_document_id("doc_aaaaaaaaaaaa")
    after_delete = pipeline._retrieval_cache_version(strategy)

    assert before != after_add
    assert after_add != after_delete

    pipeline.hybrid_retriever.min_chunk_words = 0
    assert pipeline._retrieval_cache_version(strategy) != after_delete


class _VectorStore:
    def delete_by_document_id(self, document_id: str) -> None:
        pass

    def delete_by_source(self, source_file: str) -> None:
        pass


class _BM25:
    entries: list = []

    def remove_by_document_id(self, document_id: str) -> None:
        pass

    def save(self) -> None:
        pass


class _Assets:
    def delete_document_assets(self, document_id: str) -> int:
        return 0

    def list_assets(self, document_id: str) -> list:
        return []


class _VisionCache:
    def delete_by_document_id(self, document_id: str) -> int:
        return 0


def test_deleting_a_document_clears_cached_candidates(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    document_id = "doc_bbbbbbbbbbbb"
    (uploads / f"{document_id}_policy.txt").write_text("policy", encoding="utf-8")
    service = DocumentService(
        vector_store=_VectorStore(),  # type: ignore[arg-type]
        bm25_index=_BM25(),  # type: ignore[arg-type]
        embedding_service=object(),  # type: ignore[arg-type]
        image_asset_manager=_Assets(),  # type: ignore[arg-type]
        vision_cache_manager=_VisionCache(),  # type: ignore[arg-type]
        storage_dir=str(uploads),
    )
    assert re.fullmatch(r"doc_[0-9a-f]{12}", document_id)

    cache = get_retrieval_cache()
    cache.set("policy question", [ScoredChunk(chunk=_chunk("c9", document_id), score=1.0)], version="v")
    assert cache.get("policy question", version="v") is not None

    assert service.delete_document(document_id) is not None
    assert cache.get("policy question", version="v") is None
