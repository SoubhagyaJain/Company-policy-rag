from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field

from backend.models.chunk import Chunk, ChunkMetadata, ChunkRole, ContentType
from backend.models.rag import ScoredChunk
from backend.utils.logging import logger


class MetadataFilter(BaseModel):
    key: str
    operator: str = Field(default="eq", description="eq | in | gte | lte")
    value: Any


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


class VectorStoreInterface(ABC):
    @abstractmethod
    def add_chunks(self, chunks: list[Chunk]) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        pass

    @abstractmethod
    def delete_by_source(self, source_file: str) -> None:
        pass

    @abstractmethod
    def delete_by_document_id(self, document_id: str) -> None:
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class ChromaVectorStore(VectorStoreInterface):
    """
    ChromaDB implementation of VectorStoreInterface with fallback in-memory store.
    Supports collection management, metadata filtering, and similarity search.
    """

    def __init__(
        self,
        collection_name: str = "company_policy",
        persist_dir: str = "app/storage/chroma",
    ) -> None:
        self.collection_name = collection_name
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection: Any = None
        self._memory_chunks: dict[str, Chunk] = {}
        self._init_chroma()

    def _init_chroma(self) -> None:
        try:
            import chromadb  # type: ignore
            from chromadb.config import Settings  # type: ignore

            client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChromaVectorStore initialized collection '%s' at %s", self.collection_name, self.persist_dir)
        except Exception as exc:
            logger.warning("ChromaDB initialization failed (%s). Operating in-memory vector store mode.", exc)
            self._collection = None

    def _flatten_metadata(self, meta: ChunkMetadata) -> dict[str, Any]:
        d = meta.model_dump()
        flat: dict[str, Any] = {}
        for k, v in d.items():
            if k == "extra" and isinstance(v, dict):
                for ek, ev in v.items():
                    if isinstance(ev, (str, int, float, bool)):
                        flat[f"extra_{ek}"] = ev
            elif isinstance(v, (str, int, float, bool)):
                flat[k] = v
            elif isinstance(v, (ChunkRole, ContentType)):
                flat[k] = str(v.value)
            elif v is None:
                continue
            elif isinstance(v, list):
                flat[k] = ",".join(str(x) for x in v)
            else:
                flat[k] = str(v)
        return flat

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        for chunk in chunks:
            self._memory_chunks[chunk.id] = chunk

        if self._collection is not None:
            try:
                ids: list[str] = []
                documents: list[str] = []
                embeddings: list[Any] = []
                metadatas: list[Any] = []

                for chunk in chunks:
                    if chunk.embedding is None:
                        continue
                    ids.append(chunk.id)
                    documents.append(chunk.text)
                    embeddings.append(chunk.embedding)
                    metadatas.append(self._flatten_metadata(chunk.metadata))

                if ids:
                    self._collection.upsert(
                        ids=ids,
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas,
                    )
            except Exception as exc:
                logger.warning("Failed to add chunks to Chroma collection: %s", exc)

    def _matches_filters(self, chunk: Chunk, filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        meta_dict = chunk.metadata.model_dump()
        for key, value in filters.items():
            actual = meta_dict.get(key)
            if actual is None and hasattr(chunk.metadata, key):
                actual = getattr(chunk.metadata, key)
            if isinstance(value, list):
                if actual not in value:
                    return False
            elif actual != value:
                return False
        return True

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        if not query_embedding:
            return []

        if self._collection is not None and cast(int, self._collection.count()) > 0:
            try:
                where_clause = None
                if filters:
                    conditions = []
                    for k, v in filters.items():
                        if isinstance(v, list):
                            conditions.append({k: {"$in": v}})
                        else:
                            conditions.append({k: {"$eq": v}})
                    if len(conditions) == 1:
                        where_clause = conditions[0]
                    elif len(conditions) > 1:
                        where_clause = {"$and": conditions}

                query_params: dict[str, Any] = {
                    "query_embeddings": [query_embedding],
                    "n_results": min(top_k, cast(int, self._collection.count())),
                    "include": ["documents", "metadatas", "distances"],
                }
                if where_clause:
                    query_params["where"] = where_clause

                results = self._collection.query(**query_params)
                scored_chunks: list[ScoredChunk] = []

                if results is not None and results.get("ids") and len(results["ids"]) > 0:
                    hit_ids = results["ids"][0]
                    hit_metas = (results.get("metadatas") or [[]])[0]
                    hit_docs = (results.get("documents") or [[]])[0]
                    hit_dists = (results.get("distances") or [[]])[0]

                    for cid, meta, doc, dist in zip(hit_ids, hit_metas, hit_docs, hit_dists):
                        meta_dict = cast(dict[str, Any], meta) if meta else {}
                        dist_val = float(dist) if dist is not None else 1.0
                        sim_score = max(0.0, 1.0 - dist_val)

                        chunk = self._memory_chunks.get(str(cid))
                        if chunk is None:
                            meta_obj = ChunkMetadata(
                                document_id=str(meta_dict.get("document_id", "unknown")),
                                source_file=str(meta_dict.get("source_file", "unknown")),
                                file_path=str(meta_dict.get("file_path", "")),
                                file_hash=str(meta_dict.get("file_hash", "")),
                                document_type=str(meta_dict.get("document_type", "unknown")),
                                category=str(meta_dict.get("category", "general")),
                                chunk_index=int(meta_dict.get("chunk_index", 0)),
                                page_number=int(meta_dict["page_number"]) if meta_dict.get("page_number") is not None else None,
                                section_title=str(meta_dict["section_title"]) if meta_dict.get("section_title") is not None else None,
                                section_path=str(meta_dict["section_path"]) if meta_dict.get("section_path") is not None else None,
                                chunk_strategy=str(meta_dict.get("chunk_strategy", "recursive")),
                            )
                            chunk = Chunk(id=str(cid), text=str(doc or ""), metadata=meta_obj)

                        scored_chunks.append(
                            ScoredChunk(
                                chunk=chunk,
                                score=sim_score,
                                dense_score=sim_score,
                            )
                        )
                return scored_chunks
            except Exception as exc:
                logger.warning("Chroma query failed (%s). Falling back to memory search.", exc)

        # Fallback in-memory search
        scored_memory: list[ScoredChunk] = []
        for chunk in self._memory_chunks.values():
            if not self._matches_filters(chunk, filters):
                continue
            if chunk.embedding is not None:
                sim = cosine_similarity(query_embedding, chunk.embedding)
            else:
                sim = 0.0
            scored_memory.append(ScoredChunk(chunk=chunk, score=sim, dense_score=sim))

        scored_memory.sort(key=lambda x: x.score, reverse=True)
        return scored_memory[:top_k]

    def delete_by_source(self, source_file: str) -> None:
        to_delete = [
            cid for cid, chunk in self._memory_chunks.items()
            if chunk.metadata.source_file == source_file
        ]
        for cid in to_delete:
            del self._memory_chunks[cid]

        if self._collection is not None:
            try:
                self._collection.delete(where={"source_file": source_file})
            except Exception as exc:
                logger.warning("Failed to delete from Chroma by source_file: %s", exc)

    def delete_by_document_id(self, document_id: str) -> None:
        to_delete = [
            cid for cid, chunk in self._memory_chunks.items()
            if chunk.metadata.document_id == document_id
        ]
        for cid in to_delete:
            del self._memory_chunks[cid]

        if self._collection is not None:
            try:
                self._collection.delete(where={"document_id": document_id})
            except Exception as exc:
                logger.warning("Failed to delete from Chroma by document_id: %s", exc)

    def count(self) -> int:
        if self._collection is not None:
            try:
                return cast(int, self._collection.count())
            except Exception:
                pass
        return len(self._memory_chunks)

    def clear(self) -> None:
        self._memory_chunks.clear()
        if self._collection is not None:
            try:
                all_ids = self._collection.get(include=[])["ids"]
                if all_ids:
                    self._collection.delete(ids=all_ids)
            except Exception as exc:
                logger.warning("Failed to clear Chroma collection: %s", exc)
