from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from backend.embeddings.embeddings import EmbeddingService
from backend.models.rag import Citation
from backend.utils.logging import logger
from src.config import Settings, settings as default_settings


_AUDIENCE_TERMS = {
    "employee", "employees", "contractor", "contractors", "intern", "interns",
    "manager", "managers", "director", "directors", "full-time", "part-time",
    "exempt", "non-exempt", "temporary", "vendor", "vendors",
}


def _normalized_query(query: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", query.casefold()))


def _critical_query_facts(query: str) -> tuple[frozenset[str], ...]:
    """Extract facts whose mismatch makes two otherwise similar questions unsafe to share."""
    lowered = query.casefold()
    tokens = set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", lowered))
    numbers = frozenset(re.findall(r"\b\d+(?:[.-]\d+)*\b", lowered))
    audiences = frozenset(tokens.intersection(_AUDIENCE_TERMS))
    quoted = frozenset(
        value.strip().casefold()
        for value in re.findall(r"['\"]([^'\"]{1,80})['\"]", query)
        if value.strip()
    )
    negation = frozenset({"negated"}) if tokens.intersection({"not", "never", "without", "exclude", "excluding"}) else frozenset()
    return numbers, audiences, quoted, negation


def _queries_are_interchangeable(cached_query: str, incoming_query: str) -> bool:
    if _normalized_query(cached_query) == _normalized_query(incoming_query):
        return True
    for cached_facts, incoming_facts in zip(
        _critical_query_facts(cached_query),
        _critical_query_facts(incoming_query),
    ):
        if (cached_facts or incoming_facts) and cached_facts != incoming_facts:
            return False
    return True


class CachedResponse(BaseModel):
    """Container for a retrieved semantic cache entry."""

    answer: str = Field(..., description="Cached answer text")
    citations: List[Citation] = Field(
        default_factory=list, description="Citations supporting answer"
    )
    similarity_score: float = Field(
        ..., description="Cosine similarity score (1.0 - raw_distance)"
    )
    distance: float = Field(..., description="Raw distance returned by ChromaDB")
    lookup_latency_ms: float = Field(
        ..., description="Cache lookup latency in milliseconds"
    )
    timestamp: float = Field(..., description="Epoch timestamp of storage")
    kb_version: Optional[str] = Field(
        default=None, description="Knowledge base / document version tag"
    )


class SemanticCacheManager:
    """
    Manages semantic cache lookups, storage, and invalidation using ChromaDB.
    Reuses project ChromaDB persistence and EmbeddingService.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        vector_store: Any = None,
        embedding_service: Optional[EmbeddingService] = None,
        collection_name: Optional[str] = None,
        persist_dir: Optional[Path | str] = None,
    ) -> None:
        self.settings = settings or default_settings
        self.collection_name = (
            collection_name
            or getattr(self.settings, "semantic_cache_collection_name", "semantic_cache")
        )
        if persist_dir:
            self.persist_dir = Path(persist_dir)
        else:
            self.persist_dir = getattr(self.settings, "chroma_persist_dir", Path("storage/chroma"))

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store
        self._collection: Any = None
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._init_collection()

    def _init_collection(self) -> None:
        try:
            from backend.embeddings.vector_store import get_shared_chroma_client

            client = get_shared_chroma_client(self.persist_dir)
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "SemanticCacheManager initialized collection '%s' at %s",
                self.collection_name,
                self.persist_dir,
            )
        except Exception as exc:
            logger.warning(
                "ChromaDB initialization failed for semantic cache (%s). Operating in-memory cache mode.",
                exc,
            )
            self._collection = None

    def _resolve_kb_version(self, explicit_version: Optional[str]) -> Optional[str]:
        if explicit_version is not None:
            return explicit_version
        version_fn = getattr(self.vector_store, "corpus_version", None)
        if callable(version_fn):
            try:
                return str(version_fn())
            except Exception as exc:
                logger.warning("Failed to resolve corpus version for semantic cache: %s", exc)
        return None

    def _resolve_prompt_version(self, explicit_version: Optional[str]) -> str:
        if explicit_version is not None:
            return explicit_version
        return str(getattr(self.settings, "response_prompt_version", "unknown"))

    def get(
        self,
        query: str,
        threshold: Optional[float] = None,
        kb_version: Optional[str] = None,
        model_name: Optional[str] = None,
        prompt_version: Optional[str] = None,
        cache_context: str = "",
    ) -> Optional[CachedResponse]:
        """
        Query cache for semantically matching answer.
        Returns CachedResponse if similarity >= threshold and kb_version matches, else None.
        """
        start_time = time.perf_counter()
        try:
            cache_enabled = getattr(self.settings, "semantic_cache_enabled", True)
            if not cache_enabled:
                logger.debug("Semantic cache is disabled by configuration.")
                return None

            if not query or not isinstance(query, str) or not query.strip():
                return None

            query_clean = query.strip()
            effective_kb_version = self._resolve_kb_version(kb_version)
            effective_prompt_version = self._resolve_prompt_version(prompt_version)
            effective_threshold = (
                threshold
                if threshold is not None
                else getattr(self.settings, "semantic_cache_threshold", 0.95)
            )

            query_embedding = self.embedding_service.embed_text(query_clean)
            if not query_embedding:
                return None

            # ChromaDB Lookup
            results = None
            with self._lock:
                if self._collection is not None and self._collection.count() > 0:
                    where_conditions: list[dict[str, Any]] = [
                        {"prompt_version": {"$eq": effective_prompt_version}},
                        {"cache_context": {"$eq": cache_context}},
                    ]
                    if effective_kb_version is not None:
                        where_conditions.append({"kb_version": {"$eq": effective_kb_version}})
                    if model_name is not None:
                        where_conditions.append({"model": {"$eq": model_name}})
                    where_clause: dict[str, Any] = (
                        where_conditions[0]
                        if len(where_conditions) == 1
                        else {"$and": where_conditions}
                    )
                    results = self._collection.query(
                        query_embeddings=[query_embedding],
                        n_results=1,
                        include=["documents", "metadatas", "distances"],
                        where=where_clause,
                    )

                if (
                    results
                    and results.get("ids")
                    and len(results["ids"]) > 0
                    and len(results["ids"][0]) > 0
                ):
                    raw_distance = float(results["distances"][0][0])
                    metadata = results["metadatas"][0][0] or {}
                    cached_query = str((results.get("documents") or [[""]])[0][0] or "")
                    similarity = min(1.0, max(0.0, 1.0 - raw_distance))
                    latency_ms = (time.perf_counter() - start_time) * 1000.0

                    if similarity < effective_threshold:
                        logger.info(
                            "Semantic cache MISS: similarity score %.4f < threshold %.4f (dist: %.4f, latency: %.2fms)",
                            similarity,
                            effective_threshold,
                            raw_distance,
                            latency_ms,
                        )
                        try:
                            from backend.api.dependencies import get_telemetry_service
                            get_telemetry_service().record_cache_event("Semantic Cache", "MISS", latency_ms, model_name=model_name)
                        except Exception:
                            pass
                        return None

                    if not _queries_are_interchangeable(cached_query, query_clean):
                        logger.info("Semantic cache MISS: critical query entities differ from cached query")
                        return None

                    cached_kb = metadata.get("kb_version") if "kb_version" in metadata else None
                    if effective_kb_version is not None and cached_kb != effective_kb_version:
                        logger.info(
                            "Semantic cache MISS (kb_version mismatch): cached='%s', requested='%s'",
                            cached_kb,
                            effective_kb_version,
                        )
                        try:
                            from backend.api.dependencies import get_telemetry_service
                            get_telemetry_service().record_cache_event("Semantic Cache", "MISS", latency_ms, model_name=model_name)
                        except Exception:
                            pass
                        return None

                    if metadata.get("prompt_version", "") != effective_prompt_version:
                        logger.info("Semantic cache MISS (prompt version mismatch)")
                        return None

                    if metadata.get("cache_context", "") != cache_context:
                        logger.info("Semantic cache MISS (retrieval context mismatch)")
                        return None

                    cached_model = metadata.get("model") if "model" in metadata else None
                    if model_name is not None and cached_model != model_name:
                        logger.debug("Semantic cache MISS (model mismatch): cached='%s', requested='%s'", cached_model, model_name)
                        try:
                            from backend.api.dependencies import get_telemetry_service
                            get_telemetry_service().record_cache_event("Semantic Cache", "MISS", latency_ms, model_name=model_name)
                        except Exception:
                            pass
                        return None

                    answer = metadata.get("answer", "")
                    citations_raw = metadata.get("citations_json", "[]")
                    citations: List[Citation] = []
                    try:
                        citations_data = json.loads(citations_raw)
                        citations = [Citation(**c) for c in citations_data]
                    except Exception as c_err:
                        logger.warning("Failed to parse cached citations JSON: %s", c_err)

                    timestamp = float(metadata.get("timestamp", time.time()))

                    logger.info(
                        "Semantic cache HIT: score %.4f >= threshold %.4f (latency: %.2fms)",
                        similarity,
                        effective_threshold,
                        latency_ms,
                    )
                    try:
                        from backend.api.dependencies import get_telemetry_service
                        get_telemetry_service().record_cache_event("Semantic Cache", "HIT", latency_ms, model_name=model_name)
                    except Exception:
                        pass
                    return CachedResponse(
                        answer=answer,
                        citations=citations,
                        similarity_score=similarity,
                        distance=raw_distance,
                        lookup_latency_ms=latency_ms,
                        timestamp=timestamp,
                        kb_version=cached_kb,
                    )

            # Fallback In-Memory Lookup (if Chroma is empty or unavailable or in unit tests)
            with self._lock:
                memory_snapshot = list(self._memory_cache.values())

            if memory_snapshot:
                from backend.embeddings.vector_store import cosine_similarity

                best_score = -1.0
                best_entry = None
                for item in memory_snapshot:
                    if item.get("prompt_version", "") != effective_prompt_version:
                        continue
                    if item.get("cache_context", "") != cache_context:
                        continue
                    if not _queries_are_interchangeable(str(item.get("query", "")), query_clean):
                        continue
                    sim = cosine_similarity(query_embedding, item["embedding"])
                    if sim > best_score:
                        best_score = sim
                        best_entry = item

                latency_ms = (time.perf_counter() - start_time) * 1000.0
                if best_entry and best_score >= effective_threshold:
                    cached_kb = best_entry.get("kb_version") if "kb_version" in best_entry else None
                    if effective_kb_version is not None and cached_kb != effective_kb_version:
                        logger.info(
                            "Semantic cache MISS in memory (kb_version mismatch): cached='%s', requested='%s'",
                            cached_kb,
                            effective_kb_version,
                        )
                        return None

                    cached_model = best_entry.get("model") if "model" in best_entry else None
                    if model_name is not None and cached_model != model_name:
                        return None

                    raw_distance = max(0.0, 1.0 - best_score)
                    return CachedResponse(
                        answer=best_entry["answer"],
                        citations=best_entry["citations"],
                        similarity_score=best_score,
                        distance=raw_distance,
                        lookup_latency_ms=latency_ms,
                        timestamp=best_entry["timestamp"],
                        kb_version=cached_kb,
                    )

        except Exception as exc:
            logger.warning("Exception in semantic cache get(): %s", exc, exc_info=True)
            return None

        return None

    def put(
        self,
        query: str,
        answer: str,
        citations: List[Citation] | List[dict],
        kb_version: Optional[str] = None,
        model_name: Optional[str] = None,
        prompt_version: Optional[str] = None,
        cache_context: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Store query response in semantic cache if answer and citations are valid.
        Safe execution guarantees cache store failures NEVER fail the caller request.
        """
        try:
            cache_enabled = getattr(self.settings, "semantic_cache_enabled", True)
            if not cache_enabled:
                return False

            if not query or not isinstance(query, str) or not query.strip():
                logger.warning("Semantic cache put rejected: empty query.")
                return False

            if not answer or not isinstance(answer, str) or not answer.strip():
                logger.warning("Semantic cache put rejected: empty answer.")
                return False

            if not citations or not isinstance(citations, list) or len(citations) == 0:
                logger.warning("Semantic cache put rejected: empty citations.")
                return False

            validated_citations: List[Citation] = []
            for c in citations:
                if isinstance(c, Citation):
                    validated_citations.append(c)
                elif isinstance(c, dict):
                    try:
                        validated_citations.append(Citation(**c))
                    except Exception as parse_err:
                        logger.warning("Invalid citation dictionary in put(): %s", parse_err)

            if not validated_citations:
                logger.warning("Semantic cache put rejected: no valid Citation objects.")
                return False

            query_clean = query.strip()
            effective_kb_version = self._resolve_kb_version(kb_version)
            effective_prompt_version = self._resolve_prompt_version(prompt_version)
            query_embedding = self.embedding_service.embed_text(query_clean)
            if not query_embedding:
                logger.warning("Semantic cache put failed: embedding generation returned empty.")
                return False

            hash_str = "|".join(
                [
                    _normalized_query(query_clean),
                    model_name or "",
                    effective_kb_version or "",
                    effective_prompt_version,
                    cache_context,
                ]
            )
            entry_id = hashlib.sha256(hash_str.encode("utf-8")).hexdigest()
            citations_json = json.dumps([c.model_dump() for c in validated_citations])
            ts = time.time()

            meta_dict: Dict[str, Any] = {
                "answer": answer,
                "citations_json": citations_json,
                "timestamp": ts,
                "kb_version": effective_kb_version if effective_kb_version is not None else "",
                "model": model_name if model_name is not None else "",
                "prompt_version": effective_prompt_version,
                "cache_context": cache_context,
            }

            if metadata:
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        meta_dict[f"custom_{k}"] = v

            with self._lock:
                if self._collection is not None:
                    self._collection.upsert(
                        ids=[entry_id],
                        documents=[query_clean],
                        embeddings=[query_embedding],
                        metadatas=[meta_dict],
                    )

                self._memory_cache[entry_id] = {
                    "query": query_clean,
                    "answer": answer,
                    "citations": validated_citations,
                    "embedding": query_embedding,
                    "timestamp": ts,
                    "kb_version": effective_kb_version if effective_kb_version is not None else "",
                    "model": model_name if model_name is not None else "",
                    "prompt_version": effective_prompt_version,
                    "cache_context": cache_context,
                }

            logger.info("Successfully stored entry in semantic cache (id: %s)", entry_id[:8])
            return True

        except Exception as exc:
            logger.warning("Exception during semantic cache put(): %s", exc, exc_info=True)
            return False

    def clear(self) -> None:
        """Delete all items in semantic_cache collection."""
        try:
            with self._lock:
                self._memory_cache.clear()
                if self._collection is not None:
                    all_ids = self._collection.get(include=[])["ids"]
                    if all_ids:
                        self._collection.delete(ids=all_ids)
                    logger.info(
                        "Cleared all entries from semantic cache collection '%s'",
                        self.collection_name,
                    )
        except Exception as exc:
            logger.warning("Failed to clear semantic cache collection: %s", exc)


SemanticCache = SemanticCacheManager
