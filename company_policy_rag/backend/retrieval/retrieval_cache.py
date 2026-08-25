from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

from backend.models.rag import ScoredChunk


class RetrievalCache:
    """
    Thread-safe in-memory LRU cache with TTL for retrieval candidate chunks and reranked results.
    Prevents redundant vector search and cross-encoder reranking on identical or normalized queries.
    """

    def __init__(self, max_size: int = 2000, default_ttl: int = 3600) -> None:
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, tuple[float, list[ScoredChunk]]] = OrderedDict()
        self._lock = threading.Lock()

    def _make_key(self, query: str, filters: Optional[dict[str, Any]], top_k: int) -> str:
        clean_q = query.strip().lower()
        filter_str = json.dumps(filters or {}, sort_keys=True)
        raw = f"{clean_q}|{filter_str}|{top_k}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(
        self, query: str, filters: Optional[dict[str, Any]] = None, top_k: int = 5
    ) -> list[ScoredChunk] | None:
        key = self._make_key(query, filters, top_k)
        now = time.time()
        with self._lock:
            if key not in self._cache:
                return None
            expiry, results = self._cache[key]
            if now > expiry:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return results

    def set(
        self,
        query: str,
        results: list[ScoredChunk],
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 5,
        ttl: Optional[int] = None,
    ) -> None:
        key = self._make_key(query, filters, top_k)
        expiry = time.time() + (ttl if ttl is not None else self._default_ttl)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (expiry, results)
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


_global_retrieval_cache = RetrievalCache()


def get_retrieval_cache() -> RetrievalCache:
    return _global_retrieval_cache
