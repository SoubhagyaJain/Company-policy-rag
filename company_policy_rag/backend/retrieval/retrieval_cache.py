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
    Thread-safe in-memory LRU cache with TTL for retrieval candidate chunks.

    Entries are keyed on a caller-supplied ``version`` (corpus fingerprint plus
    the retrieval settings that shape the candidate list), so an upload, delete,
    or settings change can never serve a stale candidate set. Document
    mutations also call :meth:`clear`.
    """

    def __init__(self, max_size: int = 2000, default_ttl: int = 3600) -> None:
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, tuple[float, list[ScoredChunk]]] = OrderedDict()
        self._lock = threading.Lock()
        # Lookup counters since this process started, for the Storage tab.
        self._hits = 0
        self._misses = 0
        self._last_hit_at: float | None = None
        self._counting_since = time.time()

    def _make_key(
        self, query: str, filters: Optional[dict[str, Any]], top_k: int, version: str = ""
    ) -> str:
        clean_q = query.strip().lower()
        filter_str = json.dumps(filters or {}, sort_keys=True, default=str)
        raw = f"{clean_q}|{filter_str}|{top_k}|{version}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 5,
        version: str = "",
    ) -> list[ScoredChunk] | None:
        key = self._make_key(query, filters, top_k, version)
        now = time.time()
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            expiry, results = self._cache[key]
            if now > expiry:
                del self._cache[key]
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            self._last_hit_at = now
            return list(results)

    def set(
        self,
        query: str,
        results: list[ScoredChunk],
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 5,
        ttl: Optional[int] = None,
        version: str = "",
    ) -> None:
        key = self._make_key(query, filters, top_k, version)
        expiry = time.time() + (ttl if ttl is not None else self._default_ttl)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (expiry, list(results))
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def stats(self) -> dict[str, Any]:
        """Entry count, limits and lookup counters since the process started."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "max_entries": self._max_size,
                "ttl_seconds": self._default_ttl,
                "hits": self._hits,
                "misses": self._misses,
                "last_hit_at": self._last_hit_at,
                "counting_since": self._counting_since,
            }


_global_retrieval_cache = RetrievalCache()


def get_retrieval_cache() -> RetrievalCache:
    return _global_retrieval_cache
