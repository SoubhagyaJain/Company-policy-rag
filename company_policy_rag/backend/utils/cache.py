"""Async Redis cache utility for the RAG backend.

Wraps redis.asyncio with graceful fallback — if Redis is unreachable,
every public method logs a warning and returns None / False so the
caller can proceed without caching.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

try:
    from redis.asyncio import Redis as AsyncRedis
    REDIS_ASYNC_AVAILABLE = True
except ImportError:
    AsyncRedis = None  # type: ignore[assignment, misc]
    REDIS_ASYNC_AVAILABLE = False

from backend.utils.logging import logger

DEFAULT_TTL: int = 3600  # 1 hour


class AsyncRedisCache:
    """Production-ready async Redis client with graceful degradation.

    All public methods catch connection / timeout errors and return a
    safe default so callers never crash due to cache unavailability.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: int = 0,
        password: Optional[str] = None,
        redis_url: Optional[str] = None,
        default_ttl: int = DEFAULT_TTL,
    ) -> None:
        self._host = host or os.getenv("REDIS_HOST", "localhost")
        self._port = int(port or os.getenv("REDIS_PORT", "6379"))
        self._db = int(db or os.getenv("REDIS_DB", "0"))
        self._password = password or os.getenv("REDIS_PASSWORD") or None
        self._redis_url = redis_url or os.getenv("REDIS_URL") or None
        self._default_ttl = default_ttl
        self._client: Optional[Any] = None

    async def _get_client(self) -> Optional[Any]:
        """Lazily create and verify the async Redis connection."""
        if self._client is not None:
            return self._client

        if not REDIS_ASYNC_AVAILABLE:
            logger.warning("redis.asyncio is not installed — cache disabled.")
            return None

        try:
            if self._redis_url:
                client = AsyncRedis.from_url(
                    self._redis_url,
                    db=self._db,
                    decode_responses=True,
                    socket_connect_timeout=3,
                    socket_timeout=3,
                )
            else:
                client = AsyncRedis(
                    host=self._host,
                    port=self._port,
                    db=self._db,
                    password=self._password,
                    decode_responses=True,
                    socket_connect_timeout=3,
                    socket_timeout=3,
                )
            await client.ping()
            self._client = client
            logger.info(
                "Async Redis connected at %s:%s (db=%s)",
                self._host, self._port, self._db,
            )
            return self._client
        except Exception as exc:
            logger.warning("Async Redis connection failed: %s — cache disabled.", exc)
            self._client = None
            return None

    # ── Core CRUD ────────────────────────────────────────────

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve a value by key. Returns None on miss or error."""
        client = await self._get_client()
        if client is None:
            return None
        try:
            raw = await client.get(key)
            if raw is None:
                return None
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return raw
        except Exception as exc:
            logger.warning("Redis GET error for '%s': %s", key, exc)
            self._client = None
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Store a value with optional TTL (seconds). Returns False on error."""
        client = await self._get_client()
        if client is None:
            return False
        expire = ttl if ttl is not None else self._default_ttl
        try:
            serialized = json.dumps(value) if isinstance(value, (dict, list, bool)) else str(value)
            if expire > 0:
                await client.setex(key, expire, serialized)
            else:
                await client.set(key, serialized)
            return True
        except Exception as exc:
            logger.warning("Redis SET error for '%s': %s", key, exc)
            self._client = None
            return False

    async def delete(self, key: str) -> bool:
        """Delete a key. Returns False on error."""
        client = await self._get_client()
        if client is None:
            return False
        try:
            result = await client.delete(key)
            return bool(result)
        except Exception as exc:
            logger.warning("Redis DELETE error for '%s': %s", key, exc)
            self._client = None
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists. Returns False on error."""
        client = await self._get_client()
        if client is None:
            return False
        try:
            return bool(await client.exists(key))
        except Exception as exc:
            logger.warning("Redis EXISTS error for '%s': %s", key, exc)
            self._client = None
            return False

    async def flush(self) -> bool:
        """Flush the current database. Returns False on error."""
        client = await self._get_client()
        if client is None:
            return False
        try:
            await client.flushdb()
            return True
        except Exception as exc:
            logger.warning("Redis FLUSHDB error: %s", exc)
            self._client = None
            return False

    # ── Cache key generation ─────────────────────────────────

    @staticmethod
    def make_key(query: str, filters: Optional[Dict[str, Any]] = None) -> str:
        """Generate a deterministic cache key from a query string and optional filters.

        Uses SHA-256 to produce a fixed-length, collision-resistant key
        prefixed with ``query:`` for namespace isolation.
        """
        payload = query.strip().lower()
        if filters:
            payload += "|" + json.dumps(filters, sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"query:{digest}"

    # ── High-level helpers ───────────────────────────────────

    async def get_query_cache(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a cached RAG response by query + filters hash."""
        key = self.make_key(query, filters)
        result = await self.get(key)
        if isinstance(result, dict):
            return result
        return None

    async def set_query_cache(
        self,
        query: str,
        response_data: Dict[str, Any],
        filters: Optional[Dict[str, Any]] = None,
        ttl: int = DEFAULT_TTL,
    ) -> bool:
        """Cache a RAG response keyed by query + filters hash."""
        key = self.make_key(query, filters)
        return await self.set(key, response_data, ttl=ttl)

    async def get_embedding_cache(self, text: str) -> Optional[List[float]]:
        """Retrieve a cached embedding vector by text hash."""
        digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        key = f"emb:{digest}"
        result = await self.get(key)
        if isinstance(result, list):
            return result
        return None

    async def set_embedding_cache(
        self,
        text: str,
        embedding: List[float],
        ttl: int = 86400,
    ) -> bool:
        """Cache an embedding vector keyed by text hash."""
        digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        key = f"emb:{digest}"
        return await self.set(key, embedding, ttl=ttl)

    async def close(self) -> None:
        """Gracefully close the Redis connection."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            finally:
                self._client = None


# ── Module-level singleton ────────────────────────────────────

_async_cache_instance: Optional[AsyncRedisCache] = None


def get_async_cache() -> AsyncRedisCache:
    """Return the global AsyncRedisCache singleton."""
    global _async_cache_instance
    if _async_cache_instance is None:
        _async_cache_instance = AsyncRedisCache()
    return _async_cache_instance
