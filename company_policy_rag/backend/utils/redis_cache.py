from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

try:
    import redis
    REDIS_INSTALLED = True
except ImportError:
    redis = None  # type: ignore[assignment]
    REDIS_INSTALLED = False

from backend.utils.logging import logger


class RedisCache:
    """
    Production-ready Redis client with a thread-safe in-memory fallback.
    Provides response caching, vector embedding caching, session storage, and general KV storage.
    Gracefully handles Redis connection failures without interrupting operations.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        db: int = 0,
        password: str | None = None,
        redis_url: str | None = None,
        default_ttl: int = 3600,
        enabled: bool = True,
    ) -> None:
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = int(port or os.getenv("REDIS_PORT", "6379"))
        self.db = int(db or os.getenv("REDIS_DB", "0"))
        self.password = password or os.getenv("REDIS_PASSWORD", None)
        self.redis_url = redis_url or os.getenv("REDIS_URL", None)
        self.default_ttl = default_ttl
        self.enabled = enabled and (os.getenv("REDIS_ENABLED", "true").lower() in ("true", "1", "yes"))

        self._redis_client: Any | None = None
        self._redis_connected = False
        self._memory_store: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

        if self.enabled and REDIS_INSTALLED:
            self._connect()

    def _connect(self) -> None:
        try:
            if self.redis_url:
                client = redis.Redis.from_url(
                    self.redis_url,
                    db=self.db,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                    decode_responses=True,
                )
            else:
                client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                    decode_responses=True,
                )
            client.ping()
            self._redis_client = client
            self._redis_connected = True
            logger.info("Connected to Redis at %s:%s (db=%s)", self.host, self.port, self.db)
        except Exception as exc:
            self._redis_client = None
            self._redis_connected = False
            logger.warning("Redis connection failed (%s). Falling back to in-memory cache.", exc)

    def is_redis_available(self) -> bool:
        """Check if Redis connection is active and operational."""
        if not self._redis_connected or not self._redis_client:
            return False
        try:
            return bool(self._redis_client.ping())
        except Exception:
            self._redis_connected = False
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve key from cache (Redis or in-memory fallback)."""
        if self._redis_connected and self._redis_client:
            try:
                raw = self._redis_client.get(key)
                if raw is not None:
                    try:
                        return json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        return raw
                return default
            except Exception as exc:
                logger.warning("Redis get error for key '%s': %s. Falling back to in-memory.", key, exc)
                self._redis_connected = False

        # In-memory fallback
        with self._lock:
            if key in self._memory_store:
                item = self._memory_store[key]
                expire_at = item.get("expire_at")
                if expire_at is None or expire_at > time.time():
                    return item.get("value")
                # Expired
                del self._memory_store[key]
            return default

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Store key in cache with optional TTL in seconds."""
        expire_ttl = ttl if ttl is not None else self.default_ttl
        serialized = json.dumps(value) if isinstance(value, (dict, list, bool)) else str(value)

        if self._redis_connected and self._redis_client:
            try:
                if expire_ttl > 0:
                    self._redis_client.setex(key, expire_ttl, serialized)
                else:
                    self._redis_client.set(key, serialized)
                return True
            except Exception as exc:
                logger.warning("Redis set error for key '%s': %s. Falling back to in-memory.", key, exc)
                self._redis_connected = False

        # In-memory fallback
        with self._lock:
            expire_at = time.time() + expire_ttl if expire_ttl > 0 else None
            self._memory_store[key] = {
                "value": value,
                "expire_at": expire_at,
            }
            return True

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        deleted = False
        if self._redis_connected and self._redis_client:
            try:
                deleted = bool(self._redis_client.delete(key))
            except Exception as exc:
                logger.warning("Redis delete error for key '%s': %s", key, exc)
                self._redis_connected = False

        with self._lock:
            if key in self._memory_store:
                del self._memory_store[key]
                deleted = True
        return deleted

    def flush(self) -> bool:
        """Clear all cached items."""
        if self._redis_connected and self._redis_client:
            try:
                self._redis_client.flushdb()
            except Exception as exc:
                logger.warning("Redis flush error: %s", exc)
                self._redis_connected = False

        with self._lock:
            self._memory_store.clear()
        return True

    # High-level specialized methods

    def get_query_cache(self, query_hash: str) -> dict[str, Any] | None:
        """Retrieve cached query response by query hash."""
        return self.get(f"query:{query_hash}")

    def set_query_cache(self, query_hash: str, response_data: dict[str, Any], ttl: int = 3600) -> bool:
        """Store query response in cache."""
        return self.set(f"query:{query_hash}", response_data, ttl=ttl)

    def get_embedding_cache(self, text_hash: str) -> list[float] | None:
        """Retrieve cached vector embedding by text hash."""
        return self.get(f"emb:{text_hash}")

    def set_embedding_cache(self, text_hash: str, embedding: list[float], ttl: int = 86400) -> bool:
        """Store vector embedding in cache."""
        return self.set(f"emb:{text_hash}", embedding, ttl=ttl)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve session data by session ID."""
        return self.get(f"session:{session_id}")

    def set_session(self, session_id: str, session_data: dict[str, Any], ttl: int = 86400) -> bool:
        """Store session data in cache."""
        return self.set(f"session:{session_id}", session_data, ttl=ttl)

    def delete_session(self, session_id: str) -> bool:
        """Delete session data from cache."""
        return self.delete(f"session:{session_id}")


_redis_cache_instance: RedisCache | None = None


def get_redis_cache() -> RedisCache:
    """Get global RedisCache singleton instance."""
    global _redis_cache_instance
    if _redis_cache_instance is None:
        _redis_cache_instance = RedisCache()
    return _redis_cache_instance


redis_cache = get_redis_cache()
