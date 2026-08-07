from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from backend.utils.redis_cache import RedisCache, get_redis_cache, redis_cache


def test_redis_cache_in_memory_fallback():
    cache = RedisCache(enabled=False)
    assert cache.is_redis_available() is False

    # Key-value set and get
    assert cache.set("test_key", {"foo": "bar"}, ttl=10) is True
    assert cache.get("test_key") == {"foo": "bar"}

    # Delete
    assert cache.delete("test_key") is True
    assert cache.get("test_key") is None


def test_redis_cache_ttl_expiration():
    cache = RedisCache(enabled=False)
    cache.set("short_key", "value", ttl=1)
    assert cache.get("short_key") == "value"
    time.sleep(1.1)
    assert cache.get("short_key") is None


def test_redis_cache_specialized_helpers():
    cache = RedisCache(enabled=False)

    # Query cache
    cache.set_query_cache("q123", {"answer": "42"})
    assert cache.get_query_cache("q123") == {"answer": "42"}

    # Embedding cache
    vec = [0.1, 0.2, 0.3]
    cache.set_embedding_cache("emb123", vec)
    assert cache.get_embedding_cache("emb123") == vec

    # Session cache
    session_data = {"user_id": "u1", "messages": []}
    cache.set_session("s123", session_data)
    assert cache.get_session("s123") == session_data
    assert cache.delete_session("s123") is True
    assert cache.get_session("s123") is None


def test_redis_cache_flush():
    cache = RedisCache(enabled=False)
    cache.set("k1", "v1")
    cache.set("k2", "v2")
    cache.flush()
    assert cache.get("k1") is None
    assert cache.get("k2") is None


def test_redis_cache_graceful_error_handling():
    cache = RedisCache(enabled=False)
    mock_redis = MagicMock()
    mock_redis.ping.side_effect = Exception("Redis network error")
    mock_redis.get.side_effect = Exception("Connection lost")
    cache._redis_client = mock_redis
    cache._redis_connected = True

    # Should catch exception, mark disconnected, and fallback to memory
    assert cache.is_redis_available() is False
    val = cache.get("nonexistent", default="fallback")
    assert val == "fallback"


def test_global_singleton():
    g_cache = get_redis_cache()
    assert g_cache is not None
    assert redis_cache is not None
