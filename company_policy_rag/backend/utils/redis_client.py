"""
Async Redis Client & Connection Pool Manager for FastAPI & Redis Pub/Sub Streaming.
Provides asynchronous connection getters, connection pooling, and healthchecks.
"""
from __future__ import annotations

import os
from typing import Optional
from dotenv import load_dotenv

try:
    import redis.asyncio as aioredis
    from redis.asyncio import Redis, ConnectionPool
    REDIS_ASYNC_AVAILABLE = True
except ImportError:
    aioredis = None  # type: ignore[assignment]
    Redis = None  # type: ignore[assignment]
    ConnectionPool = None  # type: ignore[assignment]
    REDIS_ASYNC_AVAILABLE = False

from backend.utils.logging import logger

load_dotenv()

_redis_pool: Optional[ConnectionPool] = None
_redis_client: Optional[Redis] = None


def get_redis_connection_url() -> str:
    """Construct Redis URL from environment variables."""
    url = os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_URL")
    if url:
        return url
    
    host = os.getenv("REDIS_HOST", "localhost")
    port = os.getenv("REDIS_PORT", "6379")
    db = os.getenv("REDIS_DB", "0")
    password = os.getenv("REDIS_PASSWORD", "")
    
    if password:
        return f"redis://:{password}@{host}:{port}/{db}"
    return f"redis://{host}:{port}/{db}"


def get_redis_pool() -> ConnectionPool:
    """Initialize or return existing Redis async ConnectionPool singleton."""
    global _redis_pool
    if not REDIS_ASYNC_AVAILABLE:
        raise RuntimeError("redis package is not installed or redis.asyncio unavailable.")
    
    if _redis_pool is None:
        redis_url = get_redis_connection_url()
        _redis_pool = ConnectionPool.from_url(
            redis_url,
            max_connections=20,
            decode_responses=True,
            socket_connect_timeout=2.0,
            socket_timeout=5.0,
        )
        logger.info("Initialized Redis async connection pool.")
    return _redis_pool


async def get_redis_client() -> Redis:
    """Get or create singleton async Redis client instance."""
    global _redis_client
    if not REDIS_ASYNC_AVAILABLE:
        raise RuntimeError("redis package is not installed.")
    
    if _redis_client is None:
        pool = get_redis_pool()
        _redis_client = Redis(connection_pool=pool)
    return _redis_client


async def close_redis_client() -> None:
    """Close async Redis client and release pool resources on application shutdown."""
    global _redis_client, _redis_pool
    if _redis_client is not None:
        try:
            if hasattr(_redis_client, "aclose"):
                await _redis_client.aclose()
            else:
                await _redis_client.close()
            logger.info("Closed Redis async client.")
        except Exception as exc:
            logger.warning("Error closing Redis client: %s", exc)
        _redis_client = None
    
    if _redis_pool is not None:
        try:
            await _redis_pool.disconnect()
            logger.info("Disconnected Redis connection pool.")
        except Exception as exc:
            logger.warning("Error disconnecting Redis pool: %s", exc)
        _redis_pool = None


async def check_redis_connection() -> bool:
    """
    Ping Redis server to verify connectivity.
    Returns True if connection is alive, False otherwise.
    """
    if not REDIS_ASYNC_AVAILABLE:
        return False
    try:
        client = await get_redis_client()
        pong = await client.ping()
        return bool(pong)
    except Exception as exc:
        logger.warning("Async Redis ping failed: %s", exc)
        return False


async def get_redis_pubsub():
    """Obtain a new PubSub instance for subscribing to channels."""
    client = await get_redis_client()
    return client.pubsub()
