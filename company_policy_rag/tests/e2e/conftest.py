"""
Pytest Fixtures for Opaque-Box E2E Tests.
Configures FastAPI ASGI AsyncClient, Redis Async Client, PubSub Listener, and Celery Eager/Live Worker fixtures.
"""
from __future__ import annotations

import os
from typing import AsyncGenerator, Generator, Any

import httpx
import pytest
import pytest_asyncio

from backend.api.main import create_app
from backend.tasks.celery_app import celery_app
from backend.utils.redis_client import get_redis_client, close_redis_client
from tests.e2e.helpers.redis_helper import RedisPubSubHelper, get_redis_test_url
from tests.e2e.helpers.celery_helper import CeleryTestHelper


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """
    Provides an async HTTP client connected directly to the FastAPI ASGI app.
    Supports line-by-line reading of streaming responses (`text/event-stream`).
    """
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
        headers={"Content-Type": "application/json"},
    ) as client:
        yield client


@pytest_asyncio.fixture
async def redis_client() -> AsyncGenerator[Any, None]:
    """
    Provides an async Redis client connected to the configured broker/Redis URL.
    Verifies connection via PING before yielding.
    """
    try:
        import redis.asyncio as aioredis
        url = get_redis_test_url()
        client = aioredis.from_url(
            url, decode_responses=True, socket_connect_timeout=2.0, socket_timeout=5.0
        )
        await client.ping()
        yield client
        await client.aclose()
    except Exception as exc:
        pytest.skip(f"Redis server unavailable for E2E tests: {exc}")


@pytest_asyncio.fixture
async def redis_pubsub_listener(redis_client: Any) -> RedisPubSubHelper:
    """
    Provides a RedisPubSubHelper instance bound to the active Redis client.
    """
    return RedisPubSubHelper(redis_client)


@pytest.fixture
def eager_celery() -> Generator[Any, None, None]:
    """
    Configures Celery to run background tasks synchronously in-process for testing.
    Restores original eager settings upon teardown.
    """
    if celery_app is None:
        pytest.skip("Celery app is not initialized.")

    orig_eager = celery_app.conf.task_always_eager
    orig_propagate = celery_app.conf.task_eager_propagates
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True

    try:
        yield celery_app
    finally:
        celery_app.conf.task_always_eager = orig_eager
        celery_app.conf.task_eager_propagates = orig_propagate


@pytest.fixture
def check_live_celery_worker() -> Any:
    """
    Fixtures checking if a live Celery worker is active and listening on the Redis broker.
    Skips the test if no live worker responds to ping.
    """
    if not CeleryTestHelper.is_celery_available():
        pytest.skip("Celery application unavailable.")

    if not CeleryTestHelper.ping_worker(timeout=2.0):
        pytest.skip("No live Celery worker process responding on broker.")

    return True
