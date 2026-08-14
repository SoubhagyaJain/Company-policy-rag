"""
Redis PubSub Listener & Helper Module for E2E Tests.
Provides channel subscription, event collection, publishing, and connectivity checks.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    aioredis = None  # type: ignore[assignment]
    REDIS_AVAILABLE = False


def get_redis_test_url() -> str:
    """Retrieve Redis connection URL for E2E tests."""
    return (
        os.getenv("CELERY_BROKER_URL")
        or os.getenv("REDIS_URL")
        or f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}/0"
    )


class RedisPubSubHelper:
    """Helper class for subscribing to Redis channels and listening/publishing SSE events."""

    def __init__(self, client: Optional[Any] = None) -> None:
        self._external_client = client
        self._owned_client: Optional[Any] = None

    async def get_client(self) -> Any:
        """Get or initialize active async Redis client instance."""
        if self._external_client is not None:
            return self._external_client

        if self._owned_client is None:
            if not REDIS_AVAILABLE:
                raise RuntimeError("redis package is not installed.")
            url = get_redis_test_url()
            self._owned_client = aioredis.from_url(
                url, decode_responses=True, socket_connect_timeout=2.0, socket_timeout=5.0
            )
        return self._owned_client

    async def check_redis_alive(self) -> bool:
        """Check if Redis server is reachable via PING."""
        try:
            client = await self.get_client()
            res = await client.ping()
            return bool(res)
        except Exception:
            return False

    async def collect_events(
        self,
        channel_name: str,
        timeout: float = 5.0,
        stop_events: Tuple[str, ...] = ("done", "error"),
        max_messages: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Subscribe to channel_name and collect published event JSON payloads until timeout or stop event.
        """
        client = await self.get_client()
        pubsub = client.pubsub()
        await pubsub.subscribe(channel_name)

        events: List[Dict[str, Any]] = []
        start_time = asyncio.get_event_loop().time()

        try:
            while (asyncio.get_event_loop().time() - start_time) < timeout:
                if len(events) >= max_messages:
                    break

                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.2)
                if message and message.get("type") == "message":
                    raw_data = message["data"]
                    try:
                        payload = json.loads(raw_data)
                    except (json.JSONDecodeError, TypeError):
                        payload = {"event": "raw", "data": raw_data}

                    events.append(payload)
                    event_name = payload.get("event") if isinstance(payload, dict) else None
                    if event_name in stop_events:
                        break

                await asyncio.sleep(0.01)
        finally:
            try:
                await pubsub.unsubscribe(channel_name)
                await pubsub.aclose()
            except Exception:
                pass

        return events

    async def publish_event(
        self, channel_name: str, event_name: str, data: Dict[str, Any]
    ) -> int:
        """
        Publish a JSON-encoded event object to channel_name.
        Format matches PROJECT.md contract: {"event": event_name, "data": data}.
        """
        client = await self.get_client()
        payload = json.dumps({"event": event_name, "data": data})
        return await client.publish(channel_name, payload)

    async def close(self) -> None:
        """Close owned client connection if created."""
        if self._owned_client is not None:
            try:
                await self._owned_client.aclose()
            except Exception:
                pass
            self._owned_client = None
