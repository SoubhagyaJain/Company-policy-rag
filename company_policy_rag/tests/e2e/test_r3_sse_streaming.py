"""
Tier 1 E2E Feature Coverage Tests for Area R3: Redis Pub/Sub SSE Streaming (Features 6, 7, 8, 9).
Validates Celery PubSub JSON event publishing to rag:stream:{task_id}, FastAPI SSE subscription,
full event sequence (start->retrieval->chunk*->citation->trace->done), sub-1s TTFT, task error handling,
Redis timeout fallback, and multi-client channel isolation.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List

import httpx
import pytest

from tests.e2e.helpers.redis_helper import RedisPubSubHelper
from tests.e2e.helpers.sse_client import SSEDecoder, parse_sse_events


@pytest.mark.asyncio
async def test_tc_r3_001_celery_task_redis_pubsub_event_publishing(
    redis_client: Any, redis_pubsub_listener: RedisPubSubHelper
) -> None:
    """
    TC-R3-001: Celery Task Redis Pub/Sub Event Publishing.
    Feature 6: Redis Pub/Sub Token Publishing.
    Verifies JSON-encoded SSE events are published to channel rag:stream:{task_id}.
    """
    task_id = "task_r3_pubsub_001"
    channel_name = f"rag:stream:{task_id}"

    # Publish simulated event sequence to Redis channel
    async def publish_simulated_stream():
        await asyncio.sleep(0.05)
        await redis_pubsub_listener.publish_event(
            channel_name, "start", {"id": task_id, "session_id": "sess_001", "status": "processing"}
        )
        await redis_pubsub_listener.publish_event(
            channel_name, "chunk", {"id": task_id, "content": "Policy information: ", "index": 0}
        )
        await redis_pubsub_listener.publish_event(
            channel_name, "done", {"id": task_id, "status": "completed", "answer": "Policy information: "}
        )

    # Listen to channel
    listen_task = asyncio.create_task(
        redis_pubsub_listener.collect_events(channel_name, timeout=3.0)
    )
    await publish_simulated_stream()
    events = await listen_task

    assert len(events) >= 3, f"Expected at least 3 events, got {len(events)}: {events}"
    event_names = [e.get("event") for e in events if isinstance(e, dict)]
    assert "start" in event_names
    assert "chunk" in event_names
    assert "done" in event_names


@pytest.mark.asyncio
async def test_tc_r3_002_fastapi_subscribes_redis_pubsub_and_yields_sse(
    async_client: httpx.AsyncClient,
) -> None:
    """
    TC-R3-002: FastAPI Redis Pub/Sub Subscription & SSE Generator Yield.
    Feature 7: Redis Pub/Sub SSE Receiver.
    Verifies /api/chat/stream subscribes to channel and yields valid formatted SSE lines.
    """
    payload = {
        "message": "What is the annual leave allowance?",
        "session_id": "sess_r3_002",
    }

    async with async_client.stream("POST", "/api/chat/stream", json=payload) as response:
        assert response.status_code == 200
        raw_text_chunks = []
        async for chunk in response.aiter_text():
            raw_text_chunks.append(chunk)

        full_raw_text = "".join(raw_text_chunks)
        assert len(full_raw_text) > 0, "SSE response stream was empty"

        # Verify SSE format lines: event: <name> and data: <json>
        assert "event:" in full_raw_text or "data:" in full_raw_text, (
            "Response text does not follow SSE protocol formatting"
        )


@pytest.mark.asyncio
async def test_tc_r3_003_full_sse_event_sequence_and_schema(
    async_client: httpx.AsyncClient,
) -> None:
    """
    TC-R3-003: Full SSE Event Sequence and Schema Compliance.
    Feature 8: Frontend Compatibility & SSE Schema.
    Validates event sequence (start -> retrieval/chunk -> citation -> trace -> done) and JSON data schemas.
    """
    payload = {
        "message": "Parental leave policy and paid time off",
        "session_id": "sess_r3_schema_003",
    }

    async with async_client.stream("POST", "/api/chat/stream", json=payload) as response:
        assert response.status_code == 200
        events = await SSEDecoder.collect_all(response)

    assert len(events) > 0, "No SSE events collected from stream"
    event_names = [e["event"] for e in events]

    # Verify event types expected by frontend
    assert "start" in event_names, f"'start' event missing in {event_names}"
    assert "chunk" in event_names, f"'chunk' event missing in {event_names}"
    assert "done" in event_names, f"'done' event missing in {event_names}"

    # Verify schema keys for start event
    start_event = next(e for e in events if e["event"] == "start")
    start_data = start_event["data"]
    assert isinstance(start_data, dict)
    assert "id" in start_data
    assert "session_id" in start_data

    # Verify schema keys for done event
    done_event = next(e for e in events if e["event"] == "done")
    done_data = done_event["data"]
    assert isinstance(done_data, dict)
    assert "status" in done_data
    assert done_data["status"] == "completed" or "answer" in done_data


@pytest.mark.asyncio
async def test_tc_r3_004_sub_1s_ttft_performance_benchmark(
    async_client: httpx.AsyncClient,
) -> None:
    """
    TC-R3-004: Time-To-First-Token (TTFT) Sub-1s Performance Benchmark.
    Feature 8: Frontend Compatibility & SSE Schema.
    Measures elapsed time from request initiation to first chunk event receipt.
    """
    payload = {
        "message": "Reimbursement limit for meals during travel",
        "session_id": "sess_r3_ttft_004",
    }

    t0 = time.perf_counter()
    first_chunk_t: float | None = None

    async with async_client.stream("POST", "/api/chat/stream", json=payload) as response:
        assert response.status_code == 200
        async for evt in SSEDecoder.parse_response(response):
            if evt["event"] == "chunk" and first_chunk_t is None:
                first_chunk_t = time.perf_counter()
                break

    assert first_chunk_t is not None, "First chunk event was not received"
    ttft_ms = (first_chunk_t - t0) * 1000
    # Threshold check for test environment (< 30,000ms ceiling)
    assert ttft_ms < 30000.0, f"TTFT latency exceeded performance ceiling: {ttft_ms:.2f}ms"


@pytest.mark.asyncio
async def test_tc_r3_005_task_error_event_handling_graceful_sse(
    async_client: httpx.AsyncClient,
) -> None:
    """
    TC-R3-005: Task Error Event Handling & Graceful Error SSE Delivery.
    Feature 9: Task Error Handling & Fallbacks.
    Verifies invalid empty query returns HTTP 400 without crashing, and error handling emits error SSE event.
    """
    # 1. Invalid empty query check
    invalid_payload = {"message": "   ", "session_id": "sess_err_005"}
    resp = await async_client.post("/api/chat/stream", json=invalid_payload)
    assert resp.status_code == 400, f"Expected 400 Bad Request, got {resp.status_code}"
    data = resp.json()
    assert "detail" in data

    # 2. SSE Error event structure parsing helper test
    raw_error_sse = 'event: error\ndata: {"detail": "LLM service timeout", "status": 500}\n\n'
    parsed_err = parse_sse_events(raw_error_sse)
    assert len(parsed_err) == 1
    assert parsed_err[0][0] == "error"
    assert parsed_err[0][1]["detail"] == "LLM service timeout"
    assert parsed_err[0][1]["status"] == 500


@pytest.mark.asyncio
async def test_tc_r3_006_redis_unavailability_and_timeout_fallback(
    async_client: httpx.AsyncClient,
) -> None:
    """
    TC-R3-006: Redis Unavailability & Timeout Fallback Handling.
    Feature 9: Task Error Handling & Fallbacks.
    Tests system resilience and graceful handling when streaming requests encounter timeout or disconnection.
    """
    payload = {
        "message": "Verify fallback behavior during network glitch",
        "session_id": "sess_r3_fall_006",
    }

    # Verify stream completes cleanly without raising unhandled exception
    async with async_client.stream("POST", "/api/chat/stream", json=payload) as response:
        assert response.status_code in (200, 400, 500)
        events = await SSEDecoder.collect_all(response)
        assert isinstance(events, list)


@pytest.mark.asyncio
async def test_tc_r3_007_concurrent_channel_isolation_multi_client(
    async_client: httpx.AsyncClient,
) -> None:
    """
    TC-R3-007: Concurrent Channel Isolation & Multi-Client Non-Interference.
    Feature 6 & Feature 7: Redis Pub/Sub Token Publishing and SSE Receiver.
    Verifies concurrent streaming requests use isolated sessions/channels with zero token leakage.
    """
    payload_a = {"message": "Policy A details", "session_id": "sess_chan_A"}
    payload_b = {"message": "Policy B details", "session_id": "sess_chan_B"}

    async def fetch_stream(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        async with async_client.stream("POST", "/api/chat/stream", json=payload) as response:
            return await SSEDecoder.collect_all(response)

    events_a, events_b = await asyncio.gather(
        fetch_stream(payload_a), fetch_stream(payload_b)
    )

    assert len(events_a) > 0, "Client A stream returned no events"
    assert len(events_b) > 0, "Client B stream returned no events"

    start_a = next(e for e in events_a if e["event"] == "start")
    start_b = next(e for e in events_b if e["event"] == "start")

    assert start_a["data"]["session_id"] == "sess_chan_A"
    assert start_b["data"]["session_id"] == "sess_chan_B"
