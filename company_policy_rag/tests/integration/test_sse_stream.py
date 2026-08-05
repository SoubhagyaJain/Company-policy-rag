from __future__ import annotations

import json
import time
import pytest
from fastapi.testclient import TestClient

from backend.api.dependencies import reset_dependencies
from backend.api.main import app


@pytest.fixture(autouse=True)
def cleanup_deps():
    reset_dependencies()
    yield
    reset_dependencies()


def parse_sse_events(raw_text: str):
    """Utility helper to parse raw SSE stream text into structured (event_name, data_dict) tuples."""
    events = []
    lines = raw_text.split("\n")
    current_event = None
    current_data = []

    for line in lines:
        line = line.strip()
        if line.startswith("event:"):
            current_event = line.replace("event:", "").strip()
        elif line.startswith("data:"):
            current_data.append(line.replace("data:", "").strip())
        elif line == "":
            if current_event and current_data:
                full_data_str = "\n".join(current_data)
                try:
                    parsed_json = json.loads(full_data_str)
                except Exception:
                    parsed_json = full_data_str
                events.append((current_event, parsed_json))
                current_event = None
                current_data = []

    return events


def test_sse_chat_stream_events_and_sub1s_ttft():
    client = TestClient(app)
    payload = {
        "message": "What is the employee travel reimbursement process?",
        "session_id": "sse_integration_test_sess",
    }

    t0 = time.perf_counter()
    with client.stream("POST", "/api/chat/stream", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        first_chunk_time = None
        start_event_received = False
        received_events = []

        # Read line by line from stream
        current_event = None
        current_data = []

        for line in response.iter_lines():
            if not line:
                if current_event and current_data:
                    data_str = "\n".join(current_data)
                    data_obj = json.loads(data_str)
                    received_events.append((current_event, data_obj))

                    if current_event == "start":
                        start_event_received = True

                    if current_event == "chunk" and first_chunk_time is None:
                        first_chunk_time = time.perf_counter()

                    current_event = None
                    current_data = []
                continue

            line_str = line if isinstance(line, str) else line.decode("utf-8")
            if line_str.startswith("event:"):
                current_event = line_str[6:].strip()
            elif line_str.startswith("data:"):
                current_data.append(line_str[5:].strip())

        t_total = (time.perf_counter() - t0) * 1000

        # Assertions
        assert start_event_received, "Expected 'start' SSE event was not received"
        event_names = [e[0] for e in received_events]

        assert "start" in event_names
        assert "chunk" in event_names
        assert "citation" in event_names
        assert "trace" in event_names
        assert "done" in event_names

        # Sub-1s TTFT check
        assert first_chunk_time is not None, "No text token chunks were received in stream"
        ttft_latency_ms = (first_chunk_time - t0) * 1000
        assert ttft_latency_ms < 1000.0, f"TTFT exceeded 1.0s limit: took {ttft_latency_ms:.2f}ms"

        # Verify done payload
        done_data = [e[1] for e in received_events if e[0] == "done"][0]
        assert done_data["status"] == "completed"
