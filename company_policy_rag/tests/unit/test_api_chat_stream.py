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


def test_post_chat_stream_success():
    client = TestClient(app)
    payload = {
        "message": "What is the employee resignation notice policy?",
        "session_id": "stream_test_session_1",
    }

    t0 = time.perf_counter()
    with client.stream("POST", "/api/chat/stream", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        assert response.headers.get("x-accel-buffering") == "no"
        assert "no-cache" in response.headers.get("cache-control", "")

        events = []
        current_event = None
        current_data = []

        for line in response.iter_lines():
            if not line:
                if current_event and current_data:
                    data_str = "\n".join(current_data)
                    data_obj = json.loads(data_str)
                    events.append((current_event, data_obj))
                    current_event = None
                    current_data = []
                continue

            line_str = line if isinstance(line, str) else line.decode("utf-8")
            if line_str.startswith("event:"):
                current_event = line_str[6:].strip()
            elif line_str.startswith("data:"):
                current_data.append(line_str[5:].strip())

        event_names = [e[0] for e in events]
        assert "start" in event_names
        assert "chunk" in event_names
        assert "citation" in event_names
        assert "trace" in event_names
        assert "done" in event_names

        # Check start payload
        start_obj = [e[1] for e in events if e[0] == "start"][0]
        assert start_obj["session_id"] == "stream_test_session_1"
        assert start_obj["query"] == "What is the employee resignation notice policy?"

        # Check done payload
        done_obj = [e[1] for e in events if e[0] == "done"][0]
        assert done_obj["status"] == "completed"
        assert "total_latency_ms" in done_obj


def test_post_chat_stream_empty_message():
    client = TestClient(app)
    payload = {"message": "   "}
    response = client.post("/api/chat/stream", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_post_chat_stream_auto_generates_session_id():
    client = TestClient(app)
    payload = {"message": "Tell me about remote work policies."}

    with client.stream("POST", "/api/chat/stream", json=payload) as response:
        assert response.status_code == 200
        events = []
        current_event = None
        current_data = []

        for line in response.iter_lines():
            if not line:
                if current_event and current_data:
                    data_str = "\n".join(current_data)
                    data_obj = json.loads(data_str)
                    events.append((current_event, data_obj))
                    current_event = None
                    current_data = []
                continue

            line_str = line if isinstance(line, str) else line.decode("utf-8")
            if line_str.startswith("event:"):
                current_event = line_str[6:].strip()
            elif line_str.startswith("data:"):
                current_data.append(line_str[5:].strip())

        start_obj = [e[1] for e in events if e[0] == "start"][0]
        assert start_obj["session_id"].startswith("sess_")
