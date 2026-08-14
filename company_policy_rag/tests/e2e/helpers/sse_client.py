"""
Server-Sent Events (SSE) Client Parser Helper for E2E Tests.
Parses text/event-stream responses into structured event objects and tuples.
"""
from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, List, Tuple, Union
import httpx


class SSEDecoder:
    """Parses text/event-stream chunks and responses into structured event objects."""

    @staticmethod
    def parse_raw_text(raw_text: str) -> List[Dict[str, Any]]:
        """
        Parse raw SSE format text into a list of event dictionaries.
        Each event dict has keys: 'event' (str) and 'data' (dict or str).
        """
        events: List[Dict[str, Any]] = []
        lines = raw_text.split("\n")
        current_event = "message"
        current_data: List[str] = []

        for line in lines:
            line_str = line.strip()
            if not line_str:
                if current_data:
                    full_data_str = "\n".join(current_data)
                    try:
                        parsed_data = json.loads(full_data_str)
                    except (json.JSONDecodeError, TypeError):
                        parsed_data = full_data_str
                    events.append({"event": current_event, "data": parsed_data})
                    current_event = "message"
                    current_data = []
                continue

            if line_str.startswith("event:"):
                current_event = line_str[6:].strip()
            elif line_str.startswith("data:"):
                current_data.append(line_str[5:].strip())

        if current_data:
            full_data_str = "\n".join(current_data)
            try:
                parsed_data = json.loads(full_data_str)
            except (json.JSONDecodeError, TypeError):
                parsed_data = full_data_str
            events.append({"event": current_event, "data": parsed_data})

        return events

    @classmethod
    async def parse_response(
        cls, response: httpx.Response
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Asynchronously yield event objects from an HTTPX streaming response.
        """
        current_event = "message"
        current_data: List[str] = []

        async for line in response.aiter_lines():
            line_str = line.strip() if isinstance(line, str) else line.decode("utf-8").strip()
            if not line_str:
                if current_data:
                    full_data_str = "\n".join(current_data)
                    try:
                        parsed_data = json.loads(full_data_str)
                    except (json.JSONDecodeError, TypeError):
                        parsed_data = full_data_str
                    yield {"event": current_event, "data": parsed_data}
                    current_event = "message"
                    current_data = []
                continue

            if line_str.startswith("event:"):
                current_event = line_str[6:].strip()
            elif line_str.startswith("data:"):
                current_data.append(line_str[5:].strip())

        if current_data:
            full_data_str = "\n".join(current_data)
            try:
                parsed_data = json.loads(full_data_str)
            except (json.JSONDecodeError, TypeError):
                parsed_data = full_data_str
            yield {"event": current_event, "data": parsed_data}

    @classmethod
    async def collect_all(cls, response: httpx.Response) -> List[Dict[str, Any]]:
        """
        Collect all SSE events from a streaming response into a list.
        """
        events = []
        async for evt in cls.parse_response(response):
            events.append(evt)
        return events


def parse_sse_events(raw_text: str) -> List[Tuple[str, Union[Dict[str, Any], str]]]:
    """
    Utility helper function to parse raw SSE stream text into structured (event_name, data) tuples.
    Used for spec and schema validation across test modules.
    """
    decoded = SSEDecoder.parse_raw_text(raw_text)
    return [(item["event"], item["data"]) for item in decoded]
