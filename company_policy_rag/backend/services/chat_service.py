from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from backend.models.api_dto import ChatRequest, ChatResponse
from backend.models.rag import RAGResponse
from backend.rag.pipeline import RAGPipeline
from backend.services.telemetry_service import TelemetryService
from backend.utils.logging import logger


class ChatService:
    """
    Service orchestrating RAG chat queries, multi-session management, and
    structured Server-Sent Events (SSE) streaming for sub-1s TTFT response delivery.
    """

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        telemetry_service: TelemetryService,
    ) -> None:
        self.pipeline = rag_pipeline
        self.telemetry_service = telemetry_service
        self._sessions: dict[str, list[dict[str, Any]]] = {}

    def set_active_model(self, model: str) -> str:
        """Return a safe guarantee that the backend pipeline is configured for the requested model."""
        if not model or not str(model).strip():
            raise ValueError("Model selection cannot be empty.")
        return self.pipeline.set_active_model(str(model).strip())

    def get_active_model(self) -> str:
        return self.pipeline.get_active_model()

    def execute_query(self, request: ChatRequest) -> ChatResponse:
        """Execute synchronous RAG query and record telemetry trace."""
        if not request.message or not request.message.strip():
            raise ValueError("Chat query message cannot be empty.")

        session_id = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        t_start = time.perf_counter()

        # Retrieve session conversation history before execution
        history = self._sessions.get(session_id, [])

        # Run pipeline
        rag_res: RAGResponse = self.pipeline.query(
            user_query=request.message,
            filters=request.filters,
            history=history,
            model=request.model,
        )
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

        # Record in Telemetry
        self.telemetry_service.record_from_rag_response(rag_res)

        # Record in session history
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(
            {
                "message_id": message_id,
                "role": "user",
                "content": request.message,
                "timestamp": time.time(),
            }
        )
        self._sessions[session_id].append(
            {
                "message_id": rag_res.id,
                "role": "assistant",
                "content": rag_res.answer,
                "timestamp": time.time(),
            }
        )

        return ChatResponse(
            id=rag_res.id,
            message_id=message_id,
            session_id=session_id,
            query=request.message,
            answer=rag_res.answer,
            citations=rag_res.citations,
            latency_ms=elapsed_ms,
            metrics={
                "candidate_count": rag_res.trace.retrieved_candidate_count if rag_res.trace else 0,
                "context_count": len(rag_res.context_chunks),
                "execution_time_ms": elapsed_ms,
            },
            trace=rag_res.trace,
            low_confidence=False,
            grounding_mode=request.grounding_mode or "balanced",
            model=rag_res.model or request.model or "qwen2.5:7b",
            token_usage=rag_res.token_usage or {},
        )

    async def stream_query(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """
        Stream RAG response via Server-Sent Events (SSE) with real-time LLM token streaming.
        Uses pipeline.stream_query() generator to get genuine tokens from Ollama as they are generated.
        Emits events: start, retrieval, chunk, citation, trace, done, error.
        """
        t_start = time.perf_counter()
        session_id = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        response_id = f"resp_{uuid.uuid4().hex[:12]}"
        model_name = request.model or "qwen2.5:7b"
        history = self._sessions.get(session_id, [])

        try:
            if not request.message or not request.message.strip():
                err_data = json.dumps({"detail": "Chat query message cannot be empty.", "status": 400})
                yield f"event: error\ndata: {err_data}\n\n"
                return

            # 1. Immediate start event (< 50ms TTFT header)
            start_payload = {
                "id": response_id,
                "message_id": message_id,
                "session_id": session_id,
                "query": request.message,
                "model": model_name,
                "status": "processing",
            }
            yield f"event: start\ndata: {json.dumps(start_payload)}\n\n"

            # 2. Run the streaming pipeline in a background thread, forward events via queue
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def _run_stream_pipeline():
                """Runs in a thread — iterates the synchronous generator and pushes events to the async queue safely."""
                try:
                    for event in self.pipeline.stream_query(
                        user_query=request.message,
                        filters=request.filters,
                        history=history,
                        model=request.model,
                    ):
                        loop.call_soon_threadsafe(queue.put_nowait, event)
                    loop.call_soon_threadsafe(queue.put_nowait, {"type": "__end__"})
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, {"type": "__error__", "detail": str(exc)})

            # Launch the pipeline in a thread
            loop.run_in_executor(None, _run_stream_pipeline)

            # Small delay to let the thread start
            await asyncio.sleep(0.01)

            ttft_recorded = False
            ttft_ms = 0.0
            full_answer = ""
            token_index = 0

            # 3. Consume events from the queue and yield SSE events
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=600.0)
                except TimeoutError:
                    err_data = json.dumps({"detail": "LLM generation timed out after 10 minutes.", "status": 504})
                    yield f"event: error\ndata: {err_data}\n\n"
                    return

                if event["type"] == "__end__":
                    break

                if event["type"] == "__error__":
                    logger.exception("Pipeline stream error: %s", event["detail"])
                    err_data = json.dumps({"detail": event["detail"], "status": 500})
                    yield f"event: error\ndata: {err_data}\n\n"
                    return

                if event["type"] == "retrieval_done":
                    # Signal frontend that retrieval is complete, LLM generation starting
                    retrieval_payload = {
                        "id": response_id,
                        "status": "generating",
                        "stage_timings": event.get("stage_timings", {}),
                        "candidate_count": event.get("candidate_count", 0),
                        "context_count": event.get("context_count", 0),
                        "cache_hit": event.get("cache_hit", False),
                    }
                    yield f"event: retrieval\ndata: {json.dumps(retrieval_payload)}\n\n"

                elif event["type"] == "token":
                    token_text = event["content"]
                    full_answer += token_text

                    if not ttft_recorded:
                        ttft_ms = round((time.perf_counter() - t_start) * 1000, 2)
                        ttft_recorded = True

                    chunk_payload = {
                        "id": response_id,
                        "content": token_text,
                        "index": token_index,
                    }
                    yield f"event: chunk\ndata: {json.dumps(chunk_payload)}\n\n"
                    token_index += 1

                elif event["type"] == "done":
                    total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

                    # Stream Citations
                    citations = event.get("citations", [])
                    citations_payload = {
                        "id": response_id,
                        "citations": [c.model_dump() for c in citations],
                    }
                    yield f"event: citation\ndata: {json.dumps(citations_payload)}\n\n"

                    # Record & Stream Telemetry Trace
                    trace = event.get("trace")
                    rag_response = RAGResponse(
                        id=response_id,
                        query=request.message,
                        answer=full_answer,
                        citations=citations,
                        context_chunks=event.get("context_chunks", []),
                        trace=trace,
                        model=model_name,
                        token_usage=event.get("token_usage", {}),
                    )
                    trace_summary = self.telemetry_service.record_from_rag_response(
                        rag_response, ttft_ms=ttft_ms,
                    )
                    trace_payload = {
                        "id": response_id,
                        "trace": trace_summary.model_dump(),
                    }
                    yield f"event: trace\ndata: {json.dumps(trace_payload)}\n\n"

                    # Done Event - shape the UI can materialize into the assistant message
                    citations_payload = [c.model_dump() for c in citations]
                    done_payload = {
                        "id": response_id,
                        "answer": full_answer,
                        "status": "completed",
                        "total_latency_ms": total_latency_ms,
                        "ttft_ms": ttft_ms,
                        "total_tokens": event.get("token_usage", {}).get("completion_tokens", 0),
                        "citations": citations_payload,
                        "thinking": None,
                        "timing": {
                            "ttft_ms": ttft_ms,
                            "total_latency_ms": total_latency_ms,
                        },
                        "retrieval_trace": trace_summary.model_dump() if trace_summary else None,
                        "message_id": message_id,
                        "low_confidence": False,
                        "grounding_mode": request.grounding_mode or "balanced",
                    }
                    yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

                    # Record session history
                    if session_id not in self._sessions:
                        self._sessions[session_id] = []
                    self._sessions[session_id].append(
                        {"message_id": message_id, "role": "user", "content": request.message, "timestamp": time.time()}
                    )
                    self._sessions[session_id].append(
                        {"message_id": response_id, "role": "assistant", "content": full_answer, "timestamp": time.time()}
                    )

        except Exception as exc:
            logger.exception("Error during chat stream query: %s", exc)
            err_payload = {"detail": str(exc), "status": 500}
            yield f"event: error\ndata: {json.dumps(err_payload)}\n\n"

