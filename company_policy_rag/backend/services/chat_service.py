from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

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
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}

    def execute_query(self, request: ChatRequest) -> ChatResponse:
        """Execute synchronous RAG query and record telemetry trace."""
        if not request.message or not request.message.strip():
            raise ValueError("Chat query message cannot be empty.")

        session_id = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        t_start = time.perf_counter()

        # Run pipeline
        rag_res: RAGResponse = self.pipeline.query(
            user_query=request.message,
            filters=request.filters,
        )
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

        # Record in Telemetry
        trace_summary = self.telemetry_service.record_from_rag_response(rag_res)

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
            model=request.model or "qwen2.5:7b",
            token_usage=rag_res.token_usage or {},
        )

    async def stream_query(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """
        Stream RAG response via Server-Sent Events (SSE) with sub-1s TTFT.
        Emits events: start, chunk, citation, trace, done, error.
        """
        t_start = time.perf_counter()
        session_id = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        response_id = f"resp_{uuid.uuid4().hex[:12]}"
        model_name = request.model or "qwen2.5:7b"

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

            # Offload synchronous RAG pipeline execution to thread pool
            rag_res: RAGResponse = await asyncio.to_thread(
                self.pipeline.query,
                request.message,
                request.filters,
            )

            ttft_recorded = False
            ttft_ms = 0.0

            # 2. Stream response chunks
            full_text = rag_res.answer or ""

            # Tokenize or split words/tokens for streaming playback
            tokens = full_text.split(" ")
            for idx, token in enumerate(tokens):
                chunk_str = token + (" " if idx < len(tokens) - 1 else "")
                if not ttft_recorded:
                    ttft_ms = round((time.perf_counter() - t_start) * 1000, 2)
                    ttft_recorded = True

                chunk_payload = {
                    "id": response_id,
                    "content": chunk_str,
                    "index": idx,
                }
                yield f"event: chunk\ndata: {json.dumps(chunk_payload)}\n\n"
                # Brief async sleep to allow event loop flush without blocking sub-1s TTFT
                await asyncio.sleep(0.005)

            total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

            # 3. Stream Citations
            citations_payload = {
                "id": response_id,
                "citations": [c.model_dump() for c in rag_res.citations],
            }
            yield f"event: citation\ndata: {json.dumps(citations_payload)}\n\n"

            # 4. Record & Stream Telemetry Trace
            trace_summary = self.telemetry_service.record_from_rag_response(
                rag_res,
                ttft_ms=ttft_ms,
            )
            trace_payload = {
                "id": response_id,
                "trace": trace_summary.model_dump(),
            }
            yield f"event: trace\ndata: {json.dumps(trace_payload)}\n\n"

            # 5. Stream Done Event
            done_payload = {
                "id": response_id,
                "status": "completed",
                "total_latency_ms": total_latency_ms,
                "ttft_ms": ttft_ms,
                "total_tokens": rag_res.token_usage.get("total_tokens", 0) if rag_res.token_usage else 0,
            }
            yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

            # Record session history
            if session_id not in self._sessions:
                self._sessions[session_id] = []
            self._sessions[session_id].append(
                {"message_id": message_id, "role": "user", "content": request.message, "timestamp": time.time()}
            )
            self._sessions[session_id].append(
                {"message_id": response_id, "role": "assistant", "content": full_text, "timestamp": time.time()}
            )

        except Exception as exc:
            logger.exception("Error during chat stream query: %s", exc)
            err_payload = {"detail": str(exc), "status": 500}
            yield f"event: error\ndata: {json.dumps(err_payload)}\n\n"
