from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from cachetools import TTLCache

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc

from backend.models.api_dto import ChatRequest, ChatResponse
from backend.models.conversation import (
    AnswerMode,
    ConversationEvidenceContext,
    ConversationRAGState,
    ConversationStateManager,
    ConversationTurn,
    ExpansionPlan,
    FollowUpResolution,
)
from backend.models.rag import EvidenceStatus, RAGResponse
from backend.models.telemetry_models import SeverityLevel
from backend.rag.pipeline import RAGPipeline
from backend.services.telemetry_service import TelemetryService
from backend.utils.logging import logger


def _safe_evidence_status(val: Any) -> EvidenceStatus:
    if isinstance(val, EvidenceStatus):
        return val
    raw = str(getattr(val, "value", val) or "").strip().upper()
    if raw in ("DIRECT", "SUFFICIENT", "SUFFICIENT_CONTEXT"):
        return EvidenceStatus.DIRECT
    elif raw in ("PARTIAL", "PARTIAL_CONTEXT"):
        return EvidenceStatus.PARTIAL
    elif raw in ("RELATED", "RELATED_CONTEXT"):
        return EvidenceStatus.RELATED
    elif raw in ("MISSING", "INSUFFICIENT", "NO_EVIDENCE"):
        return EvidenceStatus.MISSING
    try:
        return EvidenceStatus(raw)
    except Exception:
        return EvidenceStatus.DIRECT


def _safe_answer_mode(val: Any) -> AnswerMode:
    if isinstance(val, AnswerMode):
        return val
    raw = str(getattr(val, "value", val) or "").strip().upper()
    try:
        return AnswerMode(raw)
    except Exception:
        return AnswerMode.DIRECT


def _extract_visual_asset_ids(chunks: list[Any], citations: list[Any] | None = None) -> list[str]:
    v_ids: list[str] = []
    seen: set[str] = set()
    for sc in (chunks or []):
        c = getattr(sc, "chunk", sc)
        if hasattr(c, "metadata"):
            meta = c.metadata
            if getattr(meta, "visual_asset_ids", None):
                for v in meta.visual_asset_ids:
                    if v and str(v) not in seen:
                        seen.add(str(v))
                        v_ids.append(str(v))
            if getattr(meta, "image_assets", None):
                for ast in (meta.image_assets or []):
                    aid = str(ast.get("asset_id") or ast.get("asset_url") or ast) if isinstance(ast, dict) else str(ast)
                    if aid and aid not in seen:
                        seen.add(aid)
                        v_ids.append(aid)
            if getattr(meta, "extra", None) and isinstance(meta.extra, dict):
                aid = meta.extra.get("asset_id")
                if aid and str(aid) not in seen:
                    seen.add(str(aid))
                    v_ids.append(str(aid))
    for cit in (citations or []):
        aid = getattr(cit, "visual_asset_id", None)
        if aid and str(aid) not in seen:
            seen.add(str(aid))
            v_ids.append(str(aid))
    return v_ids



class ChatService:
    """
    Service orchestrating RAG chat queries, multi-session management, reference resolution,
    and structured Server-Sent Events (SSE) streaming with sub-1s TTFT and end-to-end telemetry.
    """

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        telemetry_service: TelemetryService,
        state_manager: ConversationStateManager | None = None,
    ) -> None:
        self.pipeline = rag_pipeline
        self.telemetry_service = telemetry_service
        self.state_manager = state_manager or ConversationStateManager()

        # Thread-safe TTL cache: Max 1000 sessions, 24-hour expiration
        self._sessions = TTLCache(maxsize=1000, ttl=86400)
        self._session_metadata: dict[str, dict[str, Any]] = {}
        self._session_lock = threading.Lock()

    def get_conversation_state(self, session_id: str) -> ConversationRAGState:
        """Retrieve isolated state for the given session_id."""
        return self.state_manager.get_state(session_id)

    def delete_session(self, session_id: str) -> None:
        """Safely evict a session from the LRU/TTL cache and state manager."""
        with self._session_lock:
            self._sessions.pop(session_id, None)
            self._session_metadata.pop(session_id, None)
            self.state_manager.delete_state(session_id)

    def clear_session(self, session_id: str) -> None:
        """Clear conversation messages for a specific session."""
        with self._session_lock:
            self._sessions.pop(session_id, None)
            self.state_manager.delete_state(session_id)

    def clear_all_sessions(self) -> None:
        """Purge all cached conversation sessions from memory."""
        with self._session_lock:
            self._sessions.clear()
            self._session_metadata.clear()
            self.state_manager.clear_all()

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
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        t_start = time.perf_counter()

        with self._session_lock:
            history = self._sessions.get(session_id, [])
            session_meta = self._session_metadata.setdefault(session_id, {})
            if request.active_document_id:
                session_meta["active_document_id"] = request.active_document_id
            if request.active_document_name:
                session_meta["active_document_name"] = request.active_document_name
            if request.selected_document_ids:
                session_meta["selected_document_ids"] = request.selected_document_ids
            if request.document_scope:
                session_meta["document_scope"] = request.document_scope

            active_doc_id = request.active_document_id or session_meta.get("active_document_id")
            active_doc_name = request.active_document_name or session_meta.get("active_document_name")
            selected_doc_ids = request.selected_document_ids or session_meta.get("selected_document_ids")
            doc_scope = request.document_scope or session_meta.get("document_scope")

        # Get current isolated conversation state
        conv_state = self.state_manager.get_state(session_id)

        # Run pipeline with document-aware scope resolution and conversation state
        thinking_level = request.thinking_detail_level
        if request.filters and ("thinking" in request.filters or "thinking_detail_level" in request.filters):
            filter_val = request.filters.get("thinking") or request.filters.get("thinking_detail_level")
            if filter_val:
                thinking_level = str(filter_val)
        if not thinking_level:
            thinking_level = "standard"
        try:
            rag_res: RAGResponse = self.pipeline.query(
                user_query=request.message,
                filters=request.filters,
                history=history,
                model=request.model,
                active_document_id=active_doc_id,
                active_document_name=active_doc_name,
                selected_document_ids=selected_doc_ids,
                document_scope=doc_scope,
                conversation_state=conv_state,
                thinking_detail_level=thinking_level,
            )
        except TypeError:
            rag_res = self.pipeline.query(
                user_query=request.message,
                filters=request.filters,
                history=history,
                model=request.model,
                active_document_id=active_doc_id,
                active_document_name=active_doc_name,
                selected_document_ids=selected_doc_ids,
                document_scope=doc_scope,
            )

        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

        # Track memory reference resolution
        if history and rag_res.trace and rag_res.trace.rewritten_query and rag_res.trace.rewritten_query != request.message:
            self.telemetry_service.record_memory_event(
                session_id=session_id,
                user_query=request.message,
                resolved_query=rag_res.trace.rewritten_query,
                referent_found=rag_res.trace.rewritten_query,
                resolution_status="SUCCESS",
                latency_ms=rag_res.trace.stage_timings_ms.get("query_rewrite", 3.0),
            )

        self.telemetry_service.record_from_rag_response(
            rag_res,
            ttft_ms=None,
            request_id=request_id,
            conversation_id=session_id,
            document_id=active_doc_id,
        )

        with self._session_lock:
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

        # Update and persist conversation state
        turn = ConversationTurn(
            turn_id=f"turn_{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            user_query=request.message,
            resolved_query=rag_res.trace.rewritten_query or request.message,
            is_followup=rag_res.trace.is_followup if rag_res.trace else False,
            topic_shift=rag_res.trace.topic_shift if rag_res.trace else False,
            intent=str(rag_res.trace.query_type or "factual") if rag_res.trace else "factual",
            answer_mode=str(rag_res.trace.answer_mode or "DIRECT") if rag_res.trace else "DIRECT",
            evidence_status=str(rag_res.trace.evidence_status or "DIRECT") if rag_res.trace else "DIRECT",

            active_topic=rag_res.trace.active_topic if (rag_res.trace and rag_res.trace.active_topic) else conv_state.active_topic,
            active_entities=rag_res.trace.active_entities if (rag_res.trace and rag_res.trace.active_entities) else conv_state.active_entities,
            retrieved_chunks=rag_res.context_chunks,
            visual_evidence=[
                c for c in rag_res.context_chunks
                if "diagram" in str(c.chunk.metadata.content_type).lower()
                or c.chunk.metadata.extra.get("is_visual_extraction")
                or c.chunk.metadata.image_assets
            ],
            citations=rag_res.citations,
            answer=rag_res.answer,
        )
        # Build ConversationEvidenceContext
        evidence_ctx = ConversationEvidenceContext(
            conversation_id=session_id,
            turn_id=turn.turn_id,
            query=request.message,
            normalized_subjects=[conv_state.active_topic] if conv_state.active_topic else [],
            verified_chunk_ids=[c.chunk.id for c in rag_res.context_chunks],
            verified_citations=rag_res.citations,
            evidence_status=_safe_evidence_status(rag_res.trace.evidence_status if rag_res.trace else "DIRECT"),
            visual_asset_ids=_extract_visual_asset_ids(rag_res.context_chunks, rag_res.citations),
            source_pages=[
                c.chunk.metadata.page_number
                for c in rag_res.context_chunks
                if getattr(c.chunk.metadata, "page_number", None) is not None
            ],
            document_ids=list({
                c.chunk.metadata.document_id
                for c in rag_res.context_chunks
                if getattr(c.chunk.metadata, "document_id", None)
            }),
            answer_mode=_safe_answer_mode(rag_res.trace.answer_mode if rag_res.trace else AnswerMode.DIRECT),
            timestamp=time.time(),
        )
        turn.evidence_context = evidence_ctx
        conv_state.evidence_contexts.append(evidence_ctx)
        conv_state.turns.append(turn)
        conv_state.last_user_query = request.message
        conv_state.last_resolved_query = rag_res.trace.rewritten_query or request.message
        conv_state.last_answer = rag_res.answer
        if rag_res.trace and rag_res.trace.active_topic:
            conv_state.active_topic = rag_res.trace.active_topic
        if rag_res.trace and rag_res.trace.active_entities:
            conv_state.active_entities = rag_res.trace.active_entities
        if rag_res.trace and rag_res.trace.query_type:
            conv_state.previous_intent = rag_res.trace.query_type
        if rag_res.trace and rag_res.trace.answer_mode:
            conv_state.previous_answer_mode = rag_res.trace.answer_mode
        if rag_res.trace and rag_res.trace.evidence_status:
            conv_state.previous_evidence_status = rag_res.trace.evidence_status
        conv_state.previous_retrieved_chunks = rag_res.context_chunks
        conv_state.previous_visual_evidence = [
            c for c in rag_res.context_chunks
            if "diagram" in str(c.chunk.metadata.content_type).lower()
            or c.chunk.metadata.extra.get("is_visual_extraction")
            or c.chunk.metadata.image_assets
        ]
        conv_state.previous_citations = rag_res.citations
        conv_state.updated_at = time.time()
        self.state_manager.update_state(session_id, conv_state)

        low_confidence = (
            (not rag_res.trace.faithfulness_passed)
            if (rag_res.trace and rag_res.trace.faithfulness_checked)
            else False
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
                "request_id": request_id,
            },
            trace=rag_res.trace,
            low_confidence=low_confidence,
            grounding_mode=request.grounding_mode or "balanced",
            model=rag_res.model or request.model or self.get_active_model(),
            document_scope=rag_res.trace.query_scope if rag_res.trace else doc_scope,
            active_document_id=rag_res.trace.active_document_id if rag_res.trace else active_doc_id,
            active_document_name=rag_res.trace.active_document_name if rag_res.trace else active_doc_name,
            token_usage=rag_res.token_usage or {},
            query_type=rag_res.trace.query_type if rag_res.trace else None,
            routing_confidence=rag_res.trace.routing_confidence if rag_res.trace else None,
            inferred_filters=rag_res.trace.inferred_filters if rag_res.trace else {},
            verification=rag_res.trace.verification_report if rag_res.trace else None,
            reasoning_summary=(
                rag_res.trace.reasoning_summary.model_dump()
                if (rag_res.trace and rag_res.trace.reasoning_summary and hasattr(rag_res.trace.reasoning_summary, "model_dump"))
                else (rag_res.trace.reasoning_summary if rag_res.trace else None)
            ),
            thinking_events=(
                [e.model_dump() if hasattr(e, "model_dump") else e for e in rag_res.trace.thinking_events]
                if (rag_res.trace and rag_res.trace.thinking_events)
                else []
            ),
        )

    async def stream_query(
        self, request: ChatRequest, cancel_token: asyncio.Event | None = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream RAG response via Server-Sent Events (SSE) with real-time LLM token streaming.
        Propagates request_id throughout start, chunk, citation, trace, and done payloads.
        """
        t_start = time.perf_counter()
        session_id = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        response_id = f"resp_{uuid.uuid4().hex[:12]}"
        raw_req_model = (request.model or "").strip()
        if not raw_req_model or raw_req_model.lower() in ("default", "fastapi rag", "none"):
            model_name = self.get_active_model() or "qwen2.5:7b"
        else:
            model_name = raw_req_model

        with self._session_lock:
            history = self._sessions.get(session_id, [])
            session_meta = self._session_metadata.setdefault(session_id, {})
            if request.active_document_id:
                session_meta["active_document_id"] = request.active_document_id
            if request.active_document_name:
                session_meta["active_document_name"] = request.active_document_name
            if request.selected_document_ids:
                session_meta["selected_document_ids"] = request.selected_document_ids
            if request.document_scope:
                session_meta["document_scope"] = request.document_scope

            active_doc_id = request.active_document_id or session_meta.get("active_document_id")
            active_doc_name = request.active_document_name or session_meta.get("active_document_name")
            selected_doc_ids = request.selected_document_ids or session_meta.get("selected_document_ids")
            doc_scope = request.document_scope or session_meta.get("document_scope")

        # Get conversation state from manager
        conv_state = self.state_manager.get_state(session_id)

        try:
            if not request.message or not request.message.strip():
                err_data = json.dumps(
                    {"detail": "Chat query message cannot be empty.", "status": 400}
                )
                yield f"event: error\ndata: {err_data}\n\n"
                return

            # 1. Immediate start event with request_id
            start_payload = {
                "id": response_id,
                "request_id": request_id,
                "message_id": message_id,
                "session_id": session_id,
                "query": request.message,
                "model": model_name,
                "document_scope": doc_scope,
                "active_document_id": active_doc_id,
                "active_document_name": active_doc_name,
                "status": "processing",
                "timestamp": datetime.now(UTC).isoformat(),
            }
            yield f"event: start\ndata: {json.dumps(start_payload)}\n\n"

            ttft_recorded = False
            ttft_ms = 0.0
            full_answer = ""
            token_index = 0
            final_rag_trace: Any = None
            final_citations: list[Any] = []
            final_context_chunks: list[Any] = []

            thinking_level = request.thinking_detail_level
            if request.filters and ("thinking" in request.filters or "thinking_detail_level" in request.filters):
                filter_val = request.filters.get("thinking") or request.filters.get("thinking_detail_level")
                if filter_val:
                    thinking_level = str(filter_val)
            if not thinking_level:
                thinking_level = "standard"
            try:
                pipeline_stream = self.pipeline.stream_query(
                    user_query=request.message,
                    filters=request.filters,
                    history=history,
                    model=request.model,
                    active_document_id=active_doc_id,
                    active_document_name=active_doc_name,
                    selected_document_ids=selected_doc_ids,
                    document_scope=doc_scope,
                    conversation_state=conv_state,
                    thinking_detail_level=thinking_level,
                    cancel_token=cancel_token,
                )
            except TypeError:
                pipeline_stream = self.pipeline.stream_query(
                    user_query=request.message,
                    filters=request.filters,
                    history=history,
                    model=request.model,
                    active_document_id=active_doc_id,
                    active_document_name=active_doc_name,
                    selected_document_ids=selected_doc_ids,
                    document_scope=doc_scope,
                    cancel_token=cancel_token,
                )

            async for event in pipeline_stream:
                if cancel_token and cancel_token.is_set():
                    break

                if event["type"] == "thinking":
                    thk_ev = event["event"]
                    thk_dict = thk_ev.model_dump() if hasattr(thk_ev, "model_dump") else thk_ev
                    yield f"event: thinking\ndata: {json.dumps(thk_dict)}\n\n"

                elif event["type"] == "retrieval_done":
                    retrieval_payload = {
                        "id": response_id,
                        "request_id": request_id,
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
                        "request_id": request_id,
                        "content": token_text,
                        "index": token_index,
                    }
                    yield f"event: chunk\ndata: {json.dumps(chunk_payload)}\n\n"
                    token_index += 1

                elif event["type"] == "done":
                    total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

                    citations = event.get("citations", [])
                    final_citations = citations
                    final_context_chunks = event.get("context_chunks", [])
                    citations_payload = {
                        "id": response_id,
                        "request_id": request_id,
                        "citations": [c.model_dump() if hasattr(c, "model_dump") else c for c in citations],
                    }
                    yield f"event: citation\ndata: {json.dumps(citations_payload)}\n\n"

                    trace = event.get("trace")
                    final_rag_trace = trace
                    rag_response = RAGResponse(
                        id=response_id,
                        query=request.message,
                        answer=full_answer,
                        citations=citations,
                        context_chunks=event.get("context_chunks", []),
                        trace=trace if trace else RAGTrace(query=request.message),
                        model=model_name,
                        token_usage=event.get("token_usage", {}),
                    )

                    # Track memory reference resolution
                    if history and trace and trace.rewritten_query and trace.rewritten_query != request.message and self.telemetry_service:
                        self.telemetry_service.record_memory_event(
                            session_id=session_id,
                            user_query=request.message,
                            resolved_query=trace.rewritten_query,
                            referent_found=trace.rewritten_query,
                            resolution_status="SUCCESS",
                            latency_ms=trace.stage_timings_ms.get("query_rewrite", 3.0),
                        )

                    trace_dict = {}
                    if self.telemetry_service:
                        try:
                            trace_summary = self.telemetry_service.record_from_rag_response(
                                rag_response,
                                ttft_ms=ttft_ms,
                                request_id=request_id,
                                conversation_id=session_id,
                                document_id=active_doc_id,
                            )
                            if hasattr(trace_summary, "model_dump"):
                                trace_dict = trace_summary.model_dump()
                            elif isinstance(trace_summary, dict):
                                trace_dict = dict(trace_summary)
                        except Exception as trace_err:
                            logger.warning("Failed to record trace summary: %s", trace_err)
                    trace_dict["request_id"] = request_id

                    trace_payload = {
                        "id": response_id,
                        "request_id": request_id,
                        "trace": trace_dict,
                    }
                    try:
                        trace_json = json.dumps(trace_payload)
                    except (TypeError, ValueError):
                        trace_json = json.dumps(trace_payload, default=str)
                    yield f"event: trace\ndata: {trace_json}\n\n"

                    low_confidence = (
                        (not trace.faithfulness_passed)
                        if (trace and trace.faithfulness_checked)
                        else False
                    )
                    done_payload = {
                        "id": response_id,
                        "request_id": request_id,
                        "turn_id": message_id,
                        "answer": full_answer,
                        "status": "completed",
                        "total_latency_ms": total_latency_ms,
                        "ttft_ms": ttft_ms,
                        "total_tokens": event.get("token_usage", {}).get("completion_tokens", 0),
                        "citations": [c.model_dump() if hasattr(c, "model_dump") else c for c in citations],
                        "thinking": event.get("thinking_events") or (trace.thinking_events if trace else []),
                        "reasoning_summary": event.get("reasoning_summary") or (trace.reasoning_summary.model_dump() if (trace and hasattr(trace.reasoning_summary, "model_dump")) else getattr(trace, "reasoning_summary", None)),
                        "thinking_events": event.get("thinking_events") or (trace.thinking_events if trace else []),
                        "timing": {
                            "ttft_ms": ttft_ms,
                            "total_latency_ms": total_latency_ms,
                            "stage_timings": trace.stage_timings_ms if trace else {},
                        },
                        "low_confidence": low_confidence,
                        "model": model_name,
                        "document_scope": doc_scope,
                        "active_document_id": active_doc_id,
                        "active_document_name": active_doc_name,
                    }
                    try:
                        done_json = json.dumps(done_payload)
                    except (TypeError, ValueError):
                        done_json = json.dumps(done_payload, default=str)
                    yield f"event: done\ndata: {done_json}\n\n"

                elif event["type"] == "complete":
                    yield f"event: complete\ndata: {json.dumps(event.get('data', {}))}\n\n"

                elif event["type"] == "error":
                    yield f"event: error\ndata: {json.dumps({'detail': event.get('detail', 'Unknown error'), 'status': 500, 'request_id': request_id})}\n\n"

            # 4. Save turn to history and update conversation state
            with self._session_lock:
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
                        "message_id": response_id,
                        "role": "assistant",
                        "content": full_answer,
                        "timestamp": time.time(),
                    }
                )

            # Persist updated turn to ConversationStateManager
            safe_resolved = (final_rag_trace.rewritten_query or request.message) if final_rag_trace else request.message
            turn = ConversationTurn(
                turn_id=f"turn_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                user_query=request.message,
                resolved_query=safe_resolved,
                is_followup=final_rag_trace.is_followup if final_rag_trace else False,
                topic_shift=final_rag_trace.topic_shift if final_rag_trace else False,
                intent=str(final_rag_trace.query_type) if (final_rag_trace and final_rag_trace.query_type) else "factual",
                answer_mode=str(final_rag_trace.answer_mode) if (final_rag_trace and final_rag_trace.answer_mode) else "DIRECT",
                evidence_status=str(final_rag_trace.evidence_status) if (final_rag_trace and final_rag_trace.evidence_status) else "DIRECT",
                active_topic=final_rag_trace.active_topic if (final_rag_trace and final_rag_trace.active_topic) else conv_state.active_topic,
                active_entities=final_rag_trace.active_entities if (final_rag_trace and final_rag_trace.active_entities) else conv_state.active_entities,
                retrieved_chunks=final_context_chunks,
                visual_evidence=[
                    c for c in final_context_chunks
                    if hasattr(c, "chunk") and (
                        "diagram" in str(c.chunk.metadata.content_type).lower()
                        or c.chunk.metadata.extra.get("is_visual_extraction")
                        or c.chunk.metadata.image_assets
                    )
                ],
                citations=final_citations,
                answer=full_answer,
            )
            # Build ConversationEvidenceContext for streaming turn
            evidence_ctx = ConversationEvidenceContext(
                conversation_id=session_id,
                turn_id=turn.turn_id,
                query=request.message,
                normalized_subjects=[conv_state.active_topic] if conv_state.active_topic else [],
                verified_chunk_ids=[c.chunk.id for c in final_context_chunks if hasattr(c, "chunk")],
                verified_citations=final_citations,
                evidence_status=_safe_evidence_status(final_rag_trace.evidence_status if final_rag_trace else "DIRECT"),
                visual_asset_ids=_extract_visual_asset_ids(final_context_chunks, final_citations),
                source_pages=[
                    c.chunk.metadata.page_number
                    for c in final_context_chunks
                    if hasattr(c, "chunk") and getattr(c.chunk.metadata, "page_number", None) is not None
                ],
                document_ids=list({
                    c.chunk.metadata.document_id
                    for c in final_context_chunks
                    if hasattr(c, "chunk") and getattr(c.chunk.metadata, "document_id", None)
                }),
                answer_mode=_safe_answer_mode(final_rag_trace.answer_mode if final_rag_trace else AnswerMode.DIRECT),
                timestamp=time.time(),
            )
            turn.evidence_context = evidence_ctx
            conv_state.evidence_contexts.append(evidence_ctx)
            conv_state.turns.append(turn)
            conv_state.last_user_query = request.message
            conv_state.last_resolved_query = safe_resolved
            conv_state.last_answer = full_answer
            if final_rag_trace and final_rag_trace.active_topic:
                conv_state.active_topic = final_rag_trace.active_topic
            if final_rag_trace and final_rag_trace.active_entities:
                conv_state.active_entities = final_rag_trace.active_entities
            if final_rag_trace and final_rag_trace.query_type:
                conv_state.previous_intent = final_rag_trace.query_type
            if final_rag_trace and final_rag_trace.answer_mode:
                conv_state.previous_answer_mode = final_rag_trace.answer_mode
            if final_rag_trace and final_rag_trace.evidence_status:
                conv_state.previous_evidence_status = final_rag_trace.evidence_status
            conv_state.previous_retrieved_chunks = final_context_chunks
            conv_state.previous_citations = final_citations
            conv_state.updated_at = time.time()
            self.state_manager.update_state(session_id, conv_state)

        except Exception as exc:
            logger.exception("Error during chat stream query: %s", exc)
            self.telemetry_service.record_error(
                component="Streaming",
                severity=SeverityLevel.ERROR,
                message=f"SSE stream error: {exc!s}",
                request_id=request_id,
                conversation_id=session_id,
                stack_trace=str(exc),
            )
            err_payload = {"detail": str(exc), "status": 500, "request_id": request_id}
            yield f"event: error\ndata: {json.dumps(err_payload)}\n\n"
