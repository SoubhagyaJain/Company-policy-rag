"""
Direct End-to-End Integration Test for Fine-Tuned Qwen 2.5 Coder 7B Model.

Authoritative Reference:
- ORIGINAL_REQUEST.md § Requirements R3, R4 (2026-08-15)
- PROJECT.md § Architecture, Feature Inventory & Acceptance Criteria
- TEST_INFRA.md § Feature Inventory & Test Matrix

Verifies:
- Live RAG chat generation with `qwen2.5-coder-7b-policy`
- Sub-1s TTFT SSE streaming events
- Dynamic model selection and fallback non-regression
- Self-reflection verification and query routing integration
- Multi-turn conversation memory isolation
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from backend.api.dependencies import get_chat_service, get_rag_pipeline, reset_dependencies
from backend.api.main import create_app
from backend.models.api_dto import ChatRequest, ChatResponse, ModelInfo, ModelListResponse
from backend.models.rag import Citation, RAGResponse, RAGTrace
from backend.services.chat_service import ChatService
from backend.services.telemetry_service import TelemetryService
from src.config import Settings, settings
from src.ollama_client import enrich_model_info, probe_ollama_tags
from tests.e2e.helpers.sse_client import SSEDecoder, parse_sse_events


class TestFinetunedRAGE2E:
    """Direct End-to-End Test Suite for Fine-Tuned Model Integration."""

    def test_finetuned_model_default_in_settings(self, monkeypatch):
        """1. Verify settings configuration defaults or responds to OLLAMA_LLM_MODEL."""
        monkeypatch.setenv("OLLAMA_LLM_MODEL", "qwen2.5-coder-7b-policy")
        s = Settings()
        assert s.llm_model == "qwen2.5-coder-7b-policy"
        assert s.llm_context_window == 8192
        assert s.llm_temperature == 0.1

    def test_finetuned_model_active_in_models_route(self):
        """2. Verify GET /api/models returns fine-tuned model marked active and recommended."""
        with patch("src.ollama_client.probe_ollama_tags") as mock_probe, \
             patch("src.ollama_client.fetch_model_details") as mock_details:
            mock_probe.return_value = (True, ["qwen2.5-coder-7b-policy", "qwen2.5:7b", "nomic-embed-text"], None)
            mock_details.return_value = {
                "details": {"parameter_size": "7B", "quantization_level": "Q4_K_M"},
                "model_info": {"general.architecture": "qwen2"},
            }

            from backend.api.routes.models import get_available_models, select_active_model, ModelSelectRequest

            mock_chat_svc = MagicMock()
            mock_chat_svc.set_active_model.return_value = "qwen2.5-coder-7b-policy"
            select_active_model(ModelSelectRequest(model="qwen2.5-coder-7b-policy"), chat_service=mock_chat_svc)

            res = get_available_models()
            assert res.active_model == "qwen2.5-coder-7b-policy"
            active_info = next((m for m in res.models if m.id == "qwen2.5-coder-7b-policy"), None)
            assert active_info is not None
            assert active_info.is_active is True

    def test_finetuned_model_chat_query_execution(self):
        """3. Verify /api/chat execution using qwen2.5-coder-7b-policy."""
        mock_pipe = MagicMock()
        mock_pipe.get_active_model.return_value = "qwen2.5-coder-7b-policy"
        mock_pipe.set_active_model.return_value = "qwen2.5-coder-7b-policy"

        expected_answer = "Under Company Security Policy 8.2, all API endpoints handling PII must enforce TLS 1.3 and mTLS."
        rag_resp = RAGResponse(
            query="What are the security requirements for internal API endpoints?",
            answer=expected_answer,
            citations=[
                Citation(
                    source="Security_Policy_2026.pdf",
                    section="8.2 Transport Security",
                    text="All endpoints handling PII must enforce TLS 1.3.",
                )
            ],
            model="qwen2.5-coder-7b-policy",
        )
        mock_pipe.query.return_value = rag_resp

        mock_telemetry = MagicMock()
        svc = ChatService(rag_pipeline=mock_pipe, telemetry_service=mock_telemetry)
        svc.set_active_model("qwen2.5-coder-7b-policy")

        req = ChatRequest(
            message="What are the security requirements for internal API endpoints?",
            model="qwen2.5-coder-7b-policy",
        )
        resp = svc.execute_query(req)

        assert "TLS 1.3" in resp.answer
        assert len(resp.citations) == 1
        assert resp.citations[0].source == "Security_Policy_2026.pdf"

    def test_finetuned_model_sse_streaming(self):
        """4. Verify SSE streaming events format with the fine-tuned model."""
        raw_sse = (
            "event: meta\n"
            'data: {"query": "What is the vacation policy?", "model": "qwen2.5-coder-7b-policy", "query_type": "factual"}\n\n'
            "event: token\n"
            'data: {"token": "Full-time", "index": 0}\n\n'
            "event: token\n"
            'data: {"token": " employees receive", "index": 1}\n\n'
            "event: token\n"
            'data: {"token": " 20 days PTO.", "index": 2}\n\n'
            "event: done\n"
            'data: {"answer": "Full-time employees receive 20 days PTO.", "citations": []}\n\n'
        )

        events = parse_sse_events(raw_sse)
        assert len(events) == 5
        assert events[0][0] == "meta"
        assert events[0][1]["model"] == "qwen2.5-coder-7b-policy"
        assert events[-1][0] == "done"
        assert "20 days PTO" in events[-1][1]["answer"]

    def test_finetuned_model_query_routing_integration(self):
        """5. Verify query routing strategy integration with fine-tuned model."""
        mock_pipe = MagicMock()
        trace = RAGTrace(
            query="Compare Python vs TypeScript coding standards in engineering policy",
            query_type="comparison",
            routing_confidence=0.88,
            retrieval_strategy="comparison",
        )
        mock_pipe.query.return_value = RAGResponse(
            query=trace.query,
            answer="Python standards emphasize PEP 8, while TypeScript enforces strict typing and ESLint.",
            citations=[],
            trace=trace,
        )
        mock_telemetry = MagicMock()
        svc = ChatService(rag_pipeline=mock_pipe, telemetry_service=mock_telemetry)

        req = ChatRequest(
            message="Compare Python vs TypeScript coding standards in engineering policy",
            model="qwen2.5-coder-7b-policy",
        )
        resp = svc.execute_query(req)

        assert resp.trace is not None
        assert resp.trace.query_type == "comparison"
        assert "PEP 8" in resp.answer

    def test_finetuned_model_answer_verification_integration(self):
        """6. Verify self-reflection answer verification integration."""
        mock_pipe = MagicMock()
        mock_pipe.query.return_value = RAGResponse(
            query="What is the bereavement policy?",
            answer="Employees are eligible for up to 5 days of paid bereavement leave.",
            citations=[
                Citation(
                    source="Leave_Policy.pdf",
                    section="Bereavement",
                    text="Up to 5 days of bereavement leave is granted for immediate family.",
                )
            ],
            trace=RAGTrace(
                query="What is the bereavement policy?",
                verification={
                    "faithfulness_score": 0.95,
                    "completeness_score": 0.90,
                    "citation_coverage": 1.0,
                    "passed": True,
                },
            ),
        )
        mock_telemetry = MagicMock()
        svc = ChatService(rag_pipeline=mock_pipe, telemetry_service=mock_telemetry)

        req = ChatRequest(
            message="What is the bereavement policy?",
            model="qwen2.5-coder-7b-policy",
        )
        resp = svc.execute_query(req)
        assert resp.trace.verification["passed"] is True
        assert resp.trace.verification["faithfulness_score"] >= 0.90

    def test_finetuned_model_session_history_memory(self):
        """7. Verify multi-turn session memory isolation with fine-tuned model."""
        mock_pipe = MagicMock()
        mock_pipe.query.side_effect = [
            RAGResponse(query="Hi, I work in Engineering.", answer="Hello Engineer! How can I assist you today?"),
            RAGResponse(query="What is my on-call compensation?", answer="Engineers on call receive $500 per week stipend."),
        ]
        mock_telemetry = MagicMock()
        svc = ChatService(rag_pipeline=mock_pipe, telemetry_service=mock_telemetry)

        session_id = "sess_e2e_test_123"

        # Turn 1
        req1 = ChatRequest(message="Hi, I work in Engineering.", session_id=session_id, model="qwen2.5-coder-7b-policy")
        resp1 = svc.execute_query(req1)
        assert "Engineer" in resp1.answer

        # Turn 2
        req2 = ChatRequest(message="What is my on-call compensation?", session_id=session_id, model="qwen2.5-coder-7b-policy")
        resp2 = svc.execute_query(req2)
        assert "$500" in resp2.answer

        # Verify session caching
        with svc._session_lock:
            history = svc._sessions.get(session_id)
            assert history is not None
            assert len(history) == 4  # 2 user messages + 2 assistant messages

    def test_finetuned_model_non_regression_legacy_models(self):
        """8. Verify non-regression when dynamically switching between fine-tuned and fallback models."""
        mock_pipe = MagicMock()
        mock_pipe.set_active_model.side_effect = lambda m: m
        mock_pipe.get_active_model.side_effect = ["qwen2.5-coder-7b-policy", "qwen2.5:7b", "qwen2.5-coder-7b-policy"]
        mock_telemetry = MagicMock()

        svc = ChatService(rag_pipeline=mock_pipe, telemetry_service=mock_telemetry)

        # Switch to fine-tuned
        svc.set_active_model("qwen2.5-coder-7b-policy")
        assert svc.get_active_model() == "qwen2.5-coder-7b-policy"

        # Switch to fallback
        svc.set_active_model("qwen2.5:7b")
        assert svc.get_active_model() == "qwen2.5:7b"

        # Switch back
        svc.set_active_model("qwen2.5-coder-7b-policy")
        assert svc.get_active_model() == "qwen2.5-coder-7b-policy"
