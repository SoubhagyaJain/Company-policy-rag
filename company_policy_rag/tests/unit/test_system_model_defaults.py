"""
Unit tests for System Model Defaults, Config Settings, API DTOs, Routes, and Service Integration.

Authoritative Reference:
- ORIGINAL_REQUEST.md (§ R3. Default Model Configuration & System Integration)
- PROJECT.md (§ Architecture, Feature Inventory F3.1, F3.2, F3.3, Interface Contracts)
- TEST_INFRA.md (§ Feature Inventory F3.1, F3.2, F3.3 & Tier 1/2 coverage)
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from pydantic import ValidationError

from backend.models.api_dto import ChatRequest, ChatResponse, ModelInfo, ModelListResponse
from backend.models.rag import RAGResponse, RAGTrace
from src.config import Settings, settings


class TestConfigSystemModelDefaults:
    """Tests for centralized system configuration model defaults and env overrides."""

    def test_settings_default_llm_model(self) -> None:
        """Settings defaults llm_model to qwen2.5:7b."""
        cfg = Settings(_env_file=None)
        assert cfg.llm_model == "qwen2.5:7b"

    def test_settings_default_metadata_extractor_model(self) -> None:
        """Settings defaults metadata_extractor_model to qwen2.5:7b."""
        cfg = Settings(_env_file=None)
        assert cfg.metadata_extractor_model == "qwen2.5:7b"

    def test_settings_default_caption_model(self) -> None:
        """Settings defaults caption_model to qwen2.5:7b."""
        cfg = Settings(_env_file=None)
        assert cfg.caption_model == "qwen2.5:7b"

    def test_settings_default_eval_llm_model(self) -> None:
        """Settings defaults eval_llm_model to qwen2.5:7b."""
        cfg = Settings(_env_file=None)
        assert cfg.eval_llm_model == "qwen2.5:7b"

    def test_env_variable_override_llm_model(self) -> None:
        """OLLAMA_LLM_MODEL environment variable successfully overrides llm_model."""
        custom_model = "custom-coder-model"
        with patch.dict(os.environ, {"OLLAMA_LLM_MODEL": custom_model}):
            cfg = Settings(_env_file=None)
            assert cfg.llm_model == custom_model

    def test_env_variable_override_metadata_extractor_model(self) -> None:
        """METADATA_EXTRACTOR_MODEL environment variable overrides metadata_extractor_model."""
        custom_model = "custom-metadata-model"
        with patch.dict(os.environ, {"METADATA_EXTRACTOR_MODEL": custom_model}):
            cfg = Settings(_env_file=None)
            assert cfg.metadata_extractor_model == custom_model

    def test_env_variable_override_caption_model(self) -> None:
        """CAPTION_MODEL environment variable overrides caption_model."""
        custom_model = "custom-caption-model"
        with patch.dict(os.environ, {"CAPTION_MODEL": custom_model}):
            cfg = Settings(_env_file=None)
            assert cfg.caption_model == custom_model

    def test_env_variable_override_eval_model(self) -> None:
        """EVAL_LLM_MODEL environment variable overrides eval_llm_model."""
        custom_model = "custom-eval-model"
        with patch.dict(os.environ, {"EVAL_LLM_MODEL": custom_model}):
            cfg = Settings(_env_file=None)
            assert cfg.eval_llm_model == custom_model

    def test_llm_context_and_temperature_defaults(self) -> None:
        """Context window is 8192 and temperature is 0.1 for high precision."""
        assert settings.llm_context_window == 8192
        assert settings.llm_temperature == 0.1


class TestAPIDTOModelDefaults:
    """Tests for API DTO models, default model fields, and request/response serialization."""

    def test_chat_request_default_model(self) -> None:
        """ChatRequest defaults model to qwen2.5:7b."""
        req_default = ChatRequest(message="What is the remote work policy?")
        assert req_default.model == "qwen2.5:7b"

    def test_chat_request_custom_model(self) -> None:
        """ChatRequest accepts explicit model override."""
        req_custom = ChatRequest(message="Test query", model="llama3.2:3b")
        assert req_custom.model == "llama3.2:3b"

    def test_chat_response_default_model(self) -> None:
        """ChatResponse default model is qwen2.5:7b."""
        resp = ChatResponse(
            query="What is the travel meal cap?",
            answer="The travel meal cap is $75/day.",
        )
        assert resp.model == "qwen2.5:7b"
        assert resp.query == "What is the travel meal cap?"
        assert resp.answer == "The travel meal cap is $75/day."

    def test_rag_response_default_model(self) -> None:
        """RAGResponse default model is qwen2.5:7b."""
        trace = RAGTrace(query="What is the bereavement policy?")
        rag_resp = RAGResponse(
            query="What is the bereavement policy?",
            answer="Employees receive up to 5 days bereavement leave.",
            trace=trace,
        )
        assert rag_resp.model == "qwen2.5:7b"

    def test_model_list_response_dto(self) -> None:
        """ModelListResponse structure with active model and available models list."""
        m1 = ModelInfo(id="qwen2.5:7b", name="Qwen 2.5 Coder 7B Policy", type="llm", loaded=True, is_active=True)
        m2 = ModelInfo(id="nomic-embed-text", name="Nomic Embed Text", type="embedding", loaded=True, is_active=False)

        list_resp = ModelListResponse(active_model="qwen2.5:7b", models=[m1, m2])
        assert list_resp.active_model == "qwen2.5:7b"
        assert len(list_resp.models) == 2
        assert list_resp.models[0].is_active is True


class TestModelRoutesIntegration:
    """Tests for backend model listing and active model selection."""

    @patch("backend.api.routes.models.probe_ollama_tags")
    def test_get_available_models_route(self, mock_probe: MagicMock) -> None:
        """GET /api/models returns available models including default active LLM."""
        from backend.api.routes.models import get_available_models

        mock_probe.return_value = (True, ["qwen2.5:7b", "nomic-embed-text"], None)
        res = get_available_models()

        assert res.active_model == "qwen2.5:7b"
        assert len(res.models) >= 1
        model_ids = [m.id for m in res.models]
        assert "qwen2.5:7b" in model_ids

    @patch("backend.api.routes.models.probe_ollama_tags")
    def test_get_available_models_fallback_when_ollama_offline(self, mock_probe: MagicMock) -> None:
        """GET /api/models returns fallback default when Ollama is unreachable."""
        from backend.api.routes.models import get_available_models

        mock_probe.return_value = (False, [], "Connection refused")
        res = get_available_models()

        assert res.active_model == "qwen2.5:7b"
        model_ids = [m.id for m in res.models]
        assert "qwen2.5:7b" in model_ids

    @patch("backend.api.routes.models.probe_ollama_tags")
    def test_select_active_model_valid(self, mock_probe: MagicMock) -> None:
        """POST /api/models/select switches active model when model is available."""
        from backend.api.routes.models import ModelSelectRequest, select_active_model

        mock_probe.return_value = (True, ["qwen2.5:7b", "llama3.2:3b"], None)
        mock_chat_service = MagicMock()

        req = ModelSelectRequest(model="qwen2.5:7b")
        res = select_active_model(req=req, chat_service=mock_chat_service)

        assert res["status"] == "switched"
        assert res["active_model"] == "qwen2.5:7b"
        mock_chat_service.set_active_model.assert_called_once_with("qwen2.5:7b")

    @patch("backend.api.routes.models.probe_ollama_tags")
    def test_select_active_model_invalid_raises_400(self, mock_probe: MagicMock) -> None:
        """POST /api/models/select raises 400 when model ID is not available."""
        from fastapi import HTTPException
        from backend.api.routes.models import ModelSelectRequest, select_active_model

        mock_probe.return_value = (True, ["qwen2.5:7b"], None)
        mock_chat_service = MagicMock()

        req = ModelSelectRequest(model="nonexistent-model-xyz")
        with pytest.raises(HTTPException) as exc_info:
            select_active_model(req=req, chat_service=mock_chat_service)
        assert exc_info.value.status_code == 400

    @staticmethod
    def _ollama_tags() -> MagicMock:
        """What Ollama's /api/tags returns: two local models and one cloud model."""
        payload = {
            "models": [
                {"name": "qwen2.5:7b", "size": 4_683_087_332, "details": {"parameter_size": "7.6B"}},
                {"name": "llama3.2:3b", "size": 2_019_393_189, "details": {"parameter_size": "3.2B"}},
                {
                    "name": "gpt-oss:120b-cloud",
                    "size": 384,
                    "remote_host": "https://ollama.com:443",
                    "remote_model": "gpt-oss:120b",
                },
            ]
        }
        response = MagicMock()
        response.read.return_value = json.dumps(payload).encode("utf-8")
        response.__enter__.return_value = response
        return response

    def test_cloud_models_are_not_listed_as_installed(self) -> None:
        """A cloud model is a manifest pointing at a remote host, not weights on this machine."""
        from src import ollama_client

        with patch.object(ollama_client, "urlopen", side_effect=lambda *a, **k: self._ollama_tags()):
            _ok, every_name, _ = ollama_client.probe_ollama_tags()
            _ok, local_names, _ = ollama_client.probe_ollama_tags(local_only=True)
            installed = [m["name"] for m in ollama_client.list_installed_models()]

        assert "gpt-oss:120b-cloud" in every_name
        assert local_names == ["llama3.2:3b", "qwen2.5:7b"]
        assert installed == ["qwen2.5:7b", "llama3.2:3b"]

    def test_model_picker_offers_and_accepts_only_installed_models(self) -> None:
        from fastapi import HTTPException
        from backend.api.routes.models import ModelSelectRequest, get_available_models, select_active_model
        from src import ollama_client

        with (
            patch.object(ollama_client, "urlopen", side_effect=lambda *a, **k: self._ollama_tags()),
            patch.object(ollama_client, "fetch_model_details", return_value={}),
        ):
            listed = [m.id for m in get_available_models().models]
            with pytest.raises(HTTPException) as exc_info:
                select_active_model(req=ModelSelectRequest(model="gpt-oss:120b-cloud"), chat_service=MagicMock())

        assert sorted(listed) == ["llama3.2:3b", "qwen2.5:7b"]
        assert exc_info.value.status_code == 400

    def test_parameter_size_is_the_model_size_not_its_sampling_settings(self) -> None:
        from src.ollama_client import _parse_param_size

        shown = {"parameters": 'temperature 0.6\nstop "<|eot_id|>"', "details": {"parameter_size": "8.2B"}}

        assert _parse_param_size(shown) == "8.2B"
        assert _parse_param_size({"parameters": "temperature 0.6"}) is None


class TestPipelineAndServiceDefaults:
    """Tests for default model handling in RAGPipeline and ChatService."""

    def test_rag_pipeline_default_model(self) -> None:
        """RAGPipeline defaults to qwen2.5:7b when llm model is unspecified."""
        from backend.rag.pipeline import RAGPipeline

        retriever = MagicMock()
        reranker = MagicMock()
        pipeline = RAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            llm=None,
        )
        assert pipeline.get_active_model() == "qwen2.5:7b"

    def test_chat_service_execute_query_default_model(self) -> None:
        """ChatService execute_query returns qwen2.5:7b default."""
        from backend.services.chat_service import ChatService
        from backend.models.rag import RAGResponse, RAGTrace

        mock_pipe = MagicMock()
        mock_pipe.get_active_model.return_value = "qwen2.5:7b"
        mock_pipe.query.return_value = RAGResponse(
            query="What is the holiday schedule?",
            answer="There are 11 paid company holidays.",
            trace=RAGTrace(query="What is the holiday schedule?"),
            model="qwen2.5:7b",
        )
        mock_telemetry = MagicMock()

        svc = ChatService(rag_pipeline=mock_pipe, telemetry_service=mock_telemetry)
        req = ChatRequest(message="What is the holiday schedule?")
        resp = svc.execute_query(req)

        assert resp.model == "qwen2.5:7b"
        assert resp.answer == "There are 11 paid company holidays."
