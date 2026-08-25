"""
Adversarial Stress & Hardening Test Suite (Tier 5) for Milestone 4:
Dynamic Model Switching, Backend RAG Routing, SSE Streaming, Modelfile Generation,
and Ollama Probe Resilience.

Authoritative Reference:
- ORIGINAL_REQUEST.md (§ R2, R3, R4)
- PROJECT.md (§ Architecture, Feature Inventory F2.3, F2.4, F3.1, F3.2, F3.3, F4.5)
- TEST_INFRA.md (§ Test Philosophy, Tier 1-5 Adversarial Verification)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.error import HTTPError, URLError
import pytest

# Ensure sys.path includes both PROJECT_ROOT and WORKSPACE_ROOT
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
for p in [str(PROJECT_ROOT), str(WORKSPACE_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Dynamic imports with fallback
try:
    from company_policy_rag.backend.models.api_dto import ChatRequest, ChatResponse, ModelInfo, ModelListResponse
    from company_policy_rag.backend.models.chunk import Chunk
    from company_policy_rag.backend.models.rag import (
        Citation,
        QueryCategory,
        QueryClassification,
        RAGResponse,
        RAGTrace,
        RetrievalStrategy,
        ScoredChunk,
        VerificationReport,
    )
    from company_policy_rag.backend.rag.pipeline import ModelManager, RAGPipeline, _LLMProxy
    from company_policy_rag.backend.services.chat_service import ChatService
    from company_policy_rag.backend.services.telemetry_service import TelemetryService
    from company_policy_rag.src.config import Settings, settings
    from company_policy_rag.src.finetuning.modelfile_generator import (
        DEFAULT_ENTERPRISE_SYSTEM_PROMPT,
        ModelfileGenerator,
        generate_modelfile,
        normalize_gguf_path,
        parse_modelfile,
    )
    from company_policy_rag.src.finetuning.ollama_registrar import (
        OllamaRegistrar,
        find_ollama_binary,
        get_model_details,
        probe_ollama_tags,
        register_model_api,
        register_model_cli,
        register_model_in_ollama,
        verify_model_registered,
    )
    from company_policy_rag.src.ollama_client import (
        enrich_model_info,
        fetch_model_details,
        filter_chat_models,
        list_enriched_models,
        preload_model,
        unload_model,
    )
except ImportError:
    from backend.models.api_dto import ChatRequest, ChatResponse, ModelInfo, ModelListResponse
    from backend.models.chunk import Chunk
    from backend.models.rag import (
        Citation,
        QueryCategory,
        QueryClassification,
        RAGResponse,
        RAGTrace,
        RetrievalStrategy,
        ScoredChunk,
        VerificationReport,
    )
    from backend.rag.pipeline import ModelManager, RAGPipeline, _LLMProxy
    from backend.services.chat_service import ChatService
    from backend.services.telemetry_service import TelemetryService
    from src.config import Settings, settings
    from src.finetuning.modelfile_generator import (
        DEFAULT_ENTERPRISE_SYSTEM_PROMPT,
        ModelfileGenerator,
        generate_modelfile,
        normalize_gguf_path,
        parse_modelfile,
    )
    from src.finetuning.ollama_registrar import (
        OllamaRegistrar,
        find_ollama_binary,
        get_model_details,
        probe_ollama_tags,
        register_model_api,
        register_model_cli,
        register_model_in_ollama,
        verify_model_registered,
    )
    from src.ollama_client import (
        enrich_model_info,
        fetch_model_details,
        filter_chat_models,
        list_enriched_models,
        preload_model,
        unload_model,
    )


# ============================================================================
# HELPER FIXTURES & MOCK GENERATORS
# ============================================================================

def make_dummy_scored_chunk(chunk_id: str = "c1", text: str = "Policy content snippet.") -> ScoredChunk:
    ch = Chunk(
        id=chunk_id,
        text=text,
        document_id="doc_1",
        chunk_index=0,
        token_count=20,
    )
    return ScoredChunk(chunk=ch, score=0.95, dense_score=0.9, bm25_score=10.0, rrf_score=0.03)


def make_mock_pipeline():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [make_dummy_scored_chunk()]
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [make_dummy_scored_chunk()]
    mock_router = MagicMock()
    mock_router.classify.return_value = QueryClassification(
        category=QueryCategory.FACTUAL,
        confidence=0.95,
        strategy=RetrievalStrategy(name="factual", dense_top_k=5, bm25_top_k=5, rerank_top_n=3),
    )
    mock_rewriter = MagicMock()
    mock_rewriter.is_conversational.return_value = False
    mock_rewriter.rewrite.return_value = MagicMock(rewritten_query="What is the travel policy?", is_comprehensive_list=False)

    mock_llm = MagicMock()
    mock_llm.model = "qwen2.5-coder-7b-policy"
    mock_llm.complete.return_value = "The travel meal reimbursement is $75/day. [Source 1]"

    pipeline = RAGPipeline(
        hybrid_retriever=mock_retriever,
        reranker=mock_reranker,
        query_router=mock_router,
        query_rewriter=mock_rewriter,
        llm=mock_llm,
    )
    return pipeline


# ============================================================================
# 1. CONCURRENT RAG QUERIES WITH DYNAMIC MODEL SWITCHING
# ============================================================================

class TestConcurrentDynamicModelSwitching:
    """Stress-test concurrent RAG queries while dynamic model switching occurs."""

    def test_concurrent_rag_queries_with_dynamic_model_switches(self):
        """
        Spawn 30 concurrent worker threads executing RAG queries with alternating
        per-request model selections ('qwen2.5-coder-7b-policy', 'llama3:latest', None)
        while a background thread constantly mutates global pipeline active model.
        Verifies thread-safety, zero exceptions, and isolation of per-request models.
        """
        pipeline = make_mock_pipeline()
        telemetry = TelemetryService(max_traces=500)
        chat_service = ChatService(rag_pipeline=pipeline, telemetry_service=telemetry)

        errors = []
        completed_queries = []
        lock = threading.Lock()
        stop_switching = threading.Event()

        models_to_test = [
            "qwen2.5-coder-7b-policy",
            "llama3:latest",
            "qwen2.5:7b",
            "deepseek-r1:8b",
            None,  # Should use global active model
        ]

        def switcher_worker():
            idx = 0
            while not stop_switching.is_set():
                m = models_to_test[idx % 4]
                try:
                    chat_service.set_active_model(m)
                except Exception as e:
                    with lock:
                        errors.append(("switcher", e))
                idx += 1
                time.sleep(0.005)

        def query_worker(worker_id: int):
            for i in range(10):
                target_model = models_to_test[(worker_id + i) % len(models_to_test)]
                req = ChatRequest(
                    message=f"Query {worker_id}-{i}: What is the policy?",
                    model=target_model,
                    session_id=f"session_{worker_id}",
                )
                try:
                    resp = chat_service.execute_query(req)
                    with lock:
                        completed_queries.append((worker_id, i, target_model, resp.model))
                except Exception as e:
                    with lock:
                        errors.append(("query", worker_id, i, e))
                time.sleep(0.002)

        switcher_thread = threading.Thread(target=switcher_worker, daemon=True)
        switcher_thread.start()

        query_threads = [threading.Thread(target=query_worker, args=(w_id,)) for w_id in range(15)]
        for t in query_threads:
            t.start()
        for t in query_threads:
            t.join(timeout=15.0)

        stop_switching.set()
        switcher_thread.join(timeout=2.0)

        assert len(errors) == 0, f"Encountered concurrency errors: {errors}"
        assert len(completed_queries) == 150, f"Expected 150 completed queries, got {len(completed_queries)}"

        # Verify that explicit model requests were respected
        for w_id, i, target_model, resp_model in completed_queries:
            if target_model is not None:
                assert resp_model == target_model, f"Expected model {target_model}, got {resp_model}"
            else:
                assert resp_model is not None

    def test_llm_proxy_concurrency_and_model_restoration(self):
        """
        Adversarially test _LLMProxy under multi-threaded contention.
        Verify that wrapping shared LLM instances does not leak state across threads.
        """
        mock_shared_llm = MagicMock()
        mock_shared_llm.model = "base-default-model"

        def complete_side_effect(prompt, **kwargs):
            time.sleep(0.001)
            # Verify the model property at the moment of execution
            return f"Answer for {mock_shared_llm.model}"

        mock_shared_llm.complete.side_effect = complete_side_effect

        results = []
        errors = []
        lock = threading.Lock()

        def proxy_caller(thread_id: int, assigned_model: str):
            proxy = _LLMProxy(mock_shared_llm, assigned_model)
            for _ in range(10):
                try:
                    ans = proxy.complete(f"Prompt {thread_id}")
                    with lock:
                        results.append((assigned_model, ans))
                except Exception as exc:
                    with lock:
                        errors.append((thread_id, exc))

        threads = []
        test_models = ["qwen2.5-coder-7b-policy", "llama3:latest", "mistral:7b", "gemma:2b"]
        for i in range(16):
            t = threading.Thread(target=proxy_caller, args=(i, test_models[i % len(test_models)]))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert len(errors) == 0, f"Proxy caller failed with errors: {errors}"
        assert len(results) == 160
        # Check that the underlying shared LLM was restored to base model or preserved
        assert mock_shared_llm.model == "base-default-model"

    def test_model_manager_non_blocking_preload_resilience(self):
        """
        Stress ModelManager rapid switching without deadlocks or thread leaks.
        """
        mgr = ModelManager(initial_model="qwen2.5-coder-7b-policy")
        with patch("company_policy_rag.backend.rag.pipeline.preload_model", side_effect=Exception("Preload connection failure")):
            for i in range(50):
                mgr.set_model(f"model_switch_{i}")
                assert mgr.current_model == f"model_switch_{i}"


# ============================================================================
# 2. SSE STREAMING INTERRUPTION, EMPTY CHUNKS & TOKEN BOUNDARIES
# ============================================================================

class TestSSEStreamingStressAndBoundaries:
    """Stress-test SSE streaming interruption, empty chunk handling, and token preservation."""

    @pytest.mark.asyncio
    async def test_sse_streaming_immediate_cancellation(self):
        """Test stream cancellation before first token is yielded."""
        pipeline = make_mock_pipeline()
        telemetry = TelemetryService(max_traces=100)
        chat_service = ChatService(rag_pipeline=pipeline, telemetry_service=telemetry)

        req = ChatRequest(message="What is the bereavement policy?", model="qwen2.5-coder-7b-policy")
        cancel_token = asyncio.Event()
        cancel_token.set()  # Cancel immediately

        events = []
        async for event in chat_service.stream_query(req, cancel_token=cancel_token):
            events.append(event)

        # Should exit quickly without done event
        done_events = [e for e in events if "event: done" in e]
        assert len(done_events) == 0

    @pytest.mark.asyncio
    async def test_sse_streaming_midstream_cancellation(self):
        """Test stream cancellation mid-token generation."""
        pipeline = make_mock_pipeline()
        telemetry = TelemetryService(max_traces=100)
        chat_service = ChatService(rag_pipeline=pipeline, telemetry_service=telemetry)

        req = ChatRequest(message="What is the code of conduct policy?", model="qwen2.5-coder-7b-policy")
        cancel_token = asyncio.Event()

        events = []
        chunk_count = 0
        async for event in chat_service.stream_query(req, cancel_token=cancel_token):
            events.append(event)
            if "event: chunk" in event:
                chunk_count += 1
                if chunk_count >= 2:
                    cancel_token.set()  # Abort mid-stream

        done_events = [e for e in events if "event: done" in e]
        assert len(done_events) == 0, "Done event should not be yielded when stream is cancelled"

    @pytest.mark.asyncio
    async def test_token_boundary_preservation_and_exact_reconstruction(self):
        """
        Stress test token chunk assembly and verify exact bitwise reconstruction
        of complex multi-line text with code snippets, quotes, and unicode.
        """
        complex_policy_text = (
            "Enterprise Policy (§ 4.2):\n"
            "1. Remote work is permitted up to 2 days/week.\n"
            "2. Expense reimbursement: €75/day max for meals.\n"
            "3. Code snippet:\n"
            "```bash\n"
            "ollama run qwen2.5-coder-7b-policy:latest\n"
            "```\n"
            "Contact: hr-support@company.com — Confidential [Source 1]."
        )

        pipeline = make_mock_pipeline()
        pipeline.llm.complete.return_value = complex_policy_text
        telemetry = TelemetryService(max_traces=100)
        chat_service = ChatService(rag_pipeline=pipeline, telemetry_service=telemetry)

        req = ChatRequest(message="Explain policy 4.2 in detail", model="qwen2.5-coder-7b-policy")

        chunks = []
        done_payload = None

        async for event_str in chat_service.stream_query(req):
            for block in event_str.strip().split("\n\n"):
                if not block.strip():
                    continue
                lines = block.strip().split("\n")
                if len(lines) >= 2 and lines[0].startswith("event:"):
                    event_type = lines[0].replace("event:", "").strip()
                    data_str = lines[1].replace("data:", "").strip()
                    data = json.loads(data_str)

                    if event_type == "chunk":
                        chunks.append(data["content"])
                    elif event_type == "done":
                        done_payload = data

        reconstructed = "".join(chunks)
        assert done_payload is not None, "Expected done event in stream"
        assert done_payload["answer"] == complex_policy_text
        assert reconstructed == complex_policy_text, f"Mismatch in reconstructed tokens:\nExpected: {complex_policy_text!r}\nGot: {reconstructed!r}"

    @pytest.mark.asyncio
    async def test_empty_and_whitespace_query_stream_error_handling(self):
        """Verify stream yields structured 400 error event for empty/whitespace input."""
        pipeline = make_mock_pipeline()
        telemetry = TelemetryService(max_traces=100)
        chat_service = ChatService(rag_pipeline=pipeline, telemetry_service=telemetry)

        for empty_msg in ["", "   ", "\t\n  "]:
            req = ChatRequest(message=empty_msg, model="qwen2.5-coder-7b-policy")
            events = []
            async for event in chat_service.stream_query(req):
                events.append(event)

            assert len(events) == 1
            assert "event: error" in events[0]
            assert "400" in events[0]


# ============================================================================
# 3. MODELFILE GENERATION WITH EXTREME HYPERPARAMETERS & ESCAPES
# ============================================================================

class TestModelfileAdversarialGeneration:
    """Stress-test Modelfile generation with extreme hyperparameter values and special tokens."""

    def test_extreme_context_window_values(self, tmp_path):
        """Test Modelfile generation with extreme num_ctx (131072, 1, 1048576)."""
        for extreme_ctx in [131072, 1, 1048576, 262144]:
            out_file = tmp_path / f"Modelfile_ctx_{extreme_ctx}"
            content = generate_modelfile(
                gguf_path="./models/qwen.gguf",
                output_path=out_file,
                num_ctx=extreme_ctx,
            )
            assert f"PARAMETER num_ctx {extreme_ctx}" in content
            parsed = parse_modelfile(out_file)
            assert int(parsed["parameters"]["num_ctx"]) == extreme_ctx

    def test_extreme_temperature_and_sampling_parameters(self, tmp_path):
        """Test extreme temperature values (0.0, 2.0, 100.0) and custom sampling params."""
        custom_params = {
            "top_k": 100,
            "top_p": 0.999,
            "repeat_last_n": 256,
            "mirostat": 2,
            "mirostat_eta": 0.1,
            "mirostat_tau": 5.0,
        }
        out_file = tmp_path / "Modelfile_extreme_temp"
        content = generate_modelfile(
            gguf_path="models/qwen2.5-coder.gguf",
            output_path=out_file,
            temperature=0.0,
            parameters=custom_params,
        )
        assert "PARAMETER temperature 0.0" in content
        assert "PARAMETER top_k 100" in content
        assert "PARAMETER top_p 0.999" in content
        assert "PARAMETER mirostat 2" in content

        parsed = parse_modelfile(out_file)
        assert float(parsed["parameters"]["temperature"]) == 0.0
        assert parsed["parameters"]["mirostat"] == "2"

    def test_special_stop_token_escapes(self, tmp_path):
        """Test comprehensive stop token lists with special chat markers and brackets."""
        stop_tokens = [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|fim_prefix|>",
            "<|fim_middle|>",
            "<|fim_suffix|>",
            "[INST]",
            "[/INST]",
            "<s>",
            "</s>",
            "### Instruction:",
            "### Response:",
        ]
        out_file = tmp_path / "Modelfile_stops"
        content = generate_modelfile(
            gguf_path="models/model.gguf",
            output_path=out_file,
            stop_tokens=stop_tokens,
        )
        for stop in stop_tokens:
            assert f'PARAMETER stop "{stop}"' in content

        parsed = parse_modelfile(out_file)
        assert len(parsed["stop_tokens"]) == len(stop_tokens)
        for stop in stop_tokens:
            assert stop in parsed["stop_tokens"]

    def test_adversarial_system_prompt_handling(self, tmp_path):
        """Test Modelfile with empty prompt fallback and multiline prompts containing markdown/quotes."""
        # 1. Empty prompt -> falls back to enterprise prompt
        out1 = tmp_path / "Modelfile_empty_sys"
        generate_modelfile(gguf_path="model.gguf", output_path=out1, system_prompt="   ")
        parsed1 = parse_modelfile(out1)
        assert parsed1["system"] == DEFAULT_ENTERPRISE_SYSTEM_PROMPT

        # 2. Multiline prompt with code blocks
        custom_prompt = (
            "You are a specialized policy AI.\n"
            "Follow rules:\n"
            "- Never hallucinate\n"
            "- Always cite [Source N]\n"
            "Example:\n"
            "```json\n"
            '{"status": "ok"}\n'
            "```"
        )
        out2 = tmp_path / "Modelfile_complex_sys"
        generate_modelfile(gguf_path="model.gguf", output_path=out2, system_prompt=custom_prompt)
        parsed2 = parse_modelfile(out2)
        assert "Never hallucinate" in parsed2["system"]
        assert '{"status": "ok"}' in parsed2["system"]

    def test_posix_path_normalization_edge_cases(self):
        """Test Windows backslash normalization to POSIX forward slash across various formats."""
        assert normalize_gguf_path(r"C:\Users\jains\models\qwen.gguf") == "C:/Users/jains/models/qwen.gguf"
        assert normalize_gguf_path(r"\\server\share\models\qwen.gguf") == "//server/share/models/qwen.gguf"
        assert normalize_gguf_path("/opt/models/qwen.gguf") == "/opt/models/qwen.gguf"

        with pytest.raises(ValueError):
            normalize_gguf_path("")
        with pytest.raises(ValueError):
            normalize_gguf_path("   ")


# ============================================================================
# 4. OLLAMA REGISTRAR RESILIENCE (LATENCY, MALFORMED JSON, DROPS)
# ============================================================================

class TestOllamaRegistrarResilience:
    """Stress-test Ollama registrar against network latency spikes, malformed JSON, and daemon drops."""

    @patch("company_policy_rag.src.finetuning.ollama_registrar.urlopen")
    def test_probe_ollama_tags_malformed_json_resilience(self, mock_urlopen: MagicMock):
        """Verify probe_ollama_tags handles non-JSON, invalid schemas, and empty responses safely."""
        malformed_payloads = [
            b"<!DOCTYPE html><html><body>502 Bad Gateway</body></html>",
            b"GARBAGE_BINARY_DATA\x00\x01\x02\xff\xfe",
            b"",
            b"   ",
            b'{"models": "this_should_be_a_list_not_a_string"}',
            b'{"models": [123, null, "string_without_name_key", {}]}',
            b'{"unknown_root": 42}',
        ]

        for payload in malformed_payloads:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.read.return_value = payload
            mock_urlopen.return_value.__enter__.return_value = mock_resp

            ok, models, err = probe_ollama_tags()
            if payload == b'{"models": [123, null, "string_without_name_key", {}]}':
                assert ok is True
                assert models == []  # Safely filters out items without 'name'
            elif b"this_should_be_a_list" in payload or b"unknown_root" in payload:
                assert ok is True
                assert models == []
            else:
                assert ok is False
                assert models == []
                assert err is not None

    @patch("company_policy_rag.src.finetuning.ollama_registrar.urlopen")
    def test_get_model_details_malformed_json_and_http_errors(self, mock_urlopen: MagicMock):
        """Verify get_model_details returns empty dict on malformed JSON or HTTP exceptions."""
        # 1. Malformed JSON
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"NOT_VALID_JSON"
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        details = get_model_details("qwen2.5-coder-7b-policy")
        assert details == {}

        # 2. HTTP 404 / 500 error
        mock_urlopen.side_effect = HTTPError(url="http://localhost:11434/api/show", code=404, msg="Model Not Found", hdrs={}, fp=None)
        details_404 = get_model_details("nonexistent_model")
        assert details_404 == {}

        # 3. Connection Reset / Timeout
        mock_urlopen.side_effect = ConnectionResetError("Connection reset by peer")
        details_reset = get_model_details("qwen2.5-coder-7b-policy")
        assert details_reset == {}

    @patch("company_policy_rag.src.finetuning.ollama_registrar.urlopen")
    def test_network_latency_spike_and_timeout(self, mock_urlopen: MagicMock):
        """Verify probe and registration functions catch TimeoutError gracefully."""
        mock_urlopen.side_effect = TimeoutError("The read operation timed out")

        ok, models, err = probe_ollama_tags(timeout=0.01)
        assert ok is False
        assert models == []
        assert "timed out" in str(err).lower()

        reg_ok = register_model_api(
            model_name="qwen2.5-coder-7b-policy",
            modelfile_path=PROJECT_ROOT / "Modelfile.gemma4-fast",
            timeout=0.01,
        )
        assert reg_ok is False

    @patch("company_policy_rag.src.finetuning.ollama_registrar.register_model_cli")
    @patch("company_policy_rag.src.finetuning.ollama_registrar.register_model_api")
    def test_dual_channel_fallback_behavior(self, mock_api: MagicMock, mock_cli: MagicMock):
        """
        Verify register_model_in_ollama:
        1. When prefer_api=True and API fails, CLI fallback is called.
        2. When prefer_api=False and CLI fails, API fallback is called.
        3. When both fail, returns False safely.
        """
        modelfile = PROJECT_ROOT / "Modelfile.gemma4-fast"

        # Case 1: API fails -> CLI succeeds
        mock_api.return_value = False
        mock_cli.return_value = True
        res1 = register_model_in_ollama("qwen2.5-coder-7b-policy", modelfile, prefer_api=True)
        assert res1 is True
        mock_api.assert_called_once()
        mock_cli.assert_called_once()

        mock_api.reset_mock()
        mock_cli.reset_mock()

        # Case 2: CLI fails -> API succeeds
        mock_cli.return_value = False
        mock_api.return_value = True
        res2 = register_model_in_ollama("qwen2.5-coder-7b-policy", modelfile, prefer_api=False)
        assert res2 is True
        mock_cli.assert_called_once()
        mock_api.assert_called_once()

        mock_api.reset_mock()
        mock_cli.reset_mock()

        # Case 3: Both fail -> returns False
        mock_api.return_value = False
        mock_cli.return_value = False
        res3 = register_model_in_ollama("qwen2.5-coder-7b-policy", modelfile, prefer_api=True)
        assert res3 is False


# ============================================================================
# 5. OLLAMA CLIENT ENRICHMENT & METADATA RESILIENCE
# ============================================================================

class TestOllamaClientEnrichmentStress:
    """Stress test model metadata enrichment with missing, corrupt, and abnormal inputs."""

    def test_filter_chat_models_markers(self):
        """Verify filter_chat_models removes all embedding variations while preserving chat models."""
        models = [
            "nomic-embed-text:latest",
            "qwen2.5-coder-7b-policy:latest",
            "mxbai-embed-large:latest",
            "bge-m3:latest",
            "llama3.2:3b",
            "custom-embed-v1:latest",
        ]
        filtered = filter_chat_models(models)
        assert "qwen2.5-coder-7b-policy:latest" in filtered
        assert "llama3.2:3b" in filtered
        assert len(filtered) == 2

    @patch("company_policy_rag.src.ollama_client.fetch_model_details")
    def test_enrich_model_info_with_corrupt_details(self, mock_fetch: MagicMock):
        """Verify enrich_model_info never crashes on malformed details responses."""
        corrupt_details_list = [
            {},
            {"details": None, "model_info": "not_a_dict"},
            {"details": {"parameter_size": "INVALID_NUMBER_XYZ_B"}},
            {"details": {"parameter_size": "7.0B", "quantization_level": "Q4_K_M"}},
        ]

        for details in corrupt_details_list:
            mock_fetch.return_value = details
            info = enrich_model_info("qwen2.5-coder-7b-policy", recommended="qwen2.5-coder-7b-policy")
            assert info["id"] == "qwen2.5-coder-7b-policy"
            assert "Recommended" in info["badges"]
            assert isinstance(info["badges"], list)

    @patch("company_policy_rag.src.ollama_client.urlopen")
    def test_unload_and_preload_model_resilience(self, mock_urlopen: MagicMock):
        """Verify unload_model and preload_model handle network exceptions safely without crashing."""
        mock_urlopen.side_effect = URLError("Ollama connection refused")

        assert unload_model("qwen2.5-coder-7b-policy") is False
        assert preload_model("qwen2.5-coder-7b-policy") is False
        assert unload_model("") is False
        assert preload_model("") is False
