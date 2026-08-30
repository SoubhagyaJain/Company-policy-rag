from __future__ import annotations

from unittest.mock import MagicMock

from backend.models.api_dto import ChatRequest
from backend.rag.pipeline import RAGPipeline
from backend.services.chat_service import ChatService


class _CopyableLLM:
    def __init__(self, model: str) -> None:
        self.model = model

    def complete(self, _prompt: str, **_kwargs: object) -> str:
        return self.model


def _pipeline(llm: object) -> RAGPipeline:
    return RAGPipeline(
        hybrid_retriever=MagicMock(),
        reranker=MagicMock(),
        llm=llm,
    )


def test_omitted_request_model_uses_dynamic_active_model() -> None:
    request = ChatRequest(message="Hello")
    pipeline = _pipeline(_CopyableLLM("qwen2.5:7b"))
    pipeline.set_active_model("llama3.2:3b")

    client, selected = pipeline._get_effective_llm(None)

    assert request.model == "qwen2.5:7b"
    assert selected == "llama3.2:3b"
    assert client.complete("test") == "llama3.2:3b"


def test_chat_service_distinguishes_omitted_and_explicit_models() -> None:
    omitted = ChatRequest(message="Hello")
    explicit = ChatRequest(message="Hello", model="gemma2:2b")

    assert ChatService._request_model_override(omitted) is None
    assert ChatService._request_model_override(explicit) == "gemma2:2b"


def test_per_request_models_use_isolated_clients() -> None:
    base = _CopyableLLM("qwen2.5:7b")
    pipeline = _pipeline(base)

    llama_client, llama_name = pipeline._get_effective_llm("llama3.2:3b")
    gemma_client, gemma_name = pipeline._get_effective_llm("gemma2:2b")

    assert llama_name == "llama3.2:3b"
    assert gemma_name == "gemma2:2b"
    assert llama_client is not gemma_client
    assert llama_client.complete("test") == "llama3.2:3b"
    assert gemma_client.complete("test") == "gemma2:2b"
    assert base.model == "qwen2.5:7b"
