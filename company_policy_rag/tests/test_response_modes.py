from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from backend.models.api_dto import ChatRequest
from backend.models.chunk import Chunk, ChunkMetadata, ChunkRole
from backend.models.rag import Citation, QueryCategory, RAGResponse, RAGTrace, ScoredChunk
from backend.rag.context_compression import ContextCompressor
from backend.rag.citations import CitationEngine
from backend.rag.query_router import QueryRouter
from backend.rag.response_modes import RESPONSE_MODES, get_response_mode_config
from backend.rag.semantic_cache import SemanticCacheManager
from backend.services.chat_service import ChatService


@pytest.mark.parametrize("mode", ["compact", "standard", "detailed"])
def test_chat_request_accepts_each_response_mode(mode: str) -> None:
    assert ChatRequest(message="Explain hybrid retrieval", response_mode=mode).response_mode == mode


def test_chat_request_defaults_to_standard_and_rejects_unknown_mode() -> None:
    assert ChatRequest(message="Explain hybrid retrieval").response_mode == "standard"
    with pytest.raises(ValidationError):
        ChatRequest(message="Explain hybrid retrieval", response_mode="verbose")


def test_chat_api_rejects_unknown_response_mode() -> None:
    from fastapi.testclient import TestClient

    from backend.api.main import app

    response = TestClient(app).post(
        "/api/chat",
        json={"message": "Explain hybrid retrieval", "response_mode": "verbose"},
    )
    assert response.status_code == 422


def test_mode_budgets_are_strictly_ordered() -> None:
    compact = RESPONSE_MODES["compact"]
    standard = RESPONSE_MODES["standard"]
    detailed = RESPONSE_MODES["detailed"]

    assert compact.retrieval_top_k < standard.retrieval_top_k < detailed.retrieval_top_k
    assert compact.rerank_top_k < standard.rerank_top_k < detailed.rerank_top_k
    assert compact.max_context_tokens < standard.max_context_tokens < detailed.max_context_tokens
    assert compact.max_output_tokens < standard.max_output_tokens < detailed.max_output_tokens
    assert compact.max_citations < standard.max_citations < detailed.max_citations


def test_mode_config_parameterizes_one_intent_aware_strategy() -> None:
    base = QueryRouter().get_strategy_for_category(QueryCategory.COMPARISON)
    compact = get_response_mode_config("compact").apply_to(base)
    detailed = get_response_mode_config("detailed").apply_to(base)

    assert compact.dense_top_k == compact.bm25_top_k == 4
    assert compact.rerank_top_n == 3
    assert detailed.dense_top_k == detailed.bm25_top_k == 15
    assert detailed.rerank_top_n == 10
    assert compact.enable_multi_query is base.enable_multi_query is detailed.enable_multi_query
    assert compact.min_score_ratio == base.min_score_ratio == detailed.min_score_ratio


def _scored_chunk(index: int, words: int = 70) -> ScoredChunk:
    metadata = ChunkMetadata(
        document_id="doc_modes",
        source_file="modes.pdf",
        file_path="modes.pdf",
        file_hash="hash_modes",
        document_type="pdf",
        category="guidebook",
        chunk_index=index,
        page_number=index + 1,
        section_title=f"Section {index + 1}",
        chunk_strategy="recursive",
        node_role=ChunkRole.STANDALONE,
    )
    return ScoredChunk(
        chunk=Chunk(
            id=f"chunk_mode_{index}",
            text=" ".join(f"word{index}" for _ in range(words)),
            metadata=metadata,
        ),
        score=1.0 - index * 0.1,
    )


def test_context_packing_keeps_whole_chunks_and_respects_mode_budget() -> None:
    compressor = ContextCompressor()
    chunks = [_scored_chunk(index) for index in range(8)]

    compact_chunks, compact_tokens = compressor.pack_to_token_budget(
        chunks, RESPONSE_MODES["compact"].max_context_tokens
    )
    detailed_chunks, detailed_tokens = compressor.pack_to_token_budget(
        chunks, RESPONSE_MODES["detailed"].max_context_tokens
    )

    assert len(compact_chunks) < len(detailed_chunks)
    assert compact_tokens <= RESPONSE_MODES["compact"].max_context_tokens
    assert detailed_tokens <= RESPONSE_MODES["detailed"].max_context_tokens
    assert [chunk.chunk.id for chunk in compact_chunks] == [
        chunk.chunk.id for chunk in chunks[: len(compact_chunks)]
    ]
    assert all(chunk.chunk.text.endswith("word" + str(index)) for index, chunk in enumerate(chunks))


def test_citation_fallback_uses_mode_specific_coverage_caps() -> None:
    engine = CitationEngine()
    chunks = [_scored_chunk(index) for index in range(8)]

    counts = {
        mode: len(
            engine.select_citations(
                answer_text="Answer without explicit source tags.",
                generation_chunks=chunks,
                max_citations=config.max_citations,
            )
        )
        for mode, config in RESPONSE_MODES.items()
    }

    assert counts == {"compact": 2, "standard": 4, "detailed": 6}


class _ConstantEmbeddings:
    def embed_text(self, _text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


def _memory_semantic_cache() -> SemanticCacheManager:
    cache = SemanticCacheManager.__new__(SemanticCacheManager)
    cache.settings = SimpleNamespace(
        semantic_cache_enabled=True,
        semantic_cache_threshold=0.95,
        response_prompt_version="response_modes_test",
    )
    cache.vector_store = None
    cache.embedding_service = _ConstantEmbeddings()
    cache._collection = None
    cache._memory_cache = {}
    cache._lock = threading.Lock()
    return cache


def test_semantic_cache_identity_is_partitioned_by_response_mode() -> None:
    cache = _memory_semantic_cache()
    citation = Citation(
        source_index=1,
        chunk_id="chunk_1",
        document_id="doc_1",
        source_file="policy.pdf",
        snippet="Hybrid retrieval combines dense and lexical evidence.",
    )
    compact_context = '{"response_mode":"compact"}'
    detailed_context = '{"response_mode":"detailed"}'

    assert cache.put(
        "How does hybrid retrieval work?",
        "Compact answer",
        [citation],
        model_name="qwen",
        cache_context=compact_context,
    )
    assert cache.get(
        "How does hybrid retrieval work?",
        model_name="qwen",
        cache_context=compact_context,
    ) is not None
    assert cache.get(
        "How does hybrid retrieval work?",
        model_name="qwen",
        cache_context=detailed_context,
    ) is None


class _RecordingPipeline:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def query(self, **kwargs) -> RAGResponse:
        self.calls.append(kwargs)
        mode = kwargs["response_mode"]
        trace = RAGTrace(
            query=kwargs["user_query"],
            rewritten_query=kwargs["user_query"],
            response_mode=mode,
            evidence_status="DIRECT",
        )
        return RAGResponse(
            query=kwargs["user_query"],
            answer=f"{mode} answer",
            trace=trace,
            model="test-model",
        )

    def get_active_model(self) -> str:
        return "test-model"


class _NoopTelemetry:
    def record_from_rag_response(self, *_args, **_kwargs):
        return None


def test_switching_modes_in_one_conversation_is_request_scoped() -> None:
    pipeline = _RecordingPipeline()
    service = ChatService(pipeline, _NoopTelemetry())  # type: ignore[arg-type]
    session_id = "mode_switch_session"

    service.execute_query(
        ChatRequest(message="What is BM25?", session_id=session_id, response_mode="detailed")
    )
    service.execute_query(
        ChatRequest(
            message="How does it compare with embeddings?",
            session_id=session_id,
            response_mode="compact",
        )
    )

    assert [call["response_mode"] for call in pipeline.calls] == ["detailed", "compact"]
    assert pipeline.calls[1]["history"]
    user_history = [
        message for message in service._sessions[session_id] if message["role"] == "user"
    ]
    assert [message["response_mode"] for message in user_history] == ["detailed", "compact"]
