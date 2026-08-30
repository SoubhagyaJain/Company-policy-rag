from __future__ import annotations

import pytest

from backend.models.chunk import Chunk, ChunkMetadata
from backend.models.rag import Citation
from backend.rag.pipeline import RAGPipeline, _format_history_for_prompt
from backend.rag.semantic_cache import SemanticCacheManager


class _FailingRetriever:
    def retrieve(self, *args, **kwargs):  # pragma: no cover - failure is the assertion
        raise AssertionError("document retrieval must not run in general chat mode")


class _GeneralLLM:
    model = "test-general"

    def complete(self, prompt: str, **kwargs):
        assert "General chat mode" in prompt
        return "A direct general answer."


class _StreamingGeneralLLM:
    model = "test-general-stream"

    def complete(self, prompt: str, **kwargs):  # pragma: no cover - assertion path
        raise AssertionError("streaming mode must not wait for complete()")

    def stream_complete(self, prompt: str, **kwargs):
        assert "General chat mode" in prompt
        for delta in ("Fast ", "general ", "answer."):
            yield type("CompletionDelta", (), {"delta": delta})()


class _CountingEmbedding:
    def __init__(self) -> None:
        self.calls = 0

    def embed_text(self, text: str) -> list[float]:
        self.calls += 1
        return [1.0, 0.0, 0.0]


def _page_chunk(
    chunk_id: str,
    *,
    physical: int,
    display: int,
    document_id: str = "doc-1",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=f"Facts on printed page {display}",
        metadata=ChunkMetadata(
            document_id=document_id,
            source_file="handbook.pdf",
            page_number=physical,
            internal_page_index=physical - 1,
            display_page_number=display,
            page_label=str(display),
        ),
    )


def test_general_chat_control_is_removed_from_metadata_filters() -> None:
    metadata, mode = RAGPipeline._split_control_filters(
        {"chat_mode": "general", "category": "Finance"}
    )
    assert mode == "general"
    assert metadata == {"category": "Finance"}


def test_general_chat_bypasses_document_retrieval() -> None:
    pipeline = RAGPipeline(hybrid_retriever=_FailingRetriever(), llm=_GeneralLLM())
    response = pipeline.query(
        "Explain why the sky looks blue",
        filters={"chat_mode": "general"},
        history=[{"role": "user", "content": "Let us discuss science."}],
    )

    assert response.answer == "A direct general answer."
    assert response.citations == []
    assert response.trace.retrieval_strategy == "general_chat_bypass"
    assert response.trace.retrieved_candidate_count == 0


@pytest.mark.asyncio
async def test_general_chat_stream_uses_real_model_deltas() -> None:
    pipeline = RAGPipeline(
        hybrid_retriever=_FailingRetriever(),
        llm=_StreamingGeneralLLM(),
    )

    events = [
        event
        async for event in pipeline.stream_query(
            "Give me a quick explanation",
            filters={"chat_mode": "general"},
        )
    ]

    token_events = [event["content"] for event in events if event["type"] == "token"]
    done = next(event for event in events if event["type"] == "done")
    retrieval_index = next(i for i, event in enumerate(events) if event["type"] == "retrieval_done")
    first_token_index = next(i for i, event in enumerate(events) if event["type"] == "token")

    assert "".join(token_events) == "Fast general answer."
    assert done["answer"] == "Fast general answer."
    assert done["trace"].retrieval_strategy == "general_chat_bypass"
    assert retrieval_index < first_token_index


def test_printed_page_resolves_to_physical_pdf_sheet() -> None:
    chunk = _page_chunk("page-74", physical=74, display=73)
    physical_only = Chunk(
        id="page-12",
        text="Legacy page metadata",
        metadata=ChunkMetadata(
            document_id="doc-2",
            source_file="legacy.pdf",
            page_number=12,
            internal_page_index=11,
        ),
    )
    pipeline = object.__new__(RAGPipeline)
    pipeline.docstore = {chunk.id: chunk, physical_only.id: physical_only}

    resolved = pipeline._resolve_page_number_filter(
        73,
        active_document_id="doc-1",
        active_document_name=None,
        allowed_document_ids=["doc-1"],
    )

    assert resolved == 74
    assert pipeline._resolve_page_number_filter(
        12,
        active_document_id="doc-2",
        active_document_name=None,
        allowed_document_ids=["doc-2"],
    ) == 12
    assert pipeline._resolve_page_number_filter(
        999,
        active_document_id="doc-1",
        active_document_name=None,
        allowed_document_ids=["doc-1"],
    ) is None


def test_history_prompt_is_bounded_and_keeps_newest_turn() -> None:
    history = [
        {"role": "user", "content": "old " * 200},
        {"role": "assistant", "content": "middle " * 200},
        {"role": "user", "content": "newest fact"},
    ]
    formatted = _format_history_for_prompt(history, max_turns=3, max_chars=120)

    assert len(formatted) <= 170  # heading/newlines plus the bounded content budget
    assert "newest fact" in formatted


def test_exact_semantic_cache_hit_skips_second_embedding(tmp_path) -> None:
    embedding = _CountingEmbedding()
    cache = SemanticCacheManager(
        collection_name="exact_fast_path",
        persist_dir=tmp_path / "chroma",
        embedding_service=embedding,
    )
    cache.clear()
    cache._collection = None
    citation = Citation(
        source_index=1,
        chunk_id="chunk-1",
        document_id="doc-1",
        source_file="handbook.pdf",
        snippet="Verified fact",
    )

    assert cache.put("What is the PTO limit?", "Twenty days.", [citation])
    assert embedding.calls == 1
    hit = cache.get("  what IS the PTO limit? ")

    assert hit is not None
    assert hit.answer == "Twenty days."
    assert hit.similarity_score == 1.0
    assert embedding.calls == 1
