"""Streaming verification gate.

High-risk (policy / numeric) answers must be buffered and verified before the
user sees them: no tokens are streamed live during generation, and the request
keeps a retry budget. Low-risk factual answers keep true token streaming.
"""

from __future__ import annotations

import pytest

from backend.models.chunk import Chunk, ChunkMetadata, ChunkRole
from backend.rag.pipeline import RAGPipeline, _is_high_risk_query
from backend.retrieval.bm25 import BM25SearchIndex
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.vector import DenseVectorRetriever
from backend.embeddings.embeddings import EmbeddingService
from backend.embeddings.vector_store import ChromaVectorStore


# ── 1. Risk classifier (pure) ───────────────────────────────────────────────

@pytest.mark.parametrize(
    "query",
    [
        "If I finish at 9 pm, when does my break end?",   # explicit time
        "Can I expense a $5,000 laptop stand?",           # explicit amount
        "How many vacation days do I accrue per month?",  # calculation/entitlement
    ],
)
def test_high_risk_queries_flagged(query: str) -> None:
    assert _is_high_risk_query(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "What is the resignation policy?",
        "Summarize the health benefits section.",
        "Who approves remote work?",
    ],
)
def test_low_risk_queries_not_flagged(query: str) -> None:
    assert _is_high_risk_query(query) is False


# ── 2. End-to-end emit gating ───────────────────────────────────────────────

class _StreamingLLM:
    """Fake LLM exposing both complete() and stream_complete()."""

    def __init__(self) -> None:
        self.model = "qwen2.5:7b"
        self.complete_calls = 0
        self.stream_calls = 0
        self.prompts: list[str] = []

    def complete(self, prompt: str, **kwargs) -> str:
        self.complete_calls += 1
        self.prompts.append(prompt)
        return "Employees must give two weeks written notice [Source 1]."

    def stream_complete(self, prompt: str, **kwargs):
        self.stream_calls += 1
        for word in ["Employees", " must", " give", " two", " weeks", " notice", " [Source 1]."]:
            yield word


@pytest.fixture
def pipeline_and_llm(tmp_path):
    meta = ChunkMetadata(
        document_id="doc_001",
        source_file="employee_handbook.pdf",
        category="policy",
        chunk_index=0,
        page_number=3,
        section_title="Resignation Policy",
        node_role=ChunkRole.STANDALONE,
    )
    chunks = [
        Chunk(
            id="chunk_001",
            text=(
                "Employees giving notice of resignation must submit two weeks notice "
                "in writing to HR. Vacation accrues at one day per month."
            ),
            metadata=meta,
        )
    ]
    vstore = ChromaVectorStore(collection_name="stream_gate_test", persist_dir=str(tmp_path / "chroma"))
    vstore.add_chunks(chunks)
    embed = EmbeddingService(cache_enabled=True)
    dense = DenseVectorRetriever(vector_store=vstore, embedding_service=embed)
    bm25 = BM25SearchIndex(storage_dir=str(tmp_path / "bm25"))
    bm25.build_index(chunks)
    hybrid = HybridRetriever(dense_retriever=dense, bm25_index=bm25)

    llm = _StreamingLLM()
    pipeline = RAGPipeline(hybrid_retriever=hybrid, docstore={c.id: c for c in chunks}, llm=llm)
    return pipeline, llm


def test_high_risk_answer_is_buffered_not_streamed_live(pipeline_and_llm) -> None:
    pipeline, llm = pipeline_and_llm
    deltas: list[str] = []
    pipeline.query(
        user_query="How many vacation days do I accrue and when must I give notice by 5 pm?",
        stream_callback=deltas.append,
    )
    # High-risk: no live token streaming; the answer is produced via buffered
    # complete() so verification can gate it before the user sees anything.
    assert llm.stream_calls == 0
    assert llm.complete_calls >= 1
    # Nothing (or at most a single already-verified block) is emitted mid-loop;
    # multi-delta live streaming must not have happened.
    assert len(deltas) <= 1


def test_low_risk_answer_streams_live(pipeline_and_llm) -> None:
    pipeline, llm = pipeline_and_llm
    deltas: list[str] = []
    pipeline.query(
        user_query="What is the resignation notice policy?",
        stream_callback=deltas.append,
    )
    # Low-risk: true token streaming via stream_complete().
    assert llm.stream_calls >= 1
    assert len(deltas) > 1


def _issued_llm_faithfulness_audit(llm: "_StreamingLLM") -> bool:
    """True if the LLM was asked to run the grounding-audit verification prompt."""
    return any("grounding auditor" in p.lower() for p in llm.prompts)


def test_high_risk_query_invokes_llm_faithfulness_judge(pipeline_and_llm) -> None:
    pipeline, llm = pipeline_and_llm
    pipeline.query(user_query="How many vacation days do I accrue per month by 5 pm?")
    assert _issued_llm_faithfulness_audit(llm), "high-risk answer should be LLM-audited"


def test_low_risk_query_does_not_invoke_llm_faithfulness_judge(pipeline_and_llm) -> None:
    pipeline, llm = pipeline_and_llm
    pipeline.query(user_query="What is the resignation notice policy?")
    assert not _issued_llm_faithfulness_audit(llm), "low-risk answer must skip the LLM audit"
