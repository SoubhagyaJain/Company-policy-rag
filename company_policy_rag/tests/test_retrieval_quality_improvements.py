"""Unit tests for retrieval-quality improvements.

Covers three self-contained changes that do not require Ollama or a real
embedding/reranker model to be present:

1. BGE asymmetric query instruction (``EmbeddingService.embed_query``).
2. Degraded-retrieval health signal (``EmbeddingService.is_using_fallback``).
3. Cross-encoder reranking on full chunk text over a widened candidate pool
   (``CrossEncoderReranker``), i.e. no lossy 350-character truncation.
"""

from __future__ import annotations

from backend.embeddings.embeddings import EmbeddingService, _default_query_instruction
from backend.models.chunk import Chunk, ChunkMetadata
from backend.models.rag import ScoredChunk
from backend.retrieval.reranker import CrossEncoderReranker


def _force_fallback_embedder(model_name: str) -> EmbeddingService:
    """Return a service that skips model loading and uses the hash fallback."""
    svc = EmbeddingService(model_name=model_name)
    svc._model = None
    svc._model_loaded = True  # mark load attempted -> deterministic fallback path
    return svc


# ── 1. BGE query instruction ────────────────────────────────────────────────

def test_bge_model_gets_query_instruction() -> None:
    assert _default_query_instruction("BAAI/bge-small-en-v1.5").startswith("Represent this sentence")


def test_non_bge_model_has_no_instruction() -> None:
    assert _default_query_instruction("nomic-embed-text") == ""
    assert _default_query_instruction("intfloat/multilingual-e5-base") == ""


def test_embed_query_applies_instruction_for_bge() -> None:
    svc = _force_fallback_embedder("BAAI/bge-small-en-v1.5")
    query = "how many vacation days do employees get"
    # The query embedding must equal embedding the instruction-prefixed text,
    # and must differ from the instruction-free (passage-style) embedding.
    assert svc.embed_query(query) == svc.embed_text(f"{svc.query_instruction}{query}")
    assert svc.embed_query(query) != svc.embed_text(query)


def test_embed_query_is_identity_without_instruction() -> None:
    svc = _force_fallback_embedder("nomic-embed-text")
    query = "how many vacation days do employees get"
    assert svc.embed_query(query) == svc.embed_text(query)


def test_embed_query_handles_empty_text() -> None:
    svc = _force_fallback_embedder("BAAI/bge-small-en-v1.5")
    assert svc.embed_query("") == [0.0] * svc.dimension


# ── 2. Degraded-retrieval health signal ─────────────────────────────────────

def test_is_using_fallback_true_when_model_unavailable() -> None:
    svc = _force_fallback_embedder("BAAI/bge-small-en-v1.5")
    assert svc.is_using_fallback is True


def test_is_using_fallback_false_before_load_attempt() -> None:
    svc = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")
    # No load attempted yet -> not (yet) a known-degraded state.
    assert svc.is_using_fallback is False


def test_is_using_fallback_false_when_model_present() -> None:
    svc = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")
    svc._model = object()  # stand-in for a loaded model
    svc._model_loaded = True
    assert svc.is_using_fallback is False


# ── 3. Reranker: full text + widened pool ───────────────────────────────────

class _RecordingCrossEncoder:
    """Fake cross-encoder that records the (query, text) pairs it scores."""

    def __init__(self) -> None:
        self.seen_pairs: list[list[str]] = []

    def predict(self, pairs, batch_size=None, show_progress_bar=False):  # noqa: D401
        self.seen_pairs = list(pairs)
        # Rank by text length so ordering is deterministic and testable.
        return [float(len(text)) for _q, text in pairs]


def _sc(idx: int, text: str) -> ScoredChunk:
    meta = ChunkMetadata(document_id="doc1", source_file="handbook.pdf", chunk_index=idx)
    return ScoredChunk(chunk=Chunk(id=f"c{idx}", text=text, metadata=meta), score=1.0 / (idx + 1))


def _reranker_with_fake_model(**kwargs) -> tuple[CrossEncoderReranker, _RecordingCrossEncoder]:
    rr = CrossEncoderReranker(**kwargs)
    fake = _RecordingCrossEncoder()
    rr._model = fake
    rr._model_loaded = True
    return rr, fake


def test_reranker_passes_full_chunk_text_not_truncated() -> None:
    long_text = "A" * 1200  # far beyond the old 350-char cut
    rr, fake = _reranker_with_fake_model(top_n=3, min_ratio=0.0)
    rr.rerank("q", [_sc(0, long_text)])
    assert fake.seen_pairs, "reranker should have scored at least one pair"
    scored_text = fake.seen_pairs[0][1]
    assert len(scored_text) == 1200, "chunk text must not be truncated before scoring"


def test_reranker_pool_wider_than_top_n() -> None:
    # 16 candidates; with top_n=3 the old pool was max(6, 8)=8. The widened
    # default pool max(top_n*4, 20) must score all 16.
    candidates = [_sc(i, f"chunk {i} " * 5) for i in range(16)]
    rr, fake = _reranker_with_fake_model(top_n=3, min_ratio=0.0)
    rr.rerank("q", candidates)
    assert len(fake.seen_pairs) == 16


def test_reranker_pool_size_override_is_respected() -> None:
    candidates = [_sc(i, f"chunk {i}") for i in range(30)]
    rr, fake = _reranker_with_fake_model(top_n=3, min_ratio=0.0, pool_size=12)
    rr.rerank("q", candidates)
    assert len(fake.seen_pairs) == 12


def test_reranker_returns_at_most_top_n() -> None:
    candidates = [_sc(i, f"chunk {i} body text") for i in range(16)]
    rr, _ = _reranker_with_fake_model(top_n=4, min_ratio=0.0)
    result = rr.rerank("q", candidates)
    assert len(result) <= 4
    assert all(sc.rerank_score is not None for sc in result)
