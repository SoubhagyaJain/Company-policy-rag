"""Retrieval experiment flags: defaults preserve behavior, each flag does one thing."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest

from backend.models.chunk import Chunk, ChunkMetadata
from backend.models.rag import RetrievalStrategy, ScoredChunk
from backend.rag import pipeline as pipeline_module
from backend.rag.pipeline import RAGPipeline, _apply_retrieval_experiment_flags
from backend.retrieval import reranker as reranker_module
from backend.retrieval.bm25 import BM25SearchIndex
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.reranker import CrossEncoderReranker
from src.config import settings


def _chunk(cid: str, text: str, **meta) -> Chunk:
    return Chunk(
        id=cid,
        text=text,
        metadata=ChunkMetadata(document_id="doc-1", source_file="Guide Book.pdf", **meta),
    )


def _sc(cid: str, score: float, text: str = "body text here") -> ScoredChunk:
    return ScoredChunk(chunk=_chunk(cid, text), score=score)


# ── BM25 ────────────────────────────────────────────────────────────────────


def _bm25(tmp_path, **kwargs) -> BM25SearchIndex:
    index = BM25SearchIndex(storage_dir=str(tmp_path / "bm25"), **kwargs)
    index.build_index(
        [
            _chunk("plan", "The planning pattern breaks a task into steps."),
            _chunk("tools", "Tools let an agent call external APIs."),
            _chunk("memory", "Memory keeps context across turns."),
        ]
    )
    return index


def test_bm25_does_not_pad_results_with_zero_score_chunks(tmp_path) -> None:
    index = _bm25(tmp_path)
    assert index.search("quantum entanglement", top_k=3) == []
    hits = index.search("planning", top_k=3)
    assert [h.chunk.id for h in hits] == ["plan"]


def test_bm25_keeps_matching_chunks_whose_idf_is_not_positive(tmp_path) -> None:
    # In a one-document corpus every term's IDF is negative, so a real match
    # scores <= 0. It must still be returned; only chunks with no shared term go.
    index = BM25SearchIndex(storage_dir=str(tmp_path / "bm25"), metadata_fields="")
    index.build_index([_chunk("only", "notice period policy")])
    assert [h.chunk.id for h in index.search("notice", top_k=3)] == ["only"]
    assert index.search("vacation", top_k=3) == []


def test_bm25_zero_score_padding_can_be_restored(tmp_path) -> None:
    index = _bm25(tmp_path, skip_zero_scores=False)
    assert len(index.search("quantum entanglement", top_k=3)) == 3


def test_bm25_passes_k1_and_b_to_scorer(tmp_path) -> None:
    index = _bm25(tmp_path, k1=0.9, b=0.4)
    assert index._bm25.k1 == 0.9
    assert index._bm25.b == 0.4


def test_bm25_stemming_matches_inflections(tmp_path) -> None:
    plain = _bm25(tmp_path)
    stemmed = _bm25(tmp_path, stemming=True)
    assert plain.search("plans", top_k=3) == []
    assert [h.chunk.id for h in stemmed.search("plans", top_k=3)] == ["plan"]


def test_bm25_metadata_fields_control_indexed_text(tmp_path) -> None:
    with_filename = _bm25(tmp_path)
    text_only = _bm25(tmp_path, metadata_fields="")
    # "guide" only occurs in the source filename.
    assert len(with_filename.search("guide", top_k=3)) == 3
    assert text_only.search("guide", top_k=3) == []


def test_bm25_reconfigure_retokenizes_only_when_tokenization_changes(tmp_path) -> None:
    index = _bm25(tmp_path)
    tokens_before = index._tokenized_corpus
    index.reconfigure(k1=1.2, b=0.5)
    assert index._tokenized_corpus is tokens_before
    index.reconfigure(stemming=True)
    assert index._tokenized_corpus is not tokens_before
    assert [h.chunk.id for h in index.search("plans", top_k=3)] == ["plan"]


def test_bm25_load_reuses_legacy_pickle_and_retokenizes_on_config_change(tmp_path) -> None:
    index = _bm25(tmp_path)
    index.save()
    # Simulate a pickle written before tokenizer_config existed.
    pkl = tmp_path / "bm25" / "index.pkl"
    pkl.write_bytes(pickle.dumps({"tokenized_corpus": index._tokenized_corpus}))

    legacy = BM25SearchIndex(storage_dir=str(tmp_path / "bm25"))
    assert legacy.load()
    assert legacy._tokenized_corpus == index._tokenized_corpus

    stemmed = BM25SearchIndex(storage_dir=str(tmp_path / "bm25"), stemming=True)
    assert stemmed.load()
    assert [h.chunk.id for h in stemmed.search("plans", top_k=3)] == ["plan"]


# ── Hybrid retriever ────────────────────────────────────────────────────────


class _ListRetriever:
    def __init__(self, hits: list[ScoredChunk]) -> None:
        self.hits = hits

    def retrieve(self, query, top_k=25, filters=None):
        return self.hits[:top_k]

    def search(self, query, top_k=25, filters=None):
        return self.hits[:top_k]


def test_hybrid_single_list_is_scored_on_rrf_scale() -> None:
    dense_hits = [_sc("a", 0.83), _sc("b", 0.71)]
    for hit in dense_hits:
        hit.dense_score = hit.score
    dense = _ListRetriever(dense_hits)
    hybrid = HybridRetriever(dense, _ListRetriever([]), reranker=object(), rrf_k=60)
    result = hybrid.retrieve("q", dense_top_k=2, bm25_top_k=2)
    assert [r.chunk.id for r in result] == ["a", "b"]
    assert result[0].score == pytest.approx(1 / 61)
    assert result[0].dense_score == pytest.approx(0.83)


def test_hybrid_trace_records_each_index_ranking() -> None:
    dense = _ListRetriever([_sc("a", 0.9), _sc("b", 0.8)])
    sparse = _ListRetriever([_sc("b", 7.0), _sc("c", 3.0)])
    hybrid = HybridRetriever(dense, sparse, reranker=object())
    trace: dict = {}
    hybrid.retrieve("q", dense_top_k=2, bm25_top_k=2, trace=trace)
    assert trace["dense"] == ["a", "b"]
    assert trace["bm25"] == ["b", "c"]
    assert trace["fused"][0] == "b"
    assert "dense_ms" in trace and "bm25_ms" in trace


def test_hybrid_min_chunk_words_skips_short_chunks_and_refills_depth() -> None:
    dense = _ListRetriever(
        [_sc("title", 0.9, "AI AGENTS"), _sc("body1", 0.8, "one two three four five six"), _sc("body2", 0.7, "a b c d e f g")]
    )
    hybrid = HybridRetriever(dense, _ListRetriever([]), reranker=object(), min_chunk_words=5)
    result = hybrid.retrieve("q", dense_top_k=2, bm25_top_k=2)
    assert [r.chunk.id for r in result] == ["body1", "body2"]


# ── Reranker ────────────────────────────────────────────────────────────────


class _FakeCrossEncoder:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls = 0

    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        self.calls += 1
        return [self.scores[text] for _, text in pairs]


def _reranker(scores: dict[str, float], **kwargs) -> CrossEncoderReranker:
    rr = CrossEncoderReranker(**kwargs)
    rr._model = _FakeCrossEncoder(scores)
    rr._model_loaded = True
    return rr


def _candidates(n: int) -> list[ScoredChunk]:
    return [_sc(f"c{i}", 1.0 / (i + 1), text=f"t{i}") for i in range(n)]


def test_reranker_filter_can_be_disabled() -> None:
    scores = {"t0": 0.9, "t1": 0.1, "t2": 0.05}
    on = _reranker(scores, top_n=3, min_ratio=0.45)
    off = _reranker(scores, top_n=3, min_ratio=0.45, score_filter_enabled=False)
    assert [c.chunk.id for c in on.rerank("q", _candidates(3))] == ["c0"]
    assert [c.chunk.id for c in off.rerank("q", _candidates(3))] == ["c0", "c1", "c2"]


def test_reranker_honors_min_keep() -> None:
    scores = {"t0": 0.9, "t1": 0.1, "t2": 0.05}
    rr = _reranker(scores, top_n=3, min_ratio=0.45, min_keep=2)
    assert [c.chunk.id for c in rr.rerank("q", _candidates(3))] == ["c0", "c1"]


@pytest.mark.parametrize(
    ("pool_size", "expected"),
    [(None, 20), (0, 20), (-1, 30), (12, 12), (40, 30)],
)
def test_reranker_pool_size_modes(pool_size, expected) -> None:
    rr = CrossEncoderReranker(top_n=3, pool_size=pool_size)
    assert rr.resolve_pool_size(3, 30) == expected


def test_reranker_trace_records_prefilter_and_postfilter() -> None:
    scores = {"t0": 0.2, "t1": 0.9, "t2": 0.01}
    rr = _reranker(scores, top_n=3, min_ratio=0.45)
    trace: dict = {}
    rr.rerank("q", _candidates(3), trace=trace)
    assert trace["pool"] == ["c0", "c1", "c2"]
    assert trace["prefilter"] == ["c1", "c0", "c2"]
    assert trace["postfilter"] == ["c1"]
    assert trace["scores"]["c1"] == pytest.approx(0.9)
    assert trace["model_ms"] >= 0


def test_shared_reranker_models_are_keyed_by_model_name(monkeypatch) -> None:
    base, large = object(), object()
    monkeypatch.setattr(
        reranker_module,
        "_shared_reranker_models",
        {("m-base", "cpu", 512): base, ("m-large", "cpu", 512): large},
    )
    a = CrossEncoderReranker(model_name="m-base", device="cpu")
    b = CrossEncoderReranker(model_name="m-large", device="cpu")
    a._init_model()
    b._init_model()
    assert a._model is base
    assert b._model is large


# ── Pipeline flags ──────────────────────────────────────────────────────────


def _strategy(**overrides) -> RetrievalStrategy:
    values = dict(dense_top_k=8, bm25_top_k=8, rrf_k=60, rerank_top_n=6, min_score_ratio=0.45)
    values.update(overrides)
    return RetrievalStrategy(**values)


def test_experiment_flags_default_to_no_change() -> None:
    router = _strategy(dense_top_k=30, bm25_top_k=30)
    strategy = _strategy()
    _apply_retrieval_experiment_flags(router, strategy)
    assert strategy == _strategy()


def test_experiment_flags_depth_mode_ratio_and_rrf(monkeypatch) -> None:
    monkeypatch.setattr(settings, "retrieval_depth_mode", "max")
    monkeypatch.setattr(settings, "hybrid_rrf_k", 20)
    monkeypatch.setattr(settings, "rerank_score_ratio_override", 0.25)
    strategy = _strategy()
    _apply_retrieval_experiment_flags(_strategy(dense_top_k=30, bm25_top_k=30), strategy)
    assert (strategy.dense_top_k, strategy.bm25_top_k) == (30, 30)
    assert strategy.rrf_k == 20
    assert strategy.min_score_ratio == 0.25

    monkeypatch.setattr(settings, "retrieval_depth_override", 15)
    _apply_retrieval_experiment_flags(_strategy(dense_top_k=30, bm25_top_k=30), strategy)
    assert (strategy.dense_top_k, strategy.bm25_top_k) == (15, 15)


def _gather_stub(per_query: dict[str, list[ScoredChunk]]) -> RAGPipeline:
    pipe = RAGPipeline.__new__(RAGPipeline)

    class _HR:
        supports_stage_trace = True

        def retrieve(self, query, dense_top_k=25, bm25_top_k=25, filters=None, rrf_k=60, trace=None):
            if trace is not None:
                trace.update({"query": query, "fused": [sc.chunk.id for sc in per_query[query]]})
            return per_query[query]

    pipe.hybrid_retriever = _HR()
    return pipe


def test_subquery_merge_default_keeps_max_raw_score() -> None:
    pipe = _gather_stub({"q1": [_sc("a", 0.5), _sc("b", 0.4)], "q2": [_sc("b", 0.9)]})
    cands, _ = pipe._gather_hybrid_candidates(["q1", "q2"], None, _strategy())
    assert {c.chunk.id: c.score for c in cands}["b"] == 0.9


def test_subquery_merge_rrf_fuses_ranks_and_records_trace(monkeypatch) -> None:
    monkeypatch.setattr(settings, "subquery_merge_mode", "rrf")
    # "z" has a huge raw score from a single-list query but is ranked once;
    # "b" is ranked in both lists and must win under RRF.
    pipe = _gather_stub(
        {"q1": [_sc("a", 0.03), _sc("b", 0.02)], "q2": [_sc("z", 12.0), _sc("b", 11.0)]}
    )
    trace: list = []
    cands, _ = pipe._gather_hybrid_candidates(["q1", "q2"], None, _strategy(), stage_trace=trace)
    ranked = sorted(cands, key=lambda c: c.score, reverse=True)
    assert ranked[0].chunk.id == "b"
    assert [t["query"] for t in trace] == ["q1", "q2"]


def test_disabled_reranker_keeps_retrieval_order(monkeypatch) -> None:
    monkeypatch.setattr(settings, "enable_reranker", False)

    class _ExplodingReranker:
        def rerank(self, *args, **kwargs):
            raise AssertionError("reranker must not run when ENABLE_RERANKER=false")

    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe.reranker = _ExplodingReranker()
    candidates = [_sc("a", 0.03), _sc("b", 0.02), _sc("c", 0.01)]

    # Stop right after the rerank stage: the governing-clause selector is the
    # next collaborator and is irrelevant to this flag.
    class _Stop(Exception):
        pass

    class _Selector:
        def select(self, *args, **kwargs):
            raise _Stop

    pipe.governing_clause_selector = _Selector()
    thinking = SimpleNamespace(
        start_stage=lambda *a, **k: None,
        complete_stage=lambda *a, **k: None,
        degrade_stage=lambda *a, **k: None,
    )
    ctx = pipeline_module.QueryContext(
        user_query="q",
        thinking_sm=thinking,
        current_strategy=_strategy(rerank_top_n=2),
        candidate_chunks=candidates,
        rewrite_res=SimpleNamespace(rewritten_query="q"),
    )
    with pytest.raises(_Stop):
        pipe._stage_rerank_and_context(ctx, "")
    assert ctx.retrieval_stages["reranker_enabled"] is False
    assert ctx.retrieval_stages["post_rerank"] == ["a", "b"]
