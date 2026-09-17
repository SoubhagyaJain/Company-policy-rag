"""Regression tests for the two measured downstream evidence losses.

1. Scope: a query that says "the guidebook" / "this document" with no active
   document got CURRENT_DOCUMENT scope with no identity, and scope enforcement
   rejected every retrieved candidate (SCOPE_UNBOUND_REFERENCE_MODE).
2. Context assembly: governing-clause selection replaced the ranked hand-off
   with picks from the whole candidate pool, discarding the rank-1 chunk
   (CONTEXT_ASSEMBLY_MODE / CONTEXT_RANK_ANCHOR_K).
"""

from __future__ import annotations

import pytest

from backend.models.chunk import Chunk, ChunkMetadata
from backend.models.rag import ScoredChunk
from backend.rag.pipeline import RAGPipeline
from backend.rag.policy_reliability import GoverningClauseSelector, merge_governing_context
from backend.rag.scope_resolver import DocumentRetrievalScope, DocumentScopeResolver
from src.config import Settings, settings

GUIDE_ID = "doc_guide"
HAND_ID = "doc_hand"
GUIDE_FILE = "AI Agents guidebook.pdf"
HAND_FILE = "Employee Handbook.pdf"
ONE_DOC = {GUIDE_ID: GUIDE_FILE}
TWO_DOCS = {GUIDE_ID: GUIDE_FILE, HAND_ID: HAND_FILE}


def test_measured_fixes_are_the_defaults() -> None:
    assert Settings.model_fields["scope_unbound_reference_mode"].default == "resolve"
    assert Settings.model_fields["context_assembly_mode"].default == "rank_anchor"
    assert Settings.model_fields["min_chunk_words"].default == 5


# ── Scope resolver ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["strict", "resolve"])
def test_active_document_wins_over_a_named_reference(mode: str) -> None:
    # The query names "the guidebook" but the user has the handbook open:
    # the active document binds and no other document may leak in.
    decision = DocumentScopeResolver(mode).resolve_scope(
        "What does the guidebook say about leave?",
        active_document_id=HAND_ID,
        active_document_name=HAND_FILE,
        known_documents=TWO_DOCS,
    )
    assert decision.scope == DocumentRetrievalScope.CURRENT_DOCUMENT
    assert decision.active_document_id == HAND_ID
    assert decision.allowed_document_ids == [HAND_ID]


@pytest.mark.parametrize("mode", ["strict", "resolve"])
def test_explicit_filename_reference_binds_in_both_modes(mode: str) -> None:
    decision = DocumentScopeResolver(mode).resolve_scope(
        "What does Employee Handbook.pdf say about leave?", known_documents=TWO_DOCS
    )
    assert decision.scope == DocumentRetrievalScope.CURRENT_DOCUMENT
    assert decision.active_document_id == HAND_ID
    assert decision.allowed_document_ids == [HAND_ID]


def test_strict_mode_keeps_legacy_scope_without_identity() -> None:
    decision = DocumentScopeResolver("strict").resolve_scope(
        "How many building blocks does the guidebook describe?", known_documents=ONE_DOC
    )
    assert decision.scope == DocumentRetrievalScope.CURRENT_DOCUMENT
    assert decision.active_document_id is None
    assert decision.allowed_document_ids == []


@pytest.mark.parametrize(
    "query",
    [
        "How many building blocks does the guidebook describe?",
        "Summarize this document",
        "What is in the doc?",
    ],
)
def test_resolve_binds_implicit_reference_to_the_only_indexed_document(query: str) -> None:
    decision = DocumentScopeResolver("resolve").resolve_scope(query, known_documents=ONE_DOC)
    assert decision.scope == DocumentRetrievalScope.CURRENT_DOCUMENT
    assert decision.active_document_id == GUIDE_ID
    assert decision.active_document_name == GUIDE_FILE
    assert decision.allowed_document_ids == [GUIDE_ID]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What does the guidebook say about memory?", GUIDE_ID),
        ("What does the handbook say about leave?", HAND_ID),
    ],
)
def test_resolve_binds_named_noun_to_the_matching_document(query: str, expected: str) -> None:
    decision = DocumentScopeResolver("resolve").resolve_scope(query, known_documents=TWO_DOCS)
    assert decision.scope == DocumentRetrievalScope.CURRENT_DOCUMENT
    assert decision.allowed_document_ids == [expected]


@pytest.mark.parametrize(
    ("query", "known"),
    [
        ("Summarize this document", TWO_DOCS),
        ("What does the guidebook say?", {"a": "guidebook v1.pdf", "b": "guidebook v2.pdf"}),
        ("What does the guidebook say?", None),
    ],
)
def test_resolve_searches_globally_when_no_single_document_matches(query: str, known) -> None:
    decision = DocumentScopeResolver("resolve").resolve_scope(query, known_documents=known)
    assert decision.scope == DocumentRetrievalScope.GLOBAL
    assert decision.active_document_id is None
    assert decision.allowed_document_ids == []


def test_resolve_keeps_page_reference_on_the_bound_document() -> None:
    decision = DocumentScopeResolver("resolve").resolve_scope(
        "What is on page 5 of the guidebook?", known_documents=ONE_DOC
    )
    assert decision.active_document_id == GUIDE_ID
    assert decision.page_number == 5


def test_resolver_reads_the_setting_when_no_mode_is_given(monkeypatch) -> None:
    query = "What does the guidebook say about memory?"
    monkeypatch.setattr(settings, "scope_unbound_reference_mode", "strict")
    assert DocumentScopeResolver().resolve_scope(query, known_documents=ONE_DOC).active_document_id is None
    monkeypatch.setattr(settings, "scope_unbound_reference_mode", "resolve")
    assert DocumentScopeResolver().resolve_scope(query, known_documents=ONE_DOC).active_document_id == GUIDE_ID


# ── Pipeline helpers ────────────────────────────────────────────────────────


def _chunk(cid: str, text: str, document_id: str = GUIDE_ID, source_file: str = GUIDE_FILE) -> Chunk:
    return Chunk(
        id=cid,
        text=text,
        metadata=ChunkMetadata(document_id=document_id, source_file=source_file, page_number=1),
    )


class _Retriever:
    """Returns fixed hits; honors a document_id filter unless ``rogue``."""

    def __init__(self, chunks: list[Chunk], scores: dict[str, float], rogue: bool = False) -> None:
        self.chunks = chunks
        self.scores = scores
        self.rogue = rogue
        self.filters_seen: list[dict | None] = []

    def retrieve(self, query, dense_top_k=8, bm25_top_k=8, filters=None, rrf_k=60):
        self.filters_seen.append(filters)
        allowed = (filters or {}).get("document_id")
        hits = [
            ScoredChunk(chunk=c, score=self.scores[c.id])
            for c in self.chunks
            if self.rogue or not allowed or c.metadata.document_id == allowed
        ]
        return sorted(hits, key=lambda sc: sc.score, reverse=True)


class _FixedOrderReranker:
    """Cross-encoder stand-in: ranks candidates by a fixed relevance map."""

    def __init__(self, relevance: dict[str, float]) -> None:
        self.relevance = relevance

    def rerank(self, query, candidates, top_n=None, min_ratio=None):
        scored = [
            sc.model_copy(update={"rerank_score": self.relevance.get(sc.chunk.id, 0.0)}) for sc in candidates
        ]
        scored.sort(key=lambda sc: sc.rerank_score, reverse=True)
        return scored[: top_n or len(scored)]


@pytest.fixture
def quiet_settings(monkeypatch):
    for key, value in {
        "retrieval_cache_enabled": False,
        "vision_enabled": False,
        "enable_lazy_vision_fallback": False,
        "enable_llm_multi_query": False,
        "enable_reranker": True,
    }.items():
        monkeypatch.setattr(settings, key, value)
    return monkeypatch


def _pipeline(retriever, reranker, docstore: dict[str, Chunk], scope_mode: str | None = None) -> RAGPipeline:
    return RAGPipeline(
        hybrid_retriever=retriever,
        reranker=reranker,
        docstore=docstore,
        llm=None,
        scope_resolver=DocumentScopeResolver(scope_mode),
    )


def _ids(chunks: list[ScoredChunk]) -> list[str]:
    return [sc.chunk.id for sc in chunks]


# ── Scope through the pipeline ──────────────────────────────────────────────

_GUIDE_CHUNKS = [
    _chunk("g-memory", "Agent memory stores context between turns so the agent can recall earlier steps."),
    _chunk("g-tools", "Tools let an agent call external APIs and read their results."),
]
_HAND_CHUNK = _chunk(
    "h-memory", "Memory allowance for staff laptops covers upgrades to agent workstations.", HAND_ID, HAND_FILE
)
_SCORES = {"g-memory": 0.9, "g-tools": 0.5, "h-memory": 0.95}
_QUERY = "How does the guidebook describe agent memory?"


def test_single_document_reference_keeps_candidates_only_in_resolve_mode(quiet_settings) -> None:
    docstore = {c.id: c for c in _GUIDE_CHUNKS}
    reranker = _FixedOrderReranker({"g-memory": 0.9, "g-tools": 0.2})

    strict = _pipeline(_Retriever(_GUIDE_CHUNKS, _SCORES), reranker, docstore, "strict")
    ctx = strict.run_retrieval_stages(_QUERY)
    assert ctx.candidate_chunks == []  # the measured failure: every candidate rejected

    resolved = _pipeline(_Retriever(_GUIDE_CHUNKS, _SCORES), reranker, docstore, "resolve")
    ctx = resolved.run_retrieval_stages(_QUERY)
    assert ctx.scope_decision.active_document_id == GUIDE_ID
    assert "g-memory" in _ids(ctx.expanded_chunks)
    assert {sc.chunk.metadata.document_id for sc in ctx.expanded_chunks} == {GUIDE_ID}


def test_named_reference_never_leaks_another_document(quiet_settings) -> None:
    chunks = [*_GUIDE_CHUNKS, _HAND_CHUNK]
    docstore = {c.id: c for c in chunks}
    # The retriever ignores filters and ranks the handbook chunk first.
    retriever = _Retriever(chunks, _SCORES, rogue=True)
    pipe = _pipeline(retriever, _FixedOrderReranker({"h-memory": 0.99, "g-memory": 0.9}), docstore, "resolve")
    ctx = pipe.run_retrieval_stages(_QUERY)
    assert ctx.scope_decision.allowed_document_ids == [GUIDE_ID]
    assert any((f or {}).get("document_id") == GUIDE_ID for f in retriever.filters_seen)
    assert "h-memory" not in _ids(ctx.expanded_chunks)
    assert {sc.chunk.metadata.document_id for sc in ctx.expanded_chunks} == {GUIDE_ID}
    assert ctx.cross_document_count >= 1


def test_active_document_filter_still_binds_in_resolve_mode(quiet_settings) -> None:
    chunks = [*_GUIDE_CHUNKS, _HAND_CHUNK]
    docstore = {c.id: c for c in chunks}
    pipe = _pipeline(
        _Retriever(chunks, _SCORES, rogue=True), _FixedOrderReranker({"g-memory": 0.99}), docstore, "resolve"
    )
    ctx = pipe.run_retrieval_stages(_QUERY, filters={"document_id": HAND_ID})
    assert ctx.scope_decision.allowed_document_ids == [HAND_ID]
    assert _ids(ctx.expanded_chunks) == ["h-memory"]


def test_ambiguous_reference_searches_all_documents_instead_of_returning_nothing(quiet_settings) -> None:
    chunks = [*_GUIDE_CHUNKS, _HAND_CHUNK]
    docstore = {c.id: c for c in chunks}
    reranker = _FixedOrderReranker({"g-memory": 0.9, "h-memory": 0.5})
    query = "How does this document describe agent memory?"

    strict = _pipeline(_Retriever(chunks, _SCORES), reranker, docstore, "strict")
    assert strict.run_retrieval_stages(query).candidate_chunks == []

    resolved = _pipeline(_Retriever(chunks, _SCORES), reranker, docstore, "resolve")
    ctx = resolved.run_retrieval_stages(query)
    assert ctx.scope_decision.scope == DocumentRetrievalScope.GLOBAL
    assert "g-memory" in _ids(ctx.expanded_chunks)


# ── Context assembly ────────────────────────────────────────────────────────


def _sc(cid: str, text: str = "text", score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(chunk=_chunk(cid, text), score=score, rerank_score=score)


def test_merge_governing_mode_returns_selector_order_unchanged() -> None:
    ranked = [_sc("r1"), _sc("r2")]
    governing = [_sc("p"), _sc("x1"), _sc("x2")]
    assert _ids(merge_governing_context(ranked, governing, max_chunks=6)) == ["p", "x1", "x2"]
    assert _ids(merge_governing_context(ranked, governing, max_chunks=6, mode="rank_anchor", anchor_k=0)) == [
        "p",
        "x1",
        "x2",
    ]


def test_rank_anchor_keeps_primary_first_then_top_ranked_then_selector_picks() -> None:
    ranked = [_sc(f"r{i}") for i in range(1, 7)]
    governing = [_sc("p"), *[_sc(f"x{i}") for i in range(1, 6)]]
    merged = merge_governing_context(ranked, governing, max_chunks=6, mode="rank_anchor", anchor_k=2)
    assert _ids(merged) == ["p", "r1", "r2", "x1", "x2", "x3"]


def test_rank_anchor_dedupes_when_the_primary_is_a_top_ranked_chunk() -> None:
    ranked = [_sc("r1"), _sc("r2"), _sc("r3")]
    governing = [_sc("r1"), _sc("x1"), _sc("r3")]
    merged = merge_governing_context(ranked, governing, max_chunks=6, mode="rank_anchor", anchor_k=2)
    assert _ids(merged) == ["r1", "r2", "x1", "r3"]


def test_rank_anchor_preserves_governing_clause_primary_and_exception() -> None:
    # Policy scenario from test_policy_reliability: the reranker's top chunk is
    # an unrelated rule; the governing clause and its exception come from the pool.
    def policy(cid: str, section: str, text: str, score: float) -> ScoredChunk:
        return ScoredChunk(
            chunk=Chunk(
                id=cid,
                text=text,
                metadata=ChunkMetadata(document_id="policy", source_file="rules.pdf", section_title=section),
            ),
            score=score,
            rerank_score=score,
        )

    unrelated = policy("premises", "UNATTENDED PREMISES", "Employees must obtain express owner permission before entering unattended premises.", 8.5)
    primary = policy("own-account", "22.0 WORKING ON OWN ACCOUNT", "Employees must not perform private electrical work on their own account without authorization.", 3.0)
    family = policy("family-exception", "22.0 WORKING ON OWN ACCOUNT", "Immediate family includes a sister. However, work with a commercial value above $500 requires authorization.", 2.5)
    selector = GoverningClauseSelector()
    selection = selector.select(
        "Can I perform a $900 electrical job for my sister?", [unrelated], candidate_pool=[unrelated, primary, family]
    )
    governing = selector.order_for_context(selection, max_chunks=6)
    merged = merge_governing_context([unrelated], governing, max_chunks=6, mode="rank_anchor", anchor_k=2)
    assert merged[0].chunk.id == governing[0].chunk.id
    assert {"own-account", "family-exception"} <= set(_ids(merged))
    assert any(calc.kind == "threshold_comparison" for calc in selection.calculations)


# A legal-textbook pattern from the eval: the reranker's rank-1 chunk answers the
# question in plain prose, while pool chunks full of "shall" / "however" /
# "includes" win the governing-clause roles and fill every context slot.
_ANSWER = "Dictatorship is rule by one person holding unlimited power over the state and its people."
_DISTRACTORS = {
    **{
        f"pool-{i}": (
            f"Constitutional topic {i}: a monarch holding power shall rule one person at a time. "
            "However, succession passes by inheritance."
        )
        for i in range(1, 5)
    },
    **{
        f"pool-{i}": (
            f"Constitutional topic {i}: sovereignty means the power to rule, "
            "and a council shall hold it for one person."
        )
        for i in range(5, 8)
    },
}


def _evidence_loss_pipeline(monkeypatch, mode: str) -> RAGPipeline:
    monkeypatch.setattr(settings, "context_assembly_mode", mode)
    monkeypatch.setattr(settings, "context_rank_anchor_k", 2)
    chunks = [_chunk("answer", _ANSWER), *[_chunk(cid, text) for cid, text in _DISTRACTORS.items()]]
    scores = {"answer": 0.2, **{cid: 0.9 - i * 0.05 for i, cid in enumerate(_DISTRACTORS)}}
    relevance = {"answer": 0.99, **{cid: 0.1 for cid in _DISTRACTORS}}
    return _pipeline(_Retriever(chunks, scores), _FixedOrderReranker(relevance), {c.id: c for c in chunks}, "resolve")


_LOSS_QUERY = "What do we call rule by one person holding unlimited power?"


def test_governing_mode_discards_the_rank1_chunk(quiet_settings) -> None:
    ctx = _evidence_loss_pipeline(quiet_settings, "governing").run_retrieval_stages(_LOSS_QUERY)
    stages = ctx.retrieval_stages
    assert stages["post_rerank"][0] == "answer"
    assert "answer" not in stages["final_context"]  # the measured regression


def test_rank_anchor_mode_delivers_the_rank1_chunk_and_keeps_the_primary(quiet_settings) -> None:
    ctx = _evidence_loss_pipeline(quiet_settings, "rank_anchor").run_retrieval_stages(_LOSS_QUERY)
    stages = ctx.retrieval_stages
    assert stages["post_rerank"][0] == "answer"
    assert "answer" in stages["final_context"]
    assert stages["final_context"][0] == stages["governing_roles"]["primary"][0]
    for key in ("governing_selection", "post_governing", "post_packing", "final_context"):
        assert key in stages


def test_pipeline_shares_an_initially_empty_docstore(quiet_settings) -> None:
    # The API builds the pipeline before any upload, so the library's docstore is
    # still empty; documents added later must be visible to scope resolution.
    docstore: dict[str, Chunk] = {}
    pipe = _pipeline(_Retriever(_GUIDE_CHUNKS, _SCORES), _FixedOrderReranker({"g-memory": 0.9}), docstore, "resolve")
    assert pipe.docstore is docstore

    docstore.update({c.id: c for c in _GUIDE_CHUNKS})
    ctx = pipe.run_retrieval_stages(_QUERY)
    assert ctx.scope_decision.active_document_id == GUIDE_ID
