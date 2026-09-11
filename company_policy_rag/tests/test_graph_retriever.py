"""Tests for GraphRetriever expansion: budgets, scoring, hub suppression, clamping."""

from __future__ import annotations

from pathlib import Path

from backend.graph.models import EdgeType
from backend.graph.store import PolicyGraphStore
from backend.models.chunk import Chunk, ChunkMetadata
from backend.models.rag import ScoredChunk
from backend.retrieval.graph_retriever import GraphRetriever


def make_chunk(document_id: str, index: int, text: str = "text", **meta: object) -> Chunk:
    return Chunk(
        text=text,
        metadata=ChunkMetadata(
            document_id=document_id,
            source_file=f"{document_id}.pdf",
            chunk_index=index,
            **meta,
        ),
    )


def build(tmp_path: Path, chunks: list[Chunk]) -> tuple[GraphRetriever, PolicyGraphStore]:
    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks(chunks)
    docstore = {c.id: c for c in chunks}
    return GraphRetriever(store, docstore), store


def seed(chunk: Chunk, score: float = 1.0) -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=score)


# ── basic reachability ──────────────────────────────────────────────────────


def test_expands_to_referenced_clause(tmp_path: Path) -> None:
    """The case flat retrieval misses: a clause the top hit defers to."""
    referrer = make_chunk("d1", 0, "Subject to the exceptions in Section 7.3.", section_number="4.2")
    target = make_chunk("d1", 1, "Exceptions.", section_number="7.3", clause_id="7.3")
    retriever, _ = build(tmp_path, [referrer, target])

    result = retriever.expand([seed(referrer)])
    assert target.id in {sc.chunk.id for sc in result.chunks}


def test_expands_to_adjacent_chunk(tmp_path: Path) -> None:
    first = make_chunk("d1", 0, section_path="A")
    second = make_chunk("d1", 1, section_path="A")
    retriever, _ = build(tmp_path, [first, second])

    result = retriever.expand([seed(first)])
    assert second.id in {sc.chunk.id for sc in result.chunks}


def test_seeds_are_never_returned_as_expansions(tmp_path: Path) -> None:
    first = make_chunk("d1", 0, section_path="A")
    second = make_chunk("d1", 1, section_path="A")
    retriever, _ = build(tmp_path, [first, second])

    result = retriever.expand([seed(first), seed(second)])
    assert result.chunks == []


def test_empty_graph_returns_nothing(tmp_path: Path) -> None:
    chunk = make_chunk("d1", 0)
    store = PolicyGraphStore(storage_dir=tmp_path)
    retriever = GraphRetriever(store, {chunk.id: chunk})

    result = retriever.expand([seed(chunk)])
    assert result.chunks == []
    assert result.stop_reason == "empty_graph"


def test_no_seeds_returns_nothing(tmp_path: Path) -> None:
    retriever, _ = build(tmp_path, [make_chunk("d1", 0)])
    assert retriever.expand([]).chunks == []


def test_chunk_missing_from_docstore_is_skipped(tmp_path: Path) -> None:
    """Graph nodes hold ids, not text; an id the docstore cannot resolve is dropped."""
    first = make_chunk("d1", 0, section_path="A")
    second = make_chunk("d1", 1, section_path="A")
    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks([first, second])
    retriever = GraphRetriever(store, {first.id: first})  # second absent

    assert retriever.expand([seed(first)]).chunks == []


# ── budgets ─────────────────────────────────────────────────────────────────


def test_max_hops_limits_reach(tmp_path: Path) -> None:
    """A reference chain across documents needs one real hop per link."""
    chain = [
        make_chunk("d0", 0, "See Section 1.0.", section_number="9.0"),
        make_chunk("d1", 0, "See Section 2.0.", section_number="1.0", clause_id="1.0"),
        make_chunk("d2", 0, "See Section 3.0.", section_number="2.0", clause_id="2.0"),
        make_chunk("d3", 0, "End of chain.", section_number="3.0", clause_id="3.0"),
    ]
    retriever, _ = build(tmp_path, chain)

    one_hop = retriever.expand([seed(chain[0])], max_hops=1, hop_decay=0.9)
    three_hop = retriever.expand([seed(chain[0])], max_hops=3, hop_decay=0.9)

    assert len(one_hop.chunks) == 1
    assert len(three_hop.chunks) > len(one_hop.chunks)


def test_max_expanded_caps_returned_chunks(tmp_path: Path) -> None:
    chunks = [make_chunk("d1", i, section_path="A") for i in range(20)]
    retriever, _ = build(tmp_path, chunks)

    result = retriever.expand([seed(chunks[0])], max_hops=5, max_expanded=3, hop_decay=0.95)
    assert len(result.chunks) <= 3


def test_budget_nodes_stops_traversal(tmp_path: Path) -> None:
    chunks = [make_chunk("d1", i, section_path="A") for i in range(30)]
    retriever, _ = build(tmp_path, chunks)

    result = retriever.expand([seed(chunks[0])], max_hops=10, budget_nodes=3, hop_decay=0.99)
    assert result.stop_reason == "budget_nodes"
    assert result.visited <= 3


def test_timeout_stops_traversal(tmp_path: Path) -> None:
    chunks = [make_chunk("d1", i, section_path="A") for i in range(50)]
    retriever, _ = build(tmp_path, chunks)

    result = retriever.expand([seed(chunks[0])], max_hops=10, timeout_ms=0, hop_decay=0.99)
    assert result.stop_reason == "timeout"


def test_edge_type_filter_restricts_expansion(tmp_path: Path) -> None:
    """The reference-trigger path expands along REFERENCES edges only."""
    referrer = make_chunk("d1", 0, "See Section 7.3.", section_path="A", section_number="4.2")
    neighbor = make_chunk("d1", 1, "Adjacent prose.", section_path="A")
    target = make_chunk("d1", 2, "Exceptions.", section_path="B", section_number="7.3", clause_id="7.3")
    retriever, _ = build(tmp_path, [referrer, neighbor, target])

    # REFERENCES alone still reaches the referenced clause's text — descending out
    # of the clause node is dereferencing, not a semantic hop — but must not drag
    # in section siblings, which would need a CONTAINS step from the seed chunk.
    result = retriever.expand([seed(referrer)], edge_types={EdgeType.REFERENCES}, max_hops=2)
    ids = {sc.chunk.id for sc in result.chunks}
    assert target.id in ids
    assert neighbor.id not in ids


# ── scoring ─────────────────────────────────────────────────────────────────


def test_expanded_scores_never_reach_the_weakest_seed(tmp_path: Path) -> None:
    """The graph adds candidates; it must never outrank real retrieval."""
    chunks = [make_chunk("d1", i, section_path="A") for i in range(5)]
    retriever, _ = build(tmp_path, chunks)

    seeds = [seed(chunks[0], score=0.9), seed(chunks[1], score=0.5)]
    result = retriever.expand(seeds, max_hops=3, hop_decay=0.99)

    assert result.chunks
    assert all(sc.score < 0.5 for sc in result.chunks)


def test_score_decays_with_distance(tmp_path: Path) -> None:
    chain = [make_chunk("d1", i, section_path="A") for i in range(4)]
    retriever, _ = build(tmp_path, chain)

    result = retriever.expand([seed(chain[0])], max_hops=3)
    by_id = {sc.chunk.id: sc for sc in result.chunks}
    if chain[1].id in by_id and chain[2].id in by_id:
        assert by_id[chain[1].id].graph_score > by_id[chain[2].id].graph_score


def test_multiple_paths_take_max_not_sum(tmp_path: Path) -> None:
    """Summing paths would reward well-connected hubs; max keeps scores honest."""
    hub = make_chunk("d1", 0, "Hub.", section_path="A", key_entities=["alpha", "beta", "gamma"])
    others = [
        make_chunk("d1", i, f"Body {i}.", section_path="A", key_entities=["alpha", "beta", "gamma"])
        for i in range(1, 4)
    ]
    retriever, _ = build(tmp_path, [hub, *others])

    result = retriever.expand([seed(hub, score=1.0)], max_hops=2, hop_decay=0.6)
    # Reachable via three separate entity bridges; the score must reflect the best
    # single path, so it can never exceed one hop-decay step from the seed.
    assert all(sc.graph_score <= 1.0 * 0.6 for sc in result.chunks)


def test_graph_provenance_is_recorded(tmp_path: Path) -> None:
    referrer = make_chunk("d1", 0, "See Section 7.3.", section_number="4.2")
    target = make_chunk("d1", 1, "Exceptions.", section_number="7.3", clause_id="7.3")
    retriever, _ = build(tmp_path, [referrer, target])

    result = retriever.expand([seed(referrer)])
    expanded = next(sc for sc in result.chunks if sc.chunk.id == target.id)
    assert expanded.graph_hops and expanded.graph_hops >= 1
    assert EdgeType.REFERENCES.value in (expanded.graph_path or "")
    assert expanded.graph_score is not None


# ── hub suppression ─────────────────────────────────────────────────────────


def test_hub_entity_is_not_traversed(tmp_path: Path) -> None:
    """A generic entity shared by most chunks is a dead end, not a bridge."""
    sharers = [make_chunk("d1", i, f"Body {i}.", key_entities=["company"]) for i in range(20)]
    others = [make_chunk("d2", i, f"Other {i}.") for i in range(10)]
    retriever, _ = build(tmp_path, [*sharers, *others])

    permissive = retriever.expand(
        [seed(sharers[0])], edge_types={EdgeType.MENTIONS}, hub_degree_threshold=1000
    )
    suppressed = retriever.expand(
        [seed(sharers[0])], edge_types={EdgeType.MENTIONS}, hub_degree_threshold=5
    )
    assert len(permissive.chunks) > 0
    assert suppressed.chunks == []


def test_universal_entity_is_damped_to_zero(tmp_path: Path) -> None:
    """An entity every chunk mentions has no discriminative power at all."""
    chunks = [make_chunk("d1", i, f"Body {i}.", key_entities=["company"]) for i in range(10)]
    retriever, _ = build(tmp_path, chunks)

    result = retriever.expand(
        [seed(chunks[0])], edge_types={EdgeType.MENTIONS}, hub_degree_threshold=1000
    )
    assert result.chunks == []


def test_rare_entity_still_bridges(tmp_path: Path) -> None:
    a = make_chunk("d1", 0, "Body.", key_entities=["obscure-statute-1978"])
    b = make_chunk("d2", 0, "Body.", key_entities=["obscure-statute-1978"])
    filler = [make_chunk("d3", i, "Filler.") for i in range(20)]
    retriever, _ = build(tmp_path, [a, b, *filler])

    result = retriever.expand([seed(a)], edge_types={EdgeType.MENTIONS}, max_hops=2)
    assert b.id in {sc.chunk.id for sc in result.chunks}


# ── structural nodes are waypoints, not results ─────────────────────────────


def test_only_chunk_nodes_become_candidates(tmp_path: Path) -> None:
    chunks = [
        make_chunk("d1", 0, section_path="A", clause_id="1.1", policy_id="HR-1", key_entities=["x"]),
        make_chunk("d1", 1, section_path="A", clause_id="1.2", policy_id="HR-1", key_entities=["x"]),
    ]
    retriever, _ = build(tmp_path, chunks)

    result = retriever.expand([seed(chunks[0])], max_hops=3)
    # Everything returned resolves to a real chunk; sections/policies/entities are
    # traversed through but never handed to the reranker.
    assert all(sc.chunk.id in {c.id for c in chunks} for sc in result.chunks)
