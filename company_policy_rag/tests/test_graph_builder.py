"""Tests for StructuralGraphBuilder: hierarchy, adjacency, entities, stable ids."""

from __future__ import annotations

from backend.graph.builder import StructuralGraphBuilder
from backend.graph.models import (
    EdgeType,
    NodeType,
    chunk_node_id,
    normalize_label,
    policy_node_id,
)
from backend.models.chunk import Chunk, ChunkMetadata, ChunkRole


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


def edges_of(edges, edge_type: EdgeType):
    return [e for e in edges if e.edge_type is edge_type]


def test_hierarchy_document_section_clause_chunk() -> None:
    chunk = make_chunk(
        "d1",
        0,
        section_path="Pay > Overtime",
        section_number="4.2",
        clause_id="4.2",
    )
    nodes, edges, _ = StructuralGraphBuilder().build([chunk])

    types = {n.node_type for n in nodes}
    assert types == {
        NodeType.DOCUMENT,
        NodeType.SECTION,
        NodeType.CLAUSE,
        NodeType.CHUNK,
    }
    # doc -> section -> clause -> chunk
    assert len(edges_of(edges, EdgeType.CONTAINS)) == 3


def test_hierarchy_skips_missing_levels() -> None:
    """A document with no section or clause metadata links straight to its chunks."""
    chunk = make_chunk("d1", 0)
    nodes, edges, _ = StructuralGraphBuilder().build([chunk])

    assert {n.node_type for n in nodes} == {NodeType.DOCUMENT, NodeType.CHUNK}
    contains = edges_of(edges, EdgeType.CONTAINS)
    assert len(contains) == 1
    assert contains[0].source == "doc:d1"


def test_next_edges_link_consecutive_chunks_in_same_section() -> None:
    chunks = [
        make_chunk("d1", 0, section_path="A"),
        make_chunk("d1", 1, section_path="A"),
        make_chunk("d1", 2, section_path="A"),
    ]
    _, edges, _ = StructuralGraphBuilder().build(chunks)
    assert len(edges_of(edges, EdgeType.NEXT)) == 2


def test_next_edges_do_not_cross_sections_or_gaps() -> None:
    chunks = [
        make_chunk("d1", 0, section_path="A"),
        make_chunk("d1", 1, section_path="B"),  # different section
        make_chunk("d1", 5, section_path="A"),  # non-consecutive index
    ]
    _, edges, _ = StructuralGraphBuilder().build(chunks)
    assert edges_of(edges, EdgeType.NEXT) == []


def test_parent_child_edges_from_metadata() -> None:
    parent = make_chunk("d1", 0, node_role=ChunkRole.PARENT)
    child = make_chunk("d1", 1, node_role=ChunkRole.CHILD, parent_id=parent.id)
    _, edges, _ = StructuralGraphBuilder().build([parent, child])

    parent_edges = edges_of(edges, EdgeType.PARENT_OF)
    assert len(parent_edges) == 1
    assert parent_edges[0].source == chunk_node_id(parent.id)
    assert parent_edges[0].target == chunk_node_id(child.id)


def test_parent_edge_skipped_when_parent_not_in_batch() -> None:
    child = make_chunk("d1", 0, parent_id="chunk_does_not_exist")
    _, edges, _ = StructuralGraphBuilder().build([child])
    assert edges_of(edges, EdgeType.PARENT_OF) == []


def test_policy_node_avoids_pairwise_explosion() -> None:
    """N chunks under one policy must produce N document edges, not N^2 chunk pairs."""
    chunks = [make_chunk(f"d{i}", 0, policy_id="HR-104") for i in range(12)]
    nodes, edges, _ = StructuralGraphBuilder().build(chunks)

    policy_nodes = [n for n in nodes if n.node_type is NodeType.POLICY]
    assert len(policy_nodes) == 1
    assert policy_nodes[0].id == policy_node_id("HR-104")

    governed = edges_of(edges, EdgeType.GOVERNED_BY)
    assert len(governed) == 12  # one per document, not 12*11/2 chunk pairs
    assert all(e.source.startswith("doc:") for e in governed)


def test_entity_labels_normalize_to_one_node() -> None:
    chunks = [
        make_chunk("d1", 0, key_entities=["FLSA"]),
        make_chunk("d2", 0, key_entities=["  flsa "]),
        make_chunk("d3", 0, key_entities=["(FLSA)"]),
    ]
    nodes, _, _ = StructuralGraphBuilder().build(chunks)
    entity_nodes = [n for n in nodes if n.node_type is NodeType.ENTITY]

    assert len(entity_nodes) == 1
    assert entity_nodes[0].id == "entity:flsa"
    assert len(entity_nodes[0].chunk_ids) == 3


def test_topic_edges_off_by_default() -> None:
    chunk = make_chunk("d1", 0, topic_tags=["compliance"])
    nodes, edges, _ = StructuralGraphBuilder().build([chunk])

    assert not [n for n in nodes if n.node_type is NodeType.TOPIC]
    assert edges_of(edges, EdgeType.TAGGED) == []


def test_topic_edges_when_enabled() -> None:
    chunk = make_chunk("d1", 0, topic_tags=["Compliance"])
    nodes, edges, _ = StructuralGraphBuilder(enable_topic_edges=True).build([chunk])

    topics = [n for n in nodes if n.node_type is NodeType.TOPIC]
    assert [n.id for n in topics] == ["topic:compliance"]
    assert len(edges_of(edges, EdgeType.TAGGED)) == 1


def test_node_ids_stable_across_rebuilds() -> None:
    chunks = [
        make_chunk("d1", 0, section_path="Pay > Overtime", clause_id="4.2", policy_id="HR-104")
    ]
    first, _, _ = StructuralGraphBuilder().build(chunks)
    second, _, _ = StructuralGraphBuilder().build(chunks)

    assert sorted(n.id for n in first) == sorted(n.id for n in second)


def test_build_is_deterministic_for_edges() -> None:
    chunks = [
        make_chunk("d1", 0, section_path="A", clause_id="1.1"),
        make_chunk("d1", 1, section_path="A", clause_id="1.2"),
    ]
    _, first, _ = StructuralGraphBuilder().build(chunks)
    _, second, _ = StructuralGraphBuilder().build(chunks)

    key = lambda e: (e.source, e.target, e.edge_type.value)  # noqa: E731
    assert sorted(map(key, first)) == sorted(map(key, second))


def test_edge_weights_come_from_edge_specs() -> None:
    chunks = [make_chunk("d1", 0, section_path="A"), make_chunk("d1", 1, section_path="A")]
    _, edges, _ = StructuralGraphBuilder().build(chunks)

    next_edge = edges_of(edges, EdgeType.NEXT)[0]
    contains_edge = edges_of(edges, EdgeType.CONTAINS)[0]
    # NEXT is stronger evidence of relatedness than mere containment.
    assert next_edge.weight > contains_edge.weight


def test_no_self_edges() -> None:
    chunk = make_chunk("d1", 0, key_entities=["thing"])
    _, edges, _ = StructuralGraphBuilder().build([chunk])
    assert all(e.source != e.target for e in edges)


def test_blank_entities_are_ignored() -> None:
    chunk = make_chunk("d1", 0, key_entities=["", "   ", "..."])
    nodes, edges, _ = StructuralGraphBuilder().build([chunk])
    assert not [n for n in nodes if n.node_type is NodeType.ENTITY]
    assert edges_of(edges, EdgeType.MENTIONS) == []


def test_normalize_label_preserves_interior_punctuation() -> None:
    # "HR-104" and "hr 104" are different identifiers and must not collapse.
    assert normalize_label("HR-104") != normalize_label("HR 104")
