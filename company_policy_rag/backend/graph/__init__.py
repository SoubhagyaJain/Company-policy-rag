"""Knowledge-graph layer for Graph RAG.

Materialises the document structure that ingestion already extracts into
``ChunkMetadata`` (section hierarchy, parent/child chunks, clause
cross-references, entities, topics) as a traversable graph, so retrieval can
expand a fused candidate set along real relationships instead of treating every
chunk as an island.

The graph stores structure only — never text. Chunk text stays in Chroma, BM25
and the docstore; graph nodes carry ``chunk_ids`` so a traversal result maps
straight back to ``Chunk`` objects.
"""

from backend.graph.models import (
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    PendingReference,
    Provenance,
    chunk_node_id,
    clause_node_id,
    document_node_id,
    entity_node_id,
    normalize_label,
    policy_node_id,
    section_node_id,
    topic_node_id,
)

__all__ = [
    "EdgeType",
    "GraphEdge",
    "GraphNode",
    "NodeType",
    "PendingReference",
    "Provenance",
    "chunk_node_id",
    "clause_node_id",
    "document_node_id",
    "entity_node_id",
    "normalize_label",
    "policy_node_id",
    "section_node_id",
    "topic_node_id",
]
