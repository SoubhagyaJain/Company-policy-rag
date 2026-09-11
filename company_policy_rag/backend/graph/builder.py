"""Deterministic graph construction from chunk metadata.

Every edge here is derived from metadata that ingestion already produces —
``section_path``, ``clause_id``, ``parent_id``/``child_ids``, ``chunk_index``,
``policy_id``, ``key_entities``, ``topic_tags`` — plus regex cross-references
parsed from chunk text. No LLM calls, no similarity, no fuzzy matching.

The build is a pure function of its inputs: the same chunks always produce the
same node ids and edge keys, which is what lets the store merge a rebuild without
duplicating structure.
"""

from __future__ import annotations

import re
from typing import Any

import networkx as nx

from backend.graph.models import (
    EDGE_SPECS,
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    PendingReference,
    chunk_node_id,
    clause_node_id,
    document_node_id,
    entity_node_id,
    normalize_label,
    policy_node_id,
    section_node_id,
    topic_node_id,
)
from backend.models.chunk import Chunk
from backend.utils.logging import logger

# Cross-reference patterns. Each yields one capture group: the referenced
# identifier. Deliberately conservative — a pattern that over-matches produces
# wrong edges, which is worse than producing none.
_SECTION_REF_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"§\s*(\d+(?:\.\d+)*(?:\([a-z0-9]+\))?)", re.IGNORECASE),
    re.compile(r"\bSections?\s+(\d+(?:\.\d+)*(?:\([a-z0-9]+\))?)", re.IGNORECASE),
    re.compile(r"\bclauses?\s+(\d+(?:\.\d+)*(?:\([a-z0-9]+\))?)", re.IGNORECASE),
    re.compile(r"\bArticle\s+([IVXLC]+|\d+)\b", re.IGNORECASE),
)

# Explicit policy citations resolve straight to a POLICY node and skip the
# document/policy/global scope cascade entirely.
_POLICY_REF_PATTERN = re.compile(r"\bPolicy\s+([A-Z]{2,}[-\s]?\d+(?:\.\d+)*)\b")

_POLICY_TARGET_PREFIX = "policy::"


class StructuralGraphBuilder:
    """Builds graph fragments from chunks. Stateless and deterministic."""

    def __init__(self, enable_topic_edges: bool = False) -> None:
        # Topic tags are coarse ("compliance", "hr") and link large, weakly related
        # sets of chunks, so they are off unless explicitly enabled.
        self.enable_topic_edges = enable_topic_edges

    # ── public API ──────────────────────────────────────────────────────────

    def build(
        self,
        chunks: list[Chunk],
        existing_graph: nx.MultiDiGraph | None = None,
    ) -> tuple[list[GraphNode], list[GraphEdge], list[PendingReference]]:
        """Return the nodes, edges and unresolved references for ``chunks``."""
        nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []

        for chunk in chunks:
            self._build_chunk_structure(chunk, nodes, edges)

        self._build_adjacency(chunks, edges)
        self._build_parent_child(chunks, nodes, edges)

        # References resolve against the existing graph plus everything this batch
        # just created, so intra-batch references work on a first ingest.
        index = self._build_resolution_index(existing_graph, nodes)
        pending = self._build_references(chunks, index, edges)

        return list(nodes.values()), edges, pending

    def resolve_reference_target(
        self,
        ref: PendingReference,
        graph: nx.MultiDiGraph,
    ) -> str | None:
        """Re-attempt resolution of a deferred reference against the live graph."""
        index = self._build_resolution_index(graph, {})
        return self._resolve(ref.target_key, ref.scope_document_id, ref.scope_policy_id, index)

    # ── node/edge construction ──────────────────────────────────────────────

    def _build_chunk_structure(
        self,
        chunk: Chunk,
        nodes: dict[str, GraphNode],
        edges: list[GraphEdge],
    ) -> None:
        meta = chunk.metadata
        doc_id = meta.document_id
        chunk_nid = chunk_node_id(chunk.id)

        doc_nid = document_node_id(doc_id)
        self._upsert(
            nodes,
            GraphNode(
                id=doc_nid,
                node_type=NodeType.DOCUMENT,
                label=meta.source_file or doc_id,
                document_id=doc_id,
                attrs={"source_file": meta.source_file, "category": meta.category},
            ),
            chunk.id,
        )

        self._upsert(
            nodes,
            GraphNode(
                id=chunk_nid,
                node_type=NodeType.CHUNK,
                label=meta.section_title or f"chunk {meta.chunk_index}",
                document_id=doc_id,
                attrs={
                    "source_file": meta.source_file,
                    "chunk_index": meta.chunk_index,
                    "page_number": meta.page_number,
                    "section_path": meta.section_path,
                },
            ),
            chunk.id,
        )

        # Policy: a POLICY node, not pairwise chunk edges. N documents sharing a
        # policy produce N edges instead of N-squared chunk pairs.
        if meta.policy_id:
            policy_nid = policy_node_id(meta.policy_id)
            self._upsert(
                nodes,
                GraphNode(
                    id=policy_nid,
                    node_type=NodeType.POLICY,
                    label=meta.policy_id,
                    attrs={"policy_id": normalize_label(meta.policy_id)},
                ),
                chunk.id,
            )
            self._add_edge(edges, doc_nid, policy_nid, EdgeType.GOVERNED_BY)

        # Hierarchy: document -> section -> clause -> chunk, skipping levels the
        # document does not provide.
        parent_nid = doc_nid
        if meta.section_path or meta.section_number:
            section_key = meta.section_path or meta.section_number or ""
            section_nid = section_node_id(doc_id, section_key)
            self._upsert(
                nodes,
                GraphNode(
                    id=section_nid,
                    node_type=NodeType.SECTION,
                    label=meta.section_title or section_key,
                    document_id=doc_id,
                    attrs={
                        "source_file": meta.source_file,
                        "section_path": meta.section_path,
                        "section_number": normalize_label(meta.section_number or ""),
                        "section_level": meta.section_level,
                        "policy_id": normalize_label(meta.policy_id or ""),
                    },
                ),
                chunk.id,
            )
            self._add_edge(edges, parent_nid, section_nid, EdgeType.CONTAINS)
            parent_nid = section_nid

        if meta.clause_id:
            clause_nid = clause_node_id(doc_id, meta.clause_id)
            self._upsert(
                nodes,
                GraphNode(
                    id=clause_nid,
                    node_type=NodeType.CLAUSE,
                    label=meta.clause_id,
                    document_id=doc_id,
                    attrs={
                        "source_file": meta.source_file,
                        "clause_id": normalize_label(meta.clause_id),
                        "parent_section": meta.parent_section,
                        "policy_id": normalize_label(meta.policy_id or ""),
                    },
                ),
                chunk.id,
            )
            self._add_edge(edges, parent_nid, clause_nid, EdgeType.CONTAINS)
            parent_nid = clause_nid

        self._add_edge(edges, parent_nid, chunk_nid, EdgeType.CONTAINS)

        for entity in meta.key_entities or []:
            if not normalize_label(entity):
                continue
            entity_nid = entity_node_id(entity)
            self._upsert(
                nodes,
                GraphNode(
                    id=entity_nid,
                    node_type=NodeType.ENTITY,
                    label=entity,
                ),
                chunk.id,
            )
            self._add_edge(edges, chunk_nid, entity_nid, EdgeType.MENTIONS)

        if self.enable_topic_edges:
            for tag in meta.topic_tags or []:
                if not normalize_label(tag):
                    continue
                topic_nid = topic_node_id(tag)
                self._upsert(
                    nodes,
                    GraphNode(id=topic_nid, node_type=NodeType.TOPIC, label=tag),
                    chunk.id,
                )
                self._add_edge(edges, chunk_nid, topic_nid, EdgeType.TAGGED)

    def _build_adjacency(self, chunks: list[Chunk], edges: list[GraphEdge]) -> None:
        """NEXT edges between consecutive chunks within the same section.

        Lets a clause split across a chunk boundary be rejoined at retrieval time.
        """
        buckets: dict[tuple[str, str], list[Chunk]] = {}
        for chunk in chunks:
            key = (chunk.metadata.document_id, chunk.metadata.section_path or "")
            buckets.setdefault(key, []).append(chunk)

        for bucket in buckets.values():
            ordered = sorted(bucket, key=lambda c: c.metadata.chunk_index)
            for left, right in zip(ordered, ordered[1:]):
                if right.metadata.chunk_index - left.metadata.chunk_index != 1:
                    continue
                self._add_edge(
                    edges,
                    chunk_node_id(left.id),
                    chunk_node_id(right.id),
                    EdgeType.NEXT,
                )

    def _build_parent_child(
        self,
        chunks: list[Chunk],
        nodes: dict[str, GraphNode],
        edges: list[GraphEdge],
    ) -> None:
        """PARENT_OF edges from the parent/child ids the chunkers already assign."""
        known = {chunk.id for chunk in chunks}
        for chunk in chunks:
            meta = chunk.metadata
            if meta.parent_id and meta.parent_id in known:
                self._add_edge(
                    edges,
                    chunk_node_id(meta.parent_id),
                    chunk_node_id(chunk.id),
                    EdgeType.PARENT_OF,
                )
            for child_id in meta.child_ids or []:
                if child_id in known:
                    self._add_edge(
                        edges,
                        chunk_node_id(chunk.id),
                        chunk_node_id(child_id),
                        EdgeType.PARENT_OF,
                    )

    # ── cross-references ────────────────────────────────────────────────────

    def _build_references(
        self,
        chunks: list[Chunk],
        index: dict[str, list[dict[str, Any]]],
        edges: list[GraphEdge],
    ) -> list[PendingReference]:
        pending: list[PendingReference] = []
        for chunk in chunks:
            meta = chunk.metadata
            source_nid = chunk_node_id(chunk.id)
            for raw_text, target_key in self._extract_references(chunk):
                # A clause referring to its own section is noise, not a link.
                if target_key == normalize_label(meta.section_number or "") or (
                    target_key == normalize_label(meta.clause_id or "")
                ):
                    continue
                target = self._resolve(
                    target_key,
                    meta.document_id,
                    normalize_label(meta.policy_id or "") or None,
                    index,
                )
                if target is None:
                    pending.append(
                        PendingReference(
                            from_chunk_id=chunk.id,
                            raw_text=raw_text,
                            target_key=target_key,
                            scope_document_id=meta.document_id,
                            scope_policy_id=normalize_label(meta.policy_id or "") or None,
                        )
                    )
                    continue
                self._add_edge(
                    edges,
                    source_nid,
                    target,
                    EdgeType.REFERENCES,
                    provenance="reference",
                    attrs={"raw_text": raw_text, "target_key": target_key},
                )
        return pending

    @staticmethod
    def _extract_references(chunk: Chunk) -> list[tuple[str, str]]:
        """Return ``(raw_text, target_key)`` pairs found in the chunk text."""
        found: list[tuple[str, str]] = []
        seen: set[str] = set()

        for match in _POLICY_REF_PATTERN.finditer(chunk.text):
            key = _POLICY_TARGET_PREFIX + normalize_label(
                match.group(1).replace(" ", "-")
            )
            if key not in seen:
                seen.add(key)
                found.append((match.group(0), key))

        for pattern in _SECTION_REF_PATTERNS:
            for match in pattern.finditer(chunk.text):
                key = normalize_label(match.group(1))
                if not key or key in seen:
                    continue
                seen.add(key)
                found.append((match.group(0), key))

        return found

    def _build_resolution_index(
        self,
        graph: nx.MultiDiGraph | None,
        new_nodes: dict[str, GraphNode],
    ) -> dict[str, list[dict[str, Any]]]:
        """Index every addressable target by its normalised identifier.

        Entries carry their document and policy scope so the resolution cascade can
        narrow before it widens.
        """
        index: dict[str, list[dict[str, Any]]] = {}

        def record(node_id: str, node_type: str, attrs: dict[str, Any], document_id: str | None) -> None:
            if node_type == NodeType.POLICY.value:
                key = _POLICY_TARGET_PREFIX + (attrs.get("policy_id") or "")
            elif node_type == NodeType.CLAUSE.value:
                key = attrs.get("clause_id") or ""
            elif node_type == NodeType.SECTION.value:
                key = attrs.get("section_number") or ""
            else:
                return
            if not key or key == _POLICY_TARGET_PREFIX:
                return
            entry = {
                "node_id": node_id,
                "node_type": node_type,
                "document_id": document_id,
                "policy_id": attrs.get("policy_id") or "",
            }
            bucket = index.setdefault(key, [])
            if not any(e["node_id"] == entry["node_id"] for e in bucket):
                bucket.append(entry)

        if graph is not None:
            for node_id, data in graph.nodes(data=True):
                record(
                    node_id,
                    data.get("node_type", ""),
                    data.get("attrs") or {},
                    data.get("document_id"),
                )
        for node in new_nodes.values():
            record(node.id, node.node_type.value, node.attrs, node.document_id)

        return index

    @staticmethod
    def _narrow(candidates: list[dict[str, Any]]) -> str | None:
        """Pick a single target from same-scope candidates, or None if truly ambiguous.

        A section and the clause beneath it routinely share a number ("7.3"), which
        is not real ambiguity — they are the same target at different granularity.
        Precedence resolves that deterministically by preferring the most specific
        addressable unit. Two candidates of the *same* type remain ambiguous and
        yield no edge.
        """
        if len(candidates) == 1:
            return candidates[0]["node_id"]
        for preferred in (
            NodeType.POLICY.value,
            NodeType.CLAUSE.value,
            NodeType.SECTION.value,
        ):
            tier = [c for c in candidates if c["node_type"] == preferred]
            if len(tier) == 1:
                return tier[0]["node_id"]
            if len(tier) > 1:
                return None
        return None

    @classmethod
    def _resolve(
        cls,
        target_key: str,
        document_id: str | None,
        policy_id: str | None,
        index: dict[str, list[dict[str, Any]]],
    ) -> str | None:
        """Resolve a reference by narrowing scope; first unique match wins.

        Order is same document, then same policy, then corpus-globally unique.
        Zero matches or an ambiguous match at every scope means **no edge** — a
        wrong cross-reference is worse than a missing one, so nothing is guessed.
        """
        candidates = index.get(target_key) or []
        if not candidates:
            return None

        if document_id:
            scoped = [c for c in candidates if c["document_id"] == document_id]
            if scoped:
                # Ambiguity *within* the document is not widened away: if this
                # document cannot say which target it means, no other scope can.
                return cls._narrow(scoped)

        if policy_id:
            scoped = [c for c in candidates if c["policy_id"] == policy_id]
            if scoped:
                resolved = cls._narrow(scoped)
                if resolved:
                    return resolved

        return cls._narrow(candidates)

    # ── small helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _upsert(nodes: dict[str, GraphNode], node: GraphNode, chunk_id: str) -> None:
        existing = nodes.get(node.id)
        if existing is None:
            node.chunk_ids = [chunk_id]
            nodes[node.id] = node
            return
        if chunk_id not in existing.chunk_ids:
            existing.chunk_ids.append(chunk_id)

    @staticmethod
    def _add_edge(
        edges: list[GraphEdge],
        source: str,
        target: str,
        edge_type: EdgeType,
        *,
        provenance: str = "structural",
        attrs: dict[str, Any] | None = None,
    ) -> None:
        if source == target:
            return
        spec = EDGE_SPECS[edge_type]
        edges.append(
            GraphEdge(
                source=source,
                target=target,
                edge_type=edge_type,
                weight=spec.base_weight,
                confidence=1.0,
                provenance=provenance,  # type: ignore[arg-type]
                attrs=attrs or {},
            )
        )


__all__ = ["StructuralGraphBuilder"]
