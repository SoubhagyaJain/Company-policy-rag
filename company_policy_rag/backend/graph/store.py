"""Persistent policy knowledge graph.

Mirrors the ``BM25SearchIndex`` lifecycle (``add_chunks`` / ``remove_by_*`` /
``save`` / ``load``) so it slots into existing ingestion and DI code without
introducing a new pattern.

Two properties this module is responsible for and the rest of the system relies on:

* **Never raises on load.** A missing, corrupt, or version-mismatched graph
  degrades to "no graph" — retrieval stays flat and the user sees today's
  behaviour. A graph failure is never a user-visible failure.
* **Atomic saves.** Writes go to a temp file on the same filesystem and are
  swapped in with ``os.replace``, so a crash mid-write leaves the previous graph
  intact rather than a truncated one.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph

from backend.graph.models import (
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    PendingReference,
)
from backend.models.chunk import Chunk
from backend.utils.logging import logger

GRAPH_FILE = "graph.json"

# Unresolved cross-references accumulate when referenced documents have not been
# ingested yet. Capped FIFO so a pathological corpus cannot grow the graph file
# without bound.
MAX_PENDING_REFERENCES = 10_000

# Node types shared across documents. They are pruned by chunk id on delete rather
# than removed outright, because another document may still reference them.
_GLOBAL_NODE_TYPES = {
    NodeType.ENTITY.value,
    NodeType.TOPIC.value,
    NodeType.POLICY.value,
}


class PolicyGraphStore:
    """A ``MultiDiGraph`` of documents, sections, clauses, chunks and entities."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        storage_dir: str | Path = "storage/graph",
        builder: Any | None = None,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        # Imported lazily so the store stays importable (and unit-testable) on its
        # own, and so a fake builder can be injected in tests.
        if builder is None:
            from backend.graph.builder import StructuralGraphBuilder

            builder = StructuralGraphBuilder()
        self.builder = builder

        # Ingestion runs on a worker thread (run_in_threadpool) while query threads
        # read concurrently, so the graph is genuinely shared mutable state. One
        # RLock guards every mutation and every traversal snapshot. Traversal is
        # budget-bounded to a few hundred node visits, so hold time is ~1ms and a
        # reader/writer split would not pay for its complexity.
        self._lock = threading.RLock()
        self._graph = self._empty_graph()

    # ── construction helpers ────────────────────────────────────────────────

    @classmethod
    def _empty_graph(cls) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        graph.graph.update(
            {
                "schema_version": cls.SCHEMA_VERSION,
                "graph_version": 0,
                "built_at": None,
                "pending_references": [],
            }
        )
        return graph

    @property
    def graph(self) -> nx.MultiDiGraph:
        """The underlying graph. Callers must hold :attr:`lock` while traversing."""
        return self._graph

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @property
    def graph_version(self) -> int:
        """Monotonic counter bumped on every mutation.

        Community summaries record the version they were built from; a mismatch is
        how staleness is detected without re-reading the whole graph.
        """
        return int(self._graph.graph.get("graph_version", 0))

    def _bump_version(self) -> None:
        self._graph.graph["graph_version"] = self.graph_version + 1
        self._graph.graph["built_at"] = datetime.now(timezone.utc).isoformat()

    @property
    def pending_references(self) -> list[dict[str, Any]]:
        return self._graph.graph.setdefault("pending_references", [])

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "nodes": self._graph.number_of_nodes(),
                "edges": self._graph.number_of_edges(),
                "pending_references": len(self.pending_references),
                "graph_version": self.graph_version,
            }

    # ── mutation ────────────────────────────────────────────────────────────

    def apply_fragment(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        pending: list[PendingReference] | None = None,
    ) -> int:
        """Merge a built fragment into the graph.

        Nodes merge by id (``chunk_ids`` are unioned, so a global ENTITY node
        accumulates mentions across documents). Edges are keyed by edge type, so
        re-applying an identical fragment overwrites rather than duplicating —
        this is what makes a rebuild idempotent.
        """
        with self._lock:
            for node in nodes:
                self._merge_node(node)
            for edge in edges:
                self._merge_edge(edge)
            if pending:
                self._add_pending(pending)
            self._bump_version()
            return len(nodes)

    def _merge_node(self, node: GraphNode) -> None:
        payload = node.model_dump(mode="json", exclude={"id"})
        if node.id in self._graph:
            existing = self._graph.nodes[node.id]
            merged_chunk_ids = list(
                dict.fromkeys(list(existing.get("chunk_ids", [])) + list(node.chunk_ids))
            )
            existing.update(payload)
            existing["chunk_ids"] = merged_chunk_ids
        else:
            self._graph.add_node(node.id, **payload)

    def _merge_edge(self, edge: GraphEdge) -> None:
        # Explicit key = edge type: adding the same relationship twice replaces it
        # instead of stacking parallel edges. Two nodes can still be joined by
        # different relationships (REFERENCES *and* MENTIONS), which is why this is
        # a MultiDiGraph rather than a DiGraph.
        self._graph.add_edge(
            edge.source,
            edge.target,
            key=edge.edge_type.value,
            **edge.model_dump(mode="json", exclude={"source", "target"}),
        )

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Build and merge graph structure for ``chunks``.

        Re-ingestion of a document must call :meth:`remove_by_document_id` first:
        ``Chunk.id`` is generated fresh on every ingest, so merging without a
        removal would leave the previous generation's chunk nodes orphaned.
        """
        if not chunks:
            return 0
        with self._lock:
            nodes, edges, pending = self.builder.build(chunks, existing_graph=self._graph)
            count = self.apply_fragment(nodes, edges, pending)
            self._resolve_pending_references()
            return count

    def build_index(self, chunks: list[Chunk]) -> None:
        """Rebuild the whole graph from scratch."""
        with self._lock:
            self._graph = self._empty_graph()
            self.add_chunks(chunks)

    # ── pending cross-references ────────────────────────────────────────────

    def _add_pending(self, pending: list[PendingReference]) -> None:
        bucket = self.pending_references
        seen = {(p["from_chunk_id"], p["target_key"]) for p in bucket}
        for ref in pending:
            key = (ref.from_chunk_id, ref.target_key)
            if key in seen:
                continue
            bucket.append(ref.model_dump(mode="json"))
            seen.add(key)
        if len(bucket) > MAX_PENDING_REFERENCES:
            dropped = len(bucket) - MAX_PENDING_REFERENCES
            del bucket[:dropped]
            logger.warning(
                "pending_references exceeded %d; dropped %d oldest entries",
                MAX_PENDING_REFERENCES,
                dropped,
            )

    def _resolve_pending_references(self) -> int:
        """Replay unresolved references against the current node index.

        Called after every ingest so a reference to a document uploaded *later*
        still resolves. Without this, cross-reference edges would silently depend
        on upload order.
        """
        bucket = self.pending_references
        if not bucket:
            return 0

        still_pending: list[dict[str, Any]] = []
        resolved = 0
        for raw in bucket:
            ref = PendingReference(**raw)
            source_id = f"chunk:{ref.from_chunk_id}"
            if source_id not in self._graph:
                # Source chunk is gone (its document was deleted); drop the entry.
                continue
            target = self.builder.resolve_reference_target(ref, self._graph)
            if target is None:
                still_pending.append(raw)
                continue
            self._merge_edge(
                GraphEdge(
                    source=source_id,
                    target=target,
                    edge_type=EdgeType.REFERENCES,
                    weight=1.0,
                    confidence=1.0,
                    provenance="reference",
                    attrs={
                        "raw_text": ref.raw_text,
                        "target_key": ref.target_key,
                        "deferred": True,
                    },
                )
            )
            resolved += 1

        self._graph.graph["pending_references"] = still_pending
        if resolved:
            logger.info("Resolved %d deferred cross-reference(s)", resolved)
        return resolved

    # ── removal ─────────────────────────────────────────────────────────────

    def remove_by_document_id(self, document_id: str) -> None:
        with self._lock:
            self._remove_where(lambda data: data.get("document_id") == document_id)

    def remove_by_source_file(self, source_file: str) -> None:
        with self._lock:
            self._remove_where(
                lambda data: (data.get("attrs") or {}).get("source_file") == source_file
            )

    def _remove_where(self, predicate: Callable[[dict[str, Any]], bool]) -> None:
        """Drop matching nodes, then repair what pointed at them.

        Global nodes (ENTITY/TOPIC/POLICY) are shared across documents, so they are
        pruned by chunk id rather than deleted outright, and removed only once no
        document references them any more.
        """
        doomed = [n for n, data in self._graph.nodes(data=True) if predicate(data)]
        if not doomed:
            return
        doomed_set = set(doomed)
        doomed_chunk_ids: set[str] = set()
        for node_id in doomed:
            doomed_chunk_ids.update(self._graph.nodes[node_id].get("chunk_ids", []))

        # A cross-reference into a deleted document is not gone — it is unresolved
        # again. Return it to the pending queue so a re-upload restores the edge.
        revived: list[PendingReference] = []
        for source, target, key, data in list(self._graph.edges(keys=True, data=True)):
            if key != EdgeType.REFERENCES.value:
                continue
            if target in doomed_set and source not in doomed_set:
                attrs = data.get("attrs") or {}
                revived.append(
                    PendingReference(
                        from_chunk_id=source.removeprefix("chunk:"),
                        raw_text=attrs.get("raw_text", ""),
                        target_key=attrs.get("target_key", ""),
                        scope_document_id=self._graph.nodes[source].get("document_id"),
                    )
                )

        self._graph.remove_nodes_from(doomed)

        # Prune the deleted document's chunk ids out of surviving global nodes.
        for node_id, data in list(self._graph.nodes(data=True)):
            chunk_ids = data.get("chunk_ids") or []
            if not chunk_ids:
                continue
            kept = [c for c in chunk_ids if c not in doomed_chunk_ids]
            if len(kept) == len(chunk_ids):
                continue
            if not kept and data.get("node_type") in _GLOBAL_NODE_TYPES:
                self._graph.remove_node(node_id)
            else:
                data["chunk_ids"] = kept

        if revived:
            self._add_pending(revived)
        self._bump_version()

    def clear(self) -> None:
        with self._lock:
            self._graph = self._empty_graph()

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self, storage_dir: str | Path | None = None) -> None:
        """Persist atomically: temp file on the same filesystem, fsync, ``os.replace``.

        A crash mid-write leaves the previous ``graph.json`` fully intact.
        """
        target_dir = Path(storage_dir) if storage_dir else self.storage_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / GRAPH_FILE

        with self._lock:
            payload = json_graph.node_link_data(self._graph, edges="links")

        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=str(target_dir), prefix=".graph-", suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
        except BaseException:
            # Leave the previous graph.json untouched and clean up the temp file.
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

        logger.info(
            "Saved policy graph (%d nodes, %d edges, v%d) to %s",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
            self.graph_version,
            target_dir,
        )

    def load(self, storage_dir: str | Path | None = None) -> bool:
        """Load the graph. Returns False instead of raising, always.

        Retrieval must survive a missing or damaged graph with no user-visible
        effect, so every failure path here is a logged ``False``.
        """
        target_dir = Path(storage_dir) if storage_dir else self.storage_dir
        target = target_dir / GRAPH_FILE

        if not target.is_file():
            logger.info("No policy graph at %s; graph features disabled.", target)
            return False

        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            version = int(payload.get("graph", {}).get("schema_version", -1))
            if version != self.SCHEMA_VERSION:
                logger.warning(
                    "Policy graph schema v%s does not match v%s — rebuild required.",
                    version,
                    self.SCHEMA_VERSION,
                )
                return False
            graph = json_graph.node_link_graph(
                payload, directed=True, multigraph=True, edges="links"
            )
        except Exception as exc:
            self._quarantine(target, exc)
            return False

        with self._lock:
            self._graph = graph
            self._graph.graph.setdefault("pending_references", [])
        logger.info(
            "Loaded policy graph (%d nodes, %d edges, v%d) from %s",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            self.graph_version,
            target_dir,
        )
        return True

    @staticmethod
    def _quarantine(target: Path, exc: Exception) -> None:
        """Preserve a corrupt graph for debugging rather than deleting it."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        quarantined = target.with_name(f"{target.name}.corrupt-{stamp}")
        try:
            os.replace(target, quarantined)
            logger.warning(
                "Policy graph at %s is unreadable (%s); quarantined to %s",
                target,
                exc,
                quarantined.name,
            )
        except OSError:
            logger.warning("Policy graph at %s is unreadable (%s).", target, exc)
