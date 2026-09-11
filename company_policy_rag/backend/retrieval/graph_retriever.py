"""Graph-based candidate expansion.

This is **not** a retriever in the dense/BM25 sense. It runs *after* fusion has
produced a ranked candidate set and adds chunks that are structurally connected to
the strong hits — a clause that the top hit defers to, the parent section it sits
under, the paragraph it continues into.

Two invariants keep this safe:

* **Expansion never promotes.** Every expanded score is clamped strictly below the
  weakest seed score, so the graph cannot reorder anything dense/BM25 found. Only
  the cross-encoder decides what actually reaches the context.
* **Expansion is bounded.** Hops, returned chunks, node visits and wall-clock time
  are all capped; whichever binds first stops the walk and returns what it has.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import Any

import networkx as nx

from backend.graph.models import EDGE_SPECS, EdgeType, NodeType
from backend.models.chunk import Chunk
from backend.models.rag import ScoredChunk
from backend.utils.logging import logger

# Traversal stops when the running path score falls below this; below it, the
# cross-encoder would discard the chunk anyway and the visit is wasted budget.
_MIN_PATH_SCORE = 1e-4


class GraphExpansionResult:
    """Expanded chunks plus why the walk stopped, for telemetry."""

    __slots__ = ("chunks", "visited", "stop_reason", "elapsed_ms")

    def __init__(
        self,
        chunks: list[ScoredChunk],
        visited: int,
        stop_reason: str,
        elapsed_ms: float,
    ) -> None:
        self.chunks = chunks
        self.visited = visited
        self.stop_reason = stop_reason
        self.elapsed_ms = elapsed_ms


class GraphRetriever:
    """Best-first expansion over the policy graph."""

    def __init__(self, store: Any, docstore: dict[str, Chunk] | None = None) -> None:
        self.store = store
        # Graph nodes hold chunk ids, never text: expanded ids are resolved back to
        # Chunk objects through the same docstore the pipeline already carries.
        self.docstore = docstore if docstore is not None else {}

    def expand(
        self,
        seeds: list[ScoredChunk],
        *,
        max_hops: int = 2,
        max_expanded: int = 12,
        edge_types: set[EdgeType] | None = None,
        budget_nodes: int = 300,
        timeout_ms: int = 150,
        hop_decay: float = 0.6,
        hub_degree_threshold: int = 50,
    ) -> GraphExpansionResult:
        started = time.perf_counter()
        if not seeds or self.store is None:
            return GraphExpansionResult([], 0, "no_seeds", 0.0)

        allowed = edge_types or set(EDGE_SPECS)
        seed_chunk_ids = {sc.chunk.id for sc in seeds}
        # The clamp that keeps expansion from ever outranking real retrieval.
        weakest_seed = min((sc.score or 0.0) for sc in seeds)

        best: dict[str, tuple[float, int, str]] = {}
        visited = 0
        stop_reason = "exhausted"

        with self.store.lock:
            graph = self.store.graph
            if graph.number_of_nodes() == 0:
                return GraphExpansionResult([], 0, "empty_graph", 0.0)

            total_chunks = self._chunk_node_count(graph)

            # Max-heap over running path score: the best path to any node is found
            # first, so `best` is a max across paths for free, and exhausting the
            # budget cuts the weakest paths rather than arbitrary ones.
            heap: list[tuple[float, int, str]] = []
            counter = 0
            seen_best: dict[str, float] = {}
            for sc in seeds:
                node_id = f"chunk:{sc.chunk.id}"
                if node_id not in graph:
                    continue
                score = sc.score or 0.0
                heapq.heappush(heap, (-score, counter, node_id))
                counter += 1
                seen_best[node_id] = score

            hops: dict[str, int] = {f"chunk:{sc.chunk.id}": 0 for sc in seeds}
            paths: dict[str, str] = {f"chunk:{sc.chunk.id}": "" for sc in seeds}

            while heap:
                if visited >= budget_nodes:
                    stop_reason = "budget_nodes"
                    break
                if (time.perf_counter() - started) * 1000 >= timeout_ms:
                    stop_reason = "timeout"
                    break

                neg_score, _, node_id = heapq.heappop(heap)
                score = -neg_score
                depth = hops.get(node_id, 0)
                visited += 1

                if depth >= max_hops:
                    continue

                for neighbor, edge_type, edge_score in self._neighbors(
                    graph, node_id, allowed, hub_degree_threshold, total_chunks
                ):
                    # Decay counts chunk-to-chunk steps only. Waypoints
                    # (SECTION/CLAUSE/ENTITY/POLICY) are addressing machinery, not
                    # content, so passing through one is not a semantic step and
                    # _neighbors has already made descending out of one free.
                    lands_on_chunk = self._is_chunk_node(graph, neighbor)
                    step_decay = hop_decay if lands_on_chunk else 1.0

                    next_score = score * step_decay * edge_score
                    if next_score < _MIN_PATH_SCORE:
                        continue
                    # A strictly-better path to an already-seen node is worth
                    # re-expanding; an equal or worse one is not.
                    if neighbor in seen_best and seen_best[neighbor] >= next_score:
                        continue

                    seen_best[neighbor] = next_score
                    hops[neighbor] = depth + 1 if lands_on_chunk else depth
                    path = paths.get(node_id, "")
                    paths[neighbor] = (
                        f"{path}>{edge_type.value}" if path else edge_type.value
                    )
                    heapq.heappush(heap, (-next_score, counter, neighbor))
                    counter += 1

                    chunk_id = self._chunk_id_of(graph, neighbor)
                    if chunk_id is None or chunk_id in seed_chunk_ids:
                        continue
                    prior = best.get(chunk_id)
                    if prior is None or next_score > prior[0]:
                        best[chunk_id] = (next_score, hops[neighbor], paths[neighbor])

        expanded = self._materialize(best, weakest_seed, max_expanded)
        elapsed = (time.perf_counter() - started) * 1000
        if len(best) > max_expanded and stop_reason == "exhausted":
            stop_reason = "max_expanded"

        logger.debug(
            "Graph expansion: %d seeds -> %d candidates (%d visited, %s, %.1fms)",
            len(seeds),
            len(expanded),
            visited,
            stop_reason,
            elapsed,
        )
        return GraphExpansionResult(expanded, visited, stop_reason, elapsed)

    # ── traversal helpers ───────────────────────────────────────────────────

    def _neighbors(
        self,
        graph: nx.MultiDiGraph,
        node_id: str,
        allowed: set[EdgeType],
        hub_degree_threshold: int,
        total_chunks: int,
    ) -> list[tuple[str, EdgeType, float]]:
        """Neighbours of ``node_id``, capped per edge type and damped for hubs.

        Edges are walked in both directions: a clause referenced *by* a strong hit
        and the clause that references it are both relevant.
        """
        by_type: dict[EdgeType, list[tuple[str, float]]] = {}
        # Descending out of a waypoint is dereferencing a pointer, not a semantic
        # choice, so CONTAINS is always available there. That is what lets a
        # REFERENCES-only expansion actually reach the text of the clause it was
        # pointed at, instead of dead-ending on the clause node.
        from_waypoint = not self._is_chunk_node(graph, node_id)

        for _, target, key, data in graph.out_edges(node_id, keys=True, data=True):
            self._collect(
                by_type, allowed, key, data, target, graph, hub_degree_threshold,
                total_chunks, from_waypoint, is_reverse=False,
            )
        for source, _, key, data in graph.in_edges(node_id, keys=True, data=True):
            spec = EDGE_SPECS.get(self._as_edge_type(key))
            if spec is None or not spec.traverse_reverse:
                continue
            self._collect(
                by_type, allowed, key, data, source, graph, hub_degree_threshold,
                total_chunks, from_waypoint, is_reverse=True,
            )

        out: list[tuple[str, EdgeType, float]] = []
        for edge_type, candidates in by_type.items():
            spec = EDGE_SPECS[edge_type]
            candidates.sort(key=lambda item: item[1], reverse=True)
            for neighbor, weight in candidates[: spec.max_neighbors]:
                out.append((neighbor, edge_type, weight))
        return out

    def _collect(
        self,
        by_type: dict[EdgeType, list[tuple[str, float]]],
        allowed: set[EdgeType],
        key: str,
        data: dict[str, Any],
        neighbor: str,
        graph: nx.MultiDiGraph,
        hub_degree_threshold: int,
        total_chunks: int,
        from_waypoint: bool = False,
        is_reverse: bool = False,
    ) -> None:
        edge_type = self._as_edge_type(key)
        if edge_type is None:
            return

        # Descending out of a waypoint dereferences a pointer: it is always
        # available, and free. That is what lets a REFERENCES-only expansion read
        # the clause it was pointed at instead of dead-ending on the clause node.
        # Climbing (reverse CONTAINS) is broadening, not dereferencing, so it stays
        # subject to the edge-type filter and pays full weight — otherwise a walk
        # could ascend to the document root and descend into every other section.
        free_descent = from_waypoint and not is_reverse and edge_type is EdgeType.CONTAINS
        if edge_type not in allowed and not free_descent:
            return

        neighbor_data = graph.nodes.get(neighbor, {})
        node_type = neighbor_data.get("node_type")

        # Hub suppression. A generic entity ("company") linking hundreds of chunks
        # is worthless as a bridge and expensive to walk. The node stays in the
        # graph; it just stops being traversable.
        if node_type in _BRIDGE_NODE_TYPES:
            degree = graph.degree(neighbor)
            if degree > hub_degree_threshold:
                return

        weight = float(data.get("weight", 1.0)) * float(data.get("confidence", 1.0))

        # IDF damping for shared-vocabulary bridges, computed from live degree
        # rather than baked in at build time — otherwise it goes stale as soon as
        # another document mentions the same entity.
        if node_type in _BRIDGE_NODE_TYPES and total_chunks > 1:
            mentions = max(1, len(neighbor_data.get("chunk_ids") or []))
            idf = math.log(total_chunks / mentions) / math.log(total_chunks)
            weight *= max(0.0, min(1.0, idf))

        if free_descent:
            weight = 1.0

        if weight <= 0.0:
            return
        by_type.setdefault(edge_type, []).append((neighbor, weight))

    @staticmethod
    def _as_edge_type(key: str) -> EdgeType | None:
        try:
            return EdgeType(key)
        except ValueError:
            return None

    @staticmethod
    def _is_chunk_node(graph: nx.MultiDiGraph, node_id: str) -> bool:
        return graph.nodes.get(node_id, {}).get("node_type") == NodeType.CHUNK.value

    @staticmethod
    def _chunk_id_of(graph: nx.MultiDiGraph, node_id: str) -> str | None:
        """Chunk id for a CHUNK node; None for structural and bridge nodes.

        Only chunk nodes become candidates — a SECTION or ENTITY node is a waypoint,
        not something that can be put in front of the model.
        """
        data = graph.nodes.get(node_id, {})
        if data.get("node_type") != NodeType.CHUNK.value:
            return None
        chunk_ids = data.get("chunk_ids") or []
        return chunk_ids[0] if chunk_ids else None

    @staticmethod
    def _chunk_node_count(graph: nx.MultiDiGraph) -> int:
        return sum(
            1
            for _, data in graph.nodes(data=True)
            if data.get("node_type") == NodeType.CHUNK.value
        )

    def _materialize(
        self,
        best: dict[str, tuple[float, int, str]],
        weakest_seed: float,
        max_expanded: int,
    ) -> list[ScoredChunk]:
        """Resolve chunk ids to ScoredChunks, clamped below the weakest seed."""
        ranked = sorted(best.items(), key=lambda item: item[1][0], reverse=True)
        # Strictly below the weakest seed, so expansion can never outrank a chunk
        # that dense or BM25 actually retrieved.
        ceiling = weakest_seed * 0.999 if weakest_seed > 0 else 0.0

        out: list[ScoredChunk] = []
        for chunk_id, (score, hops, path) in ranked:
            if len(out) >= max_expanded:
                break
            chunk = self.docstore.get(chunk_id)
            if chunk is None:
                continue
            capped = min(score, ceiling) if ceiling > 0 else score
            out.append(
                ScoredChunk(
                    chunk=chunk,
                    score=capped,
                    graph_score=score,
                    graph_hops=hops,
                    graph_path=path,
                )
            )
        return out


# Node types that exist to connect chunks rather than to be retrieved.
_BRIDGE_NODE_TYPES = {
    NodeType.ENTITY.value,
    NodeType.TOPIC.value,
    NodeType.POLICY.value,
}


__all__ = ["GraphRetriever", "GraphExpansionResult"]
