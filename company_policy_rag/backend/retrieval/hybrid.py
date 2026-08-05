from __future__ import annotations

from typing import Any, Dict, List, Optional
from backend.models.rag import ScoredChunk
from backend.retrieval.bm25 import BM25SearchIndex
from backend.retrieval.vector import DenseVectorRetriever
from backend.utils.logging import logger


def reciprocal_rank_fusion(
    ranked_lists: List[List[ScoredChunk]],
    rrf_k: int = 60,
) -> List[ScoredChunk]:
    """
    Merges multiple ranked lists of ScoredChunks using Reciprocal Rank Fusion (RRF).
    Formula: RRF_Score(doc) = sum( 1.0 / (rrf_k + rank) ) across ranked lists.
    Preserves highest dense_score and sparse_score attributes.
    """
    rrf_scores: Dict[str, float] = {}
    best_chunks: Dict[str, ScoredChunk] = {}
    dense_scores: Dict[str, Optional[float]] = {}
    sparse_scores: Dict[str, Optional[float]] = {}

    for node_list in ranked_lists:
        for rank, sc in enumerate(node_list, start=1):
            cid = sc.chunk.id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
            if cid not in best_chunks:
                best_chunks[cid] = sc

            if sc.dense_score is not None:
                if cid not in dense_scores or dense_scores[cid] is None or sc.dense_score > (dense_scores[cid] or 0.0):
                    dense_scores[cid] = sc.dense_score
            if sc.sparse_score is not None:
                if cid not in sparse_scores or sparse_scores[cid] is None or sc.sparse_score > (sparse_scores[cid] or 0.0):
                    sparse_scores[cid] = sc.sparse_score

    if not rrf_scores:
        return []

    fused: List[ScoredChunk] = []
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    for rank, (cid, score) in enumerate(sorted_items, start=1):
        orig = best_chunks[cid]
        fused.append(
            ScoredChunk(
                chunk=orig.chunk,
                score=score,
                dense_score=dense_scores.get(cid, orig.dense_score),
                sparse_score=sparse_scores.get(cid, orig.sparse_score),
                rank=rank,
            )
        )

    return fused


class HybridRetriever:
    """
    Executes parallel dense vector and BM25 lexical searches and merges results via RRF.
    """

    def __init__(
        self,
        dense_retriever: DenseVectorRetriever,
        bm25_index: BM25SearchIndex,
        rrf_k: int = 60,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_index = bm25_index
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        dense_top_k: int = 25,
        bm25_top_k: int = 25,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[ScoredChunk]:
        """Execute hybrid search with Reciprocal Rank Fusion."""
        if not query.strip():
            return []

        dense_hits = self.dense_retriever.retrieve(query, top_k=dense_top_k, filters=filters)
        bm25_hits = self.bm25_index.search(query, top_k=bm25_top_k, filters=filters)

        if not bm25_hits:
            logger.debug("BM25 returned 0 hits; returning dense hits only.")
            for rank, sc in enumerate(dense_hits, start=1):
                sc.rank = rank
            return dense_hits

        if not dense_hits:
            logger.debug("Dense retriever returned 0 hits; returning BM25 hits only.")
            for rank, sc in enumerate(bm25_hits, start=1):
                sc.rank = rank
            return bm25_hits

        fused = reciprocal_rank_fusion([dense_hits, bm25_hits], rrf_k=self.rrf_k)
        logger.debug("Hybrid search fused %d dense + %d BM25 -> %d chunks", len(dense_hits), len(bm25_hits), len(fused))
        return fused
