from __future__ import annotations

from typing import Any

from backend.models.rag import ScoredChunk
from backend.retrieval.bm25 import BM25SearchIndex
from backend.retrieval.reranker import CrossEncoderReranker
from backend.retrieval.vector import DenseVectorRetriever
from backend.utils.logging import logger


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredChunk]],
    rrf_k: int = 60,
) -> list[ScoredChunk]:
    """
    Merges multiple ranked lists of ScoredChunks using Reciprocal Rank Fusion (RRF).
    Formula: RRF_Score(doc) = sum( 1.0 / (rrf_k + rank) ) across ranked lists.
    Preserves highest dense_score and sparse_score attributes.
    """
    rrf_scores: dict[str, float] = {}
    best_chunks: dict[str, ScoredChunk] = {}
    dense_scores: dict[str, float | None] = {}
    sparse_scores: dict[str, float | None] = {}

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

    fused: list[ScoredChunk] = []
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
        reranker: CrossEncoderReranker | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_index = bm25_index
        self.reranker = reranker or CrossEncoderReranker()
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        dense_top_k: int = 25,
        bm25_top_k: int = 25,
        filters: dict[str, Any] | None = None,
        rrf_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Execute hybrid search with Reciprocal Rank Fusion."""
        if not query.strip():
            return []

        effective_rrf_k = rrf_k if rrf_k is not None else self.rrf_k

        logger.info(f"Executing dense retrieval for query: {query}")
        dense_hits = self.dense_retriever.retrieve(query, top_k=dense_top_k, filters=filters)
        
        logger.info(f"Executing BM25 retrieval for query: {query}")
        bm25_hits = self.bm25_index.search(query, top_k=bm25_top_k, filters=filters)
        logger.info("BM25 retrieval complete")

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

        fused = reciprocal_rank_fusion([dense_hits, bm25_hits], rrf_k=effective_rrf_k)
        logger.debug("Hybrid search fused %d dense + %d BM25 -> %d chunks (rrf_k=%d)", len(dense_hits), len(bm25_hits), len(fused), effective_rrf_k)
        return fused
