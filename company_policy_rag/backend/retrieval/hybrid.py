from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
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


def _word_count(sc: ScoredChunk) -> int:
    return len((sc.chunk.text or "").split())


def _timed(fn, *args, **kwargs) -> tuple[Any, float]:
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000


class HybridRetriever:
    """
    Executes parallel dense vector and BM25 lexical searches and merges results via RRF.
    """

    # Pipelines pass a per-sub-query ``trace`` dict only to retrievers that set this.
    supports_stage_trace = True

    def __init__(
        self,
        dense_retriever: DenseVectorRetriever,
        bm25_index: BM25SearchIndex,
        reranker: CrossEncoderReranker | None = None,
        rrf_k: int = 60,
        min_chunk_words: int = 0,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_index = bm25_index
        self.reranker = reranker or CrossEncoderReranker()
        self.rrf_k = rrf_k
        # Chunks with this many words or fewer (title pages, bare headings) are
        # skipped at query time. 0 keeps every chunk.
        self.min_chunk_words = min_chunk_words

    def _warm_dense_model(self) -> None:
        """Force the embedding model to load on the calling thread (idempotent)."""
        embedding_service = getattr(self.dense_retriever, "embedding_service", None)
        init = getattr(embedding_service, "_init_model", None)
        if callable(init):
            try:
                init()
            except Exception:
                pass

    def retrieve(
        self,
        query: str,
        dense_top_k: int = 25,
        bm25_top_k: int = 25,
        filters: dict[str, Any] | None = None,
        rrf_k: int | None = None,
        trace: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Execute hybrid search with Reciprocal Rank Fusion.

        When ``trace`` is given it receives the dense, BM25, and fused rankings
        (chunk ids) and per-index latency.
        """
        if not query.strip():
            return []

        effective_rrf_k = rrf_k if rrf_k is not None else self.rrf_k
        min_words = max(0, int(self.min_chunk_words or 0))
        # Over-fetch when short chunks are skipped so each list still fills its depth.
        dense_fetch = dense_top_k * 2 + 10 if min_words else dense_top_k
        bm25_fetch = bm25_top_k * 2 + 10 if min_words else bm25_top_k

        # Load the embedding model on THIS (calling) thread before fanning out:
        # the first `import sentence_transformers` pulls in native libs (pyarrow)
        # that can crash if first imported inside a worker thread on some
        # platforms. Idempotent once loaded.
        self._warm_dense_model()

        # Dense and BM25 are independent and each releases the GIL during its
        # heavy work (torch inference / native scoring / Chroma I/O), so running
        # them concurrently overlaps rather than serializes. Fusion is unchanged,
        # and an exception in either propagates identically to the sequential
        # version (callers fall back to a single index on error).
        logger.info(f"Executing dense + BM25 retrieval for query: {query}")
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(
                _timed, self.dense_retriever.retrieve, query, top_k=dense_fetch, filters=filters
            )
            bm25_future = executor.submit(
                _timed, self.bm25_index.search, query, top_k=bm25_fetch, filters=filters
            )
            dense_hits, dense_ms = dense_future.result()
            bm25_hits, bm25_ms = bm25_future.result()
        logger.info("Dense + BM25 retrieval complete")

        if min_words:
            dense_hits = [sc for sc in dense_hits if _word_count(sc) > min_words][:dense_top_k]
            bm25_hits = [sc for sc in bm25_hits if _word_count(sc) > min_words][:bm25_top_k]

        if trace is not None:
            trace.update(
                {
                    "query": query,
                    "dense": [sc.chunk.id for sc in dense_hits],
                    "bm25": [sc.chunk.id for sc in bm25_hits],
                    "dense_ms": round(dense_ms, 2),
                    "bm25_ms": round(bm25_ms, 2),
                }
            )

        if not bm25_hits or not dense_hits:
            # One index came back empty. Score the surviving list on the RRF
            # scale anyway: callers merge hits from several sub-queries by score,
            # and a raw cosine (0-1) or BM25 (unbounded) score would outrank
            # every fused 1/(k + rank) score regardless of relevance.
            single = dense_hits or bm25_hits
            logger.debug(
                "%s returned 0 hits; ranking %s hits alone.",
                "BM25" if not bm25_hits else "Dense retriever",
                "dense" if not bm25_hits else "BM25",
            )
            fused = reciprocal_rank_fusion([single], rrf_k=effective_rrf_k) if single else []
        else:
            fused = reciprocal_rank_fusion([dense_hits, bm25_hits], rrf_k=effective_rrf_k)
            logger.debug("Hybrid search fused %d dense + %d BM25 -> %d chunks (rrf_k=%d)", len(dense_hits), len(bm25_hits), len(fused), effective_rrf_k)
        if trace is not None:
            trace["fused"] = [sc.chunk.id for sc in fused]
        return fused
