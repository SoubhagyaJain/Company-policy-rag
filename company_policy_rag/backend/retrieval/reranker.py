from __future__ import annotations

from typing import Any

from backend.models.rag import ScoredChunk
from backend.utils.logging import logger


class RelativeScoreThresholdPostprocessor:
    """
    Filters candidate chunks scoring below `min_ratio` of top reranker logit score.
    Adaptive per-query cutoff: top_score * min_ratio.
    Example: Top logit = 8.0, min_ratio = 0.45 -> drop chunks with score < 3.6.
    """

    def __init__(self, min_ratio: float = 0.45, min_keep: int = 1) -> None:
        self.min_ratio = min_ratio
        self.min_keep = min_keep

    def filter(self, chunks: list[ScoredChunk], min_ratio: float | None = None) -> list[ScoredChunk]:
        if not chunks:
            return chunks

        effective_ratio = self.min_ratio if min_ratio is None else min_ratio
        scored = [c for c in chunks if c.rerank_score is not None]
        if not scored:
            # Fall back to using candidate .score
            scored = chunks

        top_score = max((c.rerank_score if c.rerank_score is not None else c.score) for c in scored)
        if top_score <= 0:
            sorted_chunks = sorted(chunks, key=lambda c: c.rerank_score if c.rerank_score is not None else c.score, reverse=True)
            return sorted_chunks[: self.min_keep]

        threshold = top_score * effective_ratio
        filtered = [
            c for c in chunks
            if (c.rerank_score if c.rerank_score is not None else c.score) >= threshold
        ]

        if len(filtered) < self.min_keep:
            filtered = sorted(
                chunks,
                key=lambda c: c.rerank_score if c.rerank_score is not None else c.score,
                reverse=True,
            )[: self.min_keep]

        logger.debug(
            "RelativeScoreThresholdPostprocessor: kept %d/%d chunks (top=%.3f, threshold=%.3f, ratio=%.2f)",
            len(filtered),
            len(chunks),
            top_score,
            threshold,
            effective_ratio,
        )
        return filtered


_shared_reranker_model: Any | None = None
_shared_reranker_model_loaded: bool = False


class CrossEncoderReranker:
    """
    BAAI/bge-reranker-large cross-encoder reranker wrapper with device auto-detection,
    relative score threshold filtering, and missing dependency fallback.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        top_n: int = 5,
        device: str = "auto",
        min_ratio: float = 0.45,
        max_length: int = 512,
        pool_size: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.top_n = top_n
        self.device = device
        self.min_ratio = min_ratio
        # Token budget the cross-encoder sees per (query, chunk) pair. bge-reranker
        # handles 512 tokens; the model truncates internally, so we pass full chunk
        # text instead of a lossy character cut that hid mid-chunk governing clauses.
        self.max_length = max_length
        # How many top fused candidates to score. Wider than top_n so a chunk the
        # RRF stage ranked mid-list can still be promoted by the reranker.
        self.pool_size = pool_size
        self.postprocessor = RelativeScoreThresholdPostprocessor(min_ratio=min_ratio)
        self._model = None
        self._model_loaded = False

    def _init_model(self) -> None:
        global _shared_reranker_model, _shared_reranker_model_loaded
        if self._model_loaded:
            return
        if _shared_reranker_model_loaded:
            self._model = _shared_reranker_model
            self._model_loaded = True
            return

        self._model_loaded = True
        try:
            import os
            import torch  # type: ignore
            from sentence_transformers import CrossEncoder  # type: ignore
            dev = "cuda" if (self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available())) else "cpu"
            logger.info("Loading CrossEncoder reranker model %s on device %s", self.model_name, dev)
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:
                num_cpus = os.cpu_count() or 8
                torch.set_num_threads(num_cpus)
                self._model = CrossEncoder(self.model_name, device=dev, max_length=self.max_length)
            except Exception as local_err:
                logger.info("Local cached reranker model not found (%s). Fallback ranking enabled.", local_err)
                self._model = None
        except Exception as exc:
            logger.warning("Failed to load CrossEncoder reranker (%s). Fallback ranking enabled.", exc)
            self._model = None

        _shared_reranker_model = self._model
        _shared_reranker_model_loaded = True

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_n: int | None = None,
        min_ratio: float | None = None,
    ) -> list[ScoredChunk]:
        """Rerank candidate chunks using cross-encoder logit scoring and relative thresholding."""
        if not candidates:
            return []

        self._init_model()

        effective_top_n = self.top_n if top_n is None else top_n
        effective_min_ratio = self.min_ratio if min_ratio is None else min_ratio

        pool = self.pool_size if self.pool_size is not None else max(effective_top_n * 4, 20)
        candidate_pool_limit = min(len(candidates), pool)
        candidates_to_rerank = candidates[:candidate_pool_limit]
        if self._model is not None:
            try:
                import torch
                # Pass full chunk text; the cross-encoder truncates by tokens
                # (self.max_length) rather than losing mid-chunk clauses to a
                # fixed character cut.
                pairs = [[query, (c.chunk.text or "")] for c in candidates_to_rerank]
                logger.info("Starting CrossEncoder prediction for %d pairs...", len(pairs))
                with torch.inference_mode():
                    logits = self._model.predict(
                        pairs,
                        batch_size=min(len(pairs), 16),
                        show_progress_bar=False,
                    )
                logger.info("CrossEncoder prediction complete.")
                if hasattr(logits, "tolist"):
                    logits = logits.tolist()

                reranked_candidates: list[ScoredChunk] = []
                for sc, logit in zip(candidates_to_rerank, logits):
                    score_val = float(logit)
                    reranked_candidates.append(
                        ScoredChunk(
                            chunk=sc.chunk,
                            score=score_val,
                            rerank_score=score_val,
                            sparse_score=sc.sparse_score,
                            dense_score=sc.dense_score,
                        )
                    )
                reranked_candidates.sort(key=lambda c: c.rerank_score or -999.0, reverse=True)
            except Exception as exc:
                logger.warning("CrossEncoder prediction error (%s). Using candidate scores.", exc)
                reranked_candidates = [
                    ScoredChunk(
                        chunk=sc.chunk,
                        score=sc.score,
                        rerank_score=sc.score,
                        sparse_score=sc.sparse_score,
                        dense_score=sc.dense_score,
                    ) for sc in candidates_to_rerank
                ]
                reranked_candidates.sort(key=lambda c: c.score, reverse=True)
        else:
            # Fallback when CrossEncoder is missing
            reranked_candidates = [
                ScoredChunk(
                    chunk=sc.chunk,
                    score=sc.score,
                    rerank_score=sc.score,
                    sparse_score=sc.sparse_score,
                    dense_score=sc.dense_score,
                ) for sc in candidates_to_rerank
            ]
            reranked_candidates.sort(key=lambda c: c.score, reverse=True)

        filtered = self.postprocessor.filter(reranked_candidates, min_ratio=effective_min_ratio)
        result = filtered[: effective_top_n]
        for rank, sc in enumerate(result, start=1):
            sc.rank = rank
        return result
