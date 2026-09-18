from __future__ import annotations

import time
from typing import Any

from backend.models.rag import ScoredChunk
from backend.utils.logging import logger


class RelativeScoreThresholdPostprocessor:
    """
    Filters candidate chunks scoring below `min_ratio` of the top reranker score.
    Adaptive per-query cutoff: top_score * min_ratio.
    Example: top score = 0.9, min_ratio = 0.45 -> drop chunks with score < 0.405.
    sentence-transformers CrossEncoder applies a sigmoid for single-label models,
    so bge-reranker scores arrive in [0, 1], not as raw logits.
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


# Loaded cross-encoders, keyed by (model_name, device, max_length). A single
# shared slot would hand a second reranker the first one's model whatever
# model_name it asked for.
_shared_reranker_models: dict[tuple[str, str, int], Any] = {}

# Cap on ids recorded per list in a rerank trace.
_TRACE_LIMIT = 100


class CrossEncoderReranker:
    """
    BAAI/bge-reranker cross-encoder reranker wrapper with device auto-detection,
    relative score threshold filtering, and missing dependency fallback.
    """

    # Pipelines pass a per-request ``trace`` dict only to rerankers that set this.
    supports_stage_trace = True

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        top_n: int = 5,
        device: str = "auto",
        min_ratio: float = 0.45,
        max_length: int = 512,
        pool_size: int | None = None,
        score_filter_enabled: bool = True,
        min_keep: int = 1,
        batch_size: int = 16,
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
        # None or 0 = legacy max(top_n * 4, 20); negative = every candidate.
        self.pool_size = pool_size
        self.score_filter_enabled = score_filter_enabled
        self.batch_size = max(1, int(batch_size))
        self.postprocessor = RelativeScoreThresholdPostprocessor(min_ratio=min_ratio, min_keep=min_keep)
        self._model = None
        self._model_loaded = False

    @property
    def min_keep(self) -> int:
        return self.postprocessor.min_keep

    @min_keep.setter
    def min_keep(self, value: int) -> None:
        self.postprocessor.min_keep = value

    def _init_model(self) -> None:
        if self._model_loaded:
            return

        self._model_loaded = True
        try:
            import os
            if self.device in ("cpu", "cuda"):
                dev = self.device
            else:
                import torch  # type: ignore
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            key = (self.model_name, dev, self.max_length)
            if key in _shared_reranker_models:
                self._model = _shared_reranker_models[key]
                return
            import torch  # type: ignore
            from sentence_transformers import CrossEncoder  # type: ignore
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
            _shared_reranker_models[key] = self._model
        except Exception as exc:
            logger.warning("Failed to load CrossEncoder reranker (%s). Fallback ranking enabled.", exc)
            self._model = None

    def resolve_pool_size(self, top_n: int, candidate_count: int) -> int:
        """Number of leading candidates the cross-encoder will score."""
        if self.pool_size is None or self.pool_size == 0:
            pool = max(top_n * 4, 20)
        elif self.pool_size < 0:
            pool = candidate_count
        else:
            pool = self.pool_size
        return min(candidate_count, pool)

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_n: int | None = None,
        min_ratio: float | None = None,
        trace: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Rerank candidate chunks using cross-encoder scoring and relative thresholding.

        When ``trace`` is given it receives the scored pool, the full ranking
        before the score filter, the ranking after it, and model timing.
        """
        if not candidates:
            return []

        self._init_model()

        effective_top_n = self.top_n if top_n is None else top_n
        effective_min_ratio = self.min_ratio if min_ratio is None else min_ratio

        candidate_pool_limit = self.resolve_pool_size(effective_top_n, len(candidates))
        candidates_to_rerank = candidates[:candidate_pool_limit]
        model_ms = 0.0
        fallback = self._model is None
        if self._model is not None:
            try:
                import torch
                # Pass full chunk text; the cross-encoder truncates by tokens
                # (self.max_length) rather than losing mid-chunk clauses to a
                # fixed character cut.
                pairs = [[query, (c.chunk.text or "")] for c in candidates_to_rerank]
                logger.info("Starting CrossEncoder prediction for %d pairs...", len(pairs))
                t0 = time.perf_counter()
                with torch.inference_mode():
                    logits = self._model.predict(
                        pairs,
                        batch_size=min(len(pairs), self.batch_size),
                        show_progress_bar=False,
                    )
                model_ms = (time.perf_counter() - t0) * 1000
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
                fallback = True
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

        if self.score_filter_enabled:
            filtered = self.postprocessor.filter(reranked_candidates, min_ratio=effective_min_ratio)
        else:
            filtered = reranked_candidates
        result = filtered[: effective_top_n]
        for rank, sc in enumerate(result, start=1):
            sc.rank = rank

        if trace is not None:
            trace.update(
                {
                    "query": query,
                    "model": self.model_name,
                    "fallback": fallback,
                    "candidate_count": len(candidates),
                    "pool_size": len(candidates_to_rerank),
                    "pool": [c.chunk.id for c in candidates_to_rerank][:_TRACE_LIMIT],
                    "prefilter": [c.chunk.id for c in reranked_candidates][:_TRACE_LIMIT],
                    "scores": {
                        c.chunk.id: round(float(c.rerank_score or 0.0), 6)
                        for c in reranked_candidates[:_TRACE_LIMIT]
                    },
                    "score_filter_enabled": self.score_filter_enabled,
                    "min_ratio": effective_min_ratio,
                    "min_keep": self.postprocessor.min_keep,
                    "postfilter": [c.chunk.id for c in filtered][:_TRACE_LIMIT],
                    "returned": [c.chunk.id for c in result],
                    "model_ms": round(model_ms, 2),
                }
            )
        return result
