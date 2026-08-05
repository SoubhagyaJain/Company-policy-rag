from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional
from backend.utils.logging import logger


def normalize_vector(vector: List[float]) -> List[float]:
    """Normalize vector to unit length for cosine similarity calculations."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


class EmbeddingCache:
    """In-memory + hash-based embedding cache to avoid recomputing vector embeddings."""

    def __init__(self) -> None:
        self._cache: Dict[str, List[float]] = {}

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        key = self._hash_text(text)
        return self._cache.get(key)

    def set(self, text: str, embedding: List[float]) -> None:
        key = self._hash_text(text)
        self._cache[key] = embedding

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class EmbeddingService:
    """
    Service wrapper for dense vector embeddings with caching, batching, and normalization.
    Uses sentence-transformers / FastEmbed if available, with deterministic fallback.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_enabled: bool = True,
        dimension: int = 384,
    ) -> None:
        self.model_name = model_name
        self.cache_enabled = cache_enabled
        self.dimension = dimension
        self.cache = EmbeddingCache() if cache_enabled else None
        self._model: Optional[Any] = None
        self._model_loaded: bool = False

    def _init_model(self) -> None:
        if self._model_loaded:
            return
        self._model_loaded = True
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            logger.info("Loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:
            logger.warning("Could not load SentenceTransformer model %s (%s). Using fallback embedder.", self.model_name, exc)
            self._model = None

    def _fallback_embed(self, text: str) -> List[float]:
        """Deterministic pseudo-embedding for testing when ML packages are unavailable."""
        tokens = text.lower().split()
        vector = [0.0] * self.dimension
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            val = ((h >> 8) % 1000) / 1000.0 - 0.5
            vector[idx] += val
        return normalize_vector(vector)

    def embed_text(self, text: str) -> List[float]:
        """Embed a single query or text string."""
        if not text:
            return [0.0] * self.dimension

        if self.cache is not None:
            cached = self.cache.get(text)
            if cached is not None:
                return cached

        self._init_model()
        if self._model is not None:
            try:
                raw_emb = self._model.encode(text, convert_to_numpy=True)
                vector = normalize_vector(raw_emb.tolist())
            except Exception as exc:
                logger.warning("Embedding failed for text: %s. Using fallback.", exc)
                vector = self._fallback_embed(text)
        else:
            vector = self._fallback_embed(text)

        if self.cache is not None:
            self.cache.set(text, vector)
        return vector

    def embed_chunks(self, texts: List[str]) -> List[List[float]]:
        """Batch embed a list of document chunk texts."""
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        missing_indices: List[int] = []
        missing_texts: List[str] = []

        for idx, text in enumerate(texts):
            if self.cache is not None:
                cached = self.cache.get(text)
                if cached is not None:
                    results[idx] = cached
                    continue
            missing_indices.append(idx)
            missing_texts.append(text)

        if missing_texts:
            self._init_model()
            computed_vectors: List[List[float]] = []
            if self._model is not None:
                try:
                    raw_embs = self._model.encode(missing_texts, convert_to_numpy=True)
                    computed_vectors = [normalize_vector(emb.tolist()) for emb in raw_embs]
                except Exception as exc:
                    logger.warning("Batch embedding failed (%s). Using fallback.", exc)
                    computed_vectors = [self._fallback_embed(t) for t in missing_texts]
            else:
                computed_vectors = [self._fallback_embed(t) for t in missing_texts]

            for idx, text, vec in zip(missing_indices, missing_texts, computed_vectors):
                results[idx] = vec
                if self.cache is not None:
                    self.cache.set(text, vec)

        return [vec if vec is not None else [0.0] * self.dimension for vec in results]
