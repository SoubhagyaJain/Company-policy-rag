from __future__ import annotations

import hashlib
import math
from typing import Any

from backend.utils.logging import logger


def normalize_vector(vector: list[float]) -> list[float]:
    """Normalize vector to unit length for cosine similarity calculations."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


import threading
from collections import OrderedDict


class EmbeddingCache:
    """In-memory + hash-based thread-safe LRU embedding cache."""

    def __init__(self, max_size: int = 10000) -> None:
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> list[float] | None:
        key = self._hash_text(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def set(self, text: str, embedding: list[float]) -> None:
        key = self._hash_text(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = embedding
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


_shared_embedding_model: Any | None = None
_shared_embedding_model_loaded: bool = False


def _default_query_instruction(model_name: str) -> str:
    """Query-side instruction for asymmetric retrieval models.

    BGE English v1.x retrieval models are trained to embed *queries* with a
    short instruction while *passages* are embedded without one. Omitting it on
    the query side measurably degrades dense recall. Passage embeddings
    (``embed_chunks``) intentionally never receive an instruction.
    """
    name = model_name.lower()
    if "bge" in name and "en" in name:
        return "Represent this sentence for searching relevant passages: "
    return ""


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
        query_instruction: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_enabled = cache_enabled
        self.dimension = dimension
        self.cache = EmbeddingCache() if cache_enabled else None
        self._model: Any | None = None
        self._model_loaded: bool = False
        # Applied to search queries only (see ``embed_query``); auto-derived from
        # the model family unless explicitly overridden.
        self.query_instruction: str = (
            query_instruction
            if query_instruction is not None
            else _default_query_instruction(model_name)
        )

    @property
    def is_using_fallback(self) -> bool:
        """True once load was attempted and the real model is unavailable.

        When this is True, dense vectors come from the deterministic hash
        fallback and are **not** comparable to stored real-model vectors, so
        dense retrieval is effectively random. Callers should surface this as a
        degraded-retrieval signal rather than trusting the results.
        """
        return self._model_loaded and self._model is None

    def _init_model(self) -> None:
        global _shared_embedding_model, _shared_embedding_model_loaded
        if self._model_loaded:
            return
        if _shared_embedding_model_loaded:
            self._model = _shared_embedding_model
            self._model_loaded = True
            return

        self._model_loaded = True
        try:
            import os

            from sentence_transformers import SentenceTransformer  # type: ignore
            logger.info("Loading embedding model %s", self.model_name)
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            # Pin embeddings to CPU by default. On a 6 GB GPU the ~5.1 GB text LLM
            # already owns the VRAM; letting sentence-transformers auto-select CUDA
            # would allocate a ~1 GB CUDA context during chat and steal it from the
            # LLM. Overridable via EMBEDDING_DEVICE for larger cards.
            embedding_device = os.getenv("EMBEDDING_DEVICE", "cpu")
            try:
                self._model = SentenceTransformer(self.model_name, device=embedding_device)
            except Exception as local_err:
                logger.error(
                    "Embedding model %s could not be loaded (%s). Falling back to hash "
                    "embeddings — DENSE RETRIEVAL IS DEGRADED and results are not "
                    "comparable to stored vectors.",
                    self.model_name,
                    local_err,
                )
                self._model = None
        except Exception as exc:
            logger.error(
                "Could not import/load SentenceTransformer for %s (%s). Falling back to "
                "hash embeddings — DENSE RETRIEVAL IS DEGRADED.",
                self.model_name,
                exc,
            )
            self._model = None

        _shared_embedding_model = self._model
        _shared_embedding_model_loaded = True

    def _fallback_embed(self, text: str) -> list[float]:
        """Deterministic pseudo-embedding for testing when ML packages are unavailable."""
        tokens = text.lower().split()
        vector = [0.0] * self.dimension
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            val = ((h >> 8) % 1000) / 1000.0 - 0.5
            vector[idx] += val
        return normalize_vector(vector)

    def embed_text(self, text: str) -> list[float]:
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

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query for dense retrieval.

        Applies the model's query-side instruction (BGE asymmetric search) so
        the query vector matches instruction-free passage vectors produced by
        ``embed_chunks``. For symmetric uses (e.g. query-to-query cache
        matching) call ``embed_text`` instead. When no instruction is
        configured this is identical to ``embed_text``.
        """
        if not text:
            return [0.0] * self.dimension
        if self.query_instruction:
            return self.embed_text(f"{self.query_instruction}{text}")
        return self.embed_text(text)

    def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        """Batch embed a list of document chunk texts."""
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

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
            computed_vectors: list[list[float]] = []
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
