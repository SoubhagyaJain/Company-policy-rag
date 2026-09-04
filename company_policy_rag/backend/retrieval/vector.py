from __future__ import annotations

from typing import Any

from backend.embeddings.embeddings import EmbeddingService
from backend.embeddings.vector_store import VectorStoreInterface
from backend.models.rag import ScoredChunk


class DenseVectorRetriever:
    """
    Dense vector retriever executing similarity search via EmbeddingService and VectorStoreInterface.
    """

    def __init__(
        self,
        vector_store: VectorStoreInterface,
        embedding_service: EmbeddingService,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    def retrieve(
        self,
        query: str,
        top_k: int = 25,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Generate query vector embedding and perform top-k vector store similarity search."""
        if not query.strip():
            return []
        from backend.utils.logging import logger
        logger.info(f"Dense retrieve: computing embedding for {query}")
        # Query-side instruction (asymmetric retrieval); passages are embedded
        # without one at index time. Falls back to plain embedding for models
        # that define no instruction.
        query_emb = self.embedding_service.embed_query(query)
        logger.info("Dense retrieve: embedding computed, querying vector store")
        res = self.vector_store.search(query_emb, top_k=top_k, filters=filters)
        logger.info("Dense retrieve: vector store query complete")
        return res
