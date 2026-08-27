---
type: "query"
date: "2026-08-27T02:13:28.629533+00:00"
question: "What are the true active chat request and document ingestion lifecycles, and which components are on the critical path?"
contributor: "graphify"
source_nodes: ["ChatService", "RAGPipeline", "DocumentService", "ChromaVectorStore", "AdaptiveChunker"]
---

# Q: What are the true active chat request and document ingestion lifecycles, and which components are on the critical path?

## Answer

The active FastAPI chat path is backend/api/routes/chat.py -> backend/services/chat_service.py -> backend/rag/pipeline.py, which performs routing, scoped hybrid retrieval, reranking, context construction, generation, verification, citations, telemetry, and SSE delivery. The active ingestion path is backend/api/routes/documents.py -> backend/services/document_service.py -> loader/chunker/embedding/vector/BM25 indexes. ChatService, RAGPipeline, DocumentService, ChromaVectorStore, BM25Index, and the citation/verifier path are critical.

## Source Nodes

- ChatService
- RAGPipeline
- DocumentService
- ChromaVectorStore
- AdaptiveChunker