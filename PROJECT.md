# PROJECT: Enterprise-Grade RAG System (`company_policy_rag`)

## Architecture
Clean modular layout separating backend, frontend, shared schemas, tests, and docker deployment:
```
c:\Users\jains\OneDrive\Desktop\Rag-chatbot\company_policy_rag\
├── backend/
│   ├── api/          # FastAPI routers (/api/chat, /api/chat/stream, /api/documents, /api/admin, /api/health)
│   ├── services/     # Business logic & orchestrators (chat_service, document_service, eval_service)
│   ├── rag/          # Core RAG pipeline, multi-query, query rewriting, context compression, citations, semantic_cache
│   ├── retrieval/    # Hybrid retriever (BM25 + Dense vector, RRF), reranking (bge-reranker), metadata filters
│   ├── embeddings/   # Dense vector embedding providers & local cache
│   ├── ingestion/    # Multi-format loaders (PDF, DOCX, TXT, MD, HTML, CSV, JSON) & adaptive chunkers
│   ├── models/       # Pydantic schemas, data models, API DTOs
│   ├── evaluation/   # Faithfulness & Answer Relevancy LLM-as-judge evaluation engine
│   └── utils/        # Telemetry, structured logging, Redis cache, helpers
├── frontend/
│   ├── app/          # Next.js 15 App Router (pages, layout, streaming API routes)
│   ├── components/   # Anthropic cream UI, liquid glass components, Chat, Citations, DocumentManager
│   ├── hooks/        # Custom React hooks (useChatStream, useDocuments, useSessions)
│   ├── lib/          # API client, markdown components, tailwind config, utils
│   └── styles/       # Cream palette, liquid glass backdrop blur styles, global CSS
├── shared/           # Common type definitions and schema specs
├── tests/            # Pytest test suite for backend unit & integration testing
│   ├── unit/
│   └── integration/
├── docker/           # Production Dockerfiles and docker-compose.yml
├── data/             # Document storage and evaluation golden datasets
└── scripts/          # Evaluation scripts (evaluate.py) and CLI tools
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Multi-format Document Ingestion | Load PDF, DOCX, TXT, MD, HTML, CSV, JSON preserving page numbers, hierarchy & tables | M1 | Survey (R1) |
| 2 | Adaptive Multi-Strategy Chunking | Recursive, Semantic, Markdown, Heading-aware, Table-aware automatic chunk selection | M1 | Survey (R1) |
| 3 | Hybrid Dense + Sparse Retrieval | BM25 + Vector Search with Reciprocal Rank Fusion (RRF) & metadata filtering | M2 | Survey (R3) |
| 4 | Advanced RAG Query Engine | Multi-query decomposition, query rewriting, context compression, parent expansion | M2 | Survey (R3) |
| 5 | Cross-Encoder Reranking & Citations | BAAI bge-reranker-large score filtering and structured verifiable citations | M2 | Survey (R1, R3) |
| 6 | FastAPI Backend Routes & Streaming | Sub-1s TTFT SSE streaming chat, document upload/management, health & models endpoints | M3 | Survey (R1) |
| 7 | Admin & Observability Telemetry | Endpoint reporting chunks, similarity scores, rerank scores, sources, tokens, latency | M3 | Survey (R1) |
| 8 | Anthropic-inspired Next.js 15 UI | Cream palette (`#FAF9F5`), liquid glass UI, dark mode, Framer Motion animations | M4 | Survey (R2) |
| 9 | Streaming Chat & Citation UX | SSE chat, markdown code highlighting, interactive citation drawer/cards | M4 | Survey (R2) |
| 10| Document Manager & Multi-session UI| Upload up to 100MB files, manage 100+ documents, multi-session sidebar history | M4 | Survey (R2) |
| 11| Redis Caching & System Performance | Response caching, embedding cache, session store, in-memory fallback | M5 | Survey (R4) |
| 12| Production Docker & CI/CD Pipeline | Multi-stage Docker compose (FastAPI, Next.js, Redis), Pytest/Jest, Pyright/mypy, ESLint | M5 | Survey (R4) |
| 13| Golden RAG Evaluation Gate | Automated evaluate.py on golden dataset verifying Faithfulness >= 0.90 & Relevancy >= 0.75 | M6 | Survey (Criteria) |
| 14| Semantic Cache Storage & Metric Config | ChromaDB `semantic_cache` collection, Cosine similarity metric mapping ($1-d$), threshold config, embedding reuse | M_SC1 | User Request (R1, R3) |
| 15| RAG Pipeline Cache Lookup Hit/Miss | Pre-rewrite lookup in `pipeline.py`, cached answers & citations, retrieval/LLM bypass, non-blocking cache write | M_SC2 | User Request (R2, R4, R5) |
| 16| Invalidation, Streaming & Concurrency | Document versioning metadata invalidation, hit token streaming simulation, live miss streaming, thread-safe writes | M_SC3 | User Request (R6, R7, R8) |
| 17| Comprehensive Cache Test Suite & Audit | 13-point automated test suite (hits, misses, threshold, streaming, invalidation, non-blocking, bypass) + Audit | M_SC4 | User Request (Testing) |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Backend Clean Layout & Ingestion Engine | Modular `backend/` structure, multi-format loaders, adaptive chunkers | None | DONE |
| M2 | Advanced Retrieval & RAG Pipeline | Hybrid search, multi-query, query rewrite, reranker, context compression, citations | M1 | DONE |
| M3 | FastAPI Web Application & Observability | API routes, SSE streaming server, admin observability & telemetry endpoints | M2 | DONE |
| M4 | Next.js 15 Anthropic-inspired UI | Next.js 15 App Router, cream aesthetic, streaming UI, citation cards, document manager | M3 | DONE |
| M5 | Production Infra, Redis & Testing | Docker compose, Redis cache, pytest/jest test suites, pyright/mypy/eslint config | M4 | DONE |
| M6 | Golden Evaluation Gate & Victory Audit | Run evaluate.py (Faithfulness >= 0.90, Relevancy >= 0.75), type checks, test suite, docker build | M5 | DONE |
| M_SC1 | Semantic Cache Storage & Metric Config | ChromaDB `semantic_cache` collection, Cosine similarity metric mapping, threshold config, $O(1)$ vector cache | None | DONE |
| M_SC2 | Cache Lookup Integration & Hit/Miss Pipeline | Pre-rewrite lookup in `pipeline.py`, hit retrieval/LLM bypass, citation tracking, non-blocking cache write | M_SC1 | PLANNED |
| M_SC3 | Cache Invalidation, Streaming & Concurrency | Document versioning invalidation, simulated SSE hit streaming, live miss streaming, thread-safe writes | M_SC2 | PLANNED |
| M_SC4 | Comprehensive Test Suite & Victory Verification | 13-scenario automated test suite, hit/miss/threshold/streaming/bypass tests, Forensic Auditor check | M_SC3 | PLANNED |

## Interface Contracts

### Backend API Endpoints (FastAPI)
- `POST /api/chat` -> `{ message: string, session_id?: string, model?: string, filters?: dict }` -> `{ id: string, answer: string, citations: Citation[], latency_ms: number, metrics: dict }`
- `POST /api/chat/stream` -> SSE stream: `event: start`, `event: chunk`, `event: citation`, `event: trace`, `event: done`
- `POST /api/documents/upload` -> `multipart/form-data` -> `{ document_id: string, filename: string, chunks_indexed: number, status: string }`
- `GET /api/documents` -> `{ documents: DocumentSummary[] }`
- `GET /api/admin/observability` -> `{ total_queries: number, avg_latency_ms: number, token_usage: dict, recent_traces: Trace[] }`
- `GET /api/health` -> `{ status: "ok", redis: boolean, vector_db: boolean, models_loaded: boolean }`

## Code Layout
Target Directory: `c:\Users\jains\OneDrive\Desktop\Rag-chatbot\company_policy_rag\`
- Code files belong exclusively in `backend/`, `frontend/`, `shared/`, `tests/`, `docker/`, `scripts/`, `data/`.
- No source code or test files in `.agents/`.
