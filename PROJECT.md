# Project: Company Policy RAG with Agentic Intelligence Layer

## Architecture

The system is a production-grade RAG chatbot for company policy Q&A with an Agentic Intelligence Layer built with FastAPI, LlamaIndex, ChromaDB, BM25 (RRF), Cross-Encoder reranking (`BAAI/bge-reranker-large`), Ollama LLMs, and a Next.js 16 / React 19 frontend.

```
[User / Next.js 16 UI]
       │
       │ POST /api/chat/stream or POST /api/chat
       ▼
[FastAPI Router: backend/api/routes/chat.py]
       │
       ▼
[ChatService: backend/services/chat_service.py]
       │ (Session TTL Cache / History resolution / SSE Stream Event Generator)
       ▼
[RAGPipeline: backend/rag/pipeline.py]
       │
       ├─► [R1: QueryRouter] ── classify (FACTUAL, COMPARISON, ENUMERATION, PROCEDURAL, CONVERSATIONAL)
       │                        └── If CONVERSATIONAL: instant greeting bypass (no DB/retrieval)
       │                        └── Select dynamic RetrievalStrategy (top_k, rerank_top_n, multi_query, etc.)
       │
       ├─► [R3: QueryMetadataInferer] ── infer department, policy_id, topic_tags, category
       │                                 └── Pre-filter hybrid retrieval (with Fallback Relaxation)
       │
       ├─► [Step 0: SemanticCacheManager] ── Cosine similarity >= 0.95 ──► Return cached answer
       │
       ├─► [Step 1-2: QueryRewriter & MultiQueryGenerator]
       │
       ├─► [Step 3: HybridRetriever] ── Dense Vector (ChromaDB) + Sparse Lexical (BM25) with RRF (k=60)
       │
       ├─► [Step 4: CrossEncoderReranker] ── BAAI/bge-reranker-large + Relative Score Thresholding
       │
       ├─► [Step 5: ContextCompressor] ── Parent Context Expansion & Formatting
       │
       ├─► [Step 6: LLM Grounded Synthesis] ── Ollama / LlamaIndex / Fallback Grounded Synthesis
       │
       ├─► [Step 7: CitationEngine] ── Verifiable bracketed [Source N] extraction
       │
       └─► [R2: SelfReflectionVerifier & RetryEngine]
             ├─ Evaluate Faithfulness, Completeness, Citation Coverage, Coherence
             ├─ If score < threshold & attempt < 2 ──► Adjust retrieval parameters & retry loop
             └─ Hard cap at 2 retries ──► Graceful fallback with low_confidence=True
       │
       ▼
[TelemetryService: backend/services/telemetry_service.py]
       │ (Records full RAGTrace with routing, verification & filter telemetry)
       ▼
[SSE Events / Stream Delivery: start -> retrieval -> chunk -> citation -> trace -> done]
```

---

## Feature Inventory

Every requirement from `ORIGINAL_REQUEST.md` and codebase survey is inventoried below with its assigned milestone:

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Query Classification | Classify incoming queries into at least 4 types (FACTUAL, COMPARISON, ENUMERATION, PROCEDURAL, CONVERSATIONAL) with measurable accuracy | M1 | ORIGINAL_REQUEST §R1 |
| 2 | Strategy Selector | Dynamic retrieval strategy selection (top_k, rerank_top_n, multi_query, parent_expansion, temp) based on query type | M1 | ORIGINAL_REQUEST §R1 |
| 3 | Conversational Bypass | Greeting and conversational queries bypass vector retrieval cleanly | M1 | ORIGINAL_REQUEST §R1 |
| 4 | Routing Observability | Decision, confidence, and strategy surfaced in SSE trace events (`event: trace`) and response metadata | M1 | ORIGINAL_REQUEST §R1 |
| 5 | Post-Generation Verification | Self-reflection verifier evaluating faithfulness, completeness, citation coverage, and coherence | M2 | ORIGINAL_REQUEST §R2 |
| 6 | Autonomous Retry Loop | Autonomous retrieval adjustment & re-synthesis when verification score is below threshold | M2 | ORIGINAL_REQUEST §R2 |
| 7 | Retry Hard Cap & Fallback | Hard cap at 2 retries with graceful fallback to prevent infinite loops | M2 | ORIGINAL_REQUEST §R2 |
| 8 | Verification Observability | Verification scores and retry attempts logged in observability traces and SSE events | M2 | ORIGINAL_REQUEST §R2 |
| 9 | Ingestion Metadata Extraction | Automatically extract category, department, effective dates, policy IDs, key entities, and topic tags during ingestion | M3 | ORIGINAL_REQUEST §R3 |
| 10 | ChromaDB Metadata Storage | Flatten and store structured metadata in ChromaDB alongside chunk metadata | M3 | ORIGINAL_REQUEST §R3 |
| 11 | Query Metadata Inference | Infer relevant metadata filters from query text and apply pre-filtering before hybrid search | M3 | ORIGINAL_REQUEST §R3 |
| 12 | Filter Fallback Relaxation | Automatically relax filters and retry retrieval if filtered search returns 0 candidates | M3 | ORIGINAL_REQUEST §R3 |
| 13 | Frontend Agentic UI | Display query classification badges, verification status indicators, 4D score progress bars, and metadata filter tags in Next.js UI | M4 | ORIGINAL_REQUEST §R4 |
| 14 | Configurable Feature Toggles | Environment variables and config flags to enable/disable agentic features | M4 | ORIGINAL_REQUEST §R4 |
| 15 | Non-Regression & Pipeline Continuity | All existing unit and integration tests pass; streaming, caching, and memory continue to work | M4 | ORIGINAL_REQUEST §R4 |
| 16 | E2E 4-Tier Test Suite | Comprehensive opaque-box test suite verifying all 4 tiers (Feature, Boundary, Combinations, Real-World Scenarios) | E2E | ORIGINAL_REQUEST Acceptance Criteria |
| 17 | Adversarial Hardening (Tier 5) | Adversarial test coverage and edge case verification | E2E | Project Quality Standards |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Test Suite Track | Build/verify 4-tier opaque-box E2E test suite (`TEST_INFRA.md`, `TEST_READY.md`) | none | DONE |
| M1 | Query Router & Strategy Selector | Query classifier, strategy mapping, conversational bypass, SSE trace integration | none | DONE |
| M2 | Self-Reflection & Verification | 4D verification evaluator, retry engine, 2-retry hard cap, telemetry trace | M1 | DONE |
| M3 | Dynamic Metadata Extraction & Filtering | Ingestion metadata extraction, ChromaDB storage, query filter inference, fallback relaxation | none | IN_PROGRESS |
| M4 | Integration, Frontend UI & Non-Regression | UI badges/indicators/tags, config toggles, enum import fix, full test suite green | M1, M2, M3 | PLANNED |
| Final | Final E2E Pass & Hardening | 100% pass on Tiers 1-4, Tier 5 adversarial hardening, Forensic Audit | E2E, M4 | PLANNED |

---

## Interface Contracts

### Query Router (`backend/rag/query_router.py`)
- `QueryCategory(str, Enum)`: `FACTUAL`, `COMPARISON`, `ENUMERATION`, `PROCEDURAL`, `CONVERSATIONAL`
- `RetrievalStrategy(BaseModel)`: `dense_top_k: int`, `bm25_top_k: int`, `rrf_k: int`, `rerank_top_n: int`, `min_score_ratio: float`, `enable_multi_query: bool`, `enable_parent_expansion: bool`, `temperature: float`
- `QueryClassification(BaseModel)`: `category: QueryCategory`, `confidence: float`, `strategy: RetrievalStrategy`, `reasoning: str`
- `QueryRouter.classify(query: str, history: list[dict] | None = None) -> QueryClassification`

### Self-Reflection Verifier (`backend/rag/verifier.py` & `backend/rag/retry_engine.py`)
- `VerificationReport(BaseModel)`: `faithfulness: float`, `completeness: float`, `citation_coverage: float`, `coherence: float`, `composite_score: float`, `passed: bool`, `critique: str`, `missing_aspects: list[str]`, `unsupported_claims: list[str]`
- `SelfReflectionVerifier.verify(query: str, answer: str, context_chunks: list[ScoredChunk], citations: list[Citation]) -> VerificationReport`
- `RetryEngine.prepare_retry(attempt: int, report: VerificationReport, current_strategy: RetrievalStrategy, query: str) -> tuple[RetrievalStrategy, str | None]` (hard capped at attempt < 2)

### Dynamic Metadata Extractor & Inferer (`backend/ingestion/metadata_extractor.py` & `backend/rag/filter_extractor.py`)
- `DocumentMetadataExtractor.extract(text: str, filename: str = "") -> ExtractedDocumentMetadata`
- `QueryMetadataInferer.infer_filters(query: str) -> dict[str, Any]` (e.g. `{"department": "IT"}`, `{"topic_tags": ["it_security"]}`)

### SSE Stream Trace Contract (`backend/services/chat_service.py` & `frontend/lib/types.ts`)
- `event: trace`: payload includes `query_type`, `routing_confidence`, `retrieval_strategy`, `inferred_filters`, `applied_filters`, `filter_relaxed`, `verification_report`, `verification_score`, `retry_count`, `retry_reasons`
- `event: done`: payload includes complete `trace`, `answer`, `citations`, `timing`, `low_confidence`

---

## Code Layout

- Backend Root: `company_policy_rag/backend/`
  - API Routes: `backend/api/routes/` (`chat.py`, `documents.py`, `admin.py`, `health.py`, `models.py`)
  - API Main & Dependencies: `backend/api/main.py`, `backend/api/dependencies.py`
  - Services: `backend/services/` (`chat_service.py`, `document_service.py`, `telemetry_service.py`)
  - RAG Core: `backend/rag/` (`pipeline.py`, `query_router.py`, `verifier.py`, `retry_engine.py`, `filter_extractor.py`, `semantic_cache.py`, `query_rewrite.py`, `context_compression.py`)
  - Retrieval & Search: `backend/retrieval/` (`hybrid.py`, `vector.py`, `bm25.py`, `reranker.py`)
  - Ingestion & Loaders: `backend/ingestion/` (`metadata_extractor.py`, `loaders/`, `chunkers/`)
  - Embeddings & Storage: `backend/embeddings/` (`embeddings.py`, `vector_store.py`)
  - Models & Schemas: `backend/models/` (`rag.py`, `api_dto.py`, `chunk.py`, `document.py`)
- Config & Core: `company_policy_rag/src/config.py`, `company_policy_rag/src/ollama_client.py`
- Frontend Root: `company_policy_rag/frontend/`
  - App Pages: `frontend/app/` (`page.tsx`, `admin/page.tsx`, `layout.tsx`)
  - Components: `frontend/components/` (`ChatMessage.tsx`, `ChatWindow.tsx`, `AdminView.tsx`, `CitationDrawer.tsx`, `CitationCard.tsx`, `Header.tsx`)
  - Hooks: `frontend/hooks/` (`useChatStream.ts`, `useObservability.ts`, `useSessions.ts`, `useDocuments.ts`)
  - Library & Types: `frontend/lib/` (`types.ts`, `api-client.ts`)
- Tests: `company_policy_rag/tests/`
  - Unit: `tests/unit/`
  - Integration: `tests/integration/`
  - E2E: `tests/e2e/` (`test_e2e_agentic_layer.py`, `test_e2e_tier1_features.py`, `test_e2e_tier2_boundaries.py`, `test_e2e_tier3_combinations.py`, `test_e2e_tier4_scenarios.py`)
