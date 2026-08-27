# Graph Report - backend  (2026-08-26)

## Corpus Check
- 84 files · ~59,482 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1125 nodes · 2829 edges · 64 communities (56 shown, 8 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 297 edges (avg confidence: 0.95)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Vector Storage
- Visual Assets
- RAG Data Models
- Chat Endpoint
- Conversation Models
- Citation Assembly
- Pipeline Response
- Thinking Trace
- Vision Page Services
- Async Redis Cache
- Admin Trace Cleanup
- Telemetry Persistence
- Document Metadata
- Telemetry Schema
- Document Scope Resolution
- Cache Operations
- Ingestion Loaders
- Telemetry Service
- API Dependencies
- Document API
- Health Endpoint
- Loader Base Classes
- Query Rewriting
- Upload Routes
- HTML Loader
- Embedding Service
- Document DTOs
- Page Identity
- Hybrid Retrieval
- Vision Cache
- Conversation State
- Logical Document Structure
- Redis Client
- Printed Page Detection
- Metadata Inference
- Chat Session Routes
- Evidence Gate
- BM25 Index
- Section Tracking
- Semantic Cache Queries
- PDF Loader
- Cross Encoder Reranking
- Celery Setup
- Embedding Cache
- Retrieval Cache
- Loader Metadata
- JSON Loader
- Telemetry Enums
- Semantic Cache Storage
- Heading Context
- Model Switching
- Text Loader
- Telemetry Errors
- Hashing Utilities
- Vision Circuit Breaker
- Model Management
- Redis Cache Service
- Redis Session Storage
- Trace Sanitization
- Dense Retrieval
- Logging Setup
- Timing Utilities
- Backend Package
- Task Package

## God Nodes (most connected - your core abstractions)
1. `TelemetryService` - 68 edges
2. `ScoredChunk` - 62 edges
3. `RAGPipeline` - 56 edges
4. `RawDocument` - 55 edges
5. `Chunk` - 51 edges
6. `DocumentService` - 45 edges
7. `ImageAssetManager` - 37 edges
8. `TelemetryDB` - 36 edges
9. `ChatService` - 33 edges
10. `BaseLoader` - 29 edges

## Surprising Connections (you probably didn't know these)
- `get_observability_summary_endpoint()` --uses--> `ObservabilitySummary`  [INFERRED]
  company_policy_rag/backend/api/routes/admin.py → company_policy_rag/backend/models/telemetry_models.py
- `get_observability_summary_endpoint()` --uses--> `DocumentService`  [INFERRED]
  company_policy_rag/backend/api/routes/admin.py → company_policy_rag/backend/services/document_service.py
- `get_observability_summary_endpoint()` --uses--> `TelemetryService`  [INFERRED]
  company_policy_rag/backend/api/routes/admin.py → company_policy_rag/backend/services/telemetry_service.py
- `get_subsystem_health_endpoint()` --uses--> `DocumentService`  [INFERRED]
  company_policy_rag/backend/api/routes/admin.py → company_policy_rag/backend/services/document_service.py
- `get_subsystem_health_endpoint()` --uses--> `TelemetryService`  [INFERRED]
  company_policy_rag/backend/api/routes/admin.py → company_policy_rag/backend/services/telemetry_service.py

## Import Cycles
- None detected.

## Communities (64 total, 8 thin omitted)

### Community 0 - "Vector Storage"
Cohesion: 0.05
Nodes (45): ChromaVectorStore, get_shared_chroma_client(), ABC, Any, Path, ChromaDB implementation of VectorStoreInterface with fallback in-memory store.…, Safely unpacks ChromaDB primitive metadata dictionary back into structured…, unpack_chroma_metadata() (+37 more)

### Community 1 - "Visual Assets"
Cohesion: 0.09
Nodes (17): ImageAsset, ImageAssetManager, Any, Path, Manages standalone extraction, persistence, indexing, and serving of original…, Persist original high-resolution image asset and record in registry., Extract embedded original images and high-fidelity page graphics from PDF…, Find image asset by exact hash, hash prefix, or asset_id. (+9 more)

### Community 2 - "RAG Data Models"
Cohesion: 0.11
Nodes (23): BaseModel, QueryClassification, RAGTrace, RetrievalStrategy, VerificationReport, MultiQueryGenerator, Decomposes multi-part, comprehensive, or enumeration queries into sub-queries…, Generate deduplicated list of sub-queries for retrieval aggregation. (+15 more)

### Community 3 - "Chat Endpoint"
Cohesion: 0.08
Nodes (25): post_chat(), Synchronous RAG Chat query endpoint., ChatRequest, ChatResponse, ConversationEvidenceContext, ConversationTurn, Any, Encapsulates a single turn in a multi-turn conversation for auditing and replay. (+17 more)

### Community 4 - "Conversation Models"
Cohesion: 0.10
Nodes (27): AnswerMode, ConversationRAGState, ExpansionPlan, FollowUpResolution, BaseModel, Enum, str, State tracking conversation context, grounded evidence, active topic, and… (+19 more)

### Community 5 - "Citation Assembly"
Cohesion: 0.10
Nodes (22): Citation, Preferred display page string: display_page_number -> page_label -> physical…, ScoredChunk, CitationEngine, _clean_section_title(), _compute_confidence(), Normalize rerank logit score or candidate score into [0.05, 0.99] confidence…, Map answer text [Source N] tags or relevance scores to Citation models. (+14 more)

### Community 6 - "Pipeline Response"
Cohesion: 0.10
Nodes (21): RAGResponse, ThinkingDetailLevel, _detect_fidelity_mode(), _format_evidence_status_directive(), _format_history_for_prompt(), _LLMProxy, Any, Path (+13 more)

### Community 7 - "Thinking Trace"
Cohesion: 0.14
Nodes (21): ReasoningSummary, ThinkingEvent, ThinkingStage, ThinkingStatus, Any, Deterministic state machine managing the lifecycle of Safe Thinking Events.…, Determine whether an event at this stage should be emitted given the current…, Generate a deterministic, safe title for the given stage and status. (+13 more)

### Community 8 - "Vision Page Services"
Cohesion: 0.11
Nodes (18): Downscale image for fast VLM inference while preserving visual details. For…, Vision module for multi-modal document understanding, OCR, code extraction, and…, Compute deterministic SHA256 hex digest of image bytes., Any, Enum, Path, str, Service orchestrating visual page detection heuristics, code screenshot OCR,… (+10 more)

### Community 9 - "Async Redis Cache"
Cohesion: 0.09
Nodes (18): AsyncRedisCache, get_async_cache(), Any, Async Redis cache utility for the RAG backend. Wraps redis.asyncio with…, Store a value with optional TTL (seconds). Returns False on error., Delete a key. Returns False on error., Check if a key exists. Returns False on error., Flush the current database. Returns False on error. (+10 more)

### Community 10 - "Admin Trace Cleanup"
Cohesion: 0.10
Nodes (27): clear_legacy_traces(), clear_observability_telemetry(), delete_legacy_trace(), delete_query_trace_endpoint(), get_error_incidents(), get_filtered_query_traces(), get_legacy_query_traces(), get_legacy_trace_detail_by_id() (+19 more)

### Community 11 - "Telemetry Persistence"
Cohesion: 0.13
Nodes (9): DocumentIngestionTrace, Any, Path, Process queued SQL insert statements in batch mode., Thread-safe SQLite persistent telemetry repository with async write-behind…, Delete a single query trace by trace_id or request_id., Purge all telemetry database records., TelemetryDB (+1 more)

### Community 12 - "Document Metadata"
Cohesion: 0.12
Nodes (15): DocumentMetadataExtractor, Any, Extracts structured metadata (department, effective date, policy ID, entities,…, Convert extracted metadata into ChromaDB-compatible primitive dictionary., Extract departmental ownership from header patterns or body keyword density., Extract alphanumeric policy identifier/code., Normalize raw date string into ISO 8601 format (YYYY-MM-DD)., Extract effective date or last revised date from header. (+7 more)

### Community 13 - "Telemetry Schema"
Cohesion: 0.27
Nodes (21): AlertItem, CacheTelemetry, CacheTypeMetrics, EvidenceItem, GroundingTelemetry, IngestionStageTelemetry, IngestionTelemetry, LatencyBreakdown (+13 more)

### Community 14 - "Document Scope Resolution"
Cohesion: 0.11
Nodes (16): DocumentRetrievalScope, DocumentScopeDecision, DocumentScopeResolver, BaseModel, Enum, str, Document Scope Resolver for Document-Aware RAG Retrieval. Resolves query scope…, Check if query explicitly asks for global search across all documents. (+8 more)

### Community 15 - "Cache Operations"
Cohesion: 0.11
Nodes (13): Any, Store key in cache with optional TTL in seconds., Clear all cached items., Retrieve cached query response by query hash., Store query response in cache., Retrieve cached vector embedding by text hash., Store vector embedding in cache., Retrieve session data by session ID. (+5 more)

### Community 16 - "Ingestion Loaders"
Cohesion: 0.13
Nodes (17): CSVLoader, Any, Path, Loader for CSV (.csv) tabular data documents., DocxLoader, Loader for Microsoft Word (.docx) documents., get_loader_for_file(), load_document() (+9 more)

### Community 17 - "Telemetry Service"
Cohesion: 0.12
Nodes (10): ObservabilityMetrics, TraceSummary, QueryTraceRecord, Enqueue QueryTraceRecord for persistent SQLite storage., Path, Record legacy TraceSummary into circular deque and bridge to DB., Bridge from RAGResponse to canonical QueryTraceRecord & TraceSummary., Production-grade Unified Telemetry & Observability Hub. Bridges in-memory low-… (+2 more)

### Community 18 - "API Dependencies"
Cohesion: 0.22
Nodes (15): get_chat_service(), get_conversation_state_manager(), get_document_service(), get_rag_pipeline(), get_semantic_cache_manager(), get_telemetry_service(), Reset singletons (useful for test isolation)., reset_dependencies() (+7 more)

### Community 19 - "Document API"
Cohesion: 0.14
Nodes (18): delete_document_by_id(), get_document_details(), get_document_image_file(), get_document_ingestion_status(), get_document_page_image_file(), get_document_visual_asset_file(), list_document_image_assets(), list_indexed_documents() (+10 more)

### Community 20 - "Health Endpoint"
Cohesion: 0.18
Nodes (14): get_health_status(), get, Return health readiness check including vector_db, redis, and model statuses., get_available_models(), get, List available LLM, Embedding, and Reranker model specifications., DocumentListResponse, DocumentSummary (+6 more)

### Community 21 - "Loader Base Classes"
Cohesion: 0.43
Nodes (8): BaseLoader, ABC, Abstract Base Class for document loaders., DocumentCategory, DocumentType, Enum, str, RawDocument

### Community 22 - "Query Rewriting"
Cohesion: 0.18
Nodes (9): QueryRewriteResult, _format_history_for_rewrite(), Any, QueryRewriter, Query normalization, deterministic term expansion, and LLM-based query…, Detect pure greetings, smalltalk, and pleasantries that do not require document…, Determines if a query is a follow-up question referencing previous context by…, Robust non-LLM query rewrite fallback for multi-turn dialogues. Extracts the… (+1 more)

### Community 23 - "Upload Routes"
Cohesion: 0.15
Nodes (13): post, Upload and index document file up to 100MB supporting PDF, DOCX, TXT, MD, HTML,…, Retry indexing for a previously uploaded document without re-uploading the file., retry_document_indexing(), upload_document_file(), DocumentUploadResponse, IngestionStatusResponse, Path (+5 more)

### Community 24 - "HTML Loader"
Cohesion: 0.13
Nodes (10): Any, Path, HTMLLoader, Any, Path, Loader for HTML (.html, .htm) documents., Any, Path (+2 more)

### Community 25 - "Embedding Service"
Cohesion: 0.20
Nodes (9): EmbeddingService, normalize_vector(), Normalize vector to unit length for cosine similarity calculations., Deterministic pseudo-embedding for testing when ML packages are unavailable., Embed a single query or text string., Batch embed a list of document chunk texts., Service wrapper for dense vector embeddings with caching, batching, and…, MetadataFilter (+1 more)

### Community 26 - "Document DTOs"
Cohesion: 0.21
Nodes (11): DocumentDetailResponse, IngestionStage, IngestionStatus, Enum, str, StageProgress, DocumentService, Any (+3 more)

### Community 27 - "Page Identity"
Cohesion: 0.16
Nodes (8): PageIdentity, Any, BaseModel, Convert to flat dictionary for embedding in chunk/document metadata., Resolved human-facing display label with fallback hierarchy:…, Factory method to construct a canonical PageIdentity from partial or complete…, Canonical Page Identity Contract. Reconciles: 1. internal_page_index: 0-based…, Check whether a query identifier (e.g. 98, '98', 'Page 98', 'p.98', 99) matches…

### Community 28 - "Hybrid Retrieval"
Cohesion: 0.22
Nodes (9): tokenize(), HybridRetriever, Any, Merges multiple ranked lists of ScoredChunks using Reciprocal Rank Fusion…, Executes parallel dense vector and BM25 lexical searches and merges results via…, Execute hybrid search with Reciprocal Rank Fusion., reciprocal_rank_fusion(), DenseVectorRetriever (+1 more)

### Community 29 - "Vision Cache"
Cohesion: 0.17
Nodes (9): Any, Path, Persist vision extraction result to disk., Purge all entries from the vision cache., Persistent on-disk cache for vision-language model extractions. Prevents re-…, Check if an image recently failed extraction to prevent repeated timeouts., Record an extraction failure for an image hash., Retrieve cached extraction result if present and valid. (+1 more)

### Community 30 - "Conversation State"
Cohesion: 0.13
Nodes (8): ConversationStateManager, Thread-safe, isolated in-memory cache of ConversationRAGState objects. Ensures…, Retrieve existing state or create a fresh empty state for the given…, Persist updated conversation state into thread-safe cache., Update conversation state for the given conversation_id with deep copy…, Evict a specific conversation from state cache. Returns True if existed, False…, Purge all conversation states from memory., Check if conversation_id exists in active cache.

### Community 31 - "Logical Document Structure"
Cohesion: 0.18
Nodes (9): BlockType, DocumentBlock, LogicalDocument, LogicalSection, Enum, str, Document representation preserving cross-page logical sections and multi-modal…, An individual atomic block of content within a logical section. (+1 more)

### Community 32 - "Redis Client"
Cohesion: 0.17
Nodes (14): check_redis_connection(), close_redis_client(), get_redis_client(), get_redis_connection_url(), get_redis_pool(), get_redis_pubsub(), Async Redis Client & Connection Pool Manager for FastAPI & Redis Pub/Sub…, Ping Redis server to verify connectivity. Returns True if connection is alive,… (+6 more)

### Community 33 - "Printed Page Detection"
Cohesion: 0.19
Nodes (8): Detect human-visible printed page number via PrintedPageDetector., _is_valid_roman(), PrintedPageDetector, Process an entire document's pages with sequence continuity reconciliation.…, Detect PageIdentity for a single standalone page., Generic Sequence-Aware Printed Page Number & Label Detector. Extracts human-…, Inspect header (top 3 lines) and footer (bottom 3 lines) of a page text to…, Evaluate a single header/footer line for page number candidates.

### Community 34 - "Metadata Inference"
Cohesion: 0.19
Nodes (8): Any, QueryMetadataInferer, Detect explicit policy ID references in query., Detect topic domain from query., Infers structured metadata filters (department, policy_id, topic_tags,…, Detect document category if specifically requested., Infer ChromaDB / BM25 compatible filter dictionary from query string and…, Detect department(s) mentioned in the query. Disambiguates English pronoun 'it'…

### Community 35 - "Chat Session Routes"
Cohesion: 0.17
Nodes (12): clear_all_chat_sessions(), clear_chat_session_messages(), delete_chat_session(), post_chat_stream(), delete, post, Purge all conversation sessions across the entire system., Sub-1s TTFT SSE streaming endpoint. Detects client disconnects and cancels LLM… (+4 more)

### Community 36 - "Evidence Gate"
Cohesion: 0.32
Nodes (9): EvidenceStatus, Enum, str, QueryCategory, compute_monotonic_evidence_status(), EvidenceSufficiencyGate, EvidenceSufficiencyResult, Pre-generation evaluation gate verifying that retrieved evidence is sufficient… (+1 more)

### Community 37 - "BM25 Index"
Cohesion: 0.22
Nodes (4): BM25SearchIndex, Any, Search BM25 index and return ScoredChunk list sorted by BM25 relevance score., Okapi BM25 lexical index supporting tokenization, disk persistence, metadata…

### Community 38 - "Section Tracking"
Cohesion: 0.36
Nodes (9): Context manager that records elapsed milliseconds., timer(), _build_section_patterns(), clean_title(), is_noise_line(), is_valid_roman(), parse_section_heading(), SectionHeading (+1 more)

### Community 39 - "Semantic Cache Queries"
Cohesion: 0.20
Nodes (8): cosine_similarity(), CachedResponse, BaseModel, Container for a retrieved semantic cache entry., Delete all items in semantic_cache collection., Manages semantic cache lookups, storage, and invalidation using ChromaDB.…, Query cache for semantically matching answer. Returns CachedResponse if…, SemanticCacheManager

### Community 40 - "PDF Loader"
Cohesion: 0.31
Nodes (6): PDFLoader, Any, Path, Loader for PDF documents with canonical page numbering, cross-page logical…, detect_continuation_signals(), Detect cross-page continuation cues in text.

### Community 41 - "Cross Encoder Reranking"
Cohesion: 0.22
Nodes (5): CrossEncoderReranker, Filters candidate chunks scoring below `min_ratio` of top reranker logit score.…, Rerank candidate chunks using cross-encoder logit scoring and relative…, BAAI/bge-reranker-large cross-encoder reranker wrapper with device auto-…, RelativeScoreThresholdPostprocessor

### Community 42 - "Celery Setup"
Cohesion: 0.22
Nodes (9): get_redis_url(), healthcheck_task(), ping_task(), Any, Celery Application Module for Asynchronous RAG Task Processing. Configures…, Helper to construct or retrieve Redis connection URL for Celery., Lightweight healthcheck test task., Comprehensive Celery worker health check task. (+1 more)

### Community 44 - "Retrieval Cache"
Cohesion: 0.33
Nodes (3): Any, Thread-safe in-memory LRU cache with TTL for retrieval candidate chunks and…, RetrievalCache

### Community 45 - "Loader Metadata"
Cohesion: 0.33
Nodes (4): Any, Path, Load file content into RawDocument instances preserving hierarchy and metadata., Return True if loader handles this file extension or format.

### Community 46 - "JSON Loader"
Cohesion: 0.43
Nodes (4): JSONLoader, Any, Path, Loader for JSON and JSONL (.json, .jsonl) documents.

### Community 47 - "Telemetry Enums"
Cohesion: 0.48
Nodes (7): AlertStatus, EvidenceContentType, GroundingStatus, Enum, str, SeverityLevel, SubsystemStatus

### Community 48 - "Semantic Cache Storage"
Cohesion: 0.29
Nodes (4): Any, Path, Store query response in semantic cache if answer and citations are valid. Safe…, Settings

### Community 49 - "Heading Context"
Cohesion: 0.33
Nodes (3): Any, Pushes new heading, popping any equal or deeper level headings., SectionContext

### Community 50 - "Model Switching"
Cohesion: 0.33
Nodes (6): ModelSelectRequest, BaseModel, post, Switch active LLM model and push it down into the live backend pipeline., select_active_model(), put

### Community 51 - "Text Loader"
Cohesion: 0.40
Nodes (4): Any, Path, Loader for plain text (.txt) documents., TxtLoader

### Community 53 - "Hashing Utilities"
Cohesion: 0.33
Nodes (5): compute_file_hash(), compute_string_hash(), Path, Compute 16-character SHA-256 hash of a text string., Compute 16-character SHA-256 hash of a file.

### Community 55 - "Model Management"
Cohesion: 0.40
Nodes (3): ModelManager, Lightweight, stateless model router modeled after Antigravity. Manages active…, Update active model and optionally trigger non-blocking background preload.

### Community 56 - "Redis Cache Service"
Cohesion: 0.50
Nodes (3): get_redis_cache(), Get global RedisCache singleton instance., Redis

### Community 60 - "Logging Setup"
Cohesion: 0.67
Nodes (3): Configure module-level logger with stdout handler., setup_logging(), Logger

### Community 61 - "Timing Utilities"
Cohesion: 0.67
Nodes (3): Decorator to log function execution time., timed(), F

## Knowledge Gaps
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ScoredChunk` connect `Citation Assembly` to `Vector Storage`, `RAG Data Models`, `Chat Endpoint`, `Conversation Models`, `Evidence Gate`, `Pipeline Response`, `BM25 Index`, `Cross Encoder Reranking`, `Retrieval Cache`, `Telemetry Schema`, `Telemetry Service`, `Dense Retrieval`, `Hybrid Retrieval`?**
  _High betweenness centrality (0.140) - this node is a cross-community bridge._
- **Why does `RAGPipeline` connect `Pipeline Response` to `Vector Storage`, `RAG Data Models`, `Metadata Inference`, `Conversation Models`, `Citation Assembly`, `Evidence Gate`, `Thinking Trace`, `Semantic Cache Queries`, `Cross Encoder Reranking`, `Vision Page Services`, `Chat Endpoint`, `Document Scope Resolution`, `API Dependencies`, `Query Rewriting`, `Hybrid Retrieval`?**
  _High betweenness centrality (0.120) - this node is a cross-community bridge._
- **Why does `Chunk` connect `Vector Storage` to `RAG Data Models`, `BM25 Index`, `Citation Assembly`, `Pipeline Response`, `Embedding Service`, `Document DTOs`, `Hybrid Retrieval`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Are the 43 inferred relationships involving `TelemetryService` (e.g. with `clear_legacy_traces()` and `clear_observability_telemetry()`) actually correct?**
  _`TelemetryService` has 43 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `ScoredChunk` (e.g. with `ChromaVectorStore` and `VectorStoreInterface`) actually correct?**
  _`ScoredChunk` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 36 inferred relationships involving `RAGPipeline` (e.g. with `Chunk` and `ChunkMetadata`) actually correct?**
  _`RAGPipeline` has 36 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `RawDocument` (e.g. with `AdaptiveChunker` and `BaseChunker`) actually correct?**
  _`RawDocument` has 17 INFERRED edges - model-reasoned connections that need verification._