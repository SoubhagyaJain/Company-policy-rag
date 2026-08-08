# Graph Report - C:/Users/jains/OneDrive/Desktop/Rag-chatbot/company_policy_rag  (2026-08-07)

## Corpus Check
- 282 files · ~186,400 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2640 nodes · 6659 edges · 142 communities (113 shown, 29 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 388 edges (avg confidence: 0.56)
- Token cost: 0 input · 10,081 output

## Community Hubs (Navigation)
- Frontend Pages And Views
- API Dependency Injection
- Generation And Synthesis
- Adaptive Document Chunkers
- API Client And Types
- Document Service Routes
- Embedding Service
- FastAPI App Entry
- Indexing And Config
- Document Loaders
- Vector Store Embeddings
- Query Processing Pipeline
- Chunk And BM25 Models
- Retriever Factory
- Frontend NPM Dependencies
- Streamlit Path Bootstrap
- Query Corpus Detection
- Loader Base Class
- Chat Service Orchestration
- Indexing Discovery
- Loader Chunk Methods
- BM25 And Code Retrieval
- BM25 Index Implementation
- Agent Topic Pipeline
- Chainlit Chat Interface
- Async Redis Cache
- Chunker Base Class
- Aether NPM Dependencies
- Agent And Tools
- Frontend TypeScript Config
- Streamlit Chat Component
- Assistant Turn Processing
- Prompt Templates
- Postprocessing Retriever
- Policy Topic Pipeline
- Loader Factory
- Legal Document Upload
- PDF Parsers
- Building Block Pipeline
- PDF Image Extraction
- Tool Code Pipeline
- Session Memory
- Streamlit Health Component
- Hierarchical Chunking
- Chroma Telemetry Client
- Aether TypeScript Config
- Streaming Response Synthesizer
- Feedback API Routes
- Citation Engine
- Code Validation Pipeline
- Streamlit Documents Component
- Diagnostic Script Entrypoints
- Diagram Caption Builder
- Golden Dataset Loader
- Hybrid Retriever
- Adversarial API Tests
- Code Validation Trace
- Citation Selection
- Citation Filtering
- Section Tracker
- Chainlit Session Init
- Model Filtering And Labels
- Human Judge Agreement
- Code Validation Diagnostics
- Retrieval Metrics
- Language Detection
- Citation Source Extraction
- Query And Embedding Cache
- Evaluation Entrypoint
- Redis Cache Client
- LLM Judge Evaluation
- Ollama Model Client
- Chroma Vector Store
- Interview Notes Generator
- Heading Detection
- CLI Entrypoints
- Reranker Postprocessor
- Pipeline Timing
- Backend FastAPI Factory
- CI Agent Topic Gate
- Config Tests
- Logging Utilities
- Project Documentation Notes
- Prompt Brace Escaping
- Section Title Detection
- Section Heading Parser
- Embedding Cache
- Query Engine Base
- Chat Request Routes
- Grounding Mode
- Eval Failure Analysis
- CI Eval Gate
- Citation Formatting
- Document API Tests
- Chat API Tests
- Health And Models Tests
- CI Tool Code Gate
- Category Inference
- Tool Code Prompt Tests
- Admin API Tests
- Chroma Error Types
- Grounding And Faithfulness
- Next.js Root Layout
- Section Heading Model
- Cache Delete Operations
- Weak Case Comparison
- Cross Encoder Reranker
- Hybrid Search Fusion
- Docker Entrypoint Script
- Ephemeral Chroma Fixture
- Streamlit Components Init
- Streamlit UI Init
- Backend Package Init
- Cache Flush Operations
- Cache Set Operations
- Redis Connection Check
- Citation Trust Tracking
- Topic Retrieval Pipelines
- Project Roadmap Tracks
- Next.js Config
- Next TypeScript Definitions
- Aether API Init
- Pytest Configuration
- Unit Tests Init
- Document Delete Tests
- Document Double Delete Tests
- Empty File Upload Tests
- Corrupt File Upload Tests
- Malformed JSON Upload Tests
- Unsupported File Upload Tests
- Admin Traces Tests
- Backend Package Metadata
- Docker Hub Project Name
- Suggested Prompt Catalog
- Source Renderer

## God Nodes (most connected - your core abstractions)
1. `Chunk` - 95 edges
2. `RawDocument` - 85 edges
3. `ScoredChunk` - 74 edges
4. `DocumentMetadata` - 40 edges
5. `AdaptiveChunker` - 39 edges
6. `ChunkMetadata` - 37 edges
7. `DocumentService` - 36 edges
8. `build_index()` - 35 edges
9. `ChromaVectorStore` - 34 edges
10. `BaseChunker` - 34 edges

## Surprising Connections (you probably didn't know these)
- `Test: global redis_cache singleton` --conceptually_related_to--> `Microservices stack (backend + frontend + redis + chroma)`  [INFERRED]
  tests/unit/test_redis_cache.py → docker-compose.yml
- `Microservices stack (backend + frontend + redis + chroma)` --semantically_similar_to--> `Pre-built Docker Hub compose (soubhagya007/rag-chatbot)`  [INFERRED] [semantically similar]
  docker-compose.yml → docker-compose.dockerhub.yml
- `Docker runtime deps (no jupyter/chainlit/pytest)` --semantically_similar_to--> `Full development dependencies (incl. chainlit, jupyter, fastapi)`  [INFERRED] [semantically similar]
  requirements-docker.txt → requirements.txt
- `MetadataFilter` --uses--> `Settings`  [INFERRED]
  backend/embeddings/vector_store.py → src/config.py
- `TestChunkerEdgeCases` --uses--> `AdaptiveChunker`  [INFERRED]
  tests/unit/test_chunker_correctness.py → backend/ingestion/chunkers/adaptive_chunker.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Streamlit page bootstrap pattern (path fix → bootstrap_app → inject_global_styles → sidebar)** — app_ensure_path_ensure_project_root, app_ui_bootstrap, app_ui_theme, app_ui_sidebar_render_sidebar_controls, app_ui_sidebar_render_sidebar_status [EXTRACTED 0.95]
- **RAG session lifecycle (initialize → reload → cache)** — app_ui_session_ensure_backend_ready, app_ui_session_ensure_session_state, app_ui_session_reload_rag_session, app_ui_session_ensure_query_engine, app_ui_session_index_health [INFERRED 0.85]
- **Per-answer trust + citation rendering flow** — app_ui_components_citations_render_sources_compact, ui_citations_render_sources, app_ui_components_trust_render_trust_panel, app_ui_components_trust_citation_quality_summary, app_ui_components_chat_render_chat_interface [INFERRED 0.85]
- **GitHub Actions CI pipeline (lint+test+build)** — github_workflows_ci_yml_backend_lint, github_workflows_ci_yml_backend_test, github_workflows_ci_yml_frontend_lint_build, github_workflows_ci_yml_docker_build [EXTRACTED 1.00]
- **Dual phase frameworks (Track A and Track B)** — concept_track_a_infrastructure, concept_track_b_relevancy_recovery, concept_track_c_topic_pipelines [EXTRACTED 0.95]
- **Retrieval correctness test components** — tests_unit_test_retrieval_correctness_reciprocal_rank_fusion, tests_unit_test_retrieval_correctness_hybrid_retriever, tests_unit_test_retrieval_correctness_cross_encoder_reranker, tests_unit_test_retrieval_correctness_relative_score_threshold_postprocessor [EXTRACTED 1.00]

## Communities (142 total, 29 thin omitted)

### Community 0 - "Frontend Pages And Views"
Cohesion: 0.07
Nodes (51): HomePage(), AdminView(), formatTokens(), getHealthBg(), getHealthColor(), getHealthIcon(), rowVariants, ChatMessage() (+43 more)

### Community 1 - "API Dependency Injection"
Cohesion: 0.06
Nodes (55): get_chat_service(), get_rag_pipeline(), get_telemetry_service(), Reset singletons (useful for test isolation)., reset_dependencies(), get_observability_metrics(), get_query_traces(), get_trace_detail_by_id() (+47 more)

### Community 2 - "Generation And Synthesis"
Cohesion: 0.05
Nodes (68): CompactAndRefine, run_timed_query(), apply_faithfulness_guard(), build_grounded_response_synthesizer(), _collapse_guidebook_edge_answer(), _collapse_planning_block_answer(), _edge_abstention_for_query(), generate_grounded_answer_with_trace() (+60 more)

### Community 3 - "Adaptive Document Chunkers"
Cohesion: 0.07
Nodes (41): AdaptiveChunker, Adaptive chunker that inspects document properties and selects the optimal…, MarkdownAwareChunker, Markdown-aware chunker protecting code blocks and respecting header boundaries., Recursive text chunker using hierarchical separators., RecursiveChunker, Semantic chunker grouping coherent sentences and paragraphs into semantic…, SemanticChunker (+33 more)

### Community 4 - "API Client And Types"
Cohesion: 0.06
Nodes (53): api(), ChatMessage, ChatResponse, CorpusScope, fetchHealth(), fetchModels(), GroundingMode, HealthResponse (+45 more)

### Community 5 - "Document Service Routes"
Cohesion: 0.07
Nodes (50): get_document_service(), delete_document_by_id(), get_document_details(), list_indexed_documents(), get, post, Upload and index document file up to 100MB supporting PDF, DOCX, TXT, MD, HTML,…, Retrieve list of indexed documents with summary metadata. (+42 more)

### Community 6 - "Embedding Service"
Cohesion: 0.05
Nodes (30): EmbeddingService, Embed a single query or text string., Batch embed a list of document chunk texts., Service wrapper for dense vector embeddings with caching, batching, and…, Deterministic pseudo-embedding for testing when ML packages are unavailable., ChromaVectorStore, cosine_similarity(), ABC (+22 more)

### Community 7 - "FastAPI App Entry"
Cohesion: 0.07
Nodes (46): HealthResponse, ModelsResponse, create_app(), FastAPI, Aether API — FastAPI entrypoint., chat(), chat_stream(), ChatRequest (+38 more)

### Community 8 - "Indexing And Config"
Cohesion: 0.07
Nodes (39): Diagnose Chroma index state outside Streamlit., Hierarchical parent-child chunking with code-block protection. Parents…, Path, Single source of truth for all hyperparameters and paths. Production rationale:…, Resolve runtime project root for dev, pip install, and Docker layouts., _resolve_project_root(), Diagram caption nodes for figure/diagram retrieval. Captions are embedded in…, clear_parent_store() (+31 more)

### Community 9 - "Document Loaders"
Cohesion: 0.08
Nodes (41): HTMLLoader, Loader for HTML (.html, .htm) documents., JSONLoader, Any, Path, Loader for JSON and JSONL (.json, .jsonl) documents., LoaderFactory, Factory registry for document loaders. (+33 more)

### Community 10 - "Vector Store Embeddings"
Cohesion: 0.09
Nodes (28): normalize_vector(), Normalize vector to unit length for cosine similarity calculations., MetadataFilter, BaseModel, ChunkMetadata, BaseModel, QueryRewriteResult, ContextCompressor (+20 more)

### Community 11 - "Query Processing Pipeline"
Cohesion: 0.07
Nodes (46): _append_unique_query(), augment_query_for_retrieval(), augment_query_with_guidebook_terms(), augment_query_with_policy_terms(), build_multi_retrieval_queries(), _context_block_from_query(), is_comprehensive_list_query(), is_guidebook_edge_case_query() (+38 more)

### Community 12 - "Chunk And BM25 Models"
Cohesion: 0.11
Nodes (31): Chunk, ScoredChunk, Replace child chunks with parent document sections when available,…, Build BM25 index over a list of document chunks., _searchable_text(), tokenize(), Any, Merges multiple ranked lists of ScoredChunks using Reciprocal Rank Fusion… (+23 more)

### Community 13 - "Retriever Factory"
Cohesion: 0.08
Nodes (42): get_bm25_corpus_size(), build_retriever(), _check_reranker_dependencies(), get_final_top_k(), get_initial_top_k(), get_node_postprocessors(), get_reranker(), get_reranker_install_hints() (+34 more)

### Community 14 - "Frontend NPM Dependencies"
Cohesion: 0.05
Nodes (40): autoprefixer, clsx, framer-motion, dependencies, clsx, framer-motion, lucide-react, next (+32 more)

### Community 15 - "Streamlit Path Bootstrap"
Cohesion: 0.12
Nodes (33): ensure_project_root(), Path, Add project root to sys.path (importable without the app.* prefix)., Insert company_policy_rag root so `app.*` and `src.*` imports resolve., Admin document management page., Admin system health and diagnostics page., Streamlit multipage entry point, Streamlit entry point for the Company Policy RAG multipage app. Run from the… (+25 more)

### Community 16 - "Query Corpus Detection"
Cohesion: 0.10
Nodes (25): detect_query_corpus(), _query_matches_triggers(), Infer policy vs guidebook corpus from query text when sidebar scope is 'all'.…, corpus_retrieval_filters(), filter_nodes_by_metadata(), node_matches_filters(), Any, NodeWithScore (+17 more)

### Community 17 - "Loader Base Class"
Cohesion: 0.15
Nodes (20): BaseLoader, ABC, Any, Path, Abstract Base Class for document loaders., Load file content into RawDocument instances preserving hierarchy and metadata., Return True if loader handles this file extension or format., CSVLoader (+12 more)

### Community 18 - "Chat Service Orchestration"
Cohesion: 0.12
Nodes (33): _BackendState, _build_turn_result(), _extract_query_answer(), Framework-agnostic chat orchestration for Streamlit and FastAPI., _run_agent_turn(), _run_direct_turn(), begin_citation_turn(), get_generation_nodes_this_turn() (+25 more)

### Community 19 - "Indexing Discovery"
Cohesion: 0.09
Nodes (33): discover_pdf_files(), _document_type_for_path(), documents_to_nodes(), enrich_documents_with_sections(), enrich_nodes_with_sections(), _file_hash(), get_node_parser(), load_all_documents() (+25 more)

### Community 20 - "Loader Chunk Methods"
Cohesion: 0.08
Nodes (18): Any, Path, Any, Path, PDFLoader, Any, Path, Loader for PDF documents with PyMuPDF (fitz) / pypdf support. (+10 more)

### Community 21 - "BM25 And Code Retrieval"
Cohesion: 0.12
Nodes (33): get_bm25_index(), Return loaded BM25 index, or None if missing/disabled., Rebuild BM25 if corpus size mismatches Chroma chunk count., sync_bm25_with_chroma(), _boost_nodes_matching_markers(), _boost_score_for_pool(), _cached_guidebook_code_node_ids(), clear_code_node_cache() (+25 more)

### Community 22 - "BM25 Index Implementation"
Cohesion: 0.15
Nodes (27): BM25CorpusEntry, BM25Index, clear_bm25_storage(), _corpus_path(), index_exists_check(), _index_path(), load_bm25_index(), Path (+19 more)

### Community 23 - "Agent Topic Pipeline"
Cohesion: 0.14
Nodes (31): AgentTopicKind, classify_agent_topic_query(), _content_text(), ensure_agent_topic_in_results(), finalize_agent_topic_context(), _has_usable_manager_definition(), _inject_from_pool(), _matches_content_markers() (+23 more)

### Community 24 - "Chainlit Chat Interface"
Cohesion: 0.10
Nodes (25): on_chat_end (Chainlit), on_message (Chainlit), Chainlit chat interface for the Company Policy RAG agent. Run: chainlit run…, Send the agent answer with an attached Sources section. Sources are Chainlit…, Handle user message via ReAct agent; attach rich source citations., Compact header above expandable source elements., _send_answer_with_citations, _sources_header() (+17 more)

### Community 25 - "Async Redis Cache"
Cohesion: 0.09
Nodes (18): AsyncRedisCache, get_async_cache(), Any, Async Redis cache utility for the RAG backend. Wraps redis.asyncio with…, Store a value with optional TTL (seconds). Returns False on error., Delete a key. Returns False on error., Check if a key exists. Returns False on error., Flush the current database. Returns False on error. (+10 more)

### Community 26 - "Chunker Base Class"
Cohesion: 0.20
Nodes (15): BaseChunker, ABC, Abstract Base Class for document chunkers., Split RawDocument list into Chunk list according to strategy., Estimate token count (approx 4 chars per token)., HeadingAwareChunker, Hierarchical heading-aware chunker for legal, policy, and compliance documents., ChunkRole (+7 more)

### Community 27 - "Aether NPM Dependencies"
Cohesion: 0.06
Nodes (30): dependencies, lucide-react, react, react-dom, devDependencies, tailwindcss, @tailwindcss/vite, @types/react (+22 more)

### Community 28 - "Agent And Tools"
Cohesion: 0.11
Nodes (29): AgentOutput, QueryEngineTool, build_policy_search_tool(), chat_with_memory(), configure_llm(), create_agent(), extract_agent_response(), get_agent_context() (+21 more)

### Community 29 - "Frontend TypeScript Config"
Cohesion: 0.07
Nodes (29): compilerOptions, allowJs, esModuleInterop, incremental, isolatedModules, jsx, lib, module (+21 more)

### Community 30 - "Streamlit Chat Component"
Cohesion: 0.12
Nodes (26): apply_queue_user_prompt(), Any, queue_user_prompt(), Chat welcome, history, and ChatGPT-style turn handling., Main chat loop: history, pending generation, and input., Append user message and defer generation (testable without Streamlit)., _render_assistant_extras(), render_chat_history() (+18 more)

### Community 31 - "Assistant Turn Processing"
Cohesion: 0.13
Nodes (28): apply_complete_assistant_turn(), complete_assistant_turn(), process_pending_turn(), Generate assistant reply for the queued user prompt., Persist assistant message and clear pending state., _run_turn(), ensure_backend_ready(), ensure_query_engine() cached (+20 more)

### Community 32 - "Prompt Templates"
Cohesion: 0.12
Nodes (27): PromptTemplate, format_node_for_prompt(), format_nodes_for_prompt(), get_agent_system_prompt(), get_faithfulness_guard_prompt(), get_refine_template(), get_text_qa_template(), NodeWithScore (+19 more)

### Community 33 - "Postprocessing Retriever"
Cohesion: 0.14
Nodes (19): promote_code_tool_nodes(), Move on-topic code/currency chunks to the front of the final context., apply_postprocessors(), _diversify_comprehensive_nodes(), _PostprocessingRetriever, prepare_retrieval_query(), preprocess_query(), Any (+11 more)

### Community 34 - "Policy Topic Pipeline"
Cohesion: 0.16
Nodes (27): _boost_nodes_matching_markers(), _boost_score_for_pool(), classify_policy_topic_query(), ensure_policy_topic_in_results(), finalize_policy_topic_context(), _inject_bm25_policy_nodes(), inject_policy_topic_chunks(), _matches_markers() (+19 more)

### Community 35 - "Loader Factory"
Cohesion: 0.13
Nodes (23): get_loader_for_file(), load_document(), Any, Path, Convenience method to load a document file using the appropriate loader., IngestionRequest, IngestionResult, BaseModel (+15 more)

### Community 36 - "Legal Document Upload"
Cohesion: 0.13
Nodes (25): process_legal_removal(), list_legal_documents(), Path, Save and index legal PDF uploads from the Streamlit UI., Return a safe PDF filename (basename only, allowed chars, .pdf suffix)., Validate and write an uploaded PDF to data/legal/., Return a PDF path under data/legal/; reject path traversal., Delete a legal PDF from disk and remove its chunks from Chroma. (+17 more)

### Community 37 - "PDF Parsers"
Cohesion: 0.13
Nodes (21): ParserName, _detect_block_types(), load_pdf_as_documents(), _load_with_marker(), _load_with_pypdf(), _marker_available(), _parse_page_number(), Document (+13 more)

### Community 38 - "Building Block Pipeline"
Cohesion: 0.16
Nodes (25): classify_guidebook_topic_query(), ensure_guidebook_topic_in_results(), finalize_guidebook_topic_context(), GuidebookTopicKind, _matches_markers(), _node_text(), _promote_matching(), Enum (+17 more)

### Community 39 - "PDF Image Extraction"
Cohesion: 0.18
Nodes (24): _by_source_path(), extract_pdf_images(), get_page_images(), images_dir_for_hash(), _load_by_source(), _pixmap_to_rgb(), Any, Path (+16 more)

### Community 40 - "Tool Code Pipeline"
Cohesion: 0.17
Nodes (24): classify_tool_code_query(), _is_code_node(), _matches_markers(), _node_text(), _promote_nodes_with_markers(), Enum, NodeWithScore, str (+16 more)

### Community 41 - "Session Memory"
Cohesion: 0.14
Nodes (23): build_retrieval_query(), create_session_memory(), format_history_block(), get_history_messages(), memory_stats(), ChatMemoryBuffer, Short-term conversation memory for multi-turn policy Q&A. Design (production-…, Lightweight stats for Chainlit startup / debugging. (+15 more)

### Community 42 - "Streamlit Health Component"
Cohesion: 0.13
Nodes (21): Yield answer text in small chunks for st.write_stream., stream_answer_chunks(), format_eval_metrics(), load_last_eval_run(), probe_ollama_tags(), Any, Path, System health helpers (pure functions for tests + Streamlit page). (+13 more)

### Community 43 - "Hierarchical Chunking"
Cohesion: 0.13
Nodes (20): _approx_token_count(), _build_parent_text(), _child_splitter(), _chunk_code(), _chunk_prose(), documents_to_hierarchical_nodes(), _group_documents_for_parents(), HierarchicalNodes (+12 more)

### Community 44 - "Chroma Telemetry Client"
Cohesion: 0.13
Nodes (20): ClientAPI, ProductTelemetryClient, ProductTelemetryEvent, NoOpProductTelemetry, No-op Chroma product telemetry client. chromadb 0.5.x still invokes…, Drop all Chroma telemetry events without contacting posthog., _chroma_settings(), _collection_names() (+12 more)

### Community 45 - "Aether TypeScript Config"
Cohesion: 0.09
Nodes (22): compilerOptions, allowImportingTsExtensions, isolatedModules, jsx, lib, module, moduleDetection, moduleResolution (+14 more)

### Community 46 - "Streaming Response Synthesizer"
Cohesion: 0.17
Nodes (19): QueryTextType, RESPONSE_TYPE, _finalize_answer(), _format_text_chunks(), NodeWithScore, One metadata-rich chunk per retrieved node for compact synthesis., extract_thinking(), is_reasoning_model() (+11 more)

### Community 47 - "Feedback API Routes"
Cohesion: 0.14
Nodes (20): Rating, get_feedback_summary(), get, post, Answer feedback endpoints., submit_feedback(), feedback_summary(), _feedback_path() (+12 more)

### Community 48 - "Citation Engine"
Cohesion: 0.19
Nodes (10): CitationEngine, Extracts explicit [Source N] tags from generated answer text and maps them to…, make_llama_node(), make_scored_chunk(), NodeWithScore, Empirical test suite for citation verification and parsing. Tests: 1. Citation…, Test suite for metadata mapping, page attribution, and verbatim quotes., Test suite for edge cases when LLM outputs invalid source tags. (+2 more)

### Community 49 - "Code Validation Pipeline"
Cohesion: 0.19
Nodes (10): answer_contains_code(), apply_code_validation_pipeline(), context_contains_code(), NodeWithScore, Validate code grounding; self-correct once on failure; strip or fallback if…, should_validate_code(), _node(), NodeWithScore (+2 more)

### Community 50 - "Streamlit Documents Component"
Cohesion: 0.18
Nodes (19): format_indexing_result(), index_changed(), list_handbook_pdfs(), process_legal_uploads(), Any, Path, Legal PDF upload and handbook indexing helpers., Run scripts/index_documents.py; returns (success, output). (+11 more)

### Community 51 - "Diagnostic Script Entrypoints"
Cohesion: 0.18
Nodes (18): main(), main(), _preview(), main(), configure_llama_index(), create_query_engine(), create_retriever(), index_exists() (+10 more)

### Community 52 - "Diagram Caption Builder"
Cohesion: 0.14
Nodes (16): build_caption_nodes(), _heuristic_caption(), _llm_caption(), _load_manifest(), _manifest_path(), _page_section_metadata(), Any, Document (+8 more)

### Community 53 - "Golden Dataset Loader"
Cohesion: 0.19
Nodes (11): _aggregate_for_cases(), CaseResult, filter_cases_by_corpus(), GoldenCase, load_golden_dataset(), Filter golden cases by corpus (all | policy | guidebook)., Load versioned golden dataset from JSON., Unit tests for evaluation framework (no Ollama required). (+3 more)

### Community 54 - "Hybrid Retriever"
Cohesion: 0.17
Nodes (15): bm25_nodes_for_query(), HybridRetriever, Any, NodeWithScore, QueryBundle, Hybrid dense + BM25 retrieval with reciprocal rank fusion., Merge multiple ranked node lists using RRF. score(node) = sum(1 / (k + rank))…, Run BM25 search and return NodeWithScore list. (+7 more)

### Community 55 - "Adversarial API Tests"
Cohesion: 0.14
Nodes (16): TestClient, Simulate client dropping connection after start event without breaking server., Stress test concurrent upload and deletion requests., test_chat_empty_and_whitespace_payload(), test_chat_huge_payload(), test_chat_invalid_filters_and_fields(), test_concurrent_document_uploads_and_deletions(), test_delete_non_existent_document() (+8 more)

### Community 56 - "Code Validation Trace"
Cohesion: 0.16
Nodes (16): _apply_fail_mode(), CodeValidationTrace, _needs_prose_fallback(), _parse_validation_verdict(), LLM, Post-generation code-line validation and self-correction (Phase 3). Runs after…, Drop lines like 'the tool is defined as:' when no code block follows., True when stripped answer still looks broken or self-contradictory. (+8 more)

### Community 57 - "Citation Selection"
Cohesion: 0.16
Nodes (13): RetrieverQueryEngine, log_retrieval_stage(), _node_label(), Citation selection for policy RAG — precision over recall. Production-rag…, Append nodes from a query-engine response to the current turn. Called by…, Structured log of chunk counts and top sections at each pipeline stage., record_generation_sources(), build_grounded_query_engine() (+5 more)

### Community 58 - "Citation Filtering"
Cohesion: 0.19
Nodes (16): extract_cited_source_indices(), filter_nodes_by_relevance_score(), Any, NodeWithScore, Keep nodes scoring at least min_ratio × top reranker score., Select UI citations from generation nodes only (never a parallel retrieval).…, Parse 1-based [Source N] tags from a grounded answer. Handles: [Source 1],…, select_citations_for_answer() (+8 more)

### Community 59 - "Section Tracker"
Cohesion: 0.20
Nodes (9): enrich_text_with_section_context(), Current hierarchical position while scanning a document., Maintains a hierarchical section stack while scanning documents in reading…, Push heading onto stack, clearing deeper levels., Scan all lines in text and apply every heading found (in order)., Apply tracker to a text block: propagate prior context, then scan for new…, SectionContext, SectionTracker (+1 more)

### Community 60 - "Chainlit Session Init"
Cohesion: 0.17
Nodes (16): on_chat_start (Chainlit), Initialize agent and retriever once per user session., main(), parse_args(), Namespace, Path, resolve_pdf_paths(), build_index() (+8 more)

### Community 61 - "Model Filtering And Labels"
Cohesion: 0.16
Nodes (16): patch, filter_chat_models(), format_model_label(), Exclude embedding-only models from LLM picker options., Human-readable label for UI (qwen2.5:7b -> Qwen2.5 7B)., client(), fixture, Tests for Aether FastAPI routes. (+8 more)

### Community 62 - "Human Judge Agreement"
Cohesion: 0.26
Nodes (14): main(), agreement_within(), cohen_kappa_binary(), compare_human_llm(), load_eval_run_cases(), mae(), pearson_r(), Any (+6 more)

### Community 63 - "Code Validation Diagnostics"
Cohesion: 0.18
Nodes (11): _load_case(), main(), extract_code_lines(), heuristic_code_grounded(), HeuristicResult, _line_matches_context(), _normalize_code_line(), _normalize_context_for_matching() (+3 more)

### Community 64 - "Retrieval Metrics"
Cohesion: 0.20
Nodes (11): _chunk_searchable_text(), compute_retrieval_metrics(), is_chunk_relevant(), NodeWithScore, Combine metadata + text into a single lowercase searchable string., Fuzzy relevance check against golden relevant_sections keywords. Matches if any…, Returns (hit_rate, context_precision, context_recall, relevant_count,…, _node() (+3 more)

### Community 65 - "Language Detection"
Cohesion: 0.23
Nodes (14): append_language_hint(), detect_query_language(), language_instruction_for_query(), Response language detection and prompt instructions., Heuristic language detection from the user question., Return the language the assistant must write in., Short instruction appended to generation/agent prompts., Append a language hint to the user query for the LLM. (+6 more)

### Community 66 - "Citation Source Extraction"
Cohesion: 0.20
Nodes (5): Parse 1-based [Source N] tags from answer text., Map answer text [Source N] tags or relevance scores to Citation models., Citation, Test suite for [Source N] extraction and regex validation., TestCitationParsingRegex

### Community 67 - "Query And Embedding Cache"
Cohesion: 0.15
Nodes (8): Any, Store key in cache with optional TTL in seconds., Retrieve cached query response by query hash., Store query response in cache., Retrieve cached vector embedding by text hash., Retrieve session data by session ID., Store session data in cache., Retrieve key from cache (Redis or in-memory fallback).

### Community 68 - "Evaluation Entrypoint"
Cohesion: 0.22
Nodes (13): main(), parse_args(), Namespace, Path, _resolve_dataset_path(), EvalRun, format_results_table(), _mean() (+5 more)

### Community 69 - "Redis Cache Client"
Cohesion: 0.26
Nodes (10): get_redis_cache(), Production-ready Redis client with a thread-safe in-memory fallback. Provides…, Get global RedisCache singleton instance., RedisCache, Test: global redis_cache singleton, test_redis_cache_flush(), Test: RedisCache graceful error handling, Test: RedisCache in-memory fallback (+2 more)

### Community 70 - "LLM Judge Evaluation"
Cohesion: 0.21
Nodes (14): evaluate_case(), _get_judge_llm(), judge_answer_relevancy(), judge_faithfulness(), _parse_judge_json(), Any, Ollama, VectorStoreIndex (+6 more)

### Community 71 - "Ollama Model Client"
Cohesion: 0.27
Nodes (13): enrich_model_info(), fetch_model_details(), list_enriched_models(), _param_size_numeric(), _parse_family(), _parse_param_size(), _parse_quantization(), probe_ollama_tags() (+5 more)

### Community 72 - "Chroma Vector Store"
Cohesion: 0.20
Nodes (12): Collection, _delete_chunks_for_source(), _filter_paths_for_incremental(), get_chroma_vector_store(), _get_indexed_file_hashes(), Build LlamaIndex ChromaVectorStore + StorageContext., Map source_file → file_hash for incremental indexing decisions., Remove all chunks for a source file before re-indexing an updated PDF. (+4 more)

### Community 73 - "Interview Notes Generator"
Cohesion: 0.20
Nodes (7): para(), Follow-up answer bank for interview-notes.html. Maps (parent_question_title,…, Escape *text* and wrap each double-newline-separated block in <p>...</p>., add_qs(), esc(), q(), source_card()

### Community 74 - "Heading Detection"
Cohesion: 0.17
Nodes (11): Match, _build_section_patterns(), _clean_title(), _heading_from_match(), _is_noise_line(), _is_valid_roman(), Declarative section-heading pattern. Extensibility: append new SectionPattern…, Regex patterns tuned for Employee Handbooks, HR policies, and legal docs. Level… (+3 more)

### Community 75 - "CLI Entrypoints"
Cohesion: 0.24
Nodes (11): chat_main(), eval_main(), index_main(), _load_script_main(), Path, Console entry points for the soubhagya-policy-rag PyPI package., Index PDFs from data/policies/ and data/legal/., Run golden-set evaluation. (+3 more)

### Community 76 - "Reranker Postprocessor"
Cohesion: 0.23
Nodes (10): BaseNodePostprocessor, NodeWithScore, QueryBundle, Drop nodes scoring below `min_ratio` of the top reranker score. Cross-encoder…, RelativeScoreThresholdPostprocessor, _node(), NodeWithScore, Tests for retrieval post-processors. (+2 more)

### Community 77 - "Pipeline Timing"
Cohesion: 0.24
Nodes (9): percentile(), PipelineTiming, Linear-interpolation percentile (p in 0..100)., Per-query stage timings in milliseconds., summarize_ms(), test_percentile_p50_p95(), test_percentile_single(), test_pipeline_timing_retrieve_total() (+1 more)

### Community 78 - "Backend FastAPI Factory"
Cohesion: 0.22
Nodes (8): create_app(), FastAPI, FastAPI application factory configuring CORS, routers, and global error…, cleanup_deps(), fixture, test_post_chat_stream_auto_generates_session_id(), test_post_chat_stream_empty_message(), test_post_chat_stream_success()

### Community 79 - "CI Agent Topic Gate"
Cohesion: 0.25
Nodes (9): main(), parse_args(), Namespace, main(), parse_args(), Namespace, Logger, Configure module-level logger with file + console handlers. (+1 more)

### Community 81 - "Logging Utilities"
Cohesion: 0.20
Nodes (9): F, Logger, Configure module-level logger with stdout handler., Context manager that records elapsed milliseconds., Decorator to log function execution time., setup_logging(), timed(), timer() (+1 more)

### Community 82 - "Project Documentation Notes"
Cohesion: 0.24
Nodes (10): Command reference document, NoOpProductTelemetry (disable chromadb posthog), Pre-built Docker Hub compose (soubhagya007/rag-chatbot), Microservices stack (backend + frontend + redis + chroma), Backend Lint Job (ruff + mypy), Backend Test Job (pytest with redis:7-alpine), Docker Compose Build Job (depends on all), Frontend Lint + Build Jobs (next lint, tsc) (+2 more)

### Community 83 - "Prompt Brace Escaping"
Cohesion: 0.29
Nodes (7): _escape_format_braces(), Escape braces so str.format on prompt templates does not treat code as…, _escape_prompt_braces(), get_code_validation_prompt(), Escape braces so str.format on prompt templates does not treat code as…, Tests for post-generation code validation (no Ollama required)., TestPromptFormatting

### Community 84 - "Section Title Detection"
Cohesion: 0.24
Nodes (6): detect_section_title(), Return all headings found in text (top-to-bottom order)., Backward-compatible helper: return the first detected heading label in text.…, scan_text_for_headings(), Unit tests for section detection, metadata, and citation utilities., TestHandbookScenarios

### Community 85 - "Section Heading Parser"
Cohesion: 0.33
Nodes (3): parse_section_heading(), Parse a single line into a SectionHeading, or return None. Production…, TestSectionDetection

### Community 87 - "Query Engine Base"
Cohesion: 0.25
Nodes (5): Any, Deterministic grounded response fallback when LLM service is offline., Streaming RAG pipeline: runs retrieval synchronously, then yields real-time LLM…, Execute end-to-end RAG pipeline and return structured RAGResponse with trace…, HybridRetriever

### Community 88 - "Chat Request Routes"
Cohesion: 0.29
Nodes (8): post_chat(), post_chat_stream(), ChatRequest, ChatResponse, post, StreamingResponse, Synchronous RAG Chat query endpoint., Sub-1s Time-To-First-Token (TTFT) Server-Sent Events (SSE) streaming chat…

### Community 89 - "Grounding Mode"
Cohesion: 0.32
Nodes (8): ChatMode, apply_grounding_mode(), _ensure_backend(), Use the same retriever as the cached query engine (rewrite + rerank pipeline)., Generator of SSE strings for streaming chat., _retrieve_nodes_for_turn(), run_chat_stream(), _settings_fingerprint()

### Community 90 - "Eval Failure Analysis"
Cohesion: 0.43
Nodes (7): _classify_failures(), _count_type(), _load_run(), main(), _print_bucket(), _print_summary(), Path

### Community 91 - "CI Eval Gate"
Cohesion: 0.43
Nodes (7): _check_metrics(), main(), parse_args(), Namespace, Path, _thresholds(), _write_baseline()

### Community 92 - "Citation Formatting"
Cohesion: 0.29
Nodes (7): format_citation(), format_citations(), NodeWithScore, TextNode, Normalize chunk metadata into a citation dict for the UI and agent. Rich…, Format a list of retrieved nodes as citations., TestCitationFormatting

### Community 93 - "Document API Tests"
Cohesion: 0.25
Nodes (7): cleanup_deps(), fixture, test_delete_nonexistent_document(), test_get_nonexistent_document(), test_upload_document_text_file(), test_upload_markdown_file_adaptive(), test_upload_oversized_file_rejected()

### Community 94 - "Chat API Tests"
Cohesion: 0.29
Nodes (6): cleanup_deps(), fixture, test_post_chat_auto_generates_session_id(), test_post_chat_empty_message(), test_post_chat_success(), test_post_chat_with_category_filter()

### Community 95 - "Health And Models Tests"
Cohesion: 0.29
Nodes (6): cleanup_deps(), fixture, test_get_health(), test_get_models(), test_select_active_model_invalid(), test_select_active_model_success()

### Community 96 - "CI Tool Code Gate"
Cohesion: 0.53
Nodes (5): _check_relevancy(), _check_retrieval(), main(), parse_args(), Namespace

### Community 97 - "Category Inference"
Cohesion: 0.40
Nodes (4): infer_category(), Path, Derive a human-readable category from filename and folder., TestCategoryInference

### Community 98 - "Tool Code Prompt Tests"
Cohesion: 0.33
Nodes (3): Regression guards for tool/code few-shot prompts., Example K GOOD must not show def convert_currency when excerpts are prose-only., test_example_k_does_not_teach_invented_convert_currency_def()

### Community 99 - "Admin API Tests"
Cohesion: 0.33
Nodes (5): cleanup_deps(), fixture, test_get_admin_observability(), test_get_admin_trace_detail_success_and_404(), test_get_admin_traces()

### Community 100 - "Chroma Error Types"
Cohesion: 0.40
Nodes (5): BaseException, _chroma_not_found_errors(), _is_chroma_corruption(), Exception types for missing Chroma collections (0.5.x and 1.x)., True when persisted Chroma metadata is unreadable (version mismatch / corrupt…

### Community 101 - "Grounding And Faithfulness"
Cohesion: 0.40
Nodes (5): Balanced vs Strict Grounding Trade-off, Faithfulness Guard (SUPPORTED/UNSUPPORTED classifier), Human Eval Rubric (faithfulness + answer relevancy 0-1), normalize_balanced_answer (strip double-ending abstention), Strict Grounding Enforcement (abstain when unsupported)

### Community 102 - "Next.js Root Layout"
Cohesion: 0.40
Nodes (3): inter, jetbrainsMono, metadata

### Community 103 - "Section Heading Model"
Cohesion: 0.40
Nodes (5): SectionHeading, _headings_from_section_path(), Reconstruct heading stack from a breadcrumb path for tracker seeding. Each…, Parsed heading from a single line of policy/legal text., SectionHeading

### Community 105 - "Weak Case Comparison"
Cohesion: 0.67
Nodes (3): _cases_by_id(), main(), Path

### Community 106 - "Cross Encoder Reranker"
Cohesion: 0.67
Nodes (3): Cross-Encoder Reranking (BAAI/bge-reranker-large), CrossEncoderReranker (bge-reranker-large), RelativeScoreThresholdPostprocessor

### Community 107 - "Hybrid Search Fusion"
Cohesion: 0.67
Nodes (3): Hybrid Search (Dense + Sparse BM25 via RRF), HybridRetriever (BM25 + dense fusion), reciprocal_rank_fusion (RRF) function

### Community 109 - "Ephemeral Chroma Fixture"
Cohesion: 0.67
Nodes (3): fixture, Ensure tests run against isolated ephemeral vector store to avoid SQLite file…, setup_ephemeral_chroma()

## Ambiguous Edges - Review These
- `render_sidebar_controls()` → `render_sidebar_status()`  [AMBIGUOUS]
  app/ui/sidebar.py · relation: calls
- `probe_ollama_tags() UI wrapper` → `probe_ollama_tags() UI wrapper`  [AMBIGUOUS]
  app/ui/components/health.py · relation: calls

## Knowledge Gaps
- **122 isolated node(s):** `entrypoint.sh script`, `PYTHONPATH`, `name`, `private`, `version` (+117 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **29 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `render_sidebar_controls()` and `render_sidebar_status()`?**
  _Edge tagged AMBIGUOUS (relation: calls) - confidence is low._
- **What is the exact relationship between `probe_ollama_tags() UI wrapper` and `probe_ollama_tags() UI wrapper`?**
  _Edge tagged AMBIGUOUS (relation: calls) - confidence is low._
- **Why does `RawDocument` connect `Adaptive Document Chunkers` to `Loader Factory`, `Document Loaders`, `Loader Base Class`, `Loader Chunk Methods`, `Chunker Base Class`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Why does `Chunk` connect `Chunk And BM25 Models` to `API Dependency Injection`, `Citation Source Extraction`, `Adaptive Document Chunkers`, `Loader Factory`, `Document Service Routes`, `Embedding Service`, `Document Loaders`, `Vector Store Embeddings`, `Citation Engine`, `Loader Chunk Methods`, `Query Engine Base`, `Chunker Base Class`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Why does `ScoredChunk` connect `Chunk And BM25 Models` to `API Dependency Injection`, `Citation Source Extraction`, `Loader Factory`, `Document Service Routes`, `Embedding Service`, `Vector Store Embeddings`, `Citation Engine`, `Query Engine Base`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Are the 30 inferred relationships involving `Chunk` (e.g. with `ChromaVectorStore` and `MetadataFilter`) actually correct?**
  _`Chunk` has 30 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `RawDocument` (e.g. with `AdaptiveChunker` and `BaseChunker`) actually correct?**
  _`RawDocument` has 20 INFERRED edges - model-reasoned connections that need verification._