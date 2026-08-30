export interface Citation {
  id: string;
  source_index?: number;
  document_id?: string;
  document_name?: string;
  title: string;
  source: string;
  chunk_text: string;
  snippet?: string;
  score: number;
  relevance_score?: number;
  page?: number;
  page_number?: number;
  physical_page_number?: number;
  display_page_number?: string | number | null;
  page_label?: string;
  internal_page_index?: number;
  heading?: string;
  section_title?: string | null;
  section_path?: string | null;
  category?: string;
  evidence_type?: string;
  visual_asset_id?: string | null;
  visual_status?: string | null;
  url?: string;
  image_url?: string | null;
  image_assets?: any[];
}

export type QueryCategory =
  | 'factual'
  | 'comparison'
  | 'enumeration'
  | 'procedural'
  | 'conversational'
  | 'implementation'
  | 'code'
  | 'explanation'
  | 'architecture';

export type QueryCategoryType = QueryCategory | string;
export type ResponseMode = 'compact' | 'standard' | 'detailed';

export interface VerificationReport {
  faithfulness?: number;
  completeness?: number;
  citation_coverage?: number;
  coherence?: number;
  composite_score?: number;
  passed?: boolean;
  critique?: string | null;
  missing_aspects?: string[];
  unsupported_claims?: string[];
  retry_count?: number;
}

export interface QueryTrace {
  trace_id: string;
  request_id?: string;
  timestamp: string;
  original_query: string;
  query?: string;
  resolved_query?: string;
  query_rewritten?: string;
  expanded_queries?: string[];
  sub_queries?: string[];
  total_chunks_retrieved: number;
  top_rerank_score?: number;
  rerank_latency_ms?: number;
  total_latency_ms: number;
  prompt_tokens: number;
  completion_tokens: number;
  model: string;

  // Agentic telemetry fields
  query_type?: string;
  routing_confidence?: number;
  retrieval_strategy?: string;
  retrieval_required?: boolean;
  conversational_bypass?: boolean;
  evidence_required?: boolean;
  inferred_filters?: Record<string, any>;
  applied_filters?: Record<string, any>;
  filter_relaxed?: boolean;
  verification_score?: number;
  verification?: VerificationReport | null;
  faithfulness_passed?: boolean;
  retry_count?: number;
  retry_reasons?: string[];
  cache_hit?: boolean;
  cache_similarity?: number | null;
  stage_timings?: Record<string, number>;
  similarity_scores?: number[];
  rerank_scores?: number[];
  sources_used?: string[];
  anchor_section?: string | null;
  page_identity?: string | null;
  text_candidates?: number;
  visual_candidates?: number;
  final_text_evidence?: number;
  final_visual_evidence?: number;
  visual_asset_status?: string | null;
  vision_status?: string | null;
  evidence_status?: string | null;
  grounding_status?: string | null;
  evidence_text_count?: number;
  evidence_code_count?: number;
  evidence_diagram_count?: number;
  evidence_table_count?: number;
  section_expansion_used?: boolean;
  vision_used?: boolean;
  vision_model?: string | null;
  vision_cache_status?: string | null;
  tokens_per_second?: number | null;
  ttft_ms?: number | null;
  error?: string | null;
  safe_context_preview?: string | null;
  response_mode?: ResponseMode;
  retrieval_top_k?: number;
  rerank_top_k?: number;
  context_tokens?: number;
  generation_max_tokens?: number;
}

export * from '../types/thinking';
import { ThinkingEvent, ReasoningSummary, ThinkingDetailLevel } from '../types/thinking';

export interface ChatMessageData {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  citations?: Citation[];
  trace?: QueryTrace;
  thinking_events?: ThinkingEvent[];
  reasoning_summary?: ReasoningSummary;
  thinking_detail_level?: ThinkingDetailLevel;
  response_mode?: ResponseMode;
  model?: string;
  isStreaming?: boolean;
  error?: string;
}

export interface ChatSession {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessageData[];
  filters?: {
    category?: string;
    document_id?: string;
  };
}

export interface StageProgress {
  stage: string;
  status: string; // PENDING | IN_PROGRESS | COMPLETED | FAILED | SKIPPED
  message?: string | null;
  duration_ms?: number;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface IngestionStatusResponse {
  document_id: string;
  job_id: string;
  filename: string;
  status: string; // READY | TEXT_INDEXING | VISION_PROCESSING | READY_WITH_VISION | PARTIALLY_INDEXED | FAILED | indexed
  progress: number; // 0 to 100
  current_stage: string;
  text_ready: boolean;
  pages_processed: number;
  pages_total: number;
  sections_detected?: number;
  chunks_created: number;
  chunks_indexed: number;
  vision_status?: string; // NONE | PENDING | PROCESSING | COMPLETED | PARTIAL | SKIPPED | DEGRADED | READY_ON_DEMAND
  vision_pages_processed?: number;
  vision_pages_total?: number;
  error?: string | null;
  failed_stage?: string | null;
  stages?: StageProgress[];
  created_at: string;
  updated_at: string;
  duration_ms?: number;
  can_retry?: boolean;
}

export interface DocumentItem {
  id: string;
  filename: string;
  category: string;
  chunks_count: number;
  file_size: number; // bytes
  uploaded_at: string;
  status: 'indexed' | 'processing' | 'failed' | 'READY' | 'TEXT_INDEXING' | 'VISION_PROCESSING' | 'READY_WITH_VISION' | 'PARTIALLY_INDEXED' | string;
  progress?: number;
  current_stage?: string;
  text_ready?: boolean;
  vision_status?: string;
  vision_pages_processed?: number;
  vision_pages_total?: number;
  visual_assets_count?: number;
  image_assets?: any[];
  error?: string;
  failed_stage?: string;
  file_type?: string;
  file_hash?: string;
  pages_count?: number;
  storage_state?: 'HEALTHY' | 'FILE_ONLY' | 'INDEX_ONLY' | string;
}

export interface DuplicateDocumentSummary {
  dry_run: boolean;
  duplicate_groups: number;
  duplicates_found: number;
  duplicates_removed: number;
  groups: Array<{
    file_hash: string;
    filename: string;
    keep_document_id: string;
    duplicate_count: number;
  }>;
}

export interface HealthStatus {
  status: 'ok' | 'degraded' | 'error';
  redis: boolean;
  vector_db: boolean;
  models_loaded: boolean;
  backend_version?: string;
}

export interface ObservabilityData {
  total_queries: number;
  avg_latency_ms: number;
  avg_ttft_ms?: number;
  p95_latency_ms?: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  active_documents?: number;
  indexed_chunks?: number;
  similarity_avg?: number;
  rerank_avg?: number;
  health: HealthStatus;
  recent_traces: QueryTrace[];
}

export interface FilterOptions {
  chat_mode?: 'documents' | 'general';
  category?: string;
  source_file?: string;
  document_id?: string;
}

// ── Production Observability Canonical Types ────────────────

export type SubsystemStatusType = 'healthy' | 'degraded' | 'unavailable' | 'disabled';

export interface SubsystemHealth {
  api: SubsystemStatusType;
  ollama: SubsystemStatusType;
  vector_db: SubsystemStatusType;
  bm25: SubsystemStatusType;
  embedding_model: SubsystemStatusType;
  text_model: SubsystemStatusType;
  vision_model: SubsystemStatusType;
  semantic_cache: SubsystemStatusType;
  vision_cache: SubsystemStatusType;
  memory: SubsystemStatusType;
  uptime_seconds: number;
  error_rate: number;
  active_model_text: string;
  active_model_vision: string;
  details?: Record<string, string>;
}

export interface QueryMetricsData {
  total_queries: number;
  p50_latency_ms?: number | null;
  p95_latency_ms?: number | null;
  p99_latency_ms?: number | null;
  avg_latency_ms?: number | null;
  avg_ttft_ms?: number | null;
  avg_tokens_per_second?: number | null;
  avg_prompt_tokens?: number | null;
  avg_completion_tokens?: number | null;
  error_rate: number;
  requests_per_minute: number;
}

export interface LatencyBreakdownData {
  request_received_ms?: number | null;
  query_classification_ms?: number | null;
  conversation_memory_ms?: number | null;
  query_rewrite_ms?: number | null;
  embedding_ms?: number | null;
  bm25_ms?: number | null;
  vector_search_ms?: number | null;
  hybrid_fusion_ms?: number | null;
  reranking_ms?: number | null;
  section_expansion_ms?: number | null;
  visual_detection_ms?: number | null;
  vision_extraction_ms?: number | null;
  context_build_ms?: number | null;
  qwen_prefill_ms?: number | null;
  ttft_ms?: number | null;
  generation_ms?: number | null;
  streaming_ms?: number | null;
  response_serialization_ms?: number | null;
  total_latency_ms: number;
}

export interface EvidenceItemData {
  chunk_id: string;
  document_id?: string | null;
  source_file?: string | null;
  page_number?: number | null;
  page_label?: string | null;
  section_title?: string | null;
  section_id?: string | null;
  content_type: 'text' | 'code' | 'diagram' | 'table';
  snippet: string;
  dense_score?: number | null;
  sparse_score?: number | null;
  rrf_score?: number | null;
  rerank_score?: number | null;
  selected: boolean;
  image_url?: string | null;
  extra?: Record<string, any>;
}

export interface RetrievalQualityData {
  retrieval_hit_rate: number;
  avg_candidate_count: number;
  avg_rerank_score: number;
  avg_final_chunk_count: number;
  evidence_sufficiency_rate: number;
  measured_metrics?: Record<string, any>;
  proxy_metrics?: Record<string, any>;
  evaluation_metrics?: Record<string, any>;
}

export interface GroundingTelemetryData {
  supported_claims_pct?: number | null;
  unsupported_claims_pct?: number | null;
  inferred_claims_pct?: number | null;
  citation_count: number;
  citation_coverage_pct?: number | null;
  grounding_status: 'grounded' | 'partially_grounded' | 'unsupported' | 'conversational_bypass' | 'not_applicable';
  supported_claims?: string[];
  unsupported_claims?: string[];
  inferred_claims?: string[];
}

export interface VisionFailureRecordData {
  id: string;
  timestamp: string;
  document_id?: string | null;
  source_file?: string | null;
  page_number?: number | null;
  visual_type: string;
  error_type: string;
  duration_ms: number;
  request_id?: string | null;
  message: string;
}

export interface VisionTelemetryData {
  model_name: string;
  visual_pages_detected: number;
  code_screenshots: number;
  diagrams: number;
  tables: number;
  requests_count: number;
  success_count: number;
  failure_count: number;
  timeout_count: number;
  avg_latency_ms?: number | null;
  p95_latency_ms?: number | null;
  cache_hit_rate?: number | null;
  negative_cache_hit_rate?: number | null;
  circuit_breaker_state: string;
  recent_failures: VisionFailureRecordData[];
}

export interface TextModelTelemetryData {
  model_name: string;
  requests_count: number;
  p50_latency_ms?: number | null;
  p95_latency_ms?: number | null;
  avg_ttft_ms?: number | null;
  avg_tokens_per_second?: number | null;
  avg_prompt_tokens?: number | null;
  avg_completion_tokens?: number | null;
  total_tokens: number;
  errors_count: number;
}

export interface ModelTelemetrySummaryData {
  text_model: TextModelTelemetryData;
  vision_model: VisionTelemetryData;
}

export interface TokenTelemetryData {
  avg_system_prompt_tokens: number;
  avg_memory_tokens: number;
  avg_user_query_tokens: number;
  avg_rag_context_tokens: number;
  avg_prompt_tokens: number;
  p95_prompt_tokens: number;
  avg_completion_tokens: number;
  p95_completion_tokens: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  total_tokens: number;
}

export interface MemoryResolutionEventData {
  id: string;
  timestamp: string;
  session_id?: string | null;
  user_query: string;
  resolved_query: string;
  referent_found?: string | null;
  resolution_status: string;
  latency_ms: number;
}

export interface MemoryTelemetryData {
  active_sessions: number;
  messages_today: number;
  memory_hit_rate?: number | null;
  reference_resolution_success_rate?: number | null;
  summary_updates: number;
  avg_memory_latency_ms?: number | null;
  avg_recent_turn_tokens: number;
  avg_summary_tokens: number;
  avg_memory_retrieval_tokens: number;
  recent_resolutions: MemoryResolutionEventData[];
}

export interface DocumentIngestionTraceData {
  document_id: string;
  filename: string;
  category: string;
  file_size_bytes: number;
  status: string;
  current_stage: string;
  pages_count: number;
  chunks_count: number;
  sections_count: number;
  visual_assets_count: number;
  vision_success_count: number;
  vision_failed_count: number;
  created_at: string;
  total_duration_ms: number;
  error?: string | null;
  stages?: StageProgress[];
}

export interface IngestionTelemetryData {
  documents_processed: number;
  ready_count: number;
  processing_count: number;
  failed_count: number;
  pages_processed: number;
  chunks_indexed: number;
  embeddings_generated: number;
  vector_index_ready: boolean;
  bm25_index_ready: boolean;
  visual_assets_total: number;
  vision_success_count: number;
  vision_failed_count: number;
  recent_ingestions: DocumentIngestionTraceData[];
}

export interface CacheTypeMetricsData {
  name: string;
  hits: number;
  misses: number;
  hit_rate?: number | null;
  avg_hit_latency_ms?: number | null;
  avg_miss_latency_ms?: number | null;
  evictions: number;
  size_entries: number;
  size_bytes: number;
}

export interface CacheTelemetryData {
  semantic_cache: CacheTypeMetricsData;
  embedding_cache: CacheTypeMetricsData;
  retrieval_cache: CacheTypeMetricsData;
  vision_cache: CacheTypeMetricsData;
  negative_vision_cache: CacheTypeMetricsData;
}

export interface ErrorIncidentData {
  incident_id: string;
  timestamp: string;
  request_id?: string | null;
  document_id?: string | null;
  conversation_id?: string | null;
  component: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  duration_ms?: number | null;
  retry_count: number;
  stack_trace?: string | null;
  details?: Record<string, any>;
}

export interface AlertItemData {
  alert_id: string;
  rule_name: string;
  severity: 'healthy' | 'warning' | 'critical';
  current_value: string | number;
  threshold_value: string | number;
  message: string;
  triggered_at?: string | null;
  active: boolean;
}

export interface TimeSeriesPointData {
  timestamp: string;
  latency_p50_ms: number;
  latency_p95_ms: number;
  latency_p99_ms: number;
  requests_count: number;
  prompt_tokens: number;
  completion_tokens: number;
  avg_chunks: number;
  avg_rerank_score: number;
  vision_requests: number;
  vision_timeouts: number;
  vision_cache_hits: number;
  errors_count: number;
}

export interface ObservabilitySummaryData {
  time_range: string;
  health: SubsystemHealth;
  query_metrics: QueryMetricsData;
  latency_breakdown: LatencyBreakdownData;
  retrieval_quality: RetrievalQualityData;
  grounding: GroundingTelemetryData;
  models: ModelTelemetrySummaryData;
  tokens: TokenTelemetryData;
  memory: MemoryTelemetryData;
  ingestion: IngestionTelemetryData;
  caches: CacheTelemetryData;
  alerts: AlertItemData[];
  recent_traces: QueryTrace[];
  recent_incidents: ErrorIncidentData[];
  time_series: TimeSeriesPointData[];
}

export interface TelemetryFilterOptions {
  timeRange?: string; // 5m | 15m | 1h | 6h | 24h | 7d
  documentId?: string;
  conversationId?: string;
  intent?: string;
  model?: string;
  status?: string;
  grounding?: string;
  vision?: string;
  cache?: string;
  hasError?: boolean;
}
