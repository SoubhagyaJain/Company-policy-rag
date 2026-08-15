export interface Citation {
  id: string;
  document_id?: string;
  title: string;
  source: string;
  chunk_text: string;
  score: number;
  page?: number;
  heading?: string;
  category?: string;
  url?: string;
}

export type QueryCategory =
  | 'factual'
  | 'comparison'
  | 'enumeration'
  | 'procedural'
  | 'conversational';

export type QueryCategoryType = QueryCategory | string;

export interface VerificationReport {
  faithfulness: number;
  completeness: number;
  citation_coverage: number;
  coherence: number;
  composite_score: number;
  passed: boolean;
  critique?: string | null;
  missing_aspects?: string[];
  unsupported_claims?: string[];
  retry_count?: number;
}

export interface QueryTrace {
  trace_id: string;
  timestamp: string;
  original_query: string;
  query_rewritten?: string;
  expanded_queries?: string[];
  total_chunks_retrieved: number;
  top_rerank_score: number;
  rerank_latency_ms: number;
  total_latency_ms: number;
  prompt_tokens: number;
  completion_tokens: number;
  model: string;

  // Agentic telemetry fields
  query_type?: string;
  routing_confidence?: number;
  retrieval_strategy?: string;
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
}


export interface ChatMessageData {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  citations?: Citation[];
  trace?: QueryTrace;
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

export interface DocumentItem {
  id: string;
  filename: string;
  category: string;
  chunks_count: number;
  file_size: number; // bytes
  uploaded_at: string;
  status: 'indexed' | 'processing' | 'failed';
  file_type?: string;
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
  category?: string;
  source_file?: string;
  document_id?: string;
}


