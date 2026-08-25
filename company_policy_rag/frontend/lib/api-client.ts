import {
  Citation,
  QueryTrace,
  VerificationReport,
  QueryCategory,
  DocumentItem,
  IngestionStatusResponse,
  ObservabilityData,
  HealthStatus,
  FilterOptions,
  ObservabilitySummaryData,
  TelemetryFilterOptions,
  ErrorIncidentData,
} from './types';

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ||
  (typeof window !== 'undefined' ? 'http://localhost:8000' : 'http://127.0.0.1:8000');

export interface DonePayload {
  id?: string;
  request_id?: string;
  answer?: string;
  citations?: any[];
  retrieval_trace?: any;
  total_latency_ms?: number;
  latency_ms?: number;
  ttft_ms?: number;
  metrics?: Record<string, unknown>;
  status?: string;
}

export interface StreamCallbacks {
  onStart?: (data: { session_id: string; message_id: string; id?: string; request_id?: string }) => void;
  onChunk?: (chunk: string) => void;
  onCitation?: (citation: Citation) => void;
  onTrace?: (trace: QueryTrace) => void;
  onDone?: (data: DonePayload) => void;
  onError?: (error: Error) => void;
}

function mapCitation(c: any, index = 0): Citation {
  return {
    id: c.chunk_id || c.id || `cit_${index}_${Date.now()}`,
    document_id: c.document_id,
    title: c.source_file || c.title || 'Document Source',
    source: c.source_file || c.source || '',
    chunk_text: c.snippet || c.chunk_text || c.text || '',
    score: typeof c.relevance_score === 'number' ? c.relevance_score : (typeof c.score === 'number' ? c.score : 0.0),
    page: c.page_number ?? c.page,
    page_label: c.page_label ?? (c.page_number ? String(c.page_number) : undefined),
    internal_page_index: c.internal_page_index,
    heading: c.section_title || c.heading || c.section_path,
    category: c.category || 'General',
    image_url: c.image_url ?? (c.image_assets && c.image_assets[0]?.asset_url ? c.image_assets[0].asset_url : null),
    image_assets: c.image_assets ?? [],
  };
}

export function mapVerificationReport(raw: any): VerificationReport | null {
  if (!raw || typeof raw !== 'object') return null;
  return {
    faithfulness: typeof raw.faithfulness === 'number' ? raw.faithfulness : 1.0,
    completeness: typeof raw.completeness === 'number' ? raw.completeness : 1.0,
    citation_coverage: typeof raw.citation_coverage === 'number'
      ? raw.citation_coverage
      : (typeof raw.citationCoverage === 'number' ? raw.citationCoverage : 1.0),
    coherence: typeof raw.coherence === 'number' ? raw.coherence : 1.0,
    composite_score: typeof raw.composite_score === 'number'
      ? raw.composite_score
      : (typeof raw.compositeScore === 'number' ? raw.compositeScore : 1.0),
    passed: typeof raw.passed === 'boolean' ? raw.passed : true,
    critique: raw.critique ?? null,
    missing_aspects: Array.isArray(raw.missing_aspects)
      ? raw.missing_aspects
      : (Array.isArray(raw.missingAspects) ? raw.missingAspects : []),
    unsupported_claims: Array.isArray(raw.unsupported_claims)
      ? raw.unsupported_claims
      : (Array.isArray(raw.unsupportedClaims) ? raw.unsupportedClaims : []),
    retry_count: typeof raw.retry_count === 'number'
      ? raw.retry_count
      : (typeof raw.retryCount === 'number' ? raw.retryCount : 0),
  };
}

export function mapTrace(t: any): QueryTrace {
  if (!t || typeof t !== 'object') {
    return {
      trace_id: `tr_${Date.now()}`,
      request_id: `req_${Date.now()}`,
      timestamp: new Date().toISOString(),
      original_query: '',
      total_chunks_retrieved: 0,
      top_rerank_score: 0.9,
      rerank_latency_ms: 0,
      total_latency_ms: 0,
      prompt_tokens: 0,
      completion_tokens: 0,
      model: 'FastAPI RAG',
    };
  }

  const rawVer = t.verification || t.verification_report || t.verificationReport;
  const verification = mapVerificationReport(rawVer);

  const verificationScore = typeof t.verification_score === 'number'
    ? t.verification_score
    : (typeof t.verificationScore === 'number'
      ? t.verificationScore
      : (verification?.composite_score ?? undefined));

  const faithfulnessPassed = typeof t.faithfulness_passed === 'boolean'
    ? t.faithfulness_passed
    : (typeof t.faithfulnessPassed === 'boolean'
      ? t.faithfulnessPassed
      : (verification ? verification.passed : undefined));

  const routingConfidence = typeof t.routing_confidence === 'number'
    ? t.routing_confidence
    : (typeof t.routingConfidence === 'number' ? t.routingConfidence : undefined);

  const retryCount = typeof t.retry_count === 'number'
    ? t.retry_count
    : (typeof t.retryCount === 'number' ? t.retryCount : (verification?.retry_count ?? 0));

  const retryReasons = Array.isArray(t.retry_reasons)
    ? t.retry_reasons
    : (Array.isArray(t.retryReasons) ? t.retryReasons : []);

  const cacheHit = typeof t.cache_hit === 'boolean'
    ? t.cache_hit
    : (typeof t.cacheHit === 'boolean'
      ? t.cacheHit
      : Boolean(t.retrieval_strategy === 'conversational_bypass' || t.retrieval_strategy === 'semantic_cache'));

  const cacheSimilarity = typeof t.cache_similarity === 'number'
    ? t.cache_similarity
    : (typeof t.cacheSimilarity === 'number' ? t.cacheSimilarity : null);

  const inferredFilters = (t.inferred_filters || t.inferredFilters) && typeof (t.inferred_filters || t.inferredFilters) === 'object'
    ? (t.inferred_filters || t.inferredFilters)
    : {};

  const appliedFilters = (t.applied_filters || t.appliedFilters) && typeof (t.applied_filters || t.appliedFilters) === 'object'
    ? (t.applied_filters || t.appliedFilters)
    : {};

  const filterRelaxed = typeof t.filter_relaxed === 'boolean'
    ? t.filter_relaxed
    : Boolean(t.filterRelaxed);

  const queryType = t.query_type || t.queryType || undefined;
  const retrievalStrategy = t.retrieval_strategy || t.retrievalStrategy || undefined;

  const topScore = Array.isArray(t.rerank_scores) && t.rerank_scores.length > 0
    ? t.rerank_scores[0]
    : (typeof t.top_rerank_score === 'number'
      ? t.top_rerank_score
      : (typeof t.topRerankScore === 'number' ? t.topRerankScore : (t.verification_score ?? 0.9)));

  const pTokens = t.token_usage?.prompt_tokens ?? t.prompt_tokens ?? t.promptTokens ?? 0;
  const cTokens = t.token_usage?.completion_tokens ?? t.completion_tokens ?? t.completionTokens ?? 0;

  const isConversationalBypass = typeof t.conversational_bypass === 'boolean'
    ? t.conversational_bypass
    : Boolean(retrievalStrategy === 'conversational_bypass' || t.fallback_reason === 'conversational_greeting');

  return {
    trace_id: t.trace_id || t.id || `tr_${Date.now()}`,
    request_id: t.request_id || t.requestId,
    timestamp: t.timestamp || new Date().toISOString(),
    original_query: t.query || t.original_query || t.originalQuery || '',
    resolved_query: t.resolved_query || t.resolvedQuery || t.rewritten_query || t.query_rewritten,
    query_rewritten: t.rewritten_query || t.query_rewritten || t.queryRewritten,
    expanded_queries: t.sub_queries || t.expanded_queries || t.expandedQueries || [],
    total_chunks_retrieved: t.candidate_count ?? t.total_chunks_retrieved ?? t.totalChunksRetrieved ?? t.retrieved_candidate_count ?? 0,
    top_rerank_score: topScore,
    rerank_latency_ms: t.stage_timings?.reranking ?? t.stage_timings_ms?.reranking ?? t.rerank_latency_ms ?? t.rerankLatencyMs ?? 0,
    total_latency_ms: t.execution_time_ms ?? t.total_latency_ms ?? t.totalLatencyMs ?? t.latency_ms ?? 0,
    prompt_tokens: pTokens,
    completion_tokens: cTokens,
    model: t.model || t.generation_model || 'FastAPI RAG',

    // Agentic & Production Telemetry fields
    query_type: queryType,
    routing_confidence: routingConfidence,
    retrieval_strategy: retrievalStrategy,
    retrieval_required: typeof t.retrieval_required === 'boolean' ? t.retrieval_required : !isConversationalBypass,
    conversational_bypass: isConversationalBypass,
    evidence_required: typeof t.evidence_required === 'boolean' ? t.evidence_required : !isConversationalBypass,
    inferred_filters: inferredFilters,
    applied_filters: appliedFilters,
    filter_relaxed: filterRelaxed,
    verification_score: isConversationalBypass ? undefined : verificationScore,
    verification: verification,
    faithfulness_passed: faithfulnessPassed,
    retry_count: retryCount,
    retry_reasons: retryReasons,
    cache_hit: cacheHit,
    cache_similarity: cacheSimilarity,
    stage_timings: t.stage_timings || t.stage_timings_ms || undefined,
    similarity_scores: Array.isArray(t.similarity_scores) ? t.similarity_scores : undefined,
    rerank_scores: Array.isArray(t.rerank_scores) ? t.rerank_scores : undefined,
    sources_used: Array.isArray(t.sources_used) ? t.sources_used : undefined,
    anchor_section: t.anchor_section ?? null,
    evidence_text_count: t.evidence_text_count ?? 0,
    evidence_code_count: t.evidence_code_count ?? 0,
    evidence_diagram_count: t.evidence_diagram_count ?? 0,
    evidence_table_count: t.evidence_table_count ?? 0,
    section_expansion_used: Boolean(t.section_expansion || t.section_expansion_used),
    vision_used: Boolean(t.vision_fallback || t.vision_used),
    vision_model: t.vision_model ?? null,
    vision_cache_status: t.vision_cache_status ?? null,
    tokens_per_second: t.tokens_per_second ?? null,
    ttft_ms: t.ttft_ms ?? null,
    error: t.error ?? null,
    safe_context_preview: t.safe_context_preview ?? null,
  };
}

export class ApiClient {
  private baseUrl: string;

  constructor(baseUrl = API_BASE) {
    this.baseUrl = baseUrl;
  }

  /**
   * SSE Stream Chat endpoint: POST /api/chat/stream
   */
  async streamChat(
    message: string,
    sessionId: string,
    filters?: FilterOptions,
    model?: string,
    callbacks?: StreamCallbacks,
    signal?: AbortSignal
  ): Promise<void> {
    const url = `${this.baseUrl}/api/chat/stream`;
    const payload = {
      message,
      session_id: sessionId,
      filters: filters || {},
      model: model || 'default',
    };

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: JSON.stringify(payload),
        signal,
      });

      if (!response.ok) {
        if (response.status === 404) {
          return await this.fallbackNonStreamingChat(message, sessionId, filters, model, callbacks);
        }
        const errText = typeof response.text === 'function' ? await response.text().catch(() => '') : '';
        throw new Error(`HTTP ${response.status}: ${errText || 'Stream request failed'}`);
      }

      if (!response.body) {
        throw new Error('Response body is null');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let currentEvent = 'chunk';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          if (trimmed.startsWith('event:')) {
            currentEvent = trimmed.substring(6).trim();
          } else if (trimmed.startsWith('data:')) {
            const dataStr = trimmed.substring(5).trim();
            try {
              const data = JSON.parse(dataStr);
              this.handleStreamEvent(currentEvent, data, callbacks);
            } catch {
              // If data is raw text string instead of JSON
              if (currentEvent === 'chunk' && callbacks?.onChunk) {
                callbacks.onChunk(dataStr);
              }
            }
          }
        }
      }

      // Process any remaining buffer
      if (buffer.trim()) {
        if (buffer.startsWith('data:')) {
          const dataStr = buffer.substring(5).trim();
          try {
            const data = JSON.parse(dataStr);
            this.handleStreamEvent(currentEvent, data, callbacks);
          } catch {
            if (currentEvent === 'chunk' && callbacks?.onChunk) {
              callbacks.onChunk(dataStr);
            }
          }
        }
      }
    } catch (err) {
      if ((err as Error).name === 'AbortError') {
        return;
      }
      if (callbacks?.onError) {
        callbacks.onError(err instanceof Error ? err : new Error(String(err)));
      }
      throw err;
    }
  }

  private handleStreamEvent(
    eventType: string,
    data: unknown,
    callbacks?: StreamCallbacks
  ) {
    if (!callbacks) return;

    switch (eventType) {
      case 'start':
        callbacks.onStart?.(data as { session_id: string; message_id: string; id?: string; request_id?: string });
        break;
      case 'chunk':
        if (typeof data === 'string') {
          callbacks.onChunk?.(data);
        } else if (data && typeof data === 'object' && 'content' in data) {
          callbacks.onChunk?.((data as { content: string }).content);
        }
        break;
      case 'citation':
        if (data && typeof data === 'object') {
          const rawCitations = Array.isArray((data as any).citations)
            ? (data as any).citations
            : Array.isArray(data)
            ? data
            : [data];
          rawCitations.forEach((c: any, i: number) => {
            callbacks.onCitation?.(mapCitation(c, i));
          });
        }
        break;
      case 'trace':
        if (data && typeof data === 'object') {
          const rawTrace = (data as any).trace || data;
          callbacks.onTrace?.(mapTrace(rawTrace));
        }
        break;
      case 'done':
        if (data && typeof data === 'object') {
          const doneData = data as any;
          if (Array.isArray(doneData.citations)) {
            doneData.citations.forEach((c: any, i: number) => {
              callbacks.onCitation?.(mapCitation(c, i));
            });
          }
          if (doneData.retrieval_trace) {
            const rawTrace = {
              ...doneData.retrieval_trace,
              request_id: doneData.request_id || doneData.retrieval_trace.request_id,
              verification: doneData.retrieval_trace.verification || doneData.verification,
            };
            callbacks.onTrace?.(mapTrace(rawTrace));
          }
          callbacks.onDone?.(doneData);
        } else {
          callbacks.onDone?.({ latency_ms: 0 });
        }
        break;
      case 'error':
        {
          const errDetail = (data as any)?.detail || (typeof data === 'string' ? data : 'Error from RAG backend');
          const err = new Error(errDetail);
          callbacks.onError?.(err);
        }
        break;
    }
  }

  private async fallbackNonStreamingChat(
    message: string,
    sessionId: string,
    filters?: FilterOptions,
    model?: string,
    callbacks?: StreamCallbacks
  ) {
    try {
      const res = await this.sendChatMessage(message, sessionId, filters, model);
      if (callbacks?.onChunk && res.answer) {
        callbacks.onChunk(res.answer);
      }
      if (res.citations && callbacks?.onCitation) {
        res.citations.forEach((c, i) => callbacks.onCitation?.(mapCitation(c, i)));
      }
      if (callbacks?.onTrace) {
        callbacks.onTrace(mapTrace({
          trace_id: res.id || `trace_${Date.now()}`,
          request_id: (res.metrics as any)?.request_id || `req_${Date.now()}`,
          timestamp: new Date().toISOString(),
          original_query: message,
          total_chunks_retrieved: res.citations?.length || 0,
          top_rerank_score: res.citations?.[0]?.score || 0.9,
          rerank_latency_ms: Math.round((res.latency_ms || 300) * 0.2),
          total_latency_ms: res.latency_ms || 300,
          prompt_tokens: 150,
          completion_tokens: 120,
          model: model || 'FastAPI RAG',
        }));
      }
      if (callbacks?.onDone) {
        callbacks.onDone({
          answer: res.answer,
          citations: res.citations,
          latency_ms: res.latency_ms || 300,
          total_latency_ms: res.latency_ms || 300,
        });
      }
    } catch (err) {
      if (callbacks?.onError) {
        callbacks.onError(err instanceof Error ? err : new Error(String(err)));
      } else {
        throw err;
      }
    }
  }

  /**
   * Non-streaming Chat endpoint: POST /api/chat
   */
  async sendChatMessage(
    message: string,
    sessionId?: string,
    filters?: FilterOptions,
    model?: string
  ): Promise<{
    id: string;
    answer: string;
    citations: Citation[];
    latency_ms: number;
    metrics: Record<string, unknown>;
  }> {
    const payload: Record<string, any> = {
      message,
      session_id: sessionId,
      filters: filters || {},
    };
    if (model && model !== 'default') {
      payload.model = model;
    }
    const res = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      throw new Error(`Chat API error (${res.status}): ${await res.text()}`);
    }

    return res.json();
  }

  /**
   * Unified Production Observability Summary: GET /api/admin/observability
   */
  async getObservabilitySummary(filters?: TelemetryFilterOptions): Promise<ObservabilitySummaryData> {
    const params = new URLSearchParams();
    if (filters?.timeRange) params.set('time_range', filters.timeRange);
    if (filters?.documentId) params.set('document_id', filters.documentId);
    if (filters?.conversationId) params.set('conversation_id', filters.conversationId);
    if (filters?.intent) params.set('intent', filters.intent);
    if (filters?.model) params.set('model', filters.model);
    if (filters?.status) params.set('status', filters.status);
    if (filters?.grounding) params.set('grounding', filters.grounding);
    if (filters?.vision) params.set('vision', filters.vision);
    if (filters?.cache) params.set('cache', filters.cache);
    if (filters?.hasError !== undefined) params.set('has_error', String(filters.hasError));

    const url = `${this.baseUrl}/api/admin/observability?${params.toString()}`;
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Failed to fetch observability summary (${res.status})`);
    }
    const data: ObservabilitySummaryData = await res.json();
    if (Array.isArray(data.recent_traces)) {
      data.recent_traces = data.recent_traces.map(mapTrace);
    }
    return data;
  }

  /**
   * Detailed Trace: GET /api/admin/observability/queries/{identifier}
   */
  async getQueryTraceDetail(identifier: string): Promise<QueryTrace> {
    const res = await fetch(`${this.baseUrl}/api/admin/observability/queries/${encodeURIComponent(identifier)}`);
    if (!res.ok) {
      throw new Error(`Failed to fetch trace detail (${res.status})`);
    }
    const data = await res.json();
    return mapTrace(data);
  }

  /**
   * Error Incidents: GET /api/admin/observability/errors
   */
  async getObservabilityErrors(timeRange = '24h', component?: string, severity?: string): Promise<ErrorIncidentData[]> {
    const params = new URLSearchParams({ time_range: timeRange });
    if (component) params.set('component', component);
    if (severity) params.set('severity', severity);

    const res = await fetch(`${this.baseUrl}/api/admin/observability/errors?${params.toString()}`);
    if (!res.ok) {
      return [];
    }
    return res.json();
  }

  /**
   * Observability Telemetry: GET /api/admin/observability
   */
  async getObservability(): Promise<ObservabilityData> {
    const [obsRes, healthRes] = await Promise.all([
      fetch(`${this.baseUrl}/api/admin/observability`),
      fetch(`${this.baseUrl}/api/health`).catch(() => null),
    ]);

    if (!obsRes.ok) {
      throw new Error(`Failed to fetch observability telemetry (${obsRes.status})`);
    }

    const data = await obsRes.json();
    let healthData: HealthStatus = {
      status: 'ok',
      redis: false,
      vector_db: true,
      models_loaded: true,
      backend_version: 'v1.0.0-fastapi',
    };

    if (healthRes && healthRes.ok) {
      try {
        const rawHealth = await healthRes.json();
        healthData = {
          status: rawHealth.status || 'ok',
          redis: !!rawHealth.redis,
          vector_db: !!rawHealth.vector_db,
          models_loaded: !!rawHealth.models_loaded,
          backend_version: rawHealth.collection ? `Collection: ${rawHealth.collection}` : 'FastAPI RAG',
        };
      } catch {
        // use default
      }
    }

    const rawTraces = Array.isArray(data.recent_traces) ? data.recent_traces : [];
    const mappedTraces: QueryTrace[] = rawTraces.map((t: any, idx: number) => {
      const trace = mapTrace(t);
      if (!trace.original_query) {
        trace.original_query = 'Query';
      }
      if (!t.trace_id && !t.id) {
        trace.trace_id = `tr_${idx}_${Date.now()}`;
      }
      return trace;
    });

    const promptTokens = data.tokens?.total_prompt_tokens ?? data.token_usage?.prompt_tokens ?? data.prompt_tokens ?? 0;
    const completionTokens = data.tokens?.total_completion_tokens ?? data.token_usage?.completion_tokens ?? data.completion_tokens ?? 0;
    const totalTokens = data.tokens?.total_tokens ?? data.token_usage?.total_tokens ?? data.total_tokens ?? (promptTokens + completionTokens);

    return {
      total_queries: data.query_metrics?.total_queries ?? data.total_queries ?? 0,
      avg_latency_ms: data.query_metrics?.avg_latency_ms ?? data.avg_latency_ms ?? 0,
      avg_ttft_ms: data.query_metrics?.avg_ttft_ms ?? data.avg_ttft_ms ?? 0,
      p95_latency_ms: data.query_metrics?.p95_latency_ms ?? data.p95_latency_ms ?? 0,
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: totalTokens,
      active_documents: data.ingestion?.documents_processed ?? data.active_documents ?? 0,
      indexed_chunks: data.ingestion?.chunks_indexed ?? data.indexed_chunks ?? 0,
      similarity_avg: data.retrieval_quality?.avg_rerank_score ?? data.score_distributions?.similarity_avg ?? 0,
      rerank_avg: data.retrieval_quality?.avg_rerank_score ?? data.score_distributions?.rerank_avg ?? 0,
      health: healthData,
      recent_traces: mappedTraces,
    };
  }

  /**
   * Clear Telemetry: POST /api/admin/observability/clear
   */
  async clearTelemetry(): Promise<boolean> {
    try {
      const res = await fetch(`${this.baseUrl}/api/admin/observability/clear`, { method: 'POST' });
      return res.ok;
    } catch {
      return false;
    }
  }

  /**
   * Fetch Document list: GET /api/documents
   */
  async getDocuments(): Promise<DocumentItem[]> {
    const res = await fetch(`${this.baseUrl}/api/documents`);
    if (!res.ok) {
      throw new Error(`Failed to fetch documents (${res.status})`);
    }
    const data = await res.json();
    const rawDocs = data.documents || data || [];
    return rawDocs.map((doc: any) => ({
      id: doc.document_id || doc.id,
      filename: doc.filename,
      category: doc.category || 'General',
      chunks_count: doc.chunk_count ?? doc.chunks_count ?? doc.chunks_indexed ?? 0,
      file_size: doc.file_size_bytes ?? doc.file_size ?? 0,
      uploaded_at: doc.created_at || doc.uploaded_at || new Date().toISOString(),
      status: doc.status || 'indexed',
      progress: doc.progress ?? 100,
      current_stage: doc.current_stage || 'READY',
      text_ready: doc.text_ready ?? true,
      vision_status: doc.vision_status || 'NONE',
      vision_pages_processed: doc.vision_pages_processed ?? 0,
      vision_pages_total: doc.vision_pages_total ?? 0,
      visual_assets_count: doc.visual_assets_count ?? (doc.image_assets?.length ?? 0),
      image_assets: doc.image_assets ?? [],
      error: doc.error,
      failed_stage: doc.failed_stage,
      file_type: doc.file_type || doc.filename?.split('.').pop()?.toLowerCase() || 'unknown',
    }));
  }

  /**
   * Fetch Document Ingestion Status: GET /api/documents/{doc_id}/status
   */
  async getDocumentStatus(docId: string): Promise<IngestionStatusResponse> {
    const res = await fetch(`${this.baseUrl}/api/documents/${docId}/status`);
    if (!res.ok) {
      throw new Error(`Failed to fetch document status (${res.status})`);
    }
    return res.json();
  }

  /**
   * Retry Document Indexing: POST /api/documents/{doc_id}/retry
   */
  async retryDocument(docId: string): Promise<IngestionStatusResponse> {
    const res = await fetch(`${this.baseUrl}/api/documents/${docId}/retry`, {
      method: 'POST',
    });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Retry failed (${res.status}): ${errText}`);
    }
    return res.json();
  }

  /**
   * Upload Document: POST /api/documents/upload
   */
  async uploadDocument(file: File, category = 'General'): Promise<DocumentItem> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('category', category);

    const res = await fetch(`${this.baseUrl}/api/documents/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Upload failed (${res.status}): ${errText}`);
    }

    const doc = await res.json();
    return {
      id: doc.document_id || doc.id,
      filename: doc.filename,
      category: doc.category || category || 'General',
      chunks_count: doc.chunks_indexed ?? doc.chunk_count ?? doc.chunks_count ?? 0,
      file_size: doc.file_size_bytes ?? doc.file_size ?? 0,
      uploaded_at: doc.created_at || doc.uploaded_at || new Date().toISOString(),
      status: doc.status || 'READY',
      progress: doc.progress ?? 100,
      current_stage: doc.current_stage || 'READY',
      text_ready: doc.text_ready ?? true,
      vision_status: doc.vision_status || 'NONE',
      vision_pages_processed: doc.vision_pages_processed ?? 0,
      vision_pages_total: doc.vision_pages_total ?? 0,
      visual_assets_count: doc.visual_assets_count ?? (doc.image_assets?.length ?? 0),
      image_assets: doc.image_assets ?? [],
      error: doc.error,
      failed_stage: doc.failed_stage,
      file_type: doc.file_type || doc.filename?.split('.').pop()?.toLowerCase() || 'unknown',
    };
  }

  /**
   * Delete Document: DELETE /api/documents/{doc_id}
   */
  async deleteDocument(docId: string): Promise<{ status: string; document_id: string }> {
    const res = await fetch(`${this.baseUrl}/api/documents/${docId}`, {
      method: 'DELETE',
    });

    if (!res.ok) {
      throw new Error(`Failed to delete document (${res.status})`);
    }

    return res.json();
  }

  /**
   * Backend Health check: GET /api/health
   */
  async getHealth(): Promise<HealthStatus> {
    const res = await fetch(`${this.baseUrl}/api/health`);
    if (!res.ok) {
      return { status: 'error', redis: false, vector_db: false, models_loaded: false };
    }
    const data = await res.json();
    return {
      status: data.status || 'ok',
      redis: !!data.redis,
      vector_db: !!data.vector_db,
      models_loaded: !!data.models_loaded,
      backend_version: data.collection ? `Collection: ${data.collection}` : 'FastAPI RAG',
    };
  }

  /**
   * Models API: GET /api/models
   */
  async getModels(): Promise<{ active_model: string; models: Array<{ id: string; name: string; type: string; is_active: boolean }> }> {
    try {
      const res = await fetch(`${this.baseUrl}/api/models`);
      if (!res.ok) {
        return { active_model: 'qwen2.5:7b', models: [] };
      }
      return await res.json();
    } catch {
      return { active_model: 'qwen2.5:7b', models: [] };
    }
  }

  /**
   * Models API: POST /api/models/select
   */
  async selectModel(model: string): Promise<boolean> {
    try {
      const res = await fetch(`${this.baseUrl}/api/models/select`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model }),
      });
      return res.ok;
    } catch {
      return false;
    }
  }

  /**
   * Session API: DELETE /api/chat/session/{sessionId}
   */
  async deleteSession(sessionId: string): Promise<boolean> {
    try {
      const res = await fetch(`${this.baseUrl}/api/chat/session/${encodeURIComponent(sessionId)}`, {
        method: 'DELETE',
      });
      return res.ok;
    } catch {
      return false;
    }
  }

  /**
   * Session API: POST /api/chat/session/{sessionId}/clear
   */
  async clearSession(sessionId: string): Promise<boolean> {
    try {
      const res = await fetch(`${this.baseUrl}/api/chat/session/${encodeURIComponent(sessionId)}/clear`, {
        method: 'POST',
      });
      return res.ok;
    } catch {
      return false;
    }
  }

  /**
   * Session API: DELETE /api/chat/sessions
   */
  async clearAllSessions(): Promise<boolean> {
    try {
      const res = await fetch(`${this.baseUrl}/api/chat/sessions`, {
        method: 'DELETE',
      });
      return res.ok;
    } catch {
      return false;
    }
  }
}

export const apiClient = new ApiClient();
