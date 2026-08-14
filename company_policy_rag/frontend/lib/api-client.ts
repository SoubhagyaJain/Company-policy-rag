import {
  Citation,
  QueryTrace,
  DocumentItem,
  ObservabilityData,
  HealthStatus,
  FilterOptions,
} from './types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';

export interface DonePayload {
  id?: string;
  answer?: string;
  citations?: any[];
  retrieval_trace?: any;
  total_latency_ms?: number;
  latency_ms?: number;
  metrics?: Record<string, unknown>;
  status?: string;
}

export interface StreamCallbacks {
  onStart?: (data: { session_id: string; message_id: string; id?: string }) => void;
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
    heading: c.section_title || c.heading || c.section_path,
    category: c.category || 'General',
  };
}

function mapTrace(t: any): QueryTrace {
  return {
    trace_id: t.trace_id || `tr_${Date.now()}`,
    timestamp: t.timestamp || new Date().toISOString(),
    original_query: t.query || t.original_query || '',
    query_rewritten: t.rewritten_query || t.query_rewritten,
    expanded_queries: t.sub_queries || t.expanded_queries || [],
    total_chunks_retrieved: t.candidate_count ?? t.total_chunks_retrieved ?? 0,
    top_rerank_score: t.rerank_scores?.[0] ?? t.top_rerank_score ?? 0.9,
    rerank_latency_ms: t.stage_timings?.reranking ?? t.rerank_latency_ms ?? 0,
    total_latency_ms: t.execution_time_ms ?? t.total_latency_ms ?? 0,
    prompt_tokens: t.token_usage?.prompt_tokens ?? t.prompt_tokens ?? 0,
    completion_tokens: t.token_usage?.completion_tokens ?? t.completion_tokens ?? 0,
    model: t.model || 'FastAPI RAG',
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
        // Fallback to non-streaming if stream endpoint returns 404 or fails
        if (response.status === 404) {
          return this.fallbackNonStreamingChat(message, sessionId, filters, model, callbacks);
        }
        const errorText = await response.text();
        throw new Error(`Chat API error (${response.status}): ${errorText}`);
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
        callbacks.onStart?.(data as { session_id: string; message_id: string });
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
            callbacks.onTrace?.(mapTrace(doneData.retrieval_trace));
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
        callbacks.onTrace({
          trace_id: res.id || `trace_${Date.now()}`,
          timestamp: new Date().toISOString(),
          original_query: message,
          total_chunks_retrieved: res.citations?.length || 0,
          top_rerank_score: res.citations?.[0]?.score || 0.9,
          rerank_latency_ms: Math.round((res.latency_ms || 300) * 0.2),
          total_latency_ms: res.latency_ms || 300,
          prompt_tokens: 150,
          completion_tokens: 120,
          model: model || 'FastAPI RAG',
        });
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
    const res = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        session_id: sessionId,
        filters: filters || {},
        model: model || 'default',
      }),
    });

    if (!res.ok) {
      throw new Error(`Chat API error (${res.status}): ${await res.text()}`);
    }

    return res.json();
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
      file_type: doc.file_type || doc.filename?.split('.').pop()?.toLowerCase() || 'unknown'
    }));
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
      status: doc.status || 'indexed',
      file_type: doc.file_type || doc.filename?.split('.').pop()?.toLowerCase() || 'unknown'
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
      const topScore = Array.isArray(t.rerank_scores) && t.rerank_scores.length > 0
        ? t.rerank_scores[0]
        : (typeof t.top_rerank_score === 'number' ? t.top_rerank_score : 0.88);

      const pTokens = t.token_usage?.prompt_tokens ?? t.prompt_tokens ?? 0;
      const cTokens = t.token_usage?.completion_tokens ?? t.completion_tokens ?? 0;

      return {
        trace_id: t.trace_id || `tr_${idx}_${Date.now()}`,
        timestamp: t.timestamp || new Date().toISOString(),
        original_query: t.query || t.original_query || 'Query',
        query_rewritten: t.rewritten_query || t.query_rewritten,
        expanded_queries: t.sub_queries || t.expanded_queries || [],
        total_chunks_retrieved: t.candidate_count ?? t.total_chunks_retrieved ?? 0,
        top_rerank_score: topScore,
        rerank_latency_ms: t.stage_timings?.reranking ?? t.rerank_latency_ms ?? 0,
        total_latency_ms: t.execution_time_ms ?? t.total_latency_ms ?? 0,
        prompt_tokens: pTokens,
        completion_tokens: cTokens,
        model: t.model || 'FastAPI RAG (Qwen 2.5)',
      };
    });

    const promptTokens = data.token_usage?.prompt_tokens ?? data.prompt_tokens ?? 0;
    const completionTokens = data.token_usage?.completion_tokens ?? data.completion_tokens ?? 0;
    const totalTokens = data.token_usage?.total_tokens ?? (promptTokens + completionTokens);

    return {
      total_queries: data.total_queries ?? 0,
      avg_latency_ms: data.avg_latency_ms ?? 0,
      avg_ttft_ms: data.avg_ttft_ms ?? 0,
      p95_latency_ms: data.p95_latency_ms ?? 0,
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: totalTokens,
      active_documents: data.active_documents ?? 0,
      indexed_chunks: data.indexed_chunks ?? 0,
      similarity_avg: data.score_distributions?.similarity_avg ?? 0,
      rerank_avg: data.score_distributions?.rerank_avg ?? 0,
      health: healthData,
      recent_traces: mappedTraces,
    };
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
}

export const apiClient = new ApiClient();
