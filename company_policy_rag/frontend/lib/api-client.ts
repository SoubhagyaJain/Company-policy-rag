import {
  Citation,
  QueryTrace,
  DocumentItem,
  ObservabilityData,
  HealthStatus,
  FilterOptions,
} from './types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';

export interface StreamCallbacks {
  onStart?: (data: { session_id: string; message_id: string }) => void;
  onChunk?: (chunk: string) => void;
  onCitation?: (citation: Citation) => void;
  onTrace?: (trace: QueryTrace) => void;
  onDone?: (data: { latency_ms: number; metrics?: Record<string, unknown> }) => void;
  onError?: (error: Error) => void;
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
      } else {
        throw err;
      }
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
        callbacks.onCitation?.(data as Citation);
        break;
      case 'trace':
        callbacks.onTrace?.(data as QueryTrace);
        break;
      case 'done':
        callbacks.onDone?.(data as { latency_ms: number; metrics?: Record<string, unknown> });
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
    const res = await this.sendChatMessage(message, sessionId, filters, model);
    if (callbacks?.onChunk) {
      callbacks.onChunk(res.answer);
    }
    if (res.citations && callbacks?.onCitation) {
      res.citations.forEach((c) => callbacks.onCitation?.(c));
    }
    if (res.metrics && callbacks?.onTrace) {
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
      callbacks.onDone({ latency_ms: res.latency_ms || 300 });
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
    const res = await fetch(`${this.baseUrl}/api/admin/observability`);
    if (!res.ok) {
      throw new Error(`Failed to fetch observability telemetry (${res.status})`);
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
    return res.json();
  }
}

export const apiClient = new ApiClient();
