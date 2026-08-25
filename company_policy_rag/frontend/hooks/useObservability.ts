import { useState, useEffect, useCallback, useRef } from 'react';
import {
  ObservabilityData,
  ObservabilitySummaryData,
  HealthStatus,
  TelemetryFilterOptions,
  QueryTrace,
} from '../lib/types';
import { apiClient } from '../lib/api-client';

const INITIAL_OBSERVABILITY: ObservabilityData = {
  total_queries: 0,
  avg_latency_ms: 0,
  avg_ttft_ms: 0,
  p95_latency_ms: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
  active_documents: 0,
  indexed_chunks: 0,
  similarity_avg: 0,
  rerank_avg: 0,
  health: {
    status: 'ok',
    redis: false,
    vector_db: true,
    models_loaded: true,
    backend_version: 'FastAPI RAG',
  },
  recent_traces: [],
};

export function useObservability() {
  const [data, setData] = useState<ObservabilityData>(INITIAL_OBSERVABILITY);
  const [summary, setSummary] = useState<ObservabilitySummaryData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [selectedTrace, setSelectedTrace] = useState<QueryTrace | null>(null);

  // Filter & Refresh Configuration
  const [timeRange, setTimeRange] = useState<string>('24h');
  const [filters, setFilters] = useState<TelemetryFilterOptions>({ timeRange: '24h' });
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  const [refreshIntervalMs, setRefreshIntervalMs] = useState<number>(3000);

  const filtersRef = useRef(filters);
  filtersRef.current = filters;

  const fetchTelemetry = useCallback(async (isBackground = false) => {
    if (!isBackground) {
      setLoading(true);
    } else {
      setIsRefreshing(true);
    }
    setError(null);

    try {
      const [summaryData, obsData] = await Promise.all([
        apiClient.getObservabilitySummary(filtersRef.current).catch(() => null),
        apiClient.getObservability().catch(() => null),
      ]);

      if (summaryData) {
        setSummary(summaryData);
      }
      if (obsData) {
        setData(obsData);
      } else if (summaryData) {
        setData({
          total_queries: summaryData.query_metrics.total_queries ?? 0,
          avg_latency_ms: summaryData.query_metrics.avg_latency_ms ?? 0,
          avg_ttft_ms: summaryData.query_metrics.avg_ttft_ms ?? 0,
          p95_latency_ms: summaryData.query_metrics.p95_latency_ms ?? 0,
          prompt_tokens: summaryData.tokens.total_prompt_tokens ?? 0,
          completion_tokens: summaryData.tokens.total_completion_tokens ?? 0,
          total_tokens: summaryData.tokens.total_tokens ?? 0,
          active_documents: summaryData.ingestion.documents_processed ?? 0,
          indexed_chunks: summaryData.ingestion.chunks_indexed ?? 0,
          similarity_avg: summaryData.retrieval_quality.avg_rerank_score ?? 0.95,
          rerank_avg: summaryData.retrieval_quality.avg_rerank_score ?? 0.95,
          health: {
            status: summaryData.health.api === 'healthy' ? 'ok' : 'degraded',
            redis: false,
            vector_db: summaryData.health.vector_db === 'healthy',
            models_loaded: summaryData.health.text_model === 'healthy',
            backend_version: `Qwen2.5 / ${summaryData.health.active_model_text}`,
          },
          recent_traces: summaryData.recent_traces,
        });
      }
      setLastUpdated(new Date());
    } catch (e: any) {
      console.warn('Observability telemetry fetch error:', e);
      setError(e.message || 'Failed to fetch observability telemetry');
    } finally {
      setLoading(false);
      setIsRefreshing(false);
    }
  }, []);

  const updateFilters = useCallback((newFilters: Partial<TelemetryFilterOptions>) => {
    setFilters((prev) => {
      const updated = { ...prev, ...newFilters };
      filtersRef.current = updated;
      return updated;
    });
    fetchTelemetry(false);
  }, [fetchTelemetry]);

  const handleTimeRangeChange = useCallback((newRange: string) => {
    setTimeRange(newRange);
    updateFilters({ timeRange: newRange });
  }, [updateFilters]);

  useEffect(() => {
    fetchTelemetry(false);
  }, [fetchTelemetry]);

  useEffect(() => {
    if (!autoRefresh || refreshIntervalMs <= 0) return;

    const interval = setInterval(() => {
      fetchTelemetry(true);
    }, refreshIntervalMs);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshIntervalMs, fetchTelemetry]);

  const clearData = useCallback(async () => {
    setLoading(true);
    await apiClient.clearTelemetry();
    await fetchTelemetry(false);
  }, [fetchTelemetry]);

  return {
    summary,
    data,
    health: data.health,
    loading,
    isRefreshing,
    error,
    lastUpdated,
    selectedTrace,
    setSelectedTrace,
    timeRange,
    setTimeRange: handleTimeRangeChange,
    filters,
    setFilters: updateFilters,
    autoRefresh,
    setAutoRefresh,
    refreshIntervalMs,
    setRefreshIntervalMs,
    refreshMetrics: () => fetchTelemetry(false),
    clearTelemetry: clearData,
  };
}
