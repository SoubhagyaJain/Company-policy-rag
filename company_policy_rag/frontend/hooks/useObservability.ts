import { useState, useEffect, useCallback } from 'react';
import { ObservabilityData, HealthStatus } from '../lib/types';
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
  const [loading, setLoading] = useState<boolean>(false);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [health, setHealth] = useState<HealthStatus>({
    status: 'ok',
    redis: false,
    vector_db: true,
    models_loaded: true,
  });

  const fetchMetrics = useCallback(async () => {
    try {
      const [obsData, healthData] = await Promise.all([
        apiClient.getObservability().catch(() => null),
        apiClient.getHealth().catch(() => null),
      ]);

      if (obsData) {
        setData(obsData);
        setLastUpdated(new Date());
      }
      if (healthData) {
        setHealth(healthData);
      }
    } catch (e) {
      console.warn('Observability telemetry fetch error:', e);
    }
  }, []);

  useEffect(() => {
    // Initial fetch
    fetchMetrics();
    // Real-time polling every 2.5 seconds
    const interval = setInterval(fetchMetrics, 2500);
    return () => clearInterval(interval);
  }, [fetchMetrics]);

  return {
    data,
    health,
    loading,
    lastUpdated,
    refreshMetrics: async () => {
      setLoading(true);
      await fetchMetrics();
      setLoading(false);
    },
  };
}
