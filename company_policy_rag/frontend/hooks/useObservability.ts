import { useState, useEffect, useCallback } from 'react';
import { ObservabilityData, HealthStatus } from '../lib/types';
import { apiClient } from '../lib/api-client';

const DEMO_OBSERVABILITY: ObservabilityData = {
  total_queries: 1420,
  avg_latency_ms: 285.4,
  prompt_tokens: 345200,
  completion_tokens: 182400,
  total_tokens: 527600,
  health: {
    status: 'ok',
    redis: true,
    vector_db: true,
    models_loaded: true,
    backend_version: 'v1.4.2-fastapi',
  },
  recent_traces: [
    {
      trace_id: 'tr_8912a7',
      timestamp: new Date(Date.now() - 1000 * 60 * 2).toISOString(),
      original_query: 'What is the PTO rollover policy at year end?',
      query_rewritten: 'annual paid time off rollover unused vacation balance policy',
      expanded_queries: [
        'PTO carryover limit per year',
        'unused leave balance forfeiture policy',
      ],
      total_chunks_retrieved: 8,
      top_rerank_score: 0.942,
      rerank_latency_ms: 38.2,
      total_latency_ms: 245.0,
      prompt_tokens: 340,
      completion_tokens: 115,
      model: 'FastAPI + HyDE + BGE-Reranker',
    },
    {
      trace_id: 'tr_8912a6',
      timestamp: new Date(Date.now() - 1000 * 60 * 15).toISOString(),
      original_query: 'How do I submit an expense for remote home office equipment?',
      query_rewritten: 'remote employee home office hardware expense reimbursement process',
      expanded_queries: [
        'home office stipend allowance limit',
        'submitting receipts in Concur or Workday',
      ],
      total_chunks_retrieved: 6,
      top_rerank_score: 0.915,
      rerank_latency_ms: 44.1,
      total_latency_ms: 310.5,
      prompt_tokens: 410,
      completion_tokens: 160,
      model: 'FastAPI + HyDE + BGE-Reranker',
    },
    {
      trace_id: 'tr_8912a5',
      timestamp: new Date(Date.now() - 1000 * 60 * 42).toISOString(),
      original_query: 'What are the rules regarding secondary employment or freelancing?',
      query_rewritten: 'outside employment conflict of interest dual employment policy',
      expanded_queries: [
        'freelancing permission disclosure form',
        'moonlighting policy intellectual property',
      ],
      total_chunks_retrieved: 10,
      top_rerank_score: 0.887,
      rerank_latency_ms: 51.0,
      total_latency_ms: 365.2,
      prompt_tokens: 520,
      completion_tokens: 195,
      model: 'FastAPI + HyDE + BGE-Reranker',
    },
  ],
};

export function useObservability() {
  const [data, setData] = useState<ObservabilityData>(DEMO_OBSERVABILITY);
  const [loading, setLoading] = useState<boolean>(true);
  const [health, setHealth] = useState<HealthStatus>({
    status: 'ok',
    redis: true,
    vector_db: true,
    models_loaded: true,
  });

  const fetchMetrics = useCallback(async () => {
    setLoading(true);
    try {
      const [obsData, healthData] = await Promise.all([
        apiClient.getObservability().catch(() => null),
        apiClient.getHealth().catch(() => null),
      ]);

      if (obsData) {
        setData(obsData);
      }
      if (healthData) {
        setHealth(healthData);
      }
    } catch (e) {
      console.warn('Observability endpoint fallback:', e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchMetrics();
    // Auto refresh telemetry every 30 seconds
    const interval = setInterval(fetchMetrics, 30000);
    return () => clearInterval(interval);
  }, [fetchMetrics]);

  return {
    data,
    health,
    loading,
    refreshMetrics: fetchMetrics,
  };
}
