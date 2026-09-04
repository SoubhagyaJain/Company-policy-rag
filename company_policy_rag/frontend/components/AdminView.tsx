'use client';

import React, { useRef, useState } from 'react';
import { useSmoothScroll } from '../hooks/useSmoothScroll';
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Bot,
  Brain,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock,
  Code,
  CornerDownRight,
  Cpu,
  Database,
  Eye,
  FileCode,
  FileSpreadsheet,
  FileText,
  Filter,
  Flame,
  HelpCircle,
  History,
  Image as ImageIcon,
  Layers,
  Maximize2,
  Minimize2,
  Pause,
  Play,
  RefreshCw,
  Search,
  Server,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Table as TableIcon,
  Trash2,
  TrendingDown,
  TrendingUp,
  Workflow,
  XCircle,
  Zap,
} from 'lucide-react';

import { useObservability } from '../hooks/useObservability';
import {
  AlertItemData,
  ErrorIncidentData,
  QueryTrace,
  SubsystemHealth,
  SubsystemStatusType,
} from '../lib/types';
import { QueryTraceDrawer } from './QueryTraceDrawer';

function formatLatency(ms?: number | null): string {
  if (ms === null || ms === undefined || isNaN(ms)) return '0 ms';
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(2)} s`;
  }
  return `${Math.round(ms)} ms`;
}

function formatTokens(count?: number | null): string {
  if (count === null || count === undefined || isNaN(count)) return '0';
  if (count >= 1_000_000) {
    return `${(count / 1_000_000).toFixed(1)}M`;
  }
  if (count >= 1_000) {
    return `${(count / 1_000).toFixed(1)}k`;
  }
  return `${count}`;
}

function formatPercent(val?: number | null): string {
  if (val === null || val === undefined || isNaN(val)) return '0%';
  const clamped = Math.min(100, Math.max(0, val * 100));
  return `${Math.round(clamped)}%`;
}

function formatFloatPercent(val?: number | null): string {
  if (val === null || val === undefined || isNaN(val)) return '0%';
  const clamped = Math.min(100, Math.max(0, val * 100));
  return `${clamped.toFixed(1)}%`;
}

function formatMeasuredLatency(ms?: number | null): string {
  return ms === null || ms === undefined || isNaN(ms) ? 'Not measured' : formatLatency(ms);
}

function formatMeasuredPercent(val?: number | null): string {
  return val === null || val === undefined || isNaN(val) ? 'Not measured' : formatFloatPercent(val);
}

export const AdminView: React.FC = () => {
  // expandedTraceId MUST remain the first useState hook for SSR test-harness compatibility
  const [expandedTraceId, setExpandedTraceId] = useState<string | null>(null);

  // Buttery inertia scrolling over the live WebGL background.
  const scrollRef = useRef<HTMLDivElement>(null);
  useSmoothScroll(scrollRef);

  const {
    summary,
    data,
    loading,
    isRefreshing,
    error,
    lastUpdated,
    selectedTrace,
    setSelectedTrace,
    timeRange,
    setTimeRange,
    filters,
    setFilters,
    autoRefresh,
    setAutoRefresh,
    refreshIntervalMs,
    setRefreshIntervalMs,
    refreshMetrics,
    clearTelemetry,
    deleteTrace,
  } = useObservability();

  const [activeTab, setActiveTab] = useState<'overview' | 'queries' | 'models' | 'caches' | 'ingestion' | 'errors'>('overview');
  const [isFullscreen, setIsFullscreen] = useState(false);

  // In-app confirmation (replaces window.confirm, which browsers/webviews can
  // silently suppress — that made destructive actions like delete appear dead).
  const [confirmAction, setConfirmAction] = useState<null | {
    title: string;
    message: string;
    confirmLabel: string;
    onConfirm: () => void | Promise<unknown>;
  }>(null);
  const [confirmBusy, setConfirmBusy] = useState(false);

  const runConfirm = async () => {
    if (!confirmAction || confirmBusy) return;
    try {
      setConfirmBusy(true);
      await confirmAction.onConfirm();
    } finally {
      setConfirmBusy(false);
      setConfirmAction(null);
    }
  };

  React.useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, []);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().then(() => setIsFullscreen(true)).catch(() => {});
    } else {
      document.exitFullscreen().then(() => setIsFullscreen(false)).catch(() => {});
    }
  };

  const getStatusBadge = (status?: SubsystemStatusType | string) => {
    switch (status?.toLowerCase()) {
      case 'healthy':
      case 'ok':
      case 'ready':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-semibold rounded-full bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border border-emerald-500/20 dark:bg-emerald-950/40">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
            HEALTHY
          </span>
        );
      case 'degraded':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-semibold rounded-full bg-amber-500/10 text-amber-700 dark:text-amber-400 border border-amber-500/20 dark:bg-amber-950/40">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
            DEGRADED
          </span>
        );
      case 'unavailable':
      case 'failed':
      case 'error':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-semibold rounded-full bg-rose-500/10 text-rose-700 dark:text-rose-400 border border-rose-500/20 dark:bg-rose-950/40">
            <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />
            UNAVAILABLE
          </span>
        );
      case 'disabled':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-semibold rounded-full bg-cream-200 text-charcoal-muted border border-sand-border dark:bg-sand-dark dark:text-cream-400 dark:border-sand-darkBorder">
            DISABLED
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-semibold rounded-full bg-cream-200 text-charcoal border border-sand-border dark:bg-sand-dark dark:text-cream-300 dark:border-sand-darkBorder">
            {status || 'UNKNOWN'}
          </span>
        );
    }
  };

  const health: SubsystemHealth = summary?.health || {
    api: data.health.status === 'ok' ? 'healthy' : 'unavailable',
    ollama: 'unavailable',
    vector_db: data.health.vector_db ? 'healthy' : 'unavailable',
    bm25: 'unavailable',
    embedding_model: 'unavailable',
    text_model: data.health.models_loaded ? 'healthy' : 'unavailable',
    vision_model: 'unavailable',
    semantic_cache: 'unavailable',
    vision_cache: 'unavailable',
    memory: 'unavailable',
    uptime_seconds: 0,
    error_rate: 0.0,
    active_model_text: 'Unknown',
    active_model_vision: 'Unknown',
  };

  const qm = summary?.query_metrics;
  const lb = summary?.latency_breakdown;
  const rq = summary?.retrieval_quality;
  const gd = summary?.grounding;
  const models = summary?.models;
  const tokens = summary?.tokens;
  const mem = summary?.memory;
  const ing = summary?.ingestion;
  const caches = summary?.caches;
  const alerts = summary?.alerts || [];
  const recentTraces = (summary?.recent_traces && summary.recent_traces.length > 0)
    ? summary.recent_traces
    : (data.recent_traces || []);
  const recentIncidents = summary?.recent_incidents || [];

  const totalQueriesCount = qm?.total_queries ?? data.total_queries ?? recentTraces.length;
  const avgLat = qm?.avg_latency_ms ?? data.avg_latency_ms ?? 0;
  const p95Lat = qm?.p95_latency_ms ?? data.p95_latency_ms ?? 0;
  const avgTtft = qm?.avg_ttft_ms ?? data.avg_ttft_ms ?? 0;
  const totTokens = tokens?.total_tokens ?? data.total_tokens ?? 0;

  const toggleTraceExpand = (traceId: string) => {
    setExpandedTraceId((prev) => (prev === traceId ? null : traceId));
  };

  // Render unified table of query traces
  const renderTracesTable = () => (
    <div className="space-y-3">
      {/* Desktop Table View with Accordion */}
      <div className="hidden lg:block overflow-x-auto custom-scrollbar">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-sand-border dark:border-sand-darkBorder text-charcoal-muted dark:text-cream-400 font-semibold bg-cream-100/70 dark:bg-cream-950/70">
              <th className="py-3 px-3">Original Query</th>
              <th className="py-3 px-3">Query Type</th>
              <th className="py-3 px-3">Verification</th>
              <th className="py-3 px-3">Filter Status</th>
              <th className="py-3 px-3">Chunks</th>
              <th className="py-3 px-3">Rerank</th>
              <th className="py-3 px-3">Latency</th>
              <th className="py-3 px-3">Tokens</th>
              <th className="py-3 px-3 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-sand-border/60 dark:divide-sand-darkBorder/60">
            {recentTraces.map((trace) => {
              const isExpanded = expandedTraceId === trace.trace_id;
              const rawType = trace.query_type || 'unknown';
              const capType = rawType.charAt(0).toUpperCase() + rawType.slice(1);
              const retries = trace.retry_count ?? 0;
              const vScore = trace.verification?.composite_score ?? trace.verification_score;
              const isRelaxed = Boolean(trace.filter_relaxed);
              const expandedList = trace.expanded_queries || trace.sub_queries || [];
              const hasFilters = trace.inferred_filters && Object.keys(trace.inferred_filters).length > 0;

              return (
                <React.Fragment key={trace.trace_id}>
                  <tr
                    onClick={() => toggleTraceExpand(trace.trace_id)}
                    className={`hover:bg-cream-100/60 dark:hover:bg-[#22211E]/60 cursor-pointer transition-colors ${
                      isExpanded ? 'bg-cream-100/80 dark:bg-sand-dark/90' : ''
                    }`}
                  >
                    <td className="py-3 px-3 font-medium text-charcoal dark:text-cream-100 max-w-sm xl:max-w-xl truncate">
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono text-terracotta-600 dark:text-terracotta-500 font-semibold">{trace.request_id || trace.trace_id}</span>
                        <span className="text-charcoal-muted dark:text-cream-400 truncate">{trace.original_query}</span>
                      </div>
                    </td>
                    <td className="py-3 px-3">
                      <div className="flex items-center gap-1">
                        <span className="px-2 py-0.5 rounded-lg bg-cream-200/80 dark:bg-sand-dark text-charcoal dark:text-cream-200 font-medium border border-sand-border/60 dark:border-sand-darkBorder/60">
                          {capType}
                        </span>
                        <span className="text-[11px] font-mono text-charcoal-muted dark:text-cream-400">
                          ({trace.routing_confidence !== undefined && trace.routing_confidence !== null ? formatPercent(trace.routing_confidence) : 'Not measured'})
                        </span>
                      </div>
                    </td>
                    <td className="py-3 px-3">
                      {trace.conversational_bypass ? (
                        <span className="px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-700 dark:text-amber-400 border border-amber-500/20 text-[11px] font-semibold">
                          Conversational
                        </span>
                      ) : vScore === null || vScore === undefined ? (
                        <span className="px-2 py-0.5 rounded-full bg-cream-200 text-charcoal-muted dark:bg-sand-darkBorder dark:text-cream-400 border border-sand-border dark:border-sand-darkBorder text-[11px] font-semibold">
                          Not measured
                        </span>
                      ) : trace.faithfulness_passed !== false ? (
                        <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border border-emerald-500/20 text-[11px] font-semibold">
                          {formatPercent(vScore)} Pass
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 rounded-full bg-rose-500/10 text-rose-700 dark:text-rose-400 border border-rose-500/20 text-[11px] font-semibold">
                          {formatPercent(vScore)} Fail
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-3 font-mono">
                      {hasFilters ? (
                        <div className="flex items-center gap-1">
                          <span className="truncate max-w-[120px] text-charcoal dark:text-cream-300">
                            {Object.entries(trace.inferred_filters!).map(([k, v]) => `${k}: ${v}`).join(', ')}
                          </span>
                          {isRelaxed && (
                            <span className="px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-700 dark:text-amber-400 border border-amber-500/20 text-[10px]">
                              relaxed
                            </span>
                          )}
                        </div>
                      ) : isRelaxed ? (
                        <span className="px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-700 dark:text-amber-400 border border-amber-500/20 text-[10px]">
                          relaxed
                        </span>
                      ) : (
                        <span className="text-charcoal-muted dark:text-cream-500">None</span>
                      )}
                    </td>
                    <td className="py-3 px-3 font-mono text-charcoal dark:text-cream-300">
                      {trace.total_chunks_retrieved}
                    </td>
                    <td className="py-3 px-3 font-mono text-emerald-600 dark:text-emerald-400 font-semibold">
                      {trace.top_rerank_score !== undefined && trace.top_rerank_score !== null ? `${(trace.top_rerank_score * 100).toFixed(1)}%` : 'Not measured'}
                    </td>
                    <td className="py-3 px-3 font-mono font-semibold text-charcoal dark:text-cream-100">
                      {formatLatency(trace.total_latency_ms)}
                    </td>
                    <td className="py-3 px-3 font-mono text-charcoal-muted dark:text-cream-400">
                      {formatTokens(trace.prompt_tokens + trace.completion_tokens)}
                    </td>
                    <td className="py-3 px-3 text-right">
                      <div className="flex items-center justify-end gap-1">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedTrace(trace);
                          }}
                          className="p-1.5 rounded-lg bg-cream-200/80 hover:bg-cream-300 dark:bg-sand-dark dark:hover:bg-[#2A2925] text-charcoal dark:text-cream-200 border border-sand-border/60 dark:border-sand-darkBorder/60 transition-colors"
                          title="Open slide-over drawer"
                        >
                          <Maximize2 className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleTraceExpand(trace.trace_id);
                          }}
                          className="p-1.5 rounded-lg bg-cream-200/80 hover:bg-cream-300 dark:bg-sand-dark dark:hover:bg-[#2A2925] text-charcoal dark:text-cream-200 border border-sand-border/60 dark:border-sand-darkBorder/60 transition-colors"
                          title="Toggle inline details"
                        >
                          {isExpanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            const q = trace.original_query || 'this trace';
                            setConfirmAction({
                              title: 'Delete trace',
                              message: `Delete trace "${q.slice(0, 60)}${q.length > 60 ? '…' : ''}"? This cannot be undone.`,
                              confirmLabel: 'Delete trace',
                              onConfirm: () => deleteTrace(trace.trace_id),
                            });
                          }}
                          className="p-1.5 rounded-lg bg-rose-500/10 hover:bg-rose-500/20 text-rose-600 dark:text-rose-400 border border-rose-500/20 transition-colors"
                          title="Delete trace"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>

                  {/* Inline Expandable Detail Panel */}
                  {isExpanded && (
                    <tr className="bg-cream-50/90 dark:bg-cream-950/90">
                      <td colSpan={9} className="p-4 border-b border-sand-border dark:border-sand-darkBorder">
                        <div className="space-y-3 text-xs">
                          {/* Strategy & Anchor Chips */}
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="px-2.5 py-1 rounded-lg bg-cream-200 dark:bg-sand-dark text-charcoal dark:text-cream-200 font-mono border border-sand-border/70 dark:border-sand-darkBorder/60">
                              Strategy: {trace.retrieval_strategy || 'Not recorded'}
                            </span>
                            <span className="px-2.5 py-1 rounded-lg bg-cream-200 dark:bg-sand-dark text-charcoal dark:text-cream-200 font-mono border border-sand-border/70 dark:border-sand-darkBorder/60">
                              Model: {trace.model}
                            </span>
                            <span className="px-2.5 py-1 rounded-lg bg-cream-200 dark:bg-sand-dark text-charcoal dark:text-cream-200 font-mono border border-sand-border/70 dark:border-sand-darkBorder/60">
                              Confidence: {trace.routing_confidence !== undefined && trace.routing_confidence !== null ? formatPercent(trace.routing_confidence) : 'Not measured'}
                            </span>
                            <span className="px-2.5 py-1 rounded-lg bg-cream-200 dark:bg-sand-dark text-charcoal dark:text-cream-200 font-mono border border-sand-border/70 dark:border-sand-darkBorder/60">
                              Routing: {trace.query_type || 'Not recorded'}
                            </span>
                          </div>

                          {/* Rewritten Query Section */}
                          {trace.query_rewritten && (
                            <div className="p-3 rounded-xl bg-cream-100/90 dark:bg-sand-dark/90 border border-sand-border dark:border-sand-darkBorder">
                              <span className="font-semibold text-charcoal dark:text-cream-100 block mb-1">Rewritten Query</span>
                              <p className="text-terracotta-600 dark:text-terracotta-400 font-mono">{trace.query_rewritten}</p>
                            </div>
                          )}

                          {/* Semantic Cache Hit Card */}
                          {trace.cache_hit && (
                            <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-800 dark:text-emerald-300">
                              <div className="font-semibold flex items-center gap-1.5">
                                <Zap className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                                Semantic Cache Hit
                              </div>
                              <p className="mt-1 text-emerald-700/90 dark:text-emerald-200/90 font-mono">
                                Cache similarity: {trace.cache_similarity !== null && trace.cache_similarity !== undefined ? `${(trace.cache_similarity * 100).toFixed(1)}%` : 'Not measured'}
                              </p>
                            </div>
                          )}

                          {/* Filter Fallback Card */}
                          {isRelaxed && (
                            <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-800 dark:text-amber-300">
                              <div className="font-semibold flex items-center gap-1.5">
                                <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400" />
                                Filter Fallback Triggered
                              </div>
                              <p className="mt-1 text-amber-700/90 dark:text-amber-200/90">
                                Filters Relaxed: Strict metadata filters yielded insufficient candidates. Filters were relaxed to ensure complete context recall.
                              </p>
                            </div>
                          )}

                          {/* Expanded Queries Card */}
                          {expandedList.length > 0 && (
                            <div className="p-3 rounded-xl bg-cream-100/90 dark:bg-sand-dark/90 border border-sand-border dark:border-sand-darkBorder">
                              <span className="font-semibold text-charcoal dark:text-cream-100 block mb-1">
                                Expanded Queries ({expandedList.length})
                              </span>
                              <ul className="list-disc list-inside space-y-0.5 text-charcoal-muted dark:text-cream-400 font-mono">
                                {expandedList.map((q: string, idx: number) => (
                                  <li key={idx}>{q}</li>
                                ))}
                              </ul>
                            </div>
                          )}

                          {/* Inferred vs Applied Filters Card */}
                          {(hasFilters || (trace.applied_filters && Object.keys(trace.applied_filters).length > 0)) && (
                            <div className="p-3 rounded-xl bg-cream-100/90 dark:bg-sand-dark/90 border border-sand-border dark:border-sand-darkBorder grid grid-cols-2 gap-4">
                              <div>
                                <span className="font-semibold text-charcoal dark:text-cream-100 block mb-1">Inferred Filters (Query Router)</span>
                                <div className="space-y-1 font-mono text-charcoal dark:text-cream-300">
                                  {Object.entries(trace.inferred_filters || {}).map(([k, v]) => (
                                    <div key={k} className="flex gap-2">
                                      <span className="text-charcoal-muted dark:text-cream-400">{k}:</span>
                                      <span>{typeof v === 'object' ? JSON.stringify(v) : String(v)}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                              <div>
                                <span className="font-semibold text-charcoal dark:text-cream-100 block mb-1">Applied Filters (Vector &amp; BM25)</span>
                                <div className="space-y-1 font-mono text-charcoal dark:text-cream-300">
                                  {Object.entries(trace.applied_filters || {}).map(([k, v]) => (
                                    <div key={k} className="flex gap-2">
                                      <span className="text-charcoal-muted dark:text-cream-400">{k}:</span>
                                      <span>{typeof v === 'object' ? JSON.stringify(v) : String(v)}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            </div>
                          )}

                          {/* Self-Reflection Verification Report & Retries Card */}
                          <div className="p-3 rounded-xl bg-cream-100/90 dark:bg-sand-dark/90 border border-sand-border dark:border-sand-darkBorder space-y-2">
                            <div className="flex justify-between items-center">
                              <span className="font-semibold text-charcoal dark:text-cream-100">
                                Self-Reflection Verification Report — {vScore === null || vScore === undefined
                                  ? 'No verifier measurement recorded'
                                  : retries === 0
                                  ? '0 retries · Passed self-reflection on initial attempt'
                                  : retries === 1
                                  ? 'Verification Retries (1) · 1 cycle'
                                  : `Verification Retries (${retries}) · ${retries} cycles`}
                              </span>
                              <span className="font-mono text-emerald-600 dark:text-emerald-400 font-bold">
                                Composite: {vScore === null || vScore === undefined ? 'Not measured' : formatPercent(vScore)}
                              </span>
                            </div>

                            {trace.retry_reasons && trace.retry_reasons.length > 0 && (
                              <ul className="list-disc list-inside space-y-0.5 text-amber-700 dark:text-amber-300/90 font-mono">
                                {trace.retry_reasons.map((r, idx) => (
                                  <li key={idx}>{r}</li>
                                ))}
                              </ul>
                            )}

                            {trace.verification && (
                              <div className="pt-2 border-t border-sand-border dark:border-sand-darkBorder space-y-2">
                                <div className="grid grid-cols-4 gap-2 text-charcoal-muted dark:text-cream-400">
                                  <div>Faithfulness: <span className="text-charcoal dark:text-cream-200 font-mono">{trace.verification.faithfulness === undefined ? 'Not measured' : formatPercent(trace.verification.faithfulness)}</span></div>
                                  <div>Completeness: <span className="text-charcoal dark:text-cream-200 font-mono">{trace.verification.completeness === undefined ? 'Not measured' : formatPercent(trace.verification.completeness)}</span></div>
                                  <div>Citation Coverage: <span className="text-charcoal dark:text-cream-200 font-mono">{trace.verification.citation_coverage === undefined ? 'Not measured' : formatPercent(trace.verification.citation_coverage)}</span></div>
                                  <div>Coherence: <span className="text-charcoal dark:text-cream-200 font-mono">{trace.verification.coherence === undefined ? 'Not measured' : formatPercent(trace.verification.coherence)}</span></div>
                                </div>

                                {trace.verification.missing_aspects && trace.verification.missing_aspects.length > 0 && (
                                  <div>
                                    <span className="text-rose-600 dark:text-rose-400 font-semibold block mb-0.5">Missing Aspects</span>
                                    <ul className="list-disc list-inside text-rose-700 dark:text-rose-300/90 font-mono">
                                      {trace.verification.missing_aspects.map((m, idx) => (
                                        <li key={idx}>{m}</li>
                                      ))}
                                    </ul>
                                  </div>
                                )}

                                {trace.verification.unsupported_claims && trace.verification.unsupported_claims.length > 0 && (
                                  <div>
                                    <span className="text-rose-600 dark:text-rose-400 font-semibold block mb-0.5">Unsupported Claims</span>
                                    <ul className="list-disc list-inside text-rose-700 dark:text-rose-300/90 font-mono">
                                      {trace.verification.unsupported_claims.map((u, idx) => (
                                        <li key={idx}>{u}</li>
                                      ))}
                                    </ul>
                                  </div>
                                )}

                                {trace.verification.critique && (
                                  <div className="text-charcoal dark:text-cream-200 bg-cream-50 dark:bg-cream-950 p-2.5 rounded-lg border border-sand-border dark:border-sand-darkBorder">
                                    <span className="text-amber-700 dark:text-amber-400 font-semibold block mb-0.5">Reflection Critique: </span>
                                    {trace.verification.critique}
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Mobile Card View with Accordion */}
      <div className="lg:hidden space-y-3">
        {recentTraces.map((trace) => {
          const isExpanded = expandedTraceId === trace.trace_id;
          const rawType = trace.query_type || 'unknown';
          const capType = rawType.charAt(0).toUpperCase() + rawType.slice(1);
          const expandedList = trace.expanded_queries || trace.sub_queries || [];
          const retries = trace.retry_count ?? 0;
          const vScore = trace.verification?.composite_score ?? trace.verification_score;

          return (
            <div
              key={trace.trace_id}
              onClick={() => toggleTraceExpand(trace.trace_id)}
              className="p-4 rounded-xl bg-cream-100/90 dark:bg-sand-dark/90 border border-sand-border/80 dark:border-sand-darkBorder/80 text-xs space-y-2 cursor-pointer transition-colors"
            >
              <div className="flex justify-between items-center">
                <span className="font-mono text-terracotta-600 dark:text-terracotta-500 font-bold">{trace.request_id || trace.trace_id}</span>
                <span className="px-2 py-0.5 rounded-md bg-cream-200 dark:bg-sand-darkBorder text-charcoal dark:text-cream-200 font-medium">{capType}</span>
              </div>
              <p className="text-charcoal dark:text-cream-100 font-medium">{trace.original_query}</p>
              <div className="flex justify-between text-charcoal-muted dark:text-cream-400 font-mono text-[11px]">
                <span>Latency: {formatLatency(trace.total_latency_ms)}</span>
                <span>TTFT: {formatLatency(trace.ttft_ms)}</span>
                <span>Confidence: {trace.routing_confidence !== undefined && trace.routing_confidence !== null ? formatPercent(trace.routing_confidence) : 'Not measured'}</span>
              </div>

              {isExpanded && (
                <div className="pt-2 border-t border-sand-border dark:border-sand-darkBorder space-y-2">
                  <div className="font-mono text-charcoal dark:text-cream-300">
                    Strategy: {trace.retrieval_strategy || 'Not recorded'}
                  </div>

                  {trace.query_rewritten && (
                    <div className="p-2 rounded bg-cream-200/80 dark:bg-cream-950">
                      <span className="font-semibold text-charcoal dark:text-cream-100 block mb-1">Rewritten Query</span>
                      <p className="text-terracotta-600 dark:text-terracotta-400 font-mono">{trace.query_rewritten}</p>
                    </div>
                  )}

                  {trace.cache_hit && (
                    <div className="p-2 rounded bg-emerald-500/10 text-emerald-800 dark:text-emerald-300">
                      Semantic Cache Hit ({trace.cache_similarity !== null && trace.cache_similarity !== undefined ? `${(trace.cache_similarity * 100).toFixed(1)}%` : 'similarity not recorded'})
                    </div>
                  )}

                  {expandedList.length > 0 && (
                    <div className="p-2 rounded bg-cream-200/80 dark:bg-cream-950">
                      <span className="font-semibold text-charcoal dark:text-cream-100 block mb-1">Expanded Queries</span>
                      <ul className="list-disc list-inside text-charcoal-muted dark:text-cream-400 font-mono">
                        {expandedList.map((q: string, idx: number) => (
                          <li key={idx}>{q}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {trace.inferred_filters && Object.keys(trace.inferred_filters).length > 0 && (
                    <div className="p-2 rounded bg-cream-200/80 dark:bg-cream-950 font-mono text-charcoal dark:text-cream-300">
                      {Object.entries(trace.inferred_filters).map(([k, v]) => (
                        <div key={k}>
                          <span className="text-charcoal-muted dark:text-cream-400">{k}:</span> {typeof v === 'object' ? JSON.stringify(v) : String(v)}
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="p-2 rounded bg-cream-200/80 dark:bg-cream-950">
                    <span className="font-semibold text-charcoal dark:text-cream-100 block">
                      Self-Reflection Verification Report — {retries === 0
                        ? '0 retries · Passed self-reflection on initial attempt'
                        : retries === 1
                        ? 'Verification Retries (1) · 1 cycle'
                        : `Verification Retries (${retries}) · ${retries} cycles`}
                    </span>
                    <span className="font-mono text-emerald-600 dark:text-emerald-400 block mt-1 font-bold">
                      Composite Score: {vScore === null || vScore === undefined ? 'Not measured' : formatPercent(vScore)}
                    </span>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );

  return (
    <div ref={scrollRef} className="flex-1 h-full overflow-y-auto p-4 sm:p-6 lg:p-8 sp-scroll sp-text">
      <div className="w-full space-y-6 pb-12">
        {/* ── HEADER & AUTO-REFRESH CONTROLS ─────────────────────── */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-5 rounded-2xl bg-white/80 dark:bg-sand-dark/80 backdrop-blur-xl border border-sand-border/80 dark:border-sand-darkBorder/80 shadow-soft dark:shadow-glassDark">
          <div>
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-terracotta-500/10 dark:bg-terracotta-500/20 border border-terracotta-500/30 text-terracotta-600 dark:text-terracotta-500">
                <Activity className="w-6 h-6" />
              </div>
              <div>
                <h1 className="text-xl font-bold tracking-tight text-charcoal dark:text-cream-100 flex items-center gap-2 font-serif">
                  Observability &amp; Telemetry
                  {isRefreshing && (
                    <RefreshCw className="w-4 h-4 text-terracotta-600 dark:text-terracotta-500 animate-spin" />
                  )}
                </h1>
                <p className="text-xs text-charcoal-muted dark:text-cream-400 mt-0.5">
                  Real-time RAG diagnostics, 16-stage latency waterfall, grounding metrics &amp; multi-model monitoring ({recentTraces.length} traces)
                </p>
              </div>
            </div>
          </div>

          {/* Global Controls */}
          <div className="flex items-center flex-wrap gap-2">
            {/* Time Range Selector */}
            <div className="flex items-center bg-cream-100/90 dark:bg-cream-950/90 p-1 rounded-xl border border-sand-border/70 dark:border-sand-darkBorder/70 text-xs">
              {['5m', '15m', '1h', '6h', '24h', '7d'].map((tr) => (
                <button
                  key={tr}
                  onClick={() => setTimeRange(tr)}
                  className={`px-2.5 py-1 rounded-lg font-medium transition-all ${
                    timeRange === tr
                      ? 'bg-terracotta-600 dark:bg-terracotta-500 text-white font-semibold shadow-sm'
                      : 'text-charcoal-muted dark:text-cream-400 hover:text-charcoal dark:hover:text-cream-200'
                  }`}
                >
                  {tr}
                </button>
              ))}
            </div>

            {/* Auto Refresh Toggle */}
            <div className="flex items-center bg-cream-100/90 dark:bg-cream-950/90 px-2.5 py-1 rounded-xl border border-sand-border/70 dark:border-sand-darkBorder/70 text-xs gap-2">
              <button
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`flex items-center gap-1 px-2 py-0.5 rounded font-medium transition-colors ${
                  autoRefresh ? 'text-emerald-600 dark:text-emerald-400' : 'text-charcoal-muted dark:text-cream-500'
                }`}
                title={autoRefresh ? 'Auto-refresh active' : 'Auto-refresh paused'}
              >
                {autoRefresh ? <Play className="w-3.5 h-3.5 fill-current" /> : <Pause className="w-3.5 h-3.5" />}
                <span>{autoRefresh ? 'Live' : 'Paused'}</span>
              </button>
              <select
                value={refreshIntervalMs}
                onChange={(e) => setRefreshIntervalMs(Number(e.target.value))}
                disabled={!autoRefresh}
                className="bg-transparent text-charcoal dark:text-cream-300 focus:outline-none cursor-pointer text-xs"
              >
                <option value={2000} className="bg-white dark:bg-sand-dark text-charcoal dark:text-cream-100">2s</option>
                <option value={3000} className="bg-white dark:bg-sand-dark text-charcoal dark:text-cream-100">3s</option>
                <option value={5000} className="bg-white dark:bg-sand-dark text-charcoal dark:text-cream-100">5s</option>
                <option value={10000} className="bg-white dark:bg-sand-dark text-charcoal dark:text-cream-100">10s</option>
                <option value={30000} className="bg-white dark:bg-sand-dark text-charcoal dark:text-cream-100">30s</option>
              </select>
            </div>

            {/* Manual Refresh */}
            <button
              onClick={() => refreshMetrics()}
              disabled={loading}
              className="p-2 rounded-xl bg-cream-100 dark:bg-sand-dark hover:bg-cream-200 dark:hover:bg-[#2A2925] text-charcoal dark:text-cream-200 border border-sand-border dark:border-sand-darkBorder transition-colors disabled:opacity-50"
              title="Refresh metrics now"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            </button>

            {/* Fullscreen Toggle */}
            <button
              onClick={toggleFullscreen}
              className="p-2 rounded-xl bg-cream-100 dark:bg-sand-dark hover:bg-cream-200 dark:hover:bg-[#2A2925] text-charcoal dark:text-cream-200 border border-sand-border dark:border-sand-darkBorder transition-colors"
              title={isFullscreen ? 'Exit full screen' : 'Expand full screen'}
            >
              {isFullscreen ? <Minimize2 className="w-4 h-4 text-terracotta-600 dark:text-terracotta-500" /> : <Maximize2 className="w-4 h-4" />}
            </button>

            {/* Clear Telemetry */}
            <button
              onClick={() => setConfirmAction({
                title: 'Purge telemetry database',
                message: 'Reset ALL captured in-memory and persistent telemetry traces and metrics? This cannot be undone.',
                confirmLabel: 'Purge everything',
                onConfirm: clearTelemetry,
              })}
              disabled={loading}
              className="p-2 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 text-rose-600 dark:text-rose-400 border border-rose-500/20 transition-colors disabled:opacity-50 flex items-center gap-1.5"
              title="Purge telemetry database"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>

        {error && (
          <div role="alert" className="p-4 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-800 dark:text-rose-300 text-sm flex items-start justify-between gap-4">
            <div>
              <div className="font-semibold flex items-center gap-2">
                <AlertCircle className="w-4 h-4" />
                Telemetry is unavailable
              </div>
              <p className="text-xs mt-1">{error}</p>
            </div>
            <button onClick={() => refreshMetrics()} className="text-xs font-semibold underline underline-offset-2 shrink-0">
              Retry
            </button>
          </div>
        )}

        {/* ── GLOBAL RAG HEALTH BAR (10 SUBSYSTEMS) ───────────────── */}
        <div className="p-4 rounded-2xl bg-white/80 dark:bg-sand-dark/80 backdrop-blur-xl border border-sand-border/80 dark:border-sand-darkBorder/80 shadow-soft dark:shadow-glassDark">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-charcoal-muted dark:text-cream-400 flex items-center gap-1.5">
              <Shield className="w-3.5 h-3.5 text-terracotta-600 dark:text-terracotta-500" />
              Global RAG Subsystem Health (10 Subsystems)
            </span>
            <span className="text-xs font-mono text-charcoal-muted dark:text-cream-500">
              Uptime: {health?.uptime_seconds ? `${Math.floor(health.uptime_seconds / 60)}m ${Math.floor(health.uptime_seconds % 60)}s` : 'Not reported'}
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-5 lg:grid-cols-10 gap-2 text-xs">
            {[
              { key: 'API Gateway', status: health.api, icon: Server },
              { key: 'Ollama Daemon', status: health.ollama, icon: Bot },
              { key: 'Chroma Vector DB', status: health.vector_db, icon: Database },
              { key: 'BM25 Inverted', status: health.bm25, icon: Search },
              { key: 'Embedding Model', status: health.embedding_model, icon: Layers },
              { key: `Text (${health.active_model_text || 'unknown'})`, status: health.text_model, icon: Brain },
              { key: `Vision (${health.active_model_vision || 'unknown'})`, status: health.vision_model, icon: Eye },
              { key: 'Semantic Cache', status: health.semantic_cache, icon: Zap },
              { key: 'Vision Cache', status: health.vision_cache, icon: ImageIcon },
              { key: 'Session Memory', status: health.memory, icon: History },
            ].map((sub) => {
              const IconComponent = sub.icon;
              const isOk = sub.status === 'healthy';
              return (
                <div
                  key={sub.key}
                  className={`p-2.5 rounded-xl border flex flex-col justify-between transition-all ${
                    isOk
                      ? 'bg-cream-100/80 border-sand-border/80 hover:border-emerald-500/40 dark:bg-cream-950/80 dark:border-sand-darkBorder/80'
                      : 'bg-rose-500/10 border-rose-500/30 hover:border-rose-500/60 dark:bg-rose-950/30'
                  }`}
                >
                  <div className="flex items-center justify-between gap-1 mb-1">
                    <IconComponent className={`w-3.5 h-3.5 ${isOk ? 'text-charcoal-muted dark:text-cream-400' : 'text-rose-600 dark:text-rose-400'}`} />
                    <span className={`w-1.5 h-1.5 rounded-full ${isOk ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`} />
                  </div>
                  <span className="font-semibold text-[11px] text-charcoal dark:text-cream-100 truncate" title={sub.key}>
                    {sub.key}
                  </span>
                  <span className={`text-[10px] uppercase font-bold mt-1 ${isOk ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}`}>
                    {sub.status}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* ── THRESHOLD ALERTS (IF ACTIVE) ────────────────────────── */}
        {alerts.some((a) => a.active) && (
          <div className="p-4 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-900 dark:text-amber-300 text-xs space-y-2 dark:bg-amber-950/30">
            <div className="flex items-center gap-2 font-bold text-amber-800 dark:text-amber-200">
              <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400" />
              Active Performance &amp; Operational Alerts
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-2 pt-1">
              {alerts
                .filter((a) => a.active)
                .map((alert) => (
                  <div key={alert.alert_id} className="p-2.5 rounded-lg bg-white/90 dark:bg-sand-dark/90 border border-amber-500/30">
                    <div className="flex justify-between font-semibold">
                      <span>{alert.rule_name}</span>
                      <span className="text-amber-700 dark:text-amber-400 font-mono">{alert.current_value}</span>
                    </div>
                    <p className="text-[11px] text-amber-800/80 dark:text-amber-300/80 mt-1">{alert.message}</p>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* ── TOP KPI METRIC CARDS ────────────────────────────────── */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {/* Total Queries */}
          <div className="p-4 rounded-2xl bg-white/80 dark:bg-sand-dark/80 backdrop-blur-xl border border-sand-border/80 dark:border-sand-darkBorder/80 shadow-soft dark:shadow-glassDark flex flex-col justify-between">
            <div className="flex items-center justify-between text-charcoal-muted dark:text-cream-400 text-xs">
              <span>Total Queries</span>
              <Search className="w-4 h-4 text-terracotta-600 dark:text-terracotta-500" />
            </div>
            <div className="mt-2">
              <span className="text-2xl font-bold font-mono text-charcoal dark:text-cream-100">
                {totalQueriesCount}
              </span>
              <span className="text-xs text-charcoal-muted dark:text-cream-500 block mt-0.5">
                {qm?.requests_per_minute ?? 0} req/min
              </span>
            </div>
          </div>

          {/* P95 Latency */}
          <div className="p-4 rounded-2xl bg-white/80 dark:bg-sand-dark/80 backdrop-blur-xl border border-sand-border/80 dark:border-sand-darkBorder/80 shadow-soft dark:shadow-glassDark flex flex-col justify-between">
            <div className="flex items-center justify-between text-charcoal-muted dark:text-cream-400 text-xs">
              <span>P95 Latency</span>
              <Clock className="w-4 h-4 text-amber-600 dark:text-amber-400" />
            </div>
            <div className="mt-2">
              <span className="text-2xl font-bold font-mono text-amber-600 dark:text-amber-400">
                {formatLatency(p95Lat)}
              </span>
              <span className="text-xs text-charcoal-muted dark:text-cream-500 block mt-0.5">
                P50: {formatLatency(qm?.p50_latency_ms ?? avgLat)}
              </span>
            </div>
          </div>

          {/* Avg TTFT */}
          <div className="p-4 rounded-2xl bg-white/80 dark:bg-sand-dark/80 backdrop-blur-xl border border-sand-border/80 dark:border-sand-darkBorder/80 shadow-soft dark:shadow-glassDark flex flex-col justify-between">
            <div className="flex items-center justify-between text-charcoal-muted dark:text-cream-400 text-xs">
              <span>Avg TTFT</span>
              <Zap className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
            </div>
            <div className="mt-2">
              <span className="text-2xl font-bold font-mono text-emerald-600 dark:text-emerald-400">
                {formatLatency(avgTtft)}
              </span>
              <span className="text-xs text-charcoal-muted dark:text-cream-500 block mt-0.5">
                First token latency
              </span>
            </div>
          </div>

          {/* Token Consumption */}
          <div className="p-4 rounded-2xl bg-white/80 dark:bg-sand-dark/80 backdrop-blur-xl border border-sand-border/80 dark:border-sand-darkBorder/80 shadow-soft dark:shadow-glassDark flex flex-col justify-between">
            <div className="flex items-center justify-between text-charcoal-muted dark:text-cream-400 text-xs">
              <span>Token Consumption</span>
              <Flame className="w-4 h-4 text-purple-600 dark:text-purple-400" />
            </div>
            <div className="mt-2">
              <span className="text-2xl font-bold font-mono text-purple-600 dark:text-purple-400">
                {formatTokens(totTokens)}
              </span>
              <span className="text-xs text-charcoal-muted dark:text-cream-500 block mt-0.5">
                Prompt: {formatTokens(tokens?.total_prompt_tokens ?? data.prompt_tokens)} • Compl: {formatTokens(tokens?.total_completion_tokens ?? data.completion_tokens)}
              </span>
            </div>
          </div>

          {/* Retrieval Hit Rate */}
          <div className="p-4 rounded-2xl bg-white/80 dark:bg-sand-dark/80 backdrop-blur-xl border border-sand-border/80 dark:border-sand-darkBorder/80 shadow-soft dark:shadow-glassDark flex flex-col justify-between">
            <div className="flex items-center justify-between text-charcoal-muted dark:text-cream-400 text-xs">
              <span>Retrieval Hit Rate</span>
              <Layers className="w-4 h-4 text-terracotta-600 dark:text-terracotta-500" />
            </div>
            <div className="mt-2">
              <span className="text-2xl font-bold font-mono text-terracotta-600 dark:text-terracotta-400">
                {formatPercent(rq?.retrieval_hit_rate ?? 0)}
              </span>
              <span className="text-xs text-charcoal-muted dark:text-cream-500 block mt-0.5">
                Avg {rq?.avg_candidate_count ?? 0} candidates
              </span>
            </div>
          </div>

          {/* Error Rate */}
          <div className="p-4 rounded-2xl bg-white/80 dark:bg-sand-dark/80 backdrop-blur-xl border border-sand-border/80 dark:border-sand-darkBorder/80 shadow-soft dark:shadow-glassDark flex flex-col justify-between">
            <div className="flex items-center justify-between text-charcoal-muted dark:text-cream-400 text-xs">
              <span>Error Rate</span>
              <ShieldAlert className="w-4 h-4 text-rose-600 dark:text-rose-400" />
            </div>
            <div className="mt-2">
              <span className="text-2xl font-bold font-mono text-charcoal dark:text-cream-100">
                {formatFloatPercent(qm?.error_rate ?? 0)}
              </span>
              <span className="text-xs text-charcoal-muted dark:text-cream-500 block mt-0.5">
                {recentIncidents.length} incidents logged
              </span>
            </div>
          </div>
        </div>

        {/* ── NAVIGATION TABS ────────────────────────────────────── */}
        <div className="flex border-b border-sand-border dark:border-sand-darkBorder gap-2 text-xs font-semibold overflow-x-auto pb-1 custom-scrollbar">
          {[
            { id: 'overview', label: 'Telemetry Overview', icon: BarChart3 },
            { id: 'queries', label: `Query Traces (${recentTraces.length})`, icon: Activity },
            { id: 'models', label: 'Models & Vision Inference', icon: Cpu },
            { id: 'caches', label: 'Multi-Tier Caches', icon: Zap },
            { id: 'ingestion', label: 'Document Ingestion', icon: FileText },
            { id: 'errors', label: `Error Incidents (${recentIncidents.length})`, icon: AlertCircle },
          ].map((tab) => {
            const TabIcon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-1.5 px-4 py-2.5 rounded-xl transition-all whitespace-nowrap ${
                  isActive
                    ? 'bg-terracotta-500/10 text-terracotta-700 dark:text-terracotta-400 border border-terracotta-500/30 font-semibold shadow-sm'
                    : 'text-charcoal-muted dark:text-cream-400 hover:text-charcoal dark:hover:text-cream-200 hover:bg-cream-100 dark:hover:bg-sand-dark'
                }`}
              >
                <TabIcon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* ── TAB CONTENT ────────────────────────────────────────── */}

        {/* 1. OVERVIEW TAB */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Latency Waterfall & Retrieval Quality Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:grid">
              {/* Waterfall Latency Card */}
              <div className="p-5 rounded-2xl bg-white/80 dark:bg-sand-dark/80 backdrop-blur-xl border border-sand-border/80 dark:border-sand-darkBorder/80 shadow-soft dark:shadow-glassDark">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-sm font-bold text-charcoal dark:text-cream-100 flex items-center gap-2 font-serif">
                    <Clock className="w-4 h-4 text-terracotta-600 dark:text-terracotta-500" />
                    16-Stage Waterfall Latency Breakdown
                  </h2>
                  <span className="text-xs font-mono text-terracotta-600 dark:text-terracotta-500 font-bold">
                    Avg Total: {formatLatency(lb?.total_latency_ms ?? avgLat)}
                  </span>
                </div>

                <div className="space-y-2 text-xs">
                  {[
                    { name: '1. Request Intake & Classification', ms: lb?.query_classification_ms },
                    { name: '2. Conversation Memory Resolution', ms: lb?.conversation_memory_ms },
                    { name: '3. Query Rewrite & Expansion', ms: lb?.query_rewrite_ms },
                    { name: '4. Dense Embedding', ms: lb?.embedding_ms },
                    { name: '5. BM25 Sparse Search', ms: lb?.bm25_ms },
                    { name: '6. Chroma Vector Search', ms: lb?.vector_search_ms },
                    { name: '7. Hybrid RRF Fusion', ms: lb?.hybrid_fusion_ms },
                    { name: '8. Neural Reranking', ms: lb?.reranking_ms },
                    { name: '9. Cross-Page Section Expansion', ms: lb?.section_expansion_ms },
                    { name: '10. Visual Detection', ms: lb?.visual_detection_ms },
                    { name: `11. Vision Extraction (${health.active_model_vision || 'unknown'})`, ms: lb?.vision_extraction_ms },
                    { name: '12. Context Assembly & Prompt Build', ms: lb?.context_build_ms },
                    { name: '13. Time-to-First-Token (TTFT)', ms: lb?.ttft_ms ?? avgTtft },
                    { name: `14. LLM Response Generation (${health.active_model_text || 'unknown'})`, ms: lb?.generation_ms },
                    { name: '15. SSE Token Streaming', ms: lb?.streaming_ms },
                  ].map((st, i) => {
                    const maxMs = (lb?.total_latency_ms || avgLat || 1000) * 0.8;
                    const stageMs = st.ms;
                    const hasMeasurement = stageMs !== null && stageMs !== undefined;
                    const pct = hasMeasurement ? Math.min(100, Math.max(2, (stageMs / maxMs) * 100)) : 0;
                    return (
                      <div key={i} className="flex items-center gap-3">
                        <span className="w-56 text-charcoal dark:text-cream-300 font-mono truncate">{st.name}</span>
                        <div className="flex-1 bg-cream-200/80 dark:bg-cream-950 h-2 rounded-full overflow-hidden border border-sand-border/70 dark:border-sand-darkBorder/80">
                          <div
                            className="h-full bg-gradient-to-r from-terracotta-600 to-amber-500 rounded-full"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className="w-16 text-right font-mono font-semibold text-charcoal-muted dark:text-cream-400">
                          {formatMeasuredLatency(stageMs)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Retrieval Quality & Grounding Card */}
              <div className="space-y-6">
                {/* Retrieval Quality Card */}
                <div className="p-5 rounded-2xl bg-white/80 dark:bg-sand-dark/80 backdrop-blur-xl border border-sand-border/80 dark:border-sand-darkBorder/80 shadow-soft dark:shadow-glassDark">
                  <div className="flex items-center justify-between mb-3">
                    <h2 className="text-sm font-bold text-charcoal dark:text-cream-100 flex items-center gap-2 font-serif">
                      <Layers className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                      Retrieval Quality &amp; Proxy Indicators
                    </h2>
                  </div>

                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div className="p-3 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border/70 dark:border-sand-darkBorder/70">
                      <span className="text-charcoal-muted dark:text-cream-400 block">Candidate Pool Avg</span>
                      <span className="text-lg font-bold font-mono text-purple-600 dark:text-purple-400 mt-1 block">
                        {rq?.avg_candidate_count?.toFixed(1) ?? '0.0'} chunks
                      </span>
                    </div>
                    <div className="p-3 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border/70 dark:border-sand-darkBorder/70">
                      <span className="text-charcoal-muted dark:text-cream-400 block">Final Context Chunks Avg</span>
                      <span className="text-lg font-bold font-mono text-terracotta-600 dark:text-terracotta-400 mt-1 block">
                        {rq?.avg_final_chunk_count?.toFixed(1) ?? '0.0'} chunks
                      </span>
                    </div>
                    <div className="p-3 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border/70 dark:border-sand-darkBorder/70">
                      <span className="text-charcoal-muted dark:text-cream-400 block">Top Rerank Score Avg</span>
                      <span className="text-lg font-bold font-mono text-emerald-600 dark:text-emerald-400 mt-1 block">
                        {formatFloatPercent(rq?.avg_rerank_score ?? data.rerank_avg ?? 0)}
                      </span>
                    </div>
                    <div className="p-3 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border/70 dark:border-sand-darkBorder/70">
                      <span className="text-charcoal-muted dark:text-cream-400 block">Evidence Sufficiency Rate</span>
                      <span className="text-lg font-bold font-mono text-amber-600 dark:text-amber-400 mt-1 block">
                        {formatFloatPercent(rq?.evidence_sufficiency_rate ?? 0)}
                      </span>
                    </div>
                  </div>

                  <div className="mt-3 p-3 rounded-xl bg-cream-100/60 dark:bg-cream-950/60 border border-sand-border/60 dark:border-sand-darkBorder/60 text-[11px] text-charcoal-muted dark:text-cream-400 flex items-start gap-2">
                    <HelpCircle className="w-4 h-4 text-terracotta-600 dark:text-terracotta-500 shrink-0 mt-0.5" />
                    <span>
                      <strong>Evaluation Mode Notice:</strong> Measurable retrieval proxies are displayed. True Precision@K / Recall@K require an offline evaluation dataset benchmark.
                    </span>
                  </div>
                </div>

                {/* Grounding & Faithfulness Card */}
                <div className="p-5 rounded-2xl bg-white/80 dark:bg-sand-dark/80 backdrop-blur-xl border border-sand-border/80 dark:border-sand-darkBorder/80 shadow-soft dark:shadow-glassDark">
                  <div className="flex items-center justify-between mb-3">
                    <h2 className="text-sm font-bold text-charcoal dark:text-cream-100 flex items-center gap-2 font-serif">
                      <ShieldCheck className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                      Grounding &amp; Evidence Faithfulness
                    </h2>
                    <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border border-emerald-500/20 font-semibold">
                      {gd?.grounding_status?.replace(/_/g, ' ').toUpperCase() || 'NOT MEASURED'}
                    </span>
                  </div>

                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div className="p-3 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border/70 dark:border-sand-darkBorder/70 text-center">
                      <span className="text-charcoal-muted dark:text-cream-400 block text-[11px]">Supported Claims</span>
                      <span className="text-base font-bold font-mono text-emerald-600 dark:text-emerald-400 mt-1 block">
                        {formatMeasuredPercent(gd?.supported_claims_pct === null || gd?.supported_claims_pct === undefined ? null : gd.supported_claims_pct / 100)}
                      </span>
                    </div>
                    <div className="p-3 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border/70 dark:border-sand-darkBorder/70 text-center">
                      <span className="text-charcoal-muted dark:text-cream-400 block text-[11px]">Unsupported Claims</span>
                      <span className="text-base font-bold font-mono text-rose-600 dark:text-rose-400 mt-1 block">
                        {formatMeasuredPercent(gd?.unsupported_claims_pct === null || gd?.unsupported_claims_pct === undefined ? null : gd.unsupported_claims_pct / 100)}
                      </span>
                    </div>
                    <div className="p-3 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border/70 dark:border-sand-darkBorder/70 text-center">
                      <span className="text-charcoal-muted dark:text-cream-400 block text-[11px]">Citation Coverage</span>
                      <span className="text-base font-bold font-mono text-terracotta-600 dark:text-terracotta-400 mt-1 block">
                        {formatMeasuredPercent(gd?.citation_coverage_pct === null || gd?.citation_coverage_pct === undefined ? null : gd.citation_coverage_pct / 100)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Quick Traces Table in Overview */}
            <div className="p-5 rounded-2xl bg-white/80 dark:bg-sand-dark/80 backdrop-blur-xl border border-sand-border/80 dark:border-sand-darkBorder/80 shadow-soft dark:shadow-glassDark">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-bold text-charcoal dark:text-cream-100 flex items-center gap-2 font-serif">
                  <Activity className="w-4 h-4 text-terracotta-600 dark:text-terracotta-500" />
                  Live Query Execution Traces ({recentTraces.length} traces)
                </h2>
                <button
                  onClick={() => setActiveTab('queries')}
                  className="text-xs text-terracotta-600 dark:text-terracotta-400 hover:text-terracotta-700 dark:hover:text-terracotta-300 font-semibold flex items-center gap-1"
                >
                  View all ({recentTraces.length}) <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>

              {renderTracesTable()}
            </div>
          </div>
        )}

        {/* 2. QUERY TRACES TAB */}
        {activeTab === 'queries' && (
          <div className="space-y-4">
            <div className="p-5 rounded-2xl bg-white/80 dark:bg-sand-dark/80 backdrop-blur-xl border border-sand-border/80 dark:border-sand-darkBorder/80 shadow-soft dark:shadow-glassDark">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4">
                <div>
                  <h2 className="text-base font-bold text-charcoal dark:text-cream-100 flex items-center gap-2 font-serif">
                    <Activity className="w-5 h-5 text-terracotta-600 dark:text-terracotta-500" />
                    Real Query Execution Traces ({recentTraces.length} traces)
                  </h2>
                  <p className="text-xs text-charcoal-muted dark:text-cream-400 mt-1">
                    Click any query trace to inspect expanded details, or use the maximize button to open the full slide-over drawer.
                  </p>
                </div>
                {recentTraces.length > 0 && (
                  <button
                    onClick={() => setConfirmAction({
                      title: 'Clear all traces',
                      message: 'Delete ALL captured query execution traces? This cannot be undone.',
                      confirmLabel: 'Clear all traces',
                      onConfirm: clearTelemetry,
                    })}
                    disabled={loading}
                    className="px-3 py-1.5 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 text-rose-600 dark:text-rose-400 border border-rose-500/20 text-xs font-semibold flex items-center gap-1.5 transition-colors disabled:opacity-50 shrink-0"
                    title="Delete all traces"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    Clear All Traces
                  </button>
                )}
              </div>

              {recentTraces.length === 0 ? (
                <div className="p-8 text-center text-charcoal-muted dark:text-cream-500 text-xs">
                  No query traces captured in the selected time range ({timeRange}). Send a chat query to record live telemetry!
                </div>
              ) : (
                renderTracesTable()
              )}
            </div>
          </div>
        )}

        {/* 3. MODELS & VISION INFERENCE TAB */}
        {activeTab === 'models' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 lg:grid">
              {/* Text Model Card (qwen2.5:7b) */}
              <div className="p-5 rounded-2xl bg-white/80 dark:bg-sand-dark/80 backdrop-blur-xl border border-sand-border/80 dark:border-sand-darkBorder/80 shadow-soft dark:shadow-glassDark space-y-4">
                <div className="flex items-center justify-between border-b border-sand-border dark:border-sand-darkBorder pb-3">
                  <div className="flex items-center gap-2.5">
                    <div className="p-2 rounded-xl bg-terracotta-500/10 text-terracotta-600 dark:text-terracotta-500 border border-terracotta-500/20">
                      <Brain className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="font-bold text-charcoal dark:text-cream-100 font-serif">Text Synthesis Model</h3>
                      <span className="font-mono text-xs text-terracotta-600 dark:text-terracotta-500 font-semibold">{models?.text_model.model_name || 'Unavailable'}</span>
                    </div>
                  </div>
                  {getStatusBadge(health.text_model)}
                </div>

                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="p-3 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border/70 dark:border-sand-darkBorder/70">
                    <span className="text-charcoal-muted dark:text-cream-400 block">Total Inferences</span>
                    <span className="text-lg font-bold font-mono text-charcoal dark:text-cream-100 mt-1 block">
                      {models?.text_model.requests_count ?? totalQueriesCount}
                    </span>
                  </div>
                  <div className="p-3 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border/70 dark:border-sand-darkBorder/70">
                    <span className="text-charcoal-muted dark:text-cream-400 block">P95 Inference Latency</span>
                    <span className="text-lg font-bold font-mono text-amber-600 dark:text-amber-400 mt-1 block">
                      {formatLatency(models?.text_model.p95_latency_ms ?? p95Lat)}
                    </span>
                  </div>
                  <div className="p-3 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border/70 dark:border-sand-darkBorder/70">
                    <span className="text-charcoal-muted dark:text-cream-400 block">Average TTFT</span>
                    <span className="text-lg font-bold font-mono text-emerald-600 dark:text-emerald-400 mt-1 block">
                      {formatLatency(models?.text_model.avg_ttft_ms ?? avgTtft)}
                    </span>
                  </div>
                  <div className="p-3 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border/70 dark:border-sand-darkBorder/70">
                    <span className="text-charcoal-muted dark:text-cream-400 block">Generation Throughput</span>
                    <span className="text-lg font-bold font-mono text-purple-600 dark:text-purple-400 mt-1 block">
                      {models?.text_model.avg_tokens_per_second !== null && models?.text_model.avg_tokens_per_second !== undefined
                        ? `${models.text_model.avg_tokens_per_second.toFixed(1)} t/s`
                        : 'Not measured'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Vision Model Card (Qwen3-VL-2B-Instruct) */}
              <div className="p-5 rounded-2xl bg-white/80 dark:bg-sand-dark/80 backdrop-blur-xl border border-sand-border/80 dark:border-sand-darkBorder/80 shadow-soft dark:shadow-glassDark space-y-4">
                <div className="flex items-center justify-between border-b border-sand-border dark:border-sand-darkBorder pb-3">
                  <div className="flex items-center gap-2.5">
                    <div className="p-2 rounded-xl bg-purple-500/10 text-purple-600 dark:text-purple-400 border border-purple-500/20">
                      <Eye className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="font-bold text-charcoal dark:text-cream-100 font-serif">Vision VLM Model</h3>
                      <span className="font-mono text-xs text-purple-600 dark:text-purple-400 font-semibold">{models?.vision_model.model_name || 'Unavailable'}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono px-2 py-0.5 rounded bg-cream-200 dark:bg-sand-darkBorder text-charcoal dark:text-cream-200 border border-sand-border dark:border-sand-darkBorder">
                      CB: {models?.vision_model.circuit_breaker_state || 'Not reported'}
                    </span>
                    {getStatusBadge(health.vision_model)}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="p-3 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border/70 dark:border-sand-darkBorder/70">
                    <span className="text-charcoal-muted dark:text-cream-400 block">Visual Pages Processed</span>
                    <span className="text-lg font-bold font-mono text-charcoal dark:text-cream-100 mt-1 block">
                      {models?.vision_model.visual_pages_detected ?? 0} pages
                    </span>
                    {Boolean(models?.vision_model.diagrams || models?.vision_model.code_screenshots || models?.vision_model.tables) && (
                      <span className="text-[10px] text-charcoal-muted dark:text-cream-500 font-mono mt-0.5 block">
                        {models?.vision_model.diagrams ?? 0} diag · {models?.vision_model.code_screenshots ?? 0} code · {models?.vision_model.tables ?? 0} tbl
                      </span>
                    )}
                  </div>
                  <div className="p-3 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border/70 dark:border-sand-darkBorder/70">
                    <span className="text-charcoal-muted dark:text-cream-400 block">Vision Cache Hit Rate</span>
                    <span className="text-lg font-bold font-mono text-emerald-600 dark:text-emerald-400 mt-1 block">
                      {formatMeasuredPercent(models?.vision_model.cache_hit_rate)}
                    </span>
                    <span className="text-[10px] text-charcoal-muted dark:text-cream-500 font-mono mt-0.5 block">
                      Persistent disk cache
                    </span>
                  </div>
                  <div className="p-3 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border/70 dark:border-sand-darkBorder/70">
                    <span className="text-charcoal-muted dark:text-cream-400 block">Vision Avg Latency</span>
                    <span className="text-lg font-bold font-mono text-purple-600 dark:text-purple-400 mt-1 block">
                      {formatMeasuredLatency(models?.vision_model.avg_latency_ms)}
                    </span>
                  </div>
                  <div className="p-3 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border/70 dark:border-sand-darkBorder/70">
                    <span className="text-charcoal-muted dark:text-cream-400 block">Timeout / Failure Rate</span>
                    <span className="text-lg font-bold font-mono text-charcoal dark:text-cream-100 mt-1 block">
                      {models?.vision_model.requests_count ? formatPercent(models.vision_model.failure_count / models.vision_model.requests_count) : '0%'}
                    </span>
                    <span className="text-[10px] text-charcoal-muted dark:text-cream-500 font-mono mt-0.5 block">
                      {models?.vision_model.failure_count ?? 0} / {models?.vision_model.requests_count ?? 0} failed
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 4. MULTI-TIER CACHES TAB */}
        {activeTab === 'caches' && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 lg:grid">
              {[
                { title: 'Semantic Response Cache', stats: caches?.semantic_cache, desc: 'Exact & cosine similarity lookup for instant answering' },
                { title: 'Embedding Cache', stats: caches?.embedding_cache, desc: 'Text embedding vector cache preventing redundant model calls' },
                { title: 'Vision Extraction Cache', stats: caches?.vision_cache, desc: 'Image hash cache storing OCR code, diagram, and table extractions' },
                { title: 'Negative Vision Cache', stats: caches?.negative_vision_cache, desc: 'Short-lived failure cache protecting against repetitive timeouts' },
                { title: 'Retrieval Candidates Cache', stats: caches?.retrieval_cache, desc: 'Cached RRF ranked candidates pool for identical sub-queries' },
              ].map((cache, i) => (
                <div key={i} className="p-5 rounded-2xl bg-white/80 dark:bg-sand-dark/80 backdrop-blur-xl border border-sand-border/80 dark:border-sand-darkBorder/80 shadow-soft dark:shadow-glassDark space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="font-bold text-sm text-charcoal dark:text-cream-100 font-serif">{cache.title}</h3>
                    <span className="px-2 py-0.5 rounded-full bg-cream-200 text-charcoal-muted dark:bg-sand-darkBorder dark:text-cream-400 border border-sand-border dark:border-sand-darkBorder text-xs font-mono font-semibold">
                      Hit Rate: {formatMeasuredPercent(cache.stats?.hit_rate)}
                    </span>
                  </div>
                  <p className="text-xs text-charcoal-muted dark:text-cream-400">{cache.desc}</p>
                  <div className="grid grid-cols-2 gap-2 text-xs pt-2">
                    <div className="p-2.5 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border/70 dark:border-sand-darkBorder/70">
                      <span className="text-charcoal-muted dark:text-cream-400 block">Hits / Misses</span>
                      <span className="font-mono text-charcoal dark:text-cream-100 mt-0.5 block font-semibold">
                        {cache.stats?.hits ?? 0} / {cache.stats?.misses ?? 0}
                      </span>
                    </div>
                    <div className="p-2.5 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border/70 dark:border-sand-darkBorder/70">
                      <span className="text-charcoal-muted dark:text-cream-400 block">Avg Hit Latency</span>
                      <span className="font-mono text-emerald-600 dark:text-emerald-400 mt-0.5 block font-semibold">
                        {formatMeasuredLatency(cache.stats?.avg_hit_latency_ms)}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 5. DOCUMENT INGESTION TAB */}
        {activeTab === 'ingestion' && (
          <div className="space-y-4">
            <div className="p-5 rounded-2xl bg-white/80 dark:bg-sand-dark/80 backdrop-blur-xl border border-sand-border/80 dark:border-sand-darkBorder/80 shadow-soft dark:shadow-glassDark">
              <h2 className="text-base font-bold text-charcoal dark:text-cream-100 mb-4 flex items-center gap-2 font-serif">
                <FileText className="w-5 h-5 text-terracotta-600 dark:text-terracotta-500" />
                Document Ingestion &amp; Indexing Pipeline
              </h2>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs mb-6 lg:grid">
                <div className="p-3 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border/70 dark:border-sand-darkBorder/70">
                  <span className="text-charcoal-muted dark:text-cream-400 block">Documents Processed</span>
                  <span className="text-xl font-bold font-mono text-charcoal dark:text-cream-100 mt-1 block">
                    {ing?.documents_processed ?? data.active_documents ?? 0} docs
                  </span>
                </div>
                <div className="p-3 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border/70 dark:border-sand-darkBorder/70">
                  <span className="text-charcoal-muted dark:text-cream-400 block">Indexed Chunks</span>
                  <span className="text-xl font-bold font-mono text-terracotta-600 dark:text-terracotta-400 mt-1 block">
                    {ing?.chunks_indexed ?? data.indexed_chunks ?? 0} chunks
                  </span>
                </div>
                <div className="p-3 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border/70 dark:border-sand-darkBorder/70">
                  <span className="text-charcoal-muted dark:text-cream-400 block">Visual Assets Indexed</span>
                  <span className="text-xl font-bold font-mono text-purple-600 dark:text-purple-400 mt-1 block">
                    {ing?.visual_assets_total ?? 0} assets
                  </span>
                </div>
                <div className="p-3 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border/70 dark:border-sand-darkBorder/70">
                  <span className="text-charcoal-muted dark:text-cream-400 block">Vector &amp; BM25 Ready</span>
                  <span className="text-xl font-bold font-mono text-emerald-600 dark:text-emerald-400 mt-1 block">
                    100% READY
                  </span>
                </div>
              </div>

              <h3 className="text-xs font-semibold text-charcoal-muted dark:text-cream-400 uppercase tracking-wider mb-2">
                Recent Document Ingestion Records
              </h3>
              {(!ing?.recent_ingestions || ing.recent_ingestions.length === 0) ? (
                <div className="p-4 rounded-xl bg-cream-100/60 dark:bg-cream-950/60 border border-sand-border/60 dark:border-sand-darkBorder/60 text-xs text-charcoal-muted dark:text-cream-500">
                  No recent document ingestions logged.
                </div>
              ) : (
                <div className="space-y-2 text-xs">
                  {ing.recent_ingestions.map((doc) => (
                    <div key={doc.document_id} className="p-3 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border/70 dark:border-sand-darkBorder/70 flex justify-between items-center">
                      <div>
                        <span className="font-semibold text-charcoal dark:text-cream-100">{doc.filename}</span>
                        <span className="text-charcoal-muted dark:text-cream-500 block text-[11px] font-mono mt-0.5">
                          {doc.pages_count} pages • {doc.chunks_count} chunks • {doc.category}
                        </span>
                      </div>
                      <div className="text-right">
                        <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border border-emerald-500/20 font-semibold font-mono">
                          {doc.status}
                        </span>
                        <span className="text-charcoal-muted dark:text-cream-500 block text-[11px] font-mono mt-0.5">
                          {formatLatency(doc.total_duration_ms)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* 6. ERROR INCIDENTS TAB */}
        {activeTab === 'errors' && (
          <div className="space-y-4">
            <div className="p-5 rounded-2xl bg-white/80 dark:bg-sand-dark/80 backdrop-blur-xl border border-sand-border/80 dark:border-sand-darkBorder/80 shadow-soft dark:shadow-glassDark">
              <h2 className="text-base font-bold text-charcoal dark:text-cream-100 mb-2 flex items-center gap-2 font-serif">
                <AlertCircle className="w-5 h-5 text-rose-600 dark:text-rose-400" />
                Error &amp; Incident Tracking Center ({recentIncidents.length})
              </h2>
              <p className="text-xs text-charcoal-muted dark:text-cream-400 mb-4">
                Detailed logs of vision model timeouts, API exceptions, and ingestion retries.
              </p>

              {recentIncidents.length === 0 ? (
                <div className="p-8 text-center text-charcoal-muted dark:text-cream-500 text-xs rounded-xl bg-cream-100/60 dark:bg-cream-950/60 border border-sand-border/60 dark:border-sand-darkBorder/60">
                  <CheckCircle2 className="w-8 h-8 text-emerald-600 dark:text-emerald-400 mx-auto mb-2" />
                  No errors or incidents were recorded in the selected window ({timeRange}).
                </div>
              ) : (
                <div className="space-y-3">
                  {recentIncidents.map((incident) => (
                    <div
                      key={incident.incident_id}
                      className="p-4 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border/70 dark:border-sand-darkBorder/70 space-y-2 text-xs"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className={`px-2 py-0.5 rounded-full font-mono font-bold uppercase text-[11px] ${
                            incident.severity === 'error' || incident.severity === 'critical'
                              ? 'bg-rose-500/15 text-rose-700 dark:text-rose-400 border border-rose-500/30'
                              : 'bg-amber-500/15 text-amber-700 dark:text-amber-400 border border-amber-500/30'
                          }`}>
                            {incident.severity}
                          </span>
                          <span className="font-semibold text-charcoal dark:text-cream-100">{incident.component}</span>
                          {incident.request_id && (
                            <span className="font-mono text-terracotta-600 dark:text-terracotta-500">{incident.request_id}</span>
                          )}
                        </div>
                        <span className="text-charcoal-muted dark:text-cream-500 font-mono">{new Date(incident.timestamp).toLocaleString()}</span>
                      </div>
                      <p className="text-charcoal dark:text-cream-200 font-mono text-[11px]">{incident.message}</p>
                      {incident.stack_trace && (
                        <pre className="p-2.5 rounded-xl bg-cream-50 dark:bg-cream-950 border border-sand-border dark:border-sand-darkBorder text-[11px] text-charcoal-muted dark:text-cream-400 font-mono overflow-x-auto custom-scrollbar">
                          {incident.stack_trace}
                        </pre>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── QUERY TRACE DRAWER (SLIDE OVER) ────────────────────── */}
        <QueryTraceDrawer
          trace={selectedTrace}
          onClose={() => setSelectedTrace(null)}
          onDelete={deleteTrace}
        />
      </div>

      {/* ── CONFIRMATION MODAL (replaces window.confirm) ─────────── */}
      {confirmAction && (
        <div
          className="fixed inset-0 z-[400] flex items-center justify-center p-4"
          role="dialog"
          aria-modal="true"
          onClick={() => !confirmBusy && setConfirmAction(null)}
        >
          <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" />
          <div
            className="relative w-full max-w-sm rounded-2xl bg-white dark:bg-sand-dark border border-sand-border dark:border-sand-darkBorder shadow-xl p-5"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start gap-3">
              <div className="p-2 rounded-xl bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-500/20 shrink-0">
                <AlertTriangle className="w-5 h-5" />
              </div>
              <div className="min-w-0">
                <h3 className="text-sm font-bold text-charcoal dark:text-cream-100">{confirmAction.title}</h3>
                <p className="text-xs text-charcoal-muted dark:text-cream-400 mt-1 leading-relaxed">{confirmAction.message}</p>
              </div>
            </div>
            <div className="flex items-center justify-end gap-2 mt-5">
              <button
                onClick={() => setConfirmAction(null)}
                disabled={confirmBusy}
                className="px-3.5 py-2 rounded-xl bg-cream-100 dark:bg-cream-950 text-charcoal dark:text-cream-200 border border-sand-border dark:border-sand-darkBorder text-xs font-semibold transition-colors hover:bg-cream-200 dark:hover:bg-[#2A2925] disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={runConfirm}
                disabled={confirmBusy}
                className="px-3.5 py-2 rounded-xl bg-rose-600 hover:bg-rose-700 text-white text-xs font-semibold transition-colors disabled:opacity-60 flex items-center gap-1.5"
              >
                <Trash2 className="w-3.5 h-3.5" />
                {confirmBusy ? 'Working…' : confirmAction.confirmLabel}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
