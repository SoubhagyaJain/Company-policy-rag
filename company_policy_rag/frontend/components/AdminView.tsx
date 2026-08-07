'use client';

import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity,
  Clock,
  Zap,
  Heart,
  Database,
  Cpu,
  RefreshCw,
  Search,
  ArrowRight,
  Layers,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  XCircle,
  AlertTriangle,
} from 'lucide-react';

import { LiquidGlassCard } from '@/components/LiquidGlassCard';
import { useObservability } from '@/hooks/useObservability';
import { formatLatency, formatDate } from '@/lib/utils';
import type { HealthStatus } from '@/lib/types';

/* ─── Helpers ──────────────────────────────────── */
function getHealthColor(status: HealthStatus['status']) {
  switch (status) {
    case 'ok':
      return 'text-emerald-600 dark:text-emerald-400';
    case 'degraded':
      return 'text-amber-600 dark:text-amber-400';
    case 'error':
      return 'text-rose-600 dark:text-rose-400';
  }
}

function getHealthBg(status: HealthStatus['status']) {
  switch (status) {
    case 'ok':
      return 'bg-emerald-500/10 border-emerald-500/20';
    case 'degraded':
      return 'bg-amber-500/10 border-amber-500/20';
    case 'error':
      return 'bg-rose-500/10 border-rose-500/20';
  }
}

function getHealthIcon(status: HealthStatus['status']) {
  switch (status) {
    case 'ok':
      return <CheckCircle2 className="w-4 h-4 text-emerald-500" />;
    case 'degraded':
      return <AlertTriangle className="w-4 h-4 text-amber-500" />;
    case 'error':
      return <XCircle className="w-4 h-4 text-rose-500" />;
  }
}

function ServiceDot({ active }: { active: boolean }) {
  return (
    <span
      className={`inline-block w-2 h-2 rounded-full ${
        active
          ? 'bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.5)]'
          : 'bg-rose-500 shadow-[0_0_6px_rgba(244,63,94,0.5)]'
      }`}
    />
  );
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

/* ─── Trace row variants ───────────────────────── */
const rowVariants = {
  hidden: { opacity: 0, y: 10 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.04, duration: 0.2 },
  }),
};

/* ─── Admin Observability View ─────────────────── */
export function AdminView() {
  const { data, health, loading, refreshMetrics } = useObservability();

  /* Auto-refresh every 10 seconds (additional to the hook's 30s) */
  useEffect(() => {
    const interval = setInterval(refreshMetrics, 10_000);
    return () => clearInterval(interval);
  }, [refreshMetrics]);

  const [expandedTrace, setExpandedTrace] = useState<string | null>(null);

  const toggleTrace = (id: string) => {
    setExpandedTrace((prev) => (prev === id ? null : id));
  };

  return (
    <div className="flex-1 h-[calc(100vh-57px)] overflow-y-auto p-4 sm:p-6 lg:p-8 custom-scrollbar bg-[#FAF9F5] dark:bg-[#141413]">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* ── Page heading ────────────────────────── */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <h1 className="font-serif font-bold text-2xl text-charcoal dark:text-cream-100">
              Observability & Telemetry
            </h1>
            <p className="text-sm text-charcoal-muted dark:text-cream-400 mt-0.5">
              Real-time RAG pipeline metrics, query traces, and system health.
            </p>
          </div>

          <button
            onClick={refreshMetrics}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-cream-100 dark:bg-sand-dark border border-sand-border dark:border-sand-darkBorder text-xs font-medium text-charcoal dark:text-cream-200 hover:bg-cream-200 dark:hover:bg-[#2A2925] transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh Now
          </button>
        </div>

        {/* ── Stats cards ─────────────────────────── */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <LiquidGlassCard className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-4 h-4 text-terracotta-600" />
              <span className="text-[11px] font-medium text-charcoal-muted dark:text-cream-400">
                Total Queries
              </span>
            </div>
            <p className="text-2xl font-bold font-mono text-charcoal dark:text-cream-100">
              {data.total_queries.toLocaleString()}
            </p>
          </LiquidGlassCard>

          <LiquidGlassCard className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <Clock className="w-4 h-4 text-amber-500" />
              <span className="text-[11px] font-medium text-charcoal-muted dark:text-cream-400">
                Avg Latency
              </span>
            </div>
            <p className="text-2xl font-bold font-mono text-charcoal dark:text-cream-100">
              {formatLatency(data.avg_latency_ms)}
            </p>
          </LiquidGlassCard>

          <LiquidGlassCard className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <Zap className="w-4 h-4 text-sky-500" />
              <span className="text-[11px] font-medium text-charcoal-muted dark:text-cream-400">
                Token Usage
              </span>
            </div>
            <p className="text-2xl font-bold font-mono text-charcoal dark:text-cream-100">
              {formatTokens(data.total_tokens)}
            </p>
            <p className="text-[10px] font-mono text-charcoal-muted dark:text-cream-500 mt-0.5">
              {formatTokens(data.prompt_tokens)} prompt · {formatTokens(data.completion_tokens)} completion
            </p>
          </LiquidGlassCard>

          <LiquidGlassCard className={`p-4 border ${getHealthBg(health.status)}`}>
            <div className="flex items-center gap-2 mb-2">
              <Heart className="w-4 h-4 text-rose-500" />
              <span className="text-[11px] font-medium text-charcoal-muted dark:text-cream-400">
                System Health
              </span>
            </div>
            <div className="flex items-center gap-2">
              {getHealthIcon(health.status)}
              <p className={`text-lg font-bold font-mono uppercase ${getHealthColor(health.status)}`}>
                {health.status}
              </p>
            </div>
            {health.backend_version && (
              <p className="text-[10px] font-mono text-charcoal-muted dark:text-cream-500 mt-0.5">
                {health.backend_version}
              </p>
            )}
          </LiquidGlassCard>
        </div>

        {/* ── Service health row ──────────────────── */}
        <LiquidGlassCard className="p-4">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-charcoal-muted dark:text-cream-400 mb-3">
            Service Dependencies
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="flex items-center gap-3 p-3 rounded-xl bg-cream-100/70 dark:bg-sand-dark/70">
              <Database className="w-4 h-4 text-terracotta-600" />
              <div className="flex-1">
                <p className="text-xs font-medium text-charcoal dark:text-cream-200">Vector DB</p>
                <p className="text-[10px] text-charcoal-muted dark:text-cream-500 font-mono">
                  Chroma / FAISS
                </p>
              </div>
              <ServiceDot active={health.vector_db} />
            </div>

            <div className="flex items-center gap-3 p-3 rounded-xl bg-cream-100/70 dark:bg-sand-dark/70">
              <Layers className="w-4 h-4 text-amber-500" />
              <div className="flex-1">
                <p className="text-xs font-medium text-charcoal dark:text-cream-200">Redis Cache</p>
                <p className="text-[10px] text-charcoal-muted dark:text-cream-500 font-mono">
                  Session Store
                </p>
              </div>
              <ServiceDot active={health.redis} />
            </div>

            <div className="flex items-center gap-3 p-3 rounded-xl bg-cream-100/70 dark:bg-sand-dark/70">
              <Cpu className="w-4 h-4 text-sky-500" />
              <div className="flex-1">
                <p className="text-xs font-medium text-charcoal dark:text-cream-200">ML Models</p>
                <p className="text-[10px] text-charcoal-muted dark:text-cream-500 font-mono">
                  BGE + Reranker
                </p>
              </div>
              <ServiceDot active={health.models_loaded} />
            </div>
          </div>
        </LiquidGlassCard>

        {/* ── Recent query traces table ────────────── */}
        <LiquidGlassCard className="p-0 overflow-hidden">
          <div className="px-4 py-3 border-b border-sand-border/60 dark:border-sand-darkBorder/60 flex items-center justify-between">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-charcoal-muted dark:text-cream-400 flex items-center gap-2">
              <Search className="w-3.5 h-3.5 text-terracotta-600" />
              Recent Query Traces
            </h3>
            <span className="text-[10px] font-mono text-charcoal-muted dark:text-cream-500">
              {data.recent_traces.length} traces
            </span>
          </div>

          {/* Desktop table header */}
          <div className="hidden lg:grid grid-cols-[1fr_1fr_80px_80px_80px_80px] gap-2 px-4 py-2 text-[10px] font-semibold uppercase tracking-wider text-charcoal-muted dark:text-cream-500 border-b border-sand-border/40 dark:border-sand-darkBorder/40 bg-cream-100/40 dark:bg-sand-dark/40">
            <span>Original Query</span>
            <span>Rewritten Query</span>
            <span className="text-center">Chunks</span>
            <span className="text-center">Rerank</span>
            <span className="text-center">Latency</span>
            <span className="text-center">Tokens</span>
          </div>

          {/* Rows */}
          <div className="divide-y divide-sand-border/40 dark:divide-sand-darkBorder/40">
            <AnimatePresence>
              {data.recent_traces.map((trace, i) => (
                <motion.div
                  key={trace.trace_id}
                  custom={i}
                  variants={rowVariants}
                  initial="hidden"
                  animate="visible"
                  className="group"
                >
                  {/* Desktop row */}
                  <div
                    className="hidden lg:grid grid-cols-[1fr_1fr_80px_80px_80px_80px] gap-2 px-4 py-3 items-center hover:bg-cream-100/60 dark:hover:bg-sand-dark/40 transition-colors cursor-pointer"
                    onClick={() => toggleTrace(trace.trace_id)}
                  >
                    <p className="text-xs text-charcoal dark:text-cream-100 truncate">
                      {trace.original_query}
                    </p>
                    <p className="text-xs text-charcoal-muted dark:text-cream-400 truncate">
                      {trace.query_rewritten ?? '—'}
                    </p>
                    <p className="text-xs font-mono text-center text-charcoal dark:text-cream-200">
                      {trace.total_chunks_retrieved}
                    </p>
                    <p className="text-xs font-mono text-center text-amber-600 dark:text-amber-400 font-semibold">
                      {(trace.top_rerank_score * 100).toFixed(0)}%
                    </p>
                    <p className="text-xs font-mono text-center text-charcoal dark:text-cream-200">
                      {formatLatency(trace.total_latency_ms)}
                    </p>
                    <p className="text-xs font-mono text-center text-charcoal-muted dark:text-cream-400">
                      {trace.prompt_tokens + trace.completion_tokens}
                    </p>
                  </div>

                  {/* Mobile card */}
                  <div
                    className="lg:hidden px-4 py-3 space-y-2 cursor-pointer hover:bg-cream-100/60 dark:hover:bg-sand-dark/40 transition-colors"
                    onClick={() => toggleTrace(trace.trace_id)}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <p className="text-xs font-medium text-charcoal dark:text-cream-100 line-clamp-2">
                        {trace.original_query}
                      </p>
                      {expandedTrace === trace.trace_id ? (
                        <ChevronUp className="w-3.5 h-3.5 text-charcoal-muted shrink-0" />
                      ) : (
                        <ChevronDown className="w-3.5 h-3.5 text-charcoal-muted shrink-0" />
                      )}
                    </div>
                    <div className="flex items-center gap-3 text-[10px] font-mono text-charcoal-muted dark:text-cream-400">
                      <span>{trace.total_chunks_retrieved} chunks</span>
                      <span className="text-amber-600 font-semibold">
                        {(trace.top_rerank_score * 100).toFixed(0)}%
                      </span>
                      <span>{formatLatency(trace.total_latency_ms)}</span>
                    </div>
                  </div>

                  {/* Expanded detail */}
                  <AnimatePresence>
                    {expandedTrace === trace.trace_id && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="overflow-hidden border-t border-sand-border/30 dark:border-sand-darkBorder/30 bg-cream-100/30 dark:bg-sand-dark/30"
                      >
                        <div className="px-4 py-3 grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                          <div>
                            <span className="text-[10px] font-semibold uppercase text-charcoal-muted dark:text-cream-500">
                              Rewritten Query
                            </span>
                            <p className="text-charcoal dark:text-cream-200 mt-0.5">
                              {trace.query_rewritten ?? 'N/A'}
                            </p>
                          </div>
                          <div>
                            <span className="text-[10px] font-semibold uppercase text-charcoal-muted dark:text-cream-500">
                              Expanded Queries
                            </span>
                            {trace.expanded_queries && trace.expanded_queries.length > 0 ? (
                              <ul className="mt-0.5 space-y-0.5">
                                {trace.expanded_queries.map((q, qi) => (
                                  <li key={qi} className="flex items-start gap-1 text-charcoal dark:text-cream-200">
                                    <ArrowRight className="w-3 h-3 text-terracotta-500 mt-0.5 shrink-0" />
                                    {q}
                                  </li>
                                ))}
                              </ul>
                            ) : (
                              <p className="text-charcoal-muted dark:text-cream-400 mt-0.5">None</p>
                            )}
                          </div>
                          <div>
                            <span className="text-[10px] font-semibold uppercase text-charcoal-muted dark:text-cream-500">
                              Rerank Latency
                            </span>
                            <p className="text-charcoal dark:text-cream-200 font-mono mt-0.5">
                              {formatLatency(trace.rerank_latency_ms)}
                            </p>
                          </div>
                          <div>
                            <span className="text-[10px] font-semibold uppercase text-charcoal-muted dark:text-cream-500">
                              Model
                            </span>
                            <p className="text-charcoal dark:text-cream-200 font-mono mt-0.5">
                              {trace.model}
                            </p>
                          </div>
                          <div>
                            <span className="text-[10px] font-semibold uppercase text-charcoal-muted dark:text-cream-500">
                              Tokens (Prompt / Completion)
                            </span>
                            <p className="text-charcoal dark:text-cream-200 font-mono mt-0.5">
                              {trace.prompt_tokens} / {trace.completion_tokens}
                            </p>
                          </div>
                          <div>
                            <span className="text-[10px] font-semibold uppercase text-charcoal-muted dark:text-cream-500">
                              Timestamp
                            </span>
                            <p className="text-charcoal dark:text-cream-200 font-mono mt-0.5">
                              {formatDate(trace.timestamp)}
                            </p>
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>

          {data.recent_traces.length === 0 && (
            <div className="py-12 text-center text-sm text-charcoal-muted dark:text-cream-400">
              No query traces recorded yet.
            </div>
          )}
        </LiquidGlassCard>
      </div>
    </div>
  );
}
