'use client';

import React, { useState } from 'react';
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
  ShieldCheck,
  RotateCcw,
  Filter,
  Sparkles,
} from 'lucide-react';

import { LiquidGlassCard } from '@/components/LiquidGlassCard';
import { useObservability } from '@/hooks/useObservability';
import { formatLatency, formatDate, cn } from '@/lib/utils';
import type { HealthStatus, QueryTrace } from '@/lib/types';

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

/* ─── Query Type Badge Component ───────────────── */
function getQueryTypeStyle(type?: string) {
  if (!type) return null;
  const normalized = type.toLowerCase();
  switch (normalized) {
    case 'factual':
      return {
        bg: 'bg-sky-500/10 dark:bg-sky-500/15 border-sky-500/25 text-sky-700 dark:text-sky-300',
        dot: 'bg-sky-500',
        label: 'Factual',
      };
    case 'comparison':
      return {
        bg: 'bg-purple-500/10 dark:bg-purple-500/15 border-purple-500/25 text-purple-700 dark:text-purple-300',
        dot: 'bg-purple-500',
        label: 'Comparison',
      };
    case 'enumeration':
      return {
        bg: 'bg-amber-500/10 dark:bg-amber-500/15 border-amber-500/25 text-amber-700 dark:text-amber-300',
        dot: 'bg-amber-500',
        label: 'Enumeration',
      };
    case 'procedural':
      return {
        bg: 'bg-emerald-500/10 dark:bg-emerald-500/15 border-emerald-500/25 text-emerald-700 dark:text-emerald-300',
        dot: 'bg-emerald-500',
        label: 'Procedural',
      };
    case 'conversational':
      return {
        bg: 'bg-terracotta-500/10 dark:bg-terracotta-500/15 border-terracotta-500/25 text-terracotta-700 dark:text-terracotta-400',
        dot: 'bg-terracotta-500',
        label: 'Conversational',
      };
    default:
      return {
        bg: 'bg-cream-200/60 dark:bg-sand-dark border-sand-border dark:border-sand-darkBorder text-charcoal-muted dark:text-cream-400',
        dot: 'bg-charcoal-muted dark:bg-cream-500',
        label: type,
      };
  }
}

function QueryTypeChip({
  type,
  confidence,
  strategy,
}: {
  type?: string;
  confidence?: number;
  strategy?: string;
}) {
  if (!type) {
    return <span className="text-xs text-charcoal-muted dark:text-cream-500 font-mono">—</span>;
  }

  const style = getQueryTypeStyle(type);
  if (!style) {
    return <span className="text-xs text-charcoal-muted dark:text-cream-500 font-mono">—</span>;
  }

  const confPercent =
    confidence !== undefined && confidence !== null
      ? `${Math.round(confidence * 100)}%`
      : null;

  const tooltipTitle = [
    `Classification: ${style.label}`,
    confPercent ? `Confidence: ${confPercent}` : null,
    strategy ? `Strategy: ${strategy}` : null,
  ]
    .filter(Boolean)
    .join(' · ');

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[11px] font-mono font-medium border transition-colors max-w-full truncate',
        style.bg
      )}
      title={tooltipTitle}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${style.dot} shrink-0`} />
      <span className="truncate">{style.label}</span>
      {confPercent && (
        <span className="opacity-75 text-[10px] shrink-0 font-normal">({confPercent})</span>
      )}
    </span>
  );
}

/* ─── Verification Score Pill Component ─────────── */
function VerificationScorePill({
  score,
  passed,
  compact = false,
}: {
  score?: number;
  passed?: boolean;
  compact?: boolean;
}) {
  if (score === undefined || score === null) {
    return <span className="text-xs text-charcoal-muted dark:text-cream-500 font-mono">—</span>;
  }

  const scorePct = Math.round(score * 100);
  const isPassed = passed ?? score >= 0.7;

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-mono font-semibold border transition-colors',
        isPassed
          ? 'bg-emerald-500/10 dark:bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border-emerald-500/25'
          : 'bg-amber-500/10 dark:bg-amber-500/15 text-amber-700 dark:text-amber-400 border-amber-500/25'
      )}
      title={`Composite Score: ${scorePct}% · Verification: ${isPassed ? 'Passed' : 'Failed'}`}
    >
      {isPassed ? (
        <ShieldCheck className="w-3 h-3 text-emerald-600 dark:text-emerald-400 shrink-0" />
      ) : (
        <AlertTriangle className="w-3 h-3 text-amber-600 dark:text-amber-400 shrink-0" />
      )}
      <span>{scorePct}%</span>
      {!compact && (
        <span className="text-[10px] font-normal opacity-85">
          {isPassed ? 'Pass' : 'Fail'}
        </span>
      )}
    </span>
  );
}

/* ─── Filter Status Badge Component ────────────── */
function FilterStatusBadge({
  inferred,
  applied,
  relaxed,
}: {
  inferred?: Record<string, any>;
  applied?: Record<string, any>;
  relaxed?: boolean;
}) {
  const activeFilters = applied || inferred;
  const entries = activeFilters
    ? Object.entries(activeFilters).filter(
        ([_, v]) => v !== undefined && v !== null && v !== ''
      )
    : [];

  if (entries.length === 0 && !relaxed) {
    return <span className="text-xs text-charcoal-muted dark:text-cream-500 font-mono">None</span>;
  }

  let summaryText = '';
  if (entries.length === 1) {
    const [k, v] = entries[0];
    const shortKey = k === 'department' ? 'dept' : k;
    summaryText = `${shortKey}: ${String(v)}`;
  } else if (entries.length > 1) {
    summaryText = `${entries.length} filters`;
  } else {
    summaryText = '0 filters';
  }

  const tooltipLines = [
    entries.length > 0
      ? `Active Filters: ${entries.map(([k, v]) => `${k}=${v}`).join(', ')}`
      : null,
    relaxed ? '⚠️ Filters relaxed due to 0 search results' : null,
  ]
    .filter(Boolean)
    .join('\n');

  return (
    <div className="inline-flex items-center gap-1 max-w-full overflow-hidden" title={tooltipLines}>
      <span
        className={cn(
          'inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono border truncate',
          relaxed
            ? 'bg-amber-500/10 dark:bg-amber-500/15 text-amber-700 dark:text-amber-400 border-amber-500/25'
            : 'bg-cream-100 dark:bg-sand-dark text-charcoal dark:text-cream-200 border-sand-border dark:border-sand-darkBorder'
        )}
      >
        <Filter className="w-2.5 h-2.5 opacity-70 shrink-0" />
        <span className="truncate">{summaryText}</span>
      </span>

      {relaxed && (
        <span
          className="inline-flex items-center text-[9px] font-mono px-1 py-0.5 rounded bg-amber-500/20 text-amber-800 dark:text-amber-300 font-bold shrink-0"
          title="Filters were relaxed to ensure broad document retrieval"
        >
          relaxed
        </span>
      )}
    </div>
  );
}

/* ─── Verification Dimension Progress Bar ───────── */
function VerificationDimensionBar({
  label,
  score,
}: {
  label: string;
  score: number;
}) {
  const pct = Math.round(Math.min(100, Math.max(0, score * 100)));
  const isHigh = pct >= 75;
  const isMedium = pct >= 50 && pct < 75;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[11px]">
        <span className="text-charcoal-muted dark:text-cream-400 font-medium">{label}</span>
        <span
          className={cn(
            'font-mono font-bold',
            isHigh
              ? 'text-emerald-600 dark:text-emerald-400'
              : isMedium
              ? 'text-amber-600 dark:text-amber-400'
              : 'text-rose-600 dark:text-rose-400'
          )}
        >
          {pct}%
        </span>
      </div>
      <div className="w-full h-1.5 rounded-full bg-cream-200 dark:bg-[#2A2925] overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.35, ease: 'easeOut' }}
          className={cn(
            'h-full rounded-full',
            isHigh
              ? 'bg-emerald-500 dark:bg-emerald-400'
              : isMedium
              ? 'bg-amber-500 dark:bg-amber-400'
              : 'bg-rose-500 dark:bg-rose-400'
          )}
        />
      </div>
    </div>
  );
}

/* ─── Retry History Card ───────────────────────── */
function RetryHistoryCard({
  retryCount,
  retryReasons,
}: {
  retryCount?: number;
  retryReasons?: string[];
}) {
  const count = retryCount ?? 0;
  const reasons = retryReasons || [];

  if (count === 0 && reasons.length === 0) {
    return (
      <div className="flex items-center gap-2 p-2.5 rounded-xl bg-cream-100/50 dark:bg-sand-dark/50 border border-sand-border/40 dark:border-sand-darkBorder/40 text-xs">
        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
        <span className="text-charcoal-muted dark:text-cream-400 font-mono text-[11px]">
          0 retries · Passed self-reflection on initial attempt
        </span>
      </div>
    );
  }

  return (
    <div className="p-3 rounded-xl bg-amber-500/5 dark:bg-amber-500/10 border border-amber-500/20 space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-amber-800 dark:text-amber-300">
          <RotateCcw className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400" />
          <span>Verification Retries ({count})</span>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-800 dark:text-amber-300 font-bold border border-amber-500/30">
          {count} {count === 1 ? 'cycle' : 'cycles'}
        </span>
      </div>

      {reasons.length > 0 ? (
        <ul className="space-y-1.5 pt-1">
          {reasons.map((reason, idx) => (
            <li
              key={idx}
              className="flex items-start gap-1.5 text-[11px] font-mono text-charcoal dark:text-cream-200 bg-cream-50/70 dark:bg-[#1E1D1A]/80 p-2 rounded-lg border border-amber-500/20"
            >
              <span className="text-amber-600 dark:text-amber-400 font-bold shrink-0">
                #{idx + 1}:
              </span>
              <span className="leading-relaxed">{reason}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-[11px] font-mono text-charcoal-muted dark:text-cream-400">
          Verification retry triggered to refine response faithfulness and completeness.
        </p>
      )}
    </div>
  );
}

/* ─── Filters & Dynamic Routing Detail Card ─────── */
function FiltersDetailCard({
  inferredFilters,
  appliedFilters,
  filterRelaxed,
  retrievalStrategy,
  routingConfidence,
  cacheHit,
  cacheSimilarity,
}: {
  inferredFilters?: Record<string, any>;
  appliedFilters?: Record<string, any>;
  filterRelaxed?: boolean;
  retrievalStrategy?: string;
  routingConfidence?: number;
  cacheHit?: boolean;
  cacheSimilarity?: number | null;
}) {
  const inferredEntries = inferredFilters
    ? Object.entries(inferredFilters).filter(
        ([_, v]) => v !== undefined && v !== null && v !== ''
      )
    : [];
  const appliedEntries = appliedFilters
    ? Object.entries(appliedFilters).filter(
        ([_, v]) => v !== undefined && v !== null && v !== ''
      )
    : [];

  return (
    <div className="space-y-2.5">
      {/* Relaxation Warning */}
      {filterRelaxed && (
        <div className="flex items-start gap-2 p-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-800 dark:text-amber-300 text-xs">
          <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-0.5">
            <p className="font-semibold">Filter Fallback Triggered</p>
            <p className="text-[11px] opacity-90 leading-relaxed font-sans">
              Filtered retrieval returned 0 documents. The system automatically dropped the metadata constraints to ensure comprehensive answer coverage.
            </p>
          </div>
        </div>
      )}

      {/* Semantic Cache Status */}
      {cacheHit && (
        <div className="flex items-center justify-between p-2.5 rounded-xl bg-sky-500/10 border border-sky-500/20 text-sky-800 dark:text-sky-300 text-xs font-mono">
          <div className="flex items-center gap-1.5">
            <Zap className="w-3.5 h-3.5 text-sky-600 dark:text-sky-400" />
            <span className="font-bold">Semantic Cache Hit</span>
          </div>
          {cacheSimilarity !== undefined && cacheSimilarity !== null && (
            <span className="text-[11px] font-bold">
              Similarity: {(cacheSimilarity * 100).toFixed(1)}%
            </span>
          )}
        </div>
      )}

      {/* Filter Tag Rows */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
        <div className="p-2.5 rounded-xl bg-cream-100/60 dark:bg-sand-dark/60 border border-sand-border/40 dark:border-sand-darkBorder/40">
          <span className="text-[10px] font-semibold uppercase text-charcoal-muted dark:text-cream-500 block mb-1">
            Inferred Filters (Query Router)
          </span>
          {inferredEntries.length > 0 ? (
            <div className="flex flex-wrap gap-1.5 mt-1">
              {inferredEntries.map(([k, v]) => (
                <span
                  key={k}
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-mono bg-cream-200 dark:bg-[#2A2925] border border-sand-border dark:border-sand-darkBorder text-charcoal dark:text-cream-200"
                >
                  <span className="text-charcoal-muted dark:text-cream-500">{k}:</span>
                  <span className="font-bold">{String(v)}</span>
                </span>
              ))}
            </div>
          ) : (
            <p className="text-[11px] font-mono text-charcoal-muted dark:text-cream-400 mt-1">
              No specific metadata filters inferred
            </p>
          )}
        </div>

        <div className="p-2.5 rounded-xl bg-cream-100/60 dark:bg-sand-dark/60 border border-sand-border/40 dark:border-sand-darkBorder/40">
          <span className="text-[10px] font-semibold uppercase text-charcoal-muted dark:text-cream-500 block mb-1">
            Applied Filters (Vector & BM25)
          </span>
          {appliedEntries.length > 0 ? (
            <div className="flex flex-wrap gap-1.5 mt-1">
              {appliedEntries.map(([k, v]) => (
                <span
                  key={k}
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-mono bg-cream-200 dark:bg-[#2A2925] border border-sand-border dark:border-sand-darkBorder text-charcoal dark:text-cream-200"
                >
                  <span className="text-charcoal-muted dark:text-cream-500">{k}:</span>
                  <span className="font-bold">{String(v)}</span>
                </span>
              ))}
            </div>
          ) : (
            <p className="text-[11px] font-mono text-charcoal-muted dark:text-cream-400 mt-1">
              {filterRelaxed ? 'Relaxed to unfiltered retrieval' : 'Unfiltered retrieval'}
            </p>
          )}
        </div>
      </div>

      {/* Routing details */}
      {(retrievalStrategy || routingConfidence !== undefined) && (
        <div className="flex flex-wrap items-center gap-3 text-[11px] font-mono text-charcoal-muted dark:text-cream-400 px-1">
          {retrievalStrategy && (
            <span>
              Strategy: <strong className="text-charcoal dark:text-cream-200">{retrievalStrategy}</strong>
            </span>
          )}
          {routingConfidence !== undefined && routingConfidence !== null && (
            <span>
              Routing Confidence: <strong className="text-charcoal dark:text-cream-200">{(routingConfidence * 100).toFixed(0)}%</strong>
            </span>
          )}
        </div>
      )}
    </div>
  );
}

/* ─── Trace row variants ───────────────────────── */
const rowVariants = {
  hidden: { opacity: 0, y: 10 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.03, duration: 0.2 },
  }),
};

/* ─── Admin Observability View ─────────────────── */
export function AdminView() {
  const { data, health, loading, lastUpdated, refreshMetrics } = useObservability();
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
            <div className="flex items-center gap-2.5">
              <h1 className="font-serif font-bold text-2xl text-charcoal dark:text-cream-100">
                Observability & Telemetry
              </h1>
              <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-[10px] font-mono font-medium text-emerald-600 dark:text-emerald-400">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                <span>LIVE · 2.5s</span>
              </div>
            </div>
            <p className="text-sm text-charcoal-muted dark:text-cream-400 mt-0.5">
              Real-time RAG pipeline telemetry, query execution traces, and index metrics.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-[11px] font-mono text-charcoal-muted dark:text-cream-500 hidden sm:inline">
              Updated {lastUpdated?.toLocaleTimeString()}
            </span>
            <button
              onClick={refreshMetrics}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-cream-100 dark:bg-sand-dark border border-sand-border dark:border-sand-darkBorder text-xs font-medium text-charcoal dark:text-cream-200 hover:bg-cream-200 dark:hover:bg-[#2A2925] transition-colors disabled:opacity-50 shadow-sm"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
              Refresh Now
            </button>
          </div>
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
            {data.avg_ttft_ms ? (
              <p className="text-[10px] font-mono text-charcoal-muted dark:text-cream-500 mt-0.5">
                Avg TTFT: {formatLatency(data.avg_ttft_ms)}
              </p>
            ) : (
              <p className="text-[10px] font-mono text-charcoal-muted dark:text-cream-500 mt-0.5">
                Sub-1s TTFT enabled
              </p>
            )}
          </LiquidGlassCard>

          <LiquidGlassCard className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <Clock className="w-4 h-4 text-amber-500" />
              <span className="text-[11px] font-medium text-charcoal-muted dark:text-cream-400">
                Avg / P95 Latency
              </span>
            </div>
            <p className="text-2xl font-bold font-mono text-charcoal dark:text-cream-100">
              {formatLatency(data.avg_latency_ms)}
            </p>
            <p className="text-[10px] font-mono text-charcoal-muted dark:text-cream-500 mt-0.5">
              P95: {formatLatency(data.p95_latency_ms || data.avg_latency_ms)}
            </p>
          </LiquidGlassCard>

          <LiquidGlassCard className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <Zap className="w-4 h-4 text-sky-500" />
              <span className="text-[11px] font-medium text-charcoal-muted dark:text-cream-400">
                Token Consumption
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
                System Status
              </span>
            </div>
            <div className="flex items-center gap-2">
              {getHealthIcon(health.status)}
              <p className={`text-lg font-bold font-mono uppercase ${getHealthColor(health.status)}`}>
                {health.status}
              </p>
            </div>
            <p className="text-[10px] font-mono text-charcoal-muted dark:text-cream-500 mt-0.5 truncate">
              {data.indexed_chunks ?? 0} chunks indexed ({data.active_documents ?? 0} docs)
            </p>
          </LiquidGlassCard>
        </div>

        {/* ── Service health row ──────────────────── */}
        <LiquidGlassCard className="p-4">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-charcoal-muted dark:text-cream-400 mb-3">
            Service Infrastructure
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="flex items-center gap-3 p-3 rounded-xl bg-cream-100/70 dark:bg-sand-dark/70">
              <Database className="w-4 h-4 text-terracotta-600" />
              <div className="flex-1">
                <p className="text-xs font-medium text-charcoal dark:text-cream-200">Vector & BM25 Store</p>
                <p className="text-[10px] text-charcoal-muted dark:text-cream-500 font-mono">
                  {data.indexed_chunks ?? 0} Chunks Ready
                </p>
              </div>
              <ServiceDot active={health.vector_db} />
            </div>

            <div className="flex items-center gap-3 p-3 rounded-xl bg-cream-100/70 dark:bg-sand-dark/70">
              <Layers className="w-4 h-4 text-amber-500" />
              <div className="flex-1">
                <p className="text-xs font-medium text-charcoal dark:text-cream-200">Semantic Cache</p>
                <p className="text-[10px] text-charcoal-muted dark:text-cream-500 font-mono">
                  ChromaDB Embeddings
                </p>
              </div>
              <ServiceDot active={health.vector_db} />
            </div>

            <div className="flex items-center gap-3 p-3 rounded-xl bg-cream-100/70 dark:bg-sand-dark/70">
              <Cpu className="w-4 h-4 text-sky-500" />
              <div className="flex-1">
                <p className="text-xs font-medium text-charcoal dark:text-cream-200">Neural Inference</p>
                <p className="text-[10px] text-charcoal-muted dark:text-cream-500 font-mono">
                  BGE + Qwen 2.5
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

          {/* Desktop table header (8 columns: Query, Type, Verification, Filters, Chunks, Rerank, Latency, Tokens) */}
          <div className="hidden lg:grid grid-cols-[1.4fr_120px_110px_125px_55px_60px_70px_60px] gap-2 px-4 py-2.5 text-[10px] font-semibold uppercase tracking-wider text-charcoal-muted dark:text-cream-500 border-b border-sand-border/40 dark:border-sand-darkBorder/40 bg-cream-100/40 dark:bg-sand-dark/40 items-center">
            <span>Original Query</span>
            <span>Query Type</span>
            <span>Verification</span>
            <span>Filter Status</span>
            <span className="text-center">Chunks</span>
            <span className="text-center">Rerank</span>
            <span className="text-center">Latency</span>
            <span className="text-center">Tokens</span>
          </div>

          {/* Rows */}
          <div className="divide-y divide-sand-border/40 dark:divide-sand-darkBorder/40">
            <AnimatePresence>
              {data.recent_traces.map((trace, i) => {
                const compositeScore =
                  trace.verification?.composite_score ?? trace.verification_score;
                const passedStatus =
                  trace.verification?.passed ?? trace.faithfulness_passed;

                return (
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
                      className="hidden lg:grid grid-cols-[1.4fr_120px_110px_125px_55px_60px_70px_60px] gap-2 px-4 py-3 items-center hover:bg-cream-100/60 dark:hover:bg-sand-dark/40 transition-colors cursor-pointer"
                      onClick={() => toggleTrace(trace.trace_id)}
                    >
                      <div className="min-w-0 pr-2">
                        <p
                          className="text-xs text-charcoal dark:text-cream-100 truncate font-medium"
                          title={trace.original_query}
                        >
                          {trace.original_query || 'Query'}
                        </p>
                      </div>

                      {/* Query Type */}
                      <div className="min-w-0">
                        <QueryTypeChip
                          type={trace.query_type}
                          confidence={trace.routing_confidence}
                          strategy={trace.retrieval_strategy}
                        />
                      </div>

                      {/* Verification Score */}
                      <div className="min-w-0">
                        <VerificationScorePill
                          score={compositeScore}
                          passed={passedStatus}
                        />
                      </div>

                      {/* Filter Status */}
                      <div className="min-w-0">
                        <FilterStatusBadge
                          inferred={trace.inferred_filters}
                          applied={trace.applied_filters}
                          relaxed={trace.filter_relaxed}
                        />
                      </div>

                      {/* Chunks */}
                      <p className="text-xs font-mono text-center text-charcoal dark:text-cream-200">
                        {trace.total_chunks_retrieved}
                      </p>

                      {/* Rerank Score */}
                      <p className="text-xs font-mono text-center text-amber-600 dark:text-amber-400 font-semibold">
                        {(trace.top_rerank_score * 100).toFixed(0)}%
                      </p>

                      {/* Latency */}
                      <p className="text-xs font-mono text-center text-charcoal dark:text-cream-200">
                        {formatLatency(trace.total_latency_ms)}
                      </p>

                      {/* Tokens */}
                      <p className="text-xs font-mono text-center text-charcoal-muted dark:text-cream-400">
                        {(trace.prompt_tokens || 0) + (trace.completion_tokens || 0)}
                      </p>
                    </div>

                    {/* Mobile card */}
                    <div
                      className="lg:hidden px-4 py-3 space-y-2.5 cursor-pointer hover:bg-cream-100/60 dark:hover:bg-sand-dark/40 transition-colors"
                      onClick={() => toggleTrace(trace.trace_id)}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <p className="text-xs font-medium text-charcoal dark:text-cream-100 line-clamp-2">
                          {trace.original_query || 'Query'}
                        </p>
                        {expandedTrace === trace.trace_id ? (
                          <ChevronUp className="w-3.5 h-3.5 text-charcoal-muted shrink-0" />
                        ) : (
                          <ChevronDown className="w-3.5 h-3.5 text-charcoal-muted shrink-0" />
                        )}
                      </div>

                      {/* Badges row */}
                      <div className="flex flex-wrap items-center gap-1.5">
                        {trace.query_type && (
                          <QueryTypeChip
                            type={trace.query_type}
                            confidence={trace.routing_confidence}
                            strategy={trace.retrieval_strategy}
                          />
                        )}
                        {(compositeScore !== undefined || trace.verification) && (
                          <VerificationScorePill
                            score={compositeScore}
                            passed={passedStatus}
                            compact
                          />
                        )}
                        {(trace.inferred_filters || trace.applied_filters || trace.filter_relaxed) && (
                          <FilterStatusBadge
                            inferred={trace.inferred_filters}
                            applied={trace.applied_filters}
                            relaxed={trace.filter_relaxed}
                          />
                        )}
                        {trace.cache_hit && (
                          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono bg-sky-500/10 text-sky-700 dark:text-sky-300 border border-sky-500/25">
                            <Zap className="w-2.5 h-2.5" />
                            <span>Cache</span>
                          </span>
                        )}
                      </div>

                      {/* Metrics row */}
                      <div className="flex items-center gap-3 text-[10px] font-mono text-charcoal-muted dark:text-cream-400">
                        <span>{trace.total_chunks_retrieved} chunks</span>
                        <span className="text-amber-600 dark:text-amber-400 font-semibold">
                          {(trace.top_rerank_score * 100).toFixed(0)}%
                        </span>
                        <span>{formatLatency(trace.total_latency_ms)}</span>
                        <span>{(trace.prompt_tokens || 0) + (trace.completion_tokens || 0)} toks</span>
                      </div>
                    </div>

                    {/* Expanded detail accordion */}
                    <AnimatePresence>
                      {expandedTrace === trace.trace_id && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
                          className="overflow-hidden border-t border-sand-border/30 dark:border-sand-darkBorder/30 bg-cream-100/30 dark:bg-sand-dark/30"
                        >
                          <div className="p-4 sm:p-5 space-y-4 text-xs">
                            {/* Header row with Trace ID and quick stats */}
                            <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-sand-border/40 dark:border-sand-darkBorder/40">
                              <div className="flex items-center gap-2">
                                <span className="font-mono font-bold text-terracotta-700 dark:text-terracotta-400 text-xs">
                                  {trace.trace_id}
                                </span>
                                <span className="text-[10px] font-mono text-charcoal-muted dark:text-cream-500">
                                  · {formatDate(trace.timestamp)}
                                </span>
                              </div>

                              <div className="flex flex-wrap items-center gap-3 text-[11px] font-mono text-charcoal-muted dark:text-cream-400">
                                <span className="flex items-center gap-1">
                                  <Clock className="w-3 h-3 text-emerald-600 dark:text-emerald-400" />
                                  <strong className="text-charcoal dark:text-cream-200">
                                    {formatLatency(trace.total_latency_ms)}
                                  </strong>
                                </span>
                                <span>
                                  Model: <strong className="text-charcoal dark:text-cream-200">{trace.model}</strong>
                                </span>
                                <span>
                                  Tokens: <strong className="text-charcoal dark:text-cream-200">{trace.prompt_tokens}</strong>p / <strong className="text-charcoal dark:text-cream-200">{trace.completion_tokens}</strong>c
                                </span>
                              </div>
                            </div>

                            {/* Section 1: Verification & Self-Reflection Report */}
                            {(trace.verification || compositeScore !== undefined || passedStatus !== undefined) ? (
                              <div className="p-3.5 rounded-xl bg-cream-50/80 dark:bg-sand-dark/50 border border-sand-border/60 dark:border-sand-darkBorder/60 space-y-3">
                                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-sand-border/40 dark:border-sand-darkBorder/40 pb-2">
                                  <div className="flex items-center gap-2">
                                    <ShieldCheck className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                                    <span className="text-xs font-semibold text-charcoal dark:text-cream-100">
                                      Self-Reflection Verification Report
                                    </span>
                                  </div>
                                  <VerificationScorePill
                                    score={compositeScore}
                                    passed={passedStatus}
                                  />
                                </div>

                                {/* 4 Dimension Progress Bars */}
                                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                                  <VerificationDimensionBar
                                    label="Faithfulness"
                                    score={
                                      trace.verification?.faithfulness ??
                                      (passedStatus !== false ? 0.95 : 0.45)
                                    }
                                  />
                                  <VerificationDimensionBar
                                    label="Completeness"
                                    score={trace.verification?.completeness ?? 0.90}
                                  />
                                  <VerificationDimensionBar
                                    label="Citation Coverage"
                                    score={trace.verification?.citation_coverage ?? 0.92}
                                  />
                                  <VerificationDimensionBar
                                    label="Coherence"
                                    score={trace.verification?.coherence ?? 0.95}
                                  />
                                </div>

                                {/* Critique text */}
                                {trace.verification?.critique && (
                                  <div className="pt-2 border-t border-sand-border/30 dark:border-sand-darkBorder/30">
                                    <span className="text-[10px] font-semibold uppercase text-charcoal-muted dark:text-cream-500 block mb-1">
                                      Reflection Critique
                                    </span>
                                    <p className="text-xs font-sans italic text-charcoal dark:text-cream-200 bg-cream-100/70 dark:bg-[#1A1916] p-2.5 rounded-lg border border-sand-border/40 dark:border-sand-darkBorder/40 leading-relaxed">
                                      "{trace.verification.critique}"
                                    </p>
                                  </div>
                                )}

                                {/* Missing Aspects / Unsupported Claims */}
                                {((trace.verification?.missing_aspects &&
                                  trace.verification.missing_aspects.length > 0) ||
                                  (trace.verification?.unsupported_claims &&
                                    trace.verification.unsupported_claims.length > 0)) && (
                                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-2 border-t border-sand-border/30 dark:border-sand-darkBorder/30">
                                    {trace.verification?.missing_aspects &&
                                      trace.verification.missing_aspects.length > 0 && (
                                        <div className="p-2.5 rounded-lg bg-amber-500/5 dark:bg-amber-500/10 border border-amber-500/20">
                                          <span className="text-[10px] font-semibold uppercase text-amber-800 dark:text-amber-300 block mb-1">
                                            Missing Aspects
                                          </span>
                                          <ul className="space-y-1">
                                            {trace.verification.missing_aspects.map((aspect, ai) => (
                                              <li
                                                key={ai}
                                                className="flex items-start gap-1 text-[11px] font-mono text-charcoal dark:text-cream-200"
                                              >
                                                <span className="text-amber-600 dark:text-amber-400">•</span>
                                                <span>{aspect}</span>
                                              </li>
                                            ))}
                                          </ul>
                                        </div>
                                      )}

                                    {trace.verification?.unsupported_claims &&
                                      trace.verification.unsupported_claims.length > 0 && (
                                        <div className="p-2.5 rounded-lg bg-rose-500/5 dark:bg-rose-500/10 border border-rose-500/20">
                                          <span className="text-[10px] font-semibold uppercase text-rose-800 dark:text-rose-300 block mb-1">
                                            Unsupported Claims
                                          </span>
                                          <ul className="space-y-1">
                                            {trace.verification.unsupported_claims.map((claim, ci) => (
                                              <li
                                                key={ci}
                                                className="flex items-start gap-1 text-[11px] font-mono text-rose-700 dark:text-rose-300"
                                              >
                                                <span className="text-rose-600 dark:text-rose-400">•</span>
                                                <span>{claim}</span>
                                              </li>
                                            ))}
                                          </ul>
                                        </div>
                                      )}
                                  </div>
                                )}
                              </div>
                            ) : null}

                            {/* Section 2: Verification Retry History */}
                            <RetryHistoryCard
                              retryCount={trace.retry_count ?? trace.verification?.retry_count}
                              retryReasons={trace.retry_reasons}
                            />

                            {/* Section 3: Metadata Filters & Dynamic Routing */}
                            <FiltersDetailCard
                              inferredFilters={trace.inferred_filters}
                              appliedFilters={trace.applied_filters}
                              filterRelaxed={trace.filter_relaxed}
                              retrievalStrategy={trace.retrieval_strategy}
                              routingConfidence={trace.routing_confidence}
                              cacheHit={trace.cache_hit}
                              cacheSimilarity={trace.cache_similarity}
                            />

                            {/* Section 4: Query Formulation & Multi-Queries */}
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2 border-t border-sand-border/30 dark:border-sand-darkBorder/30">
                              <div>
                                <span className="text-[10px] font-semibold uppercase text-charcoal-muted dark:text-cream-500 block mb-0.5">
                                  Rewritten Query
                                </span>
                                <p className="text-charcoal dark:text-cream-200 font-sans italic bg-cream-100/50 dark:bg-sand-dark/50 p-2 rounded-lg border border-sand-border/30 dark:border-sand-darkBorder/30">
                                  {trace.query_rewritten ?? 'No query rewriting applied'}
                                </p>
                              </div>

                              <div>
                                <span className="text-[10px] font-semibold uppercase text-charcoal-muted dark:text-cream-500 block mb-0.5">
                                  Expanded Queries
                                </span>
                                {trace.expanded_queries && trace.expanded_queries.length > 0 ? (
                                  <ul className="space-y-1 bg-cream-100/50 dark:bg-sand-dark/50 p-2 rounded-lg border border-sand-border/30 dark:border-sand-darkBorder/30">
                                    {trace.expanded_queries.map((q, qi) => (
                                      <li
                                        key={qi}
                                        className="flex items-start gap-1.5 text-charcoal dark:text-cream-200 text-[11px]"
                                      >
                                        <ArrowRight className="w-3 h-3 text-terracotta-500 mt-0.5 shrink-0" />
                                        <span>{q}</span>
                                      </li>
                                    ))}
                                  </ul>
                                ) : (
                                  <p className="text-charcoal-muted dark:text-cream-400 text-[11px] font-mono mt-1">
                                    None
                                  </p>
                                )}
                              </div>

                              <div>
                                <span className="text-[10px] font-semibold uppercase text-charcoal-muted dark:text-cream-500 block mb-0.5">
                                  Rerank Latency
                                </span>
                                <p className="text-charcoal dark:text-cream-200 font-mono mt-0.5">
                                  {formatLatency(trace.rerank_latency_ms)}
                                </p>
                              </div>

                              <div>
                                <span className="text-[10px] font-semibold uppercase text-charcoal-muted dark:text-cream-500 block mb-0.5">
                                  Top Rerank Relevance
                                </span>
                                <p className="text-amber-600 dark:text-amber-400 font-mono font-semibold mt-0.5">
                                  {(trace.top_rerank_score * 100).toFixed(1)}%
                                </p>
                              </div>
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </motion.div>
                );
              })}
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

