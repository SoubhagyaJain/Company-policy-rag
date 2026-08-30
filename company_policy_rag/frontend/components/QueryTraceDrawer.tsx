'use client';

import React, { useState } from 'react';
import {
  X,
  Clock,
  Zap,
  Layers,
  FileText,
  Code,
  Image as ImageIcon,
  Table as TableIcon,
  ShieldCheck,
  AlertTriangle,
  Cpu,
  CornerDownRight,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  CheckCircle2,
  XCircle,
  Trash2,
} from 'lucide-react';
import { QueryTrace } from '../lib/types';

interface QueryTraceDrawerProps {
  trace: QueryTrace | null;
  onClose: () => void;
  onDelete?: (traceId: string) => boolean | Promise<boolean>;
}

export const QueryTraceDrawer: React.FC<QueryTraceDrawerProps> = ({ trace, onClose, onDelete }) => {
  const [showRawJson, setShowRawJson] = useState(false);
  const [activeTab, setActiveTab] = useState<'waterfall' | 'evidence' | 'grounding' | 'json'>('waterfall');
  const [isDeleting, setIsDeleting] = useState(false);

  if (!trace) return null;

  const isConversational = Boolean(
    trace.conversational_bypass ||
    trace.retrieval_strategy === 'conversational_bypass' ||
    trace.query_type === 'conversational'
  );

  const stageTimings = trace.stage_timings || {};
  const maxStageDuration = Math.max(
    ...Object.values(stageTimings).filter((v) => typeof v === 'number' && v > 0),
    trace.total_latency_ms || 1.0
  );

  const formatMs = (ms?: number | null) => {
    if (ms === null || ms === undefined || isNaN(ms)) return 'N/A';
    return `${ms.toFixed(1)} ms`;
  };

  const getContentTypeIcon = (type?: string) => {
    switch (type?.toLowerCase()) {
      case 'code':
        return <Code className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />;
      case 'diagram':
        return <ImageIcon className="w-4 h-4 text-purple-600 dark:text-purple-400" />;
      case 'table':
        return <TableIcon className="w-4 h-4 text-amber-600 dark:text-amber-400" />;
      default:
        return <FileText className="w-4 h-4 text-terracotta-600 dark:text-terracotta-500" />;
    }
  };

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-charcoal/40 dark:bg-black/70 backdrop-blur-sm flex justify-end transition-opacity duration-200">
      {/* Click outside backdrop */}
      <div className="absolute inset-0" onClick={onClose} />

      {/* Drawer Container */}
      <div className="relative w-full max-w-3xl h-full bg-[#FAF9F5] dark:bg-[#181816] border-l border-sand-border dark:border-sand-darkBorder shadow-2xl flex flex-col z-10 overflow-hidden text-charcoal dark:text-cream-100 transition-all duration-300">
        {/* Header */}
        <div className="p-6 border-b border-sand-border dark:border-sand-darkBorder bg-[#FAF9F5]/95 dark:bg-[#181816]/95 backdrop-blur-xl sticky top-0 z-20 flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="px-2.5 py-0.5 text-xs font-mono rounded-full bg-terracotta-500/10 text-terracotta-700 dark:text-terracotta-400 border border-terracotta-500/20 font-semibold">
                {trace.request_id || trace.trace_id}
              </span>
              <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-cream-200 dark:bg-sand-dark text-charcoal dark:text-cream-200 border border-sand-border dark:border-sand-darkBorder">
                {trace.query_type || 'factual'}
              </span>
              {isConversational ? (
                <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-amber-500/15 text-amber-800 dark:text-amber-400 border border-amber-500/30">
                  Conversational Bypass
                </span>
              ) : (
                <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-emerald-500/15 text-emerald-800 dark:text-emerald-400 border border-emerald-500/30">
                  RAG Pipeline Active
                </span>
              )}
            </div>
            <h2 className="text-lg font-bold font-serif text-charcoal dark:text-cream-100 mt-2 line-clamp-2">
              "{trace.original_query}"
            </h2>
            {trace.resolved_query && trace.resolved_query !== trace.original_query && (
              <p className="text-xs text-charcoal-muted dark:text-cream-400 mt-1 flex items-center gap-1.5 font-mono">
                <CornerDownRight className="w-3.5 h-3.5 text-terracotta-600 dark:text-terracotta-400 shrink-0" />
                <span>Resolved: </span>
                <span className="text-terracotta-700 dark:text-terracotta-300 font-semibold">{trace.resolved_query}</span>
              </p>
            )}
          </div>

          <div className="flex items-center gap-1.5 shrink-0">
            {onDelete && (
              <button
                onClick={async () => {
                  if (window.confirm(`Delete trace "${trace.original_query.slice(0, 30)}..."?`)) {
                    setIsDeleting(true);
                    try {
                      const deleted = await onDelete(trace.trace_id);
                      if (deleted) onClose();
                    } finally {
                      setIsDeleting(false);
                    }
                  }
                }}
                disabled={isDeleting}
                className="p-2 text-rose-600 hover:text-rose-700 hover:bg-rose-500/10 dark:text-rose-400 dark:hover:bg-rose-500/20 rounded-xl transition-colors"
                title="Delete this query trace"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            )}
            <button
              onClick={onClose}
              className="p-2 text-charcoal-muted hover:text-charcoal dark:text-cream-400 dark:hover:text-cream-100 hover:bg-cream-200 dark:hover:bg-sand-dark rounded-xl transition-colors"
              title="Close drawer"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Quick KPI Strip */}
        <div className="grid grid-cols-4 gap-2 p-4 bg-cream-100/60 dark:bg-cream-950/60 border-b border-sand-border/80 dark:border-sand-darkBorder/80 text-xs">
          <div className="p-2.5 rounded-xl bg-white/90 dark:bg-sand-dark/90 border border-sand-border/70 dark:border-sand-darkBorder/70">
            <span className="text-charcoal-muted dark:text-cream-400 block">Total Latency</span>
            <span className="text-sm font-bold text-charcoal dark:text-cream-100 font-mono">
              {formatMs(trace.total_latency_ms)}
            </span>
          </div>
          <div className="p-2.5 rounded-xl bg-white/90 dark:bg-sand-dark/90 border border-sand-border/70 dark:border-sand-darkBorder/70">
            <span className="text-charcoal-muted dark:text-cream-400 block">TTFT</span>
            <span className="text-sm font-bold text-emerald-600 dark:text-emerald-400 font-mono">
              {formatMs(trace.ttft_ms)}
            </span>
          </div>
          <div className="p-2.5 rounded-xl bg-white/90 dark:bg-sand-dark/90 border border-sand-border/70 dark:border-sand-darkBorder/70">
            <span className="text-charcoal-muted dark:text-cream-400 block">Tokens (P / C)</span>
            <span className="text-sm font-bold text-terracotta-600 dark:text-terracotta-400 font-mono">
              {trace.prompt_tokens} / {trace.completion_tokens}
            </span>
          </div>
          <div className="p-2.5 rounded-xl bg-white/90 dark:bg-sand-dark/90 border border-sand-border/70 dark:border-sand-darkBorder/70">
            <span className="text-charcoal-muted dark:text-cream-400 block">Candidates</span>
            <span className="text-sm font-bold text-purple-600 dark:text-purple-400 font-mono">
              {trace.total_chunks_retrieved} chunks
            </span>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-sand-border dark:border-sand-darkBorder bg-[#FAF9F5] dark:bg-[#181816] px-6 gap-2 overflow-x-auto custom-scrollbar">
          <button
            onClick={() => setActiveTab('waterfall')}
            className={`py-3 px-3 text-xs font-semibold border-b-2 transition-all flex items-center gap-1.5 whitespace-nowrap ${
              activeTab === 'waterfall'
                ? 'border-terracotta-600 dark:border-terracotta-500 text-terracotta-700 dark:text-terracotta-400'
                : 'border-transparent text-charcoal-muted dark:text-cream-400 hover:text-charcoal dark:hover:text-cream-200'
            }`}
          >
            <Clock className="w-3.5 h-3.5" />
            Waterfall Latency (16 Stages)
          </button>
          <button
            onClick={() => setActiveTab('evidence')}
            className={`py-3 px-3 text-xs font-semibold border-b-2 transition-all flex items-center gap-1.5 whitespace-nowrap ${
              activeTab === 'evidence'
                ? 'border-terracotta-600 dark:border-terracotta-500 text-terracotta-700 dark:text-terracotta-400'
                : 'border-transparent text-charcoal-muted dark:text-cream-400 hover:text-charcoal dark:hover:text-cream-200'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            Evidence &amp; Chunks ({trace.total_chunks_retrieved})
          </button>
          <button
            onClick={() => setActiveTab('grounding')}
            className={`py-3 px-3 text-xs font-semibold border-b-2 transition-all flex items-center gap-1.5 whitespace-nowrap ${
              activeTab === 'grounding'
                ? 'border-terracotta-600 dark:border-terracotta-500 text-terracotta-700 dark:text-terracotta-400'
                : 'border-transparent text-charcoal-muted dark:text-cream-400 hover:text-charcoal dark:hover:text-cream-200'
            }`}
          >
            <ShieldCheck className="w-3.5 h-3.5" />
            Grounding &amp; Verification
          </button>
          <button
            onClick={() => setActiveTab('json')}
            className={`py-3 px-3 text-xs font-semibold border-b-2 transition-all flex items-center gap-1.5 whitespace-nowrap ${
              activeTab === 'json'
                ? 'border-terracotta-600 dark:border-terracotta-500 text-terracotta-700 dark:text-terracotta-400'
                : 'border-transparent text-charcoal-muted dark:text-cream-400 hover:text-charcoal dark:hover:text-cream-200'
            }`}
          >
            <Code className="w-3.5 h-3.5" />
            Raw Trace JSON
          </button>
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-6 space-y-6">
          {/* Conversational Bypass Notification Banner */}
          {isConversational && (
            <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-800 dark:text-amber-300 text-xs flex items-start gap-3">
              <Zap className="w-5 h-5 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-amber-900 dark:text-amber-200">Conversational Bypass Active</h4>
                <p className="mt-1 text-amber-800/90 dark:text-amber-300/90 leading-relaxed">
                  This query was identified as a conversational greeting or query not requiring document retrieval.
                  Vector search, BM25 retrieval, and context verification were safely bypassed (retrieval_required: false, evidence_required: false).
                </p>
              </div>
            </div>
          )}

          {/* TAB: Waterfall Latency */}
          {activeTab === 'waterfall' && (
            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-charcoal dark:text-cream-100 flex items-center gap-2 font-serif">
                <Clock className="w-4 h-4 text-terracotta-600 dark:text-terracotta-500" />
                Pipeline Stage Latency Breakdown
              </h3>

              {Object.keys(stageTimings).length === 0 ? (
                <div className="p-4 rounded-xl bg-cream-100/80 dark:bg-sand-dark/80 border border-sand-border dark:border-sand-darkBorder text-xs text-charcoal-muted dark:text-cream-400">
                  Direct synchronous execution ({formatMs(trace.total_latency_ms)}).
                </div>
              ) : (
                <div className="space-y-2.5">
                  {Object.entries(stageTimings).map(([stage, duration]) => {
                    const dur = typeof duration === 'number' ? duration : 0;
                    const pct = Math.min(100, Math.max(2, (dur / maxStageDuration) * 100));
                    return (
                      <div key={stage} className="p-3 rounded-xl bg-white dark:bg-sand-dark/80 border border-sand-border/80 dark:border-sand-darkBorder/80 text-xs shadow-sm">
                        <div className="flex justify-between items-center mb-1.5">
                          <span className="font-mono text-charcoal dark:text-cream-200 capitalize">
                            {stage.replace(/_/g, ' ')}
                          </span>
                          <span className="font-mono font-semibold text-terracotta-600 dark:text-terracotta-400">{formatMs(dur)}</span>
                        </div>
                        <div className="w-full bg-cream-200 dark:bg-cream-950 h-1.5 rounded-full overflow-hidden">
                          <div
                            className="bg-gradient-to-r from-terracotta-600 to-amber-500 h-full rounded-full transition-all duration-300"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Execution Summary Table */}
              <div className="mt-6 p-4 rounded-xl bg-white dark:bg-sand-dark/80 border border-sand-border/80 dark:border-sand-darkBorder/80 text-xs space-y-2 shadow-sm">
                <h4 className="font-semibold text-charcoal dark:text-cream-100 mb-2 font-serif">Execution Context</h4>
                <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-charcoal-muted dark:text-cream-400">
                  <div>Model: <span className="text-charcoal dark:text-cream-200 font-mono">{trace.model}</span></div>
                  <div>Strategy: <span className="text-charcoal dark:text-cream-200 font-mono">{trace.retrieval_strategy || 'balanced'}</span></div>
                  <div>Anchor Section: <span className="text-charcoal dark:text-cream-200 font-mono">{trace.anchor_section || 'N/A'}</span></div>
                  <div>Section Expansion: <span className="text-charcoal dark:text-cream-200 font-mono">{trace.section_expansion_used ? 'YES' : 'NO'}</span></div>
                  <div>Vision Extraction: <span className="text-charcoal dark:text-cream-200 font-mono">{trace.vision_used ? `YES (${trace.vision_model || 'Qwen3-VL-2B-Instruct'})` : 'NO'}</span></div>
                  <div>Cache Hit: <span className="text-charcoal dark:text-cream-200 font-mono">{trace.cache_hit ? 'YES' : 'NO'}</span></div>
                </div>
              </div>
            </div>
          )}

          {/* TAB: Evidence & Chunks */}
          {activeTab === 'evidence' && (
            <div className="space-y-4">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <h3 className="text-sm font-semibold text-charcoal dark:text-cream-100 flex items-center gap-2 font-serif">
                  <Layers className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                  Retrieved &amp; Reranked Evidence
                </h3>
                <div className="flex gap-2 text-xs flex-wrap">
                  <span className="px-2 py-0.5 rounded-full bg-terracotta-500/10 text-terracotta-700 dark:text-terracotta-400 border border-terracotta-500/20 font-mono">
                    Text: {trace.evidence_text_count ?? 0}
                  </span>
                  <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border border-emerald-500/20 font-mono">
                    Code: {trace.evidence_code_count ?? 0}
                  </span>
                  <span className="px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-700 dark:text-purple-400 border border-purple-500/20 font-mono">
                    Diagrams: {trace.evidence_diagram_count ?? 0}
                  </span>
                  <span className="px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-700 dark:text-amber-400 border border-amber-500/20 font-mono">
                    Tables: {trace.evidence_table_count ?? 0}
                  </span>
                </div>
              </div>

              {trace.sources_used && trace.sources_used.length > 0 && (
                <div className="p-3 rounded-xl bg-white dark:bg-sand-dark/80 border border-sand-border/80 dark:border-sand-darkBorder/80 text-xs shadow-sm">
                  <span className="text-charcoal-muted dark:text-cream-400 block mb-1 font-semibold">Sources Used:</span>
                  <div className="flex flex-wrap gap-1.5">
                    {trace.sources_used.map((s, idx) => (
                      <span key={idx} className="px-2.5 py-0.5 rounded-lg bg-cream-200 dark:bg-sand-darkBorder text-charcoal dark:text-cream-200 font-mono border border-sand-border/60 dark:border-sand-darkBorder/60">
                        {s}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {trace.safe_context_preview && (
                <div className="p-4 rounded-xl bg-white dark:bg-sand-dark/80 border border-sand-border/80 dark:border-sand-darkBorder/80 text-xs space-y-2 shadow-sm">
                  <span className="text-charcoal-muted dark:text-cream-400 font-semibold block font-serif">Synthesized Response Preview:</span>
                  <p className="text-charcoal dark:text-cream-200 leading-relaxed font-sans">{trace.safe_context_preview}</p>
                </div>
              )}
            </div>
          )}

          {/* TAB: Grounding & Verification */}
          {activeTab === 'grounding' && (
            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-charcoal dark:text-cream-100 flex items-center gap-2 font-serif">
                <ShieldCheck className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                Grounding &amp; Faithfulness Verification
              </h3>

              {isConversational ? (
                <div className="p-4 rounded-xl bg-cream-100/80 dark:bg-sand-dark/80 border border-sand-border dark:border-sand-darkBorder text-xs text-charcoal-muted dark:text-cream-400">
                  Conversational query — grounded against base model knowledge, evidence verification skipped.
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div className="p-3.5 rounded-xl bg-white dark:bg-sand-dark/80 border border-sand-border/80 dark:border-sand-darkBorder/80 shadow-sm">
                      <span className="text-charcoal-muted dark:text-cream-400 block">Faithfulness Status</span>
                      <div className="flex items-center gap-2 mt-1">
                        {trace.faithfulness_passed === undefined ? (
                          <span className="text-sm font-bold text-charcoal-muted dark:text-cream-400">NOT MEASURED</span>
                        ) : trace.faithfulness_passed ? (
                          <>
                            <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                            <span className="text-sm font-bold text-emerald-700 dark:text-emerald-400">PASSED</span>
                          </>
                        ) : (
                          <>
                            <XCircle className="w-4 h-4 text-rose-600 dark:text-rose-400" />
                            <span className="text-sm font-bold text-rose-700 dark:text-rose-400">FAILED</span>
                          </>
                        )}
                      </div>
                    </div>

                    <div className="p-3.5 rounded-xl bg-white dark:bg-sand-dark/80 border border-sand-border/80 dark:border-sand-darkBorder/80 shadow-sm">
                      <span className="text-charcoal-muted dark:text-cream-400 block">Verification Composite Score</span>
                      <span className="text-sm font-bold text-terracotta-600 dark:text-terracotta-400 font-mono mt-1 block">
                        {trace.verification_score !== undefined && trace.verification_score !== null
                          ? (trace.verification_score * 100).toFixed(1) + '%'
                          : 'Not measured'}
                      </span>
                    </div>
                  </div>

                  {trace.verification && (
                    <div className="p-4 rounded-xl bg-white dark:bg-sand-dark/80 border border-sand-border/80 dark:border-sand-darkBorder/80 text-xs space-y-2 shadow-sm">
                      <h4 className="font-semibold text-charcoal dark:text-cream-100 font-serif">Detailed Criteria Breakdown</h4>
                      <div className="grid grid-cols-2 gap-2 text-charcoal-muted dark:text-cream-400 pt-1">
                        <div>Faithfulness: <span className="text-charcoal dark:text-cream-200 font-mono">{trace.verification.faithfulness === undefined ? 'Not measured' : `${(trace.verification.faithfulness * 100).toFixed(0)}%`}</span></div>
                        <div>Completeness: <span className="text-charcoal dark:text-cream-200 font-mono">{trace.verification.completeness === undefined ? 'Not measured' : `${(trace.verification.completeness * 100).toFixed(0)}%`}</span></div>
                        <div>Citation Coverage: <span className="text-charcoal dark:text-cream-200 font-mono">{trace.verification.citation_coverage === undefined ? 'Not measured' : `${(trace.verification.citation_coverage * 100).toFixed(0)}%`}</span></div>
                        <div>Coherence: <span className="text-charcoal dark:text-cream-200 font-mono">{trace.verification.coherence === undefined ? 'Not measured' : `${(trace.verification.coherence * 100).toFixed(0)}%`}</span></div>
                      </div>

                      {trace.verification.critique && (
                        <div className="mt-3 p-3 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border dark:border-sand-darkBorder text-charcoal dark:text-cream-200">
                          <span className="text-amber-700 dark:text-amber-400 font-semibold block mb-1">Verifier Critique:</span>
                          {trace.verification.critique}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* TAB: Raw JSON */}
          {activeTab === 'json' && (
            <div className="space-y-2">
              <div className="flex justify-between items-center text-xs text-charcoal-muted dark:text-cream-400">
                <span>Canonical QueryTraceRecord Payload</span>
                <button
                  onClick={() => navigator.clipboard.writeText(JSON.stringify(trace, null, 2))}
                  className="px-3 py-1.5 bg-cream-200 hover:bg-cream-300 dark:bg-sand-dark dark:hover:bg-[#2A2925] text-charcoal dark:text-cream-200 rounded-lg transition-colors border border-sand-border dark:border-sand-darkBorder font-medium"
                >
                  Copy JSON
                </button>
              </div>
              <pre className="p-4 rounded-xl bg-cream-100/90 dark:bg-cream-950/90 border border-sand-border dark:border-sand-darkBorder text-xs font-mono text-charcoal dark:text-cream-200 overflow-x-auto max-h-[480px] custom-scrollbar">
                {JSON.stringify(trace, null, 2)}
              </pre>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-sand-border dark:border-sand-darkBorder bg-[#FAF9F5]/95 dark:bg-[#181816]/95 flex justify-between items-center text-xs text-charcoal-muted dark:text-cream-400">
          <span>Timestamp: {new Date(trace.timestamp).toLocaleString()}</span>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-cream-200 hover:bg-cream-300 dark:bg-sand-dark dark:hover:bg-[#2A2925] text-charcoal dark:text-cream-100 rounded-xl font-medium transition-colors border border-sand-border dark:border-sand-darkBorder"
          >
            Close Trace
          </button>
        </div>
      </div>
    </div>
  );
};
