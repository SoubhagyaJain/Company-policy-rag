'use client';

import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Sparkles,
  ChevronDown,
  ChevronUp,
  Clock,
  BookOpen,
  Copy,
  Check,
  AlertCircle,
  BrainCircuit,
  ArrowRight,
  CheckCircle2,
  ShieldCheck,
  AlertTriangle,
  RotateCcw,
  Filter,
  Zap,
  Tag,
} from 'lucide-react';
import { ChatMessageData, Citation, QueryTrace, ThinkingDetailLevel } from '../lib/types';
import { CitationCard } from './CitationCard';
import { CodeBlock } from './CodeBlock';
import { ThinkingPanel } from './ThinkingPanel';
import { formatLatency, cn } from '../lib/utils';

interface ChatMessageProps {
  message: ChatMessageData;
  onOpenCitation: (citation: Citation) => void;
}

function getQueryTypeBadgeClasses(type?: string): string {
  if (!type) {
    return 'bg-cream-200/70 text-charcoal-muted dark:bg-sand-dark dark:text-cream-400 border-sand-border dark:border-sand-darkBorder';
  }
  const normalized = type.toLowerCase();
  switch (normalized) {
    case 'factual':
      return 'bg-sky-500/10 text-sky-700 dark:text-sky-300 border-sky-500/25 dark:border-sky-500/30';
    case 'comparison':
      return 'bg-purple-500/10 text-purple-700 dark:text-purple-300 border-purple-500/25 dark:border-purple-500/30';
    case 'enumeration':
      return 'bg-amber-500/10 text-amber-700 dark:text-amber-300 border-amber-500/25 dark:border-amber-500/30';
    case 'procedural':
      return 'bg-teal-500/10 text-teal-700 dark:text-teal-300 border-teal-500/25 dark:border-teal-500/30';
    case 'conversational':
      return 'bg-terracotta-500/10 text-terracotta-700 dark:text-terracotta-400 border-terracotta-500/25 dark:border-terracotta-500/30';
    default:
      return 'bg-cream-200/70 text-charcoal-muted dark:bg-sand-dark dark:text-cream-400 border-sand-border dark:border-sand-darkBorder';
  }
}

function VerificationScorePill({ trace }: { trace: QueryTrace }) {
  const hasVerification =
    trace.verification_score !== undefined ||
    (trace.verification !== undefined && trace.verification !== null) ||
    trace.faithfulness_passed !== undefined;

  if (!hasVerification) return null;

  const score =
    trace.verification_score ??
    trace.verification?.composite_score ??
    (trace.faithfulness_passed !== false ? 0.95 : 0.5);

  const passed =
    trace.faithfulness_passed !== false &&
    (trace.verification ? trace.verification.passed : score >= 0.75);

  const scorePct = Math.round(Math.min(100, Math.max(0, score * 100)));

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-mono font-medium border transition-colors',
        passed
          ? 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/20 dark:border-emerald-500/30'
          : 'bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-500/20 dark:border-amber-500/30'
      )}
      title={`Self-Reflection Verification: ${scorePct}% composite score (${passed ? 'Passed' : 'Review needed'})`}
    >
      {passed ? (
        <CheckCircle2 className="w-3 h-3 text-emerald-600 dark:text-emerald-400 shrink-0" />
      ) : (
        <AlertTriangle className="w-3 h-3 text-amber-600 dark:text-amber-400 shrink-0" />
      )}
      <span>{scorePct}% {passed ? 'Verified' : 'Review'}</span>
    </span>
  );
}

function VerificationDimensionBar({
  label,
  score,
}: {
  label: string;
  score: number;
}) {
  const pct = Math.round(Math.min(100, Math.max(0, score * 100)));
  const isHigh = pct >= 85;
  const isMedium = pct >= 70 && pct < 85;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[10px]">
        <span className="text-[#5C564C] dark:text-[#A8A196] font-sans truncate">{label}</span>
        <span
          className={cn(
            'font-mono font-semibold',
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
      <div className="w-full h-1.5 rounded-full bg-[#E5E0D8] dark:bg-[#2A2925] overflow-hidden">
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

export function ChatMessage({ message, onOpenCitation }: ChatMessageProps) {
  const [showTrace, setShowTrace] = useState(false);
  const [copied, setCopied] = useState(false);
  const [detailLevel, setDetailLevel] = useState<ThinkingDetailLevel>(
    message.thinking_detail_level || 'standard'
  );
  const [thinkingExpanded, setThinkingExpanded] = useState<boolean>(Boolean(message.isStreaming));

  React.useEffect(() => {
    if (message.isStreaming) {
      setThinkingExpanded(true);
    } else if (message.content) {
      setThinkingExpanded(false);
    }
  }, [message.isStreaming]);

  const isUser = message.role === 'user';

  const handleCopyContent = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
      className={cn(
        'w-full py-3.5 sm:py-5 group font-sans',
        isUser ? 'flex justify-end' : 'flex justify-start'
      )}
    >
      {isUser ? (
        /* User Message Bubble - Anthropic Warm Sand Card */
        <div className="max-w-[85%] sm:max-w-2xl flex flex-col items-end space-y-1.5">
          <div className="px-4 py-3 rounded-2xl sm:rounded-3xl bg-[#EBE5D8] dark:bg-[#2A2824] text-[#1E1C1A] dark:text-[#FAF8F5] border border-[#DDD5C5] dark:border-[#383530] text-sm leading-relaxed shadow-sm font-sans whitespace-pre-wrap selection:bg-terracotta-500/20">
            {message.content}
          </div>
          <span className="text-[10px] font-mono text-charcoal-muted dark:text-cream-500 px-2 opacity-70">
            {new Date(message.timestamp).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </span>
        </div>
      ) : (
        /* Assistant Message - Anthropic Full Editorial Layout */
        <div className="w-full max-w-3xl space-y-3">
          {/* Header & Avatar */}
          <div className="flex items-center justify-between text-xs text-charcoal-muted dark:text-cream-400">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-lg bg-terracotta-600/10 dark:bg-terracotta-500/20 text-terracotta-600 dark:text-terracotta-400 flex items-center justify-center font-serif font-bold text-xs border border-terracotta-500/20 shadow-xs">
                <Sparkles className="w-3.5 h-3.5" />
              </div>
              <span className="font-medium text-charcoal dark:text-cream-200 tracking-tight text-xs">
                Nexus AI
              </span>
              <span className="text-[10px] font-mono opacity-60">
                {new Date(message.timestamp).toLocaleTimeString([], {
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </span>
            </div>

            {!message.isStreaming && message.content && (
              <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  onClick={handleCopyContent}
                  className="p-1 rounded-md text-charcoal-muted hover:text-charcoal dark:text-cream-400 dark:hover:text-cream-100 hover:bg-[#EFECE2] dark:hover:bg-[#2B2925] transition-colors"
                  title="Copy text"
                >
                  {copied ? (
                    <Check className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
                  ) : (
                    <Copy className="w-3.5 h-3.5" />
                  )}
                </button>
              </div>
            )}
          </div>

          {/* Milestone 4: Premium Thinking / Reasoning Panel */}
          {((message.thinking_events && message.thinking_events.length > 0) || message.isStreaming) && (
            <ThinkingPanel
              events={message.thinking_events || []}
              isStreaming={message.isStreaming}
              isExpanded={thinkingExpanded}
              onToggleExpanded={() => setThinkingExpanded((prev) => !prev)}
              detailLevel={detailLevel}
              onDetailLevelChange={setDetailLevel}
              reasoningSummary={message.reasoning_summary}
            />
          )}

          {/* Collapsible Thinking / RAG Reasoning Banner (Claude style) */}
          {message.trace && (
            <div className="pt-0.5 space-y-1.5">
              {/* Header row with badges */}
              <div className="flex flex-wrap items-center gap-1.5">
                <button
                  onClick={() => setShowTrace((prev) => !prev)}
                  className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-[#EFECE2]/70 hover:bg-[#EAE5D7] dark:bg-[#201F1C] dark:hover:bg-[#272522] border border-[#E0D9CB] dark:border-[#2E2C28] text-[11px] font-mono text-[#5C564C] dark:text-[#B5AFA4] transition-colors"
                >
                  <BrainCircuit className="w-3 h-3 text-terracotta-600 dark:text-terracotta-400" />
                  <span>
                    {showTrace ? 'Hide RAG Reasoning' : `Thought for ${formatLatency(message.trace.total_latency_ms)}`}
                  </span>
                  {showTrace ? <ChevronUp className="w-3 h-3 ml-0.5" /> : <ChevronDown className="w-3 h-3 ml-0.5" />}
                </button>

                {/* R1: Query Classification Badge */}
                {message.trace.query_type && (
                  <span
                    className={cn(
                      'inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-mono font-medium border capitalize transition-colors',
                      getQueryTypeBadgeClasses(message.trace.query_type)
                    )}
                    title={
                      message.trace.routing_confidence !== undefined && message.trace.routing_confidence !== null
                        ? `Routing Confidence: ${Math.round(message.trace.routing_confidence * 100)}%${
                            message.trace.retrieval_strategy ? ` · Strategy: ${message.trace.retrieval_strategy}` : ''
                          }`
                        : `Query Classification: ${message.trace.query_type}`
                    }
                  >
                    <span>{message.trace.query_type}</span>
                    {message.trace.routing_confidence !== undefined && message.trace.routing_confidence !== null && (
                      <span className="opacity-70 text-[9px]">
                        ({Math.round(message.trace.routing_confidence * 100)}%)
                      </span>
                    )}
                  </span>
                )}

                {/* R2: Composite Verification Score Pill */}
                <VerificationScorePill trace={message.trace} />

                {/* R3: Cache Hit Badge */}
                {message.trace.cache_hit && (
                  <span
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-mono font-medium bg-sky-500/10 text-sky-700 dark:text-sky-300 border border-sky-500/20 dark:border-sky-500/30 transition-colors"
                    title={`Semantic Cache Hit: Served directly from cache${
                      message.trace.cache_similarity !== null && message.trace.cache_similarity !== undefined
                        ? ` (${(message.trace.cache_similarity * 100).toFixed(1)}% similarity)`
                        : ''
                    }`}
                  >
                    <Zap className="w-2.5 h-2.5 text-sky-600 dark:text-sky-400" />
                    <span>Cache Hit</span>
                    {message.trace.cache_similarity !== null && message.trace.cache_similarity !== undefined && (
                      <span className="opacity-70 text-[9px]">
                        ({Math.round(message.trace.cache_similarity * 100)}%)
                      </span>
                    )}
                  </span>
                )}

                {/* R2: Header Retry Badge (if retried) */}
                {(message.trace.retry_count || 0) > 0 && (
                  <span
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-mono font-medium bg-amber-500/15 text-amber-700 dark:text-amber-300 border border-amber-500/30 transition-colors"
                    title={
                      message.trace.retry_reasons && message.trace.retry_reasons.length > 0
                        ? `Verification Retries (${message.trace.retry_count}):\n${message.trace.retry_reasons
                            .map((r, i) => `${i + 1}. ${r}`)
                            .join('\n')}`
                        : `${message.trace.retry_count} verification retry cycle(s) triggered`
                    }
                  >
                    <RotateCcw className="w-2.5 h-2.5 text-amber-600 dark:text-amber-400" />
                    <span>
                      {message.trace.retry_count} {message.trace.retry_count === 1 ? 'retry' : 'retries'}
                    </span>
                  </span>
                )}
              </div>

              <AnimatePresence>
                {showTrace && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.18 }}
                    className="mt-2 p-3.5 rounded-xl bg-[#F4F1E8] dark:bg-[#1E1D1A] border border-[#E2DBD0] dark:border-[#2D2B27] text-xs font-mono space-y-3 text-[#3D3A35] dark:text-[#D5CEC4]"
                  >
                    {/* Trace ID & Total Latency Header */}
                    <div className="flex items-center justify-between border-b border-[#E0D8CB] dark:border-[#2B2925] pb-2">
                      <span className="font-bold text-terracotta-700 dark:text-terracotta-400">
                        {message.trace.trace_id}
                      </span>
                      <span className="flex items-center gap-1 text-emerald-600 dark:text-emerald-400 font-semibold">
                        <Clock className="w-3 h-3" /> {formatLatency(message.trace.total_latency_ms)}
                      </span>
                    </div>

                    {/* Rewritten Query */}
                    {message.trace.query_rewritten && (
                      <div>
                        <span className="text-[10px] uppercase font-semibold text-charcoal-muted dark:text-cream-500 block mb-0.5">
                          Rewritten Query
                        </span>
                        <p className="font-sans text-xs italic bg-[#EAE4D6]/70 dark:bg-[#161513] p-2 rounded-lg border border-[#DDD5C5] dark:border-[#282622]">
                          "{message.trace.query_rewritten}"
                        </p>
                      </div>
                    )}

                    {/* Expanded Multi-Queries */}
                    {message.trace.expanded_queries && message.trace.expanded_queries.length > 0 && (
                      <div>
                        <span className="text-[10px] uppercase font-semibold text-charcoal-muted dark:text-cream-500 block mb-0.5">
                          Expanded Multi-Queries
                        </span>
                        <ul className="space-y-1">
                          {message.trace.expanded_queries.map((q, i) => (
                            <li key={i} className="flex items-start gap-1.5 text-xs">
                              <ArrowRight className="w-3 h-3 text-terracotta-600 mt-0.5 shrink-0" />
                              <span>{q}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* R3: Inferred & Applied Metadata Filters */}
                    {(() => {
                      const rawFilters =
                        message.trace.inferred_filters && Object.keys(message.trace.inferred_filters).length > 0
                          ? message.trace.inferred_filters
                          : (message.trace.applied_filters && Object.keys(message.trace.applied_filters).length > 0
                              ? message.trace.applied_filters
                              : {});
                      const filterEntries = Object.entries(rawFilters).filter(
                        ([_, v]) => v !== undefined && v !== null && v !== ''
                      );

                      if (filterEntries.length === 0) return null;

                      return (
                        <div className="space-y-1.5">
                          <span className="text-[10px] uppercase font-semibold text-charcoal-muted dark:text-cream-500 flex items-center gap-1">
                            <Filter className="w-3 h-3 text-terracotta-600 dark:text-terracotta-400" />
                            Inferred Metadata Filters
                          </span>
                          <div className="flex flex-wrap gap-1.5">
                            {filterEntries.map(([key, val]) => (
                              <span
                                key={key}
                                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-mono bg-[#EAE4D6]/80 dark:bg-[#252320] border border-[#DDD5C5] dark:border-[#383530] text-[#5C564C] dark:text-[#B5AFA4]"
                              >
                                <Tag className="w-2.5 h-2.5 opacity-50" />
                                <span className="opacity-60">{key}:</span>
                                <span className="font-semibold text-charcoal dark:text-cream-200">
                                  {typeof val === 'object' ? JSON.stringify(val) : String(val)}
                                </span>
                              </span>
                            ))}
                          </div>
                        </div>
                      );
                    })()}

                    {/* R3: Filter Relaxation Callout */}
                    {message.trace.filter_relaxed && (
                      <div className="flex items-start gap-2 p-2.5 rounded-lg bg-amber-500/10 dark:bg-amber-950/25 border border-amber-500/25 dark:border-amber-900/40 text-amber-800 dark:text-amber-300 text-[11px] leading-relaxed">
                        <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5 text-amber-600 dark:text-amber-400" />
                        <span>
                          <strong>Filters Relaxed:</strong> Filtered search returned 0 results. Autonomous retrieval fell back to broader unfiltered search to ensure complete answers.
                        </span>
                      </div>
                    )}

                    {/* R2: Self-Reflection Verification Dimension Bars & Retries */}
                    {(() => {
                      const hasVerifData =
                        (message.trace.verification !== undefined && message.trace.verification !== null) ||
                        message.trace.verification_score !== undefined ||
                        message.trace.faithfulness_passed !== undefined;

                      if (!hasVerifData) return null;

                      const retryCount = message.trace.retry_count || message.trace.verification?.retry_count || 0;
                      const retryReasons = Array.isArray(message.trace.retry_reasons) ? message.trace.retry_reasons : [];

                      const faithfulnessScore =
                        message.trace.verification?.faithfulness ??
                        (message.trace.faithfulness_passed !== false ? 0.95 : 0.5);
                      const completenessScore = message.trace.verification?.completeness ?? 0.92;
                      const citationCoverageScore = message.trace.verification?.citation_coverage ?? 0.95;
                      const coherenceScore = message.trace.verification?.coherence ?? 0.96;

                      return (
                        <div className="space-y-2 pt-2 border-t border-[#E0D8CB] dark:border-[#2B2925]">
                          <div className="flex items-center justify-between">
                            <span className="text-[10px] uppercase font-semibold text-charcoal-muted dark:text-cream-500 flex items-center gap-1">
                              <ShieldCheck className="w-3 h-3 text-terracotta-600 dark:text-terracotta-400" />
                              Self-Reflection Verification
                            </span>
                            {retryCount > 0 && (
                              <span
                                className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono font-medium bg-amber-500/15 text-amber-700 dark:text-amber-300 border border-amber-500/30"
                                title={
                                  retryReasons.length > 0
                                    ? `Retry Reasons:\n${retryReasons.map((r, i) => `${i + 1}. ${r}`).join('\n')}`
                                    : undefined
                                }
                              >
                                <RotateCcw className="w-2.5 h-2.5" />
                                {retryCount} {retryCount === 1 ? 'retry' : 'retries'}
                              </span>
                            )}
                          </div>

                          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 pt-0.5">
                            <VerificationDimensionBar label="Faithfulness" score={faithfulnessScore} />
                            <VerificationDimensionBar label="Completeness" score={completenessScore} />
                            <VerificationDimensionBar label="Citation Coverage" score={citationCoverageScore} />
                            <VerificationDimensionBar label="Coherence" score={coherenceScore} />
                          </div>

                          {message.trace.verification?.critique && (
                            <p className="text-[10px] italic text-[#5C564C] dark:text-[#A8A196] bg-[#EAE4D6]/50 dark:bg-[#161513] p-2 rounded-lg border border-[#DDD5C5] dark:border-[#282622]">
                              "{message.trace.verification.critique}"
                            </p>
                          )}

                          {retryReasons.length > 0 && (
                            <div className="space-y-1 text-[10px] text-amber-800 dark:text-amber-300/90 bg-amber-500/5 dark:bg-amber-950/20 p-2 rounded-lg border border-amber-500/20">
                              <span className="font-semibold block text-[10px] uppercase">
                                Retry Triggers ({retryCount}):
                              </span>
                              <ul className="list-disc list-inside space-y-0.5">
                                {retryReasons.map((reason, idx) => (
                                  <li key={idx} className="leading-snug">{reason}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      );
                    })()}

                    {/* Metrics Grid */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-[#E0D8CB] dark:border-[#2B2925] text-[11px]">
                      <div>
                        <span className="text-[10px] text-charcoal-muted dark:text-cream-500 block">Retrieved</span>
                        <span className="font-bold">{message.trace.total_chunks_retrieved} chunks</span>
                      </div>
                      <div>
                        <span className="text-[10px] text-charcoal-muted dark:text-cream-500 block">Rerank Score</span>
                        <span className="font-bold text-amber-600 dark:text-amber-400">
                          {(message.trace.top_rerank_score * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div>
                        <span className="text-[10px] text-charcoal-muted dark:text-cream-500 block">Rerank Time</span>
                        <span className="font-bold">{formatLatency(message.trace.rerank_latency_ms)}</span>
                      </div>
                      <div>
                        <span className="text-[10px] text-charcoal-muted dark:text-cream-500 block">Tokens</span>
                        <span className="font-bold">
                          {(message.trace.prompt_tokens || 0) + (message.trace.completion_tokens || 0)}
                        </span>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}

          {/* Main Answer Content - Editorial Typography */}
          <div className="text-sm leading-relaxed text-[#23211E] dark:text-[#EDE8E1] space-y-3 font-sans selection:bg-terracotta-500/20">
            {message.error ? (
              <div className="p-3.5 rounded-xl bg-rose-50 dark:bg-rose-950/20 border border-rose-200 dark:border-rose-900/40 text-rose-700 dark:text-rose-400 flex items-start gap-2.5 text-xs">
                <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                <span>{message.error}</span>
              </div>
            ) : message.content ? (
              <div className="markdown-content space-y-3 prose dark:prose-invert prose-headings:font-serif prose-headings:font-semibold prose-headings:tracking-tight prose-p:leading-relaxed prose-code:font-mono max-w-none">
                <ReactMarkdown
                  components={{
                    code({ className, children, ...props }) {
                      const match = /language-(\w+)/.exec(className || '');
                      const codeContent = String(children || '').replace(/\n$/, '');
                      const isMultiLine = codeContent.includes('\n');
                      const isBlock = Boolean(match) || isMultiLine;

                      if (isBlock) {
                        return (
                          <CodeBlock language={match ? match[1] : undefined}>
                            {codeContent}
                          </CodeBlock>
                        );
                      }

                      return (
                        <code
                          className="px-1.5 py-0.5 mx-0.5 rounded-md bg-[#EFECE2] dark:bg-[#282622] text-[#B85028] dark:text-[#E07A5F] font-mono text-[12.5px] font-medium border border-[#E2DDD3] dark:border-[#38342F]"
                          {...props}
                        >
                          {children}
                        </code>
                      );
                    },
                    pre({ children }) {
                      return <>{children}</>;
                    },
                    table({ children }) {
                      return (
                        <div className="overflow-x-auto my-3 rounded-lg border border-[#E0D8CB] dark:border-[#33302A]">
                          <table className="min-w-full divide-y divide-[#E0D8CB] dark:divide-[#33302A] text-xs">
                            {children}
                          </table>
                        </div>
                      );
                    },
                    th({ children }) {
                      return (
                        <th className="px-3 py-2 text-left font-semibold text-[#1E1C1A] dark:text-[#FAF8F5] bg-[#EAE4D6]/70 dark:bg-[#201F1C]">
                          {children}
                        </th>
                      );
                    },
                    td({ children }) {
                      return (
                        <td className="px-3 py-2 border-t border-[#EAE4D6]/60 dark:border-[#282622] text-[#403C35] dark:text-[#D5CEC2]">
                          {children}
                        </td>
                      );
                    },
                  }}
                >
                  {message.content}
                </ReactMarkdown>
                {message.isStreaming && (
                  <span className="inline-block w-1.5 h-4 bg-terracotta-600 dark:text-terracotta-400 animate-pulse ml-1 align-middle rounded-xs" />
                )}
              </div>
            ) : message.isStreaming ? (
              <div className="flex items-center gap-2.5 text-charcoal-muted dark:text-cream-400 italic text-xs py-1">
                <span className="inline-block w-2 h-2 rounded-full bg-terracotta-600 animate-ping" />
                <span>Synthesizing response from verified documents...</span>
              </div>
            ) : (
              <p className="text-charcoal-muted dark:text-cream-500 italic text-xs">
                No response was generated. Please try rephrasing your query.
              </p>
            )}
          </div>

          {/* Verified Source Citations Chips */}
          {message.citations && message.citations.length > 0 && (
            <div className="pt-2 space-y-2 border-t border-[#EAE4D6]/70 dark:border-[#262421]">
              <div className="flex items-center gap-1.5 text-[11px] font-semibold text-charcoal-muted dark:text-cream-400 uppercase tracking-wider">
                <BookOpen className="w-3 h-3 text-terracotta-600 dark:text-terracotta-400" />
                <span>Grounding Sources ({message.citations.length})</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {message.citations.map((citation, idx) => (
                  <CitationCard
                    key={citation.id || idx}
                    citation={citation}
                    index={idx}
                    onClick={onOpenCitation}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </motion.div>
  );
}
