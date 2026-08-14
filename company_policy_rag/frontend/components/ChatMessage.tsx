'use client';

import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Sparkles,
  ChevronDown,
  ChevronUp,
  Cpu,
  Clock,
  BookOpen,
  Copy,
  Check,
  AlertCircle,
  BrainCircuit,
  ArrowRight,
} from 'lucide-react';
import { ChatMessageData, Citation } from '../lib/types';
import { CitationCard } from './CitationCard';
import { formatLatency, cn } from '../lib/utils';

interface ChatMessageProps {
  message: ChatMessageData;
  onOpenCitation: (citation: Citation) => void;
}

export function ChatMessage({ message, onOpenCitation }: ChatMessageProps) {
  const [showTrace, setShowTrace] = useState(false);
  const [copied, setCopied] = useState(false);

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
                Policy Assistant
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

          {/* Collapsible Thinking / RAG Reasoning Banner (Claude style) */}
          {message.trace && (
            <div className="pt-0.5">
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

              <AnimatePresence>
                {showTrace && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.18 }}
                    className="mt-2 p-3.5 rounded-xl bg-[#F4F1E8] dark:bg-[#1E1D1A] border border-[#E2DBD0] dark:border-[#2D2B27] text-xs font-mono space-y-2 text-[#3D3A35] dark:text-[#D5CEC4]"
                  >
                    <div className="flex items-center justify-between border-b border-[#E0D8CB] dark:border-[#2B2925] pb-2">
                      <span className="font-bold text-terracotta-700 dark:text-terracotta-400">
                        {message.trace.trace_id}
                      </span>
                      <span className="flex items-center gap-1 text-emerald-600 dark:text-emerald-400 font-semibold">
                        <Clock className="w-3 h-3" /> {formatLatency(message.trace.total_latency_ms)}
                      </span>
                    </div>

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
              <div className="markdown-content space-y-3 prose dark:prose-invert prose-headings:font-serif prose-headings:font-semibold prose-headings:tracking-tight prose-p:leading-relaxed prose-pre:bg-[#201F1C] prose-pre:border prose-pre:border-[#2F2D29] prose-pre:text-cream-100 prose-code:font-mono prose-code:text-terracotta-700 dark:prose-code:text-terracotta-400 max-w-none">
                <ReactMarkdown>{message.content}</ReactMarkdown>
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
