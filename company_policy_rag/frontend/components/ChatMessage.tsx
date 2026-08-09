'use client';

import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { motion } from 'framer-motion';
import {
  User,
  Sparkles,
  ChevronDown,
  ChevronUp,
  Cpu,
  Clock,
  Zap,
  BookOpen,
  Copy,
  Check,
  AlertCircle,
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
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className={cn('flex gap-3 py-4 px-2 sm:px-4', isUser ? 'justify-end' : 'justify-start')}
    >
      {!isUser && (
        <div className="w-8 h-8 rounded-xl bg-terracotta-600/10 dark:bg-terracotta-500/20 border border-terracotta-500/30 text-terracotta-700 dark:text-terracotta-400 flex items-center justify-center shrink-0 font-serif font-bold text-sm shadow-sm mt-0.5">
          <Sparkles className="w-4 h-4 text-terracotta-600" />
        </div>
      )}

      <div className={cn('max-w-3xl flex-1 space-y-2', isUser && 'flex flex-col items-end')}>
        {/* Header line */}
        <div className="flex items-center gap-2 text-xs text-charcoal-muted dark:text-cream-400">
          <span className="font-semibold text-charcoal dark:text-cream-200">
            {isUser ? 'You' : 'Policy AI Assistant'}
          </span>
          <span className="text-[10px]">
            {new Date(message.timestamp).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </span>

          {!isUser && !message.isStreaming && (
            <button
              onClick={handleCopyContent}
              className="ml-auto p-1 text-charcoal-muted dark:text-cream-400 hover:text-charcoal dark:hover:text-cream-100 transition-colors"
              title="Copy Message"
            >
              {copied ? <Check className="w-3 h-3 text-emerald-600" /> : <Copy className="w-3 h-3" />}
            </button>
          )}
        </div>

        {/* Bubble container */}
        <div
          className={cn(
            'p-4 rounded-2xl text-sm leading-relaxed transition-all shadow-sm',
            isUser
              ? 'bg-terracotta-600 text-white rounded-tr-sm font-sans'
              : 'bg-[#FAF9F5]/90 dark:bg-[#1A1917]/90 border border-[#E5E0D8] dark:border-[#2A2925] text-charcoal dark:text-cream-100 rounded-tl-sm backdrop-blur-md'
          )}
        >
          {message.error ? (
            <div className="flex items-start gap-2 text-rose-600 dark:text-rose-400">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{message.error}</span>
            </div>
          ) : isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="markdown-content space-y-3 font-serif">
              <ReactMarkdown>{message.content || 'Thinking...'}</ReactMarkdown>
              {message.isStreaming && (
                <span className="inline-block w-2 h-4 bg-terracotta-600 animate-pulse ml-1 align-middle" />
              )}
            </div>
          )}
        </div>

        {/* Citations Footer */}
        {!isUser && message.citations && message.citations.length > 0 && (
          <div className="pt-2 space-y-1.5">
            <div className="flex items-center gap-1.5 text-[11px] font-semibold text-charcoal-muted dark:text-cream-400 uppercase tracking-wider">
              <BookOpen className="w-3 h-3 text-terracotta-600" />
              <span>Verified Sources ({message.citations.length})</span>
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

        {/* Observability Telemetry Trace Expander */}
        {!isUser && message.trace && (
          <div className="pt-1">
            <button
              onClick={() => setShowTrace((prev) => !prev)}
              className="flex items-center gap-1.5 text-[11px] font-mono text-terracotta-600 dark:text-terracotta-400 hover:underline focus:outline-none"
            >
              <Cpu className="w-3 h-3" />
              <span>
                {showTrace ? 'Hide Observability Trace' : 'View RAG Observability Trace'}
              </span>
              {showTrace ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>

            {showTrace && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-2 p-3 rounded-xl bg-cream-100/80 dark:bg-sand-dark/80 border border-sand-border/70 dark:border-sand-darkBorder/70 text-[11px] font-mono space-y-2 text-charcoal/80 dark:text-cream-300"
              >
                <div className="flex items-center justify-between border-b border-sand-border/50 dark:border-sand-darkBorder/50 pb-1.5">
                  <span className="font-bold text-terracotta-700 dark:text-terracotta-400">
                    Trace ID: {message.trace.trace_id}
                  </span>
                  <span className="flex items-center gap-1 text-emerald-600 font-semibold">
                    <Clock className="w-3 h-3" /> {formatLatency(message.trace.total_latency_ms)}
                  </span>
                </div>

                {message.trace.query_rewritten && (
                  <div>
                    <span className="font-semibold text-charcoal-muted dark:text-cream-400 block">
                      Rewritten Query:
                    </span>
                    <p className="text-charcoal dark:text-cream-100 font-sans italic bg-cream-50 dark:bg-charcoal-dark p-1.5 rounded border border-sand-border/50">
                      "{message.trace.query_rewritten}"
                    </p>
                  </div>
                )}

                {message.trace.expanded_queries && message.trace.expanded_queries.length > 0 && (
                  <div>
                    <span className="font-semibold text-charcoal-muted dark:text-cream-400 block">
                      Sub-Queries Expanded:
                    </span>
                    <ul className="list-disc list-inside space-y-0.5">
                      {message.trace.expanded_queries.map((q, i) => (
                        <li key={i} className="truncate">
                          {q}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 border-t border-sand-border/50">
                  <div>
                    <span className="text-[10px] text-charcoal-muted block">Retrieved</span>
                    <span className="font-bold">{message.trace.total_chunks_retrieved} chunks</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-charcoal-muted block">Top Rerank Score</span>
                    <span className="font-bold text-amber-600">
                      {(message.trace.top_rerank_score * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-charcoal-muted block">Rerank Latency</span>
                    <span className="font-bold">{formatLatency(message.trace.rerank_latency_ms)}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-charcoal-muted block">Tokens Used</span>
                    <span className="font-bold">
                      {(message.trace.prompt_tokens || 0) + (message.trace.completion_tokens || 0)}
                    </span>
                  </div>
                </div>
              </motion.div>
            )}
          </div>
        )}
      </div>

      {isUser && (
        <div className="w-8 h-8 rounded-xl bg-charcoal dark:bg-cream-100 text-cream-100 dark:text-charcoal flex items-center justify-center shrink-0 font-bold text-xs shadow-sm mt-0.5">
          <User className="w-4 h-4" />
        </div>
      )}
    </motion.div>
  );
}
