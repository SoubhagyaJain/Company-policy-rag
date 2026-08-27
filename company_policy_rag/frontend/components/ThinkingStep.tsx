'use client';

import React from 'react';
import { motion } from 'framer-motion';
import {
  CheckCircle2,
  AlertTriangle,
  Loader2,
  Brain,
  Search,
  Layers,
  FileCheck,
  Eye,
  FileText,
  Clock,
  Sparkles,
  HelpCircle,
  SkipForward,
} from 'lucide-react';
import { ThinkingEvent, ThinkingDetailLevel } from '../types/thinking';
import { cn, formatLatency } from '../lib/utils';

export interface ThinkingStepProps {
  event: ThinkingEvent;
  detailLevel: ThinkingDetailLevel;
}

export function getStageIcon(stage: string, status: string) {
  if (status === 'running') {
    return <Loader2 className="w-3.5 h-3.5 text-terracotta-600 dark:text-terracotta-400 animate-spin shrink-0" />;
  }
  if (status === 'warning' || stage === 'degraded') {
    return <AlertTriangle className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400 shrink-0" />;
  }
  if (status === 'skipped') {
    return <SkipForward className="w-3.5 h-3.5 text-charcoal-muted/60 dark:text-cream-500/60 shrink-0" />;
  }
  if (status === 'completed') {
    return <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400 shrink-0" />;
  }

  switch (stage) {
    case 'received':
    case 'query_analysis':
    case 'query_rewrite':
      return <Brain className="w-3.5 h-3.5 text-sky-600 dark:text-sky-400 shrink-0" />;
    case 'conversation_context':
    case 'follow_up_resolution':
      return <Sparkles className="w-3.5 h-3.5 text-purple-600 dark:text-purple-400 shrink-0" />;
    case 'retrieval':
    case 'reranking':
      return <Search className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400 shrink-0" />;
    case 'evidence_analysis':
    case 'evidence_reuse':
    case 'page_expansion':
      return <Layers className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400 shrink-0" />;
    case 'visual_analysis':
      return <Eye className="w-3.5 h-3.5 text-teal-600 dark:text-teal-400 shrink-0" />;
    case 'evidence_verification':
      return <FileCheck className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400 shrink-0" />;
    case 'answer_planning':
    case 'answer_generation':
    case 'citation_building':
      return <FileText className="w-3.5 h-3.5 text-terracotta-600 dark:text-terracotta-400 shrink-0" />;
    default:
      return <HelpCircle className="w-3.5 h-3.5 text-charcoal-muted shrink-0" />;
  }
}

export function ThinkingStep({ event, detailLevel }: ThinkingStepProps) {
  const isRunning = event.status === 'running';
  const isDegraded = event.status === 'warning' || event.stage === 'degraded';
  const isSkipped = event.status === 'skipped';
  const showDetails = detailLevel === 'detailed' && event.details && Object.keys(event.details).length > 0;

  return (
    <motion.div
      initial={{ opacity: 0, x: -4 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.2 }}
      className={cn(
        'group flex flex-col py-1.5 px-2.5 rounded-lg text-xs transition-colors',
        isRunning && 'bg-terracotta-500/5 dark:bg-terracotta-500/10 border border-terracotta-500/20',
        isDegraded && 'bg-amber-500/10 dark:bg-amber-950/20 border border-amber-500/20',
        isSkipped && 'opacity-60 bg-cream-200/30 dark:bg-sand-dark/20',
        !isRunning && !isDegraded && !isSkipped && 'hover:bg-cream-200/40 dark:hover:bg-sand-dark/40'
      )}
      role="status"
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          {getStageIcon(event.stage, event.status)}
          <span
            className={cn(
              'font-medium text-xs font-sans truncate',
              isRunning
                ? 'text-terracotta-700 dark:text-terracotta-300 font-semibold'
                : isDegraded
                ? 'text-amber-800 dark:text-amber-300'
                : isSkipped
                ? 'text-charcoal-muted line-through dark:text-cream-500'
                : 'text-charcoal dark:text-cream-200'
            )}
          >
            {event.title || event.stage.replace(/_/g, ' ')}
          </span>
        </div>

        {event.duration_ms !== undefined && event.duration_ms > 0 && (
          <span className="font-mono text-[10px] text-charcoal-muted dark:text-cream-500 shrink-0 flex items-center gap-1">
            <Clock className="w-2.5 h-2.5 opacity-60" />
            {formatLatency(event.duration_ms)}
          </span>
        )}
      </div>

      {event.summary && (
        <p className="text-[11px] font-sans text-charcoal-muted dark:text-cream-400 mt-0.5 ml-5.5 leading-relaxed">
          {event.summary}
        </p>
      )}

      {showDetails && (
        <div className="mt-1 ml-5.5 flex flex-wrap gap-1.5 pt-1 border-t border-sand-border dark:border-sand-darkBorder/40">
          {Object.entries(event.details || {}).map(([key, val]) => {
            if (val === undefined || val === null || val === '') return null;
            return (
              <span
                key={key}
                className="inline-flex items-center gap-1 px-1.5 py-0.2 rounded text-[9.5px] font-mono bg-cream-200/80 dark:bg-sand-dark text-charcoal-muted dark:text-cream-400"
              >
                <span className="opacity-60">{key.replace(/_/g, ' ')}:</span>
                <span className="font-semibold text-charcoal dark:text-cream-200">
                  {typeof val === 'object' ? JSON.stringify(val) : String(val)}
                </span>
              </span>
            );
          })}
        </div>
      )}
    </motion.div>
  );
}
