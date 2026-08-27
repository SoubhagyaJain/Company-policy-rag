'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, ChevronDown, ChevronUp, Sparkles, AlertTriangle } from 'lucide-react';
import { ThinkingEvent, ThinkingDetailLevel, ReasoningSummary, filterEventsByDetailLevel } from '../types/thinking';
import { ThinkingStep } from './ThinkingStep';
import { cn, formatLatency } from '../lib/utils';

export interface ThinkingPanelProps {
  events: ThinkingEvent[];
  isStreaming?: boolean;
  isExpanded: boolean;
  onToggleExpanded: () => void;
  detailLevel: ThinkingDetailLevel;
  onDetailLevelChange?: (level: ThinkingDetailLevel) => void;
  reasoningSummary?: ReasoningSummary | null;
  totalDurationMs?: number;
}

export function ThinkingPanel({
  events,
  isStreaming = false,
  isExpanded,
  onToggleExpanded,
  detailLevel,
  onDetailLevelChange,
  reasoningSummary,
  totalDurationMs = 0,
}: ThinkingPanelProps) {
  // If detail level is 'off' or there are no events and not streaming, return null
  if (detailLevel === 'off' || (!events.length && !isStreaming)) {
    return null;
  }

  const runningEvent = events.find((e) => e.status === 'running');
  const visibleEvents = filterEventsByDetailLevel(events, detailLevel);

  const effectiveDuration =
    totalDurationMs > 0
      ? totalDurationMs
      : (reasoningSummary?.total_duration_ms ?? events.reduce((acc, ev) => acc + (ev.duration_ms || 0), 0));

  const hasDegraded =
    events.some((e) => e.status === 'warning' || e.stage === 'degraded') ||
    Boolean(reasoningSummary?.degraded_stages && reasoningSummary.degraded_stages.length > 0);

  const headerTitle = isStreaming
    ? runningEvent
      ? runningEvent.title
      : 'Thinking…'
    : `Process completed in ${formatLatency(effectiveDuration)}`;

  const panelId = `thinking-steps-${React.useId().replace(/:/g, '')}`;

  return (
    <div
      className="my-2.5 rounded-xl border border-sand-border dark:border-sand-darkBorder bg-[#FAF8F5] dark:bg-[#1A1917] overflow-hidden shadow-xs"
      role="region"
      aria-label="AI Reasoning Steps"
    >
      {/* Header Bar */}
      <div className="flex items-center gap-2 px-3.5 py-2 hover:bg-cream-200/50 dark:hover:bg-sand-dark/50 transition-colors">
        <button
          type="button"
          onClick={onToggleExpanded}
          aria-expanded={isExpanded}
          aria-controls={panelId}
          className="flex min-w-0 flex-1 items-center justify-between gap-2 text-left focus:outline-hidden focus:ring-2 focus:ring-terracotta-500/40 cursor-pointer"
        >
          <div className="flex items-center gap-2 min-w-0">
            <Brain
              className={cn(
                'w-4 h-4 shrink-0',
                isStreaming
                  ? 'text-terracotta-600 dark:text-terracotta-400 animate-pulse'
                  : hasDegraded
                  ? 'text-amber-600 dark:text-amber-400'
                  : 'text-charcoal-muted dark:text-cream-400'
              )}
            />
            <span className="text-xs font-mono font-medium text-charcoal dark:text-cream-200 truncate">
              {headerTitle}
            </span>

            {isStreaming && (
              <span className="inline-flex items-center px-1.5 py-0.2 rounded-full text-[9px] font-mono bg-terracotta-500/10 text-terracotta-600 dark:text-terracotta-400 font-semibold animate-pulse">
                Live
              </span>
            )}

            {hasDegraded && !isStreaming && (
              <span className="inline-flex items-center gap-1 px-1.5 py-0.2 rounded text-[9.5px] font-mono bg-amber-500/10 text-amber-700 dark:text-amber-400 border border-amber-500/20">
                <AlertTriangle className="w-2.5 h-2.5" />
                <span>Degraded</span>
              </span>
            )}
          </div>

          {isExpanded ? (
            <ChevronUp className="w-3.5 h-3.5 text-charcoal-muted dark:text-cream-400 shrink-0" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5 text-charcoal-muted dark:text-cream-400 shrink-0" />
          )}
        </button>

        {onDetailLevelChange && (
          <div
            className="flex shrink-0 items-center gap-1 bg-cream-200/80 dark:bg-sand-dark p-0.5 rounded-md text-[10px] font-mono"
            role="group"
            aria-label="Reasoning detail level"
          >
            {(['compact', 'standard', 'detailed'] as ThinkingDetailLevel[]).map((lvl) => (
              <button
                key={lvl}
                type="button"
                onClick={() => onDetailLevelChange(lvl)}
                className={cn(
                  'px-1.5 py-0.5 rounded capitalize transition-all cursor-pointer',
                  detailLevel === lvl
                    ? 'bg-white dark:bg-[#252320] text-charcoal dark:text-cream-100 font-semibold shadow-2xs'
                    : 'text-charcoal-muted hover:text-charcoal dark:text-cream-500 dark:hover:text-cream-200'
                )}
                aria-pressed={detailLevel === lvl}
              >
                {lvl}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Expanded Steps Container */}
      <AnimatePresence initial={false}>
        {isExpanded && (
          <motion.div
            id={panelId}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: 'easeOut' }}
            className="border-t border-sand-border dark:border-sand-darkBorder px-2.5 py-2 space-y-1 bg-[#F5F2EB]/60 dark:bg-[#141312]/60 max-h-72 overflow-y-auto"
            aria-live="polite"
          >
            {visibleEvents.length > 0 ? (
              visibleEvents.map((event) => (
                <ThinkingStep key={event.id || event.stage} event={event} detailLevel={detailLevel} />
              ))
            ) : (
              <p className="text-xs text-charcoal-muted dark:text-cream-500 py-1 px-2 italic">
                {isStreaming ? 'Synthesizing reasoning stages…' : 'No reasoning steps recorded for this detail level.'}
              </p>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
