import React, { useState } from 'react';
import { Brain, ChevronDown, ChevronUp, AlertTriangle } from 'lucide-react';
import { ThinkingEvent, ThinkingDetailLevel, ReasoningSummary, filterEventsByDetailLevel } from '../types/thinking';
import { ThinkingStep } from './ThinkingStep';

export interface ThinkingPanelProps {
  events: ThinkingEvent[];
  isStreaming?: boolean;
  isExpanded?: boolean;
  onToggleExpanded?: () => void;
  detailLevel?: ThinkingDetailLevel;
  onDetailLevelChange?: (level: ThinkingDetailLevel) => void;
  reasoningSummary?: ReasoningSummary | null;
  totalDurationMs?: number;
}

function formatDuration(ms?: number): string {
  if (!ms || ms <= 0) return '0s';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function ThinkingPanel({
  events,
  isStreaming = false,
  isExpanded: controlledExpanded,
  onToggleExpanded,
  detailLevel = 'standard',
  onDetailLevelChange,
  reasoningSummary,
  totalDurationMs = 0,
}: ThinkingPanelProps) {
  const [internalExpanded, setInternalExpanded] = useState(isStreaming);
  const isExpanded = controlledExpanded !== undefined ? controlledExpanded : internalExpanded;

  const handleToggle = () => {
    if (onToggleExpanded) {
      onToggleExpanded();
    } else {
      setInternalExpanded((prev) => !prev);
    }
  };

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
    : `Thought for ${formatDuration(effectiveDuration)}`;

  const panelId = `aether-thinking-${Math.random().toString(36).substring(2, 8)}`;

  return (
    <div
      className="mb-3 rounded-2xl border border-white/10 bg-white/5 overflow-hidden shadow-xs"
      role="region"
      aria-label="AI Reasoning Steps"
    >
      {/* Header Bar */}
      <button
        type="button"
        onClick={handleToggle}
        aria-expanded={isExpanded}
        aria-controls={panelId}
        className="w-full flex items-center justify-between px-4 py-2.5 text-left text-sm text-white/90 hover:bg-white/5 transition-colors cursor-pointer"
      >
        <div className="flex items-center gap-2 min-w-0">
          <Brain
            className={`w-4 h-4 shrink-0 ${
              isStreaming
                ? 'text-violet-400 animate-pulse'
                : hasDegraded
                ? 'text-amber-400'
                : 'text-violet-300'
            }`}
          />
          <span className="font-medium text-xs font-mono truncate">{headerTitle}</span>

          {isStreaming && (
            <span className="inline-flex items-center px-1.5 py-0.2 rounded-full text-[9px] font-mono bg-violet-500/20 text-violet-300 font-semibold animate-pulse">
              Live
            </span>
          )}

          {hasDegraded && !isStreaming && (
            <span className="inline-flex items-center gap-1 px-1.5 py-0.2 rounded text-[9.5px] font-mono bg-amber-500/20 text-amber-300 border border-amber-500/30">
              <AlertTriangle className="w-2.5 h-2.5" />
              <span>Degraded</span>
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {onDetailLevelChange && (
            <div
              className="flex items-center gap-1 bg-white/10 p-0.5 rounded-md text-[10px] font-mono"
              onClick={(e) => e.stopPropagation()}
            >
              {(['compact', 'standard', 'detailed'] as ThinkingDetailLevel[]).map((lvl) => (
                <button
                  key={lvl}
                  type="button"
                  onClick={() => onDetailLevelChange(lvl)}
                  className={`px-1.5 py-0.5 rounded capitalize transition-all cursor-pointer ${
                    detailLevel === lvl
                      ? 'bg-violet-600 text-white font-semibold shadow-xs'
                      : 'text-white/50 hover:text-white/90'
                  }`}
                  aria-pressed={detailLevel === lvl}
                >
                  {lvl}
                </button>
              ))}
            </div>
          )}

          {isExpanded ? (
            <ChevronUp className="w-4 h-4 text-white/50 shrink-0" />
          ) : (
            <ChevronDown className="w-4 h-4 text-white/50 shrink-0" />
          )}
        </div>
      </button>

      {/* Expanded Steps Container */}
      {isExpanded && (
        <div
          id={panelId}
          className="px-3 pb-3 pt-1 space-y-1 max-h-60 overflow-y-auto border-t border-white/10"
          aria-live="polite"
        >
          {visibleEvents.length > 0 ? (
            visibleEvents.map((event) => (
              <ThinkingStep key={event.id || event.stage} event={event} detailLevel={detailLevel} />
            ))
          ) : (
            <p className="text-xs text-white/40 py-1 px-2 italic">
              {isStreaming ? 'Synthesizing reasoning stages…' : 'No reasoning steps recorded.'}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
