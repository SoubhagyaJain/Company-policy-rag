'use client';

/** Space-styled reasoning trace — a collapsible, mono-labeled panel over the
 *  ThinkingEvent stream. Auto-expands while streaming. Presentation only; the
 *  events themselves are produced by useChatStream. */

import { useState } from 'react';
import { ChevronDown, ChevronRight, Loader2, Check, AlertTriangle } from 'lucide-react';
import type { ThinkingEvent } from '../../types/thinking';
import { formatLatency } from '../../lib/utils';

interface SpaceThinkingPanelProps {
  events: ThinkingEvent[];
  isStreaming?: boolean;
  totalDurationMs?: number;
}

function StatusIcon({ status }: { status: ThinkingEvent['status'] }) {
  if (status === 'running' || status === 'pending')
    return <Loader2 className="h-3 w-3 animate-spin text-[var(--sp-accent)]" />;
  if (status === 'warning' || status === 'failed')
    return <AlertTriangle className="h-3 w-3 text-amber-400" />;
  return <Check className="h-3 w-3 text-[var(--sp-accent)]" />;
}

export function SpaceThinkingPanel({ events, isStreaming, totalDurationMs }: SpaceThinkingPanelProps) {
  const [expanded, setExpanded] = useState<boolean>(Boolean(isStreaming));

  if (!events || events.length === 0) return null;

  const total =
    totalDurationMs ??
    events.reduce((sum, e) => sum + (e.duration_ms || 0), 0);
  const headline = isStreaming
    ? 'Thinking…'
    : `Thought for ${formatLatency(total || 0)}`;

  return (
    <div className="sp-card mb-2 overflow-hidden rounded-2xl">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="sp-muted flex w-full items-center gap-2 px-3.5 py-2.5 text-left transition-colors hover:text-[var(--sp-text)]"
      >
        {expanded ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
        <span className="sp-mono text-[10px] uppercase tracking-[0.22em]">{headline}</span>
        {isStreaming && <Loader2 className="ml-auto h-3 w-3 animate-spin text-[var(--sp-accent)]" />}
      </button>

      {expanded && (
        <div className="border-t border-[var(--sp-hairline)] px-3.5 py-2.5">
          <ol className="space-y-2.5">
            {events.map((ev) => (
              <li key={ev.id} className="flex gap-2.5">
                <span className="mt-0.5 flex-none">
                  <StatusIcon status={ev.status} />
                </span>
                <div className="min-w-0">
                  <div className="flex items-baseline gap-2">
                    <span className="sp-text text-[12.5px] font-medium">{ev.title}</span>
                    {ev.duration_ms ? (
                      <span className="sp-mono sp-faint text-[9px]">{formatLatency(ev.duration_ms)}</span>
                    ) : null}
                  </div>
                  {ev.summary && <p className="sp-faint text-[11.5px] leading-snug">{ev.summary}</p>}
                </div>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}

export default SpaceThinkingPanel;
