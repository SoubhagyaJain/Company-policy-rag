'use client';

/** Small glass pills summarizing a query's trace: verification score, cache hit,
 *  query classification. Reads the same QueryTrace fields as ChatMessage. */

import { ShieldCheck, Zap, GitBranch } from 'lucide-react';
import type { QueryTrace } from '../../lib/types';

interface SpaceTracePillsProps {
  trace?: QueryTrace;
}

export function SpaceTracePills({ trace }: SpaceTracePillsProps) {
  if (!trace) return null;

  const score = trace.verification_score ?? trace.verification?.composite_score;
  const passed =
    trace.faithfulness_passed === false
      ? false
      : trace.verification?.passed ?? trace.faithfulness_passed ?? undefined;
  const hasVerification = typeof score === 'number' || passed !== undefined;

  const pills: React.ReactNode[] = [];

  if (hasVerification) {
    const pct = typeof score === 'number' ? `${Math.round((score <= 1 ? score * 100 : score))}%` : null;
    pills.push(
      <span
        key="verify"
        className={`sp-tracepill sp-mono flex items-center gap-1 rounded-full px-2.5 py-1 text-[9.5px] uppercase tracking-[0.12em] ${
          passed === false ? 'text-[var(--sp-warn-text)]' : 'text-[var(--sp-accent-text)]'
        }`}
      >
        <ShieldCheck className="h-3 w-3" />
        {passed === false ? 'Unverified' : 'Grounded'}
        {pct && <span className="opacity-80">{pct}</span>}
      </span>,
    );
  }

  if (trace.cache_hit) {
    const sim =
      trace.cache_similarity !== null && trace.cache_similarity !== undefined
        ? ` ${Math.round(trace.cache_similarity * 100)}%`
        : '';
    pills.push(
      <span key="cache" className="sp-tracepill sp-mono flex items-center gap-1 rounded-full px-2.5 py-1 text-[9.5px] uppercase tracking-[0.12em]">
        <Zap className="h-3 w-3" /> Cache{sim}
      </span>,
    );
  }

  if (trace.query_type) {
    pills.push(
      <span key="qtype" className="sp-tracepill sp-mono flex items-center gap-1 rounded-full px-2.5 py-1 text-[9.5px] uppercase tracking-[0.12em]">
        <GitBranch className="h-3 w-3" /> {trace.query_type}
        {trace.routing_confidence !== undefined && trace.routing_confidence !== null && (
          <span className="opacity-80">{Math.round(trace.routing_confidence * 100)}%</span>
        )}
      </span>,
    );
  }

  // What the grounding check could not find in the sources. The answer itself is
  // never altered for these; they are listed so the reader can judge.
  const verification = trace.verification;
  const terms = passed === false ? verification?.unsupported_terms ?? [] : [];
  const statements =
    passed === false
      ? [
          ...(verification?.missing_aspects ?? []),
          ...(verification?.unsupported_claims ?? []).filter((c) => !c.startsWith('Terms not found in the sources')),
          ...(verification?.citation_errors ?? []),
        ].slice(0, 3)
      : [];

  if (pills.length === 0) return null;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-1.5">{pills}</div>
      {(terms.length > 0 || statements.length > 0) && (
        <div className="sp-muted text-[11px] leading-relaxed" role="note">
          {terms.length > 0 && (
            <p>
              <span className="text-[var(--sp-warn-text)]">Not found in the sources:</span> {terms.join(', ')}
            </p>
          )}
          {statements.map((statement) => (
            <p key={statement}>
              <span className="text-[var(--sp-warn-text)]">Check:</span> {statement}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

export default SpaceTracePills;
