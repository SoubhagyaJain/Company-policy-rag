'use client';

/** Building blocks of the storage console: dense, flat, and quiet. Colour is
 *  reserved for status, safety class and storage category; everything that
 *  carries meaning through colour also carries it through text or shape. */

import React from 'react';
import { AlertTriangle, ChevronDown, Info, RefreshCw, ShieldCheck } from 'lucide-react';

import { cn, formatBytes, formatDelta } from '@/lib/utils';
import type { StorageKind, StorageSafety } from '@/lib/types';

/* ── Section ─────────────────────────────────────────────────────────────── */

interface SectionProps {
  id: string;
  title: string;
  hint?: React.ReactNode;
  open: boolean;
  onToggle: (id: string) => void;
  aside?: React.ReactNode;
  children: React.ReactNode;
}

export function Section({ id, title, hint, open, onToggle, aside, children }: SectionProps) {
  return (
    <section id={`sc-${id}`} className="sc-rule scroll-mt-4 pt-4" aria-labelledby={`sc-${id}-title`}>
      <div className="flex items-center justify-between gap-3">
        <button
          type="button"
          onClick={() => onToggle(id)}
          aria-expanded={open}
          aria-controls={`sc-${id}-body`}
          className="sc-focus group flex min-w-0 items-baseline gap-2 rounded text-left"
        >
          <ChevronDown
            className={cn('sc-t3 h-3.5 w-3.5 shrink-0 self-center transition-transform', !open && '-rotate-90')}
            aria-hidden
          />
          <h2 id={`sc-${id}-title`} className="sc-t1 text-[14px] font-semibold tracking-[-0.005em]">
            {title}
          </h2>
          {hint && <span className="sc-t3 truncate text-[11.5px]">{hint}</span>}
        </button>
        {aside && <div className="flex shrink-0 items-center gap-2">{aside}</div>}
      </div>
      {open && (
        <div id={`sc-${id}-body`} className="mt-3">
          {children}
        </div>
      )}
    </section>
  );
}

/* ── Badges ──────────────────────────────────────────────────────────────── */

const KIND_LABELS: Record<StorageKind, string> = {
  USER_DATA: 'User data',
  PERSISTENT: 'Persistent',
  CACHE: 'Cache',
  REBUILDABLE: 'Rebuildable',
  RUNTIME_ONLY: 'Runtime only',
  SYSTEM_DATA: 'System data',
  TEMPORARY: 'Temporary',
};

const KIND_HELP: Record<StorageKind, string> = {
  USER_DATA: 'Something you put here. It cannot be recreated by the app.',
  PERSISTENT: 'Stored on disk. Survives a restart.',
  CACHE: 'Kept only to avoid repeating work.',
  REBUILDABLE: 'The app can produce it again if it is removed.',
  RUNTIME_ONLY: 'Held in memory. Gone when the backend or model process stops.',
  SYSTEM_DATA: 'Written by the app for its own use.',
  TEMPORARY: 'Left behind by a past run. Nothing reads it now.',
};

const KIND_ORDER: StorageKind[] = [
  'USER_DATA',
  'RUNTIME_ONLY',
  'TEMPORARY',
  'CACHE',
  'PERSISTENT',
  'REBUILDABLE',
  'SYSTEM_DATA',
];

export function KindBadges({ kinds, limit }: { kinds: StorageKind[]; limit?: number }) {
  const ordered = KIND_ORDER.filter((kind) => kinds.includes(kind));
  const shown = limit ? ordered.slice(0, limit) : ordered;
  return (
    <span className="inline-flex flex-wrap items-center gap-1">
      {shown.map((kind) => (
        <span
          key={kind}
          title={KIND_HELP[kind]}
          className={cn(
            'sc-badge',
            (kind === 'RUNTIME_ONLY' || kind === 'TEMPORARY') && 'sc-badge--volatile',
            kind === 'USER_DATA' && 'sc-badge--user',
          )}
        >
          {KIND_LABELS[kind]}
        </span>
      ))}
    </span>
  );
}

const SAFETY: Record<StorageSafety, { label: string; help: string; icon: React.ReactNode; tone: string }> = {
  SAFE: {
    label: 'Safe',
    help: 'Removes nothing the app still uses.',
    icon: <ShieldCheck className="h-3 w-3" aria-hidden />,
    tone: 'sc-tag--safe',
  },
  REBUILDABLE: {
    label: 'Rebuildable',
    help: 'Removed data is regenerated on demand, at a performance cost.',
    icon: <RefreshCw className="h-3 w-3" aria-hidden />,
    tone: 'sc-tag--rebuildable',
  },
  DESTRUCTIVE: {
    label: 'Destructive',
    help: 'Removed data cannot be brought back.',
    icon: <AlertTriangle className="h-3 w-3" aria-hidden />,
    tone: 'sc-tag--destructive',
  },
};

export function SafetyTag({ safety }: { safety: StorageSafety }) {
  const spec = SAFETY[safety] ?? SAFETY.DESTRUCTIVE;
  return (
    <span className={cn('sc-tag', spec.tone)} title={spec.help}>
      {spec.icon}
      {spec.label}
    </span>
  );
}

export const safetyHelp = (safety: StorageSafety) => (SAFETY[safety] ?? SAFETY.DESTRUCTIVE).help;

export type DotState = 'idle' | 'ok' | 'warn' | 'crit' | 'active';

export function StatusDot({ state, className }: { state: DotState; className?: string }) {
  return <span className={cn('sc-dot', `sc-dot--${state}`, className)} aria-hidden />;
}

/* ── Bars ────────────────────────────────────────────────────────────────── */

export interface BarSegment {
  id: string;
  label: string;
  value: number;
  /** Any CSS colour; usually a `var(--…)` token. */
  color: string;
}

interface SegmentBarProps {
  segments: BarSegment[];
  /** Full width of the bar. Defaults to the sum of the segments. */
  total?: number;
  height?: number;
  onHover?: (id: string | null) => void;
  onSelect?: (id: string) => void;
  activeId?: string | null;
  ariaLabel: string;
}

/** Stacked horizontal bar. Segments are separated by a gap in the surface
 *  colour, not by borders, and the remainder of `total` shows as empty track. */
export function SegmentBar({ segments, total, height = 10, onHover, onSelect, activeId, ariaLabel }: SegmentBarProps) {
  const sum = segments.reduce((acc, s) => acc + Math.max(0, s.value), 0);
  const full = Math.max(total ?? sum, sum, 1);
  const visible = segments.filter((s) => s.value > 0);
  return (
    <div
      className="sc-track flex w-full gap-[2px] overflow-hidden rounded-[4px]"
      style={{ height }}
      role="img"
      aria-label={ariaLabel}
      onMouseLeave={onHover ? () => onHover(null) : undefined}
    >
      {visible.map((segment) => {
        const style: React.CSSProperties = {
          width: `${(segment.value / full) * 100}%`,
          minWidth: 2,
          background: segment.color,
          opacity: activeId && activeId !== segment.id ? 0.4 : 1,
        };
        return onSelect ? (
          <button
            key={segment.id}
            type="button"
            aria-label={segment.label}
            onMouseEnter={onHover ? () => onHover(segment.id) : undefined}
            onFocus={onHover ? () => onHover(segment.id) : undefined}
            onBlur={onHover ? () => onHover(null) : undefined}
            onClick={() => onSelect(segment.id)}
            className="sc-focus h-full shrink-0 transition-opacity"
            style={style}
          />
        ) : (
          <span
            key={segment.id}
            onMouseEnter={onHover ? () => onHover(segment.id) : undefined}
            title={onHover ? undefined : `${segment.label}: ${formatBytes(segment.value)}`}
            className="h-full shrink-0 transition-opacity"
            style={style}
          />
        );
      })}
    </div>
  );
}

/** Legend for a SegmentBar: swatch, label, value. Text stays in text colours. */
export function BarLegend({
  segments,
  format = formatBytes,
  className,
}: {
  segments: BarSegment[];
  format?: (value: number) => string;
  className?: string;
}) {
  return (
    <ul className={cn('flex flex-wrap gap-x-4 gap-y-1', className)}>
      {segments.map((segment) => (
        <li key={segment.id} className="flex items-center gap-1.5 text-[11.5px]">
          <span className="h-2 w-2 shrink-0 rounded-[2px]" style={{ background: segment.color }} aria-hidden />
          <span className="sc-t2">{segment.label}</span>
          <span className="sc-num sc-t1">{format(segment.value)}</span>
        </li>
      ))}
    </ul>
  );
}

export type Tone = 'accent' | 'ok' | 'warn' | 'crit' | 'neutral';

const TONE_COLOR: Record<Tone, string> = {
  accent: 'var(--sp-accent)',
  ok: 'var(--sp-success)',
  warn: 'var(--sp-warn)',
  crit: 'var(--sc-critical)',
  neutral: 'var(--sp-text-faint)',
};

export const toneColor = (tone: Tone) => TONE_COLOR[tone];

/** Severity of a utilisation figure: the fill itself says how full it is. */
export function usageTone(fraction: number): Tone {
  if (fraction >= 0.97) return 'crit';
  if (fraction >= 0.9) return 'warn';
  return 'accent';
}

export function MicroBar({
  value,
  max,
  tone = 'accent',
  height = 4,
  className,
}: {
  value: number;
  max: number;
  tone?: Tone;
  height?: number;
  className?: string;
}) {
  const pct = max > 0 ? Math.min(100, Math.max(value > 0 ? 1.5 : 0, (value / max) * 100)) : 0;
  return (
    <div className={cn('sc-track w-full overflow-hidden rounded-full', className)} style={{ height }} aria-hidden>
      <div className="h-full rounded-full" style={{ width: `${pct}%`, background: TONE_COLOR[tone] }} />
    </div>
  );
}

/** Size of one store over time, from the recorded snapshots. */
export function Sparkline({ points, width = 96, height = 22 }: { points: number[]; width?: number; height?: number }) {
  if (points.length < 2) return null;
  const max = Math.max(...points);
  const min = Math.min(...points);
  const span = max - min || 1;
  const path = points
    .map((v, i) => `${(i / (points.length - 1)) * width},${height - 2 - ((v - min) / span) * (height - 4)}`)
    .join(' ');
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} aria-hidden className="shrink-0">
      <polyline
        points={path}
        fill="none"
        stroke="var(--sp-text-faint)"
        strokeWidth="1.25"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

/* ── Values ──────────────────────────────────────────────────────────────── */

/** A figure the runtime does not report. Says so instead of showing a number. */
export function Unavailable({ children = 'Unavailable', reason }: { children?: React.ReactNode; reason?: string }) {
  return (
    <span className="sc-t3 text-[11.5px]" title={reason}>
      {children}
    </span>
  );
}

export function Estimated({ children }: { children: React.ReactNode }) {
  return (
    <span title="Estimated, not measured">
      <span className="sc-t3">~</span>
      {children}
    </span>
  );
}

/** Change over a window. Growth is neither good nor bad, so it stays in text ink. */
export function Delta({ bytes, className }: { bytes: number | null | undefined; className?: string }) {
  if (bytes === null || bytes === undefined) {
    return (
      <span className={cn('sc-t3', className)} title="No snapshot old enough to compare against">
        —
      </span>
    );
  }
  const arrow = bytes > 0 ? '↑' : bytes < 0 ? '↓' : '';
  return (
    <span className={cn('sc-num', bytes === 0 ? 'sc-t3' : 'sc-t2', className)}>
      {arrow} {formatDelta(bytes)}
    </span>
  );
}

/** Label over value, for the dense definition grids. */
export function Field({
  label,
  children,
  title,
  className,
}: {
  label: string;
  children: React.ReactNode;
  title?: string;
  className?: string;
}) {
  return (
    <div className={cn('min-w-0', className)} title={title}>
      <dt className="sc-label">{label}</dt>
      <dd className="sc-num sc-t1 mt-1 truncate text-[12.5px]">{children}</dd>
    </div>
  );
}

export function InfoButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button type="button" onClick={onClick} className="sc-icon-btn" aria-label={`Why is this here: ${label}`} title="Why is this here?">
      <Info className="h-3.5 w-3.5" aria-hidden />
    </button>
  );
}

export function EmptyState({
  icon,
  title,
  detail,
  action,
  className,
}: {
  icon?: React.ReactNode;
  title: string;
  detail?: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn('flex items-start gap-3 px-4 py-4', className)}>
      {icon && <span className="sc-t3 mt-0.5 shrink-0">{icon}</span>}
      <div className="min-w-0">
        <p className="sc-t1 text-[12.5px] font-semibold">{title}</p>
        {detail && <p className="sc-t2 mt-0.5 text-[12px]">{detail}</p>}
        {action && <div className="mt-2">{action}</div>}
      </div>
    </div>
  );
}

/** Small hover/focus card anchored under its trigger. */
export function HoverCard({
  trigger,
  children,
  align = 'left',
  className,
}: {
  trigger: React.ReactNode;
  children: React.ReactNode;
  align?: 'left' | 'right' | 'center';
  className?: string;
}) {
  return (
    <span className={cn('group/tip relative inline-flex', className)}>
      {trigger}
      <span
        role="tooltip"
        className={cn(
          'sc-tip invisible absolute top-full mt-1.5 w-64 p-2.5 opacity-0 transition-opacity',
          'group-hover/tip:visible group-hover/tip:opacity-100 group-focus-within/tip:visible group-focus-within/tip:opacity-100',
          align === 'left' && 'left-0',
          align === 'right' && 'right-0',
          align === 'center' && 'left-1/2 -translate-x-1/2',
        )}
      >
        {children}
      </span>
    </span>
  );
}
