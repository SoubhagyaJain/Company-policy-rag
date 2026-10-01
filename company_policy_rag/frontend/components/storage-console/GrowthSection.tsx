'use client';

/** Storage over time: one restrained line chart on a single byte axis, the
 *  events that explain its steps, and a forecast once there is enough history. */

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Download, Eraser, FileText, LineChart, Minimize2 } from 'lucide-react';

import { cn, formatBytes, formatDelta, formatDuration } from '@/lib/utils';
import type { StorageEvent, StorageHistory, StorageRange, StorageSeriesPoint } from '@/lib/types';
import { categoryColor } from './Overview';
import { EmptyState } from './primitives';

const RANGES: Array<{ id: StorageRange; label: string }> = [
  { id: '24h', label: '24 hours' },
  { id: '7d', label: '7 days' },
  { id: '30d', label: '30 days' },
];

interface SeriesSpec {
  id: string;
  label: string;
  color: string;
  value: (point: StorageSeriesPoint) => number | null;
}

const TOTAL: SeriesSpec = {
  id: 'total',
  label: 'App data, total',
  color: 'var(--sp-heading)',
  value: (p) => p.app_bytes,
};

const CATEGORY_SERIES: SeriesSpec[] = [
  ['vector_db', 'Vector DB'],
  ['documents', 'Documents'],
  ['caches', 'Caches'],
  ['images', 'Images'],
  ['logs_eval', 'Logs and evaluation'],
  ['telemetry', 'Telemetry'],
  ['models', 'Models'],
].map(([id, label]) => ({
  id,
  label,
  color: categoryColor(id),
  value: (p: StorageSeriesPoint) => (id === 'models' ? p.models_bytes : (p.categories[id] ?? 0)),
}));

const ALL_SERIES = [TOTAL, ...CATEGORY_SERIES];

const EVENT_ICONS: Record<StorageEvent['kind'], React.ReactNode> = {
  document: <FileText className="h-3.5 w-3.5" aria-hidden />,
  cleanup: <Eraser className="h-3.5 w-3.5" aria-hidden />,
  compaction: <Minimize2 className="h-3.5 w-3.5" aria-hidden />,
  model: <Download className="h-3.5 w-3.5" aria-hidden />,
};

const EVENT_WORDS: Record<StorageEvent['kind'], string> = {
  document: 'Document uploaded',
  cleanup: 'Cleanup executed',
  compaction: 'Database compacted',
  model: 'Model downloaded',
};

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)));
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

const timeLabel = (ts: number, range: StorageRange) =>
  new Intl.DateTimeFormat('en-US', range === '24h' ? { hour: '2-digit', minute: '2-digit' } : { month: 'short', day: 'numeric' }).format(ts);

const fullTime = (ts: number) =>
  new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }).format(ts);

/** Round a byte span up to a tick step that reads cleanly in binary units. */
function niceStep(span: number, ticks: number): number {
  const raw = span / ticks;
  const unit = 1024 ** Math.max(0, Math.floor(Math.log(Math.max(raw, 1)) / Math.log(1024)));
  const scaled = raw / unit;
  const step = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1024].find((s) => s >= scaled) ?? 1024;
  return step * unit;
}

const HEIGHT = 220;
const PAD = { top: 12, right: 14, bottom: 30, left: 62 };

interface ChartProps {
  series: StorageSeriesPoint[];
  events: StorageEvent[];
  enabled: SeriesSpec[];
  range: StorageRange;
}

function Chart({ series, events, enabled, range }: ChartProps) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);

  const times = useMemo(() => series.map((p) => new Date(p.ts).getTime()), [series]);
  const lines = useMemo(
    () => enabled.map((spec) => ({ spec, values: series.map((p) => spec.value(p)) })),
    [enabled, series],
  );

  const all = lines.flatMap((l) => l.values).filter((v): v is number => v !== null);
  const innerW = Math.max(0, width - PAD.left - PAD.right);
  const innerH = HEIGHT - PAD.top - PAD.bottom;
  const t0 = times[0];
  const t1 = times[times.length - 1];
  const dataMin = all.length ? Math.min(...all) : 0;
  const dataMax = all.length ? Math.max(...all) : 1;
  const step = niceStep(Math.max(dataMax - dataMin, dataMax * 0.02, 1), 4);
  const yMin = Math.max(0, Math.floor(dataMin / step) * step);
  const yMax = Math.max(yMin + step, Math.ceil(dataMax / step) * step);

  const x = (ts: number) => PAD.left + (t1 === t0 ? innerW / 2 : ((ts - t0) / (t1 - t0)) * innerW);
  const y = (value: number) => PAD.top + innerH - ((value - yMin) / (yMax - yMin)) * innerH;

  const yTicks: number[] = [];
  for (let v = yMin; v <= yMax + 1; v += step) yTicks.push(v);
  const xTickCount = Math.min(5, Math.max(2, Math.floor(innerW / 150)));
  const xTicks = Array.from({ length: xTickCount }, (_, i) => t0 + ((t1 - t0) * i) / (xTickCount - 1));

  const onMove = (event: React.MouseEvent<SVGRectElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const ts = t0 + ((event.clientX - rect.left) / rect.width) * (t1 - t0);
    let nearest = 0;
    for (let i = 1; i < times.length; i += 1) {
      if (Math.abs(times[i] - ts) < Math.abs(times[nearest] - ts)) nearest = i;
    }
    setHover(nearest);
  };

  const inRange = events
    .map((event) => ({ event, ts: new Date(event.ts).getTime() }))
    .filter(({ ts }) => ts >= t0 && ts <= t1);
  const hoverX = hover !== null ? x(times[hover]) : 0;
  const nearEvents = hover !== null ? inRange.filter(({ ts }) => Math.abs(x(ts) - hoverX) < 8) : [];

  return (
    <div ref={ref} className="relative">
      {width > 0 && (
        <svg width={width} height={HEIGHT} role="img" aria-label="Storage size over time">
          {yTicks.map((tick) => (
            <g key={tick}>
              <line className="sc-grid-line" x1={PAD.left} x2={width - PAD.right} y1={y(tick)} y2={y(tick)} />
              <text className="sc-axis-text" x={PAD.left - 8} y={y(tick) + 3} textAnchor="end">
                {formatBytes(tick)}
              </text>
            </g>
          ))}
          {xTicks.map((tick, index) => (
            <text
              key={tick}
              className="sc-axis-text"
              x={x(tick)}
              y={HEIGHT - 8}
              textAnchor={index === 0 ? 'start' : index === xTicks.length - 1 ? 'end' : 'middle'}
            >
              {timeLabel(tick, t1 - t0 < 2 * 86_400_000 ? '24h' : range)}
            </text>
          ))}

          {/* Events: a tick on the baseline where something changed storage. */}
          {inRange.map(({ event, ts }, index) => (
            <line
              key={`${event.ts}-${index}`}
              x1={x(ts)}
              x2={x(ts)}
              y1={PAD.top + innerH - 7}
              y2={PAD.top + innerH}
              stroke="var(--sp-text-muted)"
              strokeWidth={2}
              strokeLinecap="round"
            />
          ))}

          {lines.map(({ spec, values }) => {
            const points = values
              .map((value, index) => (value === null ? null : `${x(times[index])},${y(value)}`))
              .filter((p): p is string => p !== null);
            if (points.length === 0) return null;
            const last = values.reduce<number>((found, value, index) => (value === null ? found : index), -1);
            return (
              <g key={spec.id}>
                <polyline
                  points={points.join(' ')}
                  fill="none"
                  stroke={spec.color}
                  strokeWidth={2}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                />
                {last >= 0 && (
                  <circle
                    cx={x(times[last])}
                    cy={y(values[last] as number)}
                    r={4}
                    fill={spec.color}
                    stroke="var(--sc-surface)"
                    strokeWidth={2}
                  />
                )}
              </g>
            );
          })}

          {hover !== null && (
            <g>
              <line className="sc-crosshair" x1={hoverX} x2={hoverX} y1={PAD.top} y2={PAD.top + innerH} />
              {lines.map(({ spec, values }) =>
                values[hover] === null ? null : (
                  <circle
                    key={spec.id}
                    cx={hoverX}
                    cy={y(values[hover] as number)}
                    r={4}
                    fill={spec.color}
                    stroke="var(--sc-surface)"
                    strokeWidth={2}
                  />
                ),
              )}
            </g>
          )}
          <rect
            x={PAD.left}
            y={PAD.top}
            width={innerW}
            height={innerH}
            fill="transparent"
            onMouseMove={onMove}
            onMouseLeave={() => setHover(null)}
          />
        </svg>
      )}

      {hover !== null && (
        <div
          className="sc-tip absolute top-2 w-60 p-2.5"
          style={{ left: hoverX > width / 2 ? Math.max(PAD.left, hoverX - 252) : hoverX + 12 }}
        >
          <p className="sc-num sc-t3 text-[11px]">{fullTime(times[hover])}</p>
          <ul className="mt-1.5 space-y-1">
            {lines.map(({ spec, values }) => (
              <li key={spec.id} className="flex items-center justify-between gap-3 text-[11.5px]">
                <span className="flex min-w-0 items-center gap-1.5">
                  <span className="h-[3px] w-3 shrink-0 rounded-full" style={{ background: spec.color }} aria-hidden />
                  <span className="sc-t2 truncate">{spec.label}</span>
                </span>
                <span className="sc-num sc-t1 shrink-0">
                  {values[hover] === null ? 'not recorded' : formatBytes(values[hover] as number, 2)}
                </span>
              </li>
            ))}
          </ul>
          {nearEvents.length > 0 && (
            <ul className="sc-rule mt-2 space-y-1 pt-2">
              {nearEvents.map(({ event }, index) => (
                <li key={index} className="sc-t1 text-[11.5px] leading-snug">
                  {event.label}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

interface GrowthSectionProps {
  history: StorageHistory;
  range: StorageRange;
  onRange: (range: StorageRange) => void;
}

export function GrowthSection({ history, range, onRange }: GrowthSectionProps) {
  const [enabledIds, setEnabledIds] = useState<string[]>(['total']);
  const [allEvents, setAllEvents] = useState(false);

  const series = history.series ?? [];
  const events = history.events ?? [];
  const forecast = history.forecast ?? null;
  const interval = history.snapshot_interval_seconds ?? 900;
  // Fixed order, so a series keeps its colour whatever else is toggled.
  const enabled = ALL_SERIES.filter((spec) => enabledIds.includes(spec.id));
  const hasModels = series.some((p) => p.models_bytes !== null);
  const latestFirst = [...events].reverse();
  const shownEvents = allEvents ? latestFirst : latestFirst.slice(0, 6);

  const toggle = (id: string) =>
    setEnabledIds((current) => {
      const next = current.includes(id) ? current.filter((v) => v !== id) : [...current, id];
      return next.length === 0 ? ['total'] : next;
    });

  const first = series[0];
  const last = series[series.length - 1];
  const change = first && last ? last.app_bytes - first.app_bytes : null;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="sp-depth inline-flex rounded-lg p-0.5" role="radiogroup" aria-label="Time range">
          {RANGES.map((option) => (
            <button
              key={option.id}
              type="button"
              role="radio"
              aria-checked={range === option.id}
              onClick={() => onRange(option.id)}
              className="sp-depth-btn rounded-md px-2.5 py-1 text-[11.5px] font-semibold"
            >
              {option.label}
            </button>
          ))}
        </div>
        {change !== null && series.length > 1 && (
          <p className="sc-t2 text-[12px]">
            App data changed by <span className="sc-num sc-t1 font-semibold">{formatDelta(change)}</span> over the recorded part of
            this range
          </p>
        )}
      </div>

      {series.length < 2 ? (
        <EmptyState
          className="sc-panel"
          icon={<LineChart className="h-4 w-4" />}
          title={series.length === 0 ? 'No storage history yet' : 'One snapshot recorded so far'}
          detail={`A size snapshot is recorded every ${formatDuration(interval * 1000)} while the backend runs. The chart appears once this range holds two.`}
        />
      ) : (
        <div className="sc-panel px-3 pb-2 pt-3">
          <div className="mb-1 flex flex-wrap items-center gap-1.5 px-1" role="group" aria-label="Series shown">
            {ALL_SERIES.filter((spec) => spec.id !== 'models' || hasModels).map((spec) => (
              <button
                key={spec.id}
                type="button"
                aria-pressed={enabledIds.includes(spec.id)}
                onClick={() => toggle(spec.id)}
                className="sc-chip flex items-center gap-1.5"
              >
                <span className="h-[3px] w-3 rounded-full" style={{ background: spec.color }} aria-hidden />
                {spec.label}
              </button>
            ))}
          </div>
          <Chart series={series} events={events} enabled={enabled} range={range} />
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-[1.5fr_1fr]">
        <div className="sc-panel">
          <p className="sc-label px-4 pt-3">What changed</p>
          {events.length === 0 ? (
            <p className="sc-t2 px-4 py-3 text-[12.5px]">No uploads, model downloads, cleanups or compactions in this range.</p>
          ) : (
            <>
              <ul className="sc-divide mt-2">
                {shownEvents.map((event, index) => {
                  const measured = event.measured_delta_bytes;
                  const bytes = measured ?? event.bytes;
                  return (
                    <li key={`${event.ts}-${index}`} className="grid grid-cols-[124px_84px_1fr] items-baseline gap-3 px-4 py-2 text-[12.5px]">
                      <span className="sc-num sc-t3 text-[11.5px]">{fullTime(new Date(event.ts).getTime())}</span>
                      <span
                        className="sc-num sc-t1 text-right"
                        title={
                          measured !== null
                            ? 'Change in app data between the snapshots around this event'
                            : event.kind === 'document'
                              ? 'Size of the uploaded file'
                              : event.kind === 'model'
                                ? 'Size of the model'
                                : 'Bytes reclaimed'
                        }
                      >
                        {formatDelta(bytes)}
                      </span>
                      <span className="flex min-w-0 items-center gap-2">
                        <span className="sc-t3 shrink-0" title={EVENT_WORDS[event.kind]}>
                          {EVENT_ICONS[event.kind]}
                        </span>
                        <span className="sc-t1 truncate" title={event.label}>
                          {event.label}
                        </span>
                        {event.detail && <span className="sc-t3 shrink-0 text-[11.5px]">{event.detail}</span>}
                        {event.large && <span className="sc-badge shrink-0">Large ingestion</span>}
                      </span>
                    </li>
                  );
                })}
              </ul>
              {events.length > 6 && (
                <div className="sc-rule px-4 py-2">
                  <button type="button" className="sc-btn sc-btn--ghost" onClick={() => setAllEvents((v) => !v)}>
                    {allEvents ? 'Show fewer' : `Show all ${events.length} events`}
                  </button>
                </div>
              )}
            </>
          )}
        </div>

        <div className="sc-panel px-4 py-3">
          <p className="sc-label">Forecast</p>
          {forecast ? (
            <>
              <p className="sc-t2 mt-1.5 text-[12px]">
                At the current growth rate of{' '}
                <span className="sc-num sc-t1 font-semibold">{formatDelta(forecast.bytes_per_day)}</span> per day:
              </p>
              <dl className="mt-2 grid grid-cols-2 gap-3">
                <div>
                  <dt className="sc-label">In 7 days</dt>
                  <dd className="sc-num sc-t1 mt-1 text-[15px] font-semibold">~{formatBytes(forecast.in_7_days_bytes, 2)}</dd>
                </div>
                <div>
                  <dt className="sc-label">In 30 days</dt>
                  <dd className="sc-num sc-t1 mt-1 text-[15px] font-semibold">~{formatBytes(forecast.in_30_days_bytes, 2)}</dd>
                </div>
              </dl>
              <p className={cn('sc-t3 mt-2 text-[11.5px] leading-relaxed')}>
                Estimate. {forecast.method} Based on {forecast.basis_points} snapshots over {forecast.basis_days} days
                {forecast.disk_free_bytes !== null ? `; ${formatBytes(forecast.disk_free_bytes)} is free on the volume.` : '.'}
              </p>
            </>
          ) : (
            <p className="sc-t2 mt-1.5 text-[12px] leading-relaxed">
              Not shown yet. A projection needs at least 3 days of snapshots
              {history.span_seconds !== undefined ? `; ${formatDuration(history.span_seconds * 1000)} are recorded.` : '.'}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
