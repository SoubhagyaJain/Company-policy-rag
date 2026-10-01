'use client';

/** The top of the console: headline figures, the storage composition bar, and
 *  the concrete findings and health checks the backend measured. */

import React, { useState } from 'react';
import { AlertTriangle, ArrowRight, Check, Info, OctagonAlert } from 'lucide-react';

import { cn, formatBytes, formatPercent } from '@/lib/utils';
import type { StorageMapSegment, StorageStore, StorageSummary } from '@/lib/types';
import { useConsole, type RuntimeState } from './context';
import { Delta, EmptyState, KindBadges, MicroBar, SegmentBar, StatusDot, Unavailable, usageTone } from './primitives';

/** Which section explains each storage-map category. */
export const CATEGORY_SECTION: Record<string, string> = {
  models: 'installed-models',
  documents: 'documents',
  vector_db: 'databases',
  caches: 'caches',
  images: 'documents',
  telemetry: 'databases',
  logs_eval: 'files',
  other: 'files',
};

const STORE_SECTION: Record<string, string> = {
  vector_index: 'databases',
  telemetry_db: 'databases',
  semantic_cache: 'caches',
  vision_cache: 'caches',
  kv_cache: 'caches',
  retrieval_cache: 'caches',
  embedding_cache: 'caches',
  conversations: 'caches',
  page_images: 'documents',
  uploads: 'documents',
  ollama_models: 'gpu',
  vision_model: 'gpu',
  docstore: 'runtime',
};

export const sectionOfStore = (storeId: string | undefined) => (storeId && STORE_SECTION[storeId]) || 'files';

export const categoryColor = (id: string) => `var(--sc-cat-${id})`;

function Cell({ label, children, className }: { label: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={cn('min-w-0 border-l border-t border-[color:var(--sp-hairline)] px-4 py-3.5', className)}>
      <p className="sc-label">{label}</p>
      {children}
    </div>
  );
}

const Figure = ({ children }: { children: React.ReactNode }) => (
  <p className="sc-num sc-t1 mt-1.5 text-[19px] font-semibold leading-none tracking-[-0.02em]">{children}</p>
);

const Sub = ({ children }: { children: React.ReactNode }) => <p className="sc-t2 mt-1.5 text-[11.5px] leading-snug">{children}</p>;

const Note = ({ children }: { children: React.ReactNode }) => <p className="sc-t3 mt-0.5 text-[11px] leading-snug">{children}</p>;

export function OverviewStrip({ summary, runtime }: { summary: StorageSummary; runtime: RuntimeState }) {
  const { totals, disk } = summary;
  const weekly = totals.growth.d7;
  const { processRss, systemTotal, systemAvailable, gpu, loadedModels, inProcessModels, operations } = runtime;
  const ramFraction = processRss !== null && systemTotal ? processRss / systemTotal : null;
  const systemUsed = systemTotal !== null && systemAvailable !== null ? systemTotal - systemAvailable : null;
  const vramFraction = gpu ? gpu.used_mb / gpu.total_mb : null;
  const modelNames = [...loadedModels.map((m) => m.name), ...inProcessModels.map((m) => m.name)];
  const indexing = operations.indexing[0];
  const cleanup = operations.cleanup;
  const compacting = cleanup?.action === 'compact';

  return (
    <div className="sc-panel overflow-hidden">
      {/* Every cell draws its left and top hairline; the outer ones are clipped. */}
      <div className="-ml-px -mt-px grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
      <Cell label="Persistent storage">
        <Figure>{formatBytes(totals.storage_bytes, 2)}</Figure>
        <Sub>
          {weekly === null || weekly === undefined ? (
            <span title="Growth appears once a snapshot at least a week old exists.">No 7-day baseline yet</span>
          ) : (
            <>
              <Delta bytes={weekly} /> this week
            </>
          )}
        </Sub>
        <Note>
          {disk ? `${formatBytes(disk.free_bytes)} free on ${disk.volume}` : `${formatBytes(totals.models_bytes)} of it models`}
        </Note>
      </Cell>

      <Cell label="Safely reclaimable">
        <Figure>{formatBytes(totals.reclaimable_bytes, 2)}</Figure>
        <Sub>
          {totals.disk_bytes > 0 ? `${formatPercent(totals.reclaimable_bytes / totals.disk_bytes, 1)} of app data` : '—'}
        </Sub>
        <Note>Orphans and unused DB pages</Note>
      </Cell>

      <Cell label="Backend RAM">
        {processRss === null ? (
          <p className="mt-2">
            <Unavailable reason="psutil is not installed in the backend environment.">Not measured</Unavailable>
          </p>
        ) : (
          <>
            <Figure>{formatBytes(processRss)}</Figure>
            {ramFraction !== null && systemTotal !== null && (
              <>
                <MicroBar value={processRss} max={systemTotal} className="mt-2" />
                <Sub>
                  {formatPercent(ramFraction)} of {formatBytes(systemTotal)}
                </Sub>
                {systemUsed !== null && <Note>System RAM {formatPercent(systemUsed / systemTotal)} in use</Note>}
              </>
            )}
          </>
        )}
      </Cell>

      <Cell label="GPU VRAM">
        {gpu && vramFraction !== null ? (
          <>
            <Figure>{(gpu.used_mb / 1024).toFixed(1)} GB</Figure>
            <MicroBar value={gpu.used_mb} max={gpu.total_mb} tone={usageTone(vramFraction)} className="mt-2" />
            <Sub>
              {formatPercent(vramFraction)} of {(gpu.total_mb / 1024).toFixed(1)} GB
            </Sub>
            <Note>{gpu.name.replace(/^NVIDIA\s+/i, '')}</Note>
          </>
        ) : (
          <p className="mt-2">
            <Unavailable reason="nvidia-smi did not report a device.">No GPU detected</Unavailable>
          </p>
        )}
      </Cell>

      <Cell label="Loaded models">
        <Figure>{modelNames.length} loaded</Figure>
        {modelNames.length === 0 ? (
          <Sub>Nothing is held in memory</Sub>
        ) : (
          <ul className="mt-1.5 space-y-0.5">
            {modelNames.slice(0, 3).map((name) => (
              <li key={name} className="sc-num sc-t2 truncate text-[11.5px]" title={name}>
                {name}
              </li>
            ))}
            {modelNames.length > 3 && <li className="sc-t3 text-[11px]">+{modelNames.length - 3} more</li>}
          </ul>
        )}
      </Cell>

      <Cell label="Active operations">
        <ul className="mt-2 space-y-1.5 text-[12px]">
          <li className="flex items-center gap-2">
            <StatusDot state={indexing ? 'active' : 'idle'} />
            <span className="sc-t2 w-[74px] shrink-0">Indexing</span>
            <span className="sc-num sc-t1 truncate" title={indexing?.filename}>
              {indexing ? `${indexing.progress}% ${indexing.filename}` : 'idle'}
            </span>
          </li>
          <li className="flex items-center gap-2">
            <StatusDot state={cleanup && !compacting ? 'active' : 'idle'} />
            <span className="sc-t2 w-[74px] shrink-0">Cleanup</span>
            <span className="sc-num sc-t1 truncate">{cleanup && !compacting ? cleanup.store : 'idle'}</span>
          </li>
          <li className="flex items-center gap-2">
            <StatusDot state={compacting ? 'active' : 'idle'} />
            <span className="sc-t2 w-[74px] shrink-0">Compaction</span>
            <span className="sc-num sc-t1 truncate">{compacting ? cleanup?.store : 'idle'}</span>
          </li>
        </ul>
      </Cell>
      </div>
    </div>
  );
}

/* ── Storage map ─────────────────────────────────────────────────────────── */

interface StorageMapProps {
  segments: StorageMapSegment[];
  stores: StorageStore[];
  includeModels: boolean;
  onIncludeModels: (next: boolean) => void;
}

export function StorageMap({ segments, stores, includeModels, onIncludeModels }: StorageMapProps) {
  const { jump, advanced } = useConsole();
  const [hovered, setHovered] = useState<string | null>(null);

  const shown = segments.filter((s) => s.size_bytes > 0 && (includeModels || s.id !== 'models'));
  const total = shown.reduce((sum, s) => sum + s.size_bytes, 0);
  const storeById = new Map(stores.map((s) => [s.id, s]));
  const active = shown.find((s) => s.id === hovered) ?? null;

  // Centre of the hovered segment along the bar, to anchor its card.
  let offset = 0;
  let anchor = 50;
  for (const segment of shown) {
    const width = total > 0 ? (segment.size_bytes / total) * 100 : 0;
    if (segment.id === hovered) anchor = offset + width / 2;
    offset += width;
  }

  if (total === 0) {
    return <EmptyState title="Nothing stored yet" detail="Upload a document and its footprint will appear here." className="sc-panel" />;
  }

  return (
    <div>
      <div className="flex flex-wrap items-end justify-between gap-3">
        <p className="sc-t2 text-[12px]">
          Total local storage{' '}
          <span className="sc-num sc-t1 ml-1 text-[15px] font-semibold">{formatBytes(total, 2)}</span>
        </p>
        <label className="sc-t2 flex cursor-pointer items-center gap-2 text-[12px]">
          <input
            type="checkbox"
            className="sc-check"
            checked={includeModels}
            onChange={(e) => onIncludeModels(e.target.checked)}
          />
          Include model files
        </label>
      </div>

      <div className="relative mt-2.5">
        <SegmentBar
          segments={shown.map((s) => ({ id: s.id, label: s.label, value: s.size_bytes, color: categoryColor(s.id) }))}
          height={22}
          activeId={hovered}
          onHover={setHovered}
          onSelect={(id) => jump(CATEGORY_SECTION[id] ?? 'files')}
          ariaLabel="Storage composition by category"
        />
        {active && (
          <div
            className="sc-tip absolute top-full mt-2 w-72 p-3"
            style={{ left: `clamp(0px, calc(${anchor}% - 144px), calc(100% - 288px))` }}
          >
            <div className="flex items-center justify-between gap-3">
              <span className="sc-t1 flex items-center gap-2 text-[12.5px] font-semibold">
                <span className="h-2 w-2 rounded-[2px]" style={{ background: categoryColor(active.id) }} aria-hidden />
                {active.label}
              </span>
              <span className="sc-num sc-t1 text-[12.5px]">{formatBytes(active.size_bytes, 2)}</span>
            </div>
            <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[11.5px]">
              <dt className="sc-t3">Share of total</dt>
              <dd className="sc-num sc-t1 text-right">{formatPercent(active.size_bytes / total, 1)}</dd>
              <dt className="sc-t3">Growth, 7 days</dt>
              <dd className="text-right">
                <Delta bytes={active.growth_d7} />
              </dd>
              <dt className="sc-t3">Reclaimable</dt>
              <dd className="sc-num sc-t1 text-right">{formatBytes(active.reclaimable_bytes)}</dd>
            </dl>
            <ul className="sc-rule mt-2 space-y-1.5 pt-2">
              {active.members
                .filter((m) => m.size_bytes > 0)
                .slice(0, 4)
                .map((member) => {
                  const store = storeById.get(member.id);
                  return (
                    <li key={member.id} className="text-[11.5px]">
                      <span className="flex items-center justify-between gap-2">
                        <span className="sc-t2 truncate">{member.label}</span>
                        <span className="sc-num sc-t1 shrink-0">{formatBytes(member.size_bytes)}</span>
                      </span>
                      {store && (
                        <span className="mt-0.5 flex items-center gap-1.5">
                          <KindBadges kinds={store.kinds} limit={2} />
                        </span>
                      )}
                      {store?.path && <span className="sc-num sc-t3 mt-0.5 block truncate text-[10.5px]">{store.path}</span>}
                    </li>
                  );
                })}
            </ul>
          </div>
        )}
      </div>

      {/* The legend doubles as the table view: every figure is readable without the colours. */}
      <ul className="mt-3 grid grid-cols-1 gap-x-8 md:grid-cols-2">
        {shown.map((segment) => (
          <li key={segment.id}>
            <button
              type="button"
              onClick={() => jump(CATEGORY_SECTION[segment.id] ?? 'files')}
              onMouseEnter={() => setHovered(segment.id)}
              onMouseLeave={() => setHovered(null)}
              className="sc-row sc-focus grid w-full grid-cols-[10px_1fr_76px_44px_84px] items-center gap-3 rounded px-1.5 py-1.5 text-left text-[12px]"
            >
              <span className="h-2.5 w-2.5 rounded-[2px]" style={{ background: categoryColor(segment.id) }} aria-hidden />
              <span className="sc-t1 truncate font-medium">{segment.label}</span>
              <span className="sc-num sc-t1 text-right">{formatBytes(segment.size_bytes)}</span>
              <span className="sc-num sc-t3 text-right">{formatPercent(segment.size_bytes / total)}</span>
              <span className="text-right text-[11.5px]" title="Change over the last 7 days">
                <Delta bytes={segment.growth_d7} />
              </span>
            </button>
          </li>
        ))}
      </ul>
      {advanced && (
        <p className="sc-t3 mt-1.5 px-1.5 text-[11px]">
          Columns: size · share · change over 7 days. Select a category to open its section.
        </p>
      )}
    </div>
  );
}

/* ── Observations and health ─────────────────────────────────────────────── */

export function Observations({ summary }: { summary: StorageSummary }) {
  const { jump } = useConsole();
  const { observations, health } = summary;

  return (
    <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
      <div className="sc-panel">
        <p className="sc-label px-4 pt-3">Storage observations</p>
        {observations.length === 0 ? (
          <EmptyState
            icon={<Check className="h-4 w-4" />}
            title="Nothing notable"
            detail="No unusual growth, orphaned data, fragmentation or memory pressure was measured."
          />
        ) : (
          <ul className="sc-divide mt-2">
            {observations.map((item) => (
              <li key={item.id} className="flex items-start gap-2.5 px-4 py-2.5">
                {item.level === 'critical' ? (
                  <OctagonAlert className="sc-crit mt-0.5 h-3.5 w-3.5 shrink-0" aria-label="Critical" />
                ) : item.level === 'warn' ? (
                  <AlertTriangle className="sc-warn mt-0.5 h-3.5 w-3.5 shrink-0" aria-label="Warning" />
                ) : (
                  <Info className="sc-t3 mt-0.5 h-3.5 w-3.5 shrink-0" aria-label="Note" />
                )}
                <p className="sc-t1 min-w-0 flex-1 text-[12.5px] leading-relaxed">{item.text}</p>
                <button
                  type="button"
                  onClick={() => jump(sectionOfStore(item.store_id))}
                  className="sc-btn sc-btn--ghost -my-1 shrink-0"
                  aria-label="Show the section this refers to"
                >
                  View
                  <ArrowRight className="h-3 w-3" aria-hidden />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="sc-panel px-4 py-3">
        <p className="sc-label">Storage health</p>
        <ul className="mt-2 space-y-1.5">
          {health.map((check) => (
            <li key={check.id} className="flex items-start gap-2 text-[12.5px]">
              {check.ok ? (
                <Check className="sc-ok mt-0.5 h-3.5 w-3.5 shrink-0" aria-label="OK" />
              ) : (
                <AlertTriangle className="sc-warn mt-0.5 h-3.5 w-3.5 shrink-0" aria-label="Attention" />
              )}
              <span className={check.ok ? 'sc-t2' : 'sc-t1'}>{check.label}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
