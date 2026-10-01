'use client';

/** What lives on disk: the two databases, the caches, the indexes and loose
 *  files, and the model files. */

import React, { useState } from 'react';
import { ChevronRight, Database, HardDrive, Search as SearchIcon } from 'lucide-react';

import { cn, formatBytes, formatDate, formatDuration, formatPercent, formatRelativeTime } from '@/lib/utils';
import type { StorageCache, StorageHistory, StorageModelEntry, StorageStore } from '@/lib/types';
import { ActionButtons, useConsole, useRowClass, useVisible, type RuntimeState } from './context';
import { cacheItem, installedModelItem, storeItem } from './filtering';
import {
  BarLegend,
  Delta,
  EmptyState,
  Estimated,
  Field,
  InfoButton,
  KindBadges,
  MicroBar,
  SegmentBar,
  Sparkline,
  StatusDot,
  Unavailable,
  type BarSegment,
} from './primitives';

const MIB = 1024 * 1024;
// Rows of one list share the list's column tracks, so figures line up whatever the action buttons need.
const SUBGRID_ROW = 'lg:col-span-full lg:grid lg:grid-cols-subgrid';
const FRAGMENTATION_NOTE_PCT = 20;

const count = (value: number | null | undefined) =>
  value === null || value === undefined ? <Unavailable>—</Unavailable> : value.toLocaleString();

/* ── Databases ───────────────────────────────────────────────────────────── */

function databaseSegments(db: StorageStore): BarSegment[] {
  const d = db.details;
  const free = d.free_bytes ?? 0;
  if (db.id === 'vector_index') {
    const sqlite = d.sqlite_bytes ?? d.file_bytes ?? 0;
    return [
      { id: 'used', label: 'SQLite, used', value: Math.max(0, sqlite - free), color: 'var(--sp-accent)' },
      { id: 'unused', label: 'SQLite, unused', value: free, color: 'var(--sp-warn)' },
      { id: 'other', label: 'HNSW vector segments', value: d.hnsw_bytes ?? 0, color: 'var(--sp-text-faint)' },
    ];
  }
  const wal = d.wal_bytes ?? 0;
  return [
    { id: 'used', label: 'Used', value: Math.max(0, (d.file_bytes ?? db.size_bytes) - free - wal), color: 'var(--sp-accent)' },
    { id: 'unused', label: 'Unused pages', value: free, color: 'var(--sp-warn)' },
    { id: 'other', label: 'Write-ahead log', value: wal, color: 'var(--sp-text-faint)' },
  ];
}

function DatabaseCard({ db, history }: { db: StorageStore; history: StorageHistory }) {
  const { advanced, inspect, explain, now, searching } = useConsole();
  const d = db.details;
  const readable = d.page_count !== undefined;
  const bloat = d.bloat_pct ?? 0;
  const free = d.free_bytes ?? 0;
  const segments = databaseSegments(db);
  const trend = history.snapshots.map((s) => s.sizes[db.id]).filter((v): v is number => typeof v === 'number');

  return (
    <article className={cn('sc-panel flex flex-col', searching && 'sc-match')}>
      <header className="flex items-start justify-between gap-3 px-4 pt-3.5">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <h3 className="sc-t1 truncate text-[13.5px] font-semibold">{db.label}</h3>
            <InfoButton label={db.label} onClick={() => explain(db.id)} />
          </div>
          <p className="sc-t3 mt-0.5 text-[11.5px]">{d.engine}</p>
          <div className="mt-1.5">
            <KindBadges kinds={db.kinds} />
          </div>
        </div>
        <div className="shrink-0 text-right">
          <p className="sc-num sc-t1 text-[19px] font-semibold leading-none tracking-[-0.02em]">{formatBytes(db.size_bytes)}</p>
          <p className="sc-t3 mt-1 text-[11px]">allocated on disk</p>
        </div>
      </header>

      <div className="px-4 pt-3">
        {readable ? (
          <>
            <SegmentBar segments={segments} ariaLabel={`${db.label}: used, unused and other space`} />
            <BarLegend segments={segments} className="mt-2" />
          </>
        ) : (
          <p className="sp-warn rounded-md px-3 py-2 text-[12px]">
            The database file could not be read, so its internals are unavailable.
          </p>
        )}
        {readable && bloat >= FRAGMENTATION_NOTE_PCT && free >= MIB && (
          <p className="sc-warn mt-2 text-[12px]">{formatBytes(free)} may be recoverable through compaction.</p>
        )}
      </div>

      <dl className="grid grid-cols-3 gap-x-4 gap-y-3 px-4 pt-3.5 xl:grid-cols-4">
        {db.id === 'vector_index' ? (
          <>
            <Field label="Collections">{count(d.tables?.length)}</Field>
            <Field label="Chunks">{count(d.chunk_rows)}</Field>
            <Field label="Embeddings">{count(d.embedding_rows)}</Field>
            <Field label="Pending writes" title="Rows in the write log not yet purged">
              {count(d.queue_rows)}
            </Field>
          </>
        ) : (
          <>
            <Field label="Rows">{count(db.items)}</Field>
            <Field label="Tables">{count(d.tables?.length)}</Field>
            <Field label="Unused pages">{count(d.free_pages)}</Field>
            <Field label="WAL file">{d.wal_bytes !== undefined ? formatBytes(d.wal_bytes) : <Unavailable>—</Unavailable>}</Field>
          </>
        )}
        <Field label="Fragmentation">{readable ? `${bloat}%` : <Unavailable>—</Unavailable>}</Field>
        <Field label="Last write" title={db.last_modified ? formatDate(db.last_modified) : undefined}>
          {formatRelativeTime(db.last_modified, now)}
        </Field>
        <Field label="Last compaction" title={d.last_compaction ? formatDate(d.last_compaction) : undefined}>
          {d.last_compaction ? formatRelativeTime(d.last_compaction, now) : <Unavailable reason="No compaction in the recorded history">Not recorded</Unavailable>}
        </Field>
        {db.id === 'telemetry_db' ? (
          <Field label="Records span" title={`${d.oldest_record ?? '—'} → ${d.newest_record ?? '—'}`}>
            {d.oldest_record ? `${formatRelativeTime(d.oldest_record, now).replace(' ago', '')} → ${formatRelativeTime(d.newest_record, now)}` : <Unavailable>No records</Unavailable>}
          </Field>
        ) : (
          <Field label="Used">{readable ? formatBytes(Math.max(0, db.size_bytes - free)) : <Unavailable>—</Unavailable>}</Field>
        )}
      </dl>

      <div className="sc-rule mt-3.5 flex items-center justify-between gap-3 px-4 py-2.5">
        <dl className="flex items-center gap-4 text-[11.5px]">
          {(['d1', 'd7', 'd30'] as const).map((window) => (
            <div key={window} className="flex items-baseline gap-1.5">
              <dt className="sc-t3">{window === 'd1' ? '24h' : window === 'd7' ? '7d' : '30d'}</dt>
              <dd>
                <Delta bytes={db.growth?.[window]} />
              </dd>
            </div>
          ))}
        </dl>
        {advanced && <Sparkline points={trend.slice(-60)} />}
      </div>

      {advanced && db.path && (
        <p className="sc-num sc-t3 truncate px-4 pb-2 text-[10.5px]" title={db.path}>
          {db.path}
        </p>
      )}

      <footer className="sc-rule mt-auto flex flex-wrap items-center justify-end gap-1.5 px-4 py-2.5">
        <button type="button" className="sc-btn" onClick={() => inspect(db.id)}>
          <SearchIcon className="h-3 w-3" aria-hidden />
          Inspect
        </button>
        <ActionButtons storeId={db.id} storeLabel={db.label} actions={db.actions} />
      </footer>
    </article>
  );
}

export function DatabasesSection({ stores, history }: { stores: StorageStore[]; history: StorageHistory }) {
  const visible = useVisible(
    stores.filter((s) => s.group === 'databases'),
    storeItem,
  );
  if (visible.length === 0) return null;
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {visible.map((db) => (
        <DatabaseCard key={db.id} db={db} history={history} />
      ))}
    </div>
  );
}

/* ── Caches ──────────────────────────────────────────────────────────────── */

function cacheValue(cache: StorageCache): React.ReactNode {
  if (cache.id === 'conversations') return <span className="sc-t3">Session state, not a lookup cache</span>;
  if (cache.hits === null || cache.misses === null) return <Unavailable>Hit-rate telemetry unavailable</Unavailable>;
  if (cache.hit_rate === null) return <span className="sc-t3">No lookups {cache.hit_rate_window ?? ''}</span>;
  return (
    <span className="sc-t2">
      {(cache.hits ?? 0).toLocaleString()} of {((cache.hits ?? 0) + (cache.misses ?? 0)).toLocaleString()} lookups
      {cache.avg_saved_ms ? ` · saves ~${formatDuration(cache.avg_saved_ms)} per hit` : ''}
    </span>
  );
}

function CacheRow({ cache }: { cache: StorageCache }) {
  const { advanced, explain, now } = useConsole();
  const rowClass = useRowClass();
  const [open, setOpen] = useState(false);
  const fallback = cache.id === 'kv_cache' && cache.backend !== 'redis';

  return (
    <li className={rowClass(cn(SUBGRID_ROW, open && 'is-open'))}>
      {/* One line on a desktop; below that the name takes its own line and the figures wrap under it. */}
      <div className={cn('grid grid-cols-4 items-center gap-x-3 gap-y-2 px-4 py-2.5', SUBGRID_ROW)}>
        <div className="col-span-4 min-w-0 lg:col-span-1">
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => setOpen((v) => !v)}
              aria-expanded={open}
              aria-label={`${open ? 'Hide' : 'Show'} details for ${cache.label}`}
              className="sc-icon-btn -ml-1"
            >
              <ChevronRight className={cn('h-3.5 w-3.5 transition-transform', open && 'rotate-90')} aria-hidden />
            </button>
            <span className="sc-t1 truncate text-[12.5px] font-semibold">{cache.label}</span>
            <InfoButton label={cache.label} onClick={() => explain(cache.id)} />
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-1.5 pl-5">
            <KindBadges kinds={cache.kinds} limit={3} />
            {cache.id === 'kv_cache' && (
              <span className="flex items-center gap-1.5 text-[11px]">
                <StatusDot state={fallback ? 'warn' : 'ok'} />
                <span className={fallback ? 'sc-warn' : 'sc-t2'}>{fallback ? 'In-memory fallback' : 'Redis · connected'}</span>
              </span>
            )}
          </div>
        </div>

        <div className="lg:text-right">
          <p className="sc-num sc-t1 text-[12.5px]">{cache.entries.toLocaleString()}</p>
          <p className="sc-t3 text-[11px]">{cache.entries_label}</p>
        </div>

        <div className="lg:text-right" title={cache.size_note ?? undefined}>
          {cache.size_bytes === null ? (
            <Unavailable reason={cache.size_note ?? undefined}>Not measured</Unavailable>
          ) : (
            <p className="sc-num sc-t1 text-[12.5px]">
              {cache.size_estimated ? <Estimated>{formatBytes(cache.size_bytes)}</Estimated> : formatBytes(cache.size_bytes)}
            </p>
          )}
          <p className="sc-t3 text-[11px]">{cache.location === 'disk' ? 'on disk' : cache.location === 'redis' ? 'in Redis' : 'in RAM'}</p>
        </div>

        <div className="col-span-2 min-w-0 lg:col-span-1">
          {cache.hit_rate !== null ? (
            <div className="flex items-center gap-2">
              <span className="sc-num sc-t1 w-9 shrink-0 text-[12.5px]">{formatPercent(cache.hit_rate)}</span>
              <MicroBar value={cache.hit_rate} max={1} tone="ok" className="max-w-[120px]" />
              <span className="sc-t3 shrink-0 text-[11px]">hit rate</span>
            </div>
          ) : null}
          <p className="truncate text-[11.5px]" title={cache.hit_rate_window ?? undefined}>
            {cacheValue(cache)}
          </p>
        </div>

        <div className="lg:text-right">
          <p className="sc-num sc-t1 text-[12px]">{formatRelativeTime(cache.last_hit_at, now)}</p>
          <p className="sc-t3 text-[11px]">last hit</p>
        </div>

        <ActionButtons
          storeId={cache.id}
          storeLabel={cache.label}
          actions={cache.actions}
          className="col-span-3 justify-self-end lg:col-span-1"
        />
      </div>

      {open && (
        <div className="px-4 pb-3 pl-9 lg:col-span-full">
          {fallback && <p className="sc-warn mb-2 text-[12px]">Data will be lost when the server restarts.</p>}
          <dl className="grid grid-cols-2 gap-x-4 gap-y-3 md:grid-cols-4 xl:grid-cols-6">
            <Field label="TTL">{cache.ttl_seconds ? formatDuration(cache.ttl_seconds * 1000) : <Unavailable>None</Unavailable>}</Field>
            <Field label="Entry limit">{count(cache.max_entries)}</Field>
            <Field label="Oldest entry" title={cache.oldest_entry ?? undefined}>
              {cache.oldest_entry ? formatRelativeTime(cache.oldest_entry, now) : <Unavailable>Not recorded</Unavailable>}
            </Field>
            <Field label="Newest entry" title={cache.newest_entry ?? undefined}>
              {cache.newest_entry ? formatRelativeTime(cache.newest_entry, now) : <Unavailable>Not recorded</Unavailable>}
            </Field>
            <Field label="Reclaimable">{formatBytes(cache.reclaimable_bytes)}</Field>
            <Field label="Measured" title={cache.hit_rate_window ?? undefined}>
              {cache.hit_rate_window ?? <Unavailable>—</Unavailable>}
            </Field>
          </dl>
          <p className="sc-t2 mt-3 text-[12px]">
            <span className="sc-t3">Rebuild cost: </span>
            {cache.rebuild_cost ?? '—'}
          </p>
          {advanced && cache.size_note && <p className="sc-t3 mt-1 text-[11.5px]">{cache.size_note}</p>}
        </div>
      )}
    </li>
  );
}

function KvExplainer() {
  return (
    <div className="sc-panel mt-3 grid gap-px overflow-hidden md:grid-cols-2">
      <div className="px-4 py-3">
        <p className="sc-label">Application KV cache</p>
        <p className="sc-t2 mt-1.5 text-[12px] leading-relaxed">
          Redis, or an in-memory key-value store when Redis is not running. It caches application data: query responses,
          embeddings and sessions. Clearing it removes those saved entries.
        </p>
      </div>
      <div className="sc-rule px-4 py-3 md:border-l md:border-t-0 md:border-[color:var(--sp-hairline)]">
        <p className="sc-label">LLM attention KV cache</p>
        <p className="sc-t2 mt-1.5 text-[12px] leading-relaxed">
          Key and value tensors a transformer keeps while it generates. The model runtime manages them and releases them
          when a request or the model is unloaded. Its size is not exposed by the runtime; it is part of each loaded
          model&apos;s VRAM figure, and is never combined with the application cache above.
        </p>
      </div>
    </div>
  );
}

export function CachesSection({ caches, runtime }: { caches: StorageCache[]; runtime: RuntimeState }) {
  // Counters move between summaries; overlay the live poll on the in-memory caches.
  const merged = caches.map((cache) => {
    const live = runtime.caches[cache.id];
    if (!live) return cache;
    if (live.hits === null || live.misses === null) return { ...cache, entries: live.entries };
    const lookups = live.hits + live.misses;
    return {
      ...cache,
      entries: live.entries,
      hits: live.hits,
      misses: live.misses,
      hit_rate: lookups > 0 ? live.hits / lookups : null,
      last_hit_at: live.last_hit_at ?? cache.last_hit_at,
      backend: live.backend ?? cache.backend,
    };
  });
  const visible = useVisible(merged, cacheItem);
  const [explain, setExplain] = useState(false);
  if (visible.length === 0) return null;

  return (
    <div>
      <ul className="sc-panel sc-divide lg:grid lg:grid-cols-[minmax(0,1.5fr)_84px_92px_minmax(0,1.6fr)_74px_auto] lg:gap-x-3">
        {visible.map((cache) => (
          <CacheRow key={cache.id} cache={cache} />
        ))}
      </ul>
      <button type="button" onClick={() => setExplain((v) => !v)} aria-expanded={explain} className="sc-btn sc-btn--ghost mt-2">
        <ChevronRight className={cn('h-3 w-3 transition-transform', explain && 'rotate-90')} aria-hidden />
        Two different &ldquo;KV caches&rdquo;
      </button>
      {explain && <KvExplainer />}
    </div>
  );
}

/* ── Indexes and files ───────────────────────────────────────────────────── */

const FILE_STORE_ORDER = ['bm25_index', 'eval_artifacts', 'legacy_index', 'logs', 'session_libraries', 'frontend_build'];

export function FilesSection({ stores }: { stores: StorageStore[] }) {
  const { explain, now, advanced } = useConsole();
  const rowClass = useRowClass();
  const rows = FILE_STORE_ORDER.map((id) => stores.find((s) => s.id === id)).filter((s): s is StorageStore => Boolean(s));
  const visible = useVisible(rows, storeItem);
  if (visible.length === 0) return null;

  return (
    <ul className="sc-panel sc-divide lg:grid lg:grid-cols-[minmax(0,1.8fr)_96px_88px_84px_84px_auto] lg:gap-x-3">
      {visible.map((store) => (
        <li key={store.id} className={rowClass(SUBGRID_ROW)}>
          <div className={cn('grid grid-cols-4 items-center gap-x-3 gap-y-2 px-4 py-2.5', SUBGRID_ROW)}>
            <div className="col-span-4 min-w-0 lg:col-span-1">
              <div className="flex items-center gap-1.5">
                <span className="sc-t1 truncate text-[12.5px] font-semibold">{store.label}</span>
                <InfoButton label={store.label} onClick={() => explain(store.id)} />
              </div>
              <div className="mt-1">
                <KindBadges kinds={store.kinds} limit={3} />
              </div>
              {advanced && store.path && (
                <p className="sc-num sc-t3 mt-1 truncate text-[10.5px]" title={store.path}>
                  {store.path}
                </p>
              )}
            </div>
            <div className="lg:text-right">
              <p className="sc-num sc-t1 text-[12.5px]">{store.items !== null ? store.items.toLocaleString() : '—'}</p>
              <p className="sc-t3 text-[11px]">{store.items_label}</p>
            </div>
            <p className="sc-num sc-t1 text-[12.5px] lg:text-right">{formatBytes(store.size_bytes)}</p>
            <p className="text-[11.5px] lg:text-right" title="Change over the last 7 days">
              <Delta bytes={store.growth?.d7} />
            </p>
            <p className="sc-num sc-t2 text-[11.5px] lg:text-right" title={store.last_modified ?? undefined}>
              {formatRelativeTime(store.last_modified, now)}
            </p>
            <ActionButtons
              storeId={store.id}
              storeLabel={store.label}
              actions={store.actions}
              className="col-span-4 justify-self-end lg:col-span-1"
            />
          </div>
        </li>
      ))}
    </ul>
  );
}

/* ── Installed models ────────────────────────────────────────────────────── */

export function InstalledModelsSection({ models }: { models: StorageModelEntry[] }) {
  const { now, advanced } = useConsole();
  const rowClass = useRowClass();
  const sorted = [...models].sort((a, b) => b.size_bytes - a.size_bytes);
  const visible = useVisible(sorted, installedModelItem);

  if (models.length === 0) {
    return (
      <EmptyState
        className="sc-panel"
        icon={<HardDrive className="h-4 w-4" />}
        title="No model files found"
        detail="Ollama did not report any installed models, and the embedding, reranker and vision model directories are empty."
      />
    );
  }
  if (visible.length === 0) return null;
  const largest = sorted[0]?.size_bytes ?? 1;

  return (
    <div className="sc-panel overflow-hidden">
      <table className="w-full border-collapse text-[12.5px]">
        <thead>
          <tr>
            <th className="sc-th px-4 py-2">Model</th>
            <th className="sc-th px-3 py-2">Purpose</th>
            <th className="sc-th px-3 py-2">Runtime</th>
            <th className="sc-th px-3 py-2 text-right">On disk</th>
            <th className="sc-th px-3 py-2">State</th>
            <th className="sc-th px-4 py-2 text-right">Last used</th>
          </tr>
        </thead>
        <tbody className="sc-divide">
          {visible.map((model) => (
            <tr key={model.id} className={cn(rowClass(), 'border-t border-[color:var(--sp-hairline)]')}>
              <td className="px-4 py-2">
                <span className="sc-num sc-t1 font-medium">{model.name}</span>
                {advanced && (model.parameter_size || model.quantization) && (
                  <span className="sc-t3 ml-2 text-[11px]">
                    {[model.parameter_size, model.quantization].filter(Boolean).join(' · ')}
                  </span>
                )}
                {advanced && model.path && (
                  <span className="sc-num sc-t3 block truncate text-[10.5px]" title={model.path}>
                    {model.path}
                  </span>
                )}
              </td>
              <td className="sc-t2 px-3 py-2">{model.purpose}</td>
              <td className="sc-t2 px-3 py-2">{model.runtime}</td>
              <td className="px-3 py-2">
                <div className="flex items-center justify-end gap-2">
                  <MicroBar value={model.size_bytes} max={largest} tone="neutral" className="w-14" />
                  <span className="sc-num sc-t1 w-[68px] text-right">{formatBytes(model.size_bytes)}</span>
                </div>
              </td>
              <td className="px-3 py-2">
                <span className="flex items-center gap-1.5">
                  <StatusDot state={model.loaded ? 'ok' : 'idle'} />
                  <span className={model.loaded ? 'sc-t1' : 'sc-t3'}>{model.loaded ? 'Loaded' : 'Not loaded'}</span>
                </span>
              </td>
              <td className="px-4 py-2 text-right">
                {model.last_used ? (
                  <span className="sc-num sc-t2" title={`${model.last_used} (from telemetry)`}>
                    {formatRelativeTime(model.last_used, now)}
                  </span>
                ) : (
                  <Unavailable reason="No request for this model is recorded in telemetry.">Not recorded</Unavailable>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="sc-t3 sc-rule flex items-center gap-1.5 px-4 py-2 text-[11.5px]">
        <Database className="h-3 w-3" aria-hidden />
        Read-only. Remove a model with <span className="sc-num">ollama rm</span> or by deleting its directory.
      </p>
    </div>
  );
}
