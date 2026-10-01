'use client';

/** StorageView — what the app keeps on disk and in memory, and the cleanups for it.
 *  Every number and every available action comes from GET /api/admin/storage; this
 *  view only renders them and asks for confirmation before anything is removed. */

import React, { useMemo, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Archive,
  Box,
  CheckCircle2,
  Cpu,
  Database,
  Eye,
  FileText,
  FlaskConical,
  HardDrive,
  History,
  Image as ImageIcon,
  KeyRound,
  Layers,
  MessageSquare,
  RefreshCw,
  ScrollText,
  Search,
  Sparkles,
  Upload,
  X,
} from 'lucide-react';

import { LiquidGlassCard } from '@/components/LiquidGlassCard';
import { ConfirmDialog } from '@/components/ui/ConfirmDialog';
import { useStorage } from '@/hooks/useStorage';
import { useSmoothScroll } from '@/hooks/useSmoothScroll';
import { formatBytes, formatDate } from '@/lib/utils';
import type {
  StorageActionMeta,
  StorageHistory,
  StorageMemoryItem,
  StorageStore,
} from '@/lib/types';

const STORE_ICONS: Record<string, React.ReactNode> = {
  vector_index: <Database className="w-4 h-4 text-sky-500" />,
  telemetry_db: <Activity className="w-4 h-4 text-violet-500" />,
  semantic_cache: <Sparkles className="w-4 h-4 text-amber-500" />,
  page_images: <ImageIcon className="w-4 h-4 text-emerald-500" />,
  vision_cache: <Eye className="w-4 h-4 text-sky-500" />,
  logs: <ScrollText className="w-4 h-4 text-terracotta-600" />,
  eval_artifacts: <FlaskConical className="w-4 h-4 text-violet-500" />,
  session_libraries: <Archive className="w-4 h-4 text-amber-500" />,
  uploads: <Upload className="w-4 h-4 text-terracotta-600" />,
  bm25_index: <Search className="w-4 h-4 text-emerald-500" />,
};

const MEMORY_ICONS: Record<string, React.ReactNode> = {
  ollama_models: <Cpu className="w-4 h-4 text-sky-500" />,
  vision_model: <Eye className="w-4 h-4 text-emerald-500" />,
  kv_cache: <KeyRound className="w-4 h-4 text-amber-500" />,
  retrieval_cache: <Search className="w-4 h-4 text-violet-500" />,
  embedding_cache: <Layers className="w-4 h-4 text-sky-500" />,
  conversations: <MessageSquare className="w-4 h-4 text-terracotta-600" />,
  docstore: <FileText className="w-4 h-4 text-emerald-500" />,
};

const TEXT = 'text-charcoal dark:text-cream-100';
const MUTED = 'text-charcoal-muted dark:text-cream-400';
const ACTION_BTN =
  'px-2.5 py-1 rounded-lg text-[11px] font-semibold border transition-colors disabled:opacity-50 disabled:cursor-not-allowed';
const SAFE_BTN = `${ACTION_BTN} bg-emerald-500/10 border-emerald-500/25 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-500/20`;
const DANGER_BTN = `${ACTION_BTN} bg-rose-500/10 border-rose-500/25 text-rose-700 dark:text-rose-300 hover:bg-rose-500/20`;

interface PendingAction {
  storeId: string;
  storeLabel: string;
  action: StorageActionMeta;
  target?: string;
  reclaimable?: number;
}

/* Size of one store over time, drawn from the recorded snapshots. */
function Sparkline({ history, storeId }: { history: StorageHistory; storeId: string }) {
  const points = history.snapshots
    .map((s) => s.sizes[storeId])
    .filter((v): v is number => typeof v === 'number')
    .slice(-40);
  if (points.length < 2) {
    return <span className={`text-[10px] ${MUTED}`}>History starts with this visit</span>;
  }
  const max = Math.max(...points);
  const min = Math.min(...points);
  const span = max - min || 1;
  const path = points
    .map((v, i) => `${(i / (points.length - 1)) * 100},${26 - ((v - min) / span) * 22}`)
    .join(' ');
  return (
    <svg viewBox="0 0 100 28" preserveAspectRatio="none" className="h-7 w-full" aria-hidden>
      <polyline points={path} fill="none" stroke="currentColor" strokeWidth="1.5" vectorEffect="non-scaling-stroke" className="text-sky-500" />
    </svg>
  );
}

function Bar({ value, max, tone = 'bg-sky-500' }: { value: number; max: number; tone?: string }) {
  const pct = max > 0 ? Math.min(100, Math.max(value > 0 ? 2 : 0, (value / max) * 100)) : 0;
  return (
    <div className="h-1.5 w-full rounded-full bg-cream-200 dark:bg-sand-dark overflow-hidden">
      <div className={`h-full rounded-full ${tone}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

function SectionTitle({ icon, title, hint }: { icon: React.ReactNode; title: string; hint?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <h2 className={`flex items-center gap-2 text-sm font-bold ${TEXT}`}>
        {icon}
        {title}
      </h2>
      {hint && <span className={`text-[11px] ${MUTED}`}>{hint}</span>}
    </div>
  );
}

export function StorageView() {
  const { summary, history, loading, error, notice, running, refresh, runAction, cleanSafe, dismissNotice } = useStorage();
  const scrollRef = useRef<HTMLDivElement>(null);
  useSmoothScroll(scrollRef);

  const [pending, setPending] = useState<PendingAction | null>(null);
  const [confirmCleanup, setConfirmCleanup] = useState(false);
  const [days, setDays] = useState(30);
  const [loadTarget, setLoadTarget] = useState('');

  const stores = summary?.stores ?? [];
  const databases = stores.filter((s) => s.group === 'databases');
  const caches = stores.filter((s) => s.group === 'caches');
  const library = stores.filter((s) => s.group === 'library');
  const largestCache = Math.max(1, ...caches.map((s) => s.size_bytes));
  const labels = useMemo(() => {
    const map: Record<string, string> = {};
    for (const s of summary?.stores ?? []) map[s.id] = s.label;
    for (const m of summary?.memory.items ?? []) map[m.id] = m.label;
    return map;
  }, [summary]);

  // Chat models that can be loaded. The server lists only models installed on
  // this machine; embedding-only models cannot be loaded for chat.
  const loadable = (summary?.models ?? [])
    .filter((m) => m.id.startsWith('ollama:') && !/embed/i.test(m.id))
    .map((m) => m.id.slice('ollama:'.length));
  const chosenTarget = loadTarget || loadable[0] || '';

  const gpu = summary?.memory.gpu ?? null;
  const safeCount = stores.reduce((n, s) => n + s.actions.filter((a) => a.safe).length, 0);

  const ask = (store: StorageStore | StorageMemoryItem, action: StorageActionMeta, target?: string) => {
    setDays(30);
    setPending({
      storeId: store.id,
      storeLabel: store.label,
      action,
      target,
      reclaimable: 'reclaimable_bytes' in store && action.safe ? store.reclaimable_bytes : undefined,
    });
  };

  const confirmPending = async () => {
    if (!pending) return;
    await runAction(pending.storeId, pending.action.id, {
      olderThanDays: pending.action.needs_days ? days : undefined,
      target: pending.target,
    });
    setPending(null);
  };

  const actionButtons = (store: StorageStore | StorageMemoryItem, target?: string) =>
    store.actions
      .filter((a) => !a.needs_target || target)
      .map((a) => {
        const key = `${store.id}/${a.id}${target && a.needs_target ? `/${target}` : ''}`;
        return (
          <button
            key={a.id}
            type="button"
            onClick={() => ask(store, a, a.needs_target ? target : undefined)}
            disabled={running !== null}
            title={a.description}
            className={a.safe ? SAFE_BTN : DANGER_BTN}
          >
            {running === key ? 'Working…' : a.label}
          </button>
        );
      });

  return (
    <div ref={scrollRef} className="flex-1 h-full overflow-y-auto p-4 sm:p-6 lg:p-8 sp-scroll">
      <div className="max-w-5xl mx-auto space-y-6">
        {/* ── Heading ─────────────────────────────── */}
        <div className="flex items-center justify-between gap-3">
          <div>
            <h1 className="sp-display text-[32px] leading-tight">Storage</h1>
            <p className="sp-muted text-sm mt-0.5">
              Everything the app keeps on disk and in memory, and what can be cleared.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setConfirmCleanup(true)}
              disabled={running !== null || !summary || summary.totals.reclaimable_bytes === 0}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-xs font-semibold text-emerald-800 dark:text-emerald-300 hover:bg-emerald-500/20 transition-colors disabled:opacity-50"
              title="Remove orphaned data and compact the databases. Nothing the app still uses is deleted."
            >
              <Sparkles className="w-3.5 h-3.5" />
              {running === 'cleanup'
                ? 'Cleaning…'
                : `Clean up ${formatBytes(summary?.totals.reclaimable_bytes ?? 0)}`}
            </button>
            <button
              type="button"
              onClick={() => void refresh()}
              disabled={loading}
              className="sp-chip flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>

        {/* ── Banners ─────────────────────────────── */}
        {error && (
          <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-sm text-rose-700 dark:text-rose-400 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}
        {notice && (
          <div className="p-3.5 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-sm text-emerald-800 dark:text-emerald-300 flex items-center justify-between gap-2">
            <span className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 shrink-0" />
              {notice}
            </span>
            <button type="button" onClick={dismissNotice} aria-label="Dismiss" className="opacity-70 hover:opacity-100">
              <X className="w-4 h-4" />
            </button>
          </div>
        )}
        {summary?.busy && (
          <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/20 text-sm text-amber-800 dark:text-amber-300 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            A document is being indexed. Cleanups that touch the indexes wait until it finishes.
          </div>
        )}

        {!summary ? (
          <div className="flex items-center justify-center py-24">
            {loading ? (
              <RefreshCw className={`w-6 h-6 animate-spin ${MUTED}`} />
            ) : (
              <p className={`text-sm ${MUTED}`}>Storage information is unavailable. Is the backend running?</p>
            )}
          </div>
        ) : (
          <>
            {/* ── Totals ───────────────────────────── */}
            <div className="grid grid-cols-4 gap-3">
              <LiquidGlassCard variant="space" className="p-3">
                <div className="flex items-center gap-2 mb-1">
                  <HardDrive className="w-4 h-4 text-sky-500" />
                  <span className={`text-[11px] font-medium ${MUTED}`}>On disk</span>
                </div>
                <p className={`text-lg font-bold font-mono ${TEXT}`}>{formatBytes(summary.totals.disk_bytes, 2)}</p>
                <p className={`text-[11px] ${MUTED}`}>{summary.totals.stores} stores</p>
              </LiquidGlassCard>
              <LiquidGlassCard variant="space" className="p-3">
                <div className="flex items-center gap-2 mb-1">
                  <Sparkles className="w-4 h-4 text-emerald-500" />
                  <span className={`text-[11px] font-medium ${MUTED}`}>Reclaimable now</span>
                </div>
                <p className={`text-lg font-bold font-mono ${TEXT}`}>{formatBytes(summary.totals.reclaimable_bytes)}</p>
                <p className={`text-[11px] ${MUTED}`}>orphaned data and unused pages</p>
              </LiquidGlassCard>
              <LiquidGlassCard variant="space" className="p-3">
                <div className="flex items-center gap-2 mb-1">
                  <Cpu className="w-4 h-4 text-violet-500" />
                  <span className={`text-[11px] font-medium ${MUTED}`}>GPU memory</span>
                </div>
                {gpu ? (
                  <>
                    <p className={`text-lg font-bold font-mono ${TEXT}`}>
                      {(gpu.used_mb / 1024).toFixed(1)} / {(gpu.total_mb / 1024).toFixed(1)} GB
                    </p>
                    <div className="mt-1.5">
                      <Bar value={gpu.used_mb} max={gpu.total_mb} tone="bg-violet-500" />
                    </div>
                  </>
                ) : (
                  <p className={`text-sm ${MUTED}`}>No GPU detected</p>
                )}
              </LiquidGlassCard>
              <LiquidGlassCard variant="space" className="p-3">
                <div className="flex items-center gap-2 mb-1">
                  <Box className="w-4 h-4 text-amber-500" />
                  <span className={`text-[11px] font-medium ${MUTED}`}>Server memory</span>
                </div>
                <p className={`text-lg font-bold font-mono ${TEXT}`}>{formatBytes(summary.memory.process_rss_bytes, 2)}</p>
                <p className={`text-[11px] ${MUTED}`}>backend process</p>
              </LiquidGlassCard>
            </div>

            {/* ── Databases ────────────────────────── */}
            <section className="space-y-3">
              <SectionTitle
                icon={<Database className="w-4 h-4 text-sky-500" />}
                title="Databases"
                hint="Rows, unused space and size over time"
              />
              <div className="grid grid-cols-2 gap-3">
                {databases.map((db) => {
                  const bloat = db.details.bloat_pct ?? 0;
                  return (
                    <LiquidGlassCard key={db.id} variant="space" className="p-4 space-y-3">
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <p className={`flex items-center gap-2 text-sm font-semibold ${TEXT}`}>
                            {STORE_ICONS[db.id]}
                            {db.label}
                          </p>
                          <p className={`text-[11px] mt-0.5 ${MUTED}`}>{db.details.engine}</p>
                        </div>
                        <p className={`text-lg font-bold font-mono shrink-0 ${TEXT}`}>{formatBytes(db.size_bytes)}</p>
                      </div>

                      <div>
                        <div className={`flex items-center justify-between text-[11px] mb-1 ${MUTED}`}>
                          <span>Unused space</span>
                          <span className="font-mono">
                            {formatBytes(db.details.free_bytes ?? 0)} ({bloat}%)
                          </span>
                        </div>
                        <Bar value={bloat} max={100} tone={bloat >= 25 ? 'bg-amber-500' : 'bg-emerald-500'} />
                      </div>

                      <div className="space-y-1">
                        {(db.details.tables ?? []).map((t) => (
                          <div key={t.name} className="flex items-center justify-between text-xs">
                            <span className={`font-mono ${MUTED}`}>{t.name}</span>
                            <span className={`font-mono font-semibold ${TEXT}`}>{t.rows.toLocaleString()}</span>
                          </div>
                        ))}
                        {db.details.queue_rows != null && (
                          <div className="flex items-center justify-between text-xs">
                            <span className={`font-mono ${MUTED}`}>write log</span>
                            <span className={`font-mono font-semibold ${TEXT}`}>{db.details.queue_rows.toLocaleString()}</span>
                          </div>
                        )}
                      </div>

                      <Sparkline history={history} storeId={db.id} />

                      <div className="flex items-center justify-between gap-2">
                        <span className={`text-[11px] ${MUTED}`}>
                          {db.last_modified ? `Updated ${formatDate(db.last_modified)}` : ''}
                        </span>
                        <div className="flex items-center gap-1.5 flex-wrap justify-end">{actionButtons(db)}</div>
                      </div>
                    </LiquidGlassCard>
                  );
                })}
              </div>
            </section>

            {/* ── Caches & files ───────────────────── */}
            <section className="space-y-3">
              <SectionTitle
                icon={<Layers className="w-4 h-4 text-amber-500" />}
                title="Caches and files"
                hint={`${safeCount} safe cleanups available`}
              />
              <div className="space-y-2">
                {caches.map((store) => (
                  <LiquidGlassCard key={store.id} variant="space" className="p-3.5">
                    <div className="flex items-center gap-4">
                      <div className="w-9 h-9 rounded-xl bg-cream-200/70 dark:bg-sand-dark flex items-center justify-center shrink-0">
                        {STORE_ICONS[store.id] ?? <Box className="w-4 h-4" />}
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2 flex-wrap">
                          <p className={`text-sm font-semibold ${TEXT}`}>{store.label}</p>
                          {(store.details.orphaned ?? 0) > 0 && (
                            <span className="px-1.5 py-0.5 rounded-full text-[10px] font-mono bg-amber-500/10 text-amber-700 dark:text-amber-300 border border-amber-500/20">
                              {store.details.orphaned} orphaned · {formatBytes(store.reclaimable_bytes)}
                            </span>
                          )}
                        </div>
                        <p className={`text-[11px] mt-0.5 truncate ${MUTED}`} title={store.path ?? undefined}>
                          {store.description}
                        </p>
                        <div className="mt-2">
                          <Bar value={store.size_bytes} max={largestCache} />
                        </div>
                      </div>
                      <div className="w-28 shrink-0 text-right">
                        <p className={`text-sm font-bold font-mono ${TEXT}`}>
                          {store.path ? formatBytes(store.size_bytes) : '—'}
                        </p>
                        <p className={`text-[11px] ${MUTED}`}>
                          {store.items != null ? `${store.items.toLocaleString()} ${store.items_label}` : ''}
                        </p>
                      </div>
                      <div className="w-56 shrink-0 flex items-center gap-1.5 flex-wrap justify-end">
                        {actionButtons(store)}
                      </div>
                    </div>
                  </LiquidGlassCard>
                ))}
              </div>
            </section>

            {/* ── Memory ───────────────────────────── */}
            <section className="space-y-3">
              <SectionTitle
                icon={<Cpu className="w-4 h-4 text-violet-500" />}
                title="Memory"
                hint={gpu ? `${gpu.name} · ${(gpu.free_mb / 1024).toFixed(1)} GB free` : undefined}
              />
              <div className="grid grid-cols-2 gap-3">
                {summary.memory.items.map((item) => (
                  <LiquidGlassCard key={item.id} variant="space" className="p-3.5">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className={`flex items-center gap-2 text-sm font-semibold ${TEXT}`}>
                          {MEMORY_ICONS[item.id] ?? <Box className="w-4 h-4" />}
                          {item.label}
                        </p>
                        <p className={`text-[11px] mt-0.5 ${MUTED}`}>{item.description}</p>
                      </div>
                      <p className={`text-sm font-bold font-mono shrink-0 text-right ${TEXT}`}>
                        {item.id === 'vision_model' ? item.unit : `${item.value.toLocaleString()} ${item.unit}`}
                      </p>
                    </div>

                    {item.id === 'ollama_models' ? (
                      <div className="mt-3 space-y-2">
                        {(item.models ?? []).map((m) => (
                          <div key={m.name} className="flex items-center justify-between gap-2 text-xs">
                            <span className={`font-mono truncate ${TEXT}`}>
                              {m.name}
                              {m.pinned && <span className={`ml-1.5 ${MUTED}`}>pinned</span>}
                            </span>
                            <span className="flex items-center gap-2 shrink-0">
                              <span className={`font-mono ${MUTED}`}>
                                {formatBytes(m.vram_bytes)} VRAM
                                {m.context_length ? ` · ${m.context_length.toLocaleString()} ctx` : ''}
                              </span>
                              {item.actions
                                .filter((a) => a.id === 'unload')
                                .map((a) => (
                                  <button
                                    key={a.id}
                                    type="button"
                                    onClick={() => ask(item, a, m.name)}
                                    disabled={running !== null}
                                    title={a.description}
                                    className={DANGER_BTN}
                                  >
                                    {running === `${item.id}/${a.id}/${m.name}` ? 'Working…' : a.label}
                                  </button>
                                ))}
                            </span>
                          </div>
                        ))}
                        {loadable.length > 0 && (
                          <div className="flex items-center justify-end gap-2 pt-1">
                            <select
                              value={chosenTarget}
                              onChange={(e) => setLoadTarget(e.target.value)}
                              aria-label="Model to load"
                              className="sp-field-inner rounded-lg px-2 py-1 text-[11px] font-mono outline-none"
                            >
                              {loadable.map((name) => (
                                <option key={name} value={name}>
                                  {name}
                                </option>
                              ))}
                            </select>
                            {item.actions
                              .filter((a) => a.id === 'reload')
                              .map((a) => (
                                <button
                                  key={a.id}
                                  type="button"
                                  onClick={() => void runAction(item.id, a.id, { target: chosenTarget })}
                                  disabled={running !== null || !chosenTarget}
                                  title={a.description}
                                  className={SAFE_BTN}
                                >
                                  {running === `${item.id}/${a.id}/${chosenTarget}` ? 'Loading…' : 'Load'}
                                </button>
                              ))}
                          </div>
                        )}
                      </div>
                    ) : (
                      item.actions.length > 0 && (
                        <div className="mt-3 flex items-center justify-end gap-1.5">{actionButtons(item)}</div>
                      )
                    )}
                  </LiquidGlassCard>
                ))}
              </div>
            </section>

            {/* ── Read-only ────────────────────────── */}
            <section className="grid grid-cols-2 gap-3">
              <LiquidGlassCard variant="space" className="p-4 space-y-2.5">
                <SectionTitle icon={<FileText className="w-4 h-4 text-terracotta-600" />} title="Library and build" hint="Read-only here" />
                {library.map((store) => (
                  <div key={store.id} className="flex items-center justify-between gap-3 text-xs" title={store.description}>
                    <span className={`flex items-center gap-2 min-w-0 ${TEXT}`}>
                      {STORE_ICONS[store.id] ?? <Box className={`w-4 h-4 ${MUTED}`} />}
                      <span className="truncate">{store.label}</span>
                    </span>
                    <span className={`font-mono shrink-0 ${MUTED}`}>
                      {store.items != null ? `${store.items.toLocaleString()} ${store.items_label} · ` : ''}
                      <span className={`font-semibold ${TEXT}`}>{formatBytes(store.size_bytes)}</span>
                    </span>
                  </div>
                ))}
              </LiquidGlassCard>
              <LiquidGlassCard variant="space" className="p-4 space-y-2.5">
                <SectionTitle
                  icon={<Box className="w-4 h-4 text-violet-500" />}
                  title="Models"
                  hint={`${formatBytes(summary.models.reduce((sum, m) => sum + m.size_bytes, 0), 2)} · read-only`}
                />
                {summary.models.map((m) => (
                  <div key={m.id} className="flex items-center justify-between gap-3 text-xs" title={m.path ?? undefined}>
                    <span className={`truncate ${TEXT}`}>{m.label}</span>
                    <span className={`font-mono font-semibold shrink-0 ${TEXT}`}>{formatBytes(m.size_bytes)}</span>
                  </div>
                ))}
              </LiquidGlassCard>
            </section>

            {/* ── Activity ─────────────────────────── */}
            <section className="space-y-3">
              <SectionTitle icon={<History className={`w-4 h-4 ${MUTED}`} />} title="Cleanup activity" />
              <LiquidGlassCard variant="space" className="p-4">
                {history.actions.length === 0 ? (
                  <p className={`text-xs ${MUTED}`}>No cleanups have been run yet.</p>
                ) : (
                  <div className="space-y-2">
                    {history.actions
                      .slice(-12)
                      .reverse()
                      .map((a, i) => (
                        <div key={`${a.ts}-${i}`} className="flex items-center justify-between gap-3 text-xs">
                          <span className={`font-mono shrink-0 w-32 ${MUTED}`}>{formatDate(a.ts)}</span>
                          <span className={`flex-1 min-w-0 truncate ${TEXT}`}>
                            <span className="font-semibold">{labels[a.store] ?? a.store}</span>
                            <span className={MUTED}> · {a.message}</span>
                          </span>
                          <span className={`font-mono shrink-0 ${a.freed_bytes > 0 ? 'text-emerald-600 dark:text-emerald-400' : MUTED}`}>
                            {a.freed_bytes > 0 ? `−${formatBytes(a.freed_bytes)}` : `${a.removed_items} items`}
                          </span>
                        </div>
                      ))}
                  </div>
                )}
              </LiquidGlassCard>
            </section>
          </>
        )}
      </div>

      {/* ── Confirmations ──────────────────────────── */}
      <ConfirmDialog
        open={pending !== null}
        title={pending ? `${pending.action.label}: ${pending.target ?? pending.storeLabel}` : ''}
        message={
          pending
            ? `${pending.action.description}${
                pending.reclaimable ? ` About ${formatBytes(pending.reclaimable)} will be freed.` : ''
              }`
            : ''
        }
        confirmLabel={pending?.action.label ?? ''}
        busy={running !== null}
        confirmDisabled={Boolean(pending?.action.needs_days) && !(days >= 1)}
        onConfirm={() => void confirmPending()}
        onCancel={() => setPending(null)}
      >
        {pending?.action.needs_days && (
          <label className={`flex items-center gap-2 text-xs ${TEXT}`}>
            Older than
            <input
              type="number"
              min={1}
              max={3650}
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              className="sp-field-inner w-20 rounded-lg px-2 py-1 font-mono outline-none"
            />
            days
          </label>
        )}
      </ConfirmDialog>

      <ConfirmDialog
        open={confirmCleanup}
        title="Clean up safe items"
        message={`Removes images and cached pages of deleted documents and compacts both databases, freeing about ${formatBytes(
          summary?.totals.reclaimable_bytes ?? 0,
        )}. Documents, answers and telemetry records are kept.`}
        confirmLabel="Clean up"
        busy={running !== null}
        onConfirm={() => {
          void cleanSafe().then(() => setConfirmCleanup(false));
        }}
        onCancel={() => setConfirmCleanup(false)}
      />
    </div>
  );
}

export default StorageView;
