'use client';

/** StorageView — the Storage, Memory & Runtime console.
 *  Shows where disk, RAM and VRAM go, what persists, what a cleanup costs and
 *  what changed. Every figure and every available action comes from
 *  /api/admin/storage*; a figure the backend cannot measure is shown as
 *  unavailable, never estimated here. Sections live in components/storage-console/. */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertTriangle, ClipboardCheck, RefreshCw, Search, Unplug, X } from 'lucide-react';

import { useStorage } from '@/hooks/useStorage';
import { useSmoothScroll } from '@/hooks/useSmoothScroll';
import { cn, formatBytes } from '@/lib/utils';
import type { StorageSummary } from '@/lib/types';
import { ActionDialog } from './storage-console/ActionDialog';
import { ConsoleProvider, type ActionRequest, type ConsoleContextValue, type RuntimeState } from './storage-console/context';
import { Diagnostics } from './storage-console/Diagnostics';
import { CachesSection, DatabasesSection, FilesSection, InstalledModelsSection } from './storage-console/DiskSections';
import { DocumentsSection } from './storage-console/DocumentsSection';
import {
  FILTERS,
  auditItem,
  cacheItem,
  installedModelItem,
  loadedModelItem,
  makeMatcher,
  memoryItem,
  storeItem,
  type StorageFilter,
} from './storage-console/filtering';
import { GrowthSection } from './storage-console/GrowthSection';
import { AuditTable, CleanupCenter, CleanupPanel } from './storage-console/OpsSections';
import { Overlay, Toast } from './storage-console/Overlay';
import { Observations, OverviewStrip, StorageMap } from './storage-console/Overview';
import { EmptyState, SafetyTag, Section } from './storage-console/primitives';
import { ActiveOperations, GpuModelsSection, RuntimeMemorySection } from './storage-console/RuntimeSections';
import { StoreDrawer, type StoreDrawerTarget } from './storage-console/StoreDrawer';

type Mode = 'simple' | 'advanced';
const MODE_KEY = 'storage-console-mode';

// Sections that list stores row by row; the rest are summaries.
const FILE_STORES = ['bm25_index', 'eval_artifacts', 'legacy_index', 'logs', 'session_libraries', 'frontend_build'];
// Shown only in the advanced view (or while a search or filter is active).
const ADVANCED_SECTIONS = new Set(['databases', 'documents', 'caches', 'files', 'growth', 'audit', 'diagnostics']);
const SECTION_ORDER = ['databases', 'documents', 'caches', 'runtime', 'gpu', 'installed-models', 'files', 'audit'];

function runtimeOf(summary: StorageSummary, live: ReturnType<typeof useStorage>['live']): RuntimeState {
  const memory = summary.memory;
  return {
    processRss: live ? live.ram.process_rss_bytes : memory.process_rss_bytes,
    systemTotal: live ? live.ram.system_total_bytes : memory.system_total_bytes,
    systemAvailable: live ? live.ram.system_available_bytes : memory.system_available_bytes,
    gpu: live ? live.gpu : memory.gpu,
    gpuBreakdown: live ? live.gpu_breakdown : memory.gpu_breakdown,
    loadedModels: live ? live.loaded_models : memory.loaded_models,
    inProcessModels: live ? live.in_process_models : memory.in_process_models,
    operations: live ? live.operations : summary.operations,
    caches: live?.caches ?? {},
    updatedAt: live?.generated_at ?? summary.generated_at,
  };
}

export function StorageView({ onOpenLibrary }: { onOpenLibrary?: () => void }) {
  const storage = useStorage();
  const { summary, live, history, documents, range, loading, error, toast, running } = storage;
  const scrollRef = useRef<HTMLDivElement>(null);
  useSmoothScroll(scrollRef);

  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState<StorageFilter>('all');
  const [mode, setMode] = useState<Mode>('advanced');
  const [includeModels, setIncludeModels] = useState(true);
  const [closed, setClosed] = useState<Record<string, boolean>>({ diagnostics: true });
  const [pending, setPending] = useState<ActionRequest | null>(null);
  const [drawer, setDrawer] = useState<StoreDrawerTarget | null>(null);
  const [cleanupOpen, setCleanupOpen] = useState(false);
  const [confirmOrphans, setConfirmOrphans] = useState(false);

  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(MODE_KEY);
      if (saved === 'simple' || saved === 'advanced') setMode(saved);
    } catch {
      // Storage can be blocked; the default mode still works.
    }
  }, []);

  const changeMode = (next: Mode) => {
    setMode(next);
    try {
      window.localStorage.setItem(MODE_KEY, next);
    } catch {
      // Not persisted; nothing else depends on it.
    }
  };

  const advanced = mode === 'advanced';
  const searching = query.trim().length > 0;
  const narrowed = searching || filter !== 'all';
  const match = useMemo(() => makeMatcher(query, filter), [query, filter]);
  // Relative times ("18s ago") are re-read whenever fresh data arrives.
  const now = useMemo(() => Date.now(), [summary, live]); // eslint-disable-line react-hooks/exhaustive-deps

  const toggleSection = useCallback((id: string) => setClosed((current) => ({ ...current, [id]: !current[id] })), []);

  const jump = useCallback((sectionId: string) => {
    if (ADVANCED_SECTIONS.has(sectionId)) setMode('advanced');
    setClosed((current) => ({ ...current, [sectionId]: false }));
    // Wait for the section to render open before scrolling to it.
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        document.getElementById(`sc-${sectionId}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }),
    );
  }, []);

  const context = useMemo<ConsoleContextValue>(
    () => ({
      match,
      narrowed,
      searching,
      advanced,
      running,
      now,
      ask: setPending,
      explain: (storeId) => setDrawer({ storeId, focus: 'why' }),
      inspect: (storeId) => setDrawer({ storeId, focus: 'internals' }),
      jump,
    }),
    [match, narrowed, searching, advanced, running, now, jump],
  );

  const runtime = useMemo(() => (summary ? runtimeOf(summary, live) : null), [summary, live]);

  const labels = useMemo(() => {
    const map: Record<string, string> = {};
    for (const s of summary?.stores ?? []) map[s.id] = s.label;
    for (const m of summary?.memory.items ?? []) map[m.id] = m.label;
    return map;
  }, [summary]);

  // How many rows of each section pass the search and filter.
  const counts = useMemo(() => {
    if (!summary || !runtime) return null;
    const stores = summary.stores;
    const orphans = documents?.orphans;
    const orphanItems = orphans ? orphans.images.count + orphans.vision_cache.count : 0;
    return {
      databases: stores.filter((s) => s.group === 'databases' && match(storeItem(s))).length,
      documents:
        (documents?.documents ?? []).filter((d) =>
          match({ text: `${d.filename} ${d.document_id} document upload`, kinds: ['USER_DATA', 'PERSISTENT'], sizeBytes: d.total_bytes_estimated }),
        ).length +
        (match({ text: 'orphaned data artifacts images vision cache', orphaned: orphanItems, reclaimableBytes: orphans?.total_bytes ?? 0 }) ? 1 : 0),
      caches: summary.caches.filter((c) => match(cacheItem(c))).length,
      runtime: summary.memory.items.filter((i) => match(memoryItem(i))).length,
      gpu: [...runtime.loadedModels, ...runtime.inProcessModels].filter((m) => match(loadedModelItem(m))).length,
      'installed-models': summary.models.filter((m) => match(installedModelItem(m))).length,
      files: stores.filter((s) => FILE_STORES.includes(s.id) && match(storeItem(s))).length,
      audit: history.actions.filter((a) => match(auditItem(a, labels[a.store] ?? a.store))).length,
    };
  }, [summary, runtime, documents, history.actions, labels, match]);

  const matchTotal = counts ? Object.values(counts).reduce((sum, n) => sum + n, 0) : 0;

  /** A row section is shown when its mode allows it, or when the search or filter found something in it. */
  const shows = (id: keyof NonNullable<typeof counts>) => {
    if (!counts) return false;
    if (narrowed) return counts[id] > 0;
    return advanced || !ADVANCED_SECTIONS.has(id);
  };
  const showSummaries = !narrowed;

  const jumpToFirstMatch = () => {
    if (!counts) return;
    const first = SECTION_ORDER.find((id) => counts[id as keyof typeof counts] > 0);
    if (first) jump(first);
  };

  const confirmPending = async (request: ActionRequest, olderThanDays?: number) => {
    await storage.runAction(request.storeId, request.action.id, {
      olderThanDays,
      target: request.target,
      title: `${request.action.label} ${request.target ?? request.storeLabel}`,
    });
    setPending(null);
  };

  const orphanSelection = useMemo(() => {
    const orphans = documents?.orphans;
    if (!orphans) return [];
    return [orphans.images, orphans.vision_cache]
      .filter((group) => group.count > 0)
      .map((group) => ({ store_id: group.store_id, action_id: group.action_id, bytes: group.bytes, count: group.count }));
  }, [documents]);

  const sectionProps = (id: string) => ({ id, open: !closed[id], onToggle: toggleSection });

  return (
    <>
      <div ref={scrollRef} className="sc-root sp-scroll h-full flex-1 overflow-y-auto px-4 py-5 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-[1240px] space-y-4 pb-10">
          {/* ── Header ─────────────────────────────── */}
          <header className="flex flex-wrap items-end justify-between gap-x-6 gap-y-3">
            <div>
              <h1 className="sp-display text-[30px] leading-tight">Storage &amp; Runtime</h1>
              <p className="sc-t2 mt-0.5 text-[12.5px]">
                Where disk, RAM and VRAM go, what survives a restart, and what can be cleared.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <div className="relative">
                <Search className="sc-t3 pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2" aria-hidden />
                <input
                  type="search"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') jumpToFirstMatch();
                    if (e.key === 'Escape') setQuery('');
                  }}
                  placeholder="Search storage…"
                  aria-label="Search storage"
                  className="sc-input w-56 py-1.5 pl-8 pr-2.5"
                />
              </div>
              <div className="sp-depth inline-flex rounded-lg p-0.5" role="radiogroup" aria-label="Detail level">
                {(['simple', 'advanced'] as const).map((option) => (
                  <button
                    key={option}
                    type="button"
                    role="radio"
                    aria-checked={mode === option}
                    onClick={() => changeMode(option)}
                    className="sp-depth-btn rounded-md px-2.5 py-1 text-[11.5px] font-semibold capitalize"
                  >
                    {option}
                  </button>
                ))}
              </div>
              <button
                type="button"
                onClick={() => void storage.refresh({ deep: true })}
                disabled={loading}
                className="sc-btn"
                title="Rescan every directory instead of using the cached sizes"
              >
                <RefreshCw className={cn('h-3.5 w-3.5', loading && 'animate-spin')} aria-hidden />
                Refresh
              </button>
              <button type="button" onClick={() => setCleanupOpen(true)} disabled={!summary} className="sc-btn sc-btn--primary">
                <ClipboardCheck className="h-3.5 w-3.5" aria-hidden />
                Review cleanup
                {summary && summary.totals.reclaimable_bytes > 0 && (
                  <span className="sc-num opacity-80">{formatBytes(summary.totals.reclaimable_bytes)}</span>
                )}
              </button>
            </div>
          </header>

          <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Filter stores">
            {FILTERS.map((option) => (
              <button
                key={option.id}
                type="button"
                aria-pressed={filter === option.id}
                onClick={() => setFilter(option.id)}
                className="sc-chip"
              >
                {option.label}
              </button>
            ))}
            {narrowed && summary && (
              <span className="sc-t2 ml-2 flex items-center gap-2 text-[12px]">
                {matchTotal} {matchTotal === 1 ? 'match' : 'matches'}
                <button
                  type="button"
                  className="sc-btn sc-btn--ghost"
                  onClick={() => {
                    setQuery('');
                    setFilter('all');
                  }}
                >
                  <X className="h-3 w-3" aria-hidden />
                  Clear
                </button>
              </span>
            )}
          </div>

          {error && summary && (
            <p className="sp-warn flex items-center gap-2 rounded-md px-3 py-2 text-[12.5px]" role="alert">
              <AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden />
              The last refresh failed, so these figures may be out of date: {error}
            </p>
          )}

          {!summary || !runtime || !counts ? (
            <div className="sc-panel">
              {loading ? (
                <p className="sc-t2 flex items-center gap-2 px-4 py-6 text-[12.5px]">
                  <RefreshCw className="h-3.5 w-3.5 animate-spin" aria-hidden />
                  Reading storage…
                </p>
              ) : (
                <EmptyState
                  icon={<Unplug className="h-4 w-4" />}
                  title="Backend disconnected"
                  detail={`Storage information is unavailable${error ? `: ${error}` : ''}. Start the backend, then refresh.`}
                  action={
                    <button type="button" className="sc-btn" onClick={() => void storage.refresh()}>
                      Try again
                    </button>
                  }
                />
              )}
            </div>
          ) : (
            <ConsoleProvider value={context}>
              <OverviewStrip summary={summary} runtime={runtime} />
              <ActiveOperations runtime={runtime} />

              {showSummaries && (
                <>
                  <Section {...sectionProps('map')} title="Where your storage goes">
                    <StorageMap
                      segments={summary.map}
                      stores={summary.stores}
                      includeModels={includeModels}
                      onIncludeModels={setIncludeModels}
                    />
                  </Section>
                  <Section
                    {...sectionProps('observations')}
                    title="Storage observations"
                    hint={`${summary.observations.length} ${summary.observations.length === 1 ? 'finding' : 'findings'} · ${summary.health.filter((c) => !c.ok).length} health checks need attention`}
                  >
                    <Observations summary={summary} />
                  </Section>
                </>
              )}

              {narrowed && matchTotal === 0 && (
                <EmptyState
                  className="sc-panel"
                  icon={<Search className="h-4 w-4" />}
                  title="Nothing matches"
                  detail="No store, cache, model or audit entry matches this search and filter."
                />
              )}

              {shows('databases') && (
                <Section {...sectionProps('databases')} title="Databases" hint="Used and unused space, fragmentation, growth">
                  <DatabasesSection stores={summary.stores} history={history} />
                </Section>
              )}

              {shows('documents') && (
                <Section
                  {...sectionProps('documents')}
                  title="Documents and generated artifacts"
                  hint={documents ? `${documents.documents.length.toLocaleString()} documents` : undefined}
                >
                  <DocumentsSection
                    documents={documents}
                    stores={summary.stores}
                    onOpenLibrary={onOpenLibrary}
                    onRemoveOrphans={() => setConfirmOrphans(true)}
                    onScanVectors={() => void storage.scanVectorOrphans()}
                  />
                </Section>
              )}

              {shows('caches') && (
                <Section {...sectionProps('caches')} title="Caches" hint="Whether keeping each one pays off">
                  <CachesSection caches={summary.caches} runtime={runtime} />
                </Section>
              )}

              {shows('runtime') && (
                <Section {...sectionProps('runtime')} title="Runtime memory" hint="Lost on restart">
                  <RuntimeMemorySection
                    items={summary.memory.items}
                    caches={summary.caches}
                    flow={summary.memory_flow}
                    runtime={runtime}
                  />
                </Section>
              )}

              {shows('gpu') && (
                <Section
                  {...sectionProps('gpu')}
                  title="GPU and loaded models"
                  hint={`${runtime.loadedModels.length + runtime.inProcessModels.length} in memory`}
                >
                  <GpuModelsSection
                    items={summary.memory.items}
                    installed={summary.models}
                    runtime={runtime}
                    onLoad={(name) => void storage.runAction('ollama_models', 'reload', { target: name, title: `Load ${name}` })}
                  />
                </Section>
              )}

              {(shows('installed-models') || (!narrowed && summary.models.length === 0)) && (
                <Section
                  {...sectionProps('installed-models')}
                  title="Installed models"
                  hint={`${formatBytes(summary.totals.models_bytes, 2)} on disk`}
                >
                  <InstalledModelsSection models={summary.models} />
                </Section>
              )}

              {shows('files') && (
                <Section {...sectionProps('files')} title="Indexes and files" hint="Keyword index, evaluation data, logs, leftovers">
                  <FilesSection stores={summary.stores} />
                </Section>
              )}

              {showSummaries && advanced && (
                <Section {...sectionProps('growth')} title="Storage growth" hint="Snapshots, events and forecast">
                  <GrowthSection history={history} range={range} onRange={(next) => void storage.setRange(next)} />
                </Section>
              )}

              {showSummaries && (
                <Section {...sectionProps('cleanup')} title="Cleanup center" hint="Nothing runs without review">
                  <CleanupPanel
                    stores={summary.stores}
                    reclaimable={summary.totals.reclaimable_bytes}
                    onReview={() => setCleanupOpen(true)}
                  />
                </Section>
              )}

              {(narrowed ? counts.audit > 0 : advanced) && (
                <Section
                  {...sectionProps('audit')}
                  title="Cleanup and audit history"
                  hint={`${history.actions.length.toLocaleString()} recorded`}
                >
                  <AuditTable actions={history.actions} labels={labels} />
                </Section>
              )}

              {showSummaries && advanced && (
                <Section {...sectionProps('diagnostics')} title="Advanced diagnostics" hint="Hierarchy, lifecycle, paths">
                  <Diagnostics summary={summary} />
                </Section>
              )}

            </ConsoleProvider>
          )}
        </div>
      </div>

      {/* Overlays sit outside the scroll container: they have their own scrolling. */}
      <ActionDialog request={pending} busy={running !== null} onConfirm={(r, d) => void confirmPending(r, d)} onClose={() => setPending(null)} />
      <StoreDrawer target={drawer} onClose={() => setDrawer(null)} />
      <CleanupCenter
        open={cleanupOpen}
        busy={running !== null}
        onClose={() => setCleanupOpen(false)}
        onRun={(items) => storage.runCleanup(items)}
      />
      <Overlay
        open={confirmOrphans}
        onClose={() => setConfirmOrphans(false)}
        variant="modal"
        busy={running !== null}
        eyebrow="Orphaned data"
        title="Remove orphaned data"
        footer={
          <>
            <button type="button" className="sc-btn" onClick={() => setConfirmOrphans(false)} disabled={running !== null}>
              Cancel
            </button>
            <button
              type="button"
              className="sc-btn sc-btn--primary"
              disabled={running !== null || orphanSelection.length === 0}
              onClick={() =>
                void storage
                  .runCleanup(orphanSelection.map(({ store_id, action_id }) => ({ store_id, action_id })))
                  .then(() => setConfirmOrphans(false))
              }
            >
              {running === 'cleanup' ? 'Removing…' : 'Remove orphaned data'}
            </button>
          </>
        }
      >
        <SafetyTag safety="SAFE" />
        <p className="sc-t1 mt-3 text-[12.5px] leading-relaxed">
          Deletes the extracted images and cached page readings of documents that are no longer in the library. Everything that
          belongs to a current document is kept, so nothing has to be rebuilt.
        </p>
        <ul className="sc-divide sc-rule mt-3">
          {orphanSelection.map((group) => (
            <li key={group.store_id} className="flex items-center justify-between gap-3 py-2 text-[12.5px]">
              <span className="sc-t2">{labels[group.store_id] ?? group.store_id}</span>
              <span className="sc-num sc-t1">
                {group.count.toLocaleString()} · {formatBytes(group.bytes)}
              </span>
            </li>
          ))}
        </ul>
      </Overlay>
      <Toast toast={toast} onDismiss={storage.dismissToast} />
    </>
  );
}

export default StorageView;
