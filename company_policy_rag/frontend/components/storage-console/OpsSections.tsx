'use client';

/** Cleanup: the review drawer where every action is classified and costed
 *  before it runs, and the audit table of what was run and what it changed. */

import React, { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Check, ChevronRight, ClipboardCheck } from 'lucide-react';

import { apiClient } from '@/lib/api-client';
import { cn, formatBytes, formatCount, formatDate, formatDuration } from '@/lib/utils';
import type {
  StorageAuditEntry,
  StorageCleanupItem,
  StorageCleanupPlan,
  StorageCleanupSelection,
  StoragePreview,
  StorageSafety,
  StorageStore,
} from '@/lib/types';
import { DaysSelector, EstimateLine } from './ActionDialog';
import { useConsole, useRowClass } from './context';
import { auditItem } from './filtering';
import { Overlay } from './Overlay';
import { EmptyState, SafetyTag, safetyHelp, Unavailable } from './primitives';

const SAFETY_ORDER: StorageSafety[] = ['SAFE', 'REBUILDABLE', 'DESTRUCTIVE'];
const SAFETY_TITLES: Record<StorageSafety, string> = {
  SAFE: 'Safe cleanup',
  REBUILDABLE: 'Rebuildable caches',
  DESTRUCTIVE: 'Destructive',
};

const itemKey = (item: { store_id: string; action_id: string }) => `${item.store_id}/${item.action_id}`;

/* ── Cleanup panel (in the page) ─────────────────────────────────────────── */

export function CleanupPanel({ stores, reclaimable, onReview }: { stores: StorageStore[]; reclaimable: number; onReview: () => void }) {
  const candidates = stores.filter((s) => s.reclaimable_bytes > 0).sort((a, b) => b.reclaimable_bytes - a.reclaimable_bytes);
  return (
    <div className="sc-panel flex flex-wrap items-center justify-between gap-4 px-4 py-3.5">
      {reclaimable > 0 ? (
        <div className="min-w-0">
          <p className="sc-t2 text-[12.5px]">
            <span className="sc-num sc-t1 text-[17px] font-semibold tracking-[-0.02em]">{formatBytes(reclaimable, 2)}</span> can be
            reclaimed without deleting a document, an answer or a telemetry record.
          </p>
          <ul className="mt-1.5 flex flex-wrap gap-x-4 gap-y-0.5 text-[11.5px]">
            {candidates.map((store) => (
              <li key={store.id} className="flex items-center gap-1.5">
                <Check className="sc-ok h-3 w-3" aria-hidden />
                <span className="sc-t2">
                  {store.group === 'databases' ? `Compact ${store.label}` : `Orphaned in ${store.label}`}
                </span>
                <span className="sc-num sc-t1">{formatBytes(store.reclaimable_bytes)}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <div>
          <p className="sc-t1 text-[12.5px] font-semibold">Nothing to clean</p>
          <p className="sc-t2 mt-0.5 text-[12px]">All rebuildable stores are currently tidy. 0 bytes safely reclaimable.</p>
        </div>
      )}
      <button type="button" className="sc-btn sc-btn--primary" onClick={onReview}>
        <ClipboardCheck className="h-3.5 w-3.5" aria-hidden />
        Review cleanup
      </button>
    </div>
  );
}

/* ── Cleanup center (drawer) ─────────────────────────────────────────────── */

interface Choice {
  selected: boolean;
  days: number | null;
  /** Re-measured for a changed age; falls back to the plan's figures. */
  preview: StoragePreview | null;
}

function CleanupRow({
  item,
  choice,
  busy,
  onToggle,
  onDays,
}: {
  item: StorageCleanupItem;
  choice: Choice;
  busy: boolean;
  onToggle: (selected: boolean) => void;
  onDays: (days: number) => void;
}) {
  const [open, setOpen] = useState(false);
  const estimate = choice.preview ?? item;
  const hasEffect = choice.preview ? Boolean(choice.preview.estimated_bytes) || Boolean(choice.preview.affected_items) : item.has_effect;
  const disabled = busy || Boolean(item.blocked_reason) || !hasEffect;
  const showImpact = open || choice.selected;
  const id = `sc-clean-${item.store_id}-${item.action_id}`;

  return (
    <li className={cn('py-2.5', !hasEffect && !item.blocked_reason && 'opacity-60')}>
      <div className="flex items-start gap-2.5">
        <input
          id={id}
          type="checkbox"
          className="sc-check mt-[3px]"
          checked={choice.selected && !disabled}
          disabled={disabled}
          onChange={(e) => onToggle(e.target.checked)}
        />
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline justify-between gap-3">
            <label htmlFor={id} className="sc-t1 min-w-0 cursor-pointer text-[12.5px] font-medium">
              {item.label} <span className="sc-t2 font-normal">· {item.store_label}</span>
            </label>
            <span className="shrink-0 text-[12.5px]">
              {hasEffect ? <EstimateLine preview={estimate} /> : <span className="sc-t3">Nothing to remove</span>}
            </span>
          </div>

          {item.blocked_reason && (
            <p className="sc-warn mt-1 flex items-start gap-1.5 text-[11.5px]">
              <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" aria-hidden />
              {item.blocked_reason}
            </p>
          )}
          {item.needs_days && choice.days !== null && (
            <div className="mt-1.5">
              <DaysSelector value={choice.days} onChange={onDays} disabled={busy} />
            </div>
          )}

          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={showImpact}
            className="sc-focus sc-t3 mt-1 flex items-center gap-1 rounded text-[11.5px] hover:underline"
          >
            <ChevronRight className={cn('h-3 w-3 transition-transform', showImpact && 'rotate-90')} aria-hidden />
            What this does
          </button>
          {showImpact && (
            <dl className="mt-1 grid grid-cols-[64px_1fr] gap-x-3 gap-y-1 text-[11.5px] leading-relaxed">
              <dt className="sc-t3">Deletes</dt>
              <dd className="sc-t2">{item.impact.deletes}</dd>
              <dt className="sc-t3">Rebuild</dt>
              <dd className="sc-t2">{item.impact.rebuild}</dd>
              <dt className="sc-t3">Impact</dt>
              <dd className="sc-t2">{item.impact.performance}</dd>
              {estimate.note && (
                <>
                  <dt className="sc-t3">Note</dt>
                  <dd className="sc-t2">{estimate.note}</dd>
                </>
              )}
            </dl>
          )}
        </div>
      </div>
    </li>
  );
}

interface CleanupCenterProps {
  open: boolean;
  busy: boolean;
  onClose: () => void;
  onRun: (items: StorageCleanupSelection[]) => Promise<unknown>;
}

export function CleanupCenter({ open, busy, onClose, onRun }: CleanupCenterProps) {
  const [plan, setPlan] = useState<StorageCleanupPlan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [choices, setChoices] = useState<Record<string, Choice>>({});
  const [acknowledged, setAcknowledged] = useState(false);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setPlan(null);
    setError(null);
    setAcknowledged(false);
    apiClient
      .getCleanupPlan()
      .then((next) => {
        if (cancelled) return;
        setPlan(next);
        setChoices(
          Object.fromEntries(
            next.items.map((item) => [itemKey(item), { selected: item.selected, days: item.default_days, preview: null }]),
          ),
        );
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load the cleanup plan');
      });
    return () => {
      cancelled = true;
    };
  }, [open]);

  const update = (key: string, patch: Partial<Choice>) =>
    setChoices((current) => ({ ...current, [key]: { ...current[key], ...patch } }));

  const changeDays = (item: StorageCleanupItem, days: number) => {
    const key = itemKey(item);
    update(key, { days });
    if (!Number.isFinite(days) || days < 1 || days > 3650) return;
    apiClient
      .previewStorageAction(item.store_id, item.action_id, days)
      .then((preview) => setChoices((current) => (current[key]?.days === days ? { ...current, [key]: { ...current[key], preview } } : current)))
      .catch(() => undefined);
  };

  const chosen = useMemo(
    () => (plan?.items ?? []).filter((item) => choices[itemKey(item)]?.selected && !item.blocked_reason),
    [plan, choices],
  );
  const total = chosen.reduce((sum, item) => sum + ((choices[itemKey(item)]?.preview ?? item).estimated_bytes ?? 0), 0);
  const unmeasured = chosen.filter((item) => (choices[itemKey(item)]?.preview ?? item).estimated_bytes === null).length;
  const destructive = chosen.filter((item) => item.safety === 'DESTRUCTIVE');
  const invalidDays = chosen.some((item) => {
    const days = choices[itemKey(item)]?.days;
    return item.needs_days && !(typeof days === 'number' && days >= 1 && days <= 3650);
  });
  const nothingToClean = plan !== null && !plan.items.some((item) => item.has_effect);

  const run = async () => {
    await onRun(
      chosen.map((item) => {
        const days = choices[itemKey(item)]?.days;
        return {
          store_id: item.store_id,
          action_id: item.action_id,
          ...(item.needs_days && typeof days === 'number' ? { older_than_days: days } : {}),
        };
      }),
    );
    onClose();
  };

  return (
    <Overlay
      open={open}
      onClose={onClose}
      busy={busy}
      eyebrow="Cleanup center"
      title="Review cleanup"
      footer={
        <div className="flex w-full flex-wrap items-center justify-between gap-3">
          <div className="min-w-0">
            <p className="sc-label">Estimated reclaimable</p>
            <p className="sc-num sc-t1 mt-0.5 text-[16px] font-semibold tracking-[-0.02em]">
              {formatBytes(total, 2)}
              <span className="sc-t3 ml-2 text-[11.5px] font-normal">
                {chosen.length} {chosen.length === 1 ? 'action' : 'actions'}
                {unmeasured ? ` · ${unmeasured} free memory, not disk` : ''}
              </span>
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button type="button" className="sc-btn" onClick={onClose} disabled={busy}>
              Cancel
            </button>
            <button
              type="button"
              className={cn('sc-btn', destructive.length > 0 ? 'sc-btn--solid-danger' : 'sc-btn--primary')}
              disabled={busy || chosen.length === 0 || invalidDays || (destructive.length > 0 && !acknowledged)}
              onClick={() => void run()}
            >
              {busy ? 'Cleaning…' : 'Run cleanup'}
            </button>
          </div>
        </div>
      }
    >
      {error ? (
        <EmptyState icon={<AlertTriangle className="h-4 w-4" />} title="The cleanup plan could not be loaded" detail={error} />
      ) : !plan ? (
        <p className="sc-t2 text-[12.5px]">Measuring what can be reclaimed…</p>
      ) : (
        <div className="space-y-5">
          {plan.busy && (
            <p className="sp-warn flex items-start gap-2 rounded-md px-3 py-2 text-[12px]">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
              Cleanup is partly blocked: a document is being indexed, so actions that touch the indexes are unavailable until it
              finishes.
            </p>
          )}
          {nothingToClean && (
            <EmptyState
              className="sc-panel"
              icon={<Check className="h-4 w-4" />}
              title="Nothing to clean"
              detail="All rebuildable stores are currently tidy. 0 bytes safely reclaimable."
            />
          )}
          {SAFETY_ORDER.map((safety) => {
            const items = plan.items.filter((item) => item.safety === safety);
            if (items.length === 0) return null;
            return (
              <section key={safety} aria-label={SAFETY_TITLES[safety]}>
                <div className="flex items-center gap-2">
                  <SafetyTag safety={safety} />
                  <h3 className="sc-t1 text-[12.5px] font-semibold">{SAFETY_TITLES[safety]}</h3>
                </div>
                <p className="sc-t3 mt-1 text-[11.5px]">{safetyHelp(safety)}</p>
                <ul className="sc-divide sc-rule mt-2">
                  {items.map((item) => {
                    const key = itemKey(item);
                    const choice = choices[key] ?? { selected: false, days: item.default_days, preview: null };
                    return (
                      <CleanupRow
                        key={key}
                        item={item}
                        choice={choice}
                        busy={busy}
                        onToggle={(selected) => update(key, { selected })}
                        onDays={(days) => changeDays(item, days)}
                      />
                    );
                  })}
                </ul>
              </section>
            );
          })}
          {destructive.length > 0 && (
            <label className="flex cursor-pointer items-start gap-2.5 rounded-md border border-[color:var(--sc-critical-border)] bg-[color:var(--sc-critical-bg)] px-3 py-2.5 text-[12px]">
              <input type="checkbox" className="sc-check mt-0.5" checked={acknowledged} onChange={(e) => setAcknowledged(e.target.checked)} />
              <span className="sc-t1">
                {destructive.length === 1 ? 'One selected action is' : `${destructive.length} selected actions are`} destructive. What{' '}
                {destructive.length === 1 ? 'it removes' : 'they remove'} cannot be brought back.
              </span>
            </label>
          )}
          <p className="sc-t3 text-[11.5px]">Documents are never deleted from here. Delete them in the Library tab.</p>
        </div>
      )}
    </Overlay>
  );
}

/* ── Audit table ─────────────────────────────────────────────────────────── */

const AUDIT_PAGE = 12;

const ACTION_WORDS: Record<string, string> = {
  compact: 'Compact',
  clear: 'Clear',
  older_than: 'Delete old',
  remove_orphaned: 'Remove orphaned',
  remove_old: 'Remove old',
  delete: 'Delete',
  unload: 'Unload',
  reload: 'Load',
};

const bytesOrDash = (value: number | null | undefined) =>
  typeof value === 'number' ? formatBytes(value) : <span className="sc-t3">—</span>;

export function AuditTable({ actions, labels }: { actions: StorageAuditEntry[]; labels: Record<string, string> }) {
  const { match, advanced } = useConsole();
  const rowClass = useRowClass();
  const [full, setFull] = useState(false);

  const rows = useMemo(
    () => [...actions].reverse().filter((entry) => match(auditItem(entry, labels[entry.store] ?? entry.store))),
    [actions, labels, match],
  );

  if (actions.length === 0) {
    return <EmptyState className="sc-panel" title="No cleanups have been run yet" detail="Every cleanup is recorded here with what it changed." />;
  }
  if (rows.length === 0) return null;
  const shown = full ? rows : rows.slice(0, AUDIT_PAGE);

  return (
    <div className="sc-panel overflow-x-auto">
      <table className="w-full border-collapse text-[12px]">
        <thead>
          <tr>
            <th className="sc-th px-4 py-2">Time</th>
            <th className="sc-th px-2 py-2">Action</th>
            <th className="sc-th px-2 py-2">Target</th>
            <th className="sc-th px-2 py-2 text-right">Before</th>
            <th className="sc-th px-2 py-2 text-right">After</th>
            <th className="sc-th px-2 py-2 text-right">Reclaimed</th>
            <th className="sc-th px-2 py-2 text-right">Duration</th>
            <th className="sc-th px-2 py-2">Status</th>
            <th className="sc-th px-4 py-2">Initiated by</th>
            {advanced && <th className="sc-th px-4 py-2">Audit ID</th>}
          </tr>
        </thead>
        <tbody>
          {shown.map((entry, index) => {
            const failed = entry.status === 'failed';
            return (
              <tr
                key={entry.id ?? `${entry.ts}-${index}`}
                className={cn(rowClass(), 'border-t border-[color:var(--sp-hairline)]')}
                title={failed ? entry.error : entry.message}
              >
                <td className="sc-num sc-t2 whitespace-nowrap px-4 py-1.5">{formatDate(entry.ts)}</td>
                <td className="sc-t1 whitespace-nowrap px-2 py-1.5 font-medium">{ACTION_WORDS[entry.action] ?? entry.action}</td>
                <td className="sc-t1 max-w-[220px] truncate px-2 py-1.5">{entry.target ?? labels[entry.store] ?? entry.store}</td>
                <td className="sc-num sc-t2 px-2 py-1.5 text-right">{bytesOrDash(entry.before_bytes)}</td>
                <td className="sc-num sc-t2 px-2 py-1.5 text-right">{bytesOrDash(entry.after_bytes)}</td>
                <td className="sc-num sc-t1 px-2 py-1.5 text-right">
                  {entry.freed_bytes > 0 ? (
                    formatBytes(entry.freed_bytes)
                  ) : entry.removed_items > 0 ? (
                    <span className="sc-t2">{formatCount(entry.removed_items, 'items')}</span>
                  ) : (
                    <span className="sc-t3">—</span>
                  )}
                </td>
                <td className="sc-num sc-t2 px-2 py-1.5 text-right">
                  {entry.duration_ms !== undefined ? formatDuration(entry.duration_ms) : <span className="sc-t3">—</span>}
                </td>
                <td className="whitespace-nowrap px-2 py-1.5">
                  {entry.status === undefined ? (
                    <Unavailable reason="Recorded before statuses were tracked.">Not recorded</Unavailable>
                  ) : (
                    <span className={cn('flex items-center gap-1.5', failed ? 'sc-crit' : 'sc-t1')}>
                      {failed ? <AlertTriangle className="h-3 w-3" aria-hidden /> : <Check className="sc-ok h-3 w-3" aria-hidden />}
                      {failed ? 'Failed' : 'Success'}
                    </span>
                  )}
                </td>
                <td className="sc-t2 whitespace-nowrap px-4 py-1.5">
                  {entry.initiated_by === 'cleanup' ? 'Cleanup center' : entry.initiated_by === 'manual' ? 'Manual' : '—'}
                </td>
                {advanced && <td className="sc-num sc-t3 whitespace-nowrap px-4 py-1.5">{entry.id ?? '—'}</td>}
              </tr>
            );
          })}
        </tbody>
      </table>
      {rows.length > AUDIT_PAGE && (
        <div className="sc-rule px-4 py-2">
          <button type="button" className="sc-btn sc-btn--ghost" onClick={() => setFull((v) => !v)}>
            {full ? `Show the latest ${AUDIT_PAGE}` : `View full history (${rows.length.toLocaleString()})`}
          </button>
        </div>
      )}
    </div>
  );
}
