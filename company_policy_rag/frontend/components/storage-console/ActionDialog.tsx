'use client';

/** Confirmation for one action. Spells out what is deleted, whether and how it
 *  comes back, what it costs, and how much it reclaims, measured by the
 *  backend's preview before anything runs. */

import React, { useEffect, useState } from 'react';
import { AlertTriangle } from 'lucide-react';

import { apiClient } from '@/lib/api-client';
import { cn, formatBytes } from '@/lib/utils';
import type { StoragePreview } from '@/lib/types';
import type { ActionRequest } from './context';
import { Overlay } from './Overlay';
import { SafetyTag } from './primitives';

const DAY_PRESETS = [7, 30, 90];

export function DaysSelector({
  value,
  onChange,
  disabled,
}: {
  value: number;
  onChange: (days: number) => void;
  disabled?: boolean;
}) {
  const [custom, setCustom] = useState(!DAY_PRESETS.includes(value));
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="sc-t2 mr-1 text-[12px]">Older than</span>
      {DAY_PRESETS.map((days) => (
        <button
          key={days}
          type="button"
          disabled={disabled}
          aria-pressed={!custom && value === days}
          onClick={() => {
            setCustom(false);
            onChange(days);
          }}
          className="sc-chip"
        >
          {days} days
        </button>
      ))}
      <button type="button" disabled={disabled} aria-pressed={custom} onClick={() => setCustom(true)} className="sc-chip">
        Custom
      </button>
      {custom && (
        <label className="sc-t2 flex items-center gap-1.5 text-[12px]">
          <input
            type="number"
            min={1}
            max={3650}
            value={value}
            disabled={disabled}
            onChange={(e) => onChange(Number(e.target.value))}
            aria-label="Age in days"
            className="sc-input sc-num w-20 px-2 py-1"
          />
          days
        </label>
      )}
    </div>
  );
}

/** "~126 MB · 18,421 records", or a plain statement that it cannot be measured. */
export function EstimateLine({ preview }: { preview: Pick<StoragePreview, 'affected_items' | 'estimated_bytes' | 'exact'> }) {
  if (!preview.estimated_bytes && preview.affected_items === 0) return <span className="sc-t3">Nothing to remove</span>;
  const parts: string[] = [];
  if (preview.estimated_bytes !== null) {
    parts.push(`${preview.exact ? '' : '~'}${formatBytes(preview.estimated_bytes)}`);
  }
  if (preview.affected_items !== null) {
    parts.push(`${preview.affected_items.toLocaleString()} ${preview.affected_items === 1 ? 'item' : 'items'}`);
  }
  if (parts.length === 0) return <span className="sc-t3">Not measurable</span>;
  return <span className="sc-num sc-t1">{parts.join(' · ')}</span>;
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[112px_1fr] gap-3 py-2">
      <dt className="sc-label pt-0.5">{label}</dt>
      <dd className="sc-t1 text-[12.5px] leading-relaxed">{children}</dd>
    </div>
  );
}

interface ActionDialogProps {
  request: ActionRequest | null;
  busy: boolean;
  onConfirm: (request: ActionRequest, olderThanDays?: number) => void;
  onClose: () => void;
}

export function ActionDialog({ request, busy, onConfirm, onClose }: ActionDialogProps) {
  const [days, setDays] = useState(30);
  const [preview, setPreview] = useState<StoragePreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);

  const storeId = request?.storeId;
  const actionId = request?.action.id;
  const needsDays = request?.action.needs_days ?? false;
  const validDays = Number.isFinite(days) && days >= 1 && days <= 3650;

  useEffect(() => {
    setDays(30);
  }, [storeId, actionId]);

  useEffect(() => {
    if (!storeId || !actionId || (needsDays && !validDays)) {
      setPreview(null);
      return;
    }
    let cancelled = false;
    setPreviewError(null);
    // Debounced so typing a custom age does not fire a request per keystroke.
    const timer = setTimeout(() => {
      apiClient
        .previewStorageAction(storeId, actionId, needsDays ? days : undefined)
        .then((next) => {
          if (!cancelled) setPreview(next);
        })
        .catch((err: unknown) => {
          if (cancelled) return;
          setPreview(null);
          setPreviewError(err instanceof Error ? err.message : 'Preview failed');
        });
    }, 180);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [storeId, actionId, needsDays, days, validDays]);

  const action = request?.action;
  const blocked = preview?.blocked_reason ?? action?.blocked_reason ?? null;
  const destructive = action?.safety === 'DESTRUCTIVE';

  return (
    <Overlay
      open={request !== null}
      onClose={onClose}
      variant="modal"
      busy={busy}
      eyebrow={request?.target ?? request?.storeLabel}
      title={action ? `${action.label}${request?.target ? '' : ` · ${request?.storeLabel}`}` : ''}
      footer={
        <>
          <button type="button" onClick={onClose} disabled={busy} className="sc-btn">
            Cancel
          </button>
          <button
            type="button"
            onClick={() => request && onConfirm(request, needsDays ? days : undefined)}
            disabled={busy || Boolean(blocked) || (needsDays && !validDays)}
            className={cn('sc-btn', destructive ? 'sc-btn--solid-danger' : 'sc-btn--primary')}
          >
            {busy ? 'Working…' : action?.label}
          </button>
        </>
      }
    >
      {action && (
        <div>
          <SafetyTag safety={action.safety} />
          {needsDays && (
            <div className="mt-3">
              <DaysSelector value={days} onChange={setDays} disabled={busy} />
            </div>
          )}
          <dl className="sc-divide mt-3">
            <Row label="Deletes">{action.impact.deletes}</Row>
            <Row label="Rebuild">{action.impact.rebuild}</Row>
            <Row label="Impact">{action.impact.performance}</Row>
            <Row label="Reclaims">
              {preview ? (
                <>
                  <EstimateLine preview={preview} />
                  {preview.note && <span className="sc-t2 mt-0.5 block text-[11.5px]">{preview.note}</span>}
                </>
              ) : previewError ? (
                <span className="sc-t3">Preview unavailable: {previewError}</span>
              ) : (
                <span className="sc-t3">Measuring…</span>
              )}
            </Row>
          </dl>
          {blocked && (
            <p className="sp-warn mt-3 flex items-start gap-2 rounded-md px-3 py-2 text-[12px]">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
              {blocked}
            </p>
          )}
        </div>
      )}
    </Overlay>
  );
}
