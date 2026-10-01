'use client';

/** In-app confirmation for destructive actions. Replaces window.confirm, which
 *  browsers/webviews can silently suppress — that made actions like delete
 *  appear dead. */

import React from 'react';
import { AlertTriangle, Trash2 } from 'lucide-react';

interface ConfirmDialogProps {
  open: boolean;
  title: string;
  message: string;
  confirmLabel: string;
  busy?: boolean;
  confirmDisabled?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
  /** Extra controls the confirmation needs, e.g. an "older than N days" input. */
  children?: React.ReactNode;
}

export function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel,
  busy = false,
  confirmDisabled = false,
  onConfirm,
  onCancel,
  children,
}: ConfirmDialogProps) {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[400] flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      onClick={() => !busy && onCancel()}
    >
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" />
      <div
        className="relative w-full max-w-sm rounded-2xl bg-white dark:bg-sand-dark border border-sand-border dark:border-sand-darkBorder shadow-xl p-5"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-3">
          <div className="p-2 rounded-xl bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-500/20 shrink-0">
            <AlertTriangle className="w-5 h-5" />
          </div>
          <div className="min-w-0">
            <h3 className="text-sm font-bold text-charcoal dark:text-cream-100">{title}</h3>
            <p className="text-xs text-charcoal-muted dark:text-cream-400 mt-1 leading-relaxed">{message}</p>
          </div>
        </div>
        {children && <div className="mt-4">{children}</div>}
        <div className="flex items-center justify-end gap-2 mt-5">
          <button
            onClick={onCancel}
            disabled={busy}
            className="px-3.5 py-2 rounded-xl bg-cream-100 dark:bg-cream-950 text-charcoal dark:text-cream-200 border border-sand-border dark:border-sand-darkBorder text-xs font-semibold transition-colors hover:bg-cream-200 dark:hover:bg-[#1B2748] disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={busy || confirmDisabled}
            className="px-3.5 py-2 rounded-xl bg-rose-600 hover:bg-rose-700 text-white text-xs font-semibold transition-colors disabled:opacity-60 flex items-center gap-1.5"
          >
            <Trash2 className="w-3.5 h-3.5" />
            {busy ? 'Working…' : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

export default ConfirmDialog;
