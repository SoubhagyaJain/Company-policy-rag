'use client';

/** Drawer and modal on the native <dialog>: showModal() puts it in the top
 *  layer, traps focus, restores it on close and handles Esc. `closedby="any"`
 *  adds click-outside dismissal where the browser supports it; elsewhere the
 *  backdrop click is handled by hand. */

import React, { useEffect, useId, useRef } from 'react';
import { CheckCircle2, X, XCircle } from 'lucide-react';

import { cn, formatBytes, formatDuration } from '@/lib/utils';
import type { StorageToast } from '@/hooks/useStorage';

interface OverlayProps {
  open: boolean;
  onClose: () => void;
  variant?: 'drawer' | 'modal';
  eyebrow?: string;
  title: React.ReactNode;
  children: React.ReactNode;
  footer?: React.ReactNode;
  /** While an action runs the overlay cannot be dismissed. */
  busy?: boolean;
}

export function Overlay({ open, onClose, variant = 'drawer', eyebrow, title, children, footer, busy = false }: OverlayProps) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  // The page behind uses a wheel-driven smooth scroller on an ancestor element;
  // without this it would scroll the page instead of the overlay's own content.
  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    const contain = (event: WheelEvent) => event.stopPropagation();
    dialog.addEventListener('wheel', contain, { passive: true });
    return () => dialog.removeEventListener('wheel', contain);
  }, []);

  const onBackdropClick = (event: React.MouseEvent<HTMLDialogElement>) => {
    const dialog = ref.current;
    if (!dialog || busy || event.target !== dialog) return;
    if ('closedBy' in HTMLDialogElement.prototype) return; // the browser handles it
    const rect = dialog.getBoundingClientRect();
    const inside =
      rect.top <= event.clientY && event.clientY <= rect.bottom && rect.left <= event.clientX && event.clientX <= rect.right;
    if (!inside) onClose();
  };

  return (
    <dialog
      ref={ref}
      className={cn('sc-dialog sc-root', variant === 'drawer' ? 'sc-drawer' : 'sc-modal')}
      aria-labelledby={titleId}
      onClose={onClose}
      onCancel={(event) => {
        if (busy) event.preventDefault();
      }}
      onClick={onBackdropClick}
      {...({ closedby: busy ? 'none' : 'any' } as Record<string, string>)}
    >
      {open && (
        <>
          <header className="flex shrink-0 items-start justify-between gap-3 px-5 pb-3 pt-4">
            <div className="min-w-0">
              {eyebrow && <p className="sc-label">{eyebrow}</p>}
              <h2 id={titleId} className="sc-t1 mt-0.5 text-[15px] font-semibold leading-snug">
                {title}
              </h2>
            </div>
            <button type="button" onClick={onClose} disabled={busy} className="sc-icon-btn -mr-1 shrink-0" aria-label="Close">
              <X className="h-4 w-4" aria-hidden />
            </button>
          </header>
          <div className="sp-scroll sc-rule min-h-0 flex-1 overflow-y-auto px-5 py-4">{children}</div>
          {footer && <footer className="sc-rule flex shrink-0 items-center justify-end gap-2 px-5 py-3">{footer}</footer>}
        </>
      )}
    </dialog>
  );
}

const TOAST_MS = 9000;

/** Before → after feedback for the cleanup that just ran. */
export function Toast({ toast, onDismiss }: { toast: StorageToast | null; onDismiss: () => void }) {
  const dismiss = useRef(onDismiss);
  dismiss.current = onDismiss;
  const key = toast?.key;

  // Keyed on the toast itself, so the page's own re-renders do not restart the timer.
  useEffect(() => {
    if (key === undefined) return;
    const timer = setTimeout(() => dismiss.current(), TOAST_MS);
    return () => clearTimeout(timer);
  }, [key]);

  if (!toast) return null;
  const { beforeBytes, afterBytes, freedBytes } = toast;
  const measured = typeof beforeBytes === 'number' && typeof afterBytes === 'number';
  const reduction = measured && beforeBytes > 0 ? ((beforeBytes - afterBytes) / beforeBytes) * 100 : null;

  return (
    <div className="sc-root sc-toast fixed bottom-6 right-6 z-[300] w-[340px] p-3.5" role="status" aria-live="polite">
      <div className="flex items-start gap-2.5">
        {toast.ok ? (
          <CheckCircle2 className="sc-ok mt-0.5 h-4 w-4 shrink-0" aria-hidden />
        ) : (
          <XCircle className="sc-crit mt-0.5 h-4 w-4 shrink-0" aria-hidden />
        )}
        <div className="min-w-0 flex-1">
          <p className="sc-t1 text-[12.5px] font-semibold">{toast.title}</p>
          {measured && (
            <p className="sc-num sc-t1 mt-1.5 text-[13px]">
              {formatBytes(beforeBytes)} <span className="sc-t3">→</span> {formatBytes(afterBytes)}
            </p>
          )}
          {toast.ok && (
            <p className="sc-t2 mt-1 text-[11.5px]">
              {freedBytes > 0
                ? `${formatBytes(freedBytes)} reclaimed`
                : toast.removedItems > 0
                  ? `${toast.removedItems.toLocaleString()} items removed`
                  : 'Nothing needed removing'}
              {reduction !== null && reduction > 0 ? ` · ${reduction.toFixed(1)}% reduction` : ''}
              {toast.durationMs !== undefined ? ` · completed in ${formatDuration(toast.durationMs)}` : ''}
            </p>
          )}
          {toast.detail && <p className={cn('mt-1 text-[11.5px]', toast.ok ? 'sc-t2' : 'sc-crit')}>{toast.detail}</p>}
        </div>
        <button type="button" onClick={onDismiss} className="sc-icon-btn shrink-0" aria-label="Dismiss">
          <X className="h-3.5 w-3.5" aria-hidden />
        </button>
      </div>
    </div>
  );
}
