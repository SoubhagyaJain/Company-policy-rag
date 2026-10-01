'use client';

/** What every section of the storage console shares: the active search and
 *  filter, the view mode, and the handlers that open dialogs and drawers. */

import React, { createContext, useContext } from 'react';

import { cn } from '@/lib/utils';
import type {
  StorageActionMeta,
  StorageGpu,
  StorageGpuBreakdown,
  StorageInProcessModel,
  StorageLiveCache,
  StorageOllamaModel,
  StorageOperations,
} from '@/lib/types';
import type { Filterable, Matcher } from './filtering';

export interface ActionRequest {
  storeId: string;
  storeLabel: string;
  action: StorageActionMeta;
  target?: string;
}

/** RAM, VRAM, models and operations: the live poll when present, else the last summary. */
export interface RuntimeState {
  processRss: number | null;
  systemTotal: number | null;
  systemAvailable: number | null;
  gpu: StorageGpu | null;
  gpuBreakdown: StorageGpuBreakdown | null;
  loadedModels: StorageOllamaModel[];
  inProcessModels: StorageInProcessModel[];
  operations: StorageOperations;
  caches: Record<string, StorageLiveCache>;
  updatedAt: string | null;
}

export interface ConsoleContextValue {
  match: Matcher;
  /** A search term or a filter other than "All" is active. */
  narrowed: boolean;
  searching: boolean;
  advanced: boolean;
  running: string | null;
  now: number;
  ask: (request: ActionRequest) => void;
  explain: (storeId: string) => void;
  inspect: (storeId: string) => void;
  jump: (sectionId: string) => void;
}

const ConsoleContext = createContext<ConsoleContextValue | null>(null);

export const ConsoleProvider = ConsoleContext.Provider;

export function useConsole(): ConsoleContextValue {
  const value = useContext(ConsoleContext);
  if (!value) throw new Error('useConsole must be used inside the storage console');
  return value;
}

/** Row classes: hover wash, plus the accent edge on rows a search matched. */
export function useRowClass() {
  const { searching } = useConsole();
  return (extra?: string) => cn('sc-row', searching && 'sc-match', extra);
}

/** Keep the rows that pass the active search and filter. */
export function useVisible<T>(rows: T[], describe: (row: T) => Filterable): T[] {
  const { match } = useConsole();
  return rows.filter((row) => match(describe(row)));
}

interface ActionButtonsProps {
  storeId: string;
  storeLabel: string;
  actions: StorageActionMeta[];
  target?: string;
  /** Show only these action ids, in this order. */
  only?: string[];
  className?: string;
}

/** One button per cleanup offered for a store. Nothing runs without the dialog. */
export function ActionButtons({ storeId, storeLabel, actions, target, only, className }: ActionButtonsProps) {
  const { ask, running } = useConsole();
  const offered = only
    ? only.map((id) => actions.find((a) => a.id === id)).filter((a): a is StorageActionMeta => a !== undefined)
    : actions;
  const shown = offered.filter((a) => !a.needs_target || Boolean(target));
  if (shown.length === 0) return null;
  return (
    <span className={cn('inline-flex flex-wrap items-center justify-end gap-1.5', className)}>
      {shown.map((action) => {
        const key = `${storeId}/${action.id}${target && action.needs_target ? `/${target}` : ''}`;
        return (
          <button
            key={action.id}
            type="button"
            onClick={() => ask({ storeId, storeLabel, action, target: action.needs_target ? target : undefined })}
            disabled={running !== null || Boolean(action.blocked_reason)}
            title={action.blocked_reason ?? action.description}
            className={cn('sc-btn', action.safety === 'DESTRUCTIVE' && 'sc-btn--danger')}
          >
            {running === key ? 'Working…' : action.label}
          </button>
        );
      })}
    </span>
  );
}
