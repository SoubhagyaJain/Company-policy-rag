import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  StorageActionResult,
  StorageCleanupResult,
  StorageCleanupSelection,
  StorageDocuments,
  StorageHistory,
  StorageLive,
  StorageRange,
  StorageSummary,
} from '../lib/types';
import { apiClient } from '../lib/api-client';

// Directory sizes are cached server-side for a minute, so asking more often buys nothing.
const SUMMARY_INTERVAL_MS = 60_000;
// RAM, VRAM, loaded models and running operations move on their own.
const LIVE_INTERVAL_MS = 5_000;

/** Outcome of the last cleanup, shown as a toast and mirrored in the audit table. */
export interface StorageToast {
  key: number;
  title: string;
  ok: boolean;
  beforeBytes?: number | null;
  afterBytes?: number | null;
  freedBytes: number;
  removedItems: number;
  durationMs?: number;
  detail?: string;
}

export function useStorage() {
  const [summary, setSummary] = useState<StorageSummary | null>(null);
  const [live, setLive] = useState<StorageLive | null>(null);
  const [history, setHistory] = useState<StorageHistory>({ snapshots: [], actions: [] });
  const [documents, setDocuments] = useState<StorageDocuments | null>(null);
  const [range, setRangeState] = useState<StorageRange>('7d');
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<StorageToast | null>(null);
  // "store/action" (or "cleanup") of the request in flight, so its button can show progress.
  const [running, setRunning] = useState<string | null>(null);
  const activeRef = useRef(true);
  const rangeRef = useRef<StorageRange>('7d');

  const refreshLive = useCallback(async () => {
    try {
      const next = await apiClient.getStorageLive();
      if (activeRef.current) setLive(next);
    } catch {
      // The summary request reports a disconnected backend; a missed poll is not an error.
    }
  }, []);

  const refresh = useCallback(async (options: { silent?: boolean; deep?: boolean } = {}) => {
    const { silent = false, deep = false } = options;
    if (!silent) setLoading(true);
    try {
      const [nextSummary, nextHistory, nextDocuments] = await Promise.allSettled([
        apiClient.getStorage(deep),
        apiClient.getStorageHistory(rangeRef.current),
        apiClient.getStorageDocuments(),
      ]);
      if (!activeRef.current) return;
      if (nextSummary.status === 'rejected') throw nextSummary.reason;
      setSummary(nextSummary.value);
      if (nextHistory.status === 'fulfilled') setHistory(nextHistory.value);
      if (nextDocuments.status === 'fulfilled') setDocuments(nextDocuments.value);
      setError(null);
    } catch (err) {
      if (!activeRef.current) return;
      setError(err instanceof Error ? err.message : 'Failed to load storage information');
    } finally {
      if (activeRef.current && !silent) setLoading(false);
    }
  }, []);

  useEffect(() => {
    activeRef.current = true;
    void refresh();
    void refreshLive();
    const summaryTimer = setInterval(() => {
      if (!document.hidden) void refresh({ silent: true });
    }, SUMMARY_INTERVAL_MS);
    const liveTimer = setInterval(() => {
      if (!document.hidden) void refreshLive();
    }, LIVE_INTERVAL_MS);
    const onVisible = () => {
      if (!document.hidden) void refreshLive();
    };
    document.addEventListener('visibilitychange', onVisible);
    return () => {
      activeRef.current = false;
      clearInterval(summaryTimer);
      clearInterval(liveTimer);
      document.removeEventListener('visibilitychange', onVisible);
    };
  }, [refresh, refreshLive]);

  const setRange = useCallback(async (next: StorageRange) => {
    rangeRef.current = next;
    setRangeState(next);
    try {
      const nextHistory = await apiClient.getStorageHistory(next);
      if (activeRef.current) setHistory(nextHistory);
    } catch {
      // Keep the chart that is already drawn.
    }
  }, []);

  const afterChange = useCallback(async () => {
    await Promise.all([refresh({ silent: true }), refreshLive()]);
  }, [refresh, refreshLive]);

  const runAction = useCallback(
    async (
      storeId: string,
      actionId: string,
      options: { olderThanDays?: number; target?: string; title?: string } = {},
    ): Promise<StorageActionResult | null> => {
      setRunning(`${storeId}/${actionId}${options.target ? `/${options.target}` : ''}`);
      setError(null);
      try {
        const result = await apiClient.runStorageAction(storeId, actionId, options);
        setToast({
          key: Date.now(),
          title: result.message ?? options.title ?? 'Done.',
          ok: true,
          beforeBytes: result.before_bytes,
          afterBytes: result.after_bytes,
          freedBytes: result.freed_bytes,
          removedItems: result.removed_items,
          durationMs: result.duration_ms,
        });
        await afterChange();
        return result;
      } catch (err) {
        setToast({
          key: Date.now(),
          title: `${options.title ?? 'Storage action'} failed`,
          ok: false,
          freedBytes: 0,
          removedItems: 0,
          detail: err instanceof Error ? err.message : 'Storage action failed',
        });
        await afterChange();
        return null;
      } finally {
        setRunning(null);
      }
    },
    [afterChange],
  );

  const runCleanup = useCallback(
    async (items?: StorageCleanupSelection[]): Promise<StorageCleanupResult | null> => {
      setRunning('cleanup');
      setError(null);
      try {
        const result = await apiClient.cleanStorage(items);
        const skipped = result.results.filter((r) => r.skipped).length;
        const failed = result.results.filter((r) => r.failed).length;
        const done = result.results.length - skipped - failed;
        const notes = [
          skipped ? `${skipped} skipped while a document is being indexed` : '',
          failed ? `${failed} failed` : '',
        ].filter(Boolean);
        setToast({
          key: Date.now(),
          title: `Cleanup ran ${done} ${done === 1 ? 'step' : 'steps'}`,
          ok: failed === 0,
          freedBytes: result.freed_bytes,
          removedItems: result.removed_items,
          durationMs: result.duration_ms,
          detail: notes.join(' · ') || undefined,
        });
        await afterChange();
        return result;
      } catch (err) {
        setToast({
          key: Date.now(),
          title: 'Cleanup failed',
          ok: false,
          freedBytes: 0,
          removedItems: 0,
          detail: err instanceof Error ? err.message : 'Storage cleanup failed',
        });
        return null;
      } finally {
        setRunning(null);
      }
    },
    [afterChange],
  );

  /** Count vector rows whose document is gone. Reads every row, so only on request. */
  const scanVectorOrphans = useCallback(async () => {
    setRunning('deep-scan');
    try {
      const next = await apiClient.getStorageDocuments(true);
      if (activeRef.current) setDocuments(next);
    } catch (err) {
      if (activeRef.current) setError(err instanceof Error ? err.message : 'Deep scan failed');
    } finally {
      setRunning(null);
    }
  }, []);

  return {
    summary,
    live,
    history,
    documents,
    range,
    loading,
    error,
    toast,
    running,
    refresh,
    setRange,
    runAction,
    runCleanup,
    scanVectorOrphans,
    dismissToast: () => setToast(null),
  };
}

export type StorageController = ReturnType<typeof useStorage>;
