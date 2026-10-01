import { useState, useEffect, useCallback, useRef } from 'react';
import { StorageHistory, StorageSummary } from '../lib/types';
import { apiClient } from '../lib/api-client';
import { formatBytes } from '../lib/utils';

const REFRESH_INTERVAL_MS = 20_000;

export function useStorage() {
  const [summary, setSummary] = useState<StorageSummary | null>(null);
  const [history, setHistory] = useState<StorageHistory>({ snapshots: [], actions: [] });
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  // "store/action" (or "cleanup") of the request in flight, so its button can show progress.
  const [running, setRunning] = useState<string | null>(null);
  const activeRef = useRef(true);

  const refresh = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      const [nextSummary, nextHistory] = await Promise.all([
        apiClient.getStorage(),
        apiClient.getStorageHistory(),
      ]);
      if (!activeRef.current) return;
      setSummary(nextSummary);
      setHistory(nextHistory);
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
    // Memory figures (GPU, loaded models, cache entries) move without any action here.
    const timer = setInterval(() => void refresh(true), REFRESH_INTERVAL_MS);
    return () => {
      activeRef.current = false;
      clearInterval(timer);
    };
  }, [refresh]);

  const runAction = useCallback(
    async (
      storeId: string,
      actionId: string,
      options: { olderThanDays?: number; target?: string } = {},
    ): Promise<boolean> => {
      setRunning(`${storeId}/${actionId}${options.target ? `/${options.target}` : ''}`);
      setError(null);
      setNotice(null);
      try {
        const result = await apiClient.runStorageAction(storeId, actionId, options);
        const freed = result.freed_bytes > 0 ? ` Freed ${formatBytes(result.freed_bytes)}.` : '';
        setNotice(`${result.message ?? 'Done.'}${freed}`);
        await refresh(true);
        return true;
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Storage action failed');
        return false;
      } finally {
        setRunning(null);
      }
    },
    [refresh],
  );

  const cleanSafe = useCallback(async (): Promise<boolean> => {
    setRunning('cleanup');
    setError(null);
    setNotice(null);
    try {
      const result = await apiClient.cleanStorage();
      const skipped = result.results.filter((r) => r.skipped).length;
      setNotice(
        `Cleanup freed ${formatBytes(result.freed_bytes)} and removed ${result.removed_items} items.` +
          (skipped ? ` ${skipped} step skipped while a document is being indexed.` : ''),
      );
      await refresh(true);
      return true;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Storage cleanup failed');
      return false;
    } finally {
      setRunning(null);
    }
  }, [refresh]);

  return {
    summary,
    history,
    loading,
    error,
    notice,
    running,
    refresh,
    runAction,
    cleanSafe,
    dismissNotice: () => setNotice(null),
  };
}
