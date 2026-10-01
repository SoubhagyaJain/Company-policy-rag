import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatBytes(bytes: number | undefined | null, decimals = 1): string {
  if (bytes === undefined || bytes === null || isNaN(bytes) || bytes === 0) return '0 B';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(Math.abs(bytes)) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

export function formatDate(isoString: string): string {
  if (!isoString) return '';
  try {
    const date = new Date(isoString);
    if (isNaN(date.getTime())) return isoString;
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  } catch {
    return isoString;
  }
}

/** "18s ago", "42m ago", "3d ago"; an em dash when there is no timestamp. */
export function formatRelativeTime(isoString: string | null | undefined, now: number = Date.now()): string {
  if (!isoString) return '—';
  const then = new Date(isoString).getTime();
  if (isNaN(then)) return '—';
  const seconds = Math.max(0, Math.round((now - then) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

/** Compact duration: "850 ms", "2.1 s", "4m 12s", "3h 05m", "12 d". */
export function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || isNaN(ms)) return '—';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  const seconds = ms / 1000;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${String(Math.floor(seconds % 60)).padStart(2, '0')}s`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ${String(Math.floor((seconds % 3600) / 60)).padStart(2, '0')}m`;
  return `${Math.round(seconds / 86400)} d`;
}

export function formatPercent(fraction: number | null | undefined, decimals = 0): string {
  if (fraction === null || fraction === undefined || isNaN(fraction)) return '—';
  return `${(fraction * 100).toFixed(decimals)}%`;
}

/** "1 vector", "3 vectors", "1 entry": a count with its unit in the right number. */
export function formatCount(value: number, unit: string): string {
  const singular = unit.endsWith('ies') ? `${unit.slice(0, -3)}y` : unit.endsWith('s') ? unit.slice(0, -1) : unit;
  return `${value.toLocaleString()} ${value === 1 ? singular : unit}`;
}

/** Signed byte change: "+1.2 GB", "−340 MB", "±0". */
export function formatDelta(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || isNaN(bytes)) return '—';
  if (bytes === 0) return '±0';
  return `${bytes > 0 ? '+' : '−'}${formatBytes(Math.abs(bytes))}`;
}

export function formatScore(score: number | undefined | null): string {
  if (score === undefined || score === null || isNaN(score)) return 'N/A';
  let val = score;
  if (val > 1.0) {
    val = 1.0 / (1.0 + Math.exp(-val));
  } else if (val < 0.0) {
    val = 1.0 / (1.0 + Math.exp(-val));
  }
  const pct = Math.round(Math.min(99, Math.max(10, val * 100)));
  return `${pct}%`;
}

export function formatLatency(ms: number): string {
  if (!ms && ms !== 0) return 'N/A';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function generateId(prefix = 'id'): string {
  return `${prefix}_${Math.random().toString(36).substring(2, 9)}_${Date.now()}`;
}
