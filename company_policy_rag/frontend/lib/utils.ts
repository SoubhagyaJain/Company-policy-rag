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
