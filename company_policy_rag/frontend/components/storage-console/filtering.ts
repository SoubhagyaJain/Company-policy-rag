/** Search and filter predicates shared by every section of the storage console. */

import type {
  StorageAuditEntry,
  StorageCache,
  StorageInProcessModel,
  StorageInfo,
  StorageKind,
  StorageMemoryItem,
  StorageModelEntry,
  StorageOllamaModel,
  StorageStore,
} from '@/lib/types';

export type StorageFilter =
  | 'all'
  | 'persistent'
  | 'runtime'
  | 'cache'
  | 'database'
  | 'model'
  | 'rebuildable'
  | 'reclaimable'
  | 'orphaned'
  | 'large';

export const FILTERS: Array<{ id: StorageFilter; label: string }> = [
  { id: 'all', label: 'All' },
  { id: 'persistent', label: 'Persistent' },
  { id: 'runtime', label: 'Runtime' },
  { id: 'cache', label: 'Cache' },
  { id: 'database', label: 'Database' },
  { id: 'model', label: 'Model' },
  { id: 'rebuildable', label: 'Rebuildable' },
  { id: 'reclaimable', label: 'Reclaimable' },
  { id: 'orphaned', label: 'Orphaned' },
  { id: 'large', label: 'Large > 1 GB' },
];

const GIB = 1024 ** 3;

/** What a row exposes to search and filters. */
export interface Filterable {
  text: string;
  kinds?: StorageKind[];
  database?: boolean;
  model?: boolean;
  runtime?: boolean;
  persistent?: boolean;
  sizeBytes?: number | null;
  reclaimableBytes?: number;
  orphaned?: number;
}

export type Matcher = (item: Filterable) => boolean;

function passesFilter(item: Filterable, filter: StorageFilter): boolean {
  const kinds = item.kinds ?? [];
  switch (filter) {
    case 'all':
      return true;
    case 'persistent':
      return item.persistent ?? (kinds.includes('PERSISTENT') || kinds.includes('USER_DATA'));
    case 'runtime':
      return item.runtime ?? kinds.includes('RUNTIME_ONLY');
    case 'cache':
      return kinds.includes('CACHE');
    case 'database':
      return Boolean(item.database);
    case 'model':
      return Boolean(item.model);
    case 'rebuildable':
      return kinds.includes('REBUILDABLE');
    case 'reclaimable':
      return (item.reclaimableBytes ?? 0) > 0;
    case 'orphaned':
      return (item.orphaned ?? 0) > 0;
    case 'large':
      return (item.sizeBytes ?? 0) >= GIB;
  }
}

export function makeMatcher(query: string, filter: StorageFilter): Matcher {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  return (item) => {
    if (!passesFilter(item, filter)) return false;
    if (terms.length === 0) return true;
    const haystack = item.text.toLowerCase();
    return terms.every((term) => haystack.includes(term));
  };
}

const infoText = (info: StorageInfo | null | undefined) =>
  info ? [info.what, info.why, info.created_by, info.read_by, info.cleanup_policy].join(' ') : '';

export const storeItem = (store: StorageStore): Filterable => ({
  text: [store.id, store.label, store.description, store.path ?? '', store.details.engine ?? '', infoText(store.info)].join(' '),
  kinds: store.kinds,
  database: store.group === 'databases',
  sizeBytes: store.size_bytes,
  reclaimableBytes: store.reclaimable_bytes,
  orphaned: store.details.orphaned,
});

export const cacheItem = (cache: StorageCache): Filterable => ({
  text: [cache.id, cache.label, cache.backend ?? '', cache.location, infoText(cache.info)].join(' '),
  kinds: cache.kinds.includes('CACHE') ? cache.kinds : [...cache.kinds, 'CACHE'],
  runtime: cache.location !== 'disk',
  persistent: cache.location === 'disk',
  sizeBytes: cache.size_bytes,
  reclaimableBytes: cache.reclaimable_bytes,
});

export const memoryItem = (item: StorageMemoryItem): Filterable => ({
  text: [item.id, item.label, item.description, infoText(item.info)].join(' '),
  kinds: item.kinds,
  runtime: true,
  persistent: false,
  model: item.id === 'ollama_models' || item.id === 'vision_model',
});

export const loadedModelItem = (model: StorageOllamaModel | StorageInProcessModel): Filterable => ({
  text: [model.name, model.purpose, model.backend, 'model loaded memory'].join(' '),
  kinds: ['RUNTIME_ONLY'],
  model: true,
  runtime: true,
  persistent: false,
  sizeBytes: 'vram_bytes' in model ? model.size_bytes : model.weight_bytes,
});

export const installedModelItem = (model: StorageModelEntry): Filterable => ({
  text: [model.name, model.label, model.purpose, model.runtime, 'model'].join(' '),
  kinds: ['PERSISTENT'],
  model: true,
  persistent: true,
  runtime: false,
  sizeBytes: model.size_bytes,
});

export const auditItem = (entry: StorageAuditEntry, storeLabel: string): Filterable => ({
  text: [entry.store, storeLabel, entry.action, entry.message ?? '', entry.target ?? '', entry.status ?? ''].join(' '),
});
