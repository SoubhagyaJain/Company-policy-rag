/** Shapes returned by /api/admin/storage*. A field typed `| null` is one the
 *  backend could not measure; the console shows it as unavailable, never as 0. */

export type StorageKind =
  | 'PERSISTENT'
  | 'CACHE'
  | 'REBUILDABLE'
  | 'RUNTIME_ONLY'
  | 'USER_DATA'
  | 'SYSTEM_DATA'
  | 'TEMPORARY';

export type StorageSafety = 'SAFE' | 'REBUILDABLE' | 'DESTRUCTIVE';

export interface StorageInfo {
  kinds: StorageKind[];
  category: string;
  what: string;
  why: string;
  created_by: string;
  read_by: string;
  survives_restart: boolean;
  deletable: string;
  delete_effect: string;
  rebuild: string;
  cleanup_policy: string;
}

export interface StorageImpact {
  deletes: string;
  rebuild: string;
  performance: string;
}

export interface StorageActionMeta {
  id: string;
  label: string;
  description: string;
  needs_days: boolean;
  needs_target: boolean;
  safe: boolean;
  safety: StorageSafety;
  impact: StorageImpact;
  blocked_reason: string | null;
}

export interface StorageTableRow {
  name: string;
  rows: number;
}

export interface StorageGrowth {
  d1?: number | null;
  d7?: number | null;
  d30?: number | null;
}

export interface StorageStore {
  id: string;
  label: string;
  group: 'databases' | 'caches' | 'library' | string;
  description: string;
  path: string | null;
  size_bytes: number;
  items: number | null;
  items_label: string;
  file_count: number | null;
  reclaimable_bytes: number;
  last_modified: string | null;
  kinds: StorageKind[];
  category: string;
  info: StorageInfo | null;
  growth: StorageGrowth;
  details: {
    engine?: string;
    tables?: StorageTableRow[];
    queue_rows?: number | null;
    embedding_rows?: number | null;
    chunk_rows?: number | null;
    sqlite_bytes?: number;
    hnsw_bytes?: number;
    file_bytes?: number;
    wal_bytes?: number;
    page_size?: number;
    page_count?: number;
    free_pages?: number;
    free_bytes?: number;
    bloat_pct?: number;
    oldest_record?: string | null;
    newest_record?: string | null;
    last_compaction?: string | null;
    orphaned?: number;
    largest?: Array<{ name: string; size_bytes: number; orphaned?: boolean }>;
  };
  actions: StorageActionMeta[];
}

export interface StorageLoadedModel {
  name: string;
  size_bytes: number;
  vram_bytes: number;
  context_length: number | null;
  pinned: boolean;
}

/** A chat model resident in the Ollama server. */
export interface StorageOllamaModel extends StorageLoadedModel {
  id: string;
  purpose: string;
  backend: string;
  ram_bytes: number;
  gpu_pct: number | null;
  expires_at: string | null;
  parameter_size: string | null;
  quantization: string | null;
  loaded_at: string | null;
  last_request_at: string | null;
  requests: number | null;
  requests_source: string | null;
  unloadable: boolean;
}

/** A model held inside the backend process (embedding, reranker, vision). */
export interface StorageInProcessModel {
  id: string;
  name: string;
  purpose: string;
  backend: string;
  device: string | null;
  weight_bytes: number | null;
  loaded_at: string | null;
  unloadable: boolean;
}

export interface StorageMemoryItem {
  id: string;
  label: string;
  description: string;
  value: number;
  unit: string;
  kinds: StorageKind[];
  info: StorageInfo | null;
  models?: StorageLoadedModel[];
  actions: StorageActionMeta[];
}

export interface StorageGpu {
  name: string;
  total_mb: number;
  used_mb: number;
  free_mb: number;
}

export interface StorageGpuSegment {
  id: string;
  label: string;
  kind: 'model' | 'backend' | 'other';
  bytes: number;
  source: string;
}

export interface StorageGpuBreakdown extends StorageGpu {
  segments: StorageGpuSegment[];
  note: string;
}

export interface StorageModelEntry {
  id: string;
  label: string;
  name: string;
  path: string | null;
  size_bytes: number;
  purpose: string;
  runtime: string;
  loaded: boolean;
  modified_at: string | null;
  last_used: string | null;
  parameter_size?: string | null;
  quantization?: string | null;
}

export interface StorageCache {
  id: string;
  label: string;
  kinds: StorageKind[];
  info: StorageInfo | null;
  location: 'disk' | 'memory' | 'redis';
  entries: number;
  entries_label: string;
  size_bytes: number | null;
  size_estimated: boolean;
  size_note: string | null;
  reclaimable_bytes: number;
  hits: number | null;
  misses: number | null;
  hit_rate: number | null;
  hit_rate_window: string | null;
  avg_saved_ms: number | null;
  last_hit_at: string | null;
  oldest_entry: string | null;
  newest_entry: string | null;
  ttl_seconds: number | null;
  max_entries: number | null;
  backend: string | null;
  rebuild_cost: string | null;
  actions: StorageActionMeta[];
}

export interface StorageMapSegment {
  id: string;
  label: string;
  size_bytes: number;
  reclaimable_bytes: number;
  growth_d7: number | null;
  members: Array<{ id: string; label: string; size_bytes: number; kind: 'store' | 'model' }>;
}

export interface StorageObservation {
  id: string;
  level: 'info' | 'warn' | 'critical';
  text: string;
  store_id?: string;
}

export interface StorageHealthCheck {
  id: string;
  ok: boolean;
  label: string;
}

export interface StorageIndexingJob {
  document_id: string;
  filename: string;
  status: string;
  stage: string;
  progress: number;
  pages_processed: number;
  pages_total: number;
  chunks_created: number;
  chunks_indexed: number;
  vision_status: string;
  vision_pages_processed: number;
  vision_pages_total: number;
  updated_at: string;
  stages: Array<{ stage: string; status: string }>;
}

export interface StorageOperations {
  indexing: StorageIndexingJob[];
  cleanup: { store: string; action: string; target: string | null; started_at: string } | null;
  blocks_index_cleanup: boolean;
}

export interface StorageTreeNode {
  label: string;
  store?: string;
  runtime?: string;
  model?: string;
  runtime_only?: boolean;
  children?: StorageTreeNode[];
}

export interface StorageFlowStep {
  label: string;
  detail?: string;
  store?: string;
  runtime?: string;
  model?: string;
  also?: string[];
}

export interface StorageSummary {
  generated_at: string;
  busy: boolean;
  totals: {
    disk_bytes: number;
    reclaimable_bytes: number;
    stores: number;
    databases: number;
    models_bytes: number;
    storage_bytes: number;
    growth: StorageGrowth;
    history_span_seconds: number;
    orphan_bytes: number;
    orphan_items: number;
  };
  disk: { total_bytes: number; free_bytes: number; volume: string } | null;
  stores: StorageStore[];
  memory: {
    gpu: StorageGpu | null;
    process_rss_bytes: number | null;
    system_total_bytes: number | null;
    system_available_bytes: number | null;
    gpu_breakdown: StorageGpuBreakdown | null;
    loaded_models: StorageOllamaModel[];
    in_process_models: StorageInProcessModel[];
    items: StorageMemoryItem[];
  };
  models: StorageModelEntry[];
  caches: StorageCache[];
  map: StorageMapSegment[];
  observations: StorageObservation[];
  health: StorageHealthCheck[];
  operations: StorageOperations;
  hierarchy: StorageTreeNode[];
  lifecycle: StorageFlowStep[];
  memory_flow: StorageFlowStep[];
  snapshot_interval_seconds: number;
}

export interface StorageLiveCache {
  entries: number;
  hits: number | null;
  misses: number | null;
  last_hit_at: string | null;
  backend: string | null;
}

export interface StorageLive {
  generated_at: string;
  busy: boolean;
  ram: {
    process_rss_bytes: number | null;
    system_total_bytes: number | null;
    system_available_bytes: number | null;
  };
  gpu: StorageGpu | null;
  gpu_breakdown: StorageGpuBreakdown | null;
  loaded_models: StorageOllamaModel[];
  in_process_models: StorageInProcessModel[];
  vision_loaded: boolean;
  caches: Record<string, StorageLiveCache>;
  operations: StorageOperations;
}

export interface StorageActionResult {
  id?: string;
  ts?: string;
  store?: string;
  action?: string;
  target?: string | null;
  older_than_days?: number | null;
  freed_bytes: number;
  removed_items: number;
  message?: string;
  before_bytes?: number | null;
  after_bytes?: number | null;
  duration_ms?: number;
  status?: 'success' | 'failed';
  error?: string;
  initiated_by?: string;
  safety?: StorageSafety;
  skipped?: string;
  failed?: string;
}

export type StorageAuditEntry = StorageActionResult & { ts: string; store: string; action: string };

export interface StorageCleanupResult {
  freed_bytes: number;
  removed_items: number;
  duration_ms?: number;
  results: StorageActionResult[];
}

export interface StorageSnapshot {
  ts: string;
  sizes: Record<string, number>;
  items?: Record<string, number>;
  models?: number;
}

export interface StorageSeriesPoint {
  ts: string;
  app_bytes: number;
  models_bytes: number | null;
  categories: Record<string, number>;
}

export interface StorageEvent {
  ts: string;
  kind: 'document' | 'cleanup' | 'compaction' | 'model';
  label: string;
  detail?: string;
  bytes: number;
  large?: boolean;
  store_id?: string;
  measured_delta_bytes: number | null;
}

export interface StorageForecast {
  basis_days: number;
  basis_points: number;
  bytes_per_day: number;
  current_bytes: number;
  in_7_days_bytes: number;
  in_30_days_bytes: number;
  disk_free_bytes: number | null;
  method: string;
}

export type StorageRange = '24h' | '7d' | '30d';

export interface StorageHistory {
  range?: StorageRange;
  snapshots: StorageSnapshot[];
  actions: StorageAuditEntry[];
  series?: StorageSeriesPoint[];
  events?: StorageEvent[];
  forecast?: StorageForecast | null;
  span_seconds?: number;
  snapshot_interval_seconds?: number;
}

export interface StorageDocumentRow {
  document_id: string;
  filename: string;
  status: string;
  storage_state: string;
  created_at: string | null;
  pages: number;
  file_bytes: number;
  chunks: number;
  embeddings: number;
  images: { count: number; bytes: number };
  vision_cache: { entries: number; bytes: number };
  index_bytes_estimated: number | null;
  total_bytes_estimated: number;
  query_count: number;
  last_queried: string | null;
}

export interface StorageOrphanGroup {
  count: number;
  files?: number;
  bytes: number;
  items: Array<{ name: string; bytes: number; files?: number; entries?: number }>;
  store_id: string;
  action_id: string;
  safety: StorageSafety;
}

export interface StorageDocuments {
  generated_at: string;
  vector_dim: number | null;
  documents: StorageDocumentRow[];
  notes: { index_bytes: string; last_queried: string };
  orphans: {
    total_bytes: number;
    reason: string;
    images: StorageOrphanGroup;
    vision_cache: StorageOrphanGroup;
    sessions: StorageOrphanGroup;
    chunk_references: {
      bm25: number;
      docstore: number;
      vector: number | null;
      vector_checked_at: string | null;
      report_only: boolean;
    };
  };
}

export interface StorageFilesystem {
  logical_name: string;
  path: string | null;
  engine: string | null;
  kinds: StorageKind[];
  size_bytes: number;
  file_count: number | null;
  created: string | null;
  last_modified: string | null;
  last_accessed: string | null;
  rebuildable: boolean;
  cleanup_policy: string | null;
}

export interface StorageSqliteStats {
  file_bytes?: number | null;
  wal_bytes?: number | null;
  page_size?: number | null;
  page_count?: number | null;
  free_pages?: number | null;
  free_bytes?: number | null;
  bloat_pct?: number | null;
}

export interface StorageInspect {
  id: string;
  label: string;
  info: StorageInfo | null;
  filesystem: StorageFilesystem | null;
  largest?: Array<{ name: string; size_bytes: number }>;
  error?: string;
  collections?: Array<{
    name: string;
    role: string;
    rows: number;
    vector_dim: number | null;
    embedding_bytes_estimated: number | null;
  }>;
  documents?: {
    count: number;
    chunks: number;
    avg_chunk_chars: number | null;
    largest: Array<{ document_id: string; filename: string; chunks: number }>;
    oldest: { filename: string; created_at: string } | null;
    newest: { filename: string; created_at: string } | null;
  };
  sqlite?: StorageSqliteStats;
  segments?: { sqlite_bytes?: number | null; hnsw_bytes?: number | null };
  tables?: Array<{ name: string; rows: number; oldest: string | null; newest: string | null }>;
  semantic?: {
    entries: number | null;
    note?: string;
    oldest_entry?: string | null;
    newest_entry?: string | null;
    by_model?: Array<{ model: string; entries: number }>;
    for_current_library?: number | null;
  };
}

export interface StoragePreview {
  store: string;
  action: string;
  older_than_days: number | null;
  affected_items: number | null;
  estimated_bytes: number | null;
  exact: boolean;
  note: string;
  safety: StorageSafety;
  impact: StorageImpact;
  blocked_reason: string | null;
}

export interface StorageCleanupItem {
  store_id: string;
  action_id: string;
  store_label: string;
  label: string;
  safety: StorageSafety;
  impact: StorageImpact;
  blocked_reason: string | null;
  needs_days: boolean;
  default_days: number | null;
  affected_items: number | null;
  estimated_bytes: number | null;
  exact: boolean;
  note: string;
  has_effect: boolean;
  selected: boolean;
}

export interface StorageCleanupPlan {
  generated_at: string;
  busy: boolean;
  items: StorageCleanupItem[];
  safe_reclaimable_bytes: number;
}

export interface StorageCleanupSelection {
  store_id: string;
  action_id: string;
  older_than_days?: number;
}
