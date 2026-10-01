'use client';

/** What lives in memory: backend RAM, the runtime-only stores, GPU memory and
 *  the models holding it. Everything here is refreshed by the live poll. */

import React, { useState } from 'react';
import { ArrowRight, Cpu, MemoryStick } from 'lucide-react';

import { cn, formatBytes, formatCount, formatDate, formatPercent, formatRelativeTime } from '@/lib/utils';
import type {
  StorageCache,
  StorageFlowStep,
  StorageInProcessModel,
  StorageIndexingJob,
  StorageMemoryItem,
  StorageModelEntry,
  StorageOllamaModel,
} from '@/lib/types';
import { ActionButtons, useConsole, useRowClass, useVisible, type RuntimeState } from './context';
import { loadedModelItem, memoryItem } from './filtering';
import { Overlay } from './Overlay';
import {
  BarLegend,
  EmptyState,
  Estimated,
  Field,
  HoverCard,
  InfoButton,
  KindBadges,
  MicroBar,
  SegmentBar,
  StatusDot,
  Unavailable,
  usageTone,
  toneColor,
  type BarSegment,
} from './primitives';

const SUBGRID_ROW = 'lg:col-span-full lg:grid lg:grid-cols-subgrid';
const NOT_MEASURED = 'Held in backend RAM. Its size is not currently measured.';

/* ── Active operations ───────────────────────────────────────────────────── */

const STAGE_LABELS: Record<string, string> = {
  UPLOAD: 'Upload',
  TEXT_EXTRACTION: 'Text extraction',
  SECTION_DETECTION: 'Sections',
  CHUNKING: 'Chunking',
  EMBEDDINGS: 'Embedding',
  VECTOR_INDEX: 'Database writes',
  BM25_INDEX: 'Keyword index',
  FINALIZING: 'Finalizing',
};

const stageDot = (status: string) =>
  status === 'COMPLETED' ? 'ok' : status === 'IN_PROGRESS' ? 'active' : status === 'FAILED' ? 'crit' : 'idle';

const stageWord = (status: string) =>
  status === 'COMPLETED' ? 'Finished' : status === 'IN_PROGRESS' ? 'Active' : status === 'FAILED' ? 'Failed' : 'Waiting';

function IndexingJob({ job }: { job: StorageIndexingJob }) {
  const reading = job.status === 'VISION_PROCESSING';
  return (
    <li className="px-4 py-3">
      <div className="flex items-baseline justify-between gap-3">
        <p className="sc-t1 truncate text-[12.5px] font-semibold">
          {reading ? 'Reading page images of' : 'Indexing'} <span className="sc-num font-medium">{job.filename}</span>
        </p>
        <p className="sc-num sc-t1 shrink-0 text-[13px] font-semibold">{job.progress}%</p>
      </div>
      <MicroBar value={job.progress} max={100} className="mt-1.5" height={5} />
      <div className="sc-num sc-t2 mt-1.5 flex flex-wrap gap-x-4 gap-y-0.5 text-[11.5px]">
        {job.chunks_created > 0 && (
          <span>
            {job.chunks_indexed.toLocaleString()} / {job.chunks_created.toLocaleString()} chunks
          </span>
        )}
        {job.pages_total > 0 && (
          <span>
            {job.pages_processed.toLocaleString()} / {job.pages_total.toLocaleString()} pages
          </span>
        )}
        {job.vision_pages_total > 0 && (
          <span>
            {job.vision_pages_processed.toLocaleString()} / {job.vision_pages_total.toLocaleString()} pages read
          </span>
        )}
      </div>
      {job.stages.length > 0 && (
        <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
          {job.stages
            .filter((s) => STAGE_LABELS[s.stage])
            .map((s) => (
              <li key={s.stage} className="flex items-center gap-1.5 text-[11.5px]">
                <StatusDot state={stageDot(s.status)} />
                <span className="sc-t2">{STAGE_LABELS[s.stage]}</span>
                <span className="sc-t3">{stageWord(s.status)}</span>
              </li>
            ))}
        </ul>
      )}
    </li>
  );
}

/** Running work, with the reason index cleanups are held back. Hidden when idle. */
export function ActiveOperations({ runtime }: { runtime: RuntimeState }) {
  const { indexing, cleanup, blocks_index_cleanup } = runtime.operations;
  if (indexing.length === 0 && !cleanup) return null;
  return (
    <div className="sc-panel">
      <p className="sc-label px-4 pt-3">Active operations</p>
      <ul className="sc-divide">
        {indexing.map((job) => (
          <IndexingJob key={job.document_id} job={job} />
        ))}
        {cleanup && (
          <li className="flex items-center gap-2 px-4 py-3 text-[12.5px]">
            <StatusDot state="active" />
            <span className="sc-t1 font-semibold">{cleanup.action === 'compact' ? 'Compacting' : 'Cleaning'}</span>
            <span className="sc-num sc-t2">{cleanup.target ?? cleanup.store}</span>
          </li>
        )}
      </ul>
      {blocks_index_cleanup && (
        <p className="sc-rule sc-t2 px-4 py-2 text-[12px]">
          Vector compaction and cache clears that touch the indexes are unavailable while document indexing is active.
        </p>
      )}
    </div>
  );
}

/* ── Runtime memory ──────────────────────────────────────────────────────── */

interface RuntimeMemoryProps {
  items: StorageMemoryItem[];
  caches: StorageCache[];
  flow: StorageFlowStep[];
  runtime: RuntimeState;
}

export function RuntimeMemorySection({ items, caches, flow, runtime }: RuntimeMemoryProps) {
  const { advanced, explain } = useConsole();
  const rowClass = useRowClass();
  const { processRss, systemTotal, systemAvailable, loadedModels, inProcessModels } = runtime;

  const cpuWeights = inProcessModels
    .filter((m) => m.device !== 'cuda')
    .reduce((sum, m) => sum + (m.weight_bytes ?? 0), 0);
  const ollamaRam = loadedModels.reduce((sum, m) => sum + m.ram_bytes, 0);
  const systemUsed = systemTotal !== null && systemAvailable !== null ? systemTotal - systemAvailable : null;

  const ramSegments: BarSegment[] =
    processRss !== null && systemTotal !== null
      ? [
          { id: 'weights', label: 'Model weights in the backend', value: Math.min(cpuWeights, processRss), color: 'var(--sp-lavender)' },
          { id: 'backend', label: 'Backend process, other', value: Math.max(0, processRss - cpuWeights), color: 'var(--sp-accent)' },
          { id: 'ollama', label: 'Ollama models in system RAM', value: ollamaRam, color: 'var(--sc-cat-models)' },
          {
            id: 'system',
            label: 'Other system use',
            value: systemUsed !== null ? Math.max(0, systemUsed - processRss - ollamaRam) : 0,
            color: 'var(--sp-text-faint)',
          },
        ]
      : [];

  const embeddingCache = caches.find((c) => c.id === 'embedding_cache');
  const vision = inProcessModels.find((m) => m.id === 'vision');

  const live = (item: StorageMemoryItem): { value: number; unit: string } => {
    if (item.id === 'ollama_models') return { value: loadedModels.length, unit: 'loaded' };
    if (item.id === 'vision_model') return { value: vision ? 1 : 0, unit: vision ? `loaded on ${vision.device ?? 'cpu'}` : 'not loaded' };
    const counter = runtime.caches[item.id];
    return { value: counter ? counter.entries : item.value, unit: item.unit };
  };

  const memoryOf = (item: StorageMemoryItem): React.ReactNode => {
    if (item.id === 'ollama_models') {
      if (loadedModels.length === 0) return <span className="sc-t3">—</span>;
      const total = loadedModels.reduce((sum, m) => sum + m.size_bytes, 0);
      const vram = loadedModels.reduce((sum, m) => sum + m.vram_bytes, 0);
      return (
        <span title="Reported by Ollama">
          {formatBytes(total)} <span className="sc-t3">· {formatBytes(vram)} VRAM</span>
        </span>
      );
    }
    if (item.id === 'vision_model') {
      if (!vision) return <span className="sc-t3">—</span>;
      return vision.weight_bytes !== null ? (
        <span title="Size of the model's parameter tensors">{formatBytes(vision.weight_bytes)} weights</span>
      ) : (
        <Unavailable reason="The runtime did not report the weights' size.">Not measured</Unavailable>
      );
    }
    if (item.id === 'embedding_cache' && embeddingCache?.size_bytes) {
      return <Estimated>{formatBytes(embeddingCache.size_bytes)}</Estimated>;
    }
    return <Unavailable reason={NOT_MEASURED}>Not currently measured</Unavailable>;
  };

  const visible = useVisible(items, memoryItem);

  const flowFigure = (step: StorageFlowStep): { figure: string; detail: string } | null => {
    if (step.runtime) {
      const item = items.find((i) => i.id === step.runtime);
      if (!item) return null;
      const { value, unit } = live(item);
      return { figure: formatCount(value, unit), detail: item.description };
    }
    if (step.model) {
      const model = inProcessModels.find((m) => m.id === step.model);
      if (!model) return { figure: 'not loaded', detail: 'The reranker is loaded on first use, or disabled.' };
      return {
        figure: model.weight_bytes !== null ? `${formatBytes(model.weight_bytes)} on ${model.device ?? 'cpu'}` : `on ${model.device ?? 'cpu'}`,
        detail: `${model.name} · ${model.backend}`,
      };
    }
    return null;
  };

  return (
    <div className="space-y-4">
      <p className="sc-boundary sc-t2 pt-3 text-[12.5px]">
        Unlike disk storage, most of this disappears when the backend or model process stops.
      </p>

      <div className="sc-panel px-4 py-3.5">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <p className="sc-label">System RAM</p>
          {processRss !== null && systemTotal !== null ? (
            <p className="sc-num sc-t1 text-[13px]">
              <span className="font-semibold">{formatBytes(processRss)}</span>
              <span className="sc-t3"> backend · </span>
              {systemUsed !== null ? formatBytes(systemUsed) : '—'}
              <span className="sc-t3"> of {formatBytes(systemTotal)} in use</span>
            </p>
          ) : (
            <Unavailable reason="psutil is not installed in the backend environment.">Not measured</Unavailable>
          )}
        </div>
        {ramSegments.length > 0 && systemTotal !== null && (
          <>
            <div className="mt-2.5">
              <SegmentBar segments={ramSegments} total={systemTotal} height={12} ariaLabel="System RAM by consumer" />
            </div>
            <BarLegend segments={ramSegments.filter((s) => s.value > 0)} className="mt-2" />
            <p className="sc-t3 mt-1.5 text-[11.5px]">
              Weights are the loaded models&apos; parameter tensors. The Python heap is not measured separately.
            </p>
          </>
        )}
      </div>

      {visible.length > 0 && (
        <ul className="sc-panel sc-divide lg:grid lg:grid-cols-[minmax(0,1.7fr)_140px_180px_auto] lg:gap-x-3">
          {visible.map((item) => {
            const { value, unit } = live(item);
            return (
              <li key={item.id} className={rowClass(SUBGRID_ROW)}>
                <div className={cn('grid grid-cols-2 items-center gap-x-3 gap-y-1.5 px-4 py-2.5', SUBGRID_ROW)}>
                  <div className="col-span-2 min-w-0 lg:col-span-1">
                    <div className="flex items-center gap-1.5">
                      <span className="sc-t1 truncate text-[12.5px] font-semibold">{item.label}</span>
                      <InfoButton label={item.label} onClick={() => explain(item.id)} />
                      <KindBadges kinds={item.kinds} limit={2} />
                    </div>
                    <p className="sc-t3 mt-0.5 truncate text-[11.5px]" title={item.description}>
                      {item.description}
                    </p>
                  </div>
                  <p className="sc-num sc-t1 text-[12.5px] lg:text-right">
                    {item.id === 'vision_model' ? unit : formatCount(value, unit)}
                  </p>
                  <p className="sc-num sc-t1 text-[12.5px] lg:text-right">{memoryOf(item)}</p>
                  {/* Models are unloaded from their cards below, where the target is explicit. */}
                  {item.id === 'ollama_models' || item.id === 'vision_model' ? (
                    <span className="max-lg:hidden" />
                  ) : (
                    <ActionButtons
                      storeId={item.id}
                      storeLabel={item.label}
                      actions={item.actions}
                      className="col-span-2 justify-self-end lg:col-span-1"
                    />
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}

      {advanced && flow.length > 0 && (
        <div className="sc-panel px-4 py-3.5">
          <p className="sc-label">Memory along a query</p>
          <ol className="mt-2.5 flex flex-wrap items-stretch gap-y-2">
            {flow.map((step, index) => {
              const figure = flowFigure(step);
              const node = (
                <span
                  tabIndex={figure ? 0 : undefined}
                  className={cn(
                    'sc-focus flex h-full min-w-[104px] flex-col justify-center rounded-md border px-2.5 py-1.5',
                    figure ? 'border-[color:var(--sp-card-border)]' : 'border-dashed border-[color:var(--sp-hairline)]',
                  )}
                >
                  <span className="sc-t1 text-[12px] font-medium">{step.label}</span>
                  {figure && <span className="sc-num sc-t3 text-[10.5px]">{figure.figure}</span>}
                </span>
              );
              return (
                <li key={step.label} className="flex items-center">
                  {figure ? (
                    <HoverCard trigger={node} align={index > flow.length / 2 ? 'right' : 'left'}>
                      <span className="sc-t1 block text-[12px] font-semibold">{step.label}</span>
                      <span className="sc-num sc-t1 mt-1 block">{figure.figure}</span>
                      <span className="sc-t2 mt-1 block leading-relaxed">{figure.detail}</span>
                    </HoverCard>
                  ) : (
                    node
                  )}
                  {index < flow.length - 1 && <ArrowRight className="sc-t3 mx-1.5 h-3 w-3 shrink-0" aria-hidden />}
                </li>
              );
            })}
          </ol>
        </div>
      )}
    </div>
  );
}

/* ── GPU and loaded models ───────────────────────────────────────────────── */

const SEGMENT_COLOR: Record<string, string> = {
  model: 'var(--sc-cat-models)',
  backend: 'var(--sp-lavender)',
  other: 'var(--sp-text-faint)',
};

type AnyModel = StorageOllamaModel | StorageInProcessModel;
const isOllama = (model: AnyModel): model is StorageOllamaModel => 'vram_bytes' in model;

function ModelCard({
  model,
  unloadAction,
  onDetails,
}: {
  model: AnyModel;
  unloadAction: { storeId: string; storeLabel: string; actions: StorageMemoryItem['actions']; target?: string } | null;
  onDetails: () => void;
}) {
  const { now, searching } = useConsole();
  const ollama = isOllama(model);
  return (
    <article className={cn('sc-panel flex flex-col', searching && 'sc-match')}>
      <header className="px-4 pt-3">
        <p className="sc-label">
          {model.purpose} model · {model.backend}
        </p>
        <h3 className="sc-num sc-t1 mt-1 truncate text-[13.5px] font-semibold" title={model.name}>
          {model.name}
        </h3>
      </header>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-2.5 px-4 py-3">
        {ollama ? (
          <>
            <Field label="VRAM">{formatBytes(model.vram_bytes)}</Field>
            <Field label="System RAM">{model.ram_bytes > 0 ? formatBytes(model.ram_bytes) : '0 B'}</Field>
            <Field label="On GPU" title="Share of the model that is resident in VRAM">
              {model.gpu_pct !== null ? `${model.gpu_pct}%` : <Unavailable>—</Unavailable>}
            </Field>
            <Field label="Context">{model.context_length ? model.context_length.toLocaleString() : <Unavailable>—</Unavailable>}</Field>
            <Field label="Loaded since">
              <Unavailable reason="Ollama does not report when a model was loaded.">Not exposed by runtime</Unavailable>
            </Field>
            <Field label="Last request" title={model.last_request_at ? `${formatDate(model.last_request_at)} · from telemetry` : undefined}>
              {model.last_request_at ? formatRelativeTime(model.last_request_at, now) : <Unavailable>Not recorded</Unavailable>}
            </Field>
            <Field label="Requests served" title="Answered questions recorded in telemetry for this model">
              {model.requests !== null ? model.requests.toLocaleString() : <Unavailable>Not recorded</Unavailable>}
            </Field>
            <Field label="Pinned" title={model.expires_at ? `Released at ${formatDate(model.expires_at)}` : undefined}>
              {model.pinned ? 'Yes' : model.expires_at ? `No · until ${formatRelativeTime(model.expires_at, now).replace(' ago', '')}` : 'No'}
            </Field>
          </>
        ) : (
          <>
            <Field label="Weights" title="Size of the model's parameter tensors">
              {model.weight_bytes !== null ? formatBytes(model.weight_bytes) : <Unavailable>Not measured</Unavailable>}
            </Field>
            <Field label="Device">{model.device === 'cuda' ? 'GPU (cuda)' : (model.device ?? 'cpu').toUpperCase()}</Field>
            <Field label="Loaded" title={model.loaded_at ? formatDate(model.loaded_at) : undefined}>
              {model.loaded_at ? formatRelativeTime(model.loaded_at, now) : <Unavailable>Not recorded</Unavailable>}
            </Field>
            <Field label="Requests served">
              <Unavailable reason="The backend does not count requests for this model.">Not currently measured</Unavailable>
            </Field>
          </>
        )}
      </dl>
      <footer className="sc-rule mt-auto flex items-center justify-end gap-1.5 px-4 py-2.5">
        <button type="button" className="sc-btn sc-btn--ghost" onClick={onDetails}>
          Runtime details
        </button>
        {unloadAction ? (
          <ActionButtons {...unloadAction} only={['unload']} />
        ) : (
          <span className="sc-t3 text-[11.5px]" title="Loaded for the lifetime of the backend process.">
            Stays loaded
          </span>
        )}
      </footer>
    </article>
  );
}

function ModelDetails({ model }: { model: AnyModel }) {
  const rows: Array<[string, React.ReactNode]> = isOllama(model)
    ? [
        ['Model', model.name],
        ['Type', `${model.purpose} model`],
        ['Backend', model.backend],
        ['Total size in memory', formatBytes(model.size_bytes)],
        ['In VRAM', formatBytes(model.vram_bytes)],
        ['In system RAM', formatBytes(model.ram_bytes)],
        ['Share on GPU', model.gpu_pct !== null ? `${model.gpu_pct}%` : 'Not reported'],
        ['Context length', model.context_length ? `${model.context_length.toLocaleString()} tokens` : 'Not reported'],
        ['Parameters', model.parameter_size ?? 'Not reported'],
        ['Quantization', model.quantization ?? 'Not reported'],
        ['Pinned', model.pinned ? 'Yes (keep_alive = -1)' : 'No'],
        ['Released at', model.expires_at ? formatDate(model.expires_at) : model.pinned ? 'Never, while pinned' : 'Not reported'],
        ['Loaded since', 'Not exposed by Ollama'],
        ['Last request', model.last_request_at ? `${formatDate(model.last_request_at)} (telemetry)` : 'Not recorded'],
        ['Requests served', model.requests !== null ? `${model.requests.toLocaleString()} (telemetry)` : 'Not recorded'],
        ['Attention KV cache', "Not exposed by runtime. Included in this model's VRAM figure."],
      ]
    : [
        ['Model', model.name],
        ['Type', `${model.purpose} model`],
        ['Backend', model.backend],
        ['Device', model.device ?? 'cpu'],
        ['Weights', model.weight_bytes !== null ? `${formatBytes(model.weight_bytes)} (parameter tensors)` : 'Not measured'],
        ['Loaded', model.loaded_at ? formatDate(model.loaded_at) : 'Not recorded'],
        ['Requests served', 'Not currently measured'],
        ['Process', 'Shares the backend process: its memory is part of backend RAM, or of the backend VRAM figure on GPU.'],
      ];
  return (
    <dl className="sc-divide">
      {rows.map(([label, value]) => (
        <div key={label} className="grid grid-cols-[150px_1fr] gap-3 py-2">
          <dt className="sc-label pt-0.5">{label}</dt>
          <dd className="sc-num sc-t1 text-[12.5px]">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

interface GpuModelsProps {
  items: StorageMemoryItem[];
  installed: StorageModelEntry[];
  runtime: RuntimeState;
  onLoad: (modelName: string) => void;
}

export function GpuModelsSection({ items, installed, runtime, onLoad }: GpuModelsProps) {
  const { running, narrowed } = useConsole();
  const { gpu, gpuBreakdown, loadedModels, inProcessModels } = runtime;
  const [details, setDetails] = useState<AnyModel | null>(null);
  const [loadTarget, setLoadTarget] = useState('');

  const ollamaItem = items.find((i) => i.id === 'ollama_models');
  const visionItem = items.find((i) => i.id === 'vision_model');
  const models = useVisible<AnyModel>([...loadedModels, ...inProcessModels], loadedModelItem);

  // Chat models that can be loaded: installed in Ollama, not embedding-only, not already resident.
  const loadable = installed
    .filter((m) => m.runtime === 'Ollama' && m.purpose !== 'Embedding' && !m.loaded)
    .map((m) => m.name);
  const target = loadTarget && loadable.includes(loadTarget) ? loadTarget : (loadable[0] ?? '');
  const canLoad = Boolean(ollamaItem?.actions.some((a) => a.id === 'reload'));

  const segments: BarSegment[] = (gpuBreakdown?.segments ?? []).map((s) => ({
    id: s.id,
    label: s.label,
    value: s.bytes,
    color: SEGMENT_COLOR[s.kind] ?? SEGMENT_COLOR.other,
  }));
  const usedFraction = gpu ? gpu.used_mb / gpu.total_mb : 0;

  return (
    <div className="space-y-4">
      {!narrowed &&
        (gpu ? (
          <div className="sc-panel px-4 py-3.5">
            <div className="flex flex-wrap items-baseline justify-between gap-3">
              <div>
                <p className="sc-label">GPU memory</p>
                <p className="sc-t1 mt-1 text-[13.5px] font-semibold">
                  {gpu.name.replace(/^NVIDIA\s+/i, '')}
                  <span className="sc-num sc-t3 ml-2 text-[12px] font-normal">{(gpu.total_mb / 1024).toFixed(1)} GB VRAM</span>
                </p>
              </div>
              <p className="sc-num text-[13px]">
                <span className="sc-t1 text-[17px] font-semibold tracking-[-0.02em]">{(gpu.used_mb / 1024).toFixed(2)} GB</span>
                <span className="sc-t3"> used · </span>
                <span className="sc-t1">{formatBytes(gpu.free_mb * 1024 * 1024)}</span>
                <span className="sc-t3"> free · </span>
                <span style={{ color: usedFraction >= 0.9 ? toneColor(usageTone(usedFraction)) : undefined }} className="sc-t1">
                  {formatPercent(usedFraction)}
                </span>
              </p>
            </div>
            <div className="mt-2.5">
              {segments.length > 0 ? (
                <SegmentBar segments={segments} total={gpu.total_mb * 1024 * 1024} height={14} ariaLabel="GPU memory by holder" />
              ) : (
                <MicroBar value={gpu.used_mb} max={gpu.total_mb} tone={usageTone(usedFraction)} height={14} />
              )}
            </div>
            {segments.length > 0 ? (
              <>
                <BarLegend segments={segments.filter((s) => s.value > 0)} className="mt-2" />
                <p className="sc-t3 mt-1.5 text-[11.5px]">{gpuBreakdown?.note}</p>
              </>
            ) : (
              <p className="sc-t3 mt-1.5 text-[11.5px]">Process-level breakdown unavailable from current runtime.</p>
            )}
          </div>
        ) : (
          <EmptyState
            className="sc-panel"
            icon={<Cpu className="h-4 w-4" />}
            title="GPU unavailable"
            detail="nvidia-smi did not report a device, so VRAM cannot be shown. Models run on the CPU and use system RAM."
          />
        ))}

      {models.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {models.map((model) => {
            const unloadAction = isOllama(model)
              ? ollamaItem
                ? { storeId: 'ollama_models', storeLabel: model.name, actions: ollamaItem.actions, target: model.name }
                : null
              : model.id === 'vision' && visionItem
                ? { storeId: 'vision_model', storeLabel: model.name, actions: visionItem.actions }
                : null;
            return <ModelCard key={model.id + model.name} model={model} unloadAction={unloadAction} onDetails={() => setDetails(model)} />;
          })}
        </div>
      ) : (
        !narrowed && (
          <EmptyState
            className="sc-panel"
            icon={<MemoryStick className="h-4 w-4" />}
            title="No models loaded"
            detail="Nothing is resident in memory, or the Ollama server is not reachable. A model loads on the next question."
          />
        )
      )}

      {!narrowed && canLoad && loadable.length > 0 && (
        <div className="flex flex-wrap items-center justify-end gap-2">
          <label className="sc-t2 text-[12px]" htmlFor="sc-load-model">
            Load a chat model
          </label>
          <select
            id="sc-load-model"
            value={target}
            onChange={(e) => setLoadTarget(e.target.value)}
            className="sc-input sc-num px-2 py-1.5"
          >
            {loadable.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
          <button type="button" className="sc-btn" disabled={running !== null || !target} onClick={() => onLoad(target)}>
            {running === `ollama_models/reload/${target}` ? 'Loading…' : 'Load'}
          </button>
        </div>
      )}

      <Overlay open={details !== null} onClose={() => setDetails(null)} eyebrow="Runtime details" title={details?.name ?? ''}>
        {details && <ModelDetails model={details} />}
      </Overlay>
    </div>
  );
}
