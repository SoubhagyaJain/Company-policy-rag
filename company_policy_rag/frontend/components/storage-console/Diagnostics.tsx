'use client';

/** Advanced view: the storage hierarchy, the data lifecycle from upload to
 *  query-time caches, and where every persistent store sits on disk. */

import React from 'react';
import { ArrowDown } from 'lucide-react';

import { formatBytes, formatDate } from '@/lib/utils';
import type { StorageFlowStep, StorageSummary, StorageTreeNode } from '@/lib/types';
import { useConsole } from './context';
import { KindBadges } from './primitives';

interface Resolved {
  value: string;
  storeId?: string;
}

function useResolver(summary: StorageSummary) {
  const stores = new Map(summary.stores.map((s) => [s.id, s]));
  const items = new Map(summary.memory.items.map((i) => [i.id, i]));

  return (node: { store?: string; runtime?: string; model?: string }): Resolved | null => {
    if (node.store) {
      const store = stores.get(node.store);
      if (!store) return null;
      // The semantic cache has no file of its own: it is rows inside the vector database.
      const value = store.path ? formatBytes(store.size_bytes) : `${(store.items ?? 0).toLocaleString()} ${store.items_label}`;
      return { value, storeId: store.id };
    }
    if (node.runtime) {
      const item = items.get(node.runtime);
      if (!item) return null;
      return { value: `${item.value.toLocaleString()} ${item.unit}`, storeId: item.id };
    }
    if (node.model) {
      const matching = summary.models.filter((m) => m.id.startsWith(node.model as string));
      if (matching.length === 0) return null;
      return { value: formatBytes(matching.reduce((sum, m) => sum + m.size_bytes, 0)) };
    }
    return null;
  };
}

function TreeBranch({ node, resolve, prefix, last }: { node: StorageTreeNode; resolve: ReturnType<typeof useResolver>; prefix: string; last: boolean }) {
  const { explain } = useConsole();
  const resolved = resolve(node);
  const children = (node.children ?? []).filter((child) => child.children || resolve(child));
  return (
    <>
      <li className="sc-row grid grid-cols-[minmax(0,1fr)_auto] items-baseline gap-3 px-4 py-[3px]">
        <span className="flex min-w-0 items-baseline">
          <span className="sc-t3 whitespace-pre" aria-hidden>
            {prefix}
            {last ? '└── ' : '├── '}
          </span>
          {resolved?.storeId ? (
            <button type="button" onClick={() => explain(resolved.storeId as string)} className="sc-focus sc-t1 truncate rounded text-left hover:underline">
              {node.label}
            </button>
          ) : (
            <span className={node.children ? 'sc-t1 font-semibold' : 'sc-t1'}>{node.label}</span>
          )}
          {node.runtime_only && <span className="sc-badge sc-badge--volatile ml-2">Runtime only</span>}
        </span>
        <span className="sc-t2 text-right">{resolved?.value ?? ''}</span>
      </li>
      {children.map((child, index) => (
        <TreeBranch
          key={child.label}
          node={child}
          resolve={resolve}
          prefix={`${prefix}${last ? '    ' : '│   '}`}
          last={index === children.length - 1}
        />
      ))}
    </>
  );
}

function Lifecycle({ steps, resolve }: { steps: StorageFlowStep[]; resolve: ReturnType<typeof useResolver> }) {
  const { explain } = useConsole();
  const labels: Record<string, string> = {
    vision_cache: 'Vision cache',
    retrieval_cache: 'Retrieval cache',
    kv_cache: 'Application KV cache',
  };
  return (
    <ol>
      {steps.map((step, index) => {
        const resolved = resolve(step);
        return (
          <li key={step.label}>
            <div className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-3 px-4 py-2">
              <div className="min-w-0">
                <p className="sc-t1 text-[12.5px] font-semibold">{step.label}</p>
                <p className="sc-t2 mt-0.5 text-[12px] leading-relaxed">{step.detail}</p>
                {step.also && (
                  <p className="mt-1 flex flex-wrap items-center gap-1.5 text-[11.5px]">
                    <span className="sc-t3">also</span>
                    {step.also.map((id) => (
                      <button key={id} type="button" onClick={() => explain(id)} className="sc-focus sc-t2 rounded hover:underline">
                        {labels[id] ?? id}
                      </button>
                    ))}
                  </p>
                )}
              </div>
              {resolved && (
                <button
                  type="button"
                  onClick={() => resolved.storeId && explain(resolved.storeId)}
                  className="sc-focus sc-num sc-t2 shrink-0 rounded text-right text-[12px] hover:underline"
                  title="Where this artifact lives"
                >
                  {resolved.value}
                </button>
              )}
            </div>
            {index < steps.length - 1 && <ArrowDown className="sc-t3 ml-4 h-3 w-3" aria-hidden />}
          </li>
        );
      })}
    </ol>
  );
}

export function Diagnostics({ summary }: { summary: StorageSummary }) {
  const resolve = useResolver(summary);
  const onDisk = summary.stores.filter((s) => s.path);

  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="sc-panel pb-2">
          <p className="sc-label px-4 pb-1 pt-3">Storage hierarchy</p>
          <ul className="sc-num text-[12px]">
            <li className="sc-t1 px-4 py-[3px] font-semibold">Storage</li>
            {summary.hierarchy.map((node, index) => (
              <TreeBranch key={node.label} node={node} resolve={resolve} prefix="" last={index === summary.hierarchy.length - 1} />
            ))}
          </ul>
        </div>
        <div className="sc-panel pb-2">
          <p className="sc-label px-4 pb-1 pt-3">Data lifecycle</p>
          <Lifecycle steps={summary.lifecycle} resolve={resolve} />
        </div>
      </div>

      <div className="sc-panel overflow-x-auto">
        <p className="sc-label px-4 pb-1 pt-3">File system details</p>
        <table className="w-full border-collapse text-[12px]">
          <thead>
            <tr>
              <th className="sc-th px-4 py-2">Logical name</th>
              <th className="sc-th px-2 py-2">Physical path</th>
              <th className="sc-th px-2 py-2">Engine</th>
              <th className="sc-th px-2 py-2">Persistence</th>
              <th className="sc-th px-2 py-2 text-right">Files</th>
              <th className="sc-th px-2 py-2 text-right">Size</th>
              <th className="sc-th px-4 py-2 text-right">Last modified</th>
            </tr>
          </thead>
          <tbody>
            {onDisk.map((store) => (
              <tr key={store.id} className="sc-row border-t border-[color:var(--sp-hairline)]">
                <td className="sc-num sc-t1 whitespace-nowrap px-4 py-1.5">{store.id}</td>
                <td className="sc-num sc-t2 max-w-[360px] truncate px-2 py-1.5" title={store.path ?? undefined}>
                  {store.path}
                </td>
                <td className="sc-t2 whitespace-nowrap px-2 py-1.5">{store.details.engine ?? 'Files'}</td>
                <td className="px-2 py-1.5">
                  <KindBadges kinds={store.kinds} limit={2} />
                </td>
                <td className="sc-num sc-t2 px-2 py-1.5 text-right">{store.file_count !== null ? store.file_count.toLocaleString() : '—'}</td>
                <td className="sc-num sc-t1 px-2 py-1.5 text-right">{formatBytes(store.size_bytes)}</td>
                <td className="sc-num sc-t2 whitespace-nowrap px-4 py-1.5 text-right" title={store.last_modified ?? undefined}>
                  {store.last_modified ? formatDate(store.last_modified) : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="sc-t3 sc-rule px-4 py-2 text-[11.5px]">
          Sizes are cached for up to a minute; Refresh rescans every directory. Summary generated {formatDate(summary.generated_at)}.
        </p>
      </div>
    </div>
  );
}
