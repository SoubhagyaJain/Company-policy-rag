'use client';

/** One drawer per store: why it exists and what clearing it costs, where it
 *  sits on disk, and its internals (collections, tables, largest files). */

import React, { useEffect, useState } from 'react';
import { AlertTriangle } from 'lucide-react';

import { apiClient } from '@/lib/api-client';
import { formatBytes, formatDate } from '@/lib/utils';
import type { StorageInfo, StorageInspect, StorageSqliteStats } from '@/lib/types';
import { Overlay } from './Overlay';
import { EmptyState, Estimated, KindBadges, Unavailable } from './primitives';

export interface StoreDrawerTarget {
  storeId: string;
  /** Lead with the internals (Inspect) or with the explanation (the info button). */
  focus: 'why' | 'internals';
}

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mb-5 last:mb-0">
      <h3 className="sc-label mb-1.5">{title}</h3>
      {children}
    </section>
  );
}

function Pairs({ rows }: { rows: Array<[string, React.ReactNode]> }) {
  return (
    <dl className="sc-divide">
      {rows.map(([label, value]) => (
        <div key={label} className="grid grid-cols-[136px_1fr] gap-3 py-1.5">
          <dt className="sc-t3 text-[12px]">{label}</dt>
          <dd className="sc-t1 min-w-0 break-words text-[12.5px] leading-relaxed">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function Why({ info }: { info: StorageInfo }) {
  return (
    <Pairs
      rows={[
        ['What is this?', info.what],
        ['Why is it needed?', info.why],
        ['What creates it?', info.created_by],
        ['What reads it?', info.read_by],
        ['Survives a restart?', info.survives_restart ? 'Yes. It is stored on disk.' : 'No. It lives in memory only.'],
        ['Can it be deleted?', info.deletable],
        ['If it is deleted', info.delete_effect],
        ['How it rebuilds', info.rebuild],
        ['Cleanup policy', info.cleanup_policy],
      ]}
    />
  );
}

const num = (value: number | null | undefined) => (typeof value === 'number' ? value.toLocaleString() : '—');
const size = (value: number | null | undefined) => (typeof value === 'number' ? formatBytes(value) : '—');
const when = (value: string | null | undefined) => (value ? formatDate(value) : <Unavailable>Not recorded</Unavailable>);

function Sqlite({ stats }: { stats: StorageSqliteStats }) {
  return (
    <Pairs
      rows={[
        ['File size', size(stats.file_bytes)],
        ['WAL file', size(stats.wal_bytes)],
        ['Pages', `${num(stats.page_count)} × ${size(stats.page_size)}`],
        ['Unused pages', `${num(stats.free_pages)} (${size(stats.free_bytes)})`],
        ['Fragmentation', typeof stats.bloat_pct === 'number' ? `${stats.bloat_pct}%` : '—'],
      ]}
    />
  );
}

function Internals({ data }: { data: StorageInspect }) {
  return (
    <>
      {data.error && (
        <p className="sp-warn mb-4 flex items-start gap-2 rounded-md px-3 py-2 text-[12px]">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
          {data.error}
        </p>
      )}

      {data.collections && (
        <Block title="Collections">
          <table className="w-full border-collapse text-[12px]">
            <thead>
              <tr>
                <th className="sc-th py-1.5">Collection</th>
                <th className="sc-th px-2 py-1.5 text-right">Rows</th>
                <th className="sc-th px-2 py-1.5 text-right">Dim</th>
                <th className="sc-th py-1.5 text-right">Vectors, est.</th>
              </tr>
            </thead>
            <tbody>
              {data.collections.map((collection) => (
                <tr key={collection.name} className="border-t border-[color:var(--sp-hairline)]">
                  <td className="py-1.5">
                    <span className="sc-num sc-t1">{collection.name}</span>
                    <span className="sc-t3 block text-[11px]">{collection.role}</span>
                  </td>
                  <td className="sc-num sc-t1 px-2 py-1.5 text-right">{collection.rows.toLocaleString()}</td>
                  <td className="sc-num sc-t2 px-2 py-1.5 text-right">{collection.vector_dim ?? '—'}</td>
                  <td className="sc-num sc-t1 py-1.5 text-right">
                    {collection.embedding_bytes_estimated !== null ? (
                      <Estimated>{formatBytes(collection.embedding_bytes_estimated)}</Estimated>
                    ) : (
                      <Unavailable>Unavailable</Unavailable>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="sc-t3 mt-1.5 text-[11.5px]">Vector bytes are rows × dimension × 4. Chroma does not report bytes per collection.</p>
        </Block>
      )}

      {data.documents && (
        <Block title="Documents in the index">
          <Pairs
            rows={[
              ['Documents', num(data.documents.count)],
              ['Chunks', num(data.documents.chunks)],
              ['Average chunk size', data.documents.avg_chunk_chars !== null ? `${num(data.documents.avg_chunk_chars)} characters` : '—'],
              ['Oldest record', data.documents.oldest ? `${data.documents.oldest.filename} · ${formatDate(data.documents.oldest.created_at)}` : '—'],
              ['Newest record', data.documents.newest ? `${data.documents.newest.filename} · ${formatDate(data.documents.newest.created_at)}` : '—'],
            ]}
          />
          {data.documents.largest.length > 0 && (
            <ul className="mt-2">
              {data.documents.largest.map((doc) => (
                <li key={doc.document_id} className="flex items-center justify-between gap-3 py-0.5 text-[12px]">
                  <span className="sc-t2 truncate">{doc.filename}</span>
                  <span className="sc-num sc-t1 shrink-0">{doc.chunks.toLocaleString()} chunks</span>
                </li>
              ))}
            </ul>
          )}
        </Block>
      )}

      {data.tables && (
        <Block title="Tables">
          <table className="w-full border-collapse text-[12px]">
            <thead>
              <tr>
                <th className="sc-th py-1.5">Table</th>
                <th className="sc-th px-2 py-1.5 text-right">Rows</th>
                <th className="sc-th px-2 py-1.5 text-right">Oldest</th>
                <th className="sc-th py-1.5 text-right">Newest</th>
              </tr>
            </thead>
            <tbody>
              {data.tables.map((table) => (
                <tr key={table.name} className="border-t border-[color:var(--sp-hairline)]">
                  <td className="sc-num sc-t1 py-1.5">{table.name}</td>
                  <td className="sc-num sc-t1 px-2 py-1.5 text-right">{table.rows.toLocaleString()}</td>
                  <td className="sc-num sc-t2 px-2 py-1.5 text-right">{table.oldest ? formatDate(table.oldest) : '—'}</td>
                  <td className="sc-num sc-t2 py-1.5 text-right">{table.newest ? formatDate(table.newest) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Block>
      )}

      {data.sqlite && (
        <Block title="SQLite file">
          <Sqlite stats={data.sqlite} />
          {data.segments && typeof data.segments.hnsw_bytes === 'number' && (
            <p className="sc-t2 mt-1.5 text-[12px]">
              HNSW vector segments next to the file: <span className="sc-num sc-t1">{formatBytes(data.segments.hnsw_bytes)}</span>
            </p>
          )}
        </Block>
      )}

      {data.semantic && (
        <Block title="Cached answers">
          {data.semantic.entries === null ? (
            <Unavailable>{data.semantic.note ?? 'Unavailable'}</Unavailable>
          ) : (
            <Pairs
              rows={[
                ['Entries', num(data.semantic.entries)],
                ['Oldest entry', when(data.semantic.oldest_entry)],
                ['Newest entry', when(data.semantic.newest_entry)],
                [
                  'For the current library',
                  typeof data.semantic.for_current_library === 'number' ? (
                    <>
                      {num(data.semantic.for_current_library)}
                      <span className="sc-t3 block text-[11.5px]">
                        Entries written for another library version are never served again.
                      </span>
                    </>
                  ) : (
                    <Unavailable>Not available</Unavailable>
                  ),
                ],
                ['By model', (data.semantic.by_model ?? []).map((m) => `${m.model}: ${m.entries.toLocaleString()}`).join(' · ') || '—'],
              ]}
            />
          )}
        </Block>
      )}

      {data.largest && data.largest.length > 0 && (
        <Block title="Largest entries">
          <ul>
            {data.largest.map((entry) => (
              <li key={entry.name} className="flex items-center justify-between gap-3 py-0.5 text-[12px]">
                <span className="sc-num sc-t2 truncate">{entry.name}</span>
                <span className="sc-num sc-t1 shrink-0">{formatBytes(entry.size_bytes)}</span>
              </li>
            ))}
          </ul>
        </Block>
      )}
    </>
  );
}

function Filesystem({ data }: { data: StorageInspect }) {
  const fs = data.filesystem;
  if (!fs) {
    return <p className="sc-t2 text-[12.5px]">Held in memory. There is nothing on disk for this store.</p>;
  }
  return (
    <Pairs
      rows={[
        ['Logical name', <span key="n" className="sc-num">{fs.logical_name}</span>],
        ['Physical path', fs.path ? <span className="sc-num text-[12px]">{fs.path}</span> : 'Inside the vector database'],
        ['Engine', fs.engine ?? '—'],
        ['Persistence', <KindBadges key="k" kinds={fs.kinds} />],
        ['Current size', fs.path ? formatBytes(fs.size_bytes) : <Unavailable>Not measured separately</Unavailable>],
        ['File count', num(fs.file_count)],
        ['Created', when(fs.created)],
        ['Last modified', when(fs.last_modified)],
        ['Last accessed', when(fs.last_accessed)],
        ['Rebuildable?', fs.rebuildable ? 'Yes' : 'No'],
      ]}
    />
  );
}

export function StoreDrawer({ target, onClose }: { target: StoreDrawerTarget | null; onClose: () => void }) {
  const [data, setData] = useState<StorageInspect | null>(null);
  const [error, setError] = useState<string | null>(null);
  const storeId = target?.storeId;

  useEffect(() => {
    if (!storeId) return;
    let cancelled = false;
    setData(null);
    setError(null);
    apiClient
      .inspectStorage(storeId)
      .then((next) => {
        if (!cancelled) setData(next);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Inspection failed');
      });
    return () => {
      cancelled = true;
    };
  }, [storeId]);

  const why = data?.info ? (
    <Block title="Why is this here?">
      <Why info={data.info} />
    </Block>
  ) : null;
  const details = data ? (
    <Block title="Storage details">
      <Filesystem data={data} />
    </Block>
  ) : null;
  const internals = data ? <Internals data={data} /> : null;

  return (
    <Overlay
      open={target !== null}
      onClose={onClose}
      eyebrow={target?.focus === 'internals' ? 'Inspect' : 'Store'}
      title={data?.label ?? (error ? 'Unavailable' : 'Loading…')}
    >
      {error ? (
        <EmptyState icon={<AlertTriangle className="h-4 w-4" />} title="Database inspection failed" detail={error} />
      ) : !data ? (
        <p className="sc-t2 text-[12.5px]">Reading the store…</p>
      ) : target?.focus === 'internals' ? (
        <>
          {internals}
          {details}
          {why}
        </>
      ) : (
        <>
          {why}
          {details}
          {internals}
        </>
      )}
    </Overlay>
  );
}
