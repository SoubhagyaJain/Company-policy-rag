'use client';

/** Which document is responsible for which bytes, and which artifacts were
 *  left behind by documents that no longer exist. Documents are never deleted
 *  from here; that stays in the Library. */

import React, { useMemo, useState } from 'react';
import { ArrowUpRight, Check, ChevronRight, FileText, ScanSearch } from 'lucide-react';

import { cn, formatBytes, formatDate, formatRelativeTime } from '@/lib/utils';
import type { StorageDocumentRow, StorageDocuments, StorageOrphanGroup, StorageStore } from '@/lib/types';
import { useConsole, useRowClass } from './context';
import { EmptyState, Estimated, KindBadges, MicroBar, SafetyTag, Unavailable } from './primitives';

type SortKey = 'largest' | 'newest' | 'queried' | 'chunks' | 'images';

const SORTS: Array<{ id: SortKey; label: string }> = [
  { id: 'largest', label: 'Largest' },
  { id: 'newest', label: 'Newest' },
  { id: 'queried', label: 'Most queried' },
  { id: 'chunks', label: 'Most chunks' },
  { id: 'images', label: 'Most image-heavy' },
];

const SORTERS: Record<SortKey, (a: StorageDocumentRow, b: StorageDocumentRow) => number> = {
  largest: (a, b) => b.total_bytes_estimated - a.total_bytes_estimated,
  newest: (a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''),
  queried: (a, b) => b.query_count - a.query_count,
  chunks: (a, b) => b.chunks - a.chunks,
  images: (a, b) => b.images.bytes - a.images.bytes,
};

const PAGE = 12;

const STATE_NOTE: Record<string, string> = {
  INDEX_ONLY: 'Indexed, but the original file is missing',
  FILE_ONLY: 'File stored, but not indexed',
};

/** Everything that exists because this document was uploaded. */
function DependencyTree({ doc, dim }: { doc: StorageDocumentRow; dim: number | null }) {
  const branches: Array<{ label: string; where: string; size?: React.ReactNode }> = [
    { label: `${doc.chunks.toLocaleString()} chunks`, where: 'Chunk store, in memory' },
    {
      label: `${doc.embeddings.toLocaleString()} embeddings`,
      where: dim ? `Chroma, ${dim}-dimensional vectors` : 'Chroma',
      size: doc.index_bytes_estimated !== null ? <Estimated>{formatBytes(doc.index_bytes_estimated)}</Estimated> : undefined,
    },
    {
      label: `${doc.images.count.toLocaleString()} extracted images`,
      where: 'Extracted page images',
      size: formatBytes(doc.images.bytes),
    },
    {
      label: `${doc.vision_cache.entries.toLocaleString()} vision cache entries`,
      where: 'Vision cache',
      size: formatBytes(doc.vision_cache.bytes),
    },
    { label: `${doc.chunks.toLocaleString()} BM25 references`, where: 'Keyword index' },
    { label: `${doc.chunks.toLocaleString()} Chroma records`, where: 'Vector database' },
  ];
  return (
    <div className="sc-num text-[12px]">
      <p className="sc-t1 flex items-center gap-2">
        <FileText className="sc-t3 h-3.5 w-3.5" aria-hidden />
        {doc.filename}
        <span className="sc-t3">{doc.file_bytes > 0 ? formatBytes(doc.file_bytes) : 'original missing'}</span>
      </p>
      <ul className="mt-1">
        {branches.map((branch, index) => (
          <li key={branch.label} className="grid grid-cols-[18px_minmax(0,220px)_minmax(0,1fr)_auto] items-baseline gap-x-2 py-0.5">
            <span className="sc-t3" aria-hidden>
              {index === branches.length - 1 ? '└─' : '├─'}
            </span>
            <span className="sc-t1 truncate">{branch.label}</span>
            <span className="sc-t3 truncate font-sans text-[11.5px]">{branch.where}</span>
            <span className="sc-t2 text-right">{branch.size ?? ''}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function OrphanRow({
  title,
  group,
  unit,
}: {
  title: string;
  group: StorageOrphanGroup;
  unit: (group: StorageOrphanGroup) => string;
}) {
  const [open, setOpen] = useState(false);
  if (group.count === 0) return null;
  return (
    <li>
      <div className="flex items-center gap-3 px-4 py-2">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          className="sc-focus flex min-w-0 flex-1 items-center gap-1.5 rounded text-left"
        >
          <ChevronRight className={cn('sc-t3 h-3.5 w-3.5 shrink-0 transition-transform', open && 'rotate-90')} aria-hidden />
          <span className="sc-t1 text-[12.5px] font-medium">{title}</span>
          <span className="sc-t3 text-[11.5px]">{unit(group)}</span>
        </button>
        <SafetyTag safety={group.safety} />
        <span className="sc-num sc-t1 w-20 text-right text-[12.5px]">{formatBytes(group.bytes)}</span>
      </div>
      {open && (
        <ul className="pb-2 pl-10 pr-4">
          {group.items.map((item) => (
            <li key={item.name} className="sc-num sc-t2 flex items-center justify-between gap-3 py-0.5 text-[11.5px]">
              <span className="truncate">{item.name}</span>
              <span className="shrink-0">
                {item.files !== undefined ? `${item.files.toLocaleString()} files · ` : ''}
                {item.entries !== undefined ? `${item.entries.toLocaleString()} entries · ` : ''}
                {formatBytes(item.bytes)}
              </span>
            </li>
          ))}
          {group.count > group.items.length && (
            <li className="sc-t3 py-0.5 text-[11.5px]">and {(group.count - group.items.length).toLocaleString()} more</li>
          )}
        </ul>
      )}
    </li>
  );
}

interface DocumentsSectionProps {
  documents: StorageDocuments | null;
  stores: StorageStore[];
  onOpenLibrary?: () => void;
  onRemoveOrphans: () => void;
  onScanVectors: () => void;
}

export function DocumentsSection({ documents, stores, onOpenLibrary, onRemoveOrphans, onScanVectors }: DocumentsSectionProps) {
  const { match, narrowed, running, now, ask, explain } = useConsole();
  const rowClass = useRowClass();
  const [sort, setSort] = useState<SortKey>('largest');
  const [expanded, setExpanded] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);

  const uploads = stores.find((s) => s.id === 'uploads');
  const images = stores.find((s) => s.id === 'page_images');
  const sessions = stores.find((s) => s.id === 'session_libraries');

  const rows = useMemo(() => {
    const all = documents?.documents ?? [];
    return all
      .filter((doc) =>
        match({
          text: `${doc.filename} ${doc.document_id} document upload`,
          kinds: ['USER_DATA', 'PERSISTENT'],
          sizeBytes: doc.total_bytes_estimated,
        }),
      )
      .sort(SORTERS[sort]);
  }, [documents, match, sort]);

  if (!documents) {
    return <EmptyState className="sc-panel" title="Document storage is unavailable" detail="The backend did not return the per-document breakdown." />;
  }

  const { orphans } = documents;
  const largest = Math.max(1, ...rows.map((r) => r.total_bytes_estimated));
  const shown = showAll ? rows : rows.slice(0, PAGE);
  const refs = orphans.chunk_references;
  const orphanTotalItems = orphans.images.count + orphans.vision_cache.count;
  const showOrphans = !narrowed || match({ text: 'orphaned data artifacts images vision cache', orphaned: orphanTotalItems, reclaimableBytes: orphans.total_bytes });

  return (
    <div className="space-y-4">
      {/* Where the originals and their images live */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-[12px]">
        {[uploads, images].map(
          (store) =>
            store && (
              <button
                key={store.id}
                type="button"
                onClick={() => explain(store.id)}
                className="sc-focus flex items-center gap-2 rounded text-left"
                title="Why is this here?"
              >
                <span className="sc-t2">{store.label}</span>
                <span className="sc-num sc-t1 font-semibold">{formatBytes(store.size_bytes)}</span>
                <KindBadges kinds={store.kinds} limit={2} />
              </button>
            ),
        )}
      </div>

      {(documents.documents.length > 0 || !narrowed) && (
        <div>
          <div className="mb-2 flex flex-wrap items-center gap-1.5">
            <span className="sc-label mr-1">Storage by document</span>
            {SORTS.map((option) => (
              <button
                key={option.id}
                type="button"
                aria-pressed={sort === option.id}
                onClick={() => setSort(option.id)}
                className="sc-chip"
              >
                {option.label}
              </button>
            ))}
          </div>

          {rows.length === 0 ? (
            <EmptyState
              className="sc-panel"
              icon={<FileText className="h-4 w-4" />}
              title={documents.documents.length === 0 ? 'No documents in the library' : 'No document matches'}
              detail={documents.documents.length === 0 ? 'Upload a document in the Library tab to see its footprint here.' : undefined}
            />
          ) : (
            <div className="sc-panel overflow-hidden">
              <table className="w-full table-fixed border-collapse text-[12.5px]">
                <colgroup>
                  <col />
                  <col className="w-[88px]" />
                  <col className="w-[72px]" />
                  <col className="w-[104px]" />
                  <col className="w-[104px]" />
                  <col className="w-[92px]" />
                  <col className="w-[140px]" />
                  <col className="w-[124px]" />
                </colgroup>
                <thead>
                  <tr>
                    <th className="sc-th px-4 py-2">Document</th>
                    <th className="sc-th px-2 py-2 text-right">Original</th>
                    <th className="sc-th px-2 py-2 text-right">Chunks</th>
                    <th className="sc-th px-2 py-2 text-right">Images</th>
                    <th className="sc-th px-2 py-2 text-right">Vision cache</th>
                    <th className="sc-th px-2 py-2 text-right" title={documents.notes.index_bytes}>
                      Index, est.
                    </th>
                    <th className="sc-th px-2 py-2 text-right">Total, est.</th>
                    <th className="sc-th px-4 py-2 text-right" title={documents.notes.last_queried}>
                      Last queried
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {shown.map((doc) => {
                    const open = expanded === doc.document_id;
                    return (
                      <React.Fragment key={doc.document_id}>
                        <tr className={cn(rowClass(open ? 'is-open' : undefined), 'border-t border-[color:var(--sp-hairline)]')}>
                          <td className="px-4 py-2">
                            <button
                              type="button"
                              onClick={() => setExpanded(open ? null : doc.document_id)}
                              aria-expanded={open}
                              className="sc-focus flex w-full min-w-0 items-center gap-1.5 rounded text-left"
                            >
                              <ChevronRight className={cn('sc-t3 h-3.5 w-3.5 shrink-0 transition-transform', open && 'rotate-90')} aria-hidden />
                              <span className="sc-t1 truncate font-medium" title={doc.filename}>
                                {doc.filename}
                              </span>
                              {STATE_NOTE[doc.storage_state] && (
                                <span className="sc-badge shrink-0" title={STATE_NOTE[doc.storage_state]}>
                                  {doc.storage_state === 'INDEX_ONLY' ? 'Index only' : 'File only'}
                                </span>
                              )}
                            </button>
                          </td>
                          <td className="sc-num sc-t1 px-2 py-2 text-right">
                            {doc.file_bytes > 0 ? formatBytes(doc.file_bytes) : <Unavailable reason="The original file is not stored.">—</Unavailable>}
                          </td>
                          <td className="sc-num sc-t1 px-2 py-2 text-right">{doc.chunks.toLocaleString()}</td>
                          <td className="sc-num px-2 py-2 text-right">
                            <span className="sc-t1">{formatBytes(doc.images.bytes)}</span>
                            <span className="sc-t3 block text-[11px]">{doc.images.count.toLocaleString()} files</span>
                          </td>
                          <td className="sc-num px-2 py-2 text-right">
                            <span className="sc-t1">{formatBytes(doc.vision_cache.bytes)}</span>
                            <span className="sc-t3 block text-[11px]">{doc.vision_cache.entries.toLocaleString()} pages</span>
                          </td>
                          <td className="sc-num sc-t1 px-2 py-2 text-right">
                            {doc.index_bytes_estimated !== null ? (
                              <Estimated>{formatBytes(doc.index_bytes_estimated)}</Estimated>
                            ) : (
                              <Unavailable reason={documents.notes.index_bytes}>Unavailable</Unavailable>
                            )}
                          </td>
                          <td className="px-2 py-2">
                            <div className="flex items-center justify-end gap-2">
                              <MicroBar value={doc.total_bytes_estimated} max={largest} tone="neutral" className="w-12" />
                              <span className="sc-num sc-t1 w-[72px] text-right font-semibold">{formatBytes(doc.total_bytes_estimated)}</span>
                            </div>
                          </td>
                          <td className="px-4 py-2 text-right">
                            {doc.last_queried ? (
                              <span className="sc-num sc-t2" title={`${formatDate(doc.last_queried)} · ${doc.query_count.toLocaleString()} scoped queries`}>
                                {formatRelativeTime(doc.last_queried, now)}
                              </span>
                            ) : (
                              <Unavailable reason={documents.notes.last_queried}>Not recorded</Unavailable>
                            )}
                          </td>
                        </tr>
                        {open && (
                          <tr className="sc-row is-open">
                            <td colSpan={8} className="px-4 pb-3 pl-10 pt-1">
                              <DependencyTree doc={doc} dim={documents.vector_dim} />
                              <div className="mt-2 flex flex-wrap items-center gap-3">
                                {onOpenLibrary && (
                                  <button type="button" className="sc-btn" onClick={onOpenLibrary}>
                                    View in Library
                                    <ArrowUpRight className="h-3 w-3" aria-hidden />
                                  </button>
                                )}
                                <span className="sc-t3 text-[11.5px]">
                                  Added {doc.created_at ? formatDate(doc.created_at) : '—'} · {doc.pages.toLocaleString()} pages ·
                                  deleting it in the Library removes everything above.
                                </span>
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    );
                  })}
                </tbody>
              </table>
              {rows.length > PAGE && (
                <div className="sc-rule px-4 py-2">
                  <button type="button" className="sc-btn sc-btn--ghost" onClick={() => setShowAll((v) => !v)}>
                    {showAll ? 'Show fewer' : `Show all ${rows.length.toLocaleString()} documents`}
                  </button>
                </div>
              )}
            </div>
          )}
          <p className="sc-t3 mt-1.5 text-[11.5px]">
            Totals add the original file, extracted images, vision cache and estimated vectors. {documents.notes.index_bytes}
          </p>
        </div>
      )}

      {/* Orphaned data */}
      {showOrphans && (
        <div className="sc-panel">
          <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
            <div>
              <p className="sc-label">Orphaned data</p>
              {orphans.total_bytes > 0 ? (
                <p className="mt-1 flex flex-wrap items-baseline gap-x-2">
                  <span className="sc-num sc-t1 text-[17px] font-semibold tracking-[-0.02em]">{formatBytes(orphans.total_bytes)}</span>
                  <span className="sc-t2 text-[12px]">
                    {(orphans.images.files ?? 0).toLocaleString()} files · {orphans.vision_cache.count.toLocaleString()} cache
                    entries. {orphans.reason}
                  </span>
                </p>
              ) : (
                <p className="sc-t2 mt-1 flex items-center gap-1.5 text-[12.5px]">
                  <Check className="sc-ok h-3.5 w-3.5" aria-hidden />
                  No orphaned files. Every image folder and cached page belongs to a document in the library.
                </p>
              )}
            </div>
            {orphans.total_bytes > 0 && (
              <button type="button" className="sc-btn sc-btn--primary" disabled={running !== null} onClick={onRemoveOrphans}>
                {running === 'cleanup' ? 'Removing…' : 'Remove orphaned data'}
              </button>
            )}
          </div>

          <ul className="sc-divide sc-rule">
            <OrphanRow title="Extracted images" group={orphans.images} unit={(g) => `${g.count.toLocaleString()} documents`} />
            <OrphanRow title="Vision cache entries" group={orphans.vision_cache} unit={(g) => `${g.count.toLocaleString()} pages`} />
            {orphans.sessions.count > 0 && sessions && (
              <li className="flex items-center gap-3 px-4 py-2">
                <span className="sc-t1 min-w-0 flex-1 text-[12.5px] font-medium">
                  Old session directories{' '}
                  <span className="sc-t3 font-normal">
                    {orphans.sessions.count.toLocaleString()} sessions · contain the files uploaded in them
                  </span>
                </span>
                <SafetyTag safety="DESTRUCTIVE" />
                <span className="sc-num sc-t1 w-20 text-right text-[12.5px]">{formatBytes(orphans.sessions.bytes)}</span>
                {sessions.actions[0] && (
                  <button
                    type="button"
                    className="sc-btn sc-btn--danger"
                    disabled={running !== null}
                    onClick={() => ask({ storeId: sessions.id, storeLabel: sessions.label, action: sessions.actions[0] })}
                  >
                    Remove old
                  </button>
                )}
              </li>
            )}
            <li className="flex flex-wrap items-center gap-x-3 gap-y-1 px-4 py-2 text-[12.5px]">
              <span className="sc-t1 min-w-0 flex-1 font-medium">
                Chunk references{' '}
                <span className="sc-t3 font-normal">
                  report only · BM25 {refs.bm25.toLocaleString()} · chunk store {refs.docstore.toLocaleString()} · vector rows{' '}
                  {refs.vector === null ? 'not scanned' : refs.vector.toLocaleString()}
                  {refs.vector_checked_at ? ` (${formatRelativeTime(refs.vector_checked_at, now)})` : ''}
                </span>
              </span>
              <button
                type="button"
                className="sc-btn"
                disabled={running !== null}
                onClick={onScanVectors}
                title="Reads the metadata of every vector row. Nothing is deleted."
              >
                <ScanSearch className="h-3 w-3" aria-hidden />
                {running === 'deep-scan' ? 'Scanning…' : 'Scan vector rows'}
              </button>
            </li>
          </ul>
        </div>
      )}
    </div>
  );
}
