'use client';

/** Space-styled conversation sidebar. Pure presentation over the useSessions
 *  handlers already wired in app/page.tsx. Collapsible glass rail. */

import { useMemo, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Plus, PanelLeftClose, PanelLeft, Trash2, Pencil, Check, X } from 'lucide-react';
import type { ChatSession } from '../../lib/types';
import { useBlackHole } from './BlackHoleAbsorption';

interface SpaceSidebarProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  sessions: ChatSession[];
  activeSessionId: string;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string) => void;
  onRenameSession: (id: string, title: string) => void;
  /** status card */
  modelLabel?: string;
  reranker?: string;
  indexLabel?: string;
  grounded?: boolean;
}

function bucketOf(iso: string): 'Today' | 'Yesterday' | 'Earlier' {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return 'Earlier';
  const now = new Date();
  const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const t = d.getTime();
  if (t >= startToday) return 'Today';
  if (t >= startToday - 86_400_000) return 'Yesterday';
  return 'Earlier';
}

export function SpaceSidebar({
  collapsed,
  onToggleCollapse,
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  onRenameSession,
  modelLabel = 'Qwen 2.5 7B',
  reranker = 'bge-v2-m3',
  indexLabel = '2.4M chunks',
  grounded = true,
}: SpaceSidebarProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [departing, setDeparting] = useState<Set<string>>(new Set());
  const blackHole = useBlackHole();
  const listRef = useRef<HTMLDivElement>(null);

  const markDeparting = (ids: string[], on: boolean) =>
    setDeparting((prev) => {
      const next = new Set(prev);
      ids.forEach((id) => (on ? next.add(id) : next.delete(id)));
      return next;
    });

  /** Delete one chat: spiral its card into the black hole, then run the real
   *  delete (backend + state + undo/error logic all untouched). */
  const handleDelete = (id: string, e: React.MouseEvent) => {
    const row = (e.currentTarget as HTMLElement).closest('[data-session-row]') as HTMLElement | null;
    if (!blackHole || !row) {
      onDeleteSession(id);
      return;
    }
    markDeparting([id], true);
    blackHole.absorb(row, {
      onComplete: () => {
        onDeleteSession(id);
        markDeparting([id], false);
      },
    });
  };

  /** Clear all: stagger every card into the black hole one after another. */
  const handleClearAll = () => {
    if (sessions.length === 0) return;
    const ordered = [...sessions].sort((a, b) => (b.updatedAt || '').localeCompare(a.updatedAt || ''));
    if (!blackHole || !listRef.current) {
      ordered.forEach((s) => onDeleteSession(s.id));
      return;
    }
    const rows = Array.from(listRef.current.querySelectorAll<HTMLElement>('[data-session-row]'));
    const rowById = new Map(rows.map((r) => [r.dataset.sessionRow!, r]));
    markDeparting(ordered.map((s) => s.id), true);
    ordered.forEach((s, i) => {
      const row = rowById.get(s.id);
      if (!row) {
        onDeleteSession(s.id);
        return;
      }
      blackHole.absorb(row, {
        delay: i * 65,
        onComplete: () => {
          onDeleteSession(s.id);
          markDeparting([s.id], false);
        },
      });
    });
  };

  const groups = useMemo(() => {
    const order: Array<'Today' | 'Yesterday' | 'Earlier'> = ['Today', 'Yesterday', 'Earlier'];
    const map: Record<string, ChatSession[]> = { Today: [], Yesterday: [], Earlier: [] };
    [...sessions]
      .sort((a, b) => (b.updatedAt || '').localeCompare(a.updatedAt || ''))
      .forEach((s) => map[bucketOf(s.updatedAt || s.createdAt)].push(s));
    return order.filter((k) => map[k].length > 0).map((k) => ({ label: k, items: map[k] }));
  }, [sessions]);

  const startEdit = (s: ChatSession) => {
    setEditingId(s.id);
    setEditTitle(s.title);
  };
  const commitEdit = (id: string) => {
    if (editTitle.trim()) onRenameSession(id, editTitle.trim());
    setEditingId(null);
  };

  return (
    <aside
      className="sp-side sp-text flex h-full flex-col overflow-hidden rounded-[26px]"
      style={{ width: collapsed ? 74 : 286 }}
    >
      {/* Brand + collapse */}
      <div className="flex items-center justify-between gap-2.5 px-4 pb-3 pt-4">
        {!collapsed && (
          <span className="sp-mono flex min-w-0 items-center gap-2.5 text-[10.5px] uppercase tracking-[0.26em]">
            <span
              className="h-[7px] w-[7px] flex-none rounded-full"
              style={{ background: 'var(--sp-accent)', boxShadow: '0 0 12px rgba(127,227,176,0.85)' }}
            />
            <span className="truncate">Aperture RAG</span>
          </span>
        )}
        <button
          type="button"
          onClick={onToggleCollapse}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className="sp-ibtn flex h-[34px] w-[34px] flex-none items-center justify-center rounded-[11px]"
        >
          {collapsed ? <PanelLeft className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
        </button>
      </div>

      {/* New chat */}
      <div className="px-4 pb-3.5">
        <button
          type="button"
          onClick={onNewSession}
          className="sp-send flex min-h-[44px] w-full items-center gap-2.5 rounded-[14px] px-3.5 py-3 text-left text-[13.5px] font-semibold"
        >
          <Plus className="h-4 w-4 flex-none" />
          {!collapsed && <span className="flex-1 truncate">New chat</span>}
          {!collapsed && (
            <kbd className="sp-kbd sp-mono flex-none rounded-md px-1.5 py-0.5 text-[9.5px] tracking-[0.08em]">⌘K</kbd>
          )}
        </button>
      </div>

      {/* Conversation list */}
      <div ref={listRef} className="sp-scroll flex-1 overflow-y-auto px-3 pb-3">
        {groups.map((g) => (
          <div key={g.label} className="mb-1">
            {!collapsed && (
              <p className="sp-mono sp-faint px-2 py-1.5 text-[9.5px] uppercase tracking-[0.3em]">{g.label}</p>
            )}
            {g.items.map((s) => {
              const active = s.id === activeSessionId;
              const isEditing = editingId === s.id;
              const isDeparting = departing.has(s.id);
              return (
                <motion.div
                  layout
                  key={s.id}
                  data-session-row={s.id}
                  style={{ opacity: isDeparting ? 0 : 1 }}
                  className={`group mb-1 flex items-center gap-1.5 rounded-[13px] px-2.5 ${
                    active ? 'sp-item-active' : 'sp-item'
                  } ${isDeparting ? 'pointer-events-none' : ''}`}
                >
                  {isEditing ? (
                    <>
                      <input
                        autoFocus
                        value={editTitle}
                        onChange={(e) => setEditTitle(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') commitEdit(s.id);
                          if (e.key === 'Escape') setEditingId(null);
                        }}
                        className="sp-text min-w-0 flex-1 bg-transparent py-2.5 text-[13px] outline-none"
                      />
                      <button type="button" onClick={() => commitEdit(s.id)} aria-label="Save" className="p-1">
                        <Check className="h-3.5 w-3.5" />
                      </button>
                      <button type="button" onClick={() => setEditingId(null)} aria-label="Cancel" className="p-1">
                        <X className="h-3.5 w-3.5" />
                      </button>
                    </>
                  ) : (
                    <>
                      <button
                        type="button"
                        onClick={() => onSelectSession(s.id)}
                        className="min-h-[44px] min-w-0 flex-1 truncate py-2.5 text-left text-[13px]"
                        title={s.title}
                      >
                        {collapsed ? (s.title || 'Chat').slice(0, 1).toUpperCase() : s.title || 'New chat'}
                      </button>
                      {!collapsed && (
                        <span className="flex flex-none items-center opacity-0 transition-opacity group-hover:opacity-100">
                          <button type="button" onClick={() => startEdit(s)} aria-label="Rename" className="p-1">
                            <Pencil className="h-3 w-3" />
                          </button>
                          <button type="button" onClick={(e) => handleDelete(s.id, e)} aria-label="Delete" className="p-1">
                            <Trash2 className="h-3 w-3" />
                          </button>
                        </span>
                      )}
                    </>
                  )}
                </motion.div>
              );
            })}
          </div>
        ))}

        {!collapsed && sessions.length > 1 && (
          <button
            type="button"
            onClick={handleClearAll}
            className="sp-mono sp-faint mt-1 flex w-full items-center justify-center gap-1.5 rounded-[11px] px-2.5 py-2 text-[9.5px] uppercase tracking-[0.2em] transition-colors hover:text-[var(--sp-text)]"
          >
            <Trash2 className="h-3 w-3" /> Clear all
          </button>
        )}
      </div>

      {/* Status card */}
      {!collapsed && (
        <div className="px-3 pb-4">
          <div className="sp-card rounded-2xl px-3.5 py-3">
            <div className="mb-2 flex items-center gap-2">
              <span
                className="h-[6px] w-[6px] rounded-full"
                style={{ background: grounded ? 'var(--sp-accent)' : '#e0b45a' }}
              />
              <span className="sp-mono sp-muted text-[9.5px] uppercase tracking-[0.24em]">
                {grounded ? 'Grounded' : 'Degraded'}
              </span>
              <span className="sp-mono ml-auto text-[10px] text-[var(--sp-accent-text)]">98%</span>
            </div>
            <dl className="space-y-1">
              {[
                ['Reranker', reranker],
                ['Model', modelLabel],
                ['Index', indexLabel],
              ].map(([k, v]) => (
                <div key={k} className="flex items-center justify-between gap-2">
                  <dt className="sp-mono sp-faint text-[9.5px] uppercase tracking-[0.16em]">{k}</dt>
                  <dd className="sp-muted max-w-[150px] truncate text-[11px]">{v}</dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
      )}
    </aside>
  );
}

export default SpaceSidebar;
