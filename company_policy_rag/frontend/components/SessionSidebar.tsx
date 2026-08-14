'use client';

import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Plus,
  MessageSquare,
  Trash2,
  Edit2,
  Check,
  X,
  Search,
  Sparkles,
  Shield,
  Layers,
} from 'lucide-react';
import { ChatSession } from '../lib/types';
import { cn } from '../lib/utils';

interface SessionSidebarProps {
  isOpen: boolean;
  sessions: ChatSession[];
  activeSessionId: string;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string) => void;
  onRenameSession: (id: string, newTitle: string) => void;
}

function groupSessionsByDate(sessions: ChatSession[]) {
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const startOfYesterday = startOfToday - 86400000;
  const startOfLast7Days = startOfToday - 7 * 86400000;

  const groups: { label: string; items: ChatSession[] }[] = [
    { label: 'Today', items: [] },
    { label: 'Yesterday', items: [] },
    { label: 'Previous 7 Days', items: [] },
    { label: 'Older', items: [] },
  ];

  for (const session of sessions) {
    const time = new Date(session.updatedAt || session.createdAt).getTime();
    if (time >= startOfToday) {
      groups[0].items.push(session);
    } else if (time >= startOfYesterday) {
      groups[1].items.push(session);
    } else if (time >= startOfLast7Days) {
      groups[2].items.push(session);
    } else {
      groups[3].items.push(session);
    }
  }

  return groups.filter((g) => g.items.length > 0);
}

export function SessionSidebar({
  isOpen,
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  onRenameSession,
}: SessionSidebarProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState<string>('');

  const filteredSessions = useMemo(() => {
    if (!searchQuery.trim()) return sessions;
    const q = searchQuery.toLowerCase();
    return sessions.filter((s) => s.title.toLowerCase().includes(q));
  }, [sessions, searchQuery]);

  const grouped = useMemo(() => groupSessionsByDate(filteredSessions), [filteredSessions]);

  const handleStartRename = (session: ChatSession, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(session.id);
    setEditTitle(session.title);
  };

  const handleSaveRename = (id: string, e: React.FormEvent) => {
    e.preventDefault();
    if (editTitle.trim()) {
      onRenameSession(id, editTitle.trim());
    }
    setEditingId(null);
  };

  const handleCancelRename = (e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(null);
  };

  const handleDelete = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    onDeleteSession(id);
  };

  return (
    <AnimatePresence mode="wait">
      {isOpen && (
        <motion.aside
          initial={{ x: -280, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: -280, opacity: 0 }}
          transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
          className="w-72 shrink-0 h-[calc(100vh-57px)] sticky top-[57px] bg-[#F7F4EB] dark:bg-[#181715] border-r border-[#E8E2D5] dark:border-[#272522] flex flex-col justify-between p-3.5 z-20 select-none font-sans"
        >
          {/* Top Section */}
          <div className="flex flex-col h-full overflow-hidden">
            {/* Anthropic style New Chat Button */}
            <button
              onClick={onNewSession}
              className="w-full flex items-center justify-between py-2.5 px-3.5 rounded-xl bg-[#EBE5D8] hover:bg-[#E2DBCC] dark:bg-[#262421] dark:hover:bg-[#302D29] text-[#2D2A26] dark:text-[#E8E4DD] border border-[#DDD5C5] dark:border-[#383530] font-medium text-xs shadow-sm transition-all active:scale-[0.99] group mb-3"
            >
              <div className="flex items-center gap-2">
                <div className="w-5 h-5 rounded-lg bg-terracotta-600/10 dark:bg-terracotta-500/20 text-terracotta-600 dark:text-terracotta-400 flex items-center justify-center">
                  <Plus className="w-3.5 h-3.5 stroke-[2.5]" />
                </div>
                <span className="font-semibold tracking-tight">New chat</span>
              </div>
              <span className="text-[10px] font-mono text-charcoal-muted dark:text-cream-500 opacity-60">
                ⌘K
              </span>
            </button>

            {/* Quick search input if more than 3 sessions */}
            {sessions.length > 3 && (
              <div className="relative mb-2">
                <Search className="w-3 h-3 text-charcoal-muted dark:text-cream-500 absolute left-2.5 top-2.5 pointer-events-none" />
                <input
                  type="text"
                  placeholder="Search chats..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-7 pr-3 py-1.5 rounded-lg bg-[#EFECE2]/70 dark:bg-[#201F1C] border border-[#E0D9CB] dark:border-[#2D2B27] text-xs text-charcoal dark:text-cream-100 placeholder:text-charcoal-muted dark:placeholder:text-cream-500 focus:outline-none focus:border-terracotta-500 transition-colors"
                />
              </div>
            )}

            {/* Grouped Session List */}
            <div className="flex-1 overflow-y-auto space-y-4 pr-1 custom-scrollbar pt-1">
              {grouped.length === 0 ? (
                <div className="py-10 text-center text-xs text-charcoal-muted dark:text-cream-400">
                  {searchQuery ? 'No matching chats found' : 'No conversations yet'}
                </div>
              ) : (
                grouped.map((group) => (
                  <div key={group.label} className="space-y-1">
                    <div className="px-2 text-[10px] font-semibold tracking-wider text-charcoal-muted/70 dark:text-cream-500 uppercase">
                      {group.label}
                    </div>

                    {group.items.map((session) => {
                      const isActive = session.id === activeSessionId;
                      const isEditing = editingId === session.id;

                      return (
                        <div
                          key={session.id}
                          onClick={() => onSelectSession(session.id)}
                          className={cn(
                            'group relative flex items-center justify-between px-2.5 py-2 rounded-xl cursor-pointer text-xs transition-all duration-150',
                            isActive
                              ? 'bg-[#EAE4D6] dark:bg-[#282622] text-[#1E1C1A] dark:text-[#FAF8F5] font-semibold border border-[#D9D1C1] dark:border-[#383530] shadow-sm'
                              : 'text-[#4A463F] dark:text-[#C5BFB5] hover:bg-[#EFECE2]/70 dark:hover:bg-[#21201D]'
                          )}
                        >
                          <div className="flex items-center gap-2 min-w-0 flex-1">
                            <MessageSquare
                              className={cn(
                                'w-3.5 h-3.5 shrink-0 opacity-70',
                                isActive
                                  ? 'text-terracotta-600 dark:text-terracotta-400 opacity-100'
                                  : 'text-charcoal-muted dark:text-cream-400'
                              )}
                            />

                            {isEditing ? (
                              <form
                                onSubmit={(e) => handleSaveRename(session.id, e)}
                                className="flex items-center gap-1 w-full"
                              >
                                <input
                                  type="text"
                                  value={editTitle}
                                  onChange={(e) => setEditTitle(e.target.value)}
                                  className="w-full bg-[#FAF9F5] dark:bg-[#1A1917] px-1.5 py-0.5 rounded border border-terracotta-500 text-xs focus:outline-none"
                                  autoFocus
                                  onClick={(e) => e.stopPropagation()}
                                />
                                <button
                                  type="submit"
                                  onClick={(e) => e.stopPropagation()}
                                  className="p-1 hover:text-emerald-600"
                                >
                                  <Check className="w-3 h-3" />
                                </button>
                                <button
                                  type="button"
                                  onClick={handleCancelRename}
                                  className="p-1 hover:text-rose-600"
                                >
                                  <X className="w-3 h-3" />
                                </button>
                              </form>
                            ) : (
                              <p className="truncate text-xs leading-snug">
                                {session.title || 'Untitled Conversation'}
                              </p>
                            )}
                          </div>

                          {/* Hover action buttons */}
                          {!isEditing && (
                            <div className="opacity-0 group-hover:opacity-100 flex items-center gap-0.5 transition-opacity shrink-0 ml-1">
                              <button
                                onClick={(e) => handleStartRename(session, e)}
                                title="Rename"
                                className="p-1 hover:bg-[#DDD5C5] dark:hover:bg-[#383530] rounded-md text-charcoal-muted dark:text-cream-400 hover:text-charcoal transition-colors"
                              >
                                <Edit2 className="w-3 h-3" />
                              </button>
                              <button
                                onClick={(e) => handleDelete(session.id, e)}
                                title="Delete"
                                className="p-1 hover:bg-rose-100 dark:hover:bg-rose-950/40 rounded-md text-charcoal-muted hover:text-rose-600 transition-colors"
                              >
                                <Trash2 className="w-3 h-3" />
                              </button>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Bottom Specs Card */}
          <div className="pt-3 border-t border-[#E8E2D5] dark:border-[#272522]">
            <div className="p-2.5 rounded-xl bg-[#EFECE2]/70 dark:bg-[#201F1C] border border-[#E0D9CB] dark:border-[#2D2B27] text-[11px] space-y-1.5">
              <div className="flex items-center justify-between text-charcoal-muted dark:text-cream-400">
                <span className="flex items-center gap-1.5 font-medium">
                  <Shield className="w-3 h-3 text-terracotta-600" /> RAG Security
                </span>
                <span className="font-mono text-[10px] text-emerald-600 dark:text-emerald-400 font-semibold">
                  Grounded
                </span>
              </div>
              <div className="flex items-center justify-between text-charcoal-muted dark:text-cream-400">
                <span className="flex items-center gap-1.5 font-medium">
                  <Layers className="w-3 h-3 text-amber-600" /> Reranker
                </span>
                <span className="font-mono text-[10px] text-charcoal dark:text-cream-200">
                  BGE Large
                </span>
              </div>
            </div>
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  );
}
