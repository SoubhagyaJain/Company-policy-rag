'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Plus,
  MessageSquare,
  Trash2,
  Edit2,
  Check,
  X,
  Sparkles,
  Database,
  Layers,
} from 'lucide-react';
import { ChatSession } from '../lib/types';
import { formatDate, cn } from '../lib/utils';

interface SessionSidebarProps {
  isOpen: boolean;
  sessions: ChatSession[];
  activeSessionId: string;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string) => void;
  onRenameSession: (id: string, newTitle: string) => void;
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
          transition={{ duration: 0.25, ease: 'easeInOut' }}
          className="w-72 shrink-0 h-[calc(100vh-57px)] sticky top-[57px] backdrop-blur-xl bg-[#F3F0E6]/90 dark:bg-[#1A1917]/90 border-r border-[#E5E0D8]/80 dark:border-[#2A2925]/80 flex flex-col justify-between p-3.5 z-20"
        >
          {/* Top: New Session Button & Session List */}
          <div className="flex flex-col h-full overflow-hidden">
            <button
              onClick={onNewSession}
              className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl bg-terracotta-600 hover:bg-terracotta-700 text-white font-medium text-xs shadow-md shadow-terracotta-600/20 transition-all active:scale-[0.98]"
            >
              <Plus className="w-4 h-4" />
              <span>New Conversation</span>
            </button>

            <div className="mt-4 mb-2 px-1 flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-charcoal-muted dark:text-cream-400">
                Chat History
              </span>
              <span className="text-[10px] font-mono text-charcoal-muted dark:text-cream-500">
                {sessions.length} sessions
              </span>
            </div>

            {/* Sessions List */}
            <div className="flex-1 overflow-y-auto space-y-1 pr-1 custom-scrollbar">
              {sessions.length === 0 ? (
                <div className="py-8 text-center text-xs text-charcoal-muted dark:text-cream-400">
                  No active sessions
                </div>
              ) : (
                sessions.map((session) => {
                  const isActive = session.id === activeSessionId;
                  const isEditing = editingId === session.id;

                  return (
                    <div
                      key={session.id}
                      onClick={() => onSelectSession(session.id)}
                      className={cn(
                        'group relative flex items-center justify-between p-2.5 rounded-xl cursor-pointer text-xs transition-all duration-150',
                        isActive
                          ? 'bg-[#FAF9F5] dark:bg-[#252420] text-charcoal dark:text-cream-100 shadow-sm font-medium border border-[#E5E0D8] dark:border-[#33312B]'
                          : 'text-charcoal/80 dark:text-cream-300 hover:bg-cream-200/50 dark:hover:bg-sand-dark/50'
                      )}
                    >
                      <div className="flex items-center gap-2.5 min-w-0 flex-1">
                        <MessageSquare
                          className={cn(
                            'w-3.5 h-3.5 shrink-0',
                            isActive
                              ? 'text-terracotta-600 dark:text-terracotta-500'
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
                              className="w-full bg-cream-50 dark:bg-charcoal-dark px-1.5 py-0.5 rounded border border-terracotta-500 text-xs focus:outline-none"
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
                          <div className="min-w-0 flex-1">
                            <p className="truncate text-xs font-medium leading-tight">
                              {session.title || 'Untitled Session'}
                            </p>
                            <p className="text-[10px] text-charcoal-muted dark:text-cream-500 mt-0.5">
                              {formatDate(session.updatedAt || session.createdAt)}
                            </p>
                          </div>
                        )}
                      </div>

                      {/* Action buttons on hover */}
                      {!isEditing && (
                        <div className="opacity-0 group-hover:opacity-100 flex items-center gap-1 transition-opacity">
                          <button
                            onClick={(e) => handleStartRename(session, e)}
                            title="Rename"
                            className="p-1 hover:bg-cream-300/60 dark:hover:bg-charcoal-dark rounded text-charcoal-muted dark:text-cream-400 hover:text-charcoal"
                          >
                            <Edit2 className="w-3 h-3" />
                          </button>
                          <button
                            onClick={(e) => handleDelete(session.id, e)}
                            title="Delete"
                            className="p-1 hover:bg-rose-100 dark:hover:bg-rose-900/30 rounded text-charcoal-muted hover:text-rose-600"
                          >
                            <Trash2 className="w-3 h-3" />
                          </button>
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </div>

          {/* Bottom Footer Specs */}
          <div className="pt-3 border-t border-[#E5E0D8]/80 dark:border-[#2A2925]/80 space-y-2">
            <div className="p-2.5 rounded-xl bg-cream-50/70 dark:bg-charcoal-dark/70 border border-sand-border/50 dark:border-sand-darkBorder/50 text-[11px] space-y-1.5">
              <div className="flex items-center justify-between text-charcoal-muted dark:text-cream-400">
                <span className="flex items-center gap-1.5 font-medium">
                  <Database className="w-3 h-3 text-terracotta-600" /> Vector DB
                </span>
                <span className="font-mono text-[10px] text-emerald-600 dark:text-emerald-400 font-semibold">
                  Chroma / FAISS
                </span>
              </div>

              <div className="flex items-center justify-between text-charcoal-muted dark:text-cream-400">
                <span className="flex items-center gap-1.5 font-medium">
                  <Layers className="w-3 h-3 text-amber-600" /> Reranker
                </span>
                <span className="font-mono text-[10px] text-charcoal dark:text-cream-200">
                  BGE-Reranker
                </span>
              </div>

              <div className="flex items-center justify-between text-charcoal-muted dark:text-cream-400">
                <span className="flex items-center gap-1.5 font-medium">
                  <Sparkles className="w-3 h-3 text-terracotta-500" /> Multi-Query
                </span>
                <span className="font-mono text-[10px] text-charcoal dark:text-cream-200">
                  HyDE Decomp
                </span>
              </div>
            </div>
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  );
}
