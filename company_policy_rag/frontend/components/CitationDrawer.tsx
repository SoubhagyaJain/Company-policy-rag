'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X,
  BookOpen,
  Copy,
  Check,
  Award,
  Hash,
  Folder,
  FileText,
  ExternalLink,
  ShieldAlert,
} from 'lucide-react';
import { Citation } from '../lib/types';
import { formatScore, cn } from '../lib/utils';

interface CitationDrawerProps {
  isOpen: boolean;
  citation: Citation | null;
  onClose: () => void;
}

export function CitationDrawer({
  isOpen,
  citation,
  onClose,
}: CitationDrawerProps) {
  const [copied, setCopied] = useState(false);

  const handleCopyChunk = () => {
    if (citation?.chunk_text) {
      navigator.clipboard.writeText(citation.chunk_text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (!citation) return null;

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop Overlay */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-charcoal/30 dark:bg-black/60 backdrop-blur-xs z-40"
          />

          {/* Sliding Drawer */}
          <motion.aside
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 220 }}
            className="fixed top-0 right-0 h-full w-full max-w-md bg-[#FAF9F5] dark:bg-[#181715] border-l border-[#E5E0D8] dark:border-[#2E2C27] shadow-2xl z-50 flex flex-col justify-between"
          >
            {/* Drawer Header */}
            <div className="p-4 border-b border-[#E5E0D8] dark:border-[#2E2C27] flex items-center justify-between bg-cream-100/60 dark:bg-sand-dark/60">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-xl bg-terracotta-500/10 dark:bg-terracotta-500/20 text-terracotta-600 dark:text-terracotta-400 flex items-center justify-center">
                  <BookOpen className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="font-serif font-semibold text-sm text-charcoal dark:text-cream-100">
                    Source Citation Details
                  </h3>
                  <p className="text-[11px] text-charcoal-muted dark:text-cream-400">
                    Verbatim Document Extract
                  </p>
                </div>
              </div>

              <button
                onClick={onClose}
                className="p-1.5 rounded-lg hover:bg-cream-200 dark:hover:bg-sand-dark text-charcoal-muted dark:text-cream-400 hover:text-charcoal transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Drawer Content */}
            <div className="flex-1 overflow-y-auto p-5 space-y-5 custom-scrollbar">
              {/* Document Title Card */}
              <div className="p-4 rounded-2xl bg-[#F3F0E6]/80 dark:bg-[#201F1C]/80 border border-[#E5E0D8] dark:border-[#2E2C27] space-y-2">
                <div className="flex items-center justify-between text-xs text-charcoal-muted dark:text-cream-400">
                  <span className="flex items-center gap-1.5 font-medium">
                    <FileText className="w-3.5 h-3.5 text-terracotta-600" />
                    Document Name
                  </span>
                  {citation.category && (
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-terracotta-500/10 text-terracotta-700 dark:text-terracotta-400 border border-terracotta-500/20">
                      {citation.category}
                    </span>
                  )}
                </div>

                <h4 className="font-serif text-base font-bold text-charcoal dark:text-cream-100 leading-snug">
                  {citation.title}
                </h4>

                {citation.source && (
                  <p className="text-[11px] font-mono text-charcoal-muted dark:text-cream-500 truncate">
                    Path: {citation.source}
                  </p>
                )}
              </div>

              {/* Rerank Match Score Progress */}
              {citation.score !== undefined && (
                <div className="p-3.5 rounded-xl bg-[#FAF9F5] dark:bg-[#1E1D1A] border border-[#E5E0D8] dark:border-[#2E2C27] space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="flex items-center gap-1.5 font-medium text-charcoal dark:text-cream-200">
                      <Award className="w-3.5 h-3.5 text-amber-500" /> RAG Rerank Confidence
                    </span>
                    <span className="font-mono font-bold text-terracotta-600 dark:text-terracotta-400">
                      {formatScore(citation.score)}
                    </span>
                  </div>

                  <div className="w-full h-2 rounded-full bg-cream-200 dark:bg-sand-dark overflow-hidden">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-amber-500 to-terracotta-600 transition-all duration-500"
                      style={{ width: `${Math.min(100, Math.max(0, citation.score * 100))}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Section & Page Metadata */}
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-xl bg-[#FAF9F5] dark:bg-[#1E1D1A] border border-[#E5E0D8] dark:border-[#2E2C27]">
                  <span className="text-[10px] uppercase font-semibold text-charcoal-muted dark:text-cream-500 block mb-0.5">
                    Page Reference
                  </span>
                  <span className="font-mono text-xs font-bold text-charcoal dark:text-cream-100 flex items-center gap-1">
                    <Hash className="w-3 h-3 text-terracotta-500" />
                    {citation.page !== undefined ? `Page ${citation.page}` : 'N/A'}
                  </span>
                </div>

                <div className="p-3 rounded-xl bg-[#FAF9F5] dark:bg-[#1E1D1A] border border-[#E5E0D8] dark:border-[#2E2C27]">
                  <span className="text-[10px] uppercase font-semibold text-charcoal-muted dark:text-cream-500 block mb-0.5">
                    Heading / Section
                  </span>
                  <span className="text-xs font-bold text-charcoal dark:text-cream-100 truncate block">
                    {citation.heading ? `§ ${citation.heading}` : 'General Section'}
                  </span>
                </div>
              </div>

              {/* Chunk Text Container */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-charcoal-muted dark:text-cream-400">
                    Indexed Text Chunk
                  </h4>
                  <button
                    onClick={handleCopyChunk}
                    className="flex items-center gap-1 text-[11px] text-terracotta-600 dark:text-terracotta-400 hover:underline focus:outline-none"
                  >
                    {copied ? (
                      <>
                        <Check className="w-3 h-3 text-emerald-600" /> Copied!
                      </>
                    ) : (
                      <>
                        <Copy className="w-3 h-3" /> Copy Snippet
                      </>
                    )}
                  </button>
                </div>

                <div className="p-4 rounded-2xl bg-[#F3F0E6]/90 dark:bg-[#22211E]/90 border border-[#E5E0D8] dark:border-[#2E2C27] text-xs text-charcoal dark:text-cream-100 font-serif leading-relaxed italic whitespace-pre-wrap selection:bg-terracotta-500/20">
                  "{citation.chunk_text}"
                </div>
              </div>

              <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-[11px] text-amber-800 dark:text-amber-300 flex items-start gap-2">
                <ShieldAlert className="w-4 h-4 shrink-0 text-amber-600 mt-0.5" />
                <p>
                  This text snippet was extracted directly from the verified corporate policy document index.
                </p>
              </div>
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-[#E5E0D8] dark:border-[#2E2C27] bg-cream-100/60 dark:bg-sand-dark/60">
              <button
                onClick={onClose}
                className="w-full py-2 px-4 rounded-xl bg-charcoal dark:bg-cream-100 text-cream-100 dark:text-charcoal text-xs font-medium hover:bg-charcoal/90 dark:hover:bg-cream-200 transition-colors"
              >
                Done Reading
              </button>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
