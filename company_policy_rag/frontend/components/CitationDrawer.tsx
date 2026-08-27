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
  FileText,
  ShieldCheck,
  Layers,
  Sparkles,
  ExternalLink,
  ZoomIn,
  Image as ImageIcon,
} from 'lucide-react';
import { Citation } from '../lib/types';
import { formatScore } from '../lib/utils';

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
  const [isZoomed, setIsZoomed] = useState(false);

  const handleCopyChunk = () => {
    if (citation?.chunk_text) {
      navigator.clipboard.writeText(citation.chunk_text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (!citation) return null;

  const scorePct = citation.score !== undefined ? formatScore(citation.score) : null;
  const wordCount = citation.chunk_text ? citation.chunk_text.trim().split(/\s+/).length : 0;
  const charCount = citation.chunk_text ? citation.chunk_text.length : 0;
  const displayPage = citation.display_page_number ?? citation.page_label ?? citation.page;
  const isVisual = citation.evidence_type === 'DIAGRAM_ARCHITECTURE' || citation.evidence_type === 'CODE_SCREENSHOT' || citation.evidence_type === 'TABLE_DATA' || Boolean(citation.image_url);

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
            className="fixed inset-0 bg-[#141413]/40 dark:bg-black/70 backdrop-blur-xs z-40 transition-opacity"
          />

          {/* Sliding Drawer */}
          <motion.aside
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 28, stiffness: 240 }}
            className="fixed top-0 right-0 h-full w-full max-w-lg bg-[#FAF8F5] dark:bg-[#181715] border-l border-[#E2DDD5] dark:border-[#2C2A26] shadow-2xl z-50 flex flex-col justify-between font-sans"
          >
            {/* Drawer Header */}
            <div className="p-4 sm:p-5 border-b border-[#E8E2D5] dark:border-[#282622] flex items-center justify-between bg-[#F4F0E6]/80 dark:bg-[#1E1D1A]/80 backdrop-blur-md">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-2xl bg-terracotta-500/15 dark:bg-terracotta-500/25 text-terracotta-600 dark:text-terracotta-400 flex items-center justify-center border border-terracotta-500/20 shadow-xs">
                  <BookOpen className="w-4.5 h-4.5" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="font-serif font-bold text-sm text-[#23201C] dark:text-[#FAF8F5]">
                      Grounding Citation [{citation.source_index ?? 1}]
                    </h3>
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-terracotta-600 text-white">
                      Verified
                    </span>
                  </div>
                  <p className="text-xs text-[#736D62] dark:text-[#9A9385]">
                    Verbatim Document Index Extract
                  </p>
                </div>
              </div>

              <button
                onClick={onClose}
                className="p-2 rounded-xl hover:bg-[#EAE4D8] dark:hover:bg-[#282622] text-[#736D62] dark:text-[#9A9385] hover:text-[#23201C] dark:hover:text-[#FAF8F5] transition-colors"
                title="Close drawer"
              >
                <X className="w-4.5 h-4.5" />
              </button>
            </div>

            {/* Drawer Content Body */}
            <div className="flex-1 overflow-y-auto p-5 sm:p-6 space-y-5 custom-scrollbar">
              {/* Document Identity Card */}
              <div className="p-4 rounded-2xl bg-[#F2EDE2]/90 dark:bg-[#201F1C]/90 border border-[#E0D8CA] dark:border-[#2F2D28] space-y-2.5">
                <div className="flex items-center justify-between text-xs text-[#736D62] dark:text-[#9A9385]">
                  <span className="flex items-center gap-1.5 font-medium text-[11px] uppercase tracking-wider text-charcoal-muted dark:text-cream-400">
                    <FileText className="w-3.5 h-3.5 text-terracotta-600 dark:text-terracotta-400" />
                    Source Document
                  </span>
                  {citation.category && (
                    <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-medium bg-[#E5DFD0] dark:bg-[#2B2925] text-[#3D3830] dark:text-[#D5CEC2] border border-[#D8CFBF] dark:border-[#38352F]">
                      {citation.category}
                    </span>
                  )}
                </div>

                <h4 className="font-serif text-base font-bold text-[#1E1C1A] dark:text-[#FAF8F5] leading-snug break-all">
                  {citation.document_name || citation.title || citation.source || 'Policy Document'}
                </h4>

                {citation.source && citation.source !== citation.title && (
                  <p className="text-[11px] font-mono text-[#787265] dark:text-[#908A7C] truncate">
                    File: {citation.source}
                  </p>
                )}
              </div>

              {/* Rerank Match & Relevance Meter */}
              {citation.score !== undefined && (
                <div className="p-4 rounded-2xl bg-[#FAF8F5] dark:bg-[#1E1D1A] border border-[#E5DFD2] dark:border-[#2D2B26] space-y-2.5 shadow-xs">
                  <div className="flex items-center justify-between text-xs">
                    <span className="flex items-center gap-1.5 font-semibold text-[#2D2A25] dark:text-[#E8E3D8]">
                      <Award className="w-4 h-4 text-amber-600 dark:text-amber-400" /> Cross-Encoder Confidence
                    </span>
                    <span className="font-mono font-bold text-sm text-terracotta-600 dark:text-terracotta-400">
                      {scorePct}
                    </span>
                  </div>

                  <div className="w-full h-2 rounded-full bg-[#EAE4D8] dark:bg-[#2B2925] overflow-hidden">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-amber-500 via-terracotta-500 to-terracotta-600 transition-all duration-500"
                      style={{
                        width: `${Math.min(
                          100,
                          Math.max(15, citation.score > 1.0 ? 92 : citation.score * 100)
                        )}%`,
                      }}
                    />
                  </div>

                  <div className="flex items-center justify-between text-[10px] text-[#7A7468] dark:text-[#8C867B] font-mono pt-0.5">
                    <span>Ranked by BGE Cross-Encoder</span>
                    <span>High Grounding Match</span>
                  </div>
                </div>
              )}

              {/* Metadata Grid */}
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3.5 rounded-2xl bg-[#F7F4EC] dark:bg-[#1E1D1A] border border-[#E5DFD2] dark:border-[#2D2B26]">
                  <span className="text-[10px] uppercase font-semibold text-[#7A7468] dark:text-[#8C867B] block mb-1">
                    Document Page Reference
                  </span>
                  <span
                    className="font-mono text-xs font-bold text-[#23201C] dark:text-[#FAF8F5] flex items-center gap-1.5"
                    title={citation.physical_page_number && String(citation.physical_page_number) !== String(displayPage) ? `PDF physical stream sheet ${citation.physical_page_number}` : undefined}
                  >
                    <Hash className="w-3.5 h-3.5 text-terracotta-600 dark:text-terracotta-400 shrink-0" />
                    {displayPage !== undefined ? `Page ${displayPage}` : 'Full Document'}
                  </span>
                </div>

                <div className="p-3.5 rounded-2xl bg-[#F7F4EC] dark:bg-[#1E1D1A] border border-[#E5DFD2] dark:border-[#2D2B26]">
                  <span className="text-[10px] uppercase font-semibold text-[#7A7468] dark:text-[#8C867B] block mb-1">
                    Section Heading
                  </span>
                  <span className="text-xs font-bold text-[#23201C] dark:text-[#FAF8F5] truncate block">
                    {citation.heading && citation.heading !== 'CONTENTS'
                      ? `§ ${citation.heading}`
                      : 'General Policy Section'}
                  </span>
                </div>
              </div>

              {/* Original Visual Image Asset Preview (if present) */}
              {citation.image_url && (
                <div className="p-4 rounded-2xl bg-[#FAF8F5] dark:bg-[#1E1D1A] border border-[#E0D8CA] dark:border-[#2E2C27] space-y-2.5 shadow-xs">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold uppercase tracking-wider text-[#4A453D] dark:text-[#C5BEB2] flex items-center gap-1.5">
                      <ImageIcon className="w-4 h-4 text-terracotta-600" />
                      <span>Original Document Visual Asset</span>
                    </span>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => setIsZoomed(true)}
                        className="text-[11px] text-[#736D62] hover:text-[#23201C] dark:text-[#9A9385] dark:hover:text-[#FAF8F5] flex items-center gap-1 font-mono"
                      >
                        <ZoomIn className="w-3 h-3" /> Zoom
                      </button>
                      <a
                        href={citation.image_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[11px] text-terracotta-600 dark:text-terracotta-400 hover:underline flex items-center gap-1 font-mono font-medium"
                      >
                        Full Res <ExternalLink className="w-3 h-3" />
                      </a>
                    </div>
                  </div>

                  <div
                    onClick={() => setIsZoomed(true)}
                    className="group relative rounded-xl overflow-hidden border border-[#E5E0D8] dark:border-[#33302A] bg-[#111] max-h-72 flex items-center justify-center cursor-zoom-in"
                  >
                    <img
                      src={citation.image_url}
                      alt={citation.heading || 'Original Document Diagram'}
                      className="max-h-72 w-auto object-contain group-hover:scale-102 transition-transform"
                    />
                    <div className="absolute bottom-2 right-2 px-2 py-1 rounded bg-black/60 text-white text-[10px] font-mono opacity-0 group-hover:opacity-100 transition-opacity">
                      Click to expand
                    </div>
                  </div>
                  <p className="text-[10px] font-mono text-[#8C867B] dark:text-[#736E65]">
                    Original high-resolution visual evidence extracted directly from PDF Page {displayPage ?? ''}.
                  </p>
                </div>
              )}

              {/* Full Verbatim Text Chunk */}
              <div className="space-y-2">
                <div className="flex items-center justify-between px-1">
                  <div className="flex items-center gap-2">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-[#4A453D] dark:text-[#C5BEB2]">
                      Verbatim Document Extract
                    </h4>
                    <span className="text-[10px] font-mono text-[#8C867B] dark:text-[#736E65]">
                      ({wordCount} words · {charCount} chars)
                    </span>
                  </div>

                  <button
                    onClick={handleCopyChunk}
                    className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium text-terracotta-600 dark:text-terracotta-400 hover:bg-[#EAE4D8] dark:hover:bg-[#25231F] transition-colors"
                  >
                    {copied ? (
                      <>
                        <Check className="w-3.5 h-3.5 text-emerald-600" />
                        <span className="text-emerald-700 dark:text-emerald-400">Copied</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5" />
                        <span>Copy Text</span>
                      </>
                    )}
                  </button>
                </div>

                {/* Text Content Container */}
                <div className="p-4 sm:p-5 rounded-2xl bg-[#F2EDE2]/90 dark:bg-[#201F1C]/90 border border-[#DDD5C5] dark:border-[#2F2D28] text-xs sm:text-sm text-[#23201C] dark:text-[#EFEAE1] font-serif leading-relaxed whitespace-pre-wrap selection:bg-terracotta-500/20 shadow-inner">
                  {citation.chunk_text || 'No extract text available for this citation.'}
                </div>
              </div>

              {/* Verified Policy Stamp */}
              <div className="p-3.5 rounded-2xl bg-emerald-500/10 dark:bg-emerald-500/15 border border-emerald-500/25 text-xs text-emerald-800 dark:text-emerald-300 flex items-start gap-2.5">
                <ShieldCheck className="w-4 h-4 shrink-0 text-emerald-600 dark:text-emerald-400 mt-0.5" />
                <p className="leading-relaxed text-[11px]">
                  This extract is grounded directly in the verified document repository. It was retrieved via Dense Vector + BM25 search and validated by BGE Cross-Encoder reranking.
                </p>
              </div>
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-[#E8E2D5] dark:border-[#282622] bg-[#F4F0E6]/80 dark:bg-[#1E1D1A]/80 backdrop-blur-md">
              <button
                onClick={onClose}
                className="w-full py-2.5 px-4 rounded-xl bg-[#23201C] dark:bg-[#FAF8F5] text-white dark:text-[#181715] text-xs font-semibold hover:bg-[#38342E] dark:hover:bg-[#EFEAE1] shadow-sm transition-all active:scale-98"
              >
                Close Extract View
              </button>
            </div>
          </motion.aside>

          {/* Full Screen Image Zoom Modal */}
          {isZoomed && citation.image_url && (
            <div
              onClick={() => setIsZoomed(false)}
              className="fixed inset-0 z-60 bg-black/90 backdrop-blur-md flex items-center justify-center p-4 cursor-zoom-out"
            >
              <div className="relative max-w-5xl max-h-[90vh] flex flex-col items-center">
                <img
                  src={citation.image_url}
                  alt={citation.heading || 'Full Resolution Diagram'}
                  className="max-w-full max-h-[85vh] object-contain rounded-lg shadow-2xl"
                  onClick={(e) => e.stopPropagation()}
                />
                <div className="mt-3 text-white text-xs font-mono flex items-center gap-4">
                  <span>Page {displayPage ?? ''} · {citation.heading ?? 'Visual Asset'}</span>
                  <button
                    onClick={() => setIsZoomed(false)}
                    className="px-3 py-1 rounded bg-white/20 hover:bg-white/30 text-white transition-colors"
                  >
                    Close Zoom
                  </button>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </AnimatePresence>
  );
}
