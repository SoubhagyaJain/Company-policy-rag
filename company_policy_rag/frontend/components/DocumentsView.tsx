'use client';

import React, { useState, useRef, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload,
  FileText,
  Trash2,
  RefreshCw,
  CheckCircle2,
  Clock,
  XCircle,
  FileCode2,
  FileSpreadsheet,
  FileBadge,
  HardDrive,
  Layers,
  AlertTriangle,
  RotateCcw,
  Sparkles,
  Eye,
  Search,
  CopyCheck,
} from 'lucide-react';

import { LiquidGlassCard } from '@/components/LiquidGlassCard';
import { useDocuments } from '@/hooks/useDocuments';
import { useSmoothScroll } from '@/hooks/useSmoothScroll';
import { formatBytes, formatDate } from '@/lib/utils';
import type { DocumentItem } from '@/lib/types';

/* ─── Helpers ──────────────────────────────────── */
function getFileIcon(fileType?: string) {
  switch (fileType) {
    case 'pdf':
      return <FileBadge className="w-5 h-5 text-rose-500" />;
    case 'csv':
    case 'xlsx':
      return <FileSpreadsheet className="w-5 h-5 text-emerald-600" />;
    case 'json':
    case 'md':
    case 'py':
      return <FileCode2 className="w-5 h-5 text-sky-500" />;
    default:
      return <FileText className="w-5 h-5 text-terracotta-600" />;
  }
}

function getStatusBadge(doc: DocumentItem) {
  const statusLower = (doc.status || '').toLowerCase();

  if (statusLower === 'ready' || statusLower === 'indexed' || statusLower === 'ready_with_vision') {
    const assetsCount = doc.visual_assets_count ?? (doc.image_assets?.length ?? 0);
    return (
      <div className="flex items-center gap-1.5 flex-wrap">
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border border-emerald-500/20">
          <CheckCircle2 className="w-3 h-3" /> Text Ready
        </span>
        {assetsCount > 0 ? (
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[9.5px] font-mono bg-sky-500/10 text-sky-700 dark:text-sky-300 border border-sky-500/20" title={`${assetsCount} standalone high-res images extracted and available`}>
            <Eye className="w-2.5 h-2.5" /> {assetsCount} Visual {assetsCount === 1 ? 'Asset' : 'Assets'}
          </span>
        ) : (
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[9.5px] font-mono bg-cream-200 text-charcoal-muted dark:bg-sand-dark dark:text-cream-400 border border-sand-border dark:border-sand-darkBorder" title="Document contains pure text content">
            Text Only
          </span>
        )}
      </div>
    );
  }

  if (statusLower === 'processing' || statusLower === 'text_indexing' || statusLower === 'vision_processing') {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-amber-500/10 text-amber-700 dark:text-amber-400 border border-amber-500/20 animate-pulse">
        <Clock className="w-3 h-3 animate-spin" /> {doc.current_stage || 'Indexing'} ({doc.progress ?? 50}%)
      </span>
    );
  }

  if (statusLower === 'partially_indexed') {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-sky-500/10 text-sky-700 dark:text-sky-400 border border-sky-500/20">
        <CheckCircle2 className="w-3 h-3" /> Text Ready (Visual Partial)
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-rose-500/10 text-rose-700 dark:text-rose-400 border border-rose-500/20">
      <XCircle className="w-3 h-3" /> Failed
    </span>
  );
}

/* ─── List item animation variants ─────────────── */
const listItemVariants = {
  hidden: { opacity: 0, y: 16, scale: 0.97 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    scale: 1,
    transition: { delay: i * 0.05, duration: 0.25, ease: 'easeOut' },
  }),
  exit: { opacity: 0, x: -30, transition: { duration: 0.2 } },
};

/* ─── Document Manager View ────────────────────── */
export function DocumentsView() {
  const {
    documents,
    loading,
    uploading,
    uploadProgress,
    currentStage,
    stageMessage,
    error,
    duplicateCount,
    deduplicating,
    refreshDocuments,
    uploadDocument,
    retryDocument,
    deleteDocument,
    removeDuplicates,
  } = useDocuments();

  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('General');
  const [retryingId, setRetryingId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Buttery inertia scrolling over the live WebGL background.
  const scrollRef = useRef<HTMLDivElement>(null);
  useSmoothScroll(scrollRef);

  const CATEGORIES = ['General', 'HR & Benefits', 'Operations', 'IT & Security', 'Finance', 'Compliance'];
  const visibleDocuments = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    if (!query) return documents;
    return documents.filter((doc) =>
      [doc.filename, doc.category, doc.file_type, doc.file_hash]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(query)),
    );
  }, [documents, searchQuery]);

  /* ── Drag & drop handlers ────────────────────── */
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const files = Array.from(e.dataTransfer.files);
      for (const file of files) {
        await uploadDocument(file, selectedCategory);
      }
    },
    [uploadDocument, selectedCategory],
  );

  const handleFileSelect = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files ?? []);
      for (const file of files) {
        await uploadDocument(file, selectedCategory);
      }
      if (fileInputRef.current) fileInputRef.current.value = '';
    },
    [uploadDocument, selectedCategory],
  );

  const handleRetry = async (docId: string) => {
    setRetryingId(docId);
    await retryDocument(docId);
    setRetryingId(null);
  };

  return (
    <div ref={scrollRef} className="flex-1 h-full overflow-y-auto p-4 sm:p-6 lg:p-8 sp-scroll">
      <div className="max-w-5xl mx-auto space-y-6">
        {/* ── Page heading ────────────────────────── */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <h1 className="sp-heading text-2xl font-semibold tracking-tight">
              Document Manager
            </h1>
            <p className="sp-muted text-sm mt-0.5">
              Upload, index, and manage your corporate policy knowledge base.
            </p>
          </div>

          <div className="flex items-center gap-2">
            {duplicateCount > 0 && (
              <button
                onClick={() => {
                  if (window.confirm(`Remove ${duplicateCount} byte-identical duplicate documents? The best indexed copy of each file will be kept.`)) {
                    void removeDuplicates();
                  }
                }}
                disabled={deduplicating}
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-amber-500/10 border border-amber-500/30 text-xs font-semibold text-amber-800 dark:text-amber-300 hover:bg-amber-500/20 transition-colors disabled:opacity-50"
                title="Keep the newest indexed copy and remove only byte-identical duplicates"
              >
                <CopyCheck className="w-3.5 h-3.5" />
                {deduplicating ? 'Cleaning…' : `Remove ${duplicateCount} duplicates`}
              </button>
            )}
            <button
              onClick={refreshDocuments}
              disabled={loading}
              className="sp-chip flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>

        {/* ── Error banner ────────────────────────── */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-sm text-rose-700 dark:text-rose-400 flex items-center justify-between gap-2"
          >
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          </motion.div>
        )}

        {/* ── Upload dropzone ─────────────────────── */}
        <LiquidGlassCard variant="space" className="p-0 overflow-hidden">
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`relative flex flex-col items-center justify-center gap-3 p-8 border-2 border-dashed rounded-2xl transition-all cursor-pointer ${
              isDragOver
                ? 'border-terracotta-500 bg-terracotta-500/5 dark:bg-terracotta-500/10'
                : 'border-sand-border dark:border-sand-darkBorder hover:border-terracotta-500/40'
            }`}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              multiple
              accept=".pdf,.docx,.txt,.csv,.json,.md,.py,.xlsx"
              onChange={handleFileSelect}
            />

            <div className="w-12 h-12 rounded-2xl bg-terracotta-500/10 dark:bg-terracotta-500/20 flex items-center justify-center">
              <Upload className="w-6 h-6 text-terracotta-600 dark:text-terracotta-500" />
            </div>

            <div className="text-center">
              <p className="text-sm font-medium text-charcoal dark:text-cream-100">
                {isDragOver ? 'Drop files here to upload' : 'Drag & drop files or click to browse'}
              </p>
              <p className="text-xs text-charcoal-muted dark:text-cream-400 mt-1">
                PDF, DOCX, TXT, CSV, JSON, MD — max 100 MB per file
              </p>
            </div>

            {/* Category selector */}
            <div className="flex items-center gap-1.5 mt-2 flex-wrap justify-center">
              {CATEGORIES.map((cat) => (
                <button
                  key={cat}
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedCategory(cat);
                  }}
                  className={`px-2.5 py-0.5 rounded-lg text-[11px] font-mono transition-colors ${
                    selectedCategory === cat
                      ? 'bg-terracotta-600 text-white font-bold'
                      : 'bg-cream-200 dark:bg-sand-dark text-charcoal dark:text-cream-300 hover:bg-cream-300 dark:hover:bg-[#2A2925]'
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>

            {/* Stage-by-Stage Real Progress Bar */}
            {uploading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="w-full max-w-sm space-y-2 mt-3 p-3.5 rounded-xl bg-cream-100/90 dark:bg-[#1A1916] border border-sand-border dark:border-sand-darkBorder"
              >
                <div className="flex items-center justify-between text-xs font-mono font-medium">
                  <span className="text-terracotta-700 dark:text-terracotta-400 flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5 animate-spin" />
                    {currentStage || 'INDEXING'}
                  </span>
                  <span className="font-bold text-charcoal dark:text-cream-100">{uploadProgress}%</span>
                </div>

                <div className="w-full h-2 rounded-full bg-cream-200 dark:bg-sand-dark overflow-hidden">
                  <motion.div
                    className="h-full rounded-full bg-gradient-to-r from-terracotta-500 via-amber-500 to-emerald-500"
                    initial={{ width: 0 }}
                    animate={{ width: `${uploadProgress}%` }}
                    transition={{ duration: 0.35, ease: 'easeOut' }}
                  />
                </div>

                <p className="text-center text-[11px] font-sans text-charcoal-muted dark:text-cream-400 truncate">
                  {stageMessage || 'Processing document ingestion...'}
                </p>
              </motion.div>
            )}
          </div>
        </LiquidGlassCard>

        {/* ── Stats row ───────────────────────────── */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            {
              label: 'Total Documents',
              value: documents.length,
              icon: <FileText className="w-4 h-4 text-terracotta-600" />,
            },
            {
              label: 'Total Chunks',
              value: documents.reduce((sum, d) => sum + d.chunks_count, 0),
              icon: <Layers className="w-4 h-4 text-amber-500" />,
            },
            {
              label: 'Total Size',
              value: formatBytes(documents.reduce((sum, d) => sum + d.file_size, 0)),
              icon: <HardDrive className="w-4 h-4 text-sky-500" />,
            },
            {
              label: 'Ready & Indexed',
              value: documents.filter((d) => d.status === 'indexed' || d.status === 'READY' || d.status === 'READY_WITH_VISION').length,
              icon: <CheckCircle2 className="w-4 h-4 text-emerald-500" />,
            },
          ].map((stat) => (
            <LiquidGlassCard key={stat.label} variant="space" className="p-3">
              <div className="flex items-center gap-2 mb-1">
                {stat.icon}
                <span className="text-[11px] text-charcoal-muted dark:text-cream-400 font-medium">
                  {stat.label}
                </span>
              </div>
              <p className="text-lg font-bold font-mono text-charcoal dark:text-cream-100">{stat.value}</p>
            </LiquidGlassCard>
          ))}
        </div>

        {/* ── Document list ────────────────────────── */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: 'var(--sp-text-faint)' }} />
          <input
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            placeholder="Search by filename, category, type, or content hash…"
            className="sp-field-inner w-full pl-10 pr-4 py-2.5 rounded-xl text-sm outline-none backdrop-blur-md"
          />
        </div>
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <RefreshCw className="w-6 h-6 animate-spin text-charcoal-muted dark:text-cream-400" />
          </div>
        ) : visibleDocuments.length === 0 ? (
          <div className="text-center py-20 text-charcoal-muted dark:text-cream-400 text-sm">
            {documents.length === 0 ? 'No documents found. Upload your first policy document above.' : 'No documents match your search.'}
          </div>
        ) : (
          <div className="space-y-2">
            <AnimatePresence mode="popLayout">
              {visibleDocuments.map((doc, i) => (
                <motion.div
                  key={doc.id}
                  custom={i}
                  variants={listItemVariants}
                  initial="hidden"
                  animate="visible"
                  exit="exit"
                  layout
                >
                  <LiquidGlassCard variant="space" className="p-4 flex items-center gap-4" hoverEffect>
                    {/* File icon */}
                    <div className="w-10 h-10 rounded-xl bg-cream-100 dark:bg-sand-dark flex items-center justify-center shrink-0">
                      {getFileIcon(doc.file_type)}
                    </div>

                    {/* Info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <p className="text-sm font-semibold text-charcoal dark:text-cream-100 truncate">
                          {doc.filename}
                        </p>
                        {getStatusBadge(doc)}
                      </div>

                      <div className="flex items-center gap-3 mt-1 text-[11px] text-charcoal-muted dark:text-cream-400 font-mono">
                        <span>{doc.file_type?.toUpperCase() ?? 'FILE'}</span>
                        <span>•</span>
                        <span>{doc.chunks_count} chunks</span>
                        {Boolean(doc.pages_count) && <><span>•</span><span>{doc.pages_count} pages</span></>}
                        <span>•</span>
                        <span>{formatBytes(doc.file_size)}</span>
                        <span className="hidden sm:inline">•</span>
                        <span className="hidden sm:inline">{formatDate(doc.uploaded_at)}</span>
                      </div>
                      {doc.file_hash && (
                        <p className="mt-1 text-[9.5px] font-mono text-charcoal-muted/70 dark:text-cream-400/70" title={doc.file_hash}>
                          SHA-256 {doc.file_hash.slice(0, 12)}…
                        </p>
                      )}
                    </div>

                    {/* Category pill */}
                    <span className="hidden md:inline-flex px-2.5 py-0.5 rounded-full text-[10px] font-mono bg-terracotta-500/10 text-terracotta-700 dark:text-terracotta-400 border border-terracotta-500/20 shrink-0">
                      {doc.category}
                    </span>

                    {/* Action buttons */}
                    <div className="flex items-center gap-1 shrink-0">
                      {doc.status?.toLowerCase() === 'failed' && (
                        <button
                          onClick={() => handleRetry(doc.id)}
                          disabled={retryingId === doc.id}
                          className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-amber-500/10 text-amber-700 dark:text-amber-400 border border-amber-500/20 text-xs hover:bg-amber-500/20 transition-colors disabled:opacity-50"
                          title="Retry indexing from stored file"
                        >
                          <RotateCcw className={`w-3 h-3 ${retryingId === doc.id ? 'animate-spin' : ''}`} />
                          Retry
                        </button>
                      )}

                      <button
                        onClick={() => deleteDocument(doc.id)}
                        className="p-2 rounded-lg hover:bg-rose-500/10 text-charcoal-muted dark:text-cream-400 hover:text-rose-600 transition-colors shrink-0"
                        title="Delete document"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </LiquidGlassCard>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>
    </div>
  );
}
