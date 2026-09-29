'use client';

/** SpaceComposer — the chat input, built on the morphing PromptInput
 *  (components/ui/ai-chat-input): a slim pill that springs open into a card with
 *  the model picker, an answer-depth cycler, the knowledge-base filter, voice
 *  dictation and send / stop. Active document/category scope shows as chips above
 *  it. All behavior (send, filters, model switch, depth, stop) comes from
 *  useComposerControls. */

import { memo, useEffect, useMemo, useRef, useState } from 'react';
import { SlidersHorizontal, Search, X, FileText, Folder } from 'lucide-react';
import { PromptInput, type PromptEffortOption, type PromptModelOption } from '@/components/ui/ai-chat-input';
import { cn } from '@/lib/utils';
import type { FilterOptions, ResponseMode } from '../../lib/types';
import type { ComposerControls } from '../../hooks/useComposerControls';

const MAX_CHARS = 4000;

const DEPTHS: PromptEffortOption[] = [
  { value: 'compact', label: 'Compact', hint: 'Answer depth: quick answer' },
  { value: 'standard', label: 'Standard', hint: 'Answer depth: balanced' },
  { value: 'detailed', label: 'Detailed', hint: 'Answer depth: deep, with evidence' },
];

interface SpaceComposerProps {
  controls: ComposerControls;
  isStreaming: boolean;
  onSend: (content: string, filters: FilterOptions | undefined, model: string, responseMode: ResponseMode) => void;
  onCancel: () => void;
}

export const SpaceComposer = memo(function SpaceComposer({ controls, isStreaming, onSend, onCancel }: SpaceComposerProps) {
  const {
    modelsList, selectedModel, pendingModel, modelSwitchError, selectModel,
    responseMode, setResponseMode,
    filteredDocs, filteredCategories, filterSearch, setFilterSearch,
    selectedDocId, setSelectedDocId, selectedCategory, setSelectedCategory,
    selectedDocument, isFilterActive, clearFilters, buildFilters, loadDocuments,
  } = controls;

  const [filtersOpen, setFiltersOpen] = useState(false);
  const [filterTab, setFilterTab] = useState<'documents' | 'categories'>('documents');
  const filterRef = useRef<HTMLDivElement>(null);

  const models = useMemo<PromptModelOption[]>(
    () => modelsList.map((m) => ({ id: m.id, label: m.label, description: m.desc })),
    [modelsList],
  );

  useEffect(() => {
    if (filtersOpen) loadDocuments();
  }, [filtersOpen, loadDocuments]);

  useEffect(() => {
    if (!filtersOpen) return;
    const onDown = (e: MouseEvent) => {
      if (filterRef.current && !filterRef.current.contains(e.target as Node)) setFiltersOpen(false);
    };
    const onEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setFiltersOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onEsc);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onEsc);
    };
  }, [filtersOpen]);

  const submit = (value: string) => {
    const content = value.trim();
    if (!content || isStreaming) return;
    setFiltersOpen(false);
    onSend(content, buildFilters(), selectedModel, responseMode);
  };

  const filterControl = (
    <div ref={filterRef} className="relative">
      <button
        type="button"
        onMouseDown={(e) => e.preventDefault()}
        onClick={() => setFiltersOpen((o) => !o)}
        className={cn(
          'relative flex size-7 items-center justify-center rounded-full text-foreground/50 outline-none transition-all duration-200 hover:bg-accent/60 hover:text-foreground',
          (filtersOpen || isFilterActive) && 'bg-accent/60 text-foreground',
        )}
        title="Filter knowledge base"
        aria-label="Filter knowledge base"
        aria-expanded={filtersOpen}
      >
        <SlidersHorizontal className="size-3.5" />
        {isFilterActive && <span className="sp-dot absolute -right-0.5 -top-0.5 h-2 w-2" />}
      </button>
      {filtersOpen && (
        <div className="sp-pop sp-text absolute bottom-full right-0 z-50 mb-3 flex w-[340px] cursor-default flex-col gap-3 rounded-[18px] p-3.5">
          <div className="flex items-center justify-between">
            <span className="sp-text flex items-center gap-1.5 text-[12px] font-semibold">
              <SlidersHorizontal className="h-3.5 w-3.5" /> Filter Knowledge Base
            </span>
            {isFilterActive && (
              <button type="button" onClick={clearFilters} className="sp-mono text-[10px] uppercase tracking-[0.12em] text-[var(--sp-accent-text)]">
                Reset
              </button>
            )}
          </div>

          <div className="relative">
            <Search className="pointer-events-none absolute left-2.5 top-2.5 h-3.5 w-3.5 opacity-50" />
            <input
              value={filterSearch}
              onChange={(e) => setFilterSearch(e.target.value)}
              placeholder="Search documents or categories…"
              className="sp-field-inner w-full rounded-xl py-1.5 pl-8 pr-3 text-[12px] outline-none"
            />
          </div>

          <div className="sp-depth flex items-center gap-1 rounded-xl p-1 text-[11px]">
            {(['documents', 'categories'] as const).map((tab) => (
              <button
                key={tab}
                type="button"
                aria-checked={filterTab === tab}
                onClick={() => setFilterTab(tab)}
                className="sp-depth-btn flex-1 rounded-lg py-1 capitalize"
              >
                {tab} ({tab === 'documents' ? filteredDocs.length : filteredCategories.length})
              </button>
            ))}
          </div>

          <div className="sp-scroll flex max-h-56 flex-col gap-0.5 overflow-y-auto">
            <button
              type="button"
              onClick={() => (filterTab === 'documents' ? setSelectedDocId('All') : setSelectedCategory('All'))}
              className={`flex items-center gap-2 rounded-lg px-2.5 py-2 text-left text-[12px] ${
                (filterTab === 'documents' ? selectedDocId === 'All' : selectedCategory === 'All') ? 'sp-item-active' : 'sp-item'
              }`}
            >
              All {filterTab === 'documents' ? 'documents' : 'categories'}
            </button>
            {filterTab === 'documents'
              ? filteredDocs.map((d) => (
                  <button
                    key={d.id}
                    type="button"
                    onClick={() => { setSelectedDocId(d.id); setFiltersOpen(false); }}
                    className={`flex items-center gap-2 rounded-lg px-2.5 py-2 text-left text-[12px] ${selectedDocId === d.id ? 'sp-item-active' : 'sp-item'}`}
                  >
                    <FileText className="h-3.5 w-3.5 flex-none opacity-60" />
                    <span className="truncate">{d.filename}</span>
                  </button>
                ))
              : filteredCategories.map((c) => (
                  <button
                    key={c}
                    type="button"
                    onClick={() => { setSelectedCategory(c); setFiltersOpen(false); }}
                    className={`flex items-center gap-2 rounded-lg px-2.5 py-2 text-left text-[12px] ${selectedCategory === c ? 'sp-item-active' : 'sp-item'}`}
                  >
                    <Folder className="h-3.5 w-3.5 flex-none opacity-60" />
                    <span className="truncate">{c}</span>
                  </button>
                ))}
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className="flex w-full flex-col items-center">
      {/* Active scope chips */}
      {isFilterActive && (
        <div className="mb-2 flex w-full max-w-[720px] flex-wrap items-center justify-center gap-1.5 px-1">
          <span className="sp-scope sp-mono flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[9px] uppercase tracking-[0.16em]">
            <span className="sp-dot h-1.5 w-1.5" />
            Grounded
          </span>
          {selectedDocument && (
            <span className="sp-chip flex items-center gap-1 rounded-full px-2.5 py-1 text-[11px]">
              {selectedDocument.filename}
              <button type="button" onClick={() => setSelectedDocId('All')} aria-label="Remove document filter">
                <X className="h-3 w-3 opacity-60" />
              </button>
            </span>
          )}
          {selectedCategory !== 'All' && (
            <span className="sp-chip flex items-center gap-1 rounded-full px-2.5 py-1 text-[11px]">
              {selectedCategory}
              <button type="button" onClick={() => setSelectedCategory('All')} aria-label="Remove category filter">
                <X className="h-3 w-3 opacity-60" />
              </button>
            </span>
          )}
        </div>
      )}

      <PromptInput
        className="pointer-events-auto"
        cardClassName="bg-card/90 backdrop-blur-xl backdrop-saturate-150 shadow-[0_26px_60px_-30px_rgba(64,42,30,0.5)] dark:shadow-[0_28px_64px_-26px_rgba(0,3,12,0.9)]"
        placeholder="Ask across your grounded corpus…"
        collapsedWidth={460}
        expandedWidth={720}
        maxTextareaHeight={280}
        maxLength={MAX_CHARS}
        maxAttachments={0}
        models={models}
        model={selectedModel}
        onModelChange={selectModel}
        modelLoading={pendingModel !== null}
        modelError={modelSwitchError}
        efforts={DEPTHS}
        effort={responseMode}
        onEffortChange={(v) => setResponseMode(v as ResponseMode)}
        isStreaming={isStreaming}
        onStop={onCancel}
        onSubmit={submit}
        toolbarEnd={filterControl}
      />
    </div>
  );
}, composerPropsEqual);

/** The controls object is rebuilt on every parent render, so compare its
 *  fields (all state values or stable callbacks) instead of its identity. */
function composerPropsEqual(a: SpaceComposerProps, b: SpaceComposerProps) {
  if (a.isStreaming !== b.isStreaming || a.onSend !== b.onSend || a.onCancel !== b.onCancel) return false;
  const keys = Object.keys(a.controls) as Array<keyof ComposerControls>;
  return keys.every((k) => a.controls[k] === b.controls[k]);
}

export default SpaceComposer;
