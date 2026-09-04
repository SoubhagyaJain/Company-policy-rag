'use client';

/** SpaceComposer — a premium, compact floating command bar. One slim glass row:
 *  a grounded dot, an autosize input, a tight cluster of pill controls (answer
 *  depth · model · knowledge-base filter) that open upward, and a circular accent
 *  send. Active document/category scope shows as chips above the bar. All behavior
 *  (send, filters, model switch, depth, stop) comes from useComposerControls. */

import { useEffect, useRef, useState } from 'react';
import {
  ChevronDown, SlidersHorizontal, Search, X, Square, ArrowUp, Loader2,
  FileText, Folder, Gauge, Check,
} from 'lucide-react';
import type { FilterOptions, ResponseMode } from '../../lib/types';
import type { ComposerControls } from '../../hooks/useComposerControls';

const MAX_CHARS = 4000;

const DEPTHS: Array<{ label: string; mode: ResponseMode; hint: string }> = [
  { label: 'Compact', mode: 'compact', hint: 'Quick answer' },
  { label: 'Standard', mode: 'standard', hint: 'Balanced' },
  { label: 'Detailed', mode: 'detailed', hint: 'Deep, with evidence' },
];

interface SpaceComposerProps {
  controls: ComposerControls;
  isStreaming: boolean;
  onSend: (content: string, filters: FilterOptions | undefined, model: string, responseMode: ResponseMode) => void;
  onCancel: () => void;
}

export function SpaceComposer({ controls, isStreaming, onSend, onCancel }: SpaceComposerProps) {
  const {
    modelsList, selectedModel, selectedModelLabel, pendingModel, modelSwitchError, selectModel,
    responseMode, setResponseMode,
    filteredDocs, filteredCategories, filterSearch, setFilterSearch,
    selectedDocId, setSelectedDocId, selectedCategory, setSelectedCategory,
    selectedDocument, isFilterActive, clearFilters, buildFilters, loadDocuments,
  } = controls;

  const [input, setInput] = useState('');
  const [open, setOpen] = useState<null | 'filters' | 'models' | 'depth'>(null);
  const [filterTab, setFilterTab] = useState<'documents' | 'categories'>('documents');
  const [focused, setFocused] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const barRef = useRef<HTMLDivElement>(null);

  const depthLabel = DEPTHS.find((d) => d.mode === responseMode)?.label ?? 'Standard';

  useEffect(() => {
    if (open === 'filters') loadDocuments();
  }, [open, loadDocuments]);

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (barRef.current && !barRef.current.contains(e.target as Node)) setOpen(null);
    };
    const onEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(null);
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onEsc);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onEsc);
    };
  }, []);

  // Grow the textarea to fit its content, capped at a generous slice of the
  // viewport so long input stays fully visible; only scroll once past that cap.
  const autosize = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    const cap = Math.max(120, Math.min(Math.round(window.innerHeight * 0.45), 520));
    el.style.height = Math.min(el.scrollHeight, cap) + 'px';
    el.style.overflowY = el.scrollHeight > cap ? 'auto' : 'hidden';
  };

  // Keep height in sync with the value on mount, external clears, and resize.
  useEffect(() => {
    autosize();
    const onResize = () => autosize();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input]);

  const submit = () => {
    const content = input.trim();
    if (!content || isStreaming) return;
    onSend(content, buildFilters(), selectedModel, responseMode);
    setInput('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const canSend = input.trim().length > 0 && !isStreaming;
  const nearLimit = input.length > MAX_CHARS * 0.8;
  const toggle = (which: 'filters' | 'models' | 'depth') => setOpen((cur) => (cur === which ? null : which));

  return (
    <div ref={barRef} className="w-full max-w-[720px]">
      {/* Active scope chips */}
      {isFilterActive && (
        <div className="mb-2 flex flex-wrap items-center gap-1.5 px-1">
          <span className="sp-scope sp-mono flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[9px] uppercase tracking-[0.16em]">
            <span className="h-1.5 w-1.5 rounded-full" style={{ background: 'var(--sp-accent)', boxShadow: '0 0 10px rgba(127,227,176,0.9)' }} />
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

      <form
        onSubmit={(e) => { e.preventDefault(); submit(); }}
        className="sp-comp pointer-events-auto relative flex items-end gap-2 rounded-[20px] py-2 pl-3.5 pr-2"
      >
        {/* grounded dot */}
        <span
          className="mb-2 h-2 w-2 flex-none rounded-full"
          style={{ background: 'var(--sp-accent)', boxShadow: '0 0 10px rgba(127,227,176,0.85)' }}
          aria-hidden="true"
        />

        {/* input */}
        <textarea
          ref={textareaRef}
          rows={1}
          value={input}
          onChange={(e) => { setInput(e.target.value.slice(0, MAX_CHARS)); autosize(); }}
          onKeyDown={onKeyDown}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder="Ask across your grounded corpus…"
          className="sp-text sp-scroll max-h-[45vh] min-h-[24px] w-full flex-1 resize-none self-center bg-transparent py-1 text-[14.5px] leading-relaxed outline-none placeholder:opacity-50"
        />

        {/* counter — only when relevant */}
        {(nearLimit || (focused && input.length > 0)) && (
          <span className={`sp-mono mb-2 flex-none text-[9px] tracking-[0.1em] ${nearLimit ? 'text-amber-300' : 'sp-faint'}`}>
            {input.length}/{MAX_CHARS}
          </span>
        )}

        {/* control cluster */}
        <div className="mb-0.5 flex flex-none items-center gap-1">
          {/* Answer depth */}
          <div className="relative">
            <button
              type="button"
              onClick={() => toggle('depth')}
              className="sp-ibtn sp-mono flex h-8 items-center gap-1 rounded-[11px] px-2 text-[10px] uppercase tracking-[0.1em]"
              title="Answer depth"
            >
              <Gauge className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">{depthLabel}</span>
              <ChevronDown className="h-3 w-3 opacity-60" />
            </button>
            {open === 'depth' && (
              <div className="sp-pop absolute bottom-full right-0 z-20 mb-2.5 flex w-[188px] flex-col gap-0.5 rounded-2xl p-2" role="listbox" aria-label="Answer depth">
                {DEPTHS.map((d) => (
                  <button
                    key={d.mode}
                    type="button"
                    onClick={() => { setResponseMode(d.mode); setOpen(null); }}
                    className={`flex items-center justify-between gap-2 rounded-xl px-3 py-2 text-left ${responseMode === d.mode ? 'sp-item-active' : 'sp-item'}`}
                  >
                    <span className="min-w-0">
                      <span className="sp-text block text-[12.5px] font-medium">{d.label}</span>
                      <span className="sp-faint block text-[10px]">{d.hint}</span>
                    </span>
                    {responseMode === d.mode && <Check className="h-3.5 w-3.5 flex-none text-[var(--sp-accent-text)]" />}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Model */}
          <div className="relative">
            <button
              type="button"
              onClick={() => toggle('models')}
              className="sp-ibtn flex h-8 items-center gap-1.5 rounded-[11px] px-2.5 text-[11.5px] font-medium"
              title="Model"
            >
              {pendingModel && <Loader2 className="h-3 w-3 animate-spin" />}
              <span className="max-w-[92px] truncate">{selectedModelLabel}</span>
              <ChevronDown className="h-3 w-3 opacity-60" />
            </button>
            {open === 'models' && (
              <div className="sp-pop sp-scroll absolute bottom-full right-0 z-20 mb-2.5 flex max-h-[300px] w-[268px] flex-col gap-0.5 overflow-y-auto rounded-2xl p-2" role="listbox" aria-label="Model">
                {modelsList.map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => { selectModel(m.id); setOpen(null); }}
                    className={`flex flex-col gap-0.5 rounded-xl px-3 py-2 text-left ${selectedModel === m.id ? 'sp-item-active' : 'sp-item'}`}
                  >
                    <span className="sp-text text-[12.5px] font-medium">{m.label}</span>
                    <span className="sp-faint sp-mono text-[9.5px]">{m.desc}</span>
                  </button>
                ))}
                {modelSwitchError && <p className="sp-mono px-3 py-1.5 text-[10px] text-amber-400">{modelSwitchError}</p>}
              </div>
            )}
          </div>

          {/* Filter */}
          <div className="relative">
            <button
              type="button"
              onClick={() => toggle('filters')}
              className="sp-ibtn relative flex h-8 w-8 items-center justify-center rounded-[11px]"
              title="Filter knowledge base"
              aria-label="Filter knowledge base"
            >
              <SlidersHorizontal className="h-3.5 w-3.5" />
              {isFilterActive && (
                <span className="absolute -right-0.5 -top-0.5 h-2 w-2 rounded-full" style={{ background: 'var(--sp-accent)', boxShadow: '0 0 8px rgba(127,227,176,0.9)' }} />
              )}
            </button>
            {open === 'filters' && (
              <div className="sp-pop sp-text absolute bottom-full right-0 z-20 mb-2.5 flex w-[340px] flex-col gap-3 rounded-[18px] p-3.5">
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
                          onClick={() => { setSelectedDocId(d.id); setOpen(null); }}
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
                          onClick={() => { setSelectedCategory(c); setOpen(null); }}
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

          {/* divider */}
          <span className="mx-0.5 h-5 w-px" style={{ background: 'var(--sp-hairline)' }} aria-hidden="true" />

          {/* Send / Stop */}
          {isStreaming ? (
            <button
              type="button"
              onClick={onCancel}
              className="sp-ibtn flex h-9 w-9 items-center justify-center rounded-full"
              aria-label="Stop"
              title="Stop"
            >
              <Square className="h-3.5 w-3.5" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!canSend}
              className="sp-send-fab flex h-9 w-9 items-center justify-center rounded-full transition-all"
              aria-label="Send"
              title="Send"
              style={{ opacity: canSend ? 1 : 0.4, transform: canSend ? 'scale(1)' : 'scale(0.94)' }}
            >
              <ArrowUp className="h-4 w-4" />
            </button>
          )}
        </div>
      </form>
    </div>
  );
}

export default SpaceComposer;
