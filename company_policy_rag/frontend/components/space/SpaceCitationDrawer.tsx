'use client';

/** Space-styled grounding-sources drawer — slides in from the right, glass panel
 *  matching the sidebar recipe. Shows the full retrieved chunk for a citation. */

import { X, FileText, Hash } from 'lucide-react';
import type { Citation } from '../../lib/types';
import { formatScore } from '../../lib/utils';

interface SpaceCitationDrawerProps {
  isOpen: boolean;
  citation: Citation | null;
  onClose: () => void;
}

export function SpaceCitationDrawer({ isOpen, citation, onClose }: SpaceCitationDrawerProps) {
  const pg =
    citation?.display_page_number ??
    citation?.page_label ??
    citation?.page ??
    citation?.page_number ??
    citation?.physical_page_number;
  const score = citation?.relevance_score ?? citation?.score;

  return (
    <>
      {/* scrim */}
      <div
        onClick={onClose}
        className={`fixed inset-0 z-[60] bg-[#04060c]/40 backdrop-blur-[2px] transition-opacity duration-300 ${
          isOpen ? 'opacity-100' : 'pointer-events-none opacity-0'
        }`}
        aria-hidden="true"
      />
      <aside
        role="dialog"
        aria-label="Grounding source"
        className={`sp-side sp-text fixed right-0 top-0 z-[61] flex h-[100dvh] w-[min(460px,92vw)] flex-col rounded-l-[26px] transition-transform duration-[420ms] [transition-timing-function:cubic-bezier(0.22,0.61,0.36,1)] ${
          isOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        <header className="flex items-start justify-between gap-3 border-b border-[var(--sp-hairline)] px-5 py-4">
          <div className="min-w-0">
            <p className="sp-mono sp-faint mb-1 text-[9.5px] uppercase tracking-[0.3em]">Grounding source</p>
            <h3 className="flex items-center gap-2 truncate text-[15px] font-semibold">
              <FileText className="h-4 w-4 flex-none opacity-70" />
              <span className="truncate">
                {citation?.document_name || citation?.source || citation?.title || 'Source'}
              </span>
            </h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="sp-ibtn flex h-8 w-8 flex-none items-center justify-center rounded-[11px]"
          >
            <X className="h-4 w-4" />
          </button>
        </header>

        <div className="sp-scroll flex-1 overflow-y-auto px-5 py-4">
          <div className="mb-4 flex flex-wrap gap-2">
            {pg !== undefined && pg !== null && pg !== '' && (
              <span className="sp-chip sp-mono flex items-center gap-1 rounded-full px-2.5 py-1 text-[10px]">
                <Hash className="h-3 w-3 opacity-60" /> page {pg}
              </span>
            )}
            {typeof score === 'number' && (
              <span className="sp-scope sp-mono rounded-full px-2.5 py-1 text-[10px] uppercase tracking-[0.14em]">
                relevance {formatScore(score)}
              </span>
            )}
            {citation?.category && (
              <span className="sp-chip rounded-full px-2.5 py-1 text-[11px]">{citation.category}</span>
            )}
          </div>

          {(citation?.section_title || citation?.heading) && (
            <p className="sp-muted mb-2 text-[12px] font-medium">
              {citation.section_title || citation.heading}
            </p>
          )}

          <p className="sp-muted whitespace-pre-wrap text-[13.5px] leading-relaxed">
            {citation?.chunk_text || citation?.snippet || 'No source text available.'}
          </p>
        </div>
      </aside>
    </>
  );
}

export default SpaceCitationDrawer;
