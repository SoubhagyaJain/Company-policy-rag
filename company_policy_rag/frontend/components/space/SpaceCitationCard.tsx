'use client';

/** Space-styled citation chip. Opens the grounding drawer on click. */

import { FileText } from 'lucide-react';
import type { Citation } from '../../lib/types';
import { formatScore } from '../../lib/utils';

interface SpaceCitationCardProps {
  citation: Citation;
  index: number;
  onOpen: (citation: Citation) => void;
}

function pageLabel(c: Citation): string | null {
  const p =
    c.display_page_number ??
    c.page_label ??
    c.page ??
    c.page_number ??
    c.physical_page_number;
  return p !== undefined && p !== null && p !== '' ? `p.${p}` : null;
}

export function SpaceCitationCard({ citation, index, onOpen }: SpaceCitationCardProps) {
  const label = citation.document_name || citation.source || citation.title || 'Source';
  const pg = pageLabel(citation);
  const score = citation.relevance_score ?? citation.score;

  return (
    <button
      type="button"
      onClick={() => onOpen(citation)}
      className="sp-chip group flex items-center gap-2 rounded-full px-3 py-1.5 text-left transition-all hover:-translate-y-px"
      title={`Open source · ${label}${pg ? ` · ${pg}` : ''}`}
    >
      <span className="sp-mono flex h-4 min-w-4 items-center justify-center rounded-full bg-[var(--sp-accent-bg)] px-1 text-[9px] font-medium text-[var(--sp-accent-text)]">
        {citation.source_index ?? index + 1}
      </span>
      <FileText className="h-3 w-3 opacity-70" />
      <span className="max-w-[180px] truncate text-[11.5px] font-medium">{label}</span>
      {pg && <span className="sp-mono text-[9.5px] opacity-60">{pg}</span>}
      {typeof score === 'number' && (
        <span className="sp-mono text-[9.5px] opacity-50">{formatScore(score)}</span>
      )}
    </button>
  );
}

export default SpaceCitationCard;
