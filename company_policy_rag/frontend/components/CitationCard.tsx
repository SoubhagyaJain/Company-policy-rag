'use client';

import React from 'react';
import { BookOpen, ExternalLink, Hash, Award } from 'lucide-react';
import { Citation } from '../lib/types';
import { formatScore, cn } from '../lib/utils';

interface CitationCardProps {
  citation: Citation;
  index: number;
  onClick: (citation: Citation) => void;
  compact?: boolean;
}

export function CitationCard({
  citation,
  index,
  onClick,
  compact = false,
}: CitationCardProps) {
  const getScoreBadgeColor = (score: number) => {
    if (score >= 0.85) return 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/30';
    if (score >= 0.7) return 'bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-500/30';
    return 'bg-cream-200 text-charcoal-muted border-sand-border';
  };

  if (compact) {
    return (
      <button
        onClick={() => onClick(citation)}
        title={`View Source: ${citation.title}`}
        className="inline-flex items-center gap-1 px-2 py-0.5 mx-0.5 rounded-md bg-terracotta-500/10 dark:bg-terracotta-500/20 text-terracotta-700 dark:text-terracotta-400 border border-terracotta-500/30 text-[11px] font-mono hover:bg-terracotta-500/20 transition-colors"
      >
        <BookOpen className="w-3 h-3 shrink-0" />
        <span className="font-semibold">[{index + 1}]</span>
        <span className="truncate max-w-[120px] hidden sm:inline">{citation.title}</span>
      </button>
    );
  }

  return (
    <div
      onClick={() => onClick(citation)}
      className="group relative p-3 rounded-xl bg-[#FAF9F5]/90 dark:bg-[#1E1D1A]/90 border border-[#E5E0D8] dark:border-[#2E2C27] hover:border-terracotta-500/50 cursor-pointer transition-all shadow-sm hover:shadow-md"
    >
      <div className="flex items-start justify-between gap-2 mb-1.5">
        <div className="flex items-center gap-2 min-w-0">
          <span className="w-5 h-5 rounded-md bg-terracotta-600/10 text-terracotta-700 dark:text-terracotta-400 font-mono text-[11px] font-bold flex items-center justify-center shrink-0">
            {index + 1}
          </span>
          <h4 className="text-xs font-semibold text-charcoal dark:text-cream-100 truncate group-hover:text-terracotta-600 transition-colors">
            {citation.title}
          </h4>
        </div>

        {citation.score !== undefined && (
          <span
            className={cn(
              'px-1.5 py-0.5 rounded text-[10px] font-mono font-medium border shrink-0',
              getScoreBadgeColor(citation.score)
            )}
          >
            {formatScore(citation.score)} match
          </span>
        )}
      </div>

      <p className="text-[11px] text-charcoal-muted dark:text-cream-400 line-clamp-2 leading-relaxed mb-2 font-sans italic">
        "{citation.chunk_text}"
      </p>

      <div className="flex items-center justify-between text-[10px] text-charcoal-muted dark:text-cream-500 border-t border-sand-border/40 dark:border-sand-darkBorder/40 pt-1.5">
        <div className="flex items-center gap-2">
          {citation.page !== undefined && (
            <span className="flex items-center gap-0.5 font-mono">
              <Hash className="w-2.5 h-2.5" /> Page {citation.page}
            </span>
          )}
          {citation.heading && (
            <span className="truncate max-w-[140px] font-medium text-charcoal/80 dark:text-cream-300">
              § {citation.heading}
            </span>
          )}
        </div>

        <span className="flex items-center gap-1 text-terracotta-600 dark:text-terracotta-400 font-medium group-hover:underline">
          View Detail <ExternalLink className="w-2.5 h-2.5" />
        </span>
      </div>
    </div>
  );
}
