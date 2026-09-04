'use client';

import { FileText, Link2, ShieldCheck } from 'lucide-react';

import Earth from '@/components/ui/earth';

/**
 * The chat welcome visual: a connected Earth represents policy sources being
 * retrieved across the knowledge base and resolved into grounded answers.
 */
export function PolicyKnowledgeScene() {
  return (
    <div className="policy-graph" aria-hidden="true">
      <div className="policy-graph__earth">
        <Earth />
      </div>
      <div className="policy-graph__grid" />

      <div className="policy-graph__label policy-graph__label--documents">
        <FileText className="h-3 w-3" /> policy sources
      </div>
      <div className="policy-graph__label policy-graph__label--retrieval">
        <Link2 className="h-3 w-3" /> hybrid retrieval
      </div>
      <div className="policy-graph__label policy-graph__label--answer">
        <ShieldCheck className="h-3 w-3" /> grounded answer
      </div>

      <div className="policy-graph__caption">
        <span className="policy-graph__live-dot" />
        Global knowledge map · drag to explore
      </div>
    </div>
  );
}
