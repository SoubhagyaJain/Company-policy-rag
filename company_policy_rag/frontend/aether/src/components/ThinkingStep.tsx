import React from 'react';
import {
  CheckCircle2,
  AlertTriangle,
  Loader2,
  Brain,
  Search,
  Layers,
  FileCheck,
  Eye,
  FileText,
  Clock,
  Sparkles,
  HelpCircle,
  SkipForward,
} from 'lucide-react';
import { ThinkingEvent, ThinkingDetailLevel } from '../types/thinking';

export interface ThinkingStepProps {
  event: ThinkingEvent;
  detailLevel: ThinkingDetailLevel;
}

export function getStageIcon(stage: string, status: string) {
  if (status === 'running') {
    return <Loader2 className="w-3.5 h-3.5 text-violet-400 animate-spin shrink-0" />;
  }
  if (status === 'warning' || stage === 'degraded') {
    return <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0" />;
  }
  if (status === 'skipped') {
    return <SkipForward className="w-3.5 h-3.5 text-white/30 shrink-0" />;
  }
  if (status === 'completed') {
    return <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />;
  }

  switch (stage) {
    case 'received':
    case 'query_analysis':
    case 'query_rewrite':
      return <Brain className="w-3.5 h-3.5 text-sky-400 shrink-0" />;
    case 'conversation_context':
    case 'follow_up_resolution':
      return <Sparkles className="w-3.5 h-3.5 text-purple-400 shrink-0" />;
    case 'retrieval':
    case 'reranking':
      return <Search className="w-3.5 h-3.5 text-indigo-400 shrink-0" />;
    case 'evidence_analysis':
    case 'evidence_reuse':
    case 'page_expansion':
      return <Layers className="w-3.5 h-3.5 text-amber-400 shrink-0" />;
    case 'visual_analysis':
      return <Eye className="w-3.5 h-3.5 text-teal-400 shrink-0" />;
    case 'evidence_verification':
      return <FileCheck className="w-3.5 h-3.5 text-emerald-400 shrink-0" />;
    case 'answer_planning':
    case 'answer_generation':
    case 'citation_building':
      return <FileText className="w-3.5 h-3.5 text-violet-400 shrink-0" />;
    default:
      return <HelpCircle className="w-3.5 h-3.5 text-white/40 shrink-0" />;
  }
}

function formatDuration(ms?: number): string {
  if (!ms || ms <= 0) return '';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function ThinkingStep({ event, detailLevel }: ThinkingStepProps) {
  const isRunning = event.status === 'running';
  const isDegraded = event.status === 'warning' || event.stage === 'degraded';
  const isSkipped = event.status === 'skipped';
  const showDetails = detailLevel === 'detailed' && event.details && Object.keys(event.details).length > 0;

  return (
    <div
      className={`flex flex-col py-1.5 px-2.5 rounded-lg text-xs transition-colors ${
        isRunning
          ? 'bg-violet-500/10 border border-violet-500/20 text-violet-200'
          : isDegraded
          ? 'bg-amber-500/10 border border-amber-500/20 text-amber-200'
          : isSkipped
          ? 'opacity-50 text-white/40'
          : 'hover:bg-white/5 text-white/80'
      }`}
      role="status"
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          {getStageIcon(event.stage, event.status)}
          <span className="font-medium text-xs truncate">
            {event.title || event.stage.replace(/_/g, ' ')}
          </span>
        </div>

        {event.duration_ms !== undefined && event.duration_ms > 0 && (
          <span className="font-mono text-[10px] text-white/40 shrink-0 flex items-center gap-1">
            <Clock className="w-2.5 h-2.5 opacity-60" />
            {formatDuration(event.duration_ms)}
          </span>
        )}
      </div>

      {event.summary && (
        <p className="text-[11px] text-white/60 mt-0.5 ml-5.5 leading-relaxed">
          {event.summary}
        </p>
      )}

      {showDetails && (
        <div className="mt-1 ml-5.5 flex flex-wrap gap-1.5 pt-1 border-t border-white/10">
          {Object.entries(event.details || {}).map(([key, val]) => {
            if (val === undefined || val === null || val === '') return null;
            return (
              <span
                key={key}
                className="inline-flex items-center gap-1 px-1.5 py-0.2 rounded text-[9.5px] font-mono bg-white/10 text-white/70"
              >
                <span className="opacity-60">{key.replace(/_/g, ' ')}:</span>
                <span className="font-semibold text-white/90">
                  {typeof val === 'object' ? JSON.stringify(val) : String(val)}
                </span>
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}
