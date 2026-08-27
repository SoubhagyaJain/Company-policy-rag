/**
 * Aether Thinking & Reasoning Data Types.
 *
 * Mirror definitions for Aether React application.
 */

export type ThinkingStage =
  | 'received'
  | 'conversation_context'
  | 'follow_up_resolution'
  | 'query_analysis'
  | 'query_rewrite'
  | 'retrieval'
  | 'reranking'
  | 'evidence_analysis'
  | 'evidence_reuse'
  | 'page_expansion'
  | 'visual_analysis'
  | 'evidence_verification'
  | 'answer_planning'
  | 'answer_generation'
  | 'citation_building'
  | 'completed'
  | 'degraded';

export type ThinkingStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'skipped'
  | 'warning'
  | 'failed';

export type ThinkingDetailLevel = 'off' | 'compact' | 'standard' | 'detailed';

export interface ThinkingEvent {
  id: string;
  query_id: string;
  stage: ThinkingStage;
  status: ThinkingStatus;
  title: string;
  summary: string;
  details?: Record<string, any>;
  started_at?: string;
  completed_at?: string | null;
  duration_ms?: number;
}

export interface ReasoningSummary {
  intent: string;
  answer_mode: string;
  is_follow_up: boolean;
  used_conversation_context: boolean;
  reused_previous_evidence: boolean;
  retrieved_new_evidence: boolean;
  used_visual_evidence: boolean;
  evidence_status: string;
  sources_used: string[];
  degraded_stages: string[];
  total_duration_ms: number;
}

export interface ThinkingStreamState {
  events: ThinkingEvent[];
  activeStage: ThinkingStage | null;
  isExpanded: boolean;
  detailLevel: ThinkingDetailLevel;
  reasoningSummary: ReasoningSummary | null;
  totalDurationMs: number;
  hasDegradedStages: boolean;
  isStreaming: boolean;
}

export const COMPACT_STAGES: readonly ThinkingStage[] = [
  'received',
  'conversation_context',
  'retrieval',
  'evidence_verification',
  'answer_planning',
  'completed',
  'degraded',
] as const;

export function isCompactStage(stage: ThinkingStage): boolean {
  return (COMPACT_STAGES as readonly string[]).includes(stage);
}

export function filterEventsByDetailLevel(
  events: ThinkingEvent[],
  detailLevel: ThinkingDetailLevel
): ThinkingEvent[] {
  if (detailLevel === 'off') {
    return [];
  }
  if (detailLevel === 'compact') {
    return events.filter((e) => isCompactStage(e.stage));
  }
  return events;
}
