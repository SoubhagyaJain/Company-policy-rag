import { useState, useCallback, useMemo } from 'react';
import {
  ThinkingEvent,
  ThinkingDetailLevel,
  ReasoningSummary,
  ThinkingStage,
  filterEventsByDetailLevel,
} from '../types/thinking';

export interface UseThinkingStreamOptions {
  initialDetailLevel?: ThinkingDetailLevel;
  initialEvents?: ThinkingEvent[];
  initialReasoningSummary?: ReasoningSummary | null;
  autoExpandDuringStream?: boolean;
  autoCollapseOnDone?: boolean;
}

export interface UseThinkingStreamReturn {
  events: ThinkingEvent[];
  visibleEvents: ThinkingEvent[];
  activeStage: ThinkingStage | null;
  isExpanded: boolean;
  isStreaming: boolean;
  detailLevel: ThinkingDetailLevel;
  reasoningSummary: ReasoningSummary | null;
  totalDurationMs: number;
  hasDegradedStages: boolean;
  setDetailLevel: (level: ThinkingDetailLevel) => void;
  setIsExpanded: (expanded: boolean | ((prev: boolean) => boolean)) => void;
  toggleExpanded: () => void;
  handleThinkingEvent: (event: ThinkingEvent) => void;
  handleStreamStart: () => void;
  handleStreamDone: (summary?: ReasoningSummary, finalEvents?: ThinkingEvent[]) => void;
  reset: () => void;
}

export function useThinkingStream(
  options: UseThinkingStreamOptions = {}
): UseThinkingStreamReturn {
  const {
    initialDetailLevel = 'standard',
    initialEvents = [],
    initialReasoningSummary = null,
    autoExpandDuringStream = true,
    autoCollapseOnDone = true,
  } = options;

  const [events, setEvents] = useState<ThinkingEvent[]>(initialEvents);
  const [isExpanded, setIsExpanded] = useState<boolean>(initialEvents.length > 0 ? false : autoExpandDuringStream);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [detailLevel, setDetailLevel] = useState<ThinkingDetailLevel>(initialDetailLevel);
  const [reasoningSummary, setReasoningSummary] = useState<ReasoningSummary | null>(initialReasoningSummary);

  const handleStreamStart = useCallback(() => {
    setEvents([]);
    setIsStreaming(true);
    if (autoExpandDuringStream) {
      setIsExpanded(true);
    }
    setReasoningSummary(null);
  }, [autoExpandDuringStream]);

  const handleThinkingEvent = useCallback((event: ThinkingEvent) => {
    setEvents((prev) => {
      // 1. Exact match by event ID
      const idIdx = prev.findIndex((e) => e.id === event.id);
      if (idIdx >= 0) {
        const next = [...prev];
        next[idIdx] = event;
        return next;
      }

      // 2. Stage transition match
      const stageIdx = prev.findIndex((e) => e.stage === event.stage);
      if (stageIdx >= 0 && (event.status === 'completed' || event.status === 'warning' || event.status === 'failed')) {
        const next = [...prev];
        next[stageIdx] = event;
        return next;
      }

      // 3. New stage event
      return [...prev, event];
    });
  }, []);

  const handleStreamDone = useCallback(
    (summary?: ReasoningSummary, finalEvents?: ThinkingEvent[]) => {
      setIsStreaming(false);
      if (autoCollapseOnDone) {
        setIsExpanded(false);
      }
      if (summary) {
        setReasoningSummary(summary);
      }
      if (finalEvents && finalEvents.length > 0) {
        setEvents(finalEvents);
      }
    },
    [autoCollapseOnDone]
  );

  const reset = useCallback(() => {
    setEvents([]);
    setIsStreaming(false);
    setIsExpanded(false);
    setReasoningSummary(null);
  }, []);

  const toggleExpanded = useCallback(() => {
    setIsExpanded((prev) => !prev);
  }, []);

  const activeStage = useMemo<ThinkingStage | null>(() => {
    const running = events.find((e) => e.status === 'running');
    if (running) return running.stage;
    if (isStreaming && events.length > 0) {
      return events[events.length - 1].stage;
    }
    return null;
  }, [events, isStreaming]);

  const totalDurationMs = useMemo(() => {
    if (reasoningSummary?.total_duration_ms && reasoningSummary.total_duration_ms > 0) {
      return reasoningSummary.total_duration_ms;
    }
    return events.reduce((acc, ev) => acc + (ev.duration_ms || 0), 0);
  }, [events, reasoningSummary]);

  const hasDegradedStages = useMemo(() => {
    return (
      events.some((e) => e.status === 'warning' || e.stage === 'degraded') ||
      Boolean(reasoningSummary?.degraded_stages && reasoningSummary.degraded_stages.length > 0)
    );
  }, [events, reasoningSummary]);

  const visibleEvents = useMemo(() => {
    return filterEventsByDetailLevel(events, detailLevel);
  }, [events, detailLevel]);

  return {
    events,
    visibleEvents,
    activeStage,
    isExpanded,
    isStreaming,
    detailLevel,
    reasoningSummary,
    totalDurationMs,
    hasDegradedStages,
    setDetailLevel,
    setIsExpanded,
    toggleExpanded,
    handleThinkingEvent,
    handleStreamStart,
    handleStreamDone,
    reset,
  };
}
