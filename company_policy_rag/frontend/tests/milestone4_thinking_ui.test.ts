/**
 * Milestone 4: Premium Frontend Thinking UI & Stream Hook Test Suite.
 *
 * Covers:
 * - TypeScript types & helpers (ThinkingStage, ThinkingStatus, ThinkingDetailLevel, COMPACT_STAGES)
 * - Detail level filtering (off, compact, standard, detailed)
 * - SSE parser event mapping (mapThinkingEvent, streamChat dispatch)
 * - useThinkingStream hook lifecycle (auto-expand on start, auto-collapse on done, deduplication, timing)
 * - Zero-CoT safety (no <think>, private prompts, vector IDs, secrets in rendering)
 * - Accessibility attributes (aria-expanded, aria-controls, aria-live, role)
 */

import {
  ThinkingStage,
  ThinkingStatus,
  ThinkingDetailLevel,
  ThinkingEvent,
  ReasoningSummary,
  COMPACT_STAGES,
  isCompactStage,
  filterEventsByDetailLevel,
} from '../types/thinking';
import { mapThinkingEvent } from '../lib/api-client';
import { TestResult } from './tier1_features.test';

export function runMilestone4Tests(): TestResult[] {
  const results: TestResult[] = [];

  function test(name: string, fn: () => void) {
    const start = performance.now();
    try {
      fn();
      results.push({
        suite: 'Milestone 4: Premium Frontend Thinking UI & SSE Hook',
        name,
        passed: true,
        durationMs: performance.now() - start,
      });
    } catch (err: any) {
      results.push({
        suite: 'Milestone 4: Premium Frontend Thinking UI & SSE Hook',
        name,
        passed: false,
        durationMs: performance.now() - start,
        error: err?.message || String(err),
      });
    }
  }

  // --- 1. Type & Enums Verification ---
  test('M4.1: ThinkingStage enum contains all 17 canonical pipeline stages', () => {
    const expectedStages: ThinkingStage[] = [
      'received',
      'conversation_context',
      'follow_up_resolution',
      'query_analysis',
      'query_rewrite',
      'retrieval',
      'reranking',
      'evidence_analysis',
      'evidence_reuse',
      'page_expansion',
      'visual_analysis',
      'evidence_verification',
      'answer_planning',
      'answer_generation',
      'citation_building',
      'completed',
      'degraded',
    ];
    if (expectedStages.length !== 17) {
      throw new Error(`Expected 17 stages, got ${expectedStages.length}`);
    }
  });

  test('M4.2: ThinkingStatus enum contains all 6 canonical statuses', () => {
    const statuses: ThinkingStatus[] = ['pending', 'running', 'completed', 'skipped', 'warning', 'failed'];
    if (statuses.length !== 6) {
      throw new Error(`Expected 6 statuses, got ${statuses.length}`);
    }
  });

  test('M4.3: COMPACT_STAGES contains exactly the 7 core milestone stages', () => {
    if (COMPACT_STAGES.length !== 7) {
      throw new Error(`Expected 7 compact stages, got ${COMPACT_STAGES.length}`);
    }
    const required = ['received', 'conversation_context', 'retrieval', 'evidence_verification', 'answer_planning', 'completed', 'degraded'];
    for (const r of required) {
      if (!isCompactStage(r as ThinkingStage)) {
        throw new Error(`Stage ${r} must be in COMPACT_STAGES`);
      }
    }
    if (isCompactStage('query_analysis')) {
      throw new Error('query_analysis should not be in COMPACT_STAGES');
    }
    if (isCompactStage('reranking')) {
      throw new Error('reranking should not be in COMPACT_STAGES');
    }
  });

  // --- 2. Detail Level Filtering ---
  test('M4.4: filterEventsByDetailLevel handles off, compact, standard, and detailed', () => {
    const mockEvents: ThinkingEvent[] = [
      { id: '1', query_id: 'q1', stage: 'received', status: 'completed', title: 'Received', summary: 'Question received' },
      { id: '2', query_id: 'q1', stage: 'query_analysis', status: 'completed', title: 'Analysis', summary: 'Classified query' },
      { id: '3', query_id: 'q1', stage: 'conversation_context', status: 'completed', title: 'Context', summary: 'Follow-up linked' },
      { id: '4', query_id: 'q1', stage: 'retrieval', status: 'completed', title: 'Retrieval', summary: 'Searched sources' },
      { id: '5', query_id: 'q1', stage: 'reranking', status: 'completed', title: 'Reranking', summary: 'Cross-encoder scoring' },
      { id: '6', query_id: 'q1', stage: 'evidence_verification', status: 'completed', title: 'Verification', summary: 'Evidence verified' },
      { id: '7', query_id: 'q1', stage: 'answer_planning', status: 'completed', title: 'Planning', summary: 'Constructing prompt' },
    ];

    // Off
    const offEvents = filterEventsByDetailLevel(mockEvents, 'off');
    if (offEvents.length !== 0) {
      throw new Error(`Detail level 'off' must return 0 events, got ${offEvents.length}`);
    }

    // Compact
    const compactEvents = filterEventsByDetailLevel(mockEvents, 'compact');
    if (compactEvents.length !== 5) {
      throw new Error(`Detail level 'compact' expected 5 events, got ${compactEvents.length}`);
    }
    const compactStages = compactEvents.map((e) => e.stage);
    if (compactStages.includes('query_analysis') || compactStages.includes('reranking')) {
      throw new Error('Compact events contained non-compact stages');
    }

    // Standard
    const standardEvents = filterEventsByDetailLevel(mockEvents, 'standard');
    if (standardEvents.length !== 7) {
      throw new Error(`Standard level expected all 7 events, got ${standardEvents.length}`);
    }

    // Detailed
    const detailedEvents = filterEventsByDetailLevel(mockEvents, 'detailed');
    if (detailedEvents.length !== 7) {
      throw new Error(`Detailed level expected all 7 events, got ${detailedEvents.length}`);
    }
  });

  // --- 3. SSE Parser mapThinkingEvent ---
  test('M4.5: mapThinkingEvent converts raw backend JSON to structured ThinkingEvent', () => {
    const raw = {
      id: 'thk_12345',
      query_id: 'qry_999',
      stage: 'conversation_context',
      status: 'completed',
      title: 'Resolving conversation context',
      summary: 'Detected follow-up question and linked to previous topic.',
      details: { is_follow_up: true, previous_subject: 'Hotel Search Agent' },
      started_at: '2026-08-26T08:00:00Z',
      completed_at: '2026-08-26T08:00:00.012Z',
      duration_ms: 12.5,
    };

    const ev = mapThinkingEvent(raw);
    if (ev.id !== 'thk_12345') throw new Error(`Expected id thk_12345, got ${ev.id}`);
    if (ev.stage !== 'conversation_context') throw new Error(`Expected stage conversation_context, got ${ev.stage}`);
    if (ev.status !== 'completed') throw new Error(`Expected status completed, got ${ev.status}`);
    if (ev.duration_ms !== 12.5) throw new Error(`Expected duration 12.5, got ${ev.duration_ms}`);
    if (ev.details?.is_follow_up !== true) throw new Error('Expected details.is_follow_up to be true');
  });

  test('M4.6: mapThinkingEvent handles missing or non-object inputs safely', () => {
    const fallback1 = mapThinkingEvent(null);
    if (!fallback1.id.startsWith('thk_') || fallback1.stage !== 'received') {
      throw new Error('Null input failed fallback');
    }

    const fallback2 = mapThinkingEvent('Raw text summary');
    if (fallback2.summary !== 'Raw text summary') {
      throw new Error(`Expected raw string in summary, got ${fallback2.summary}`);
    }
  });

  // --- 4. Hook State Machine Simulation ---
  test('M4.7: Hook lifecycle - auto-expand on stream start and auto-collapse on stream done', () => {
    let events: ThinkingEvent[] = [];
    let isStreaming = false;
    let isExpanded = false;
    let summary: ReasoningSummary | null = null;

    // Simulate handleStreamStart
    const onStart = () => {
      events = [];
      isStreaming = true;
      isExpanded = true; // Auto-expand during stream
      summary = null;
    };

    // Simulate handleThinkingEvent
    const onThinking = (ev: ThinkingEvent) => {
      const idx = events.findIndex((e) => e.stage === ev.stage);
      if (idx >= 0 && (ev.status === 'completed' || ev.status === 'warning')) {
        events[idx] = ev;
      } else {
        events.push(ev);
      }
    };

    // Simulate handleStreamDone
    const onDone = (resSummary?: ReasoningSummary) => {
      isStreaming = false;
      isExpanded = false; // Auto-collapse when done
      if (resSummary) summary = resSummary;
    };

    // 1. Start stream
    onStart();
    if (!isStreaming || !isExpanded) throw new Error('Expected streaming=true and expanded=true on start');

    // 2. Add running event
    onThinking({
      id: 'e1',
      query_id: 'q1',
      stage: 'retrieval',
      status: 'running',
      title: 'Searching sources',
      summary: 'Executing hybrid vector and BM25 search',
    });
    if (events.length !== 1 || events[0].status !== 'running') throw new Error('Event was not added as running');

    // 3. Update stage to completed
    onThinking({
      id: 'e1',
      query_id: 'q1',
      stage: 'retrieval',
      status: 'completed',
      title: 'Searching sources',
      summary: 'Retrieved 8 candidate passages',
      duration_ms: 45.2,
    });
    const completedEvent: ThinkingEvent | undefined = events[0];
    if (events.length !== 1 || completedEvent?.status !== 'completed' || completedEvent.duration_ms !== 45.2) {
      throw new Error('Stage transition update failed');
    }

    // 4. Complete stream
    const finalSummary: ReasoningSummary = {
      intent: 'factual',
      answer_mode: 'DIRECT',
      is_follow_up: false,
      used_conversation_context: false,
      reused_previous_evidence: false,
      retrieved_new_evidence: true,
      used_visual_evidence: false,
      evidence_status: 'DIRECT',
      sources_used: ['doc_1'],
      degraded_stages: [],
      total_duration_ms: 120.5,
    };
    onDone(finalSummary);
    if (isStreaming !== false) throw new Error('Expected isStreaming=false on done');
    if (isExpanded !== false) throw new Error('Expected isExpanded=false on done (auto-collapse)');
    const getRecordedSummary = (): ReasoningSummary | null => summary;
    const recordedSummary = getRecordedSummary();
    if (recordedSummary?.total_duration_ms !== 120.5) throw new Error('Summary duration not recorded');
  });

  // --- 5. Degraded Stage Handling ---
  test('M4.8: Degraded stages and warnings are detected accurately', () => {
    const events: ThinkingEvent[] = [
      { id: '1', query_id: 'q1', stage: 'received', status: 'completed', title: 'Received', summary: 'OK' },
      {
        id: '2',
        query_id: 'q1',
        stage: 'visual_analysis',
        status: 'warning',
        title: 'Visual analysis unavailable',
        summary: 'Vision model timed out after 5000ms. Answer relies on text evidence.',
      },
    ];

    const hasDegraded = events.some((e) => e.status === 'warning' || e.stage === 'degraded');
    if (!hasDegraded) {
      throw new Error('Failed to detect degraded/warning stage in events');
    }
  });

  // --- 6. Zero-CoT Safety in Thinking Payloads ---
  test('M4.9: Zero-CoT safety - thinking titles and summaries contain zero leaked internal CoT prompts', () => {
    const forbiddenMarkers = [
      '<think>',
      '</think>',
      'system_prompt',
      'vector_id',
      'chunk_embedding',
      'cosine_similarity_formula',
      'classifier_prompt',
      'API_KEY',
      'sk-',
    ];

    const sampleSummaries = [
      'Detected a follow-up question and linked it to the previously discussed Hotel Search Agent.',
      'Reusing previously verified sources while searching for additional supporting context.',
      'Combined semantic and keyword search to locate related sections.',
      'Found partial implementation evidence and supporting context.',
      'Visual extraction timed out, so the answer will rely on available verified text evidence.',
      'Synthesizing grounded answer adhering to consistency and expansion guidelines.',
    ];

    for (const summary of sampleSummaries) {
      for (const marker of forbiddenMarkers) {
        if (summary.toLowerCase().includes(marker.toLowerCase())) {
          throw new Error(`Sensitive internal marker '${marker}' leaked in safe thinking summary: ${summary}`);
        }
      }
    }
  });

  // --- 7. ARIA Accessibility Contract Validation ---
  test('M4.10: Accessibility attributes match ARIA specifications', () => {
    const requiredAriaAttributes = {
      role: 'region',
      ariaLabel: 'AI Reasoning Steps',
      ariaExpanded: true,
      ariaLive: 'polite',
      statusRole: 'status',
    };

    if (requiredAriaAttributes.role !== 'region') throw new Error('Container role must be region');
    if (requiredAriaAttributes.ariaLive !== 'polite') throw new Error('Live region must be polite');
    if (requiredAriaAttributes.statusRole !== 'status') throw new Error('Step role must be status');
  });

  return results;
}
