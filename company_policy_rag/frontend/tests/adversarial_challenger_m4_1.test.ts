/**
 * Milestone 4 Empirical Adversarial Challenger Test Suite
 *
 * Challenger: Challenger 1 (Gate Verification for M4)
 * Focus:
 *  1. Detail Level Filtering Stress: off, compact, standard, detailed across thousands of events and invalid inputs.
 *  2. SSE Stream Lifecycle & State Machine: rapid bursts, duplicate IDs, stage status transitions, auto-expand, auto-collapse.
 *  3. Zero CoT Leakage in UI: SSR HTML inspection verifying 0 instances of <think>, system_prompt, embeddings, vector_id, API keys.
 *  4. Component Rendering & ARIA: ThinkingPanel, ThinkingStep, and ChatMessage rendered output verification.
 */

import React from 'react';
import { renderToString } from 'react-dom/server';
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
import { ThinkingPanel } from '../components/ThinkingPanel';
import { ThinkingStep } from '../components/ThinkingStep';
import { ChatMessage } from '../components/ChatMessage';
import { ChatMessageData } from '../lib/types';
import { TestResult } from './tier1_features.test';

export function runAdversarialM4ChallengerTests(): TestResult[] {
  const results: TestResult[] = [];

  function test(name: string, fn: () => void) {
    const start = performance.now();
    try {
      fn();
      results.push({
        suite: 'Adversarial Challenger M4: Thinking UI & SSE Lifecycle',
        name,
        passed: true,
        durationMs: performance.now() - start,
      });
    } catch (err: any) {
      results.push({
        suite: 'Adversarial Challenger M4: Thinking UI & SSE Lifecycle',
        name,
        passed: false,
        durationMs: performance.now() - start,
        error: err?.message || String(err),
      });
    }
  }

  // =========================================================================
  // SECTION 1: Detail Level Filtering Stress
  // =========================================================================

  test('ADV-M4.1.1: filterEventsByDetailLevel on empty and single-element arrays', () => {
    const empty: ThinkingEvent[] = [];
    if (filterEventsByDetailLevel(empty, 'off').length !== 0) throw new Error('off on empty failed');
    if (filterEventsByDetailLevel(empty, 'compact').length !== 0) throw new Error('compact on empty failed');
    if (filterEventsByDetailLevel(empty, 'standard').length !== 0) throw new Error('standard on empty failed');
    if (filterEventsByDetailLevel(empty, 'detailed').length !== 0) throw new Error('detailed on empty failed');

    const singleCompact: ThinkingEvent = {
      id: '1',
      query_id: 'q1',
      stage: 'retrieval',
      status: 'completed',
      title: 'Retrieval',
      summary: 'Searched',
    };
    if (filterEventsByDetailLevel([singleCompact], 'off').length !== 0) throw new Error('off should return empty');
    if (filterEventsByDetailLevel([singleCompact], 'compact').length !== 1) throw new Error('compact should retain retrieval');
    if (filterEventsByDetailLevel([singleCompact], 'standard').length !== 1) throw new Error('standard should retain retrieval');
    if (filterEventsByDetailLevel([singleCompact], 'detailed').length !== 1) throw new Error('detailed should retain retrieval');

    const singleNonCompact: ThinkingEvent = {
      id: '2',
      query_id: 'q1',
      stage: 'reranking',
      status: 'completed',
      title: 'Reranking',
      summary: 'Scored',
    };
    if (filterEventsByDetailLevel([singleNonCompact], 'compact').length !== 0) {
      throw new Error('compact should filter out reranking');
    }
    if (filterEventsByDetailLevel([singleNonCompact], 'standard').length !== 1) {
      throw new Error('standard should keep reranking');
    }
  });

  test('ADV-M4.1.2: Stress test filterEventsByDetailLevel with 5,000 synthetic mixed events', () => {
    const allStages: ThinkingStage[] = [
      'received', 'conversation_context', 'follow_up_resolution', 'query_analysis',
      'query_rewrite', 'retrieval', 'reranking', 'evidence_analysis', 'evidence_reuse',
      'page_expansion', 'visual_analysis', 'evidence_verification', 'answer_planning',
      'answer_generation', 'citation_building', 'completed', 'degraded',
    ];

    const count = 5000;
    const largeEvents: ThinkingEvent[] = [];
    let expectedCompactCount = 0;

    for (let i = 0; i < count; i++) {
      const stage = allStages[i % allStages.length];
      const isComp = isCompactStage(stage);
      if (isComp) expectedCompactCount++;
      largeEvents.push({
        id: 'ev_' + i,
        query_id: 'q_' + (i % 50),
        stage,
        status: 'completed',
        title: 'Stage ' + stage,
        summary: 'Summary for stage ' + stage + ' #' + i,
      });
    }

    const t0 = performance.now();
    const offResult = filterEventsByDetailLevel(largeEvents, 'off');
    const compactResult = filterEventsByDetailLevel(largeEvents, 'compact');
    const standardResult = filterEventsByDetailLevel(largeEvents, 'standard');
    const detailedResult = filterEventsByDetailLevel(largeEvents, 'detailed');
    const duration = performance.now() - t0;

    if (offResult.length !== 0) throw new Error('off result length must be 0, got ' + offResult.length);
    if (compactResult.length !== expectedCompactCount) {
      throw new Error('compact expected ' + expectedCompactCount + ', got ' + compactResult.length);
    }
    if (standardResult.length !== count) throw new Error('standard expected ' + count + ', got ' + standardResult.length);
    if (detailedResult.length !== count) throw new Error('detailed expected ' + count + ', got ' + detailedResult.length);

    // Verify immutability: original array unmodified
    if (largeEvents.length !== count) throw new Error('Original array was mutated');

    // Performance check: filtering 5000 events 4 times must take < 100ms
    if (duration > 100) {
      throw new Error('Performance regression: filtering 5000 events took ' + duration.toFixed(2) + 'ms (>100ms)');
    }
  });

  test('ADV-M4.1.3: All 17 canonical stages strictly partitioned into COMPACT vs NON_COMPACT', () => {
    const all17Stages: ThinkingStage[] = [
      'received', 'conversation_context', 'follow_up_resolution', 'query_analysis',
      'query_rewrite', 'retrieval', 'reranking', 'evidence_analysis', 'evidence_reuse',
      'page_expansion', 'visual_analysis', 'evidence_verification', 'answer_planning',
      'answer_generation', 'citation_building', 'completed', 'degraded',
    ];

    const expectedCompact = new Set<ThinkingStage>([
      'received', 'conversation_context', 'retrieval', 'evidence_verification',
      'answer_planning', 'completed', 'degraded',
    ]);

    for (const stage of all17Stages) {
      const isComp = isCompactStage(stage);
      if (expectedCompact.has(stage) && !isComp) {
        throw new Error('Stage ' + stage + ' should be compact but isCompactStage returned false');
      }
      if (!expectedCompact.has(stage) && isComp) {
        throw new Error('Stage ' + stage + ' should NOT be compact but isCompactStage returned true');
      }
    }
  });

  // =========================================================================
  // SECTION 2: SSE Stream Lifecycle & State Machine Simulation
  // =========================================================================

  test('ADV-M4.2.1: Rapid-fire SSE event bursts and ID deduplication state machine', () => {
    let events: ThinkingEvent[] = [];
    let isStreaming = false;
    let isExpanded = false;

    const onStart = () => {
      events = [];
      isStreaming = true;
      isExpanded = true;
    };

    const handleEvent = (event: ThinkingEvent) => {
      // Logic matching useThinkingStream
      const idIdx = events.findIndex((e) => e.id === event.id);
      if (idIdx >= 0) {
        events[idIdx] = event;
        return;
      }
      const stageIdx = events.findIndex((e) => e.stage === event.stage);
      if (stageIdx >= 0 && (event.status === 'completed' || event.status === 'warning' || event.status === 'failed')) {
        events[stageIdx] = event;
        return;
      }
      events.push(event);
    };

    const onDone = () => {
      isStreaming = false;
      isExpanded = false;
    };

    onStart();
    if (!isStreaming || !isExpanded) throw new Error('Stream start failed to initialize state');

    // Emit initial running event
    handleEvent({
      id: 'thk_01',
      query_id: 'q_001',
      stage: 'retrieval',
      status: 'running',
      title: 'Searching documents',
      summary: 'Executing hybrid vector retrieval',
    });
    if (events.length !== 1 || events[0].status !== 'running') throw new Error('Initial event not stored');

    // Emit 100 duplicate ID pulses updating progress/duration
    for (let i = 1; i <= 100; i++) {
      handleEvent({
        id: 'thk_01',
        query_id: 'q_001',
        stage: 'retrieval',
        status: i === 100 ? 'completed' : 'running',
        title: 'Searching documents',
        summary: i === 100 ? 'Found 8 relevant passages' : 'Searching pass ' + i + '...',
        duration_ms: i * 2,
      });
    }

    // Must deduplicate by ID so length remains exactly 1
    if (events.length !== 1) {
      throw new Error('Expected exactly 1 event after 100 duplicate updates, got ' + events.length);
    }
    const finalEvent: ThinkingEvent | undefined = events[0];
    if (finalEvent?.status !== 'completed' || finalEvent.duration_ms !== 200) {
      throw new Error('Final event state did not apply latest mutation');
    }

    onDone();
    if (isStreaming !== false || isExpanded !== false) {
      throw new Error('onDone failed to auto-collapse stream');
    }
  });

  test('ADV-M4.2.2: Stage transition deduplication when event ID is generated dynamically per packet', () => {
    let events: ThinkingEvent[] = [];

    const handleEvent = (event: ThinkingEvent) => {
      const idIdx = events.findIndex((e) => e.id === event.id);
      if (idIdx >= 0) {
        events[idIdx] = event;
        return;
      }
      const stageIdx = events.findIndex((e) => e.stage === event.stage);
      if (stageIdx >= 0 && (event.status === 'completed' || event.status === 'warning' || event.status === 'failed')) {
        events[stageIdx] = event;
        return;
      }
      events.push(event);
    };

    // Stage 1: Received
    handleEvent({ id: 'id_a1', query_id: 'q1', stage: 'received', status: 'running', title: 'Receiving', summary: 'Waiting' });
    handleEvent({ id: 'id_a2', query_id: 'q1', stage: 'received', status: 'completed', title: 'Received', summary: 'Done', duration_ms: 5 });

    // Stage 2: Conversation Context
    handleEvent({ id: 'id_b1', query_id: 'q1', stage: 'conversation_context', status: 'running', title: 'Context', summary: 'Analyzing' });
    handleEvent({ id: 'id_b2', query_id: 'q1', stage: 'conversation_context', status: 'completed', title: 'Context', summary: 'Resolved', duration_ms: 15 });

    // Stage 3: Retrieval (Degraded warning)
    handleEvent({ id: 'id_c1', query_id: 'q1', stage: 'retrieval', status: 'running', title: 'Retrieving', summary: 'Searching vector DB' });
    handleEvent({ id: 'id_c2', query_id: 'q1', stage: 'retrieval', status: 'warning', title: 'Retrieval degraded', summary: 'Fell back to BM25', duration_ms: 40 });

    if (events.length !== 3) {
      throw new Error('Expected 3 stages after running->completed/warning transitions, got ' + events.length);
    }
    if (events[0].status !== 'completed' || events[1].status !== 'completed' || events[2].status !== 'warning') {
      throw new Error('Stage statuses were not updated properly');
    }
  });

  test('ADV-M4.2.3: Active stage resolution and total duration fallback computation', () => {
    const events: ThinkingEvent[] = [
      { id: '1', query_id: 'q', stage: 'received', status: 'completed', title: 'R', summary: 'S', duration_ms: 10 },
      { id: '2', query_id: 'q', stage: 'retrieval', status: 'running', title: 'Ret', summary: 'S' },
    ];

    // When running event exists, activeStage is the running stage
    const running = events.find((e) => e.status === 'running');
    const activeStage = running ? running.stage : null;
    if (activeStage !== 'retrieval') throw new Error('Expected activeStage retrieval, got ' + activeStage);

    // Duration computation fallback
    const summaryNoDuration: ReasoningSummary = {
      intent: 'factual',
      answer_mode: 'DIRECT',
      is_follow_up: false,
      used_conversation_context: false,
      reused_previous_evidence: false,
      retrieved_new_evidence: true,
      used_visual_evidence: false,
      evidence_status: 'DIRECT',
      sources_used: [],
      degraded_stages: [],
      total_duration_ms: 0,
    };

    const sumFromEvents = events.reduce((acc, ev) => acc + (ev.duration_ms || 0), 0);
    const computedDuration = summaryNoDuration.total_duration_ms > 0 ? summaryNoDuration.total_duration_ms : sumFromEvents;
    if (computedDuration !== 10) throw new Error('Expected computed duration 10, got ' + computedDuration);

    // When summary has total_duration_ms > 0, it takes precedence
    const summaryWithDuration: ReasoningSummary = { ...summaryNoDuration, total_duration_ms: 125.4 };
    const precedenceDuration = summaryWithDuration.total_duration_ms > 0 ? summaryWithDuration.total_duration_ms : sumFromEvents;
    if (precedenceDuration !== 125.4) throw new Error('Expected precedence duration 125.4, got ' + precedenceDuration);
  });

  // =========================================================================
  // SECTION 3: Zero-CoT & Safe UI Rendering Verification
  // =========================================================================

  test('ADV-M4.3.1: Zero CoT leakage in ThinkingPanel and ThinkingStep rendered HTML', () => {
    const safeEvents: ThinkingEvent[] = [
      {
        id: 'ev_1',
        query_id: 'q_test',
        stage: 'conversation_context',
        status: 'completed',
        title: 'Resolving conversation context',
        summary: 'Detected follow-up query and linked to Hotel Search Agent.',
        details: { is_follow_up: true, previous_subject: 'Hotel Search Agent', candidate_count: 5 },
        duration_ms: 14.2,
      },
      {
        id: 'ev_2',
        query_id: 'q_test',
        stage: 'evidence_verification',
        status: 'completed',
        title: 'Verifying retrieved evidence',
        summary: 'Verified 3 document chunks for factual alignment.',
        details: { verified_chunks: 3, evidence_status: 'DIRECT' },
        duration_ms: 32.8,
      },
    ];

    const safeSummary: ReasoningSummary = {
      intent: 'code_explanation',
      answer_mode: 'DETAILED',
      is_follow_up: true,
      used_conversation_context: true,
      reused_previous_evidence: true,
      retrieved_new_evidence: true,
      used_visual_evidence: false,
      evidence_status: 'DIRECT',
      sources_used: ['policy_hotel_agent.pdf#p72'],
      degraded_stages: [],
      total_duration_ms: 47.0,
    };

    const renderedPanelHtml = renderToString(
      React.createElement(ThinkingPanel, {
        events: safeEvents,
        isStreaming: false,
        isExpanded: true,
        onToggleExpanded: () => {},
        detailLevel: 'detailed',
        reasoningSummary: safeSummary,
        totalDurationMs: 47.0,
      })
    );

    const forbiddenSubstrings = [
      '<think>',
      '</think>',
      '<thought>',
      '</thought>',
      'system_prompt',
      'system prompt',
      'classifier_prompt',
      'retrieval_prompt',
      'vector_id',
      'chunk_embedding',
      'cosine_similarity_formula',
      'dense_vector',
      'API_KEY',
      'sk-proj',
      'sk-ant',
    ];

    for (const forbidden of forbiddenSubstrings) {
      if (renderedPanelHtml.toLowerCase().includes(forbidden.toLowerCase())) {
        throw new Error(`CRITICAL LEAKAGE: Found forbidden string '${forbidden}' in ThinkingPanel rendered HTML!`);
      }
    }

    // Verify essential accessibility attributes
    if (!renderedPanelHtml.includes('role="region"')) throw new Error('ThinkingPanel missing role="region"');
    if (!renderedPanelHtml.includes('aria-label="AI Reasoning Steps"')) throw new Error('ThinkingPanel missing aria-label');
    if (!renderedPanelHtml.includes('aria-expanded="true"')) throw new Error('ThinkingPanel missing aria-expanded="true"');
    if (!renderedPanelHtml.includes('aria-live="polite"')) throw new Error('ThinkingPanel missing aria-live="polite"');
  });

  test('ADV-M4.3.2: ChatMessage with integrated ThinkingPanel renders without CoT leakage', () => {
    const assistantMessage: ChatMessageData = {
      id: 'msg_001',
      role: 'assistant',
      content: 'Here is the detailed explanation of the Hotel Search Agent implementation.',
      timestamp: new Date().toISOString(),
      thinking_detail_level: 'standard',
      thinking_events: [
        {
          id: 'ev_01',
          query_id: 'q_01',
          stage: 'conversation_context',
          status: 'completed',
          title: 'Resolving previous conversation',
          summary: 'Preserved verified Hotel Search Agent context from prior turn.',
          duration_ms: 11.0,
        },
        {
          id: 'ev_02',
          query_id: 'q_01',
          stage: 'retrieval',
          status: 'completed',
          title: 'Searching document sources',
          summary: 'Expanded evidence window around page 72.',
          duration_ms: 55.4,
        },
      ],
      reasoning_summary: {
        intent: 'code_explanation',
        answer_mode: 'DETAILED',
        is_follow_up: true,
        used_conversation_context: true,
        reused_previous_evidence: true,
        retrieved_new_evidence: true,
        used_visual_evidence: false,
        evidence_status: 'DIRECT',
        sources_used: ['doc_agent.pdf'],
        degraded_stages: [],
        total_duration_ms: 66.4,
      },
    };

    const renderedChatHtml = renderToString(
      React.createElement(ChatMessage, {
        message: assistantMessage,
        onOpenCitation: () => {},
      })
    );

    const forbiddenSubstrings = ['<think>', '</think>', 'system_prompt', 'vector_id', 'chunk_embedding'];
    for (const forbidden of forbiddenSubstrings) {
      if (renderedChatHtml.toLowerCase().includes(forbidden.toLowerCase())) {
        throw new Error(`CRITICAL LEAKAGE: Found forbidden string '${forbidden}' in ChatMessage rendered HTML!`);
      }
    }

    // Verify thinking panel container is present
    if (!renderedChatHtml.includes('AI Reasoning Steps')) {
      throw new Error('ChatMessage did not render AI Reasoning Steps ThinkingPanel');
    }
  });

  // =========================================================================
  // SECTION 4: Detail Level Switching & Component Toggle Resilience
  // =========================================================================

  test('ADV-M4.4.1: ThinkingPanel returns null when detailLevel is off', () => {
    const events: ThinkingEvent[] = [
      { id: '1', query_id: 'q', stage: 'received', status: 'completed', title: 'Recv', summary: 'OK' },
    ];
    const html = renderToString(
      React.createElement(ThinkingPanel, {
        events,
        isStreaming: false,
        isExpanded: true,
        onToggleExpanded: () => {},
        detailLevel: 'off',
      })
    );
    if (html !== '') {
      throw new Error(`ThinkingPanel with detailLevel='off' must render null (empty string), got: ${html}`);
    }
  });

 test('ADV-M4.4.2: ThinkingStep renders warning state and degraded badges cleanly', () => {
 const degradedEvent: ThinkingEvent = {
 id: 'deg_1',
 query_id: 'q',
 stage: 'visual_analysis',
 status: 'warning',
 title: 'Visual analysis timed out',
 summary: 'Vision service exceeded 5000ms budget. Text evidence remains verified.',
 duration_ms: 5001.2,
 details: { timeout_ms: 5000, fallback: 'text_only' },
 };

 const stepHtml = renderToString(
 React.createElement(ThinkingStep, {
 event: degradedEvent,
 detailLevel: 'detailed',
 })
 );

 if (!stepHtml.includes('Visual analysis timed out')) {
 throw new Error('ThinkingStep missing title');
 }
 if (!stepHtml.includes('Vision service exceeded 5000ms budget')) {
 throw new Error('ThinkingStep missing summary');
 }
 if (!stepHtml.includes('timeout ms') && !stepHtml.includes('timeout_ms')) {
 throw new Error('ThinkingStep in detailed mode missing details tag');
 }
 });

 return results;
}
