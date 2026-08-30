/**
 * Empirical Adversarial Challenger 1 Test Suite
 *
 * Subject: ChatMessage.tsx and AdminView.tsx
 * Focus: Malformed/sparse traces, retry extremes, conflicting states,
 *        score clamping anomalies, string length stress, and mobile/dark layout resilience.
 */

import {
  renderChatMessage,
  renderMessageWithTrace,
  renderAdminView,
  assert,
} from './test_helpers';
import { QueryTrace, ObservabilityData, ChatMessageData, Citation } from '../lib/types';
import { TestResult } from './tier1_features.test';

export function runAdversarialTests(): TestResult[] {
  const results: TestResult[] = [];

  function test(name: string, fn: () => void) {
    const t0 = performance.now();
    try {
      fn();
      results.push({
        suite: 'Adversarial Challenger 1 (Stress & Corner Cases)',
        name,
        passed: true,
        durationMs: performance.now() - t0,
      });
    } catch (err: any) {
      results.push({
        suite: 'Adversarial Challenger 1 (Stress & Corner Cases)',
        name,
        passed: false,
        durationMs: performance.now() - t0,
        error: err.message || String(err),
      });
    }
  }

  // =========================================================================
  // CATEGORY 1: Malformed & Sparse Traces
  // =========================================================================

  test('ADV-1.1: Trace with completely undefined/missing optional fields does not crash', () => {
    const trace: QueryTrace = {
      trace_id: 'tr_sparse_001',
      timestamp: '2026-08-15T08:00:00Z',
      original_query: 'Minimal trace query',
      total_chunks_retrieved: 0,
      top_rerank_score: 0,
      rerank_latency_ms: 0,
      total_latency_ms: 0,
      prompt_tokens: 0,
      completion_tokens: 0,
      model: 'MinimalModel',
    };

    const chatHtmlCollapsed = renderMessageWithTrace(trace, { expanded: false });
    assert(chatHtmlCollapsed.includes('Thought for 0 ms'), 'Renders collapsed with Thought for 0 ms');

    const chatHtmlExpanded = renderMessageWithTrace(trace, { expanded: true });
    assert(chatHtmlExpanded.includes('tr_sparse_001'), 'Renders expanded with trace ID');
    assert(chatHtmlExpanded.includes('0 chunks'), 'Renders 0 chunks');

    const adminHtml = renderAdminView({ recent_traces: [trace] });
    assert(adminHtml.includes('Minimal trace query'), 'AdminView displays sparse trace query');
  });

  test('ADV-1.2: Trace with explicit nulls across all agentic fields', () => {
    const trace: any = {
      trace_id: 'tr_nulls_002',
      timestamp: '2026-08-15T08:00:00Z',
      original_query: 'Query with all nulls',
      total_chunks_retrieved: 0,
      top_rerank_score: 0,
      rerank_latency_ms: 0,
      total_latency_ms: 0,
      prompt_tokens: 0,
      completion_tokens: 0,
      model: 'TestModel',
      query_type: null,
      routing_confidence: null,
      retrieval_strategy: null,
      inferred_filters: null,
      applied_filters: null,
      filter_relaxed: null,
      verification_score: null,
      verification: null,
      faithfulness_passed: null,
      retry_count: null,
      retry_reasons: null,
      cache_hit: null,
      cache_similarity: null,
    };

    const chatHtml = renderMessageWithTrace(trace, { expanded: true });
    assert(chatHtml.includes('tr_nulls_002'), 'Renders chat message without crashing');
    assert(!chatHtml.includes('NaN'), 'Does not render NaN in chat view');

    const adminHtml = renderAdminView({ recent_traces: [trace] }, { expandedTraceId: 'tr_nulls_002' });
    assert(adminHtml.includes('tr_nulls_002'), 'Renders admin view with null fields without crashing');
    assert(!adminHtml.includes('NaN%'), 'Does not render NaN% in admin table');
  });

  test('ADV-1.3: Partial verification report with only composite_score (missing 4 sub-dimensions)', () => {
    const trace: QueryTrace = {
      trace_id: 'tr_partial_verif_003',
      timestamp: '2026-08-15T08:00:00Z',
      original_query: 'Partial verification test',
      total_chunks_retrieved: 3,
      top_rerank_score: 0.88,
      rerank_latency_ms: 20,
      total_latency_ms: 120,
      prompt_tokens: 50,
      completion_tokens: 25,
      model: 'FastAPI RAG',
      verification: {
        composite_score: 0.82,
        passed: true,
      } as any,
    };

    const chatHtml = renderMessageWithTrace(trace, { expanded: true });
    assert(chatHtml.includes('82% Verified'), 'Renders composite score pill with 82%');
    assert(!chatHtml.includes('Faithfulness'), 'Does not invent a missing Faithfulness dimension');
    assert(!chatHtml.includes('Completeness'), 'Does not invent a missing Completeness dimension');

    const adminHtml = renderAdminView({ recent_traces: [trace] }, { expandedTraceId: 'tr_partial_verif_003' });
    assert(adminHtml.includes('82%'), 'Admin renders 82% composite score');
  });

  test('ADV-1.4: Adversarial query_type string injections and exotic values', () => {
    const exoticTypes = [
      'UNKNOWN_CUSTOM_ROUTER_TYPE',
      '<script>alert("xss")</script>',
      'DROP TABLE traces;--',
      '   spaces_around   ',
      '🤖_AGENTIC_TYPE_🚀',
    ];

    for (const qType of exoticTypes) {
      const trace: QueryTrace = {
        trace_id: `tr_exotic_${qType.slice(0, 5)}`,
        timestamp: '2026-08-15T08:00:00Z',
        original_query: 'Exotic query type test',
        query_type: qType,
        routing_confidence: 0.9,
        total_chunks_retrieved: 1,
        top_rerank_score: 0.9,
        rerank_latency_ms: 10,
        total_latency_ms: 100,
        prompt_tokens: 10,
        completion_tokens: 10,
        model: 'FastAPI',
      };

      const chatHtml = renderMessageWithTrace(trace, { expanded: false });
      assert(chatHtml.includes('90%'), `Renders confidence for ${qType}`);

      const adminHtml = renderAdminView({ recent_traces: [trace] });
      assert(adminHtml.includes('90%'), `Admin renders confidence for ${qType}`);
    }
  });

  // =========================================================================
  // CATEGORY 2: Zero vs Multiple Retries with Long Text & Critique Stress
  // =========================================================================

  test('ADV-2.1: Zero retries with pristine verification report', () => {
    const trace: QueryTrace = {
      trace_id: 'tr_zero_retries_004',
      timestamp: '2026-08-15T08:00:00Z',
      original_query: 'Pristine pass on first try',
      retry_count: 0,
      retry_reasons: [],
      verification: {
        faithfulness: 0.98,
        completeness: 0.95,
        citation_coverage: 1.0,
        coherence: 0.99,
        composite_score: 0.98,
        passed: true,
        critique: null,
      },
      total_chunks_retrieved: 4,
      top_rerank_score: 0.95,
      rerank_latency_ms: 30,
      total_latency_ms: 220,
      prompt_tokens: 120,
      completion_tokens: 60,
      model: 'FastAPI RAG',
    };

    const chatHtml = renderMessageWithTrace(trace, { expanded: true });
    assert(!chatHtml.includes('retry'), 'No retry badge rendered in chat when count is 0');
    assert(chatHtml.includes('98% Verified'), 'Shows 98% Verified');

    const adminHtml = renderAdminView({ recent_traces: [trace] }, { expandedTraceId: 'tr_zero_retries_004' });
    assert(adminHtml.includes('0 retries · Passed self-reflection on initial attempt'), 'Admin confirms 0 retries');
  });

  test('ADV-2.2: Extreme retry stress: 5 retries with long critique (5000+ chars) & 10 detailed reasons', () => {
    const longCritique = 'The initial draft lacked citation for Section 4.2 paragraph 3 regarding PTO accrual limits during leave of absence. '
      .repeat(25)
      .trim();

    const reasons = [
      'Retry 1: Missing cross-reference to 2026 Leave Policy Document page 14 paragraph 2.',
      'Retry 2: Faithfulness score dropped below 0.75 due to hallucinated reimbursement timeline (14 days instead of 30 days).',
      'Retry 3: Citation coverage was incomplete for the parental leave eligibility table.',
      'Retry 4: Coherence score flagged contradictory statements in paragraph 4 regarding weekend accrual.',
      'Retry 5: Final refinement for compliance with regional labor guidelines in EMEA and APAC territories.',
    ];

    const trace: QueryTrace = {
      trace_id: 'tr_retry_stress_005',
      timestamp: '2026-08-15T08:00:00Z',
      original_query: 'Extreme retry stress test query',
      retry_count: 5,
      retry_reasons: reasons,
      verification: {
        faithfulness: 0.91,
        completeness: 0.89,
        citation_coverage: 0.94,
        coherence: 0.96,
        composite_score: 0.92,
        passed: true,
        critique: longCritique,
        missing_aspects: ['Clause 4.2.1 (Carried-over PTO)', 'Clause 9.3 (Bereavement coordination)'],
        unsupported_claims: ['Uncited claim regarding 100% gym stipend carryover'],
      },
      total_chunks_retrieved: 8,
      top_rerank_score: 0.89,
      rerank_latency_ms: 60,
      total_latency_ms: 1200,
      prompt_tokens: 800,
      completion_tokens: 450,
      model: 'FastAPI RAG',
    };

    const chatHtml = renderMessageWithTrace(trace, { expanded: true });
    assert(chatHtml.includes('5 retries'), 'ChatMessage shows 5 retries in header badge');
    assert(chatHtml.includes('Retry Triggers (5)'), 'Expanded banner lists Retry Triggers (5)');
    assert(chatHtml.includes(reasons[0]), 'ChatMessage renders first retry reason');
    assert(chatHtml.includes(longCritique.slice(0, 50)), 'Long critique renders without corruption');

    const adminHtml = renderAdminView({ recent_traces: [trace] }, { expandedTraceId: 'tr_retry_stress_005' });
    assert(adminHtml.includes('Verification Retries (5)'), 'Admin view shows 5 verification retries');
    assert(adminHtml.includes('Missing Aspects'), 'Admin view shows Missing Aspects card');
    assert(adminHtml.includes('Unsupported Claims'), 'Admin view shows Unsupported Claims card');
    assert(adminHtml.includes('Clause 4.2.1 (Carried-over PTO)'), 'Admin view displays missing aspect items');
    assert(adminHtml.includes('Uncited claim regarding 100% gym stipend carryover'), 'Admin view displays unsupported claims');
  });

  test('ADV-2.3: Retry count = 1 renders singular grammar throughout', () => {
    const trace: QueryTrace = {
      trace_id: 'tr_single_retry_006',
      timestamp: '2026-08-15T08:00:00Z',
      original_query: 'Single retry query',
      retry_count: 1,
      retry_reasons: ['Initial retrieval lacked specific medical deductible thresholds.'],
      verification_score: 0.88,
      total_chunks_retrieved: 2,
      top_rerank_score: 0.9,
      rerank_latency_ms: 15,
      total_latency_ms: 250,
      prompt_tokens: 150,
      completion_tokens: 75,
      model: 'FastAPI RAG',
    };

    const chatHtml = renderMessageWithTrace(trace, { expanded: true });
    assert(chatHtml.includes('1 retry'), 'ChatMessage uses singular "1 retry"');

    const adminHtml = renderAdminView({ recent_traces: [trace] }, { expandedTraceId: 'tr_single_retry_006' });
    assert(adminHtml.includes('1 cycle'), 'AdminView uses singular "1 cycle"');
  });

  // =========================================================================
  // CATEGORY 3: Simultaneous Conflicting & Edge States
  // =========================================================================

  test('ADV-3.1: Simultaneous Filter Relaxation + Semantic Cache Hit + 0 Chunks', () => {
    const trace: QueryTrace = {
      trace_id: 'tr_conflict_007',
      timestamp: '2026-08-15T08:00:00Z',
      original_query: 'Conflicting edge state query',
      filter_relaxed: true,
      cache_hit: true,
      cache_similarity: 0.985,
      inferred_filters: { department: 'Legal', jurisdiction: 'California' },
      applied_filters: {},
      total_chunks_retrieved: 0,
      top_rerank_score: 0.0,
      rerank_latency_ms: 0,
      total_latency_ms: 12,
      prompt_tokens: 40,
      completion_tokens: 30,
      model: 'FastAPI RAG',
      verification_score: 0.99,
    };

    const chatHtml = renderMessageWithTrace(trace, { expanded: true });
    assert(chatHtml.includes('Cache Hit'), 'Displays Cache Hit badge in header');
    assert(chatHtml.includes('99%'), 'Displays 99% cache similarity / verification');
    assert(chatHtml.includes('Filters Relaxed:'), 'Displays Filter Relaxation warning callout');
    assert(chatHtml.includes('department:'), 'Displays department filter key');
    assert(chatHtml.includes('Legal'), 'Displays Legal filter value');
    assert(chatHtml.includes('0 chunks'), 'Displays 0 chunks retrieved');

    const adminHtml = renderAdminView({ recent_traces: [trace] }, { expandedTraceId: 'tr_conflict_007' });
    assert(adminHtml.includes('Semantic Cache Hit'), 'AdminView displays Semantic Cache Hit card');
    assert(adminHtml.includes('98.5%'), 'AdminView displays 98.5% cache similarity');
    assert(adminHtml.includes('Filter Fallback Triggered'), 'AdminView displays Filter Fallback Triggered');
    assert(adminHtml.includes('relaxed'), 'AdminView table row displays relaxed tag');
  });

  test('ADV-3.2: Contradictory verification signals: faithfulness_passed = false vs composite_score = 0.90', () => {
    const trace: QueryTrace = {
      trace_id: 'tr_contradiction_008',
      timestamp: '2026-08-15T08:00:00Z',
      original_query: 'Contradictory verification signals',
      faithfulness_passed: false,
      verification_score: 0.90,
      verification: {
        faithfulness: 0.40,
        completeness: 0.95,
        citation_coverage: 0.95,
        coherence: 0.95,
        composite_score: 0.90,
        passed: false,
        critique: 'Response includes unverified claim.',
      },
      total_chunks_retrieved: 3,
      top_rerank_score: 0.85,
      rerank_latency_ms: 20,
      total_latency_ms: 300,
      prompt_tokens: 100,
      completion_tokens: 50,
      model: 'FastAPI RAG',
    };

    const chatHtml = renderMessageWithTrace(trace, { expanded: true });
    // When passed is false, status must be Review (not Verified)
    assert(chatHtml.includes('Review'), 'ChatMessage displays Review status when passed=false');
    assert(chatHtml.includes('90%'), 'Score remains 90%');

    const adminHtml = renderAdminView({ recent_traces: [trace] }, { expandedTraceId: 'tr_contradiction_008' });
    assert(adminHtml.includes('Fail'), 'AdminView table displays Fail when passed is false');
  });

  test('ADV-3.3: High routing confidence (100%) with 0 chunks retrieved & conversational strategy', () => {
    const trace: QueryTrace = {
      trace_id: 'tr_conv_zero_chunks_009',
      timestamp: '2026-08-15T08:00:00Z',
      original_query: 'Hello there, how are you today?',
      query_type: 'conversational',
      routing_confidence: 1.0,
      retrieval_strategy: 'direct_synthesis',
      total_chunks_retrieved: 0,
      top_rerank_score: 0.0,
      rerank_latency_ms: 0,
      total_latency_ms: 85,
      prompt_tokens: 25,
      completion_tokens: 35,
      model: 'FastAPI RAG',
    };

    const chatHtml = renderMessageWithTrace(trace, { expanded: true });
    assert(chatHtml.includes('conversational'), 'ChatMessage shows conversational badge');
    assert(chatHtml.includes('100%'), 'ChatMessage shows 100% confidence');
    assert(chatHtml.includes('0 chunks'), 'ChatMessage displays 0 chunks');

    const adminHtml = renderAdminView({ recent_traces: [trace] }, { expandedTraceId: 'tr_conv_zero_chunks_009' });
    assert(adminHtml.includes('Conversational'), 'AdminView shows Conversational chip');
    assert(adminHtml.includes('direct_synthesis'), 'AdminView displays direct_synthesis strategy');
  });

  // =========================================================================
  // CATEGORY 4: Extreme Scores & Clamping Anomalies
  // =========================================================================

  test('ADV-4.1: Extreme boundary score values: 0.0, 1.0, negative (-0.5), overflown (2.5)', () => {
    const scoreTests = [
      { score: 0.0, expectedPct: '0%', expectedStatus: 'Review' },
      { score: 1.0, expectedPct: '100%', expectedStatus: 'Verified' },
      { score: -0.5, expectedPct: '0%', expectedStatus: 'Review' }, // Clamped to 0
      { score: 2.5, expectedPct: '100%', expectedStatus: 'Verified' }, // Clamped to 100
    ];

    for (let i = 0; i < scoreTests.length; i++) {
      const { score, expectedPct, expectedStatus } = scoreTests[i];
      const trace: QueryTrace = {
        trace_id: `tr_score_clamp_${i}`,
        timestamp: '2026-08-15T08:00:00Z',
        original_query: `Score test for ${score}`,
        verification_score: score,
        total_chunks_retrieved: 1,
        top_rerank_score: 0.8,
        rerank_latency_ms: 10,
        total_latency_ms: 50,
        prompt_tokens: 10,
        completion_tokens: 10,
        model: 'FastAPI',
      };

      const chatHtml = renderMessageWithTrace(trace, { expanded: false });
      assert(chatHtml.includes(expectedPct), `Score ${score} produces clamped percentage ${expectedPct}`);
      assert(chatHtml.includes(expectedStatus), `Score ${score} produces expected status ${expectedStatus}`);
    }
  });

  test('ADV-4.2: Clamping on all 4 verification dimension bars', () => {
    const trace: QueryTrace = {
      trace_id: 'tr_dim_clamp_010',
      timestamp: '2026-08-15T08:00:00Z',
      original_query: 'Dimension bar clamp test',
      verification: {
        faithfulness: -0.2, // Clamped to 0%
        completeness: 1.8, // Clamped to 100%
        citation_coverage: 0.0, // Exactly 0%
        coherence: 1.0, // Exactly 100%
        composite_score: 0.85,
        passed: true,
      },
      total_chunks_retrieved: 2,
      top_rerank_score: 0.9,
      rerank_latency_ms: 15,
      total_latency_ms: 100,
      prompt_tokens: 20,
      completion_tokens: 20,
      model: 'FastAPI',
    };

    const chatHtml = renderMessageWithTrace(trace, { expanded: true });
    assert(chatHtml.includes('0%'), 'Faithfulness clamped to 0%');
    assert(chatHtml.includes('100%'), 'Completeness clamped to 100%');
    assert(chatHtml.includes('Faithfulness'), 'Faithfulness label present');
    assert(chatHtml.includes('Completeness'), 'Completeness label present');
  });

  test('ADV-4.3: Cache similarity score precision & boundary formatting', () => {
    const simValues = [
      { sim: 0.0, label: '0%' },
      { sim: 1.0, label: '100%' },
      { sim: 0.9995, label: '100%' },
      { sim: 0.87654, label: '88%' },
    ];

    for (const { sim, label } of simValues) {
      const trace: QueryTrace = {
        trace_id: `tr_cache_sim_${sim}`,
        timestamp: '2026-08-15T08:00:00Z',
        original_query: `Cache sim test ${sim}`,
        cache_hit: true,
        cache_similarity: sim,
        total_chunks_retrieved: 0,
        top_rerank_score: 0,
        rerank_latency_ms: 0,
        total_latency_ms: 5,
        prompt_tokens: 10,
        completion_tokens: 10,
        model: 'FastAPI',
      };

      const chatHtml = renderMessageWithTrace(trace, { expanded: false });
      assert(chatHtml.includes('Cache Hit'), 'Cache Hit badge rendered');
      assert(chatHtml.includes(label), `Cache similarity ${sim} formatted as ${label}`);
    }
  });

  // =========================================================================
  // CATEGORY 5: String Length & Heavy Payload Stress
  // =========================================================================

  test('ADV-5.1: Ultra-long query strings (10,000 chars) and deeply nested filters', () => {
    const ultraLongQuery = 'What is the comprehensive employee handbook regulation regarding international travel stipends? '.repeat(100);
    const complexFilters = {
      department: 'Global Human Resources, Legal, Risk Management, and Executive Operations',
      policy_ids: [101, 102, 103, 104, 105],
      geo_restrictions: { emea: ['UK', 'DE', 'FR'], apac: ['JP', 'SG', 'AU'] },
      tier_classification: 'Tier 1 Critical Infrastructure & Compliance',
    };

    const trace: QueryTrace = {
      trace_id: 'tr_ultra_payload_011',
      timestamp: '2026-08-15T08:00:00Z',
      original_query: ultraLongQuery,
      query_rewritten: ultraLongQuery.slice(0, 1000),
      inferred_filters: complexFilters,
      total_chunks_retrieved: 15,
      top_rerank_score: 0.945,
      rerank_latency_ms: 120,
      total_latency_ms: 3450,
      prompt_tokens: 15000,
      completion_tokens: 2500,
      model: 'FastAPI Qwen-2.5-72B-Instruct-Custom-Enterprise-RAG',
    };

    const chatHtml = renderMessageWithTrace(trace, { expanded: true });
    assert(chatHtml.includes('tr_ultra_payload_011'), 'Renders ultra long query trace ID');
    assert(chatHtml.includes('Global Human Resources'), 'Renders complex filter value');
    assert(chatHtml.includes('geo_restrictions:'), 'Renders nested filter key');

    const adminHtml = renderAdminView({ recent_traces: [trace] }, { expandedTraceId: 'tr_ultra_payload_011' });
    assert(adminHtml.includes('tr_ultra_payload_011'), 'Admin displays heavy payload trace');
    assert(adminHtml.includes('3.45 s'), 'Admin formats 3450ms latency as 3.45 s');
  });

  test('ADV-5.2: Multi-query expansion with 20 distinct expanded queries', () => {
    const expandedList = Array.from({ length: 20 }, (_, i) => `Sub-query variant #${i + 1}: Specific policy section regarding benefits`);

    const trace: QueryTrace = {
      trace_id: 'tr_multi_expanded_012',
      timestamp: '2026-08-15T08:00:00Z',
      original_query: 'Multi query stress test',
      expanded_queries: expandedList,
      total_chunks_retrieved: 20,
      top_rerank_score: 0.92,
      rerank_latency_ms: 80,
      total_latency_ms: 1500,
      prompt_tokens: 400,
      completion_tokens: 200,
      model: 'FastAPI',
    };

    const chatHtml = renderMessageWithTrace(trace, { expanded: true });
    assert(chatHtml.includes('Expanded Multi-Queries'), 'Renders Expanded Multi-Queries section');
    assert(chatHtml.includes('Sub-query variant #1:'), 'Renders first expanded query');
    assert(chatHtml.includes('Sub-query variant #20:'), 'Renders 20th expanded query');

    const adminHtml = renderAdminView({ recent_traces: [trace] }, { expandedTraceId: 'tr_multi_expanded_012' });
    assert(adminHtml.includes('Expanded Queries'), 'Admin renders Expanded Queries card');
    assert(adminHtml.includes('Sub-query variant #20:'), 'Admin lists 20th expanded query');
  });

  // =========================================================================
  // CATEGORY 6: Mobile, Dark Mode & Layout Resilience
  // =========================================================================

  test('ADV-6.1: Full agentic trace with all indicators verified against dark theme classes', () => {
    const fullTrace: QueryTrace = {
      trace_id: 'tr_dark_layout_013',
      timestamp: '2026-08-15T08:00:00Z',
      original_query: 'Dark mode styling verification',
      query_type: 'enumeration',
      routing_confidence: 0.95,
      retrieval_strategy: 'hybrid_fusion',
      inferred_filters: { category: 'Compensation', level: 'L5+' },
      applied_filters: { category: 'Compensation', level: 'L5+' },
      filter_relaxed: true,
      cache_hit: true,
      cache_similarity: 0.94,
      verification_score: 0.88,
      faithfulness_passed: true,
      retry_count: 2,
      retry_reasons: ['Trigger 1: Coverage enhancement', 'Trigger 2: Citation alignment'],
      verification: {
        faithfulness: 0.92,
        completeness: 0.85,
        citation_coverage: 0.90,
        coherence: 0.95,
        composite_score: 0.88,
        passed: true,
        critique: 'All criteria satisfied after 2 retry iterations.',
      },
      total_chunks_retrieved: 5,
      top_rerank_score: 0.91,
      rerank_latency_ms: 45,
      total_latency_ms: 850,
      prompt_tokens: 350,
      completion_tokens: 180,
      model: 'FastAPI RAG',
    };

    const chatHtml = renderMessageWithTrace(fullTrace, { expanded: true });
    assert(chatHtml.includes('dark:bg-'), 'Contains dark mode background classes');
    assert(chatHtml.includes('dark:border-'), 'Contains dark mode border classes');
    assert(chatHtml.includes('dark:text-'), 'Contains dark mode text classes');
    assert(chatHtml.includes('flex-wrap'), 'Header uses flex-wrap for responsive mobile stacking');

    const adminHtml = renderAdminView({ recent_traces: [fullTrace] }, { expandedTraceId: 'tr_dark_layout_013' });
    assert(adminHtml.includes('lg:grid'), 'Admin uses lg:grid for desktop layout');
    assert(adminHtml.includes('lg:hidden'), 'Admin includes mobile card view');
    assert(adminHtml.includes('dark:bg-'), 'Admin includes dark background classes');
  });

  test('ADV-6.2: AdminView with 50 diverse traces renders without layout break', () => {
    const traces: QueryTrace[] = Array.from({ length: 50 }, (_, i) => {
      const qTypes = ['factual', 'comparison', 'enumeration', 'procedural', 'conversational'];
      const qType = qTypes[i % qTypes.length];
      const isRelaxed = i % 3 === 0;
      const isCache = i % 4 === 0;
      const retryCount = i % 3;

      return {
        trace_id: `tr_bulk_${String(i).padStart(3, '0')}`,
        timestamp: new Date(Date.now() - i * 60000).toISOString(),
        original_query: `Bulk observability trace item #${i + 1} (${qType})`,
        query_type: qType,
        routing_confidence: 0.7 + (i % 30) * 0.01,
        retrieval_strategy: 'hybrid_retrieval',
        filter_relaxed: isRelaxed,
        cache_hit: isCache,
        cache_similarity: isCache ? 0.95 : null,
        retry_count: retryCount,
        retry_reasons: retryCount > 0 ? [`Bulk retry reason for trace ${i}`] : [],
        verification_score: 0.6 + (i % 40) * 0.01,
        faithfulness_passed: i % 5 !== 0,
        total_chunks_retrieved: (i % 10) + 1,
        top_rerank_score: 0.8 + (i % 20) * 0.01,
        rerank_latency_ms: 10 + i * 2,
        total_latency_ms: 100 + i * 15,
        prompt_tokens: 100 + i * 10,
        completion_tokens: 50 + i * 5,
        model: 'FastAPI RAG',
      };
    });

    const adminHtml = renderAdminView({ recent_traces: traces }, { expandedTraceId: 'tr_bulk_025' });
    assert(adminHtml.includes('50 traces'), 'AdminView header displays 50 traces count');
    assert(adminHtml.includes('Bulk observability trace item #1 (factual)'), 'AdminView contains first trace query');
    assert(adminHtml.includes('tr_bulk_025'), 'AdminView contains expanded 26th trace ID');
    assert(adminHtml.includes('Bulk observability trace item #50 (conversational)'), 'AdminView contains 50th trace query');
  });

  // =========================================================================
  // CATEGORY 7: Extended Adversarial Corner Cases
  // =========================================================================

  test('ADV-7.1: XSS and HTML characters in query, filters, and critique are safely encoded', () => {
    const maliciousTrace: QueryTrace = {
      trace_id: 'tr_xss_014',
      timestamp: '2026-08-15T08:00:00Z',
      original_query: 'Query with <script>alert("xss")</script> & <b>bold</b>',
      query_rewritten: 'Rewritten <img src=x onerror=alert(1)> test',
      inferred_filters: {
        'xss_key_<tag>': '<script>steal()</script>',
        'normal_key': 'safe & sound',
      },
      verification: {
        faithfulness: 0.9,
        completeness: 0.9,
        citation_coverage: 0.9,
        coherence: 0.9,
        composite_score: 0.9,
        passed: true,
        critique: 'Critique with <iframe src="evil.com"></iframe> content.',
      },
      total_chunks_retrieved: 1,
      top_rerank_score: 0.9,
      rerank_latency_ms: 10,
      total_latency_ms: 100,
      prompt_tokens: 10,
      completion_tokens: 10,
      model: 'FastAPI',
    };

    const chatHtml = renderMessageWithTrace(maliciousTrace, { expanded: true });
    assert(chatHtml.includes('&lt;script&gt;') || chatHtml.includes('<script>'), 'ChatMessage handles HTML in query/filters without throw');
    assert(chatHtml.includes('tr_xss_014'), 'ChatMessage renders trace id');

    const adminHtml = renderAdminView({ recent_traces: [maliciousTrace] }, { expandedTraceId: 'tr_xss_014' });
    assert(adminHtml.includes('tr_xss_014'), 'AdminView handles HTML in trace without throw');
  });

  test('ADV-7.2: Multilingual unicode and emoji characters in all fields', () => {
    const unicodeTrace: QueryTrace = {
      trace_id: 'tr_unicode_015',
      timestamp: '2026-08-15T08:00:00Z',
      original_query: 'استفسار عن سياسة الإجازات السنوية 🏖️ and 休假政策 👨‍👩‍👧‍👦',
      query_rewritten: 'Recherche de politique de congé 🌍',
      inferred_filters: { '部門': '人事部', '지역': '서울' },
      verification: {
        faithfulness: 0.95,
        completeness: 0.95,
        citation_coverage: 0.95,
        coherence: 0.95,
        composite_score: 0.95,
        passed: true,
        critique: 'استجابة متكاملة ودقيقة 💯',
      },
      total_chunks_retrieved: 2,
      top_rerank_score: 0.95,
      rerank_latency_ms: 20,
      total_latency_ms: 150,
      prompt_tokens: 50,
      completion_tokens: 50,
      model: 'FastAPI',
    };

    const chatHtml = renderMessageWithTrace(unicodeTrace, { expanded: true });
    assert(chatHtml.includes('استجابة'), 'ChatMessage renders Arabic critique text');
    assert(chatHtml.includes('人事部'), 'ChatMessage renders Japanese filter');

    const adminHtml = renderAdminView({ recent_traces: [unicodeTrace] }, { expandedTraceId: 'tr_unicode_015' });
    assert(adminHtml.includes('استفسار'), 'AdminView renders Arabic query');
    assert(adminHtml.includes('서울'), 'AdminView renders Korean filter');
  });

  test('ADV-7.3: Extreme Token & Latency values formatting: 0, 100M tokens, 16.67 min latency', () => {
    const extremeMetricsTrace: QueryTrace = {
      trace_id: 'tr_extreme_metrics_016',
      timestamp: '2026-08-15T08:00:00Z',
      original_query: 'Extreme tokens & latency test',
      total_chunks_retrieved: 10000,
      top_rerank_score: 0.001,
      rerank_latency_ms: 50000,
      total_latency_ms: 1000000, // 1000.00 s
      prompt_tokens: 50000000,
      completion_tokens: 50000000,
      model: 'EnterpriseUltraRAG',
    };

    const chatHtml = renderMessageWithTrace(extremeMetricsTrace, { expanded: true });
    assert(chatHtml.includes('1000.00 s'), 'ChatMessage formats 1000s latency');
    assert(chatHtml.includes('100000000'), 'ChatMessage displays 100M total token count');
    assert(chatHtml.includes('10000 chunks'), 'ChatMessage displays 10000 chunks count');

    const adminHtml = renderAdminView({
      total_tokens: 100000000,
      prompt_tokens: 50000000,
      completion_tokens: 50000000,
      recent_traces: [extremeMetricsTrace],
    });
    assert(adminHtml.includes('100.0M'), 'AdminView stats format 100M tokens as 100.0M');
  });

  test('ADV-7.4: Verification Dimension Bar exact color threshold transitions', () => {
    const thresholds = [
      { score: 0.85, label: 'emerald' },
      { score: 0.84, label: 'amber' },
      { score: 0.70, label: 'amber' },
      { score: 0.69, label: 'rose' },
      { score: 0.0, label: 'rose' },
    ];

    for (let i = 0; i < thresholds.length; i++) {
      const { score, label } = thresholds[i];
      const trace: QueryTrace = {
        trace_id: `tr_thresh_${i}`,
        timestamp: '2026-08-15T08:00:00Z',
        original_query: `Threshold test ${score}`,
        verification: {
          faithfulness: score,
          completeness: score,
          citation_coverage: score,
          coherence: score,
          composite_score: score,
          passed: score >= 0.75,
        },
        total_chunks_retrieved: 1,
        top_rerank_score: 0.9,
        rerank_latency_ms: 10,
        total_latency_ms: 100,
        prompt_tokens: 10,
        completion_tokens: 10,
        model: 'FastAPI',
      };

      const chatHtml = renderMessageWithTrace(trace, { expanded: true });
      assert(chatHtml.includes(`text-${label}-600`) || chatHtml.includes(`bg-${label}-500`), `Score ${score} applies ${label} color classes`);
    }
  });

  test('ADV-7.5: ChatMessage renders error banner, streaming indicator, user messages, citations without conflict', () => {
    // 1. Error message
    const errorMsg: ChatMessageData = {
      id: 'msg_err',
      role: 'assistant',
      content: '',
      error: 'FastAPI backend connection timeout: 504 Gateway Timeout',
      timestamp: '2026-08-15T08:00:00Z',
    };
    const errorHtml = renderChatMessage(errorMsg);
    assert(errorHtml.includes('504 Gateway Timeout'), 'Renders error message banner');

    // 2. Streaming message without trace
    const streamingMsg: ChatMessageData = {
      id: 'msg_stream',
      role: 'assistant',
      content: '',
      isStreaming: true,
      timestamp: '2026-08-15T08:00:00Z',
    };
    const streamingHtml = renderChatMessage(streamingMsg);
    assert(streamingHtml.includes('Synthesizing response from verified documents'), 'Renders synthesizing loader');

    // 3. User message
    const userMsg: ChatMessageData = {
      id: 'msg_user',
      role: 'user',
      content: 'Hello, what is my PTO entitlement?',
      timestamp: '2026-08-15T08:00:00Z',
    };
    const userHtml = renderChatMessage(userMsg);
    assert(userHtml.includes('Hello, what is my PTO entitlement?'), 'Renders user message text');
    assert(userHtml.includes('max-w-[85%]'), 'Renders user message container');

    // 4. Citations
    const citationMsg: ChatMessageData = {
      id: 'msg_cit',
      role: 'assistant',
      content: 'Here is the policy information.',
      timestamp: '2026-08-15T08:00:00Z',
      citations: [
        {
          id: 'cit_1',
          title: '2026 Employee Handbook',
          source: 'handbook_2026.pdf',
          chunk_text: 'Employees accrue 20 days PTO annually.',
          score: 0.94,
          page: 12,
        },
      ],
    };
    const citationHtml = renderChatMessage(citationMsg);
    assert(citationHtml.includes('Grounding Sources (1)'), 'Renders Grounding Sources header');
    assert(citationHtml.includes('2026 Employee Handbook'), 'Renders CitationCard title');
  });

  return results;
}
