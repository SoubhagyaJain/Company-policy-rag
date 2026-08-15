/**
 * Tier 4 Real-World Workload Scenarios Test Suite for Agentic Intelligence UI Indicators.
 *
 * Requirements & Specifications:
 * - TEST_INFRA.md § Coverage Goals: Tier 4 (>= 6 comprehensive realistic workloads)
 */

import {
  QueryTrace,
  VerificationReport,
  ChatMessageData,
  Citation,
  ObservabilityData,
} from '../lib/types';
import {
  mapTrace,
  ApiClient,
} from '../lib/api-client';
import {
  renderChatMessage,
  renderAdminView,
  assert,
} from './test_helpers';
import { TestResult } from './tier1_features.test';

export function runTier4Tests(): TestResult[] {
  const results: TestResult[] = [];

  function test(name: string, fn: () => void) {
    const start = performance.now();
    try {
      fn();
      results.push({
        suite: 'Tier 4: Real-World Workloads',
        name,
        passed: true,
        durationMs: performance.now() - start,
      });
    } catch (err: any) {
      results.push({
        suite: 'Tier 4: Real-World Workloads',
        name,
        passed: false,
        error: err?.message || String(err),
        durationMs: performance.now() - start,
      });
    }
  }

  // =========================================================================
  // SCENARIO 1: Factual HR Vacation Policy Query
  // =========================================================================
  test('Scenario 1: Factual HR Vacation Policy Query E2E Pipeline', () => {
    const client = new ApiClient();
    let accumulatedContent = '';
    const citations: Citation[] = [];
    let finalTrace: QueryTrace | null = null;

    const rawTracePayload = {
      trace_id: 'tr_hr_vacation_2026',
      timestamp: '2026-08-15T08:15:00.000Z',
      query: 'What is the standard annual vacation allowance for full-time employees?',
      rewritten_query: 'full-time employee annual paid vacation days allowance policy',
      sub_queries: ['vacation accrual schedule', 'tenure tier vacation days'],
      candidate_count: 4,
      top_rerank_score: 0.965,
      rerank_latency_ms: 42,
      execution_time_ms: 290,
      token_usage: { prompt_tokens: 140, completion_tokens: 85, total_tokens: 225 },
      model: 'FastAPI Qwen2.5',
      query_type: 'factual',
      routing_confidence: 0.96,
      retrieval_strategy: 'dense_bm25_hybrid',
      inferred_filters: { department: 'HR', document_type: 'policy' },
      applied_filters: { department: 'HR', document_type: 'policy' },
      filter_relaxed: false,
      verification_score: 0.96,
      faithfulness_passed: true,
      retry_count: 0,
      retry_reasons: [],
      cache_hit: false,
      cache_similarity: null,
      verification: {
        faithfulness: 0.98,
        completeness: 0.95,
        citation_coverage: 0.96,
        coherence: 0.97,
        composite_score: 0.96,
        passed: true,
        critique: 'Directly verified against Section 2.1 Employee Handbook.',
        missing_aspects: [],
        unsupported_claims: [],
        retry_count: 0,
      },
    };

    // 1. Simulate SSE stream events
    const callbacks = {
      onChunk: (chunk: string) => {
        accumulatedContent += chunk;
      },
      onCitation: (c: Citation) => {
        citations.push(c);
      },
      onTrace: (t: QueryTrace) => {
        finalTrace = t;
      },
    };

    (client as any).handleStreamEvent('start', { session_id: 'sess_hr_01', message_id: 'msg_01' }, callbacks);
    (client as any).handleStreamEvent('chunk', 'Full-time employees receive 20 days of paid vacation per calendar year.', callbacks);
    (client as any).handleStreamEvent(
      'citation',
      {
        chunk_id: 'c_hr_vac_01',
        source_file: 'Employee_Handbook_2026.pdf',
        section_title: 'Section 2.1 - Vacation Policy',
        snippet: 'All full-time exempt and non-exempt staff accrue 1.67 days per month (20 days annually).',
        relevance_score: 0.965,
        category: 'HR',
      },
      callbacks
    );
    (client as any).handleStreamEvent('trace', { trace: rawTracePayload }, callbacks);
    (client as any).handleStreamEvent('done', { answer: accumulatedContent, retrieval_trace: rawTracePayload }, callbacks);

    // 2. Validate deserialized trace integrity
    assert(finalTrace !== null, 'Trace deserialized');
    assert((finalTrace as any).query_type === 'factual', 'query_type is factual');
    assert((finalTrace as any).verification_score === 0.96, 'verification_score is 0.96');
    assert((finalTrace as any).inferred_filters?.department === 'HR', 'department filter is HR');
    assert((finalTrace as any).retry_count === 0, '0 retries');

    // 3. Render Assistant ChatMessage
    const assistantMessage: ChatMessageData = {
      id: 'msg_hr_01',
      role: 'assistant',
      content: accumulatedContent,
      timestamp: '2026-08-15T08:15:00.000Z',
      citations,
      trace: finalTrace as any,
    };

    const html = renderChatMessage(assistantMessage, { expanded: true });

    // 4. Validate UI Indicators
    assert(html.includes('factual'), 'renders factual badge');
    assert(html.includes('(96%)') || html.includes('96%'), 'renders 96% routing confidence');
    assert(html.includes('96% Verified') || (html.includes('96%') && html.includes('Verified')), 'renders emerald 96% Verified pill');
    assert(html.includes('department:') && html.includes('HR'), 'renders department: HR chip');
    assert(html.includes('document_type:') && html.includes('policy'), 'renders document_type: policy chip');
    assert(html.includes('Faithfulness') && html.includes('98%'), 'renders Faithfulness 98% bar');
    assert(html.includes('Employee_Handbook_2026.pdf'), 'renders citation grounding card');
    assert(!html.includes('retry') && !html.includes('retries'), 'no retry badge when retry_count is 0');
  });

  // =========================================================================
  // SCENARIO 2: Cross-Department Benefits Comparison Query
  // =========================================================================
  test('Scenario 2: Cross-Department Benefits Comparison Query E2E Pipeline', () => {
    const rawTrace = {
      trace_id: 'tr_comp_stipend_02',
      timestamp: '2026-08-15T08:20:00.000Z',
      query: 'How does the remote work equipment stipend compare between Engineering and Sales?',
      rewritten_query: 'remote work home office equipment stipend Engineering vs Sales department policy comparison',
      sub_queries: [
        'Engineering remote hardware stipend limit',
        'Sales home office equipment allowance',
        'Eligibility criteria and reimbursement interval',
      ],
      candidate_count: 8,
      top_rerank_score: 0.942,
      rerank_latency_ms: 65,
      execution_time_ms: 480,
      token_usage: { prompt_tokens: 310, completion_tokens: 175, total_tokens: 485 },
      model: 'FastAPI Qwen2.5',
      query_type: 'comparison',
      routing_confidence: 0.92,
      retrieval_strategy: 'multi_query_comparison',
      inferred_filters: { topic: 'remote_work', category: 'stipend' },
      applied_filters: { topic: 'remote_work', category: 'stipend' },
      filter_relaxed: false,
      verification_score: 0.91,
      faithfulness_passed: true,
      retry_count: 0,
      verification: {
        faithfulness: 0.94,
        completeness: 0.89,
        citation_coverage: 0.91,
        coherence: 0.93,
        composite_score: 0.91,
        passed: true,
        critique: 'Comparison clearly outlines both Engineering ($1,500) and Sales ($1,000) allowances.',
      },
    };

    const trace = mapTrace(rawTrace);
    const message: ChatMessageData = {
      id: 'msg_comp_02',
      role: 'assistant',
      content: '### Comparison Summary\n- **Engineering**: $1,500 initial setup stipend + $500 annual refresh.\n- **Sales**: $1,000 initial setup stipend + mobile phone allowance.',
      timestamp: '2026-08-15T08:20:00.000Z',
      trace,
    };

    const html = renderChatMessage(message, { expanded: true });

    assert(html.includes('comparison'), 'renders purple comparison badge');
    assert(html.includes('(92%)') || html.includes('92%'), 'renders 92% confidence');
    assert(html.includes('91% Verified') || (html.includes('91%') && html.includes('Verified')), 'renders 91% Verified');
    assert(html.includes('Engineering remote hardware stipend limit'), 'renders subquery 1');
    assert(html.includes('Sales home office equipment allowance'), 'renders subquery 2');
    assert(html.includes('topic:') && html.includes('remote_work'), 'renders topic tag');
    assert(html.includes('category:') && html.includes('stipend'), 'renders category tag');
    assert(html.includes('Completeness') && html.includes('89%'), 'renders completeness 89%');
  });

  // =========================================================================
  // SCENARIO 3: Procedural IT Equipment Request with Verification Retries
  // =========================================================================
  test('Scenario 3: Procedural IT Equipment Request with Verification Retries', () => {
    const rawTrace = {
      trace_id: 'tr_proc_it_03',
      timestamp: '2026-08-15T08:25:00.000Z',
      query: 'What are the step-by-step procedures to request a replacement laptop under IT Policy?',
      rewritten_query: 'replacement laptop request procedure ServiceNow steps approval workflow',
      sub_queries: ['ServiceNow IT hardware request form', 'Manager approval SLA and shipping timeline'],
      candidate_count: 6,
      top_rerank_score: 0.915,
      rerank_latency_ms: 55,
      execution_time_ms: 720,
      token_usage: { prompt_tokens: 290, completion_tokens: 160, total_tokens: 450 },
      model: 'FastAPI Qwen2.5',
      query_type: 'procedural',
      routing_confidence: 0.89,
      retrieval_strategy: 'procedural_graph_search',
      inferred_filters: { department: 'IT', service: 'Hardware_Support' },
      applied_filters: { department: 'IT' },
      filter_relaxed: false,
      verification_score: 0.88,
      faithfulness_passed: true,
      retry_count: 2,
      retry_reasons: [
        'Initial revision omitted Manager Approval SLA (Section 3.2)',
        'Second revision lacked hardware return shipping instructions',
      ],
      verification: {
        faithfulness: 0.92,
        completeness: 0.86,
        citation_coverage: 0.85,
        coherence: 0.94,
        composite_score: 0.88,
        passed: true,
        critique: 'Second verification retry successfully integrated both approval SLA and return shipping logistics.',
        missing_aspects: [],
        unsupported_claims: [],
        retry_count: 2,
      },
    };

    const trace = mapTrace(rawTrace);
    const message: ChatMessageData = {
      id: 'msg_proc_03',
      role: 'assistant',
      content: '1. Navigate to ServiceNow IT Portal\n2. Submit Hardware Request form\n3. Manager approval within 48 hours\n4. Device dispatched via FedEx with prepaid return box.',
      timestamp: '2026-08-15T08:25:00.000Z',
      trace,
    };

    const html = renderChatMessage(message, { expanded: true });

    assert(html.includes('procedural'), 'renders teal procedural badge');
    assert(html.includes('2 retries'), 'renders amber 2 retries badge');
    assert(html.includes('88% Verified') || (html.includes('88%') && html.includes('Verified')), 'renders 88% Verified pill');
    assert(html.includes('Initial revision omitted Manager Approval SLA'), 'renders retry trigger reason 1');
    assert(html.includes('hardware return shipping instructions'), 'renders retry trigger reason 2');
    assert(html.includes('Second verification retry successfully integrated'), 'renders critique');
    assert(html.includes('department:') && html.includes('IT'), 'renders IT filter chip');
  });

  // =========================================================================
  // SCENARIO 4: Enumeration with Filter Fallback/Relaxation
  // =========================================================================
  test('Scenario 4: Enumeration with Filter Fallback/Relaxation', () => {
    const rawTrace = {
      trace_id: 'tr_enum_legal_04',
      timestamp: '2026-08-15T08:30:00.000Z',
      query: 'List all restricted software applications for external contractors under Legal Dept',
      candidate_count: 12,
      top_rerank_score: 0.885,
      rerank_latency_ms: 60,
      execution_time_ms: 540,
      token_usage: { prompt_tokens: 350, completion_tokens: 210, total_tokens: 560 },
      model: 'FastAPI Qwen2.5',
      query_type: 'enumeration',
      routing_confidence: 0.87,
      retrieval_strategy: 'exhaustive_chunk_scan',
      inferred_filters: { department: 'Legal', audience: 'Contractor_External_Restricted_Level_5' },
      applied_filters: {},
      filter_relaxed: true,
      verification_score: 0.84,
      faithfulness_passed: true,
      retry_count: 1,
      retry_reasons: ['Broadened retrieval to general IT software security policy'],
      verification: {
        faithfulness: 0.88,
        completeness: 0.80,
        citation_coverage: 0.84,
        coherence: 0.90,
        composite_score: 0.84,
        passed: true,
        critique: 'Filtered search returned 0 documents; retrieved company-wide contractor policy instead.',
      },
    };

    const trace = mapTrace(rawTrace);
    const message: ChatMessageData = {
      id: 'msg_enum_04',
      role: 'assistant',
      content: 'Restricted software includes: 1. Unapproved cloud storage, 2. Peer-to-peer file sharing, 3. Non-enterprise password managers.',
      timestamp: '2026-08-15T08:30:00.000Z',
      trace,
    };

    const html = renderChatMessage(message, { expanded: true });

    assert(html.includes('enumeration'), 'renders enumeration badge');
    assert(html.includes('Filters Relaxed:'), 'renders amber filter relaxation warning callout');
    assert(html.includes('Filtered search returned 0 results'), 'callout contains explanation text');
    assert(html.includes('department:') && html.includes('Legal'), 'displays inferred department tag');
    assert(html.includes('Contractor_External_Restricted_Level_5'), 'displays inferred audience tag');
    assert(html.includes('1 retry'), 'displays 1 retry badge');
  });

  // =========================================================================
  // SCENARIO 5: Conversational Greeting / General Chat (Cache Hit)
  // =========================================================================
  test('Scenario 5: Conversational Greeting / General Chat (Cache Hit)', () => {
    const rawTrace = {
      trace_id: 'tr_conv_cache_05',
      timestamp: '2026-08-15T08:35:00.000Z',
      query: 'Hello, good morning! Can you help me understand our company policies?',
      candidate_count: 0,
      top_rerank_score: 0.99,
      rerank_latency_ms: 0,
      execution_time_ms: 45,
      token_usage: { prompt_tokens: 45, completion_tokens: 35, total_tokens: 80 },
      model: 'FastAPI Qwen2.5',
      query_type: 'conversational',
      routing_confidence: 0.99,
      retrieval_strategy: 'conversational_bypass',
      inferred_filters: {},
      applied_filters: {},
      filter_relaxed: false,
      verification_score: 1.0,
      faithfulness_passed: true,
      retry_count: 0,
      cache_hit: true,
      cache_similarity: 0.985,
      verification: {
        faithfulness: 1.0,
        completeness: 1.0,
        citation_coverage: 1.0,
        coherence: 1.0,
        composite_score: 1.0,
        passed: true,
      },
    };

    const trace = mapTrace(rawTrace);
    const message: ChatMessageData = {
      id: 'msg_conv_05',
      role: 'assistant',
      content: 'Good morning! I am your AI Policy Assistant. Feel free to ask any question regarding HR, benefits, travel, or IT guidelines.',
      timestamp: '2026-08-15T08:35:00.000Z',
      trace,
    };

    const html = renderChatMessage(message, { expanded: true });

    assert(html.includes('conversational'), 'renders terracotta conversational badge');
    assert(html.includes('Cache Hit'), 'renders sky Cache Hit badge');
    assert(html.includes('99%') || html.includes('98%') || html.includes('(99%)'), 'renders similarity / confidence');
    assert(html.includes('100% Verified') || (html.includes('100%') && html.includes('Verified')), 'renders 100% Verified pill');
    assert(html.includes('45 ms') || html.includes('Thought for 45 ms') || html.includes('45ms'), 'renders fast latency banner');
  });

  // =========================================================================
  // SCENARIO 6: Admin Observability Full Audit & Trace Inspection
  // =========================================================================
  test('Scenario 6: Admin Observability Full Audit & Trace Inspection', () => {
    const traces: QueryTrace[] = [
      mapTrace({
        trace_id: 'tr_hr_vacation_2026',
        timestamp: '2026-08-15T08:15:00.000Z',
        query: 'What is the standard annual vacation allowance for full-time employees?',
        total_chunks_retrieved: 4,
        top_rerank_score: 0.965,
        total_latency_ms: 290,
        prompt_tokens: 140,
        completion_tokens: 85,
        model: 'FastAPI Qwen2.5',
        query_type: 'factual',
        routing_confidence: 0.96,
        inferred_filters: { department: 'HR' },
        verification_score: 0.96,
        faithfulness_passed: true,
      }),
      mapTrace({
        trace_id: 'tr_comp_stipend_02',
        timestamp: '2026-08-15T08:20:00.000Z',
        query: 'How does remote work stipend compare between Engineering and Sales?',
        total_chunks_retrieved: 8,
        top_rerank_score: 0.942,
        total_latency_ms: 480,
        prompt_tokens: 310,
        completion_tokens: 175,
        model: 'FastAPI Qwen2.5',
        query_type: 'comparison',
        routing_confidence: 0.92,
        inferred_filters: { topic: 'remote_work' },
        verification_score: 0.91,
        faithfulness_passed: true,
      }),
      mapTrace({
        trace_id: 'tr_proc_it_03',
        timestamp: '2026-08-15T08:25:00.000Z',
        query: 'Replacement laptop request procedure steps',
        total_chunks_retrieved: 6,
        top_rerank_score: 0.915,
        total_latency_ms: 720,
        prompt_tokens: 290,
        completion_tokens: 160,
        model: 'FastAPI Qwen2.5',
        query_type: 'procedural',
        routing_confidence: 0.89,
        retry_count: 2,
        retry_reasons: ['Reason 1', 'Reason 2'],
        inferred_filters: { department: 'IT' },
        verification_score: 0.88,
        faithfulness_passed: true,
      }),
      mapTrace({
        trace_id: 'tr_enum_legal_04',
        timestamp: '2026-08-15T08:30:00.000Z',
        query: 'Restricted software applications list',
        total_chunks_retrieved: 12,
        top_rerank_score: 0.885,
        total_latency_ms: 540,
        prompt_tokens: 350,
        completion_tokens: 210,
        model: 'FastAPI Qwen2.5',
        query_type: 'enumeration',
        routing_confidence: 0.87,
        inferred_filters: { department: 'Legal' },
        filter_relaxed: true,
        verification_score: 0.84,
        faithfulness_passed: true,
      }),
      mapTrace({
        trace_id: 'tr_conv_cache_05',
        timestamp: '2026-08-15T08:35:00.000Z',
        query: 'Hello, good morning policy assistant',
        total_chunks_retrieved: 0,
        top_rerank_score: 0.99,
        total_latency_ms: 45,
        prompt_tokens: 45,
        completion_tokens: 35,
        model: 'FastAPI Qwen2.5',
        query_type: 'conversational',
        routing_confidence: 0.99,
        cache_hit: true,
        cache_similarity: 0.985,
        verification_score: 1.0,
        faithfulness_passed: true,
      }),
    ];

    const obsData: ObservabilityData = {
      total_queries: 5,
      avg_latency_ms: 415,
      avg_ttft_ms: 190,
      p95_latency_ms: 720,
      prompt_tokens: 1135,
      completion_tokens: 665,
      total_tokens: 1800,
      active_documents: 14,
      indexed_chunks: 340,
      similarity_avg: 0.92,
      rerank_avg: 0.94,
      health: {
        status: 'ok',
        redis: true,
        vector_db: true,
        models_loaded: true,
        backend_version: 'FastAPI RAG v1.0',
      },
      recent_traces: traces,
    };

    const html = renderAdminView(obsData, { expandedTraceId: 'tr_proc_it_03' });

    assert(html.includes('Observability &amp; Telemetry') || html.includes('Observability & Telemetry'), 'AdminView rendered');
    assert(html.includes('Factual'), 'Factual chip in table');
    assert(html.includes('Comparison'), 'Comparison chip in table');
    assert(html.includes('Procedural'), 'Procedural chip in table');
    assert(html.includes('Enumeration'), 'Enumeration chip in table');
    assert(html.includes('Conversational'), 'Conversational chip in table');
    assert(html.includes('Verification Retries (2)'), 'Expanded detail shows retries for procedural trace');
  });

  return results;
}
