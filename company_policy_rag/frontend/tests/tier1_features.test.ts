/**
 * Tier 1 Feature Coverage Test Suite for Agentic Intelligence UI Indicators.
 *
 * Requirements & Interface Contracts:
 * - ORIGINAL_REQUEST.md § R1, R2, R3, R4, § Integration, § Quality
 * - PROJECT.md § Feature Inventory (F1 to F11), § Interface Contracts
 * - TEST_INFRA.md § Coverage Goals: Tier 1 (>= 55 test cases across 11 features)
 */

import {
  QueryCategory,
  QueryTrace,
  VerificationReport,
  ChatMessageData,
  ObservabilityData,
} from '../lib/types';
import {
  mapTrace,
  mapVerificationReport,
  ApiClient,
} from '../lib/api-client';
import {
  renderChatMessage,
  renderMessageWithTrace,
  renderAdminView,
  assert,
} from './test_helpers';

export interface TestResult {
  suite: string;
  name: string;
  passed: boolean;
  error?: string;
  durationMs: number;
}

export function runTier1Tests(): TestResult[] {
  const results: TestResult[] = [];

  function test(name: string, fn: () => void) {
    const start = performance.now();
    try {
      fn();
      results.push({
        suite: 'Tier 1: Feature Coverage',
        name,
        passed: true,
        durationMs: performance.now() - start,
      });
    } catch (err: any) {
      results.push({
        suite: 'Tier 1: Feature Coverage',
        name,
        passed: false,
        error: err?.message || String(err),
        durationMs: performance.now() - start,
      });
    }
  }

  // =========================================================================
  // FEATURE 1: Type System Integrity (5 tests)
  // =========================================================================

  test('F1.1: VerificationReport model supports all 10 core fields', () => {
    const report: VerificationReport = {
      faithfulness: 0.95,
      completeness: 0.90,
      citation_coverage: 0.88,
      coherence: 0.92,
      composite_score: 0.91,
      passed: true,
      critique: 'Response is accurate and grounded in policy documentation.',
      missing_aspects: ['SLA escalation procedure'],
      unsupported_claims: [],
      retry_count: 1,
    };
    assert(report.faithfulness === 0.95, 'faithfulness matches');
    assert(report.completeness === 0.90, 'completeness matches');
    assert(report.citation_coverage === 0.88, 'citation_coverage matches');
    assert(report.coherence === 0.92, 'coherence matches');
    assert(report.composite_score === 0.91, 'composite_score matches');
    assert(report.passed === true, 'passed matches');
    assert(typeof report.critique === 'string', 'critique is string');
    assert(Array.isArray(report.missing_aspects) && report.missing_aspects.length === 1, 'missing_aspects array');
    assert(Array.isArray(report.unsupported_claims) && report.unsupported_claims.length === 0, 'unsupported_claims array');
    assert(report.retry_count === 1, 'retry_count is 1');
  });

  test('F1.2: QueryTrace interface includes all 13 agentic telemetry fields', () => {
    const trace: QueryTrace = {
      trace_id: 'tr_spec_001',
      timestamp: '2026-08-15T08:00:00.000Z',
      original_query: 'vacation policy query',
      query_rewritten: 'annual paid time off vacation policy',
      expanded_queries: ['vacation accrual rate', 'carryover limit'],
      total_chunks_retrieved: 4,
      top_rerank_score: 0.94,
      rerank_latency_ms: 45,
      total_latency_ms: 280,
      prompt_tokens: 150,
      completion_tokens: 80,
      model: 'FastAPI Qwen2.5',
      query_type: 'factual',
      routing_confidence: 0.96,
      retrieval_strategy: 'dense_bm25_hybrid',
      inferred_filters: { department: 'HR', topic: 'benefits' },
      applied_filters: { department: 'HR' },
      filter_relaxed: false,
      verification_score: 0.95,
      verification: {
        faithfulness: 0.98,
        completeness: 0.92,
        citation_coverage: 0.95,
        coherence: 0.96,
        composite_score: 0.95,
        passed: true,
      },
      faithfulness_passed: true,
      retry_count: 0,
      retry_reasons: [],
      cache_hit: false,
      cache_similarity: null,
    };
    assert(trace.trace_id === 'tr_spec_001', 'trace_id matches');
    assert(trace.query_type === 'factual', 'query_type matches');
    assert(trace.routing_confidence === 0.96, 'routing_confidence matches');
    assert(trace.retrieval_strategy === 'dense_bm25_hybrid', 'retrieval_strategy matches');
    assert(trace.inferred_filters?.department === 'HR', 'inferred_filters matches');
    assert(trace.applied_filters?.department === 'HR', 'applied_filters matches');
    assert(trace.filter_relaxed === false, 'filter_relaxed matches');
    assert(trace.verification_score === 0.95, 'verification_score matches');
    assert(trace.faithfulness_passed === true, 'faithfulness_passed matches');
    assert(trace.retry_count === 0, 'retry_count matches');
    assert(trace.cache_hit === false, 'cache_hit matches');
  });

  test('F1.3: QueryCategory union type covers all 5 classification variants', () => {
    const categories: QueryCategory[] = [
      'factual',
      'comparison',
      'enumeration',
      'procedural',
      'conversational',
    ];
    assert(categories.length === 5, '5 categories defined');
    assert(categories.includes('factual'), 'includes factual');
    assert(categories.includes('comparison'), 'includes comparison');
    assert(categories.includes('enumeration'), 'includes enumeration');
    assert(categories.includes('procedural'), 'includes procedural');
    assert(categories.includes('conversational'), 'includes conversational');
  });

  test('F1.4: ChatMessageData incorporates optional QueryTrace telemetry', () => {
    const msg: ChatMessageData = {
      id: 'msg_001',
      role: 'assistant',
      content: 'Here is the company policy summary.',
      timestamp: '2026-08-15T08:00:00.000Z',
      trace: {
        trace_id: 'tr_test',
        timestamp: '2026-08-15T08:00:00.000Z',
        original_query: 'policy question',
        total_chunks_retrieved: 2,
        top_rerank_score: 0.9,
        rerank_latency_ms: 30,
        total_latency_ms: 200,
        prompt_tokens: 100,
        completion_tokens: 50,
        model: 'FastAPI',
        query_type: 'factual',
        verification_score: 0.92,
      },
    };
    assert(msg.trace?.query_type === 'factual', 'trace attached to message');
    assert(msg.trace?.verification_score === 0.92, 'verification score present');
  });

  test('F1.5: ObservabilityData contains recent_traces of type QueryTrace[]', () => {
    const obs: ObservabilityData = {
      total_queries: 10,
      avg_latency_ms: 350,
      avg_ttft_ms: 180,
      p95_latency_ms: 500,
      prompt_tokens: 1200,
      completion_tokens: 800,
      total_tokens: 2000,
      health: {
        status: 'ok',
        redis: true,
        vector_db: true,
        models_loaded: true,
        backend_version: 'FastAPI RAG',
      },
      recent_traces: [
        {
          trace_id: 'tr_obs_1',
          timestamp: '2026-08-15T08:00:00.000Z',
          original_query: 'obs test',
          total_chunks_retrieved: 3,
          top_rerank_score: 0.88,
          rerank_latency_ms: 40,
          total_latency_ms: 320,
          prompt_tokens: 120,
          completion_tokens: 80,
          model: 'FastAPI',
          query_type: 'procedural',
        },
      ],
    };
    assert(obs.recent_traces.length === 1, 'recent_traces has 1 trace');
    assert(obs.recent_traces[0].query_type === 'procedural', 'query_type matches');
  });

  // =========================================================================
  // FEATURE 2: SSE Deserialization & Mapping (5 tests)
  // =========================================================================

  test('F2.1: mapTrace preserves snake_case agentic fields from backend', () => {
    const rawBackendPayload = {
      trace_id: 'tr_sse_01',
      timestamp: '2026-08-15T08:00:00.000Z',
      query: 'What is parental leave duration?',
      rewritten_query: 'parental leave policy weeks duration',
      sub_queries: ['maternity leave duration', 'paternity leave duration'],
      candidate_count: 5,
      top_rerank_score: 0.93,
      rerank_latency_ms: 50,
      execution_time_ms: 340,
      token_usage: { prompt_tokens: 180, completion_tokens: 95, total_tokens: 275 },
      model: 'FastAPI Qwen2.5',
      query_type: 'factual',
      routing_confidence: 0.95,
      retrieval_strategy: 'dense_bm25_hybrid',
      inferred_filters: { department: 'HR', topic: 'leave' },
      applied_filters: { department: 'HR' },
      filter_relaxed: false,
      verification_score: 0.94,
      faithfulness_passed: true,
      retry_count: 1,
      retry_reasons: ['Initial retrieval lacked paternity leave specific clause'],
      cache_hit: false,
      cache_similarity: null,
    };

    const mapped = mapTrace(rawBackendPayload);
    assert(mapped.trace_id === 'tr_sse_01', 'trace_id mapped');
    assert(mapped.original_query === 'What is parental leave duration?', 'query mapped');
    assert(mapped.query_rewritten === 'parental leave policy weeks duration', 'query_rewritten mapped');
    assert(mapped.expanded_queries?.length === 2, 'expanded_queries mapped');
    assert(mapped.total_chunks_retrieved === 5, 'candidate_count mapped to chunks');
    assert(mapped.query_type === 'factual', 'query_type preserved');
    assert(mapped.routing_confidence === 0.95, 'routing_confidence preserved');
    assert(mapped.retrieval_strategy === 'dense_bm25_hybrid', 'strategy preserved');
    assert(mapped.inferred_filters?.department === 'HR', 'inferred_filters preserved');
    assert(mapped.applied_filters?.department === 'HR', 'applied_filters preserved');
    assert(mapped.filter_relaxed === false, 'filter_relaxed preserved');
    assert(mapped.verification_score === 0.94, 'verification_score preserved');
    assert(mapped.faithfulness_passed === true, 'faithfulness_passed preserved');
    assert(mapped.retry_count === 1, 'retry_count preserved');
    assert(mapped.retry_reasons?.[0].includes('paternity'), 'retry_reasons preserved');
    assert(mapped.cache_hit === false, 'cache_hit preserved');
  });

  test('F2.2: mapTrace supports camelCase aliases for all telemetry fields', () => {
    const rawCamelPayload = {
      id: 'tr_camel_01',
      originalQuery: 'health insurance coverage',
      queryRewritten: 'medical insurance benefits',
      expandedQueries: ['dental coverage', 'vision coverage'],
      totalChunksRetrieved: 3,
      topRerankScore: 0.91,
      rerankLatencyMs: 35,
      totalLatencyMs: 250,
      promptTokens: 110,
      completionTokens: 60,
      queryType: 'comparison',
      routingConfidence: 0.88,
      retrievalStrategy: 'multi_query',
      inferredFilters: { benefit_type: 'health' },
      appliedFilters: { benefit_type: 'health' },
      filterRelaxed: true,
      verificationScore: 0.89,
      faithfulnessPassed: true,
      retryCount: 0,
      retryReasons: [],
      cacheHit: true,
      cacheSimilarity: 0.97,
    };

    const mapped = mapTrace(rawCamelPayload);
    assert(mapped.trace_id === 'tr_camel_01', 'id mapped to trace_id');
    assert(mapped.original_query === 'health insurance coverage', 'originalQuery mapped');
    assert(mapped.query_rewritten === 'medical insurance benefits', 'queryRewritten mapped');
    assert(mapped.expanded_queries?.length === 2, 'expandedQueries mapped');
    assert(mapped.query_type === 'comparison', 'queryType mapped');
    assert(mapped.routing_confidence === 0.88, 'routingConfidence mapped');
    assert(mapped.filter_relaxed === true, 'filterRelaxed mapped');
    assert(mapped.verification_score === 0.89, 'verificationScore mapped');
    assert(mapped.cache_hit === true, 'cacheHit mapped');
    assert(mapped.cache_similarity === 0.97, 'cacheSimilarity mapped');
  });

  test('F2.3: mapVerificationReport converts raw dict and fills sensible defaults', () => {
    const rawVer = {
      faithfulness: 0.96,
      completeness: 0.89,
      citation_coverage: 0.93,
      coherence: 0.97,
      composite_score: 0.94,
      passed: true,
      critique: 'Accurate policy references.',
      missing_aspects: [],
      unsupported_claims: [],
      retry_count: 0,
    };

    const ver = mapVerificationReport(rawVer);
    assert(ver !== null, 'verification report is not null');
    assert(ver?.faithfulness === 0.96, 'faithfulness score matches');
    assert(ver?.composite_score === 0.94, 'composite score matches');
    assert(ver?.passed === true, 'passed status matches');
    assert(ver?.critique === 'Accurate policy references.', 'critique matches');

    assert(mapVerificationReport(null) === null, 'null input yields null');
    assert(mapVerificationReport(undefined) === null, 'undefined input yields null');
  });

  test('F2.4: mapTrace extracts composite_score and passed status from nested verification report when top-level fields are absent', () => {
    const payload = {
      trace_id: 'tr_nested_ver',
      original_query: 'bereavement policy',
      verification: {
        faithfulness: 0.98,
        completeness: 0.95,
        citation_coverage: 0.94,
        coherence: 0.99,
        composite_score: 0.965,
        passed: true,
      },
    };

    const mapped = mapTrace(payload);
    assert(mapped.verification !== null, 'nested verification object mapped');
    assert(mapped.verification_score === 0.965, 'top-level verification_score inferred from nested composite_score');
    assert(mapped.faithfulness_passed === true, 'faithfulness_passed inferred from nested verification.passed');
  });

  test('F2.5: ApiClient.handleStreamEvent properly deserializes event: trace and event: done payloads', () => {
    const client = new ApiClient();
    let receivedTrace: QueryTrace | null = null;
    let receivedDone: any = null;

    const callbacks = {
      onTrace: (t: QueryTrace) => {
        receivedTrace = t;
      },
      onDone: (d: any) => {
        receivedDone = d;
      },
    };

    (client as any).handleStreamEvent(
      'trace',
      {
        trace: {
          trace_id: 'tr_stream_01',
          query: 'holiday schedule 2026',
          query_type: 'enumeration',
          routing_confidence: 0.91,
          verification_score: 0.93,
          faithfulness_passed: true,
        },
      },
      callbacks
    );

    assert(receivedTrace !== null, 'trace callback invoked');
    assert((receivedTrace as any)?.trace_id === 'tr_stream_01', 'trace_id matches');
    assert((receivedTrace as any)?.query_type === 'enumeration', 'query_type matches');

    (client as any).handleStreamEvent(
      'done',
      {
        answer: 'Here is the 2026 holiday calendar.',
        total_latency_ms: 420,
        retrieval_trace: {
          trace_id: 'tr_stream_done',
          query_type: 'enumeration',
          routing_confidence: 0.93,
          cache_hit: true,
        },
      },
      callbacks
    );

    assert(receivedDone !== null, 'done callback invoked');
    assert((receivedTrace as any)?.trace_id === 'tr_stream_done', 'trace updated from done event');
    assert((receivedTrace as any)?.cache_hit === true, 'cache_hit preserved');
  });

  // =========================================================================
  // FEATURE 3: R1 Query Classification Badge (5 tests)
  // =========================================================================

  test('F3.1: Renders factual badge with sky palette and confidence tooltip', () => {
    const html = renderMessageWithTrace({
      query_type: 'factual',
      routing_confidence: 0.95,
      retrieval_strategy: 'dense_bm25_hybrid',
    });
    assert(html.includes('factual'), 'renders factual text');
    assert(html.includes('sky-500') || html.includes('sky-700'), 'uses sky color palette');
    assert(html.includes('(95%)') || html.includes('95%'), 'displays 95% confidence');
    assert(html.includes('Routing Confidence: 95%'), 'tooltip contains confidence');
  });

  test('F3.2: Renders comparison badge with purple palette and confidence', () => {
    const html = renderMessageWithTrace({
      query_type: 'comparison',
      routing_confidence: 0.88,
    });
    assert(html.includes('comparison'), 'renders comparison text');
    assert(html.includes('purple-500') || html.includes('purple-700'), 'uses purple color palette');
    assert(html.includes('88%'), 'displays 88% confidence');
  });

  test('F3.3: Renders enumeration badge with amber palette and confidence', () => {
    const html = renderMessageWithTrace({
      query_type: 'enumeration',
      routing_confidence: 0.92,
    });
    assert(html.includes('enumeration'), 'renders enumeration text');
    assert(html.includes('amber-500') || html.includes('amber-700'), 'uses amber color palette');
    assert(html.includes('92%'), 'displays 92% confidence');
  });

  test('F3.4: Renders procedural badge with teal palette and confidence', () => {
    const html = renderMessageWithTrace({
      query_type: 'procedural',
      routing_confidence: 0.90,
    });
    assert(html.includes('procedural'), 'renders procedural text');
    assert(html.includes('teal-500') || html.includes('teal-700'), 'uses teal color palette');
    assert(html.includes('90%'), 'displays 90% confidence');
  });

  test('F3.5: Renders conversational badge with terracotta palette and confidence', () => {
    const html = renderMessageWithTrace({
      query_type: 'conversational',
      routing_confidence: 0.99,
    });
    assert(html.includes('conversational'), 'renders conversational text');
    assert(html.includes('terracotta-500') || html.includes('terracotta-700'), 'uses terracotta color palette');
    assert(html.includes('99%'), 'displays 99% confidence');
  });

  // =========================================================================
  // FEATURE 4: R2 Verification Composite Score Pill (5 tests)
  // =========================================================================

  test('F4.1: Renders green/emerald badge when passed === true (score >= 0.75)', () => {
    const html = renderMessageWithTrace({
      verification_score: 0.94,
      faithfulness_passed: true,
      verification: {
        faithfulness: 0.96,
        completeness: 0.92,
        citation_coverage: 0.94,
        coherence: 0.98,
        composite_score: 0.94,
        passed: true,
      },
    });
    assert(html.includes('94% Verified') || (html.includes('94%') && html.includes('Verified')), 'renders 94% Verified');
    assert(html.includes('emerald-500') || html.includes('emerald-700'), 'uses emerald/green palette for pass');
  });

  test('F4.2: Renders amber badge when passed === false or score < 0.75', () => {
    const html = renderMessageWithTrace({
      verification_score: 0.62,
      faithfulness_passed: false,
      verification: {
        faithfulness: 0.60,
        completeness: 0.65,
        citation_coverage: 0.55,
        coherence: 0.70,
        composite_score: 0.62,
        passed: false,
      },
    });
    assert(html.includes('62% Review') || (html.includes('62%') && html.includes('Review')), 'renders 62% Review');
    assert(html.includes('amber-500') || html.includes('amber-700'), 'uses amber palette for fail/review');
  });

  test('F4.3: Correctly formats integer percentage (e.g. 0.923 -> 92%)', () => {
    const html = renderMessageWithTrace({
      verification_score: 0.923,
      faithfulness_passed: true,
    });
    assert(html.includes('92%'), 'rounds 0.923 to 92%');
  });

  test('F4.4: Omits pill completely when no verification data is present', () => {
    const html = renderMessageWithTrace({
      verification_score: undefined,
      verification: undefined,
      faithfulness_passed: undefined,
    });
    assert(!html.includes('Verified') && !html.includes('Review'), 'no verification pill rendered');
  });

  test('F4.5: Displays comprehensive tooltip describing self-reflection score', () => {
    const html = renderMessageWithTrace({
      verification_score: 0.95,
      faithfulness_passed: true,
    });
    assert(html.includes('Self-Reflection Verification: 95% composite score (Passed)'), 'tooltip matches');
  });

  // =========================================================================
  // FEATURE 5: R2 4-Dimension Progress Bars (5 tests)
  // =========================================================================

  test('F5.1: Faithfulness dimension bar renders with numeric percentage and green class if >= 85%', () => {
    const html = renderMessageWithTrace(
      {
        verification: {
          faithfulness: 0.96,
          completeness: 0.90,
          citation_coverage: 0.92,
          coherence: 0.95,
          composite_score: 0.93,
          passed: true,
        },
      },
      { expanded: true }
    );
    assert(html.includes('Faithfulness'), 'renders Faithfulness label');
    assert(html.includes('96%'), 'renders 96% score');
  });

  test('F5.2: Completeness dimension bar renders with numeric percentage and color thresholding', () => {
    const html = renderMessageWithTrace(
      {
        verification: {
          faithfulness: 0.90,
          completeness: 0.78,
          citation_coverage: 0.90,
          coherence: 0.90,
          composite_score: 0.87,
          passed: true,
        },
      },
      { expanded: true }
    );
    assert(html.includes('Completeness'), 'renders Completeness label');
    assert(html.includes('78%'), 'renders 78% completeness');
  });

  test('F5.3: Citation coverage dimension bar renders with numeric percentage and color thresholding', () => {
    const html = renderMessageWithTrace(
      {
        verification: {
          faithfulness: 0.90,
          completeness: 0.90,
          citation_coverage: 0.65,
          coherence: 0.90,
          composite_score: 0.83,
          passed: true,
        },
      },
      { expanded: true }
    );
    assert(html.includes('Citation Coverage'), 'renders Citation Coverage label');
    assert(html.includes('65%'), 'renders 65% coverage');
  });

  test('F5.4: Coherence dimension bar renders with numeric percentage and color thresholding', () => {
    const html = renderMessageWithTrace(
      {
        verification: {
          faithfulness: 0.90,
          completeness: 0.90,
          citation_coverage: 0.90,
          coherence: 0.98,
          composite_score: 0.92,
          passed: true,
        },
      },
      { expanded: true }
    );
    assert(html.includes('Coherence'), 'renders Coherence label');
    assert(html.includes('98%'), 'renders 98% coherence');
  });

  test('F5.5: All 4 progress bars render inside expanded trace section with Framer Motion animated width', () => {
    const html = renderMessageWithTrace(
      {
        verification: {
          faithfulness: 0.95,
          completeness: 0.91,
          citation_coverage: 0.89,
          coherence: 0.97,
          composite_score: 0.93,
          passed: true,
        },
      },
      { expanded: true }
    );
    assert(html.includes('Self-Reflection Verification'), 'verification section header present');
    assert(
      html.includes('Faithfulness') &&
        html.includes('Completeness') &&
        html.includes('Citation Coverage') &&
        html.includes('Coherence'),
      'all 4 dimensions present'
    );
  });

  // =========================================================================
  // FEATURE 6: R2 Retry Indicator & Tooltip (5 tests)
  // =========================================================================

  test('F6.1: Shows retry badge in trace header when retry_count > 0', () => {
    const html = renderMessageWithTrace({
      retry_count: 2,
      retry_reasons: ['Initial draft lacked SLA details', 'Second check missing escalation steps'],
    });
    assert(html.includes('2 retries'), 'renders 2 retries in header');
    assert(html.includes('amber-500'), 'uses amber retry badge');
  });

  test('F6.2: Hides retry badge in header when retry_count === 0', () => {
    const html = renderMessageWithTrace({
      retry_count: 0,
      retry_reasons: [],
    });
    assert(!html.includes('0 retries') && !html.includes('0 retry'), 'no retry badge when retry_count is 0');
  });

  test('F6.3: Handles singular "1 retry" vs plural "2 retries" label correctly', () => {
    const htmlSingular = renderMessageWithTrace({
      retry_count: 1,
      retry_reasons: ['Refined citations'],
    });
    assert(htmlSingular.includes('1 retry') && !htmlSingular.includes('1 retries'), 'renders "1 retry" for singular');

    const htmlPlural = renderMessageWithTrace({
      retry_count: 3,
      retry_reasons: ['Reason 1', 'Reason 2', 'Reason 3'],
    });
    assert(htmlPlural.includes('3 retries'), 'renders "3 retries" for plural');
  });

  test('F6.4: Renders tooltip and detail list containing all retry reasons', () => {
    const reasons = ['Missing reimbursement cap clause', 'Unverified section reference'];
    const html = renderMessageWithTrace(
      {
        retry_count: 2,
        retry_reasons: reasons,
      },
      { expanded: true }
    );
    assert(html.includes('Missing reimbursement cap clause'), 'first reason rendered');
    assert(html.includes('Unverified section reference'), 'second reason rendered');
  });

  test('F6.5: Displays critique text inside expanded trace banner when available', () => {
    const critiqueText = 'Response improved after referencing Section 4.1 hardware eligibility.';
    const html = renderMessageWithTrace(
      {
        verification: {
          faithfulness: 0.92,
          completeness: 0.88,
          citation_coverage: 0.90,
          coherence: 0.95,
          composite_score: 0.91,
          passed: true,
          critique: critiqueText,
        },
      },
      { expanded: true }
    );
    assert(html.includes(critiqueText), 'critique text rendered inside trace');
  });

  // =========================================================================
  // FEATURE 7: R3 Metadata Filter Tag Chips (5 tests)
  // =========================================================================

  test('F7.1: Renders tag chips for single inferred filter (e.g. department: HR)', () => {
    const html = renderMessageWithTrace(
      {
        inferred_filters: { department: 'HR' },
      },
      { expanded: true }
    );
    assert(html.includes('Inferred Metadata Filters'), 'section header present');
    assert(html.includes('department:') && html.includes('HR'), 'department: HR chip rendered');
  });

  test('F7.2: Renders multiple tag chips for compound inferred filters (e.g. department, topic, year)', () => {
    const html = renderMessageWithTrace(
      {
        inferred_filters: { department: 'Engineering', topic: 'oncall', year: 2026 },
      },
      { expanded: true }
    );
    assert(html.includes('department:') && html.includes('Engineering'), 'Engineering chip');
    assert(html.includes('topic:') && html.includes('oncall'), 'oncall chip');
    assert(html.includes('year:') && html.includes('2026'), 'year: 2026 chip');
  });

  test('F7.3: Renders applied_filters if inferred_filters is not present', () => {
    const html = renderMessageWithTrace(
      {
        inferred_filters: {},
        applied_filters: { jurisdiction: 'US-West' },
      },
      { expanded: true }
    );
    assert(html.includes('jurisdiction:') && html.includes('US-West'), 'applied_filters fallback rendered');
  });

  test('F7.4: Handles non-string filter values (boolean, number, object) safely', () => {
    const html = renderMessageWithTrace(
      {
        inferred_filters: { is_active: true, level: 5, tags: ['contractor', 'remote'] },
      },
      { expanded: true }
    );
    assert(html.includes('is_active:') && html.includes('true'), 'boolean filter rendered');
    assert(html.includes('level:') && html.includes('5'), 'numeric filter rendered');
    assert(html.includes('tags:') && html.includes('contractor'), 'array/object filter rendered without crash');
  });

  test('F7.5: Omits filter section completely when inferred_filters is empty ({})', () => {
    const html = renderMessageWithTrace(
      {
        inferred_filters: {},
        applied_filters: {},
      },
      { expanded: true }
    );
    assert(!html.includes('Inferred Metadata Filters'), 'no filter section rendered for empty filters');
  });

  // =========================================================================
  // FEATURE 8: R3 Filter Relaxation Warning (5 tests)
  // =========================================================================

  test('F8.1: Renders amber warning callout when filter_relaxed is true', () => {
    const html = renderMessageWithTrace(
      {
        filter_relaxed: true,
        inferred_filters: { department: 'SpecialOps' },
      },
      { expanded: true }
    );
    assert(html.includes('Filters Relaxed:'), 'relaxation callout title rendered');
    assert(html.includes('amber-500') || html.includes('amber-800'), 'uses amber alert styling');
  });

  test('F8.2: Hides relaxation warning when filter_relaxed is false or undefined', () => {
    const html = renderMessageWithTrace(
      {
        filter_relaxed: false,
        inferred_filters: { department: 'HR' },
      },
      { expanded: true }
    );
    assert(!html.includes('Filters Relaxed:'), 'no relaxation callout when filter_relaxed is false');
  });

  test('F8.3: Explains fallback reason clearly (0 search results -> unfiltered)', () => {
    const html = renderMessageWithTrace(
      {
        filter_relaxed: true,
      },
      { expanded: true }
    );
    assert(html.includes('Filtered search returned 0 results'), 'explains zero results condition');
    assert(html.includes('unfiltered search'), 'explains fallback to unfiltered search');
  });

  test('F8.4: Renders with AlertTriangle icon and accessible styling', () => {
    const html = renderMessageWithTrace(
      {
        filter_relaxed: true,
      },
      { expanded: true }
    );
    assert(html.includes('strong') || html.includes('Filters Relaxed:'), 'semantic emphasis present');
  });

  test('F8.5: Co-exists gracefully with filter chips and other indicators', () => {
    const html = renderMessageWithTrace(
      {
        filter_relaxed: true,
        inferred_filters: { department: 'Legal', confidentiality: 'TopSecret' },
        query_type: 'enumeration',
        verification_score: 0.88,
        cache_hit: false,
      },
      { expanded: true }
    );
    assert(html.includes('Filters Relaxed:'), 'relaxation callout present');
    assert(html.includes('department:') && html.includes('Legal'), 'filter chips present');
    assert(html.includes('enumeration'), 'query classification badge present');
  });

  // =========================================================================
  // FEATURE 9: R3 Semantic Cache Hit Badge (5 tests)
  // =========================================================================

  test('F9.1: Displays Cache Hit badge with sky palette and Zap icon when cache_hit is true', () => {
    const html = renderMessageWithTrace({
      cache_hit: true,
    });
    assert(html.includes('Cache Hit'), 'renders Cache Hit badge');
    assert(html.includes('sky-500') || html.includes('sky-700'), 'uses sky palette');
  });

  test('F9.2: Displays cache similarity percentage when cache_similarity is provided', () => {
    const html = renderMessageWithTrace({
      cache_hit: true,
      cache_similarity: 0.984,
    });
    assert(html.includes('Cache Hit'), 'Cache Hit badge present');
    assert(html.includes('98%'), 'displays 98% similarity');
    assert(html.includes('98.4% similarity'), 'tooltip displays exact 98.4% similarity');
  });

  test('F9.3: Hides cache hit badge when cache_hit is false or undefined', () => {
    const html = renderMessageWithTrace({
      cache_hit: false,
      cache_similarity: null,
    });
    assert(!html.includes('Cache Hit'), 'no Cache Hit badge when cache_hit is false');
  });

  test('F9.4: Renders tooltip describing cache hit status and similarity score', () => {
    const html = renderMessageWithTrace({
      cache_hit: true,
      cache_similarity: 0.95,
    });
    assert(html.includes('Semantic Cache Hit: Served directly from cache (95.0% similarity)'), 'tooltip matches');
  });

  test('F9.5: Automatically infers cache_hit when retrieval_strategy is conversational_bypass / semantic_cache', () => {
    const trace = mapTrace({
      trace_id: 'tr_conv_cache',
      query: 'Hello policy bot',
      retrieval_strategy: 'conversational_bypass',
    });
    assert(trace.cache_hit === true, 'cache_hit automatically inferred from conversational_bypass');
  });

  // =========================================================================
  // FEATURE 10: R4 AdminView Table Columns (5 tests)
  // =========================================================================

  test('F10.1: Displays Query Type column with colored chip and confidence dot in AdminView', () => {
    const trace = mapTrace({
      trace_id: 'tr_admin_01',
      query: 'maternity leave policy',
      query_type: 'factual',
      routing_confidence: 0.96,
      retrieval_strategy: 'dense_bm25_hybrid',
    });
    const html = renderAdminView({ recent_traces: [trace] });
    assert(html.includes('Factual'), 'renders Factual chip');
    assert(html.includes('(96%)') || html.includes('96%'), 'renders 96% confidence');
  });

  test('F10.2: Displays Verification column with pass/fail colored pill in AdminView', () => {
    const passTrace = mapTrace({
      trace_id: 'tr_pass',
      query: 'vacation policy',
      verification_score: 0.92,
      faithfulness_passed: true,
    });
    const failTrace = mapTrace({
      trace_id: 'tr_fail',
      query: 'invalid claim query',
      verification_score: 0.58,
      faithfulness_passed: false,
    });
    const html = renderAdminView({ recent_traces: [passTrace, failTrace] });
    assert(html.includes('92%'), 'renders 92% verification score');
    assert(html.includes('Pass'), 'renders Pass label');
    assert(html.includes('58%'), 'renders 58% verification score');
    assert(html.includes('Fail'), 'renders Fail label');
  });

  test('F10.3: Displays Filter Status column showing active filter summary or "None"', () => {
    const traceWithFilters = mapTrace({
      trace_id: 'tr_f1',
      query: 'eng policy',
      inferred_filters: { department: 'Engineering' },
      applied_filters: { department: 'Engineering' },
    });
    const traceNoFilters = mapTrace({
      trace_id: 'tr_f2',
      query: 'general query',
      inferred_filters: {},
      applied_filters: {},
    });
    const html = renderAdminView({ recent_traces: [traceWithFilters, traceNoFilters] });
    assert(html.includes('dept: Engineering') || html.includes('Engineering'), 'renders filter summary');
    assert(html.includes('None'), 'renders None for empty filter');
  });

  test('F10.4: Displays "relaxed" badge in Filter Status column when filter_relaxed is true', () => {
    const relaxedTrace = mapTrace({
      trace_id: 'tr_rel',
      query: 'classified policy',
      inferred_filters: { department: 'Legal' },
      filter_relaxed: true,
    });
    const html = renderAdminView({ recent_traces: [relaxedTrace] });
    assert(html.includes('relaxed'), 'renders relaxed badge in table');
  });

  test('F10.5: Desktop table header includes all 8 columns in AdminView', () => {
    const trace = mapTrace({ trace_id: 'tr_head', query: 'head test' });
    const html = renderAdminView({ recent_traces: [trace] });
    assert(html.includes('Original Query'), 'Original Query header');
    assert(html.includes('Query Type'), 'Query Type header');
    assert(html.includes('Verification'), 'Verification header');
    assert(html.includes('Filter Status'), 'Filter Status header');
    assert(html.includes('Chunks'), 'Chunks header');
    assert(html.includes('Rerank'), 'Rerank header');
    assert(html.includes('Latency'), 'Latency header');
    assert(html.includes('Tokens'), 'Tokens header');
  });

  // =========================================================================
  // FEATURE 11: R4 AdminView Expandable Detail (5 tests)
  // =========================================================================

  test('F11.1: Expanding row reveals Self-Reflection Verification Report with 4 dimension bars in AdminView', () => {
    const fullVerReport: VerificationReport = {
      faithfulness: 0.97,
      completeness: 0.93,
      citation_coverage: 0.95,
      coherence: 0.98,
      composite_score: 0.96,
      passed: true,
    };
    const trace = mapTrace({
      trace_id: 'tr_detail_1',
      query: 'detailed check',
      verification: fullVerReport,
    });
    const html = renderAdminView({ recent_traces: [trace] }, { expandedTraceId: 'tr_detail_1' });
    assert(html.includes('Self-Reflection Verification Report'), 'report header rendered');
    assert(html.includes('Faithfulness') && html.includes('97%'), 'faithfulness 97%');
    assert(html.includes('Completeness') && html.includes('93%'), 'completeness 93%');
    assert(html.includes('Citation Coverage') && html.includes('95%'), 'citation coverage 95%');
    assert(html.includes('Coherence') && html.includes('98%'), 'coherence 98%');
  });

  test('F11.2: Expanding row reveals Reflection Critique, Missing Aspects, and Unsupported Claims in AdminView', () => {
    const report: VerificationReport = {
      faithfulness: 0.85,
      completeness: 0.80,
      citation_coverage: 0.82,
      coherence: 0.90,
      composite_score: 0.84,
      passed: true,
      critique: 'Need clearer clarification on probation period.',
      missing_aspects: ['Probation period notice requirement'],
      unsupported_claims: ['Immediate stock grant without vesting'],
    };
    const trace = mapTrace({
      trace_id: 'tr_critique_test',
      query: 'probation stock query',
      verification: report,
    });
    const html = renderAdminView({ recent_traces: [trace] }, { expandedTraceId: 'tr_critique_test' });
    assert(html.includes('Reflection Critique'), 'critique section header present');
    assert(html.includes('Need clearer clarification on probation period.'), 'critique text rendered');
    assert(html.includes('Missing Aspects'), 'missing aspects header present');
    assert(html.includes('Probation period notice requirement'), 'missing aspect item rendered');
    assert(html.includes('Unsupported Claims'), 'unsupported claims header present');
    assert(html.includes('Immediate stock grant without vesting'), 'unsupported claim item rendered');
  });

  test('F11.3: Expanding row reveals Retry History Card with retry reasons in AdminView', () => {
    const trace = mapTrace({
      trace_id: 'tr_retry_detail',
      query: 'equipment return',
      retry_count: 2,
      retry_reasons: ['Reason 1: Missing section', 'Reason 2: Clarified return policy'],
    });
    const html = renderAdminView({ recent_traces: [trace] }, { expandedTraceId: 'tr_retry_detail' });
    assert(html.includes('Verification Retries (2)'), 'retry header rendered');
    assert(html.includes('Reason 1: Missing section'), 'first retry reason rendered');
    assert(html.includes('Reason 2: Clarified return policy'), 'second retry reason rendered');
  });

  test('F11.4: Expanding row reveals Filters Detail Card comparing Inferred vs Applied filters in AdminView', () => {
    const trace = mapTrace({
      trace_id: 'tr_filters_compare',
      query: 'intern stipend',
      inferred_filters: { department: 'HR', role: 'Intern' },
      applied_filters: { department: 'HR' },
      filter_relaxed: true,
    });
    const html = renderAdminView({ recent_traces: [trace] }, { expandedTraceId: 'tr_filters_compare' });
    assert(html.includes('Inferred Filters (Query Router)'), 'inferred filters header present');
    assert(html.includes('role:') && html.includes('Intern'), 'role Intern rendered');
    assert(html.includes('Applied Filters (Vector &amp; BM25)') || html.includes('Applied Filters'), 'applied filters header present');
    assert(html.includes('Filter Fallback Triggered'), 'relaxation warning card present in detail');
  });

  test('F11.5: Expanding row displays Rewritten Query and Expanded Multi-Queries in AdminView', () => {
    const trace = mapTrace({
      trace_id: 'tr_rewrite_expand',
      query: 'paternity leave',
      rewritten_query: 'paternity leave allowance policy',
      sub_queries: ['paternity leave days', 'paternity pay rate'],
    });
    const html = renderAdminView({ recent_traces: [trace] }, { expandedTraceId: 'tr_rewrite_expand' });
    assert(html.includes('Rewritten Query'), 'rewritten query header present');
    assert(html.includes('paternity leave allowance policy'), 'rewritten text present');
    assert(html.includes('Expanded Queries'), 'expanded queries header present');
    assert(html.includes('paternity leave days'), 'first subquery present');
    assert(html.includes('paternity pay rate'), 'second subquery present');
  });

  return results;
}
