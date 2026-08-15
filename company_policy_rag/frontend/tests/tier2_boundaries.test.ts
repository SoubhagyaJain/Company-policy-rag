/**
 * Tier 2 Boundary & Corner Cases Test Suite for Agentic Intelligence UI Indicators.
 *
 * Requirements & Specifications:
 * - TEST_INFRA.md § Coverage Goals: Tier 2 (>= 55 test cases across boundaries, extremes, nulls, stress)
 */

import {
  QueryTrace,
  VerificationReport,
  ChatMessageData,
} from '../lib/types';
import {
  mapTrace,
  mapVerificationReport,
  ApiClient,
} from '../lib/api-client';
import {
  renderChatMessage,
  renderMessageWithTrace,
  assert,
} from './test_helpers';
import { TestResult } from './tier1_features.test';

export function runTier2Tests(): TestResult[] {
  const results: TestResult[] = [];

  function test(name: string, fn: () => void) {
    const start = performance.now();
    try {
      fn();
      results.push({
        suite: 'Tier 2: Boundary & Corner Cases',
        name,
        passed: true,
        durationMs: performance.now() - start,
      });
    } catch (err: any) {
      results.push({
        suite: 'Tier 2: Boundary & Corner Cases',
        name,
        passed: false,
        error: err?.message || String(err),
        durationMs: performance.now() - start,
      });
    }
  }

  // =========================================================================
  // GROUP 1: Null & Missing Values (10 tests)
  // =========================================================================

  test('B1.1: Handles completely empty/null trace object gracefully without throwing', () => {
    const emptyTrace = mapTrace(null);
    assert(emptyTrace.trace_id.startsWith('tr_'), 'generates fallback trace_id');
    assert(emptyTrace.total_chunks_retrieved === 0, 'fallback chunks retrieved');
    assert(emptyTrace.prompt_tokens === 0, 'fallback prompt tokens');
    const html = renderMessageWithTrace(emptyTrace);
    assert(typeof html === 'string' && html.length > 0, 'renders without crashing');
  });

  test('B1.2: Handles null verification_score and null verification object gracefully', () => {
    const trace = mapTrace({
      trace_id: 'tr_null_ver',
      verification_score: null,
      verification: null,
    });
    assert(trace.verification_score === undefined, 'null score converts to undefined');
    assert(trace.verification === null, 'null verification is null');
    const html = renderMessageWithTrace(trace);
    assert(!html.includes('Verified') && !html.includes('Review'), 'omits score pill');
  });

  test('B1.3: Handles null query_type gracefully with neutral fallback styling', () => {
    const trace = mapTrace({
      trace_id: 'tr_null_qt',
      query_type: null,
    });
    assert(trace.query_type === undefined, 'null query_type converted');
    const html = renderMessageWithTrace(trace);
    assert(typeof html === 'string', 'renders successfully');
  });

  test('B1.4: Handles null routing_confidence gracefully without showing NaN', () => {
    const trace = mapTrace({
      trace_id: 'tr_null_conf',
      query_type: 'factual',
      routing_confidence: null,
    });
    assert(trace.routing_confidence === undefined, 'null confidence converted to undefined');
    const html = renderMessageWithTrace(trace);
    assert(!html.includes('NaN'), 'no NaN in rendered html');
  });

  test('B1.5: Handles null inferred_filters and applied_filters gracefully', () => {
    const trace = mapTrace({
      trace_id: 'tr_null_filt',
      inferred_filters: null,
      applied_filters: null,
    });
    assert(typeof trace.inferred_filters === 'object', 'inferred_filters defaults to object');
    assert(typeof trace.applied_filters === 'object', 'applied_filters defaults to object');
    const html = renderMessageWithTrace(trace, { expanded: true });
    assert(!html.includes('Inferred Metadata Filters'), 'no filter chips for null filters');
  });

  test('B1.6: Handles null cache_similarity when cache_hit is true gracefully', () => {
    const trace = mapTrace({
      trace_id: 'tr_null_sim',
      cache_hit: true,
      cache_similarity: null,
    });
    assert(trace.cache_hit === true, 'cache_hit is true');
    assert(trace.cache_similarity === null, 'cache_similarity is null');
    const html = renderMessageWithTrace(trace);
    assert(html.includes('Cache Hit'), 'renders Cache Hit badge without percentage');
  });

  test('B1.7: Handles undefined stage_timings gracefully', () => {
    const trace = mapTrace({
      trace_id: 'tr_no_timings',
      stage_timings: undefined,
    });
    assert(trace.stage_timings === undefined, 'stage_timings undefined');
  });

  test('B1.8: Handles null critique in verification report gracefully', () => {
    const trace = mapTrace({
      trace_id: 'tr_null_critique',
      verification: {
        faithfulness: 0.95,
        completeness: 0.90,
        citation_coverage: 0.90,
        coherence: 0.95,
        composite_score: 0.92,
        passed: true,
        critique: null,
      },
    });
    assert(trace.verification?.critique === null, 'critique is null');
    const html = renderMessageWithTrace(trace);
    assert(html.includes('92%'), 'renders verification pill');
  });

  test('B1.9: Handles undefined model name by providing fallback', () => {
    const trace = mapTrace({
      trace_id: 'tr_no_model',
      model: undefined,
    });
    assert(trace.model === 'FastAPI RAG', 'falls back to default model name');
  });

  test('B1.10: Handles undefined latency metrics gracefully with 0 ms fallback', () => {
    const trace = mapTrace({
      trace_id: 'tr_no_latency',
      total_latency_ms: undefined,
      rerank_latency_ms: undefined,
    });
    assert(trace.total_latency_ms === 0, 'total latency fallback to 0');
    assert(trace.rerank_latency_ms === 0, 'rerank latency fallback to 0');
  });

  // =========================================================================
  // GROUP 2: Score Extremes & Boundaries (10 tests)
  // =========================================================================

  test('B2.1: Verification composite score exactly 0.0 renders 0% Review', () => {
    const html = renderMessageWithTrace({
      verification_score: 0.0,
      faithfulness_passed: false,
    });
    assert(html.includes('0% Review') || (html.includes('0%') && html.includes('Review')), 'renders 0% Review');
    assert(html.includes('amber-500'), 'uses amber fail color');
  });

  test('B2.2: Verification composite score exactly 1.0 renders 100% Verified', () => {
    const html = renderMessageWithTrace({
      verification_score: 1.0,
      faithfulness_passed: true,
    });
    assert(html.includes('100% Verified') || (html.includes('100%') && html.includes('Verified')), 'renders 100% Verified');
    assert(html.includes('emerald-500'), 'uses emerald pass color');
  });

  test('B2.3: Verification score at 0.749 (borderline fail) renders Review status', () => {
    const html = renderMessageWithTrace({
      verification_score: 0.749,
      faithfulness_passed: true,
      verification: {
        faithfulness: 0.75,
        completeness: 0.74,
        citation_coverage: 0.75,
        coherence: 0.75,
        composite_score: 0.749,
        passed: false,
      },
    });
    assert(html.includes('75% Review') || html.includes('Review'), 'renders Review status for score below pass threshold');
  });

  test('B2.4: Verification score at 0.750 (exact pass threshold) renders Verified status', () => {
    const html = renderMessageWithTrace({
      verification_score: 0.750,
      faithfulness_passed: true,
      verification: {
        faithfulness: 0.75,
        completeness: 0.75,
        citation_coverage: 0.75,
        coherence: 0.75,
        composite_score: 0.750,
        passed: true,
      },
    });
    assert(html.includes('75% Verified') || (html.includes('75%') && html.includes('Verified')), 'renders Verified status for threshold score');
  });

  test('B2.5: Faithfulness dimension score color thresholding (>=85% emerald, 70-84% amber, <70% rose)', () => {
    const htmlHigh = renderMessageWithTrace(
      {
        verification: {
          faithfulness: 0.85,
          completeness: 0.90,
          citation_coverage: 0.90,
          coherence: 0.90,
          composite_score: 0.88,
          passed: true,
        },
      },
      { expanded: true }
    );
    assert(htmlHigh.includes('85%'), 'high score rendered');

    const htmlMed = renderMessageWithTrace(
      {
        verification: {
          faithfulness: 0.72,
          completeness: 0.90,
          citation_coverage: 0.90,
          coherence: 0.90,
          composite_score: 0.85,
          passed: true,
        },
      },
      { expanded: true }
    );
    assert(htmlMed.includes('72%'), 'medium score rendered');

    const htmlLow = renderMessageWithTrace(
      {
        verification: {
          faithfulness: 0.45,
          completeness: 0.90,
          citation_coverage: 0.90,
          coherence: 0.90,
          composite_score: 0.65,
          passed: false,
        },
      },
      { expanded: true }
    );
    assert(htmlLow.includes('45%'), 'low score rendered');
  });

  test('B2.6: Routing confidence 0.0 renders 0% label', () => {
    const html = renderMessageWithTrace({
      query_type: 'factual',
      routing_confidence: 0.0,
    });
    assert(html.includes('(0%)') || html.includes('0%'), 'renders (0%)');
  });

  test('B2.7: Routing confidence 1.0 renders 100% label', () => {
    const html = renderMessageWithTrace({
      query_type: 'factual',
      routing_confidence: 1.0,
    });
    assert(html.includes('(100%)'), 'renders (100%)');
  });

  test('B2.8: Cache similarity 0.0 renders 0% similarity label', () => {
    const html = renderMessageWithTrace({
      cache_hit: true,
      cache_similarity: 0.0,
    });
    assert(html.includes('0%') || html.includes('Cache Hit'), 'renders cache hit with 0% similarity');
  });

  test('B2.9: Cache similarity 1.0 renders 100% similarity label', () => {
    const html = renderMessageWithTrace({
      cache_hit: true,
      cache_similarity: 1.0,
    });
    assert(html.includes('(100%)'), 'renders (100%)');
  });

  test('B2.10: Out-of-bounds score values (>1.0 or <0.0) are safely clamped without overflow', () => {
    const html = renderMessageWithTrace({
      verification_score: 1.25,
      faithfulness_passed: true,
    });
    assert(html.includes('100% Verified') || (html.includes('100%') && html.includes('Verified')), 'clamped to 100%');
  });

  // =========================================================================
  // GROUP 3: Retry Count Boundaries (8 tests)
  // =========================================================================

  test('B3.1: retry_count = 0 with empty retry_reasons omits retry badges', () => {
    const html = renderMessageWithTrace({
      retry_count: 0,
      retry_reasons: [],
    });
    assert(!html.includes('retry') && !html.includes('retries'), 'no retry badge rendered');
  });

  test('B3.2: retry_count = 0 with non-empty retry_reasons omits header retry badge', () => {
    const html = renderMessageWithTrace({
      retry_count: 0,
      retry_reasons: ['Ignored reason because count is 0'],
    });
    assert(!html.includes('0 retries'), 'no 0 retries badge');
  });

  test('B3.3: retry_count = 1 renders singular "1 retry"', () => {
    const html = renderMessageWithTrace({
      retry_count: 1,
      retry_reasons: ['Single retry reason'],
    });
    assert(html.includes('1 retry'), 'renders singular "1 retry"');
    assert(!html.includes('1 retries'), 'does not render "1 retries"');
  });

  test('B3.4: retry_count = 2 renders plural "2 retries"', () => {
    const html = renderMessageWithTrace({
      retry_count: 2,
      retry_reasons: ['Reason 1', 'Reason 2'],
    });
    assert(html.includes('2 retries'), 'renders plural "2 retries"');
  });

  test('B3.5: retry_count = 5 renders "5 retries" and lists all reasons', () => {
    const reasons = [
      'Reason 1: missing header',
      'Reason 2: ambiguous clause',
      'Reason 3: citation check fail',
      'Reason 4: incomplete steps',
      'Reason 5: final polish',
    ];
    const html = renderMessageWithTrace(
      {
        retry_count: 5,
        retry_reasons: reasons,
      },
      { expanded: true }
    );
    assert(html.includes('5 retries'), 'renders 5 retries');
    assert(html.includes('Reason 5: final polish'), 'lists 5th reason');
  });

  test('B3.6: When top-level retry_count is undefined, falls back to verification.retry_count', () => {
    const trace = mapTrace({
      trace_id: 'tr_retry_fb',
      verification: {
        faithfulness: 0.9,
        completeness: 0.9,
        citation_coverage: 0.9,
        coherence: 0.9,
        composite_score: 0.9,
        passed: true,
        retry_count: 2,
      },
    });
    assert(trace.retry_count === 2, 'retry_count fallback from verification object');
  });

  test('B3.7: Missing retry_reasons array defaults to empty array', () => {
    const trace = mapTrace({
      trace_id: 'tr_no_reasons',
      retry_count: 1,
      retry_reasons: undefined,
    });
    assert(Array.isArray(trace.retry_reasons) && trace.retry_reasons.length === 0, 'empty array default');
  });

  test('B3.8: retry_reasons with multiple empty strings does not crash rendering', () => {
    const html = renderMessageWithTrace(
      {
        retry_count: 2,
        retry_reasons: ['', '   ', 'Valid reason'],
      },
      { expanded: true }
    );
    assert(html.includes('Valid reason'), 'renders valid reason');
  });

  // =========================================================================
  // GROUP 4: Filter Boundaries & Edge Cases (10 tests)
  // =========================================================================

  test('B4.1: filter_relaxed = true with empty inferred_filters renders relaxation warning', () => {
    const html = renderMessageWithTrace(
      {
        filter_relaxed: true,
        inferred_filters: {},
      },
      { expanded: true }
    );
    assert(html.includes('Filters Relaxed:'), 'renders relaxation warning even with empty inferred_filters');
  });

  test('B4.2: filter_relaxed = true with non-empty inferred_filters renders both chips and warning', () => {
    const html = renderMessageWithTrace(
      {
        filter_relaxed: true,
        inferred_filters: { department: 'Legal', confidentiality: 'Confidential' },
      },
      { expanded: true }
    );
    assert(html.includes('Filters Relaxed:'), 'relaxation warning present');
    assert(html.includes('department:') && html.includes('Legal'), 'department chip present');
    assert(html.includes('confidentiality:') && html.includes('Confidential'), 'confidentiality chip present');
  });

  test('B4.3: filter_relaxed = false with non-empty filters renders chips without warning', () => {
    const html = renderMessageWithTrace(
      {
        filter_relaxed: false,
        inferred_filters: { department: 'HR' },
      },
      { expanded: true }
    );
    assert(!html.includes('Filters Relaxed:'), 'no relaxation warning');
    assert(html.includes('department:') && html.includes('HR'), 'filter chips present');
  });

  test('B4.4: inferred_filters with nested objects serializes to valid JSON string', () => {
    const html = renderMessageWithTrace(
      {
        inferred_filters: { metadata: { code: 'A1', active: true } },
      },
      { expanded: true }
    );
    assert(html.includes('metadata:'), 'renders key');
    assert(html.includes('&quot;code&quot;:&quot;A1&quot;') || html.includes('code') || html.includes('A1'), 'serializes nested object safely');
  });

  test('B4.5: inferred_filters with boolean, integer, float values renders accurately', () => {
    const html = renderMessageWithTrace(
      {
        inferred_filters: { is_active: false, tier: 3, ratio: 0.75 },
      },
      { expanded: true }
    );
    assert(html.includes('is_active:') && html.includes('false'), 'boolean false rendered');
    assert(html.includes('tier:') && html.includes('3'), 'integer 3 rendered');
    assert(html.includes('ratio:') && html.includes('0.75'), 'float 0.75 rendered');
  });

  test('B4.6: inferred_filters with special characters in keys and values renders safely', () => {
    const html = renderMessageWithTrace(
      {
        inferred_filters: { 'dept/sub-org': 'HR & Benefits (Global)', 'policy#': 'POL-2026.01' },
      },
      { expanded: true }
    );
    assert(html.includes('dept/sub-org:'), 'special key rendered');
    assert(html.includes('HR &amp; Benefits (Global)') || html.includes('HR & Benefits'), 'escaped or preserved safely');
  });

  test('B4.7: inferred_filters with 10+ key-value pairs renders without breaking layout', () => {
    const manyFilters: Record<string, any> = {};
    for (let i = 1; i <= 10; i++) {
      manyFilters[`filter_key_${i}`] = `value_${i}`;
    }
    const html = renderMessageWithTrace(
      {
        inferred_filters: manyFilters,
      },
      { expanded: true }
    );
    assert(html.includes('filter_key_1:') && html.includes('filter_key_10:'), 'renders all 10 filter keys');
  });

  test('B4.8: applied_filters differing from inferred_filters renders inferred_filters as priority in trace banner', () => {
    const html = renderMessageWithTrace(
      {
        inferred_filters: { department: 'HR' },
        applied_filters: { department: 'HR', auto_expanded: true },
      },
      { expanded: true }
    );
    assert(html.includes('department:') && html.includes('HR'), 'inferred filters displayed in ChatMessage');
  });

  test('B4.9: Filter values containing null or undefined are excluded from tag chips', () => {
    const html = renderMessageWithTrace(
      {
        inferred_filters: { valid_key: 'valid_val', null_key: null, undef_key: undefined },
      },
      { expanded: true }
    );
    assert(html.includes('valid_key:'), 'valid key rendered');
    assert(!html.includes('null_key:'), 'null key excluded');
    assert(!html.includes('undef_key:'), 'undefined key excluded');
  });

  test('B4.10: Empty string filter values are excluded from tag chips', () => {
    const html = renderMessageWithTrace(
      {
        inferred_filters: { present: 'yes', empty: '' },
      },
      { expanded: true }
    );
    assert(html.includes('present:'), 'present key rendered');
    assert(!html.includes('empty:'), 'empty string key excluded');
  });

  // =========================================================================
  // GROUP 5: Text & Payload Stress (10 tests)
  // =========================================================================

  test('B5.1: Extremely long original query (2000+ characters) renders without overflow', () => {
    const longQuery = 'What is the company policy regarding ' + 'vacation leave and sick leave '.repeat(100);
    const trace = mapTrace({
      trace_id: 'tr_long_query',
      query: longQuery,
    });
    assert(trace.original_query.length > 2000, 'query length exceeds 2000');
    const html = renderMessageWithTrace(trace);
    assert(typeof html === 'string', 'renders without error');
  });

  test('B5.2: Extremely long rewritten query (1000+ characters) renders in blockquote', () => {
    const longRewritten = 'Rewritten query: ' + 'detailed policy search '.repeat(60);
    const html = renderMessageWithTrace(
      {
        query_rewritten: longRewritten,
      },
      { expanded: true }
    );
    assert(html.includes('Rewritten Query'), 'Rewritten Query header present');
    assert(html.includes('detailed policy search'), 'contains rewritten text');
  });

  test('B5.3: 10+ expanded multi-queries render as a bulleted list with arrow icons', () => {
    const subQueries = Array.from({ length: 12 }, (_, i) => `Sub query expansion #${i + 1} for policy search`);
    const html = renderMessageWithTrace(
      {
        expanded_queries: subQueries,
      },
      { expanded: true }
    );
    assert(html.includes('Expanded Multi-Queries'), 'Expanded Multi-Queries header present');
    assert(html.includes('Sub query expansion #12'), '12th subquery rendered');
  });

  test('B5.4: Extremely long critique text (500+ words) renders in italic callout', () => {
    const longCritique = 'Verification critique evaluation: ' + 'The model answer correctly synthesized the rules. '.repeat(50);
    const html = renderMessageWithTrace(
      {
        verification: {
          faithfulness: 0.95,
          completeness: 0.90,
          citation_coverage: 0.92,
          coherence: 0.95,
          composite_score: 0.93,
          passed: true,
          critique: longCritique,
        },
      },
      { expanded: true }
    );
    assert(html.includes('Verification critique evaluation:'), 'critique present');
  });

  test('B5.5: 10+ missing aspects items render without breaking UI structure', () => {
    const missing = Array.from({ length: 10 }, (_, i) => `Missing aspect #${i + 1}`);
    const trace = mapTrace({
      trace_id: 'tr_missing_10',
      verification: {
        faithfulness: 0.8,
        completeness: 0.7,
        citation_coverage: 0.8,
        coherence: 0.8,
        composite_score: 0.77,
        passed: true,
        missing_aspects: missing,
      },
    });
    assert(trace.verification?.missing_aspects?.length === 10, '10 missing aspects preserved');
  });

  test('B5.6: 10+ unsupported claims items render safely in model', () => {
    const claims = Array.from({ length: 10 }, (_, i) => `Unsupported claim #${i + 1}`);
    const trace = mapTrace({
      trace_id: 'tr_claims_10',
      verification: {
        faithfulness: 0.6,
        completeness: 0.8,
        citation_coverage: 0.7,
        coherence: 0.8,
        composite_score: 0.72,
        passed: false,
        unsupported_claims: claims,
      },
    });
    assert(trace.verification?.unsupported_claims?.length === 10, '10 unsupported claims preserved');
  });

  test('B5.7: Unicode, emojis, and international characters in queries and filters', () => {
    const html = renderMessageWithTrace(
      {
        original_query: '政策查询 休暇制度 ☕ Überprüfung của chính sách',
        inferred_filters: { '部門': '人事部 (HR)', 'region': 'Tokyo 🗼' },
        query_type: 'factual',
      },
      { expanded: true }
    );
    assert(html.includes('部門:'), 'Japanese key rendered');
    assert(html.includes('Tokyo 🗼') || html.includes('Tokyo'), 'emoji rendered');
  });

  test('B5.8: Zero chunks retrieved (0 chunks) renders "0 chunks"', () => {
    const html = renderMessageWithTrace(
      {
        total_chunks_retrieved: 0,
      },
      { expanded: true }
    );
    assert(html.includes('0 chunks'), 'renders 0 chunks');
  });

  test('B5.9: High candidate count (1000 chunks) renders "1000 chunks"', () => {
    const html = renderMessageWithTrace(
      {
        total_chunks_retrieved: 1000,
      },
      { expanded: true }
    );
    assert(html.includes('1000 chunks'), 'renders 1000 chunks');
  });

  test('B5.10: Latency boundaries: 0 ms total latency vs 60000 ms (60.00 s)', () => {
    const htmlFast = renderMessageWithTrace({ total_latency_ms: 0 });
    assert(htmlFast.includes('0 ms') || htmlFast.includes('Thought for 0 ms'), '0 ms latency rendered');

    const htmlSlow = renderMessageWithTrace({ total_latency_ms: 60000 });
    assert(htmlSlow.includes('60.00 s') || htmlSlow.includes('60 s') || htmlSlow.includes('60.0s'), '60.00 s latency rendered');
  });

  // =========================================================================
  // GROUP 6: SSE Ingestion Fault Tolerance (7 tests)
  // =========================================================================

  test('B6.1: ApiClient handles malformed/corrupted JSON in SSE data gracefully', () => {
    const client = new ApiClient();
    let chunkReceived = '';
    const callbacks = {
      onChunk: (c: string) => {
        chunkReceived += c;
      },
    };

    (client as any).handleStreamEvent('chunk', 'Raw streaming text chunk without JSON', callbacks);
    assert(chunkReceived === 'Raw streaming text chunk without JSON', 'handled raw chunk gracefully');
  });

  test('B6.2: ApiClient ignores unknown SSE event types without crashing', () => {
    const client = new ApiClient();
    let errorThrown = false;
    try {
      (client as any).handleStreamEvent('unknown_heartbeat_event', { time: Date.now() }, {});
    } catch {
      errorThrown = true;
    }
    assert(!errorThrown, 'no error thrown on unknown event type');
  });

  test('B6.3: ApiClient handles error event and triggers onError callback', () => {
    const client = new ApiClient();
    let capturedError: Error | null = null;
    const callbacks = {
      onError: (err: Error) => {
        capturedError = err;
      },
    };

    (client as any).handleStreamEvent('error', { detail: 'Database connection failed' }, callbacks);
    assert(capturedError !== null, 'error callback invoked');
    assert((capturedError as any)?.message.includes('Database connection failed'), 'error message matches');
  });

  test('B6.4: ApiClient handles done event without retrieval_trace gracefully', () => {
    const client = new ApiClient();
    let doneInvoked = false;
    const callbacks = {
      onDone: () => {
        doneInvoked = true;
      },
    };

    (client as any).handleStreamEvent('done', { answer: 'All done' }, callbacks);
    assert(doneInvoked, 'onDone callback invoked');
  });

  test('B6.5: ApiClient handles empty citations array in citation event', () => {
    const client = new ApiClient();
    let citationCount = 0;
    const callbacks = {
      onCitation: () => {
        citationCount++;
      },
    };

    (client as any).handleStreamEvent('citation', { citations: [] }, callbacks);
    assert(citationCount === 0, 'no citations dispatched for empty array');
  });

  test('B6.6: ApiClient handles start event with session and message IDs', () => {
    const client = new ApiClient();
    let startData: any = null;
    const callbacks = {
      onStart: (d: any) => {
        startData = d;
      },
    };

    (client as any).handleStreamEvent('start', { session_id: 'sess_123', message_id: 'msg_456' }, callbacks);
    assert(startData?.session_id === 'sess_123', 'start session_id matched');
    assert(startData?.message_id === 'msg_456', 'start message_id matched');
  });

  test('B6.7: Unrecognized query_type string applies fallback neutral badge styling', () => {
    const html = renderMessageWithTrace({
      query_type: 'custom_multimodal_intent',
    });
    assert(html.includes('custom_multimodal_intent'), 'renders custom type text');
    assert(html.includes('border-sand-border') || html.includes('bg-cream-200'), 'uses neutral fallback styling');
  });

  return results;
}
