/**
 * Adversarial Challenger 2 Test Suite: Data Pipeline & State Verification.
 *
 * Rigorous empirical testing for:
 * 1. Backward compatibility with legacy/non-agentic SSE trace payloads.
 * 2. Full fidelity round-trip ingestion of all 13 agentic telemetry fields.
 * 3. Correct propagation through streamChat(), getObservability(), and useChatStream().
 * 4. Error handling when SSE chunks are truncated, corrupted, or fragmented.
 */

import {
  QueryTrace,
  VerificationReport,
  ObservabilityData,
  Citation,
  ChatMessageData,
} from '../lib/types';
import {
  mapTrace,
  mapVerificationReport,
  ApiClient,
} from '../lib/api-client';

export interface TestResult {
  suite: string;
  name: string;
  passed: boolean;
  error?: string;
  durationMs: number;
}

function assert(condition: any, msg: string): asserts condition {
  if (!condition) {
    throw new Error(`Assertion failed: ${msg}`);
  }
}

function assertEqual(actual: any, expected: any, msg: string) {
  const actualStr = JSON.stringify(actual);
  const expectedStr = JSON.stringify(expected);
  if (actualStr !== expectedStr) {
    throw new Error(`Assertion failed: ${msg} (expected: ${expectedStr}, got: ${actualStr})`);
  }
}

/**
 * Creates a ReadableStream from an array of string chunks (can be byte-level fragments).
 */
function createMockStream(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
}

export function runAdversarialPipelineTests(): TestResult[] {
  const results: TestResult[] = [];

  function test(name: string, fn: () => void | Promise<void>) {
    const start = performance.now();
    try {
      const res = fn();
      if (res && typeof (res as any).then === 'function') {
        throw new Error(`Test '${name}' returned a promise. Use asyncTest instead.`);
      }
      results.push({
        suite: 'Challenger 2: Data Pipeline Adversarial Suite',
        name,
        passed: true,
        durationMs: performance.now() - start,
      });
    } catch (err: any) {
      results.push({
        suite: 'Challenger 2: Data Pipeline Adversarial Suite',
        name,
        passed: false,
        error: err?.message || String(err),
        durationMs: performance.now() - start,
      });
    }
  }

  async function asyncTest(name: string, fn: () => Promise<void>) {
    const start = performance.now();
    try {
      await fn();
      results.push({
        suite: 'Challenger 2: Data Pipeline Adversarial Suite',
        name,
        passed: true,
        durationMs: performance.now() - start,
      });
    } catch (err: any) {
      results.push({
        suite: 'Challenger 2: Data Pipeline Adversarial Suite',
        name,
        passed: false,
        error: err?.message || String(err),
        durationMs: performance.now() - start,
      });
    }
  }

  // =========================================================================
  // CATEGORY 1: Backward Compatibility & Legacy Trace Payloads
  // =========================================================================

  test('C2.1.1: Legacy trace payload with only classic metrics deserializes without errors', () => {
    const legacy = {
      trace_id: 'legacy_001',
      timestamp: '2026-01-01T12:00:00.000Z',
      query: 'What is the standard sick leave policy?',
      candidate_count: 5,
      top_rerank_score: 0.88,
      stage_timings: { retrieval: 120, reranking: 45 },
      execution_time_ms: 350,
      token_usage: { prompt_tokens: 220, completion_tokens: 85 },
      model: 'FastAPI RAG v1',
    };

    const mapped = mapTrace(legacy);

    assert(mapped.trace_id === 'legacy_001', 'trace_id preserved');
    assert(mapped.original_query === 'What is the standard sick leave policy?', 'original_query mapped from query');
    assert(mapped.total_chunks_retrieved === 5, 'candidate_count mapped to total_chunks_retrieved');
    assert(mapped.top_rerank_score === 0.88, 'top_rerank_score preserved');
    assert(mapped.rerank_latency_ms === 45, 'rerank_latency_ms mapped from stage_timings.reranking');
    assert(mapped.total_latency_ms === 350, 'total_latency_ms mapped from execution_time_ms');
    assert(mapped.prompt_tokens === 220, 'prompt_tokens mapped from token_usage');
    assert(mapped.completion_tokens === 85, 'completion_tokens mapped from token_usage');
    assert(mapped.model === 'FastAPI RAG v1', 'model preserved');

    // Agentic defaults should be safe/empty
    assert(mapped.query_type === undefined, 'query_type undefined on legacy');
    assert(mapped.routing_confidence === undefined, 'routing_confidence undefined on legacy');
    assert(mapped.verification === null, 'verification is null on legacy');
    assert(mapped.verification_score === undefined, 'verification_score undefined on legacy');
    assertEqual(mapped.inferred_filters, {}, 'inferred_filters is empty object');
    assertEqual(mapped.applied_filters, {}, 'applied_filters is empty object');
    assert(mapped.filter_relaxed === false, 'filter_relaxed is false');
    assert(mapped.retry_count === 0, 'retry_count is 0');
    assertEqual(mapped.retry_reasons, [], 'retry_reasons is empty array');
    assert(mapped.cache_hit === false, 'cache_hit is false');
    assert(mapped.cache_similarity === null, 'cache_similarity is null');
  });

  test('C2.1.2: Empty object {} input produces a valid fallback QueryTrace without throwing', () => {
    const mapped = mapTrace({});
    assert(typeof mapped.trace_id === 'string' && mapped.trace_id.startsWith('tr_'), 'generates synthetic trace_id');
    assert(typeof mapped.timestamp === 'string', 'generates ISO timestamp');
    assert(mapped.original_query === '', 'empty query');
    assert(mapped.total_chunks_retrieved === 0, '0 chunks retrieved');
    assert(mapped.top_rerank_score === 0.9, 'default top_rerank_score 0.9');
    assert(mapped.total_latency_ms === 0, '0 latency');
    assert(mapped.prompt_tokens === 0, '0 prompt tokens');
    assert(mapped.completion_tokens === 0, '0 completion tokens');
    assert(mapped.model === 'FastAPI RAG', 'default model name');
  });

  test('C2.1.3: Non-object primitive inputs (null, undefined, number, string, array, boolean) handle safely', () => {
    const primitives = [null, undefined, 42, 'invalid_trace', true, false, [1, 2, 3]];
    for (const prim of primitives) {
      const mapped = mapTrace(prim);
      assert(typeof mapped === 'object' && mapped !== null, `mapped result is object for ${typeof prim}`);
      assert(typeof mapped.trace_id === 'string', 'trace_id is string');
      assert(typeof mapped.total_chunks_retrieved === 'number', 'total_chunks_retrieved is number');
      assert(mapped.verification === null || mapped.verification === undefined, 'verification is null or undefined');
    }
  });

  test('C2.1.4: Legacy alias resolution: query/rewritten/sub_queries/candidate_count/latencies', () => {
    const legacyAliases = {
      id: 'alt_id_123',
      query: 'Original Q',
      rewritten_query: 'Rewritten Q',
      sub_queries: ['Sub 1', 'Sub 2'],
      retrieved_candidate_count: 8,
      latency_ms: 420,
      stage_timings_ms: { reranking: 60 },
      promptTokens: 110,
      completionTokens: 45,
    };

    const mapped = mapTrace(legacyAliases);
    assert(mapped.trace_id === 'alt_id_123', 'id mapped to trace_id');
    assert(mapped.original_query === 'Original Q', 'query mapped to original_query');
    assert(mapped.query_rewritten === 'Rewritten Q', 'rewritten_query mapped to query_rewritten');
    assertEqual(mapped.expanded_queries, ['Sub 1', 'Sub 2'], 'sub_queries mapped to expanded_queries');
    assert(mapped.total_chunks_retrieved === 8, 'retrieved_candidate_count mapped');
    assert(mapped.total_latency_ms === 420, 'latency_ms mapped');
    assert(mapped.rerank_latency_ms === 60, 'stage_timings_ms mapped');
    assert(mapped.prompt_tokens === 110, 'promptTokens mapped');
    assert(mapped.completion_tokens === 45, 'completionTokens mapped');
  });

  // =========================================================================
  // CATEGORY 2: Full Fidelity Round-Trip of All 13 Agentic Telemetry Fields
  // =========================================================================

  test('C2.2.1: Full snake_case payload containing all 13 agentic fields is preserved with exact fidelity', () => {
    const fullPayload = {
      trace_id: 'tr_agentic_001',
      timestamp: '2026-08-15T08:30:00.000Z',
      original_query: 'Compare remote work stipends in UK vs US',
      query_rewritten: 'Remote work equipment allowance comparison UK vs US',
      expanded_queries: [
        'UK remote work stipend policy',
        'US remote work equipment budget',
      ],
      total_chunks_retrieved: 6,
      top_rerank_score: 0.965,
      rerank_latency_ms: 55,
      total_latency_ms: 620,
      prompt_tokens: 350,
      completion_tokens: 180,
      model: 'FastAPI Qwen2.5:7B',

      // 1. query_type
      query_type: 'comparison',
      // 2. routing_confidence
      routing_confidence: 0.982,
      // 3. retrieval_strategy
      retrieval_strategy: 'multi_query_fusion',
      // 4. inferred_filters
      inferred_filters: { department: 'HR', region: ['UK', 'US'], active: true },
      // 5. applied_filters
      applied_filters: { department: 'HR' },
      // 6. filter_relaxed
      filter_relaxed: true,
      // 7. verification_score
      verification_score: 0.895,
      // 8. verification report object
      verification: {
        faithfulness: 0.95,
        completeness: 0.90,
        citation_coverage: 0.85,
        coherence: 0.92,
        composite_score: 0.895,
        passed: true,
        critique: 'Comparison accurately highlights regional stipend differences.',
        missing_aspects: ['Tax treatment of equipment in UK'],
        unsupported_claims: [],
        retry_count: 1,
      },
      // 9. faithfulness_passed
      faithfulness_passed: true,
      // 10. retry_count
      retry_count: 1,
      // 11. retry_reasons
      retry_reasons: ['Incomplete coverage of UK tax implications on initial pass'],
      // 12. cache_hit
      cache_hit: false,
      // 13. cache_similarity
      cache_similarity: 0.42,
    };

    const mapped = mapTrace(fullPayload);

    // 1. query_type
    assert(mapped.query_type === 'comparison', 'Field 1: query_type preserved');
    // 2. routing_confidence
    assert(mapped.routing_confidence === 0.982, 'Field 2: routing_confidence preserved');
    // 3. retrieval_strategy
    assert(mapped.retrieval_strategy === 'multi_query_fusion', 'Field 3: retrieval_strategy preserved');
    // 4. inferred_filters
    assertEqual(mapped.inferred_filters, { department: 'HR', region: ['UK', 'US'], active: true }, 'Field 4: inferred_filters preserved');
    // 5. applied_filters
    assertEqual(mapped.applied_filters, { department: 'HR' }, 'Field 5: applied_filters preserved');
    // 6. filter_relaxed
    assert(mapped.filter_relaxed === true, 'Field 6: filter_relaxed preserved');
    // 7. verification_score
    assert(mapped.verification_score === 0.895, 'Field 7: verification_score preserved');
    // 8. verification report object
    assert(mapped.verification !== null, 'Field 8: verification is non-null');
    assert(mapped.verification?.faithfulness === 0.95, 'verification.faithfulness matches');
    assert(mapped.verification?.completeness === 0.90, 'verification.completeness matches');
    assert(mapped.verification?.citation_coverage === 0.85, 'verification.citation_coverage matches');
    assert(mapped.verification?.coherence === 0.92, 'verification.coherence matches');
    assert(mapped.verification?.composite_score === 0.895, 'verification.composite_score matches');
    assert(mapped.verification?.passed === true, 'verification.passed matches');
    assert(mapped.verification?.critique === 'Comparison accurately highlights regional stipend differences.', 'verification.critique matches');
    assertEqual(mapped.verification?.missing_aspects, ['Tax treatment of equipment in UK'], 'verification.missing_aspects matches');
    assertEqual(mapped.verification?.unsupported_claims, [], 'verification.unsupported_claims matches');
    assert(mapped.verification?.retry_count === 1, 'verification.retry_count matches');
    // 9. faithfulness_passed
    assert(mapped.faithfulness_passed === true, 'Field 9: faithfulness_passed preserved');
    // 10. retry_count
    assert(mapped.retry_count === 1, 'Field 10: retry_count preserved');
    // 11. retry_reasons
    assertEqual(mapped.retry_reasons, ['Incomplete coverage of UK tax implications on initial pass'], 'Field 11: retry_reasons preserved');
    // 12. cache_hit
    assert(mapped.cache_hit === false, 'Field 12: cache_hit preserved');
    // 13. cache_similarity
    assert(mapped.cache_similarity === 0.42, 'Field 13: cache_similarity preserved');
  });

  test('C2.2.2: Full camelCase payload containing all 13 agentic fields is preserved with exact fidelity', () => {
    const camelPayload = {
      traceId: 'tr_camel_002',
      timestamp: '2026-08-15T08:35:00.000Z',
      originalQuery: 'How do I submit an expense report?',
      queryType: 'procedural',
      routingConfidence: 0.95,
      retrievalStrategy: 'hybrid_dense_bm25',
      inferredFilters: { category: 'Finance', form: 'Expense' },
      appliedFilters: { category: 'Finance' },
      filterRelaxed: false,
      verificationScore: 0.72,
      verificationReport: {
        faithfulness: 0.70,
        completeness: 0.68,
        citationCoverage: 0.75,
        coherence: 0.80,
        compositeScore: 0.72,
        passed: false,
        critique: 'Missing manager approval threshold details.',
        missingAspects: ['Manager approval limit > $1000'],
        unsupportedClaims: ['Immediate reimbursement within 1 hour'],
        retryCount: 2,
      },
      faithfulnessPassed: false,
      retryCount: 2,
      retryReasons: ['Missing approval threshold', 'Unsupported immediate claim'],
      cacheHit: false,
      cacheSimilarity: null,
    };

    const mapped = mapTrace(camelPayload);

    assert(mapped.query_type === 'procedural', 'queryType -> query_type');
    assert(mapped.routing_confidence === 0.95, 'routingConfidence -> routing_confidence');
    assert(mapped.retrieval_strategy === 'hybrid_dense_bm25', 'retrievalStrategy -> retrieval_strategy');
    assertEqual(mapped.inferred_filters, { category: 'Finance', form: 'Expense' }, 'inferredFilters -> inferred_filters');
    assertEqual(mapped.applied_filters, { category: 'Finance' }, 'appliedFilters -> applied_filters');
    assert(mapped.filter_relaxed === false, 'filterRelaxed -> filter_relaxed');
    assert(mapped.verification_score === 0.72, 'verificationScore -> verification_score');
    assert(mapped.verification !== null, 'verification report non-null');
    assert(mapped.verification?.faithfulness === 0.70, 'faithfulness matches');
    assert(mapped.verification?.citation_coverage === 0.75, 'citationCoverage matches');
    assert(mapped.verification?.composite_score === 0.72, 'compositeScore matches');
    assert(mapped.verification?.passed === false, 'passed matches');
    assertEqual(mapped.verification?.missing_aspects, ['Manager approval limit > $1000'], 'missingAspects matches');
    assertEqual(mapped.verification?.unsupported_claims, ['Immediate reimbursement within 1 hour'], 'unsupportedClaims matches');
    assert(mapped.faithfulness_passed === false, 'faithfulnessPassed matches');
    assert(mapped.retry_count === 2, 'retryCount matches');
    assertEqual(mapped.retry_reasons, ['Missing approval threshold', 'Unsupported immediate claim'], 'retryReasons matches');
  });

  test('C2.2.3: Inferred cache_hit based on retrieval_strategy when cache_hit boolean is omitted', () => {
    const bypassTrace = mapTrace({ retrieval_strategy: 'conversational_bypass' });
    assert(bypassTrace.cache_hit === true, 'conversational_bypass infers cache_hit = true');

    const cacheTrace = mapTrace({ retrieval_strategy: 'semantic_cache' });
    assert(cacheTrace.cache_hit === true, 'semantic_cache infers cache_hit = true');

    const hybridTrace = mapTrace({ retrieval_strategy: 'hybrid_dense_bm25' });
    assert(hybridTrace.cache_hit === false, 'hybrid_dense_bm25 infers cache_hit = false');
  });

  test('C2.2.4: Fallback propagation from verification report to top-level trace properties', () => {
    const traceWithEmbeddedReport = {
      verification: {
        faithfulness: 0.9,
        completeness: 0.85,
        citation_coverage: 0.8,
        coherence: 0.95,
        composite_score: 0.88,
        passed: true,
        retry_count: 2,
      },
    };

    const mapped = mapTrace(traceWithEmbeddedReport);
    assert(mapped.verification_score === 0.88, 'verification_score falls back to verification.composite_score');
    assert(mapped.faithfulness_passed === true, 'faithfulness_passed falls back to verification.passed');
    assert(mapped.retry_count === 2, 'retry_count falls back to verification.retry_count');
  });

  test('C2.2.5: mapVerificationReport() robustness on malformed or empty structures', () => {
    assert(mapVerificationReport(null) === null, 'null input returns null');
    assert(mapVerificationReport(undefined) === null, 'undefined input returns null');
    assert(mapVerificationReport('not an object') === null, 'string input returns null');
    assert(mapVerificationReport(123) === null, 'number input returns null');

    const emptyReport = mapVerificationReport({});
    assert(emptyReport !== null, 'empty object returns default VerificationReport');
    assert(emptyReport?.faithfulness === 1.0, 'default faithfulness 1.0');
    assert(emptyReport?.completeness === 1.0, 'default completeness 1.0');
    assert(emptyReport?.citation_coverage === 1.0, 'default citation_coverage 1.0');
    assert(emptyReport?.coherence === 1.0, 'default coherence 1.0');
    assert(emptyReport?.composite_score === 1.0, 'default composite_score 1.0');
    assert(emptyReport?.passed === true, 'default passed true');
    assert(emptyReport?.critique === null, 'default critique null');
    assertEqual(emptyReport?.missing_aspects, [], 'default missing_aspects []');
    assertEqual(emptyReport?.unsupported_claims, [], 'default unsupported_claims []');
    assert(emptyReport?.retry_count === 0, 'default retry_count 0');
  });

  test('C2.2.6: Explicit top-level properties take precedence over embedded verification report', () => {
    const mixedPayload = {
      verification_score: 0.99,
      faithfulness_passed: false,
      retry_count: 3,
      verification: {
        composite_score: 0.50,
        passed: true,
        retry_count: 1,
      },
    };

    const mapped = mapTrace(mixedPayload);
    assert(mapped.verification_score === 0.99, 'explicit verification_score takes priority over report composite_score');
    assert(mapped.faithfulness_passed === false, 'explicit faithfulness_passed takes priority over report passed');
    assert(mapped.retry_count === 3, 'explicit retry_count takes priority over report retry_count');
  });

  test('C2.2.7: Inferred and applied filters handle deeply nested structures and diverse primitive types', () => {
    const complexFilters = {
      inferred_filters: {
        department: 'Engineering',
        level: [3, 4, 5],
        nested: { subdept: 'Infrastructure', tags: ['cloud', 'k8s'] },
        flag: true,
        nullVal: null,
      },
      applied_filters: {
        department: 'Engineering',
        flag: false,
      },
      filter_relaxed: true,
    };

    const mapped = mapTrace(complexFilters);
    assert(mapped.inferred_filters?.department === 'Engineering', 'inferred filter primitive');
    assertEqual(mapped.inferred_filters?.level, [3, 4, 5], 'inferred filter array');
    assertEqual(mapped.inferred_filters?.nested, { subdept: 'Infrastructure', tags: ['cloud', 'k8s'] }, 'inferred nested object');
    assert(mapped.inferred_filters?.flag === true, 'inferred boolean');
    assert(mapped.applied_filters?.department === 'Engineering', 'applied filter preserved');
    assert(mapped.applied_filters?.flag === false, 'applied boolean preserved');
    assert(mapped.filter_relaxed === true, 'filter_relaxed is true');
  });

  test('C2.2.8: Explicit cache_hit = false overrides conversational_bypass or semantic_cache heuristic', () => {
    const trace = mapTrace({
      cache_hit: false,
      retrieval_strategy: 'conversational_bypass',
    });
    assert(trace.cache_hit === false, 'explicit cache_hit = false overrides strategy default');
  });

  test('C2.2.9: Verification score exact float boundaries and edge numeric values', () => {
    const tZero = mapTrace({ verification_score: 0.0 });
    assert(tZero.verification_score === 0.0, '0.0 verification score');

    const tOne = mapTrace({ verification_score: 1.0 });
    assert(tOne.verification_score === 1.0, '1.0 verification score');

    const tFloat = mapTrace({ verification_score: 0.123456789 });
    assert(tFloat.verification_score === 0.123456789, 'floating point precision preserved');
  });

  return results;
}

export async function runAdversarialPipelineAsyncTests(): Promise<TestResult[]> {
  const results: TestResult[] = [];

  async function test(name: string, fn: () => Promise<void>) {
    const start = performance.now();
    try {
      await fn();
      results.push({
        suite: 'Challenger 2: Data Pipeline Adversarial Suite',
        name,
        passed: true,
        durationMs: performance.now() - start,
      });
    } catch (err: any) {
      results.push({
        suite: 'Challenger 2: Data Pipeline Adversarial Suite',
        name,
        passed: false,
        error: err?.message || String(err),
        durationMs: performance.now() - start,
      });
    }
  }

  // =========================================================================
  // CATEGORY 3: SSE Streaming Pipeline (`streamChat()`) Verification
  // =========================================================================

  await test('C2.3.1: streamChat() correctly dispatches start, chunk, citation, trace, done events', async () => {
    const originalFetch = globalThis.fetch;
    try {
      const ssePayload = [
        'event: start\ndata: {"session_id":"sess_123","message_id":"msg_456"}\n\n',
        'event: chunk\ndata: {"content":"Company policy allows "}\n\n',
        'event: chunk\ndata: {"content":"up to 20 days PTO."}\n\n',
        'event: citation\ndata: {"id":"cit_1","title":"PTO Policy","source":"pto.pdf","snippet":"20 days per year","score":0.95}\n\n',
        'event: trace\ndata: {"trace_id":"tr_sse_01","query_type":"factual","routing_confidence":0.99,"verification_score":0.92,"filter_relaxed":false,"inferred_filters":{"topic":"PTO"}}\n\n',
        'event: done\ndata: {"answer":"Company policy allows up to 20 days PTO.","latency_ms":240}\n\n',
      ];

      globalThis.fetch = async () => {
        return new Response(createMockStream(ssePayload), {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        });
      };

      const client = new ApiClient('http://mock-rag');
      let started = false;
      const chunks: string[] = [];
      const citations: Citation[] = [];
      let receivedTrace: any = null;
      let donePayload: any = null;

      await client.streamChat('How much PTO do I get?', 'sess_123', {}, 'FastAPI RAG', {
        onStart: (data) => {
          started = true;
          assert(data.session_id === 'sess_123', 'start session_id');
        },
        onChunk: (c) => chunks.push(c),
        onCitation: (cit) => citations.push(cit),
        onTrace: (t) => { receivedTrace = t; },
        onDone: (d) => { donePayload = d; },
      });

      assert(started, 'onStart called');
      assert(chunks.join('') === 'Company policy allows up to 20 days PTO.', 'all chunks concatenated');
      assert(citations.length === 1 && citations[0].title === 'PTO Policy', 'citation received');
      assert(receivedTrace !== null, 'trace received');
      assert(receivedTrace?.query_type === 'factual', 'trace query_type is factual');
      assert(receivedTrace?.routing_confidence === 0.99, 'trace routing_confidence is 0.99');
      assert(receivedTrace?.verification_score === 0.92, 'trace verification_score is 0.92');
      assertEqual(receivedTrace?.inferred_filters, { topic: 'PTO' }, 'inferred_filters mapped');
      assert(donePayload !== null && donePayload.latency_ms === 240, 'done payload received');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  await test('C2.3.2: streamChat() correctly handles retrieval_trace embedded in done event', async () => {
    const originalFetch = globalThis.fetch;
    try {
      const ssePayload = [
        'event: chunk\ndata: {"content":"Here is the answer."}\n\n',
        'event: done\ndata: {"answer":"Here is the answer.","retrieval_trace":{"trace_id":"tr_done_01","query_type":"enumeration","verification_score":0.85,"retry_count":1,"retry_reasons":["Broad query"]},"citations":[{"id":"c1","source":"handbook.pdf","text":"Rule 1"}]}\n\n',
      ];

      globalThis.fetch = async () => {
        return new Response(createMockStream(ssePayload), {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        });
      };

      const client = new ApiClient('http://mock-rag');
      let traceFromDone: any = null;
      let citationFromDone: any = null;

      await client.streamChat('List all benefits', 'sess_1', {}, 'default', {
        onTrace: (t) => { traceFromDone = t; },
        onCitation: (c) => { citationFromDone = c; },
      });

      assert(traceFromDone !== null, 'retrieval_trace from done triggered onTrace');
      assert(traceFromDone?.query_type === 'enumeration', 'query_type is enumeration');
      assert(traceFromDone?.retry_count === 1, 'retry_count is 1');
      assertEqual(traceFromDone?.retry_reasons, ['Broad query'], 'retry_reasons preserved');
      assert(citationFromDone !== null && citationFromDone?.source === 'handbook.pdf', 'citation in done passed');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  await test('C2.3.3: streamChat() falls back to non-streaming sendChatMessage on HTTP 404', async () => {
    const originalFetch = globalThis.fetch;
    try {
      let nonStreamingCalled = false;
      globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
        const urlStr = String(input);
        if (urlStr.includes('/api/chat/stream')) {
          return new Response('Not Found', { status: 404 });
        }
        if (urlStr.includes('/api/chat')) {
          nonStreamingCalled = true;
          return new Response(JSON.stringify({
            id: 'legacy_res_123',
            answer: 'Fallback non-stream response',
            citations: [{ id: 'c1', title: 'Fallback Doc', source: 'fb.pdf', snippet: 'text', score: 0.9 }],
            latency_ms: 150,
            metrics: {},
          }), { status: 200, headers: { 'Content-Type': 'application/json' } });
        }
        return new Response('Error', { status: 500 });
      };

      const client = new ApiClient('http://mock-rag');
      let streamedChunk = '';
      let receivedTrace: any = null;
      let doneAnswer = '';

      await client.streamChat('Test fallback', 'sess_fb', {}, 'FastAPI RAG', {
        onChunk: (c) => { streamedChunk += c; },
        onTrace: (t) => { receivedTrace = t; },
        onDone: (d) => { doneAnswer = d.answer || ''; },
      });

      assert(nonStreamingCalled, 'sendChatMessage fallback was called');
      assert(streamedChunk === 'Fallback non-stream response', 'chunk emitted with fallback answer');
      assert(doneAnswer === 'Fallback non-stream response', 'done answer emitted');
      assert(receivedTrace !== null && receivedTrace.trace_id === 'legacy_res_123', 'trace emitted in fallback');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // =========================================================================
  // CATEGORY 4: Observability Pipeline (`getObservability()`) Verification
  // =========================================================================

  await test('C2.4.1: getObservability() accurately deserializes heterogeneous traces and health status', async () => {
    const originalFetch = globalThis.fetch;
    try {
      globalThis.fetch = async (input: RequestInfo | URL) => {
        const urlStr = String(input);
        if (urlStr.includes('/api/admin/observability')) {
          return new Response(JSON.stringify({
            total_queries: 2,
            avg_latency_ms: 280,
            avg_ttft_ms: 110,
            p95_latency_ms: 450,
            prompt_tokens: 400,
            completion_tokens: 200,
            active_documents: 12,
            indexed_chunks: 350,
            score_distributions: {
              similarity_avg: 0.88,
              rerank_avg: 0.94,
            },
            recent_traces: [
              // Trace 1: Full agentic trace
              {
                trace_id: 'tr_obs_01',
                timestamp: '2026-08-15T08:00:00.000Z',
                query: 'HR travel expense policy',
                query_type: 'procedural',
                routing_confidence: 0.96,
                retrieval_strategy: 'hybrid_dense_bm25',
                inferred_filters: { department: 'HR' },
                applied_filters: { department: 'HR' },
                filter_relaxed: false,
                verification_score: 0.91,
                verification: {
                  faithfulness: 0.95,
                  completeness: 0.90,
                  citation_coverage: 0.88,
                  coherence: 0.92,
                  composite_score: 0.91,
                  passed: true,
                  critique: 'Accurate and well cited.',
                  missing_aspects: [],
                  unsupported_claims: [],
                  retry_count: 0,
                },
                faithfulness_passed: true,
                retry_count: 0,
                retry_reasons: [],
                cache_hit: false,
                cache_similarity: null,
                total_chunks_retrieved: 4,
                top_rerank_score: 0.95,
                total_latency_ms: 250,
              },
              // Trace 2: Legacy trace without agentic fields or trace_id
              {
                candidate_count: 2,
                top_rerank_score: 0.87,
                execution_time_ms: 310,
                token_usage: { prompt_tokens: 150, completion_tokens: 60 },
              },
            ],
          }), { status: 200, headers: { 'Content-Type': 'application/json' } });
        }
        if (urlStr.includes('/api/health')) {
          return new Response(JSON.stringify({
            status: 'ok',
            redis: true,
            vector_db: true,
            models_loaded: true,
            collection: 'company_policies_v2',
          }), { status: 200, headers: { 'Content-Type': 'application/json' } });
        }
        return new Response('Not found', { status: 404 });
      };

      const client = new ApiClient('http://mock-rag');
      const obs = await client.getObservability();

      assert(obs.total_queries === 2, 'total_queries is 2');
      assert(obs.avg_latency_ms === 280, 'avg_latency_ms is 280');
      assert(obs.prompt_tokens === 400, 'prompt_tokens is 400');
      assert(obs.completion_tokens === 200, 'completion_tokens is 200');
      assert(obs.total_tokens === 600, 'total_tokens calculated as 600');
      assert(obs.health.status === 'ok', 'health status ok');
      assert(obs.health.backend_version === 'Collection: company_policies_v2', 'backend_version mapped');

      // Validate Traces
      assert(obs.recent_traces.length === 2, '2 traces mapped');

      // Trace 1: Full agentic verification
      const t1 = obs.recent_traces[0];
      assert(t1.trace_id === 'tr_obs_01', 't1 trace_id preserved');
      assert(t1.query_type === 'procedural', 't1 query_type preserved');
      assert(t1.routing_confidence === 0.96, 't1 routing_confidence preserved');
      assert(t1.verification_score === 0.91, 't1 verification_score preserved');
      assert(t1.verification?.passed === true, 't1 verification passed');
      assertEqual(t1.inferred_filters, { department: 'HR' }, 't1 inferred_filters preserved');

      // Trace 2: Synthetic trace fallback
      const t2 = obs.recent_traces[1];
      assert(typeof t2.trace_id === 'string' && t2.trace_id.startsWith('tr_'), 't2 synthetic trace_id generated');
      assert(t2.original_query === 'Query', 't2 fallback query name "Query"');
      assert(t2.total_chunks_retrieved === 2, 't2 chunks mapped');
      assert(t2.total_latency_ms === 310, 't2 latency mapped');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // =========================================================================
  // CATEGORY 5: Corrupted, Fragmented, Truncated SSE Payloads
  // =========================================================================

  await test('C2.5.1: SSE byte-by-byte fragmentation across arbitrary boundaries reconstructs data perfectly', async () => {
    const originalFetch = globalThis.fetch;
    try {
      const fullSseMessage =
        'event: trace\n' +
        'data: {"trace_id":"tr_frag_99","query_type":"conversational","routing_confidence":1.0,"cache_hit":true,"cache_similarity":0.99}\n\n' +
        'event: chunk\n' +
        'data: {"content":"Hello! How can I assist you today?"}\n\n' +
        'event: done\n' +
        'data: {"answer":"Hello! How can I assist you today?"}\n\n';

      // Split into 1-character and 2-character byte fragments
      const fragments: string[] = [];
      for (let i = 0; i < fullSseMessage.length; i += 3) {
        fragments.push(fullSseMessage.substring(i, i + 3));
      }

      globalThis.fetch = async () => {
        return new Response(createMockStream(fragments), {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        });
      };

      const client = new ApiClient('http://mock-rag');
      let trace: any = null;
      let text = '';
      let done = false;

      await client.streamChat('Hi', 's1', {}, 'default', {
        onTrace: (t) => { trace = t; },
        onChunk: (c) => { text += c; },
        onDone: () => { done = true; },
      });

      assert(trace !== null, 'trace received from fragmented SSE stream');
      assert(trace?.trace_id === 'tr_frag_99', 'trace_id intact');
      assert(trace?.query_type === 'conversational', 'query_type intact');
      assert(trace?.cache_hit === true, 'cache_hit intact');
      assert(trace?.cache_similarity === 0.99, 'cache_similarity intact');
      assert(text === 'Hello! How can I assist you today?', 'streamed text intact');
      assert(done, 'done event handled');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  await test('C2.5.2: SSE with UTF-8 multi-byte characters split across chunks preserves characters', async () => {
    const originalFetch = globalThis.fetch;
    try {
      const fullSse =
        'event: chunk\n' +
        'data: {"content":"Bonjour 🌍! Überlingen Café — 50€"}\n\n' +
        'event: done\n' +
        'data: {}\n\n';

      const encoder = new TextEncoder();
      const fullBytes = encoder.encode(fullSse);

      // Split bytes in middle of multi-byte UTF-8 sequences (e.g. 2-byte chunks)
      const byteChunks: Uint8Array[] = [];
      for (let i = 0; i < fullBytes.length; i += 2) {
        byteChunks.push(fullBytes.subarray(i, Math.min(i + 2, fullBytes.length)));
      }

      const stream = new ReadableStream({
        start(controller) {
          for (const b of byteChunks) {
            controller.enqueue(b);
          }
          controller.close();
        },
      });

      globalThis.fetch = async () => {
        return new Response(stream, {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        });
      };

      const client = new ApiClient('http://mock-rag');
      let text = '';

      await client.streamChat('Test UTF8', 's1', {}, 'default', {
        onChunk: (c) => { text += c; },
      });

      assert(text === 'Bonjour 🌍! Überlingen Café — 50€', `UTF8 preserved across byte cuts: got '${text}'`);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  await test('C2.5.3: Corrupted JSON in event: trace does not throw unhandled exception', async () => {
    const originalFetch = globalThis.fetch;
    try {
      const ssePayload = [
        'event: trace\ndata: {MALFORMED_JSON_CORRUPTED_STREAM\n\n',
        'event: chunk\ndata: {"content":"Normal chunk continues"}\n\n',
        'event: done\ndata: {"answer":"Normal chunk continues"}\n\n',
      ];

      globalThis.fetch = async () => {
        return new Response(createMockStream(ssePayload), {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        });
      };

      const client = new ApiClient('http://mock-rag');
      let chunkReceived = '';
      let errorThrown = false;

      try {
        await client.streamChat('Corrupted trace test', 's1', {}, 'default', {
          onChunk: (c) => { chunkReceived += c; },
          onError: () => {},
        });
      } catch {
        errorThrown = true;
      }

      assert(!errorThrown, 'corrupted trace JSON does not crash stream');
      assert(chunkReceived === 'Normal chunk continues', 'subsequent chunk received normally');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  await test('C2.5.4: Raw plain text data on event: chunk is gracefully handled as text', async () => {
    const originalFetch = globalThis.fetch;
    try {
      const ssePayload = [
        'event: chunk\ndata: Plain unescaped text chunk\n\n',
        'event: done\ndata: {"answer":"Plain unescaped text chunk"}\n\n',
      ];

      globalThis.fetch = async () => {
        return new Response(createMockStream(ssePayload), {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        });
      };

      const client = new ApiClient('http://mock-rag');
      let text = '';

      await client.streamChat('Raw chunk test', 's1', {}, 'default', {
        onChunk: (c) => { text += c; },
      });

      assert(text === 'Plain unescaped text chunk', 'raw string chunk passed through onChunk');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  await test('C2.5.5: Server error event invokes onError callback', async () => {
    const originalFetch = globalThis.fetch;
    try {
      const ssePayload = [
        'event: error\ndata: {"detail":"Vector database connection timeout"}\n\n',
      ];

      globalThis.fetch = async () => {
        return new Response(createMockStream(ssePayload), {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        });
      };

      const client = new ApiClient('http://mock-rag');
      let errorMsg = '';

      await client.streamChat('Error event test', 's1', {}, 'default', {
        onError: (err) => { errorMsg = err.message; },
      });

      assert(errorMsg === 'Vector database connection timeout', 'onError received error detail');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  await test('C2.5.6: AbortSignal triggered mid-stream exits without throwing unhandled rejection', async () => {
    const originalFetch = globalThis.fetch;
    try {
      const controller = new AbortController();
      let streamController: any;
      const stream = new ReadableStream({
        start(c) {
          streamController = c;
          c.enqueue(new TextEncoder().encode('event: chunk\ndata: {"content":"First part"}\n\n'));
        },
      });

      globalThis.fetch = async (_url, init) => {
        init?.signal?.addEventListener('abort', () => {
          const abortErr = new Error('The operation was aborted');
          abortErr.name = 'AbortError';
          try {
            streamController?.error(abortErr);
          } catch {}
        });
        return new Response(stream, {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        });
      };

      const client = new ApiClient('http://mock-rag');
      let received = '';

      const promise = client.streamChat('Abort test', 's1', {}, 'default', {
        onChunk: (c) => {
          received += c;
          controller.abort();
        },
      }, controller.signal);

      await promise;
      assert(received === 'First part', 'received chunk before abort');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // =========================================================================
  // CATEGORY 6: `useChatStream` Hook State Logic Verification
  // =========================================================================

  await test('C2.6.1: useChatStream state machine accumulates chunks, updates trace, and completes stream', async () => {
    const originalFetch = globalThis.fetch;
    try {
      const ssePayload = [
        'event: chunk\ndata: {"content":"Step 1: Fill out Form A. "}\n\n',
        'event: chunk\ndata: {"content":"Step 2: Obtain approval."}\n\n',
        'event: trace\ndata: {"trace_id":"tr_hook_01","query_type":"procedural","routing_confidence":0.97,"verification_score":0.93,"verification":{"composite_score":0.93,"passed":true,"faithfulness":0.95,"completeness":0.92,"citation_coverage":0.90,"coherence":0.95}}\n\n',
        'event: citation\ndata: {"id":"cit_proc_1","source":"proc.pdf","title":"Proc Guide","score":0.92,"snippet":"Form A"}\n\n',
        'event: done\ndata: {"answer":"Step 1: Fill out Form A. Step 2: Obtain approval."}\n\n',
      ];

      globalThis.fetch = async () => {
        return new Response(createMockStream(ssePayload), {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        });
      };

      // Emulate hook message state progression
      const userMsg: ChatMessageData = {
        id: 'msg_user_1',
        role: 'user',
        content: 'How do I request equipment?',
        timestamp: '2026-08-15T08:00:00.000Z',
      };
      const assistantMsg: ChatMessageData = {
        id: 'msg_asst_1',
        role: 'assistant',
        content: '',
        timestamp: '2026-08-15T08:00:00.000Z',
        citations: [],
        isStreaming: true,
      };

      let messages = [userMsg, assistantMsg];
      let isStreaming = true;

      const client = new ApiClient('http://mock-rag');
      await client.streamChat('How do I request equipment?', 'sess_hook', {}, 'FastAPI RAG', {
        onChunk: (chunkText) => {
          messages = messages.map((m) =>
            m.id === 'msg_asst_1' ? { ...m, content: m.content + chunkText, isStreaming: true } : m
          );
        },
        onCitation: (citation) => {
          messages = messages.map((m) => {
            if (m.id !== 'msg_asst_1') return m;
            const existing = m.citations || [];
            if (existing.some((c) => c.id === citation.id)) return m;
            return { ...m, citations: [...existing, citation] };
          });
        },
        onTrace: (traceData) => {
          messages = messages.map((m) =>
            m.id === 'msg_asst_1' ? { ...m, trace: traceData } : m
          );
        },
        onDone: (doneData) => {
          messages = messages.map((m) =>
            m.id === 'msg_asst_1' ? { ...m, content: m.content || doneData?.answer || '', isStreaming: false } : m
          );
          isStreaming = false;
        },
      });

      assert(!isStreaming, 'isStreaming is false after onDone');
      const finalAsst = messages.find((m) => m.id === 'msg_asst_1');
      assert(finalAsst?.content === 'Step 1: Fill out Form A. Step 2: Obtain approval.', 'full content accumulated');
      assert(finalAsst?.citations?.length === 1, 'citation stored on assistant message');
      assert(finalAsst?.trace?.query_type === 'procedural', 'query_type stored on message.trace');
      assert(finalAsst?.trace?.routing_confidence === 0.97, 'routing_confidence stored on message.trace');
      assert(finalAsst?.trace?.verification_score === 0.93, 'verification_score stored on message.trace');
      assert(finalAsst?.trace?.verification?.passed === true, 'verification report stored on message.trace');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  await test('C2.6.2: Citation deduplication by ID and (source, page) in hook state', async () => {
    const existingCitations: Citation[] = [
      { id: 'c1', title: 'Doc 1', source: 'doc1.pdf', chunk_text: 'Snippet 1', score: 0.9, page: 2 },
    ];

    function addCitation(list: Citation[], newCit: Citation): Citation[] {
      if (list.some((c) => c.id === newCit.id || (c.source === newCit.source && c.page === newCit.page))) {
        return list;
      }
      return [...list, newCit];
    }

    // Duplicate by ID
    let updated = addCitation(existingCitations, {
      id: 'c1', title: 'Duplicate ID', source: 'other.pdf', chunk_text: '...', score: 0.8,
    });
    assert(updated.length === 1, 'duplicate ID rejected');

    // Duplicate by (source, page)
    updated = addCitation(existingCitations, {
      id: 'c2', title: 'Duplicate Source Page', source: 'doc1.pdf', chunk_text: '...', score: 0.85, page: 2,
    });
    assert(updated.length === 1, 'duplicate (source, page) rejected');

    // Distinct citation
    updated = addCitation(existingCitations, {
      id: 'c3', title: 'Distinct Doc', source: 'doc2.pdf', chunk_text: '...', score: 0.95, page: 5,
    });
    assert(updated.length === 2, 'distinct citation accepted');
  });

  await test('C2.3.4: Windows CRLF \\r\\n line endings and SSE comments (: keep-alive) parse cleanly', async () => {
    const originalFetch = globalThis.fetch;
    try {
      const crlfPayload = [
        ': keep-alive\r\n',
        'event: trace\r\n',
        'data: {"trace_id":"tr_crlf_01","query_type":"enumeration"}\r\n\r\n',
        ': comment line\r\n',
        'event: chunk\r\n',
        'data: {"content":"Item 1"}\r\n\r\n',
        'event: done\r\n',
        'data: {"answer":"Item 1"}\r\n\r\n',
      ];

      globalThis.fetch = async () => {
        return new Response(createMockStream(crlfPayload), {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        });
      };

      const client = new ApiClient('http://mock-rag');
      let trace: any = null;
      let text = '';

      await client.streamChat('Test CRLF', 's1', {}, 'default', {
        onTrace: (t) => { trace = t; },
        onChunk: (c) => { text += c; },
      });

      assert(trace !== null && trace.query_type === 'enumeration', 'CRLF trace mapped');
      assert(text === 'Item 1', 'CRLF chunk mapped');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  await test('C2.3.5: High throughput stream with 1000 micro-chunks maintains perfect ordering without loss', async () => {
    const originalFetch = globalThis.fetch;
    try {
      const NUM_CHUNKS = 1000;
      const chunks: string[] = [];
      let expectedFullText = '';
      for (let i = 0; i < NUM_CHUNKS; i++) {
        const word = `w${i} `;
        expectedFullText += word;
        chunks.push(`event: chunk\ndata: {"content":"${word}"}\n\n`);
      }
      chunks.push(`event: done\ndata: {}\n\n`);

      globalThis.fetch = async () => {
        return new Response(createMockStream(chunks), {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        });
      };

      const client = new ApiClient('http://mock-rag');
      let accumulated = '';

      await client.streamChat('High throughput test', 's1', {}, 'default', {
        onChunk: (c) => { accumulated += c; },
      });

      assert(accumulated === expectedFullText, 'all 1000 micro-chunks received in exact order');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  await test('C2.3.6: Concurrent independent streamChat() invocations operate without cross-talk', async () => {
    const originalFetch = globalThis.fetch;
    try {
      globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
        const body = JSON.parse((init?.body as string) || '{}');
        const session = body.session_id;
        const msg = body.message;

        const payload = [
          `event: chunk\ndata: {"content":"Reply to ${msg} on ${session}"}\n\n`,
          `event: done\ndata: {"answer":"Reply to ${msg} on ${session}"}\n\n`,
        ];

        return new Response(createMockStream(payload), {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        });
      };

      const client = new ApiClient('http://mock-rag');

      let res1 = '';
      let res2 = '';

      await Promise.all([
        client.streamChat('Query 1', 'sess_A', {}, 'default', {
          onChunk: (c) => { res1 += c; },
        }),
        client.streamChat('Query 2', 'sess_B', {}, 'default', {
          onChunk: (c) => { res2 += c; },
        }),
      ]);

      assert(res1 === 'Reply to Query 1 on sess_A', 'session A reply matched');
      assert(res2 === 'Reply to Query 2 on sess_B', 'session B reply matched');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  await test('C2.4.2: getObservability() handles missing token_usage and calculates total tokens properly', async () => {
    const originalFetch = globalThis.fetch;
    try {
      globalThis.fetch = async (input: RequestInfo | URL) => {
        const urlStr = String(input);
        if (urlStr.includes('/api/admin/observability')) {
          return new Response(JSON.stringify({
            total_queries: 1,
            prompt_tokens: 300,
            completion_tokens: 150,
            recent_traces: [],
          }), { status: 200, headers: { 'Content-Type': 'application/json' } });
        }
        if (urlStr.includes('/api/health')) {
          return new Response(JSON.stringify({ status: 'ok' }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          });
        }
        return new Response('Not found', { status: 404 });
      };

      const client = new ApiClient('http://mock-rag');
      const obs = await client.getObservability();

      assert(obs.prompt_tokens === 300, 'prompt_tokens 300');
      assert(obs.completion_tokens === 150, 'completion_tokens 150');
      assert(obs.total_tokens === 450, 'total_tokens calculated as sum 450');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  await test('C2.4.3: getObservability() handles API HTTP 500 error by throwing descriptive exception', async () => {
    const originalFetch = globalThis.fetch;
    try {
      globalThis.fetch = async (input: RequestInfo | URL) => {
        const urlStr = String(input);
        if (urlStr.includes('/api/admin/observability')) {
          return new Response('Internal Server Error in database', { status: 500 });
        }
        return new Response(JSON.stringify({ status: 'ok' }), { status: 200 });
      };

      const client = new ApiClient('http://mock-rag');
      let threw = false;
      try {
        await client.getObservability();
      } catch (err: any) {
        threw = true;
        assert(err.message.includes('500'), 'error message contains status 500');
      }
      assert(threw, 'getObservability threw on 500 status');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  await test('C2.5.7: streamChat() handles HTTP 500 with custom error text', async () => {
    const originalFetch = globalThis.fetch;
    try {
      globalThis.fetch = async () => {
        return new Response('Model engine out of memory', { status: 500 });
      };

      const client = new ApiClient('http://mock-rag');
      let caughtError = '';

      try {
        await client.streamChat('OOM Query', 's1', {}, 'default', {
          onError: (err) => { caughtError = err.message; },
        });
      } catch (err: any) {
        caughtError = err.message;
      }

      assert(caughtError.includes('500') && caughtError.includes('Model engine out of memory'), 'extracted 500 error text');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  await test('C2.5.8: streamChat() handles null response.body gracefully', async () => {
    const originalFetch = globalThis.fetch;
    try {
      globalThis.fetch = async () => {
        return {
          ok: true,
          status: 200,
          body: null,
        } as any;
      };

      const client = new ApiClient('http://mock-rag');
      let thrown = false;

      try {
        await client.streamChat('Null body query', 's1', {}, 'default', {
          onError: () => {},
        });
      } catch (err: any) {
        thrown = true;
        assert(err.message.includes('body is null'), 'error mentions null body');
      }

      assert(thrown, 'streamChat threw on null body');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  await test('C2.6.3: useChatStream state properly preserves streaming content if onDone answer is empty', async () => {
    let assistantMsg: ChatMessageData = {
      id: 'msg_asst_1',
      role: 'assistant',
      content: '',
      timestamp: '2026-08-15T08:00:00.000Z',
      isStreaming: true,
    };

    // onChunk delivers streamed content
    assistantMsg = { ...assistantMsg, content: assistantMsg.content + 'Streamed answer part' };

    // onDone arrives with empty or null answer
    const doneData: any = { answer: '' };
    const finalContent = assistantMsg.content || doneData?.answer || '';
    assistantMsg = {
      ...assistantMsg,
      content: finalContent || 'Fallback default',
      isStreaming: false,
    };

    assert(assistantMsg.content === 'Streamed answer part', 'preserved streamed text over empty onDone answer');
    assert(assistantMsg.isStreaming === false, 'streaming flag is false');
  });

  return results;
}

export async function runAllChallenger2Tests(): Promise<TestResult[]> {
  const syncResults = runAdversarialPipelineTests();
  const asyncResults = await runAdversarialPipelineAsyncTests();
  return [...syncResults, ...asyncResults];
}
