/**
 * Tier 3 Cross-Feature Pairwise Combinations Test Suite for Agentic Intelligence UI Indicators.
 *
 * Requirements & Specifications:
 * - TEST_INFRA.md § Coverage Goals: Tier 3 (Cross-Feature Combinations)
 */

import {
  QueryTrace,
  VerificationReport,
  ChatMessageData,
} from '../lib/types';
import { mapTrace } from '../lib/api-client';
import {
  renderChatMessage,
  renderMessageWithTrace,
  assert,
} from './test_helpers';
import { TestResult } from './tier1_features.test';

export function runTier3Tests(): TestResult[] {
  const results: TestResult[] = [];

  function test(name: string, fn: () => void) {
    const start = performance.now();
    try {
      fn();
      results.push({
        suite: 'Tier 3: Pairwise Combinations',
        name,
        passed: true,
        durationMs: performance.now() - start,
      });
    } catch (err: any) {
      results.push({
        suite: 'Tier 3: Pairwise Combinations',
        name,
        passed: false,
        error: err?.message || String(err),
        durationMs: performance.now() - start,
      });
    }
  }

  // =========================================================================
  // PAIRWISE COMBINATIONS (21 tests)
  // =========================================================================

  test('C1: Cache Hit + Filter Relaxed simultaneously present', () => {
    const html = renderMessageWithTrace(
      {
        cache_hit: true,
        cache_similarity: 0.96,
        filter_relaxed: true,
        inferred_filters: { department: 'Legal' },
      },
      { expanded: true }
    );
    assert(html.includes('Cache Hit'), 'Cache Hit badge rendered');
    assert(html.includes('Filters Relaxed:'), 'Filter relaxation warning rendered');
    assert(html.includes('department:') && html.includes('Legal'), 'Filter chip rendered');
  });

  test('C2: Retry Count > 0 + Low Verification Score (<0.7) + Passed = False', () => {
    const html = renderMessageWithTrace(
      {
        retry_count: 2,
        retry_reasons: ['Faithfulness check failed', 'Hallucinated policy allowance'],
        verification_score: 0.55,
        faithfulness_passed: false,
        verification: {
          faithfulness: 0.50,
          completeness: 0.60,
          citation_coverage: 0.55,
          coherence: 0.65,
          composite_score: 0.55,
          passed: false,
        },
      },
      { expanded: true }
    );
    assert(html.includes('2 retries'), 'renders 2 retries');
    assert(html.includes('55% Review') || (html.includes('55%') && html.includes('Review')), 'renders amber 55% Review pill');
    assert(html.includes('Faithfulness check failed'), 'renders retry reason');
  });

  test('C3: Retry Count > 0 + High Verification Score (>=0.85) + Passed = True (Resolved Retry)', () => {
    const html = renderMessageWithTrace({
      retry_count: 1,
      retry_reasons: ['Initial revision missed specific document reference'],
      verification_score: 0.92,
      faithfulness_passed: true,
      verification: {
        faithfulness: 0.95,
        completeness: 0.90,
        citation_coverage: 0.92,
        coherence: 0.96,
        composite_score: 0.92,
        passed: true,
      },
    });
    assert(html.includes('1 retry'), 'renders 1 retry badge');
    assert(html.includes('92% Verified') || (html.includes('92%') && html.includes('Verified')), 'renders emerald 92% Verified pill');
  });

  test('C4: Conversational Query Type + Cache Hit + 0 Chunks Retrieved', () => {
    const html = renderMessageWithTrace(
      {
        query_type: 'conversational',
        routing_confidence: 0.99,
        retrieval_strategy: 'conversational_bypass',
        cache_hit: true,
        cache_similarity: 1.0,
        total_chunks_retrieved: 0,
        verification_score: 1.0,
        faithfulness_passed: true,
      },
      { expanded: true }
    );
    assert(html.includes('conversational'), 'renders conversational badge');
    assert(html.includes('Cache Hit'), 'renders Cache Hit');
    assert(html.includes('0 chunks'), 'renders 0 chunks');
    assert(html.includes('100% Verified') || (html.includes('100%') && html.includes('Verified')), 'renders 100% Verified');
  });

  test('C5: Procedural Query Type + 2 Retries + Inferred Department Filter', () => {
    const html = renderMessageWithTrace(
      {
        query_type: 'procedural',
        routing_confidence: 0.91,
        retrieval_strategy: 'procedural_graph_search',
        retry_count: 2,
        retry_reasons: ['Step 3 was incomplete', 'Added approval SLA'],
        inferred_filters: { department: 'IT_Support' },
        verification_score: 0.89,
      },
      { expanded: true }
    );
    assert(html.includes('procedural'), 'renders procedural badge');
    assert(html.includes('2 retries'), 'renders 2 retries');
    assert(html.includes('department:') && html.includes('IT_Support'), 'renders IT_Support tag');
  });

  test('C6: Comparison Query Type + Multiple Applied Filters + Expanded Queries', () => {
    const html = renderMessageWithTrace(
      {
        query_type: 'comparison',
        routing_confidence: 0.89,
        inferred_filters: { topic: 'medical_plans' },
        applied_filters: { topic: 'medical_plans', tier: 'PPO_vs_HMO' },
        expanded_queries: ['PPO deductible limits', 'HMO co-pay schedule'],
        verification_score: 0.94,
      },
      { expanded: true }
    );
    assert(html.includes('comparison'), 'renders comparison badge');
    assert(html.includes('topic:') && html.includes('medical_plans'), 'renders inferred filter tag');
    assert(html.includes('PPO deductible limits'), 'renders expanded query 1');
    assert(html.includes('HMO co-pay schedule'), 'renders expanded query 2');
  });

  test('C7: Enumeration Query Type + Filter Relaxed + Low Citation Coverage', () => {
    const html = renderMessageWithTrace(
      {
        query_type: 'enumeration',
        routing_confidence: 0.86,
        filter_relaxed: true,
        inferred_filters: { category: 'Confidential_Equipment' },
        verification: {
          faithfulness: 0.88,
          completeness: 0.85,
          citation_coverage: 0.62,
          coherence: 0.91,
          composite_score: 0.81,
          passed: true,
        },
      },
      { expanded: true }
    );
    assert(html.includes('enumeration'), 'renders enumeration badge');
    assert(html.includes('Filters Relaxed:'), 'renders filter relaxation notice');
    assert(html.includes('62%'), 'renders low citation coverage percentage');
  });

  test('C8: Factual Query Type + 100% Confidence + 0 Retries', () => {
    const html = renderMessageWithTrace({
      query_type: 'factual',
      routing_confidence: 1.0,
      retry_count: 0,
      retry_reasons: [],
      verification_score: 0.98,
      faithfulness_passed: true,
    });
    assert(html.includes('factual'), 'renders factual');
    assert(html.includes('(100%)'), 'renders 100% confidence');
    assert(!html.includes('retry') && !html.includes('retries'), 'no retry badge');
    assert(html.includes('98% Verified') || (html.includes('98%') && html.includes('Verified')), 'renders 98% Verified');
  });

  test('C9: Unknown Custom Query Type + Boolean Verification Without Invented Score', () => {
    const html = renderMessageWithTrace({
      query_type: 'multi_modal_spec',
      routing_confidence: 0.77,
      faithfulness_passed: true,
    });
    assert(html.includes('multi_modal_spec'), 'renders custom type');
    assert(html.includes('Verified'), 'renders the recorded boolean verification result');
    assert(!html.includes('95% Verified'), 'does not invent a fallback verification score');
  });

  test('C10: Cache Hit + Inferred Filters + Verification Passed', () => {
    const html = renderMessageWithTrace(
      {
        cache_hit: true,
        cache_similarity: 0.99,
        inferred_filters: { department: 'Finance', document_type: 'travel_policy' },
        verification_score: 0.96,
        faithfulness_passed: true,
      },
      { expanded: true }
    );
    assert(html.includes('Cache Hit'), 'renders Cache Hit');
    assert(html.includes('department:') && html.includes('Finance'), 'renders Finance filter');
    assert(html.includes('document_type:') && html.includes('travel_policy'), 'renders travel_policy filter');
  });

  test('C11: Filter Relaxed + Unsupported Claims + Low Faithfulness', () => {
    const report: VerificationReport = {
      faithfulness: 0.45,
      completeness: 0.70,
      citation_coverage: 0.50,
      coherence: 0.80,
      composite_score: 0.58,
      passed: false,
      unsupported_claims: ['Unlimited business class flights for domestic trips'],
    };
    const html = renderMessageWithTrace(
      {
        filter_relaxed: true,
        verification: report,
        verification_score: 0.58,
        faithfulness_passed: false,
      },
      { expanded: true }
    );
    assert(html.includes('Filters Relaxed:'), 'renders relaxation warning');
    assert(html.includes('58% Review') || (html.includes('58%') && html.includes('Review')), 'renders Review pill');
    assert(html.includes('45%'), 'renders 45% faithfulness');
  });

  test('C12: Zero Chunks Retrieved + Conversational Strategy + High Confidence', () => {
    const html = renderMessageWithTrace(
      {
        query_type: 'conversational',
        routing_confidence: 0.98,
        retrieval_strategy: 'conversational_bypass',
        total_chunks_retrieved: 0,
        verification_score: 1.0,
        faithfulness_passed: true,
      },
      { expanded: true }
    );
    assert(html.includes('0 chunks'), '0 chunks rendered');
    assert(html.includes('conversational'), 'conversational type rendered');
  });

  test('C13: 1 Retry + Missing Aspects Present + Critique Text Provided', () => {
    const critiqueMsg = 'Refined answer to include manager approval deadlines.';
    const html = renderMessageWithTrace(
      {
        retry_count: 1,
        retry_reasons: ['Missed manager approval SLA in Section 2'],
        verification: {
          faithfulness: 0.91,
          completeness: 0.88,
          citation_coverage: 0.90,
          coherence: 0.94,
          composite_score: 0.90,
          passed: true,
          critique: critiqueMsg,
          missing_aspects: ['Manager escalation timeline'],
        },
      },
      { expanded: true }
    );
    assert(html.includes('1 retry'), '1 retry badge rendered');
    assert(html.includes(critiqueMsg), 'critique message rendered');
  });

  test('C14: Cache Hit with High Similarity (99.5%) + Comparison Query', () => {
    const html = renderMessageWithTrace({
      query_type: 'comparison',
      routing_confidence: 0.93,
      cache_hit: true,
      cache_similarity: 0.995,
    });
    assert(html.includes('comparison'), 'comparison badge rendered');
    assert(html.includes('Cache Hit'), 'Cache Hit badge rendered');
    assert(html.includes('100%') || html.includes('99.5%'), 'similarity rendered');
  });

  test('C15: Low Verification Score + Faithfulness Passed = False + Filter Relaxed', () => {
    const html = renderMessageWithTrace(
      {
        verification_score: 0.40,
        faithfulness_passed: false,
        filter_relaxed: true,
      },
      { expanded: true }
    );
    assert(html.includes('40% Review') || (html.includes('40%') && html.includes('Review')), 'renders 40% Review');
    assert(html.includes('Filters Relaxed:'), 'renders Filters Relaxed');
  });

  test('C16: Multiple Expanded Queries (4) + Inferred Department + Inferred Year', () => {
    const subQ = ['Subquery 1', 'Subquery 2', 'Subquery 3', 'Subquery 4'];
    const html = renderMessageWithTrace(
      {
        expanded_queries: subQ,
        inferred_filters: { department: 'Sales', year: 2026 },
      },
      { expanded: true }
    );
    assert(html.includes('Subquery 4'), '4th subquery rendered');
    assert(html.includes('department:') && html.includes('Sales'), 'Sales tag rendered');
    assert(html.includes('year:') && html.includes('2026'), 'year tag rendered');
  });

  test('C17: Extreme Latency (12000ms) + Procedural + Retry Count 2', () => {
    const html = renderMessageWithTrace({
      total_latency_ms: 12000,
      query_type: 'procedural',
      retry_count: 2,
      retry_reasons: ['Reason 1', 'Reason 2'],
    });
    assert(html.includes('12.00 s') || html.includes('12.0s') || html.includes('12 s'), 'formats 12.00 s latency');
    assert(html.includes('procedural'), 'procedural badge rendered');
    assert(html.includes('2 retries'), '2 retries rendered');
  });

  test('C18: Zero Latency (0ms) + Cache Hit + 1.0 Cache Similarity', () => {
    const html = renderMessageWithTrace({
      total_latency_ms: 0,
      cache_hit: true,
      cache_similarity: 1.0,
    });
    assert(html.includes('0 ms') || html.includes('Thought for 0 ms'), '0 ms latency rendered');
    assert(html.includes('Cache Hit'), 'Cache Hit rendered');
    assert(html.includes('(100%)'), '100% similarity rendered');
  });

  test('C19: All 4 Dimension Scores at 100% + Factual Query + 0 Retries', () => {
    const html = renderMessageWithTrace(
      {
        query_type: 'factual',
        retry_count: 0,
        verification: {
          faithfulness: 1.0,
          completeness: 1.0,
          citation_coverage: 1.0,
          coherence: 1.0,
          composite_score: 1.0,
          passed: true,
        },
      },
      { expanded: true }
    );
    assert(html.includes('100% Verified') || (html.includes('100%') && html.includes('Verified')), '100% Verified pill');
    assert(html.includes('Faithfulness') && html.includes('100%'), '100% faithfulness');
  });

  test('C20: All 4 Dimension Scores at 0% + Enumeration Query + 2 Retries', () => {
    const html = renderMessageWithTrace(
      {
        query_type: 'enumeration',
        retry_count: 2,
        retry_reasons: ['Complete retrieval failure', 'All documents irrelevant'],
        verification: {
          faithfulness: 0.0,
          completeness: 0.0,
          citation_coverage: 0.0,
          coherence: 0.0,
          composite_score: 0.0,
          passed: false,
        },
      },
      { expanded: true }
    );
    assert(html.includes('0% Review') || (html.includes('0%') && html.includes('Review')), '0% Review pill');
    assert(html.includes('2 retries'), '2 retries badge');
    assert(html.includes('Complete retrieval failure'), 'retry reason listed');
  });

  test('C21: Dark Mode Classes & Mobile Layout with All Active Indicators', () => {
    const html = renderMessageWithTrace(
      {
        query_type: 'procedural',
        routing_confidence: 0.95,
        verification_score: 0.92,
        faithfulness_passed: true,
        cache_hit: true,
        cache_similarity: 0.98,
        retry_count: 1,
        retry_reasons: ['Added step checklist'],
        inferred_filters: { department: 'Operations' },
        filter_relaxed: true,
        verification: {
          faithfulness: 0.95,
          completeness: 0.90,
          citation_coverage: 0.92,
          coherence: 0.96,
          composite_score: 0.92,
          passed: true,
        },
      },
      { expanded: true }
    );
    assert(html.includes('dark:'), 'contains dark: Tailwind classes');
    assert(html.includes('dark:bg-'), 'contains dark background classes');
    assert(html.includes('dark:border-'), 'contains dark border classes');
    assert(html.includes('sm:') || html.includes('flex-wrap'), 'contains responsive breakpoint classes');
  });

  return results;
}
