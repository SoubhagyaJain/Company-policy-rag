/**
 * Master E2E & Component Test Runner for Agentic Intelligence UI Indicators.
 *
 * Runs all 4 Test Tiers:
 * - Tier 1: Feature Coverage (11 features x 5 test cases = 55 tests)
 * - Tier 2: Boundary & Corner Cases (55 tests)
 * - Tier 3: Cross-Feature Pairwise Combinations (21 tests)
 * - Tier 4: Real-World Workload Scenarios (6 scenarios)
 *
 * Total Test Count: 137 tests
 */

import { runTier1Tests, TestResult } from './tier1_features.test';
import { runTier2Tests } from './tier2_boundaries.test';
import { runTier3Tests } from './tier3_combinations.test';
import { runTier4Tests } from './tier4_scenarios.test';
import { runAdversarialTests as runChallenger1Tests } from './adversarial_challenger1.test';
import { runAllChallenger2Tests } from './adversarial_challenger2.test';
import { runMilestone4Tests } from './milestone4_thinking_ui.test';
import { runAdversarialM4ChallengerTests } from './adversarial_challenger_m4_1.test';

async function main() {
  console.log('\n' + '='.repeat(80));
  console.log('  AGENTIC INTELLIGENCE UI INDICATORS — COMPREHENSIVE TEST SUITE');
  console.log('='.repeat(80) + '\n');

  const startTime = performance.now();

  const ch2Results = await runAllChallenger2Tests();

  const allResults: TestResult[] = [
    ...runTier1Tests(),
    ...runTier2Tests(),
    ...runTier3Tests(),
    ...runTier4Tests(),
    ...runChallenger1Tests(),
    ...ch2Results,
    ...runMilestone4Tests(),
    ...runAdversarialM4ChallengerTests(),
  ];

  const totalDuration = performance.now() - startTime;

  // Group by suite
  const suites: Record<string, TestResult[]> = {};
  for (const res of allResults) {
    if (!suites[res.suite]) {
      suites[res.suite] = [];
    }
    suites[res.suite].push(res);
  }

  let totalPassed = 0;
  let totalFailed = 0;

  for (const [suiteName, results] of Object.entries(suites)) {
    const passedInSuite = results.filter((r) => r.passed).length;
    const failedInSuite = results.filter((r) => !r.passed).length;
    totalPassed += passedInSuite;
    totalFailed += failedInSuite;

    console.log(`\n▶ ${suiteName} (${passedInSuite}/${results.length} passed)`);
    console.log('-'.repeat(80));

    for (const r of results) {
      const statusSymbol = r.passed ? '✓ PASS' : '✗ FAIL';
      const durationStr = `${r.durationMs.toFixed(1)}ms`;
      console.log(`  ${statusSymbol} [${durationStr.padStart(7)}] ${r.name}`);
      if (!r.passed && r.error) {
        console.log(`         ERROR: ${r.error}`);
      }
    }
  }

  console.log('\n' + '='.repeat(80));
  console.log('  TEST EXECUTION SUMMARY');
  console.log('='.repeat(80));
  console.log(`  Total Suites:   ${Object.keys(suites).length}`);
  console.log(`  Total Tests:    ${allResults.length}`);
  console.log(`  Passed:         ${totalPassed}`);
  console.log(`  Failed:         ${totalFailed}`);
  console.log(`  Success Rate:   ${((totalPassed / allResults.length) * 100).toFixed(1)}%`);
  console.log(`  Total Duration: ${totalDuration.toFixed(2)}ms`);
  console.log('='.repeat(80) + '\n');

  if (totalFailed > 0) {
    console.error(`❌ Test run FAILED with ${totalFailed} failure(s).`);
    process.exit(1);
  } else {
    console.log('✅ ALL TESTS PASSED SUCCESSFULLY! (100% Pass Rate)');
    process.exit(0);
  }
}

main().catch((err) => {
  console.error('Fatal test runner error:', err);
  process.exit(1);
});
