import { runAllChallenger2Tests, TestResult } from './adversarial_challenger2.test';

async function main() {
  console.log('\n' + '='.repeat(80));
  console.log('  CHALLENGER 2 — ADVERSARIAL DATA PIPELINE & STATE VERIFICATION SUITE');
  console.log('='.repeat(80) + '\n');

  const startTime = performance.now();
  const results: TestResult[] = await runAllChallenger2Tests();
  const totalDuration = performance.now() - startTime;

  let totalPassed = 0;
  let totalFailed = 0;

  for (const r of results) {
    if (r.passed) {
      totalPassed++;
      console.log(`  ✓ PASS [${r.durationMs.toFixed(1).padStart(6)}ms] ${r.name}`);
    } else {
      totalFailed++;
      console.log(`  ✗ FAIL [${r.durationMs.toFixed(1).padStart(6)}ms] ${r.name}`);
      console.log(`         ERROR: ${r.error}`);
    }
  }

  console.log('\n' + '='.repeat(80));
  console.log('  CHALLENGER 2 TEST EXECUTION SUMMARY');
  console.log('='.repeat(80));
  console.log(`  Total Tests:    ${results.length}`);
  console.log(`  Passed:         ${totalPassed}`);
  console.log(`  Failed:         ${totalFailed}`);
  console.log(`  Success Rate:   ${((totalPassed / results.length) * 100).toFixed(1)}%`);
  console.log(`  Total Duration: ${totalDuration.toFixed(2)}ms`);
  console.log('='.repeat(80) + '\n');

  if (totalFailed > 0) {
    process.exit(1);
  } else {
    process.exit(0);
  }
}

main().catch((err) => {
  console.error('Fatal runner error:', err);
  process.exit(1);
});
