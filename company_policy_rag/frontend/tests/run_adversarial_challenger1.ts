import { runAdversarialTests } from './adversarial_challenger1.test';

async function main() {
  console.log('\n' + '='.repeat(80));
  console.log('  CHALLENGER 1: EMPIRICAL ADVERSARIAL STRESS TEST HARNESS');
  console.log('='.repeat(80) + '\n');

  const startTime = performance.now();
  const results = runAdversarialTests();
  const duration = performance.now() - startTime;

  let passed = 0;
  let failed = 0;

  for (const r of results) {
    const status = r.passed ? '✓ PASS' : '✗ FAIL';
    const dur = `${r.durationMs.toFixed(1)}ms`.padStart(7);
    console.log(`  ${status} [${dur}] ${r.name}`);
    if (!r.passed) {
      console.log(`         ERROR: ${r.error}`);
      failed++;
    } else {
      passed++;
    }
  }

  console.log('\n' + '='.repeat(80));
  console.log(`  SUMMARY: ${passed}/${results.length} PASSED (${((passed / results.length) * 100).toFixed(1)}%) in ${duration.toFixed(2)}ms`);
  console.log('='.repeat(80) + '\n');

  if (failed > 0) {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
