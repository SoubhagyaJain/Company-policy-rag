import { ApiClient } from '../lib/api-client';
import { TestResult } from './tier1_features.test';


export async function runModelSwitchingTests(): Promise<TestResult[]> {
  const results: TestResult[] = [];

  async function test(name: string, fn: () => Promise<void>) {
    const started = performance.now();
    try {
      await fn();
      results.push({
        suite: 'Dynamic Model Switching',
        name,
        passed: true,
        durationMs: performance.now() - started,
      });
    } catch (error: any) {
      results.push({
        suite: 'Dynamic Model Switching',
        name,
        passed: false,
        error: error?.message || String(error),
        durationMs: performance.now() - started,
      });
    }
  }

  const originalFetch = global.fetch;
  try {
    await test('returns the backend-acknowledged active model', async () => {
      global.fetch = async () =>
        new Response(
          JSON.stringify({ status: 'switched', active_model: 'gemma2:2b' }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );

      const result = await new ApiClient('http://test.local').selectModel('gemma2:2b');
      if (result.active_model !== 'gemma2:2b') {
        throw new Error(`Expected gemma2:2b, received ${result.active_model}`);
      }
    });

    await test('surfaces backend switch errors instead of silently succeeding', async () => {
      global.fetch = async () =>
        new Response(
          JSON.stringify({ detail: "Model 'missing' is not available" }),
          { status: 400, headers: { 'Content-Type': 'application/json' } },
        );

      let message = '';
      try {
        await new ApiClient('http://test.local').selectModel('missing');
      } catch (error) {
        message = error instanceof Error ? error.message : String(error);
      }
      if (!message.includes('not available')) {
        throw new Error(`Expected a visible backend error, received '${message}'`);
      }
    });
  } finally {
    global.fetch = originalFetch;
  }

  return results;
}
