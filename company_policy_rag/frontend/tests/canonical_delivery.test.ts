import { ApiClient } from '../lib/api-client';
import { completeMessage } from '../lib/chat-completion';
import type { ChatMessageData } from '../lib/types';
import type { TestResult } from './tier1_features.test';

export async function runCanonicalDeliveryTests(): Promise<TestResult[]> {
  const results: TestResult[] = [];
  const originalFetch = global.fetch;
  async function test(name: string, run: () => Promise<void> | void) {
    const start = performance.now();
    try { await run(); results.push({ suite: 'Canonical delivery', name, passed: true, durationMs: performance.now() - start }); }
    catch (error) { results.push({ suite: 'Canonical delivery', name, passed: false, durationMs: performance.now() - start, error: String(error) }); }
  }
  const response = (frames: string) => {
    const bytes = new TextEncoder().encode(frames);
    return new Response(new ReadableStream({ start(controller) {
      for (let i = 0; i < bytes.length; i += 7) controller.enqueue(bytes.slice(i, i + 7));
      controller.close();
    } }), { headers: { 'Content-Type': 'text/event-stream' } });
  };
  try {
    await test('transport and completion replace draft text and obsolete source cards', async () => {
      let message: ChatMessageData = { id: 'answer', role: 'assistant', content: '', timestamp: '', citations: [], isStreaming: true };
      global.fetch = async () => response(
        'event: chunk\ndata: {"content":"Wrong draft"}\n\n' +
        'event: citation\ndata: {"citations":[{"chunk_id":"old","source_file":"old.txt"}]}\n\n' +
        'event: done\ndata: {"answer":"Corrected café answer","model":"actual","citations":[{"chunk_id":"final","source_file":"final.txt"}]}\n\n');
      await new ApiClient().streamChat('question', 'session', undefined, undefined, {
        onChunk: text => { message.content += text; },
        onCitation: citation => { message.citations!.push(citation); },
        onDone: done => { message = completeMessage(message, done); },
      });
      if (message.content !== 'Corrected café answer' || message.citations?.length !== 1 || message.citations[0].id !== 'final') throw new Error('Draft or obsolete citations survived');
      if (message.model !== 'actual' || message.isStreaming || message.content_status !== 'final') throw new Error('Completion metadata diverged');
    });
    await test('an empty authoritative answer clears a draft', () => {
      const result = completeMessage({ id: 'a', role: 'assistant', content: 'draft', timestamp: '' }, { answer: '', citations: [] });
      if (result.content !== '') throw new Error('Empty final answer was ignored');
    });
    await test('EOF before done is an error, never successful completion', async () => {
      global.fetch = async () => response('event: chunk\ndata: {"content":"partial"}\n\n');
      let errors = 0;
      let completed = false;
      let rejected = false;
      try { await new ApiClient().streamChat('q', 's', undefined, undefined, { onError: () => { errors++; }, onDone: () => { completed = true; } }); }
      catch { rejected = true; }
      if (!rejected || completed || errors !== 1) throw new Error('Incomplete stream accepted');
    });
    await test('SSE error events reject the request exactly once', async () => {
      global.fetch = async () => response('event: error\ndata: {"detail":"generation failed"}\n\n');
      let errors = 0;
      let rejected = false;
      try { await new ApiClient().streamChat('q', 's', undefined, undefined, { onError: () => { errors++; } }); }
      catch { rejected = true; }
      if (!rejected || errors !== 1) throw new Error('Server error swallowed');
    });
  } finally { global.fetch = originalFetch; }
  return results;
}
