import type { DonePayload } from './api-client';
import type { ChatMessageData } from './types';

/** Replace provisional text and source cards as one canonical completion. */
export function completeMessage(message: ChatMessageData, done: DonePayload): ChatMessageData {
  return {
    ...message,
    content: typeof done.answer === 'string' ? done.answer : message.content,
    citations: Array.isArray(done.citations) ? done.citations : message.citations,
    model: done.model || message.model,
    low_confidence: done.low_confidence,
    content_status: 'final',
    isStreaming: false,
  };
}
