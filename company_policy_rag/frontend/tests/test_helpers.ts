/**
 * Test Helpers for Agentic Intelligence UI Indicators E2E Test Suite.
 */

import React from 'react';
import { renderToString } from 'react-dom/server';
import { ChatMessageData, QueryTrace, ObservabilityData } from '../lib/types';
import { ChatMessage } from '../components/ChatMessage';
import { AdminView } from '../components/AdminView';

export interface RenderChatOptions {
  expanded?: boolean;
}

export interface RenderAdminOptions {
  expandedTraceId?: string | null;
}

const internals = (React as any).__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED;

let isExpandedMode = false;
let activeObsData: ObservabilityData | null = null;
let activeExpandedTraceId: string | null = null;
let currentDisp: any = internals?.ReactCurrentDispatcher?.current;

if (internals?.ReactCurrentDispatcher) {
  Object.defineProperty(internals.ReactCurrentDispatcher, 'current', {
    configurable: true,
    get() {
      if (currentDisp) {
        return {
          ...currentDisp,
          useState: (init: any) => {
            if (activeObsData && init && typeof init === 'object' && 'recent_traces' in init) {
              return [activeObsData, () => {}];
            }
            if (activeExpandedTraceId !== null && init === null) {
              return [activeExpandedTraceId, () => {}];
            }
            if (isExpandedMode && typeof init === 'boolean') {
              return [true, () => {}];
            }
            return [init, () => {}];
          },
        };
      }
      return currentDisp;
    },
    set(val) {
      currentDisp = val;
    },
  });
}

function cleanHtml(html: string): string {
  // Strip React SSR internal comment markers: <!-- -->
  return html.replace(/<!--.*?-->/g, '');
}

/**
 * Render ChatMessage component to HTML string, supporting both collapsed header
 * and fully expanded reasoning banner modes.
 */
export function renderChatMessage(
  message: ChatMessageData,
  options: RenderChatOptions = {}
): string {
  const prevExpanded = isExpandedMode;
  isExpandedMode = Boolean(options.expanded);
  try {
    const raw = renderToString(
      React.createElement(ChatMessage, { message, onOpenCitation: () => {} })
    );
    return cleanHtml(raw);
  } finally {
    isExpandedMode = prevExpanded;
  }
}

/**
 * Render ChatMessage directly with a QueryTrace object.
 */
export function renderMessageWithTrace(
  trace: Partial<QueryTrace>,
  options: RenderChatOptions = {}
): string {
  const fullTrace: QueryTrace = {
    trace_id: 'tr_test_helper',
    timestamp: '2026-08-15T08:00:00.000Z',
    original_query: 'test query helper',
    total_chunks_retrieved: 2,
    top_rerank_score: 0.92,
    rerank_latency_ms: 25,
    total_latency_ms: 180,
    prompt_tokens: 100,
    completion_tokens: 50,
    model: 'FastAPI RAG',
    ...trace,
  };

  const message: ChatMessageData = {
    id: 'msg_test_helper',
    role: 'assistant',
    content: 'Policy answer text content.',
    timestamp: '2026-08-15T08:00:00.000Z',
    trace: fullTrace,
  };

  return renderChatMessage(message, options);
}

/**
 * Render AdminView component with custom mock ObservabilityData and optional expanded row.
 */
export function renderAdminView(
  obsData?: Partial<ObservabilityData>,
  options: RenderAdminOptions = {}
): string {
  const fullObs: ObservabilityData = {
    total_queries: obsData?.recent_traces?.length || 0,
    avg_latency_ms: 320,
    avg_ttft_ms: 150,
    p95_latency_ms: 500,
    prompt_tokens: 500,
    completion_tokens: 300,
    total_tokens: 800,
    active_documents: 5,
    indexed_chunks: 120,
    similarity_avg: 0.9,
    rerank_avg: 0.92,
    health: {
      status: 'ok',
      redis: true,
      vector_db: true,
      models_loaded: true,
      backend_version: 'FastAPI RAG',
    },
    recent_traces: [],
    ...obsData,
  };

  const prevObs = activeObsData;
  const prevExp = activeExpandedTraceId;

  activeObsData = fullObs;
  activeExpandedTraceId = options.expandedTraceId !== undefined ? options.expandedTraceId : null;

  try {
    const raw = renderToString(React.createElement(AdminView));
    return cleanHtml(raw);
  } finally {
    activeObsData = prevObs;
    activeExpandedTraceId = prevExp;
  }
}

export function assert(condition: any, msg: string) {
  if (!condition) {
    throw new Error(`Assertion failed: ${msg}`);
  }
}
