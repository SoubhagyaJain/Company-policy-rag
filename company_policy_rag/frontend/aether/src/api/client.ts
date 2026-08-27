import type {
  ThinkingEvent,
  ThinkingDetailLevel,
  ReasoningSummary,
} from "../types/thinking";

export type CorpusScope = "all" | "policy" | "guidebook";
export type GroundingMode = "balanced" | "strict";

export interface RetrievalTrace {
  chunk_count?: number;
  chunks?: Array<{
    index: number;
    section_path?: string;
    page_number?: number;
    score?: number;
    excerpt_preview?: string;
  }>;
  stages?: Record<string, number>;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  citations?: Array<Record<string, unknown>>;
  thinking?: string;
  thinking_events?: ThinkingEvent[];
  reasoning_summary?: ReasoningSummary | null;
  retrieval_trace?: RetrievalTrace | null;
  timing?: Record<string, number> | null;
  message_id?: string;
  low_confidence?: boolean;
  isStreaming?: boolean;
}

export interface ModelInfo {
  id: string;
  label: string;
  family?: string;
  parameter_size?: string;
  quantization?: string;
  badges?: string[];
}

export interface ModelsResponse {
  connected: boolean;
  error?: string;
  active_model: string;
  models: ModelInfo[];
}

export interface HealthResponse {
  index_ready: boolean;
  chunk_count: number;
  ollama_connected: boolean;
  llm_model: string;
  embed_model: string;
}

export interface ChatResponse {
  answer: string;
  citations: Array<Record<string, unknown>>;
  timing?: Record<string, number>;
  low_confidence: boolean;
  thinking?: string;
  thinking_events?: ThinkingEvent[];
  reasoning_summary?: ReasoningSummary;
  retrieval_trace?: RetrievalTrace;
  message_id?: string;
}

export interface StreamDonePayload {
  answer: string;
  thinking?: string;
  thinking_events?: ThinkingEvent[];
  reasoning_summary?: ReasoningSummary;
  citations: Array<Record<string, unknown>>;
  timing?: Record<string, number>;
  retrieval_trace?: RetrievalTrace;
  message_id?: string;
  low_confidence: boolean;
  grounding_mode: string;
}

export interface StreamHandlers {
  onRetrievalDone?: (trace: RetrievalTrace) => void;
  onThinking?: (thinking: string) => void;
  onThinkingEvent?: (event: ThinkingEvent) => void;
  onToken?: (token: string) => void;
  onDone?: (payload: StreamDonePayload) => void;
  onError?: (message: string) => void;
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || res.statusText);
  }
  return res.json() as Promise<T>;
}

function parseSseChunk(buffer: string): { events: Array<{ event: string; data: string }>; rest: string } {
  const events: Array<{ event: string; data: string }> = [];
  const parts = buffer.split("\n\n");
  const rest = parts.pop() ?? "";

  for (const part of parts) {
    if (!part.trim()) continue;
    let event = "message";
    const dataLines: string[] = [];
    for (const line of part.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (dataLines.length) events.push({ event, data: dataLines.join("\n") });
  }
  return { events, rest };
}

export function fetchModels() {
  return api<ModelsResponse>("/models");
}

export function setActiveModel(model: string) {
  return api<{ active_model: string }>("/models/active", {
    method: "PUT",
    body: JSON.stringify({ model }),
  });
}

export function fetchHealth() {
  return api<HealthResponse>("/health");
}

export function sendChat(
  message: string,
  opts: {
    corpus_scope?: CorpusScope;
    llm_model?: string | null;
    grounding_mode?: GroundingMode;
  } = {},
) {
  return api<ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify({
      message,
      corpus_scope: opts.corpus_scope ?? "all",
      chat_mode: "direct",
      llm_model: opts.llm_model,
      grounding_mode: opts.grounding_mode,
    }),
  });
}

export async function sendChatStream(
  message: string,
  opts: {
    corpus_scope?: CorpusScope;
    llm_model?: string | null;
    grounding_mode?: GroundingMode;
    thinking_detail_level?: ThinkingDetailLevel;
  },
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      corpus_scope: opts.corpus_scope ?? "all",
      chat_mode: "direct",
      llm_model: opts.llm_model,
      grounding_mode: opts.grounding_mode,
      thinking_detail_level: opts.thinking_detail_level ?? "standard",
    }),
    signal,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || res.statusText);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("Streaming not supported");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const { events, rest } = parseSseChunk(buffer);
    buffer = rest;

    for (const { event, data } of events) {
      if (event === "retrieval" || event === "retrieval_done") {
        try {
          const payload = JSON.parse(data) as RetrievalTrace;
          handlers.onRetrievalDone?.(payload);
        } catch {
          /* ignore malformed */
        }
      } else if (event === "thinking") {
        try {
          const thinkingEv = JSON.parse(data) as ThinkingEvent;
          handlers.onThinkingEvent?.(thinkingEv);
          handlers.onThinking?.(thinkingEv.summary || data);
        } catch {
          handlers.onThinking?.(data);
        }
      } else if (event === "chunk" || event === "token") {
        try {
          const payload = JSON.parse(data) as { content?: string };
          const token = typeof payload === "string" ? payload : payload?.content ?? "";
          handlers.onToken?.(token);
        } catch {
          handlers.onToken?.(data);
        }
      } else if (event === "citation") {
        // Citation frames are informational for the UI and do not need to mutate current assistant state.
      } else if (event === "trace") {
        // Trace frames are part of the final telemetry contract; ignore here because ChatThread reads the answer from done.
      } else if (event === "done") {
        try {
          const payload = JSON.parse(data) as StreamDonePayload;
          handlers.onDone?.(payload);
        } catch {
          handlers.onError?.("Invalid stream completion payload");
        }
      } else if (event === "error") {
        try {
          const parsed = JSON.parse(data) as { detail?: string; message?: string };
          handlers.onError?.(parsed.detail || parsed.message || "Stream error");
        } catch {
          handlers.onError?.(data || "Stream error");
        }
      }
    }
  }
}

export function submitFeedback(payload: {
  rating: 1 | -1;
  question: string;
  answer: string;
  model: string;
  corpus_scope: string;
  message_id?: string;
  comment?: string;
}) {
  return api<{ ok: boolean; id: string }>("/feedback", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function runEval(max_samples = 5) {
  return api<{ job_id: string; status: string }>("/eval/run", {
    method: "POST",
    body: JSON.stringify({ max_samples }),
  });
}