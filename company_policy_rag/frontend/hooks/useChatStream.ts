import { useState, useRef, useCallback } from 'react';
import { ChatMessageData, Citation, QueryTrace, FilterOptions, ThinkingEvent, ThinkingDetailLevel, ReasoningSummary } from '../lib/types';
import { apiClient } from '../lib/api-client';
import { generateId } from '../lib/utils';

type ProgressEventInput = Pick<ThinkingEvent, 'stage' | 'status' | 'title' | 'summary'>;

function createProgressEvent(input: ProgressEventInput): ThinkingEvent {
  return {
    id: `ui_${input.stage}_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
    query_id: 'ui_progress',
    duration_ms: 0,
    ...input,
  };
}

function mergeThinkingEvent(events: ThinkingEvent[], incoming: ThinkingEvent): ThinkingEvent[] {
  const byId = events.findIndex((event) => event.id === incoming.id);
  if (byId >= 0) {
    const current = events[byId];
    if (
      current.stage === incoming.stage &&
      current.status === incoming.status &&
      current.title === incoming.title &&
      current.summary === incoming.summary &&
      current.duration_ms === incoming.duration_ms
    ) {
      return events;
    }
    const next = [...events];
    next[byId] = incoming;
    return next;
  }

  // Backend events use different ids for running and completed versions of the
  // same stage. The UI's optimistic stages do too, so stage is the stable key.
  const byStage = events.findIndex((event) => event.stage === incoming.stage);
  if (byStage >= 0) {
    const current = events[byStage];
    if (
      current.status === incoming.status &&
      current.title === incoming.title &&
      current.summary === incoming.summary &&
      current.duration_ms === incoming.duration_ms
    ) {
      return events;
    }
    const next = [...events];
    next[byStage] = incoming;
    return next;
  }

  return [...events, incoming];
}

function normalizeThinkingEvents(events: ThinkingEvent[]): ThinkingEvent[] {
  return events.reduce<ThinkingEvent[]>(
    (normalized, event) => mergeThinkingEvent(normalized, event),
    []
  );
}

export function useChatStream(initialMessages: ChatMessageData[] = []) {
  const [messages, setMessages] = useState<ChatMessageData[]>(initialMessages);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const progressTimersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  const clearProgressTimers = useCallback(() => {
    progressTimersRef.current.forEach(clearTimeout);
    progressTimersRef.current = [];
  }, []);

  const openCitation = useCallback((citation: Citation) => {
    setActiveCitation(citation);
    setIsDrawerOpen(true);
  }, []);

  const closeCitationDrawer = useCallback(() => {
    setIsDrawerOpen(false);
  }, []);

  const cancelStream = useCallback(() => {
    clearProgressTimers();
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
    setMessages((prev) =>
      prev.map((msg) => (msg.isStreaming ? { ...msg, isStreaming: false } : msg))
    );
  }, [clearProgressTimers]);

  const sendMessage = useCallback(
    async (
      content: string,
      sessionId: string,
      filters?: FilterOptions,
      model = 'FastAPI RAG',
      thinkingDetailLevel: ThinkingDetailLevel = 'standard'
    ) => {
      if (!content.trim() || isStreaming) return;

      setError(null);

      const userMsg: ChatMessageData = {
        id: generateId('msg_user'),
        role: 'user',
        content,
        timestamp: new Date().toISOString(),
      };

      const assistantMsgId = generateId('msg_assistant');
      const initialProgress = createProgressEvent({
        stage: 'received',
        status: 'running',
        title: 'Understanding your question',
        summary: 'Preparing a grounded search across your selected documents.',
      });
      const initialAssistantMsg: ChatMessageData = {
        id: assistantMsgId,
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
        citations: [],
        thinking_events: thinkingDetailLevel === 'off' ? [] : [initialProgress],
        thinking_detail_level: thinkingDetailLevel,
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMsg, initialAssistantMsg]);
      setIsStreaming(true);

      const controller = new AbortController();
      abortControllerRef.current = controller;
      const backendStages = new Set<string>();

      const updateProgress = (input: ProgressEventInput | ThinkingEvent) => {
        const event = 'id' in input ? input : createProgressEvent(input);
        setMessages((prev) => {
          let changed = false;
          const next = prev.map((msg) => {
            if (msg.id !== assistantMsgId) return msg;
            const currentEvents = msg.thinking_events || [];
            const nextEvents = mergeThinkingEvent(currentEvents, event);
            if (nextEvents === currentEvents) return msg;
            changed = true;
            return { ...msg, thinking_events: nextEvents };
          });
          return changed ? next : prev;
        });
      };
      const scheduleProgress = (delayMs: number, input: ProgressEventInput) => {
        const timer = setTimeout(() => {
          if (!controller.signal.aborted && !backendStages.has(input.stage)) {
            updateProgress(input);
          }
        }, delayMs);
        progressTimersRef.current.push(timer);
      };

      // The backend immediately confirms the request, while retrieval can take
      // several seconds on local models. These are transparent product-status
      // milestones, never hidden chain-of-thought or model-private reasoning.
      if (thinkingDetailLevel !== 'off') {
        scheduleProgress(350, {
          stage: 'query_analysis',
          status: 'running',
          title: 'Analyzing your request',
          summary: 'Choosing the right retrieval strategy for this question.',
        });
        scheduleProgress(900, {
          stage: 'retrieval',
          status: 'running',
          title: 'Searching your documents',
          summary: 'Finding the most relevant source passages.',
        });
        scheduleProgress(2_400, {
          stage: 'evidence_analysis',
          status: 'running',
          title: 'Checking evidence',
          summary: 'Reviewing the strongest passages before answering.',
        });
        scheduleProgress(4_800, {
          stage: 'answer_planning',
          status: 'running',
          title: 'Drafting a grounded answer',
          summary: 'Organizing an answer with source-backed claims.',
        });
      }

      // Retry up to 3 times with backoff — handles model warm-up on first request
      const MAX_RETRIES = 3;
      const RETRY_DELAY_MS = 800;

      // SSE can deliver dozens of token events in one network read. Updating
      // React twice per token can exceed React's nested-update guard and also
      // makes markdown rendering needlessly expensive. Coalesce tokens into a
      // single message update per short paint window.
      let pendingChunkText = '';
      let chunkFlushTimer: ReturnType<typeof setTimeout> | null = null;
      let answerGenerationStarted = false;

      const flushPendingChunks = () => {
        if (chunkFlushTimer) {
          clearTimeout(chunkFlushTimer);
          chunkFlushTimer = null;
        }
        if (!pendingChunkText || controller.signal.aborted) {
          pendingChunkText = '';
          return;
        }

        const textBatch = pendingChunkText;
        pendingChunkText = '';
        const includeGenerationProgress =
          thinkingDetailLevel !== 'off' && !answerGenerationStarted;
        answerGenerationStarted = true;

        setMessages((prev) =>
          prev.map((msg) => {
            if (msg.id !== assistantMsgId) return msg;
            let thinkingEvents = msg.thinking_events || [];
            if (includeGenerationProgress) {
              thinkingEvents = mergeThinkingEvent(
                thinkingEvents,
                createProgressEvent({
                  stage: 'answer_generation',
                  status: 'running',
                  title: 'Writing the answer',
                  summary: 'Streaming a response from verified document context.',
                })
              );
            }
            return {
              ...msg,
              content: msg.content + textBatch,
              thinking_events: thinkingEvents,
              isStreaming: true,
            };
          })
        );
      };

      const queueChunk = (chunkText: string) => {
        if (!chunkText) return;
        pendingChunkText += chunkText;
        if (!chunkFlushTimer) {
          chunkFlushTimer = setTimeout(flushPendingChunks, 32);
        }
      };

      for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        if (controller.signal.aborted) {
          setIsStreaming(false);
          abortControllerRef.current = null;
          setMessages((prev) =>
            prev.map((msg) => (msg.id === assistantMsgId ? { ...msg, isStreaming: false } : msg))
          );
          return;
        }

        try {
          await apiClient.streamChat(
            content,
            sessionId,
            filters,
            model,
            {
              onStart: (_data) => {
                updateProgress(createProgressEvent({
                  stage: 'received',
                  status: 'completed',
                  title: 'Question received',
                  summary: 'Connected to the answer pipeline.',
                }));
              },
              onThinking: (thinkingEv) => {
                if (controller.signal.aborted) return;
                backendStages.add(thinkingEv.stage);
                setMessages((prev) =>
                  prev.map((msg) => {
                    if (msg.id !== assistantMsgId) return msg;
                    return {
                      ...msg,
                      thinking_events: mergeThinkingEvent(msg.thinking_events || [], thinkingEv),
                    };
                  })
                );
              },
              onChunk: (chunkText) => {
                if (controller.signal.aborted) return;
                queueChunk(chunkText);
              },
              onCitation: (citation) => {
                if (!citation || controller.signal.aborted) return;
                setMessages((prev) =>
                  prev.map((msg) => {
                    if (msg.id !== assistantMsgId) return msg;
                    const existing = msg.citations || [];
                    if (
                      existing.some(
                        (c) =>
                          c.id === citation.id ||
                          (c.source === citation.source && c.page === citation.page)
                      )
                    )
                      return msg;
                    return { ...msg, citations: [...existing, citation] };
                  })
                );
              },
              onTrace: (traceData) => {
                if (controller.signal.aborted) return;
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMsgId ? { ...msg, trace: traceData } : msg
                  )
                );
              },
              onDone: (doneData) => {
                if (controller.signal.aborted) return;
                clearProgressTimers();
                flushPendingChunks();
                setMessages((prev) =>
                  prev.map((msg) => {
                    if (msg.id !== assistantMsgId) return msg;
                    const finalContent = msg.content || doneData?.answer || '';
                    return {
                      ...msg,
                      content: finalContent || 'I am unable to answer based on the provided documents.',
                      thinking_events:
                        doneData?.thinking_events && doneData.thinking_events.length > 0
                          ? normalizeThinkingEvents(doneData.thinking_events)
                          : msg.thinking_events,
                      reasoning_summary: doneData?.reasoning_summary || msg.reasoning_summary,
                      isStreaming: false,
                    };
                  })
                );
                setIsStreaming(false);
                abortControllerRef.current = null;
              },
              onError: (err) => {
                if (controller.signal.aborted) return;
                clearProgressTimers();
                if (chunkFlushTimer) {
                  clearTimeout(chunkFlushTimer);
                  chunkFlushTimer = null;
                }
                pendingChunkText = '';
                console.warn(`Stream attempt ${attempt} error:`, err);
                if (attempt === MAX_RETRIES) {
                  const errMsg = err.message || 'Error generating response. Is the backend running?';
                  setError(errMsg);
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMsgId
                        ? {
                            ...msg,
                            isStreaming: false,
                            error: errMsg,
                            content:
                              msg.content ||
                              'Unable to connect to backend. Please ensure the FastAPI server and Ollama are running.',
                          }
                        : msg
                    )
                  );
                  setIsStreaming(false);
                  abortControllerRef.current = null;
                }
              },
            },
            controller.signal,
            thinkingDetailLevel
          );
          // Success — exit retry loop
          break;
        } catch (err) {
          if ((err as Error).name === 'AbortError' || controller.signal.aborted) {
            if (chunkFlushTimer) clearTimeout(chunkFlushTimer);
            pendingChunkText = '';
            setIsStreaming(false);
            abortControllerRef.current = null;
            setMessages((prev) =>
              prev.map((msg) => (msg.id === assistantMsgId ? { ...msg, isStreaming: false } : msg))
            );
            return;
          }

          if (attempt < MAX_RETRIES && !controller.signal.aborted) {
            // Wait before retrying
            await new Promise((res) => setTimeout(res, RETRY_DELAY_MS * attempt));
            continue;
          }

          // Final attempt failed
          const errMsg = (err as Error).message || 'Failed to communicate with RAG API';
          setError(errMsg);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMsgId
                ? {
                    ...msg,
                    isStreaming: false,
                    error: errMsg,
                    content:
                      msg.content ||
                      'Unable to connect to backend. Please ensure the FastAPI server and Ollama are running.',
                  }
                : msg
            )
          );
          setIsStreaming(false);
          abortControllerRef.current = null;
        }
      }
    },
    [clearProgressTimers, isStreaming]
  );

  const clearMessages = useCallback(() => {
    cancelStream();
    setMessages([]);
    setActiveCitation(null);
    setIsDrawerOpen(false);
    setError(null);
  }, [cancelStream]);

  return {
    messages,
    setMessages,
    isStreaming,
    sendMessage,
    cancelStream,
    clearMessages,
    activeCitation,
    openCitation,
    isDrawerOpen,
    closeCitationDrawer,
    error,
  };
}
