import { useState, useRef, useCallback } from 'react';
import { ChatMessageData, Citation, QueryTrace, FilterOptions } from '../lib/types';
import { apiClient } from '../lib/api-client';
import { generateId } from '../lib/utils';

export function useChatStream(initialMessages: ChatMessageData[] = []) {
  const [messages, setMessages] = useState<ChatMessageData[]>(initialMessages);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);

  const openCitation = useCallback((citation: Citation) => {
    setActiveCitation(citation);
    setIsDrawerOpen(true);
  }, []);

  const closeCitationDrawer = useCallback(() => {
    setIsDrawerOpen(false);
  }, []);

  const cancelStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsStreaming(false);
    }
  }, []);

  const sendMessage = useCallback(
    async (
      content: string,
      sessionId: string,
      filters?: FilterOptions,
      model = 'FastAPI RAG'
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
      const initialAssistantMsg: ChatMessageData = {
        id: assistantMsgId,
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
        citations: [],
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMsg, initialAssistantMsg]);
      setIsStreaming(true);

      const controller = new AbortController();
      abortControllerRef.current = controller;

      // Retry up to 3 times with backoff — handles model warm-up on first request
      const MAX_RETRIES = 3;
      const RETRY_DELAY_MS = 800;

      for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        try {
          await apiClient.streamChat(
            content,
            sessionId,
            filters,
            model,
            {
              onStart: (_data) => {
                // session/message id sync handled upstream
              },
              onChunk: (chunkText) => {
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMsgId
                      ? { ...msg, content: msg.content + chunkText }
                      : msg
                  )
                );
              },
              onCitation: (citation) => {
                setMessages((prev) =>
                  prev.map((msg) => {
                    if (msg.id !== assistantMsgId) return msg;
                    const existing = msg.citations || [];
                    if (existing.some((c) => c.id === citation.id)) return msg;
                    return { ...msg, citations: [...existing, citation] };
                  })
                );
              },
              onTrace: (traceData) => {
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMsgId ? { ...msg, trace: traceData } : msg
                  )
                );
              },
              onDone: () => {
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMsgId ? { ...msg, isStreaming: false } : msg
                  )
                );
                setIsStreaming(false);
                abortControllerRef.current = null;
              },
              onError: (err) => {
                // Only surface error after last attempt
                if (attempt === MAX_RETRIES) {
                  console.error('Stream error (all retries exhausted):', err);
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
                              'Unable to connect to backend. Please ensure the FastAPI server is running on port 8001.',
                          }
                        : msg
                    )
                  );
                  setIsStreaming(false);
                  abortControllerRef.current = null;
                }
              },
            },
            controller.signal
          );
          // Success — exit retry loop
          break;
        } catch (err) {
          if ((err as Error).name === 'AbortError') {
            setIsStreaming(false);
            return;
          }
          if (attempt < MAX_RETRIES) {
            // Wait before retrying
            await new Promise((res) => setTimeout(res, RETRY_DELAY_MS * attempt));
            continue;
          }
          // Final attempt failed
          const errMsg = (err as Error).message || 'Failed to communicate with RAG API';
          setError(errMsg);
          setIsStreaming(false);
        }
      }
    },
    [isStreaming]
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
