import { useState, useEffect, useCallback, useRef } from 'react';
import { ChatSession, ChatMessageData } from '../lib/types';
import { generateId } from '../lib/utils';

const STORAGE_KEY = 'rag_company_policy_sessions_v2';

function createBlankSession(title = 'New Conversation'): ChatSession {
  const now = new Date().toISOString();
  return {
    id: generateId('session'),
    title,
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

export function useSessions() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>('');
  const [isLoaded, setIsLoaded] = useState(false);

  // Load from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed: ChatSession[] = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          setSessions(parsed);
          setActiveSessionId(parsed[0].id);
          setIsLoaded(true);
          return;
        }
      }
    } catch (e) {
      console.warn('Failed to parse sessions from localStorage', e);
    }
    const initial = createBlankSession();
    setSessions([initial]);
    setActiveSessionId(initial.id);
    setIsLoaded(true);
  }, []);

  // Save sessions to localStorage whenever sessions change
  const persistSessions = useCallback((updated: ChatSession[]) => {
    setSessions(updated);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    } catch (e) {
      console.warn('Failed to save sessions to localStorage', e);
    }
  }, []);

  const activeSession = sessions.find((s) => s.id === activeSessionId) || sessions[0];

  const createNewSession = useCallback(
    (title = 'New Conversation'): ChatSession => {
      const newSession = createBlankSession(title);
      setSessions((prev) => {
        const updated = [newSession, ...prev];
        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
        } catch (e) {
          console.warn('Failed to save sessions', e);
        }
        return updated;
      });
      setActiveSessionId(newSession.id);
      return newSession;
    },
    []
  );

  const switchSession = useCallback((id: string) => {
    setActiveSessionId(id);
  }, []);

  const deleteSession = useCallback(
    (id: string) => {
      setSessions((prev) => {
        const remaining = prev.filter((s) => s.id !== id);
        let nextSessions = remaining;
        let nextActiveId = activeSessionId;

        if (remaining.length === 0) {
          const fresh = createBlankSession();
          nextSessions = [fresh];
          nextActiveId = fresh.id;
        } else if (activeSessionId === id) {
          nextActiveId = remaining[0].id;
        }

        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(nextSessions));
        } catch (e) {
          console.warn('Failed to save sessions after delete', e);
        }

        setActiveSessionId(nextActiveId);
        return nextSessions;
      });

      // Background sync with backend
      try {
        fetch(`/api/chat/session/${id}`, { method: 'DELETE' }).catch(() => null);
      } catch {
        // ignore
      }
    },
    [activeSessionId]
  );

  const renameSession = useCallback(
    (id: string, newTitle: string) => {
      if (!newTitle.trim()) return;
      setSessions((prev) => {
        const updated = prev.map((s) =>
          s.id === id
            ? { ...s, title: newTitle.trim(), updatedAt: new Date().toISOString() }
            : s
        );
        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
        } catch (e) {
          console.warn('Failed to save renamed session', e);
        }
        return updated;
      });
    },
    []
  );

  const updateSessionMessages = useCallback(
    (sessionId: string, messages: ChatMessageData[]) => {
      if (!sessionId) return;
      setSessions((prev) => {
        const targetIndex = prev.findIndex((s) => s.id === sessionId);
        if (targetIndex === -1) return prev;

        const target = prev[targetIndex];
        // Auto-generate title from first user message if current title is default
        let title = target.title;
        if (
          (title === 'New Conversation' || title === 'New Chat Session' || title === 'Company Policy Inquiry') &&
          messages.length > 0
        ) {
          const firstUserMsg = messages.find((m) => m.role === 'user');
          if (firstUserMsg && firstUserMsg.content) {
            const cleanContent = firstUserMsg.content.trim().replace(/\n/g, ' ');
            title = cleanContent.length > 38 ? cleanContent.substring(0, 38) + '...' : cleanContent;
          }
        }

        const updatedSession: ChatSession = {
          ...target,
          title,
          messages,
          updatedAt: new Date().toISOString(),
        };

        const updated = [...prev];
        updated[targetIndex] = updatedSession;

        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
        } catch (e) {
          console.warn('Failed to save session messages', e);
        }
        return updated;
      });
    },
    []
  );

  return {
    sessions,
    activeSession,
    activeSessionId,
    isLoaded,
    createNewSession,
    switchSession,
    deleteSession,
    renameSession,
    updateSessionMessages,
  };
}
