import { useState, useEffect, useCallback } from 'react';
import { ChatSession, ChatMessageData } from '../lib/types';
import { generateId } from '../lib/utils';

const STORAGE_KEY = 'rag_company_policy_sessions';

const DEFAULT_SESSION: ChatSession = {
  id: 'session_default',
  title: 'Company Policy Inquiry',
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  messages: [
    {
      id: 'welcome_1',
      role: 'assistant',
      content:
        'Welcome to the **Company Policy & Employee Guide RAG Portal**. Ask me any question regarding remote work, PTO benefits, expense reimbursement, or compliance guidelines!',
      timestamp: new Date().toISOString(),
      citations: [],
    },
  ],
};

export function useSessions() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>('session_default');

  // Load from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed: ChatSession[] = JSON.parse(saved);
        if (parsed.length > 0) {
          setSessions(parsed);
          setActiveSessionId(parsed[0].id);
          return;
        }
      }
    } catch (e) {
      console.warn('Failed to parse sessions from localStorage', e);
    }
    setSessions([DEFAULT_SESSION]);
    setActiveSessionId(DEFAULT_SESSION.id);
  }, []);

  // Save sessions to localStorage on change
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
    (title = 'New Chat Session'): string => {
      const newId = generateId('session');
      const newSession: ChatSession = {
        id: newId,
        title,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        messages: [],
      };
      const updated = [newSession, ...sessions];
      persistSessions(updated);
      setActiveSessionId(newId);
      return newId;
    },
    [sessions, persistSessions]
  );

  const switchSession = useCallback((id: string) => {
    setActiveSessionId(id);
  }, []);

  const deleteSession = useCallback(
    (id: string) => {
      const updated = sessions.filter((s) => s.id !== id);
      if (updated.length === 0) {
        const fallback = [
          {
            id: generateId('session'),
            title: 'New Chat Session',
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
            messages: [],
          },
        ];
        persistSessions(fallback);
        setActiveSessionId(fallback[0].id);
      } else {
        persistSessions(updated);
        if (activeSessionId === id) {
          setActiveSessionId(updated[0].id);
        }
      }
    },
    [sessions, activeSessionId, persistSessions]
  );

  const renameSession = useCallback(
    (id: string, newTitle: string) => {
      const updated = sessions.map((s) =>
        s.id === id ? { ...s, title: newTitle, updatedAt: new Date().toISOString() } : s
      );
      persistSessions(updated);
    },
    [sessions, persistSessions]
  );

  const updateSessionMessages = useCallback(
    (sessionId: string, messages: ChatMessageData[]) => {
      setSessions((prev) => {
        const updated = prev.map((s) => {
          if (s.id !== sessionId) return s;

          // Auto-generate title from first user message if title is default
          let title = s.title;
          if (
            (title === 'New Chat Session' || title === 'Company Policy Inquiry') &&
            messages.length > 0
          ) {
            const firstUserMsg = messages.find((m) => m.role === 'user');
            if (firstUserMsg && firstUserMsg.content) {
              title =
                firstUserMsg.content.length > 35
                  ? firstUserMsg.content.substring(0, 35) + '...'
                  : firstUserMsg.content;
            }
          }

          return {
            ...s,
            title,
            messages,
            updatedAt: new Date().toISOString(),
          };
        });

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
    createNewSession,
    switchSession,
    deleteSession,
    renameSession,
    updateSessionMessages,
  };
}
