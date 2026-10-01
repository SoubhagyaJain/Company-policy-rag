'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';

import { tabDirection, type ViewTab } from '@/components/space/SpaceTabNav';
import { DocumentsView } from '@/components/DocumentsView';
import { AdminView } from '@/components/AdminView';
import { StorageView } from '@/components/StorageView';
import { SpaceShell } from '@/components/space/SpaceShell';
import { LibraryShell } from '@/components/space/LibraryShell';
import { AlpineBackdrop } from '@/components/space/AlpineBackdrop';

import { useChatStream } from '@/hooks/useChatStream';
import { useSessions } from '@/hooks/useSessions';
import { apiClient } from '@/lib/api-client';

import type { FilterOptions, HealthStatus, ResponseMode } from '@/lib/types';


/* Direction-aware crossfade for tab content. Custom = travel direction:
 * +1 → new tab is to the right (old exits left, new enters from right), -1 → reverse.
 * Opacity + a short translate only: both stay on the compositor. No animated
 * blur() filter and no scale — scaling a panel re-rasterizes its text and forces
 * every glass (backdrop-filter) layer inside it to be re-sampled each frame,
 * which is what made tab switches stutter on integrated GPUs. */
const panelVariants = {
  enter: (d: number) => ({ opacity: 0, x: d > 0 ? 18 : d < 0 ? -18 : 0 }),
  center: { opacity: 1, x: 0 },
  exit: (d: number) => ({ opacity: 0, x: d > 0 ? -18 : d < 0 ? 18 : 0 }),
};

const reducedVariants = {
  enter: { opacity: 0 },
  center: { opacity: 1 },
  exit: { opacity: 0 },
};

export default function HomePage() {
  /* ─── Dark mode (Space UI defaults to dark) ──── */
  const [isDarkMode, setIsDarkMode] = useState(true);

  useEffect(() => {
    const stored = localStorage.getItem('rag_dark_mode');
    const dark = stored === null ? true : stored !== 'false';
    setIsDarkMode(dark);
    document.documentElement.classList.toggle('dark', dark);
  }, []);

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('rag_dark_mode', String(isDarkMode));
  }, [isDarkMode]);

  /* ─── Tabs & sidebar ─────────────────────────── */
  const reduceMotion = useReducedMotion();
  const [activeTab, setActiveTab] = useState<ViewTab>('chat');
  const [tabDir, setTabDir] = useState(0);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  /* Switch tabs while recording travel direction so content slides the right way. */
  const changeTab = useCallback((next: ViewTab) => {
    setActiveTab((cur) => {
      if (cur === next) return cur;
      setTabDir(tabDirection(cur, next));
      return next;
    });
  }, []);

  useEffect(() => {
    const desktopQuery = window.matchMedia('(min-width: 1024px)');
    const syncSidebar = () => setIsSidebarOpen(desktopQuery.matches);
    syncSidebar();
    desktopQuery.addEventListener('change', syncSidebar);
    return () => desktopQuery.removeEventListener('change', syncSidebar);
  }, []);

  /* ─── Sessions ───────────────────────────────── */
  const {
    sessions,
    activeSession,
    activeSessionId,
    isLoaded,
    createNewSession,
    switchSession,
    deleteSession,
    renameSession,
    updateSessionMessages,
  } = useSessions();

  /* ─── Chat streaming ─────────────────────────── */
  const {
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
  } = useChatStream([]);

  /* ─── Lightweight backend health for the header ─────────── */
  const [health, setHealth] = useState<HealthStatus>({
    status: 'degraded',
    redis: false,
    vector_db: false,
    models_loaded: false,
    backend_version: 'Connecting',
  });

  useEffect(() => {
    let active = true;
    const refreshHealth = async () => {
      const nextHealth = await apiClient.getHealth().catch(() => ({
        status: 'error',
        redis: false,
        vector_db: false,
        models_loaded: false,
      } as HealthStatus));
      if (active) setHealth(nextHealth);
    };

    void refreshHealth();
    window.addEventListener('focus', refreshHealth);
    window.addEventListener('online', refreshHealth);
    return () => {
      active = false;
      window.removeEventListener('focus', refreshHealth);
      window.removeEventListener('online', refreshHealth);
    };
  }, []);

  /* Sync messages from active session on initial load and session switch */
  const prevActiveSessionIdRef = React.useRef<string>('');
  useEffect(() => {
    if (isLoaded && activeSession) {
      if (prevActiveSessionIdRef.current !== activeSession.id) {
        prevActiveSessionIdRef.current = activeSession.id;
        setMessages(activeSession.messages || []);
      }
    }
  }, [activeSession, isLoaded, setMessages]);

  /* Persist chat messages back into the active session whenever messages change */
  useEffect(() => {
    if (!isLoaded || !activeSessionId || messages.length === 0) return;

    // Streaming updates arrive rapidly. Persist the latest snapshot after the
    // burst settles instead of rewriting sessions/localStorage for every token.
    const persistTimer = window.setTimeout(() => {
      updateSessionMessages(activeSessionId, messages);
    }, 180);

    return () => window.clearTimeout(persistTimer);
  }, [messages, activeSessionId, isLoaded, updateSessionMessages]);

  /* ─── Handlers ───────────────────────────────── */
  const handleSendMessage = useCallback(
    (content: string, filters?: FilterOptions, model?: string, responseMode: ResponseMode = 'standard') => {
      sendMessage(content, activeSessionId, filters, model, 'standard', responseMode);
    },
    [sendMessage, activeSessionId],
  );

  const handleNewSession = useCallback(() => {
    cancelStream();
    const newSession = createNewSession();
    prevActiveSessionIdRef.current = newSession.id;
    setMessages([]);
  }, [cancelStream, createNewSession, setMessages]);

  const handleSwitchSession = useCallback(
    (id: string) => {
      if (id === activeSessionId) return;
      cancelStream();
      if (activeSessionId) {
        updateSessionMessages(activeSessionId, messages);
      }
      switchSession(id);
      const targetSession = sessions.find((s) => s.id === id);
      prevActiveSessionIdRef.current = id;
      setMessages(targetSession?.messages || []);
      if (window.innerWidth < 1024) setIsSidebarOpen(false);
    },
    [activeSessionId, cancelStream, messages, sessions, setMessages, switchSession, updateSessionMessages],
  );

  const handleClearChat = useCallback(() => {
    cancelStream();
    closeCitationDrawer();
    clearMessages();
    if (activeSessionId) {
      updateSessionMessages(activeSessionId, []);
      apiClient.clearSession(activeSessionId).catch(() => {});
    }
  }, [cancelStream, closeCitationDrawer, clearMessages, activeSessionId, updateSessionMessages]);

  const handleDeleteSession = useCallback(
    (id: string) => {
      cancelStream();
      closeCitationDrawer();
      deleteSession(id);
      if (id === activeSessionId) {
        const remaining = sessions.filter((s) => s.id !== id);
        if (remaining.length > 0) {
          prevActiveSessionIdRef.current = remaining[0].id;
          setMessages(remaining[0].messages || []);
        } else {
          prevActiveSessionIdRef.current = '';
          setMessages([]);
        }
      }
    },
    [activeSessionId, cancelStream, closeCitationDrawer, deleteSession, sessions, setMessages]
  );

  /* ─── Shared ─── */
  const connected = health.status === 'ok' && health.vector_db;
  const toggleTheme = useCallback(() => setIsDarkMode((v) => !v), []);

  const tabContent = (() => {
    switch (activeTab) {
      case 'chat':
        return (
          <SpaceShell
            messages={messages}
            isStreaming={isStreaming}
            onSendMessage={handleSendMessage}
            onCancelStream={cancelStream}
            openCitation={openCitation}
            activeCitation={activeCitation}
            isDrawerOpen={isDrawerOpen}
            closeCitationDrawer={closeCitationDrawer}
            sessions={sessions}
            activeSessionId={activeSessionId}
            onSelectSession={handleSwitchSession}
            onNewSession={handleNewSession}
            onDeleteSession={handleDeleteSession}
            onRenameSession={renameSession}
            health={health}
            activeTab={activeTab}
            setActiveTab={changeTab}
            isLight={!isDarkMode}
            onToggleTheme={toggleTheme}
          />
        );
      case 'documents':
        return (
          <LibraryShell
            activeTab={activeTab}
            setActiveTab={changeTab}
            isLight={!isDarkMode}
            onToggleTheme={toggleTheme}
            connected={connected}
          >
            <DocumentsView />
          </LibraryShell>
        );
      case 'storage':
        return (
          <LibraryShell
            activeTab={activeTab}
            setActiveTab={changeTab}
            isLight={!isDarkMode}
            onToggleTheme={toggleTheme}
            connected={connected}
          >
            <StorageView />
          </LibraryShell>
        );
      default:
        return (
          <LibraryShell
            activeTab={activeTab}
            setActiveTab={changeTab}
            isLight={!isDarkMode}
            onToggleTheme={toggleTheme}
            connected={connected}
          >
            <AdminView />
          </LibraryShell>
        );
    }
  })();

  const panelTransition = reduceMotion
    ? { duration: 0.14, ease: [0.22, 1, 0.36, 1] as const }
    : {
        x: { type: 'spring' as const, stiffness: 460, damping: 44, mass: 0.9 },
        opacity: { duration: 0.26, ease: [0.22, 1, 0.36, 1] as const },
      };

  // Text-dense views (a live conversation, Library, Telemetry) get a deeper
  // veil and a still backdrop; the empty Ask view shows the scene in full.
  const backdropFocus = activeTab !== 'chat' || messages.length > 0;

  return (
    <div className="relative min-h-[100dvh] w-full overflow-hidden">
      {/* One persistent backdrop shared by every tab. It never unmounts, so
          switching tabs crossfades transparent content over a constant scene
          instead of tearing it down and rebuilding it. */}
      <AlpineBackdrop focus={backdropFocus} />
      {/* Sync (crossfade) mode, NOT mode="wait": the incoming tab mounts
          immediately while the outgoing one animates away, so switching tabs
          works even while the chat is streaming (rapid re-renders would
          otherwise stall mode="wait"'s exit-complete tracking and leave the new
          tab unmounted). Panels are absolutely positioned so the two overlap
          during the crossfade instead of stacking in flow. */}
      <AnimatePresence custom={tabDir} initial={false}>
        <motion.div
          key={activeTab}
          custom={tabDir}
          variants={reduceMotion ? reducedVariants : panelVariants}
          initial="enter"
          animate="center"
          exit="exit"
          transition={panelTransition}
          className="absolute inset-0 z-[1] h-[100dvh] w-full"
          style={{ willChange: 'transform, opacity', transformOrigin: 'center' }}
        >
          {tabContent}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
