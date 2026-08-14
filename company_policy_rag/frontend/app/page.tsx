'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { AnimatePresence, motion } from 'framer-motion';

import { Header, ViewTab } from '@/components/Header';
import { SessionSidebar } from '@/components/SessionSidebar';
import { ChatWindow } from '@/components/ChatWindow';
import { CitationDrawer } from '@/components/CitationDrawer';
import { DocumentsView } from '@/components/DocumentsView';
import { AdminView } from '@/components/AdminView';

import { useChatStream } from '@/hooks/useChatStream';
import { useSessions } from '@/hooks/useSessions';
import { useObservability } from '@/hooks/useObservability';

import type { FilterOptions } from '@/lib/types';

export default function HomePage() {
  /* ─── Dark mode ──────────────────────────────── */
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem('rag_dark_mode');
    if (stored === 'true') {
      setIsDarkMode(true);
      document.documentElement.classList.add('dark');
    }
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
  const [activeTab, setActiveTab] = useState<ViewTab>('chat');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

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

  /* ─── Observability (health for header) ──────── */
  const { health } = useObservability();

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
    if (isLoaded && activeSessionId && messages.length > 0) {
      updateSessionMessages(activeSessionId, messages);
    }
  }, [messages, activeSessionId, isLoaded, updateSessionMessages]);

  /* ─── Handlers ───────────────────────────────── */
  const handleSendMessage = useCallback(
    (content: string, filters?: FilterOptions, model?: string) => {
      sendMessage(content, activeSessionId, filters, model);
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
    },
    [activeSessionId, cancelStream, messages, sessions, setMessages, switchSession, updateSessionMessages],
  );

  const handleClearChat = useCallback(() => {
    clearMessages();
    if (activeSessionId) {
      updateSessionMessages(activeSessionId, []);
    }
  }, [clearMessages, activeSessionId, updateSessionMessages]);

  /* ─── Tab content ────────────────────────────── */
  const renderContent = () => {
    switch (activeTab) {
      case 'documents':
        return <DocumentsView />;
      case 'observability':
        return <AdminView />;
      case 'chat':
      default:
        return (
          <ChatWindow
            messages={messages}
            isStreaming={isStreaming}
            onSendMessage={handleSendMessage}
            onCancelStream={cancelStream}
            onClearChat={handleClearChat}
            onOpenCitation={openCitation}
          />
        );
    }
  };

  return (
    <div className="h-screen flex flex-col bg-[#FAF9F5] dark:bg-[#141413] transition-colors">
      {/* Header */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isSidebarOpen={isSidebarOpen}
        setIsSidebarOpen={setIsSidebarOpen}
        isDarkMode={isDarkMode}
        setIsDarkMode={setIsDarkMode}
        health={health}
      />

      {/* Main body: sidebar + content */}
      <div className="flex-1 flex overflow-hidden min-h-0">
        {/* Session sidebar — only relevant on chat tab */}
        {activeTab === 'chat' && (
          <SessionSidebar
            isOpen={isSidebarOpen}
            sessions={sessions}
            activeSessionId={activeSessionId}
            onSelectSession={handleSwitchSession}
            onNewSession={handleNewSession}
            onDeleteSession={deleteSession}
            onRenameSession={renameSession}
          />
        )}

        {/* Page content with page-transition animation */}
        <AnimatePresence mode="wait">
          <motion.main
            key={activeTab}
            initial={{ opacity: 0, x: 12 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -12 }}
            transition={{ duration: 0.2, ease: 'easeInOut' }}
            className="flex-1 min-h-0 min-w-0 overflow-hidden relative flex flex-col"
          >
            {renderContent()}
          </motion.main>
        </AnimatePresence>
      </div>

      {/* Citation drawer slides from right */}
      <CitationDrawer
        isOpen={isDrawerOpen}
        citation={activeCitation}
        onClose={closeCitationDrawer}
      />
    </div>
  );
}
