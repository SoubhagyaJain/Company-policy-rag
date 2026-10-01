'use client';

/** SpaceShell — the Ask view. Glass chrome (sidebar, nav, message column,
 *  composer, status) floats over the shared AlpineBackdrop rendered at the page
 *  root. Owns only presentational shell state; all RAG behavior arrives via
 *  props from app/page.tsx (useChatStream / useSessions). */

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Moon, Sun, Compass, CalendarClock, Plane, KeyRound, ArrowUpRight, ArrowDown } from 'lucide-react';

import type { ChatMessageData, ChatSession, Citation, FilterOptions, HealthStatus, ResponseMode } from '../../lib/types';
import { useComposerControls } from '../../hooks/useComposerControls';
import { useSmoothScroll } from '../../hooks/useSmoothScroll';
import { SpaceTabNav, type ViewTab } from './SpaceTabNav';
import { SpaceSidebar } from './SpaceSidebar';
import { SpaceComposer } from './SpaceComposer';
import { SpaceMessage } from './SpaceMessage';
import { SpaceCitationDrawer } from './SpaceCitationDrawer';
import { BlackHoleProvider } from './BlackHoleAbsorption';
import { backdropPoint, MOON, SUNSET_GLOW } from './AlpineBackdrop';

export type { ViewTab };

const SUGGESTED_PROMPTS = [
  { title: 'Remote Work & Stipends', prompt: 'What are the rules and eligible stipends for working remotely?', icon: Compass },
  { title: 'PTO & Rollover Policy', prompt: 'How many PTO days can I carry over into the next calendar year?', icon: CalendarClock },
  { title: 'Travel Expense Guidelines', prompt: 'What is the daily meal and hotel reimbursement limit for business travel?', icon: Plane },
  { title: 'IT Security & Passwords', prompt: 'What is the password rotation policy and MFA requirement for corporate laptops?', icon: KeyRound },
];

interface SpaceShellProps {
  messages: ChatMessageData[];
  isStreaming: boolean;
  onSendMessage: (content: string, filters: FilterOptions | undefined, model: string, responseMode: ResponseMode) => void;
  onCancelStream: () => void;

  openCitation: (c: Citation) => void;
  activeCitation: Citation | null;
  isDrawerOpen: boolean;
  closeCitationDrawer: () => void;

  sessions: ChatSession[];
  activeSessionId: string;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string) => void;
  onRenameSession: (id: string, title: string) => void;

  health: HealthStatus;
  activeTab: ViewTab;
  setActiveTab: (t: ViewTab) => void;
  isLight: boolean;
  onToggleTheme: () => void;
}

export function SpaceShell(props: SpaceShellProps) {
  const {
    messages, isStreaming,
    openCitation, activeCitation, isDrawerOpen, closeCitationDrawer,
    sessions, activeSessionId,
    health, activeTab, setActiveTab, isLight, onToggleTheme,
  } = props;

  const scrollRef = useRef<HTMLDivElement>(null);
  const controls = useComposerControls();

  // Stable handler identities for the memoized sidebar/composer. Several page
  // handlers are rebuilt on every streamed token batch (they close over
  // `messages`); routing them through a ref lets those children skip re-render
  // while an answer streams.
  const latest = useRef(props);
  latest.current = props;
  const selectSession = useCallback((id: string) => latest.current.onSelectSession(id), []);
  const newSession = useCallback(() => latest.current.onNewSession(), []);
  const deleteSession = useCallback((id: string) => latest.current.onDeleteSession(id), []);
  const renameSession = useCallback((id: string, title: string) => latest.current.onRenameSession(id, title), []);
  const sendMessage = useCallback<SpaceShellProps['onSendMessage']>(
    (content, filters, model, mode) => latest.current.onSendMessage(content, filters, model, mode),
    [],
  );
  const cancelStream = useCallback(() => latest.current.onCancelStream(), []);

  // Buttery inertia scrolling for the message pane over the live hero.
  useSmoothScroll(scrollRef);

  // Deleted chats dissolve into the moon (night) or the sunset glow (light).
  // Read fresh on every delete so it tracks the backdrop's cover crop.
  const getBlackHolePoint = useCallback(
    () => backdropPoint(isLight ? SUNSET_GLOW : MOON),
    [isLight],
  );

  const [collapsed, setCollapsed] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia('(max-width: 1024px)');
    const sync = () => setCollapsed(mq.matches);
    sync();
    mq.addEventListener('change', sync);
    return () => mq.removeEventListener('change', sync);
  }, []);

  const toggleCollapse = useCallback(() => setCollapsed((v) => !v), []);

  // Follow-the-answer auto-scroll, driven by the reader's intent rather than a
  // distance threshold: ANY upward gesture (wheel, keys, scrollbar drag, touch)
  // releases the pin immediately, so the pane never yanks back mid-scroll.
  // It re-engages only after the reader has deliberately headed back down
  // (a downward gesture, or having left the bottom zone) and reaches the
  // bottom again — or when a new message is added (they sent a question).
  // Scroll positions alone can't be trusted: one coalesced scroll event can mix
  // an auto-follow jump with the reader's first upward frame.
  const pinnedRef = useRef(true);
  const headingDownRef = useRef(false);
  const prevMsgCountRef = useRef(messages.length);
  const [showJump, setShowJump] = useState(false);

  const handleMessagesScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const dist = el.scrollHeight - el.clientHeight - el.scrollTop;
    if (!pinnedRef.current) {
      if (dist >= 32) headingDownRef.current = true;
      else if (headingDownRef.current) pinnedRef.current = true;
    }
    setShowJump(!pinnedRef.current && dist > 240);
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const release = () => {
      pinnedRef.current = false;
      headingDownRef.current = false;
    };
    const onWheel = (e: WheelEvent) => {
      if (e.deltaY < 0) release();
      else if (e.deltaY > 0) headingDownRef.current = true;
    };
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === 'TEXTAREA' || t.tagName === 'INPUT' || t.isContentEditable)) return;
      if (e.key === 'ArrowUp' || e.key === 'PageUp' || e.key === 'Home') release();
      else if (e.key === 'ArrowDown' || e.key === 'PageDown' || e.key === 'End' || e.key === ' ') headingDownRef.current = true;
    };
    // Only a press on the scrollbar itself (right of the content box) counts.
    const onPointerDown = (e: PointerEvent) => {
      if (e.target === el && e.offsetX > el.clientWidth) release();
    };
    el.addEventListener('wheel', onWheel, { passive: true });
    el.addEventListener('touchmove', release, { passive: true });
    el.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('keydown', onKey);
    return () => {
      el.removeEventListener('wheel', onWheel);
      el.removeEventListener('touchmove', release);
      el.removeEventListener('pointerdown', onPointerDown);
      window.removeEventListener('keydown', onKey);
    };
  }, []);

  // Layout effect: the pane is moved to the newest line before the browser
  // paints, so streamed text never flashes below the fold for a frame.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (messages.length !== prevMsgCountRef.current) {
      prevMsgCountRef.current = messages.length;
      pinnedRef.current = true;
      setShowJump(false);
    }
    if (pinnedRef.current) el.scrollTop = el.scrollHeight;
  }, [messages]);

  const jumpToLatest = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    setShowJump(false);
    headingDownRef.current = true;
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  }, []);

  const connected = health.status === 'ok' && health.vector_db;
  const empty = messages.length === 0;

  const sendPrompt = (text: string) =>
    sendMessage(text, controls.buildFilters(), controls.selectedModel, controls.responseMode);

  const messageList = useMemo(
    () =>
      messages.map((m) => (
        <SpaceMessage key={m.id} message={m} onOpenCitation={openCitation} />
      )),
    [messages, openCitation],
  );

  return (
    <BlackHoleProvider getTarget={getBlackHolePoint}>
    <div className="relative h-[100dvh] overflow-hidden">
        {/* Chrome */}
        <div className="relative z-10 flex h-full gap-3 p-3 sm:gap-5 sm:p-5" style={{ animation: 'chromeReveal 0.8s cubic-bezier(0.22, 1, 0.36, 1) both 0.1s' }}>
          <SpaceSidebar
            collapsed={collapsed}
            onToggleCollapse={toggleCollapse}
            sessions={sessions}
            activeSessionId={activeSessionId}
            onSelectSession={selectSession}
            onNewSession={newSession}
            onDeleteSession={deleteSession}
            onRenameSession={renameSession}
            modelLabel={controls.selectedModelLabel}
            grounded={connected}
          />

          <div className="flex min-w-0 flex-1 flex-col">
            {/* Top bar */}
            <div className="relative flex items-center justify-center">
              <SpaceTabNav activeTab={activeTab} onChange={setActiveTab} />
              <div className="absolute right-0 flex items-center gap-2">
                <span
                  className="sp-conn sp-mono hidden items-center gap-2 rounded-full px-3.5 py-2 text-[10px] uppercase tracking-[0.2em] md:inline-flex"
                  role="status"
                >
                  <span
                    className={`h-1.5 w-1.5 ${connected ? 'sp-dot sp-dot-ok' : 'sp-dot sp-dot-warn'}`}
                    style={{ animation: 'connPulse 2.4s ease-in-out infinite' }}
                  />
                  {connected ? 'Connected' : 'Offline'}
                </span>
                <button
                  type="button"
                  onClick={onToggleTheme}
                  aria-label={isLight ? 'Switch to night theme' : 'Switch to golden-hour theme'}
                  title={isLight ? 'Night' : 'Golden hour'}
                  className="sp-ibtn flex h-9 w-9 items-center justify-center rounded-full"
                >
                  {isLight ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
                </button>
              </div>
            </div>

            {/* Messages */}
            <div className="relative mt-3 flex min-h-0 flex-1 flex-col">
            <div ref={scrollRef} onScroll={handleMessagesScroll} className="sp-scroll sp-fade-top min-h-0 flex-1 overflow-y-auto">
              <div className={`mx-auto flex w-full max-w-[860px] flex-col gap-7 pb-6 ${empty ? 'min-h-full' : 'pt-4'}`}>
                {empty ? (
                  <motion.div
                    className="flex flex-1 flex-col items-center justify-center gap-9 py-[6vh] text-center"
                    initial="hidden"
                    animate="visible"
                    variants={{ hidden: {}, visible: { transition: { staggerChildren: 0.09, delayChildren: 0.15 } } }}
                  >
                    <motion.div
                      className="sp-hero-copy flex max-w-[640px] flex-col items-center px-2"
                      variants={{ hidden: { opacity: 0, y: 18 }, visible: { opacity: 1, y: 0 } }}
                      transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
                    >
                      <p className="sp-eyebrow mb-5 flex items-center gap-2.5">
                        <span className="sp-moon h-2 w-2" aria-hidden="true" />
                        Grounded policy intelligence
                      </p>
                      <h1 className="sp-display text-[clamp(38px,5.4vw,64px)] leading-[1.04] [text-wrap:balance]">
                        Ask across your <em>grounded</em> corpus
                      </h1>
                      <p className="sp-lede mt-5 max-w-[480px] text-[15px] leading-relaxed">
                        Every answer is retrieved, reranked, and verified against your documents, with sources you can open.
                      </p>
                    </motion.div>
                    <motion.div
                      className="grid w-full max-w-[760px] grid-cols-1 gap-3 sm:grid-cols-2"
                      variants={{ hidden: {}, visible: { transition: { staggerChildren: 0.07 } } }}
                    >
                      {SUGGESTED_PROMPTS.map((p) => {
                        const Icon = p.icon;
                        return (
                          <motion.button
                            key={p.title}
                            type="button"
                            onClick={() => sendPrompt(p.prompt)}
                            className="sp-card sp-hero-card group flex items-start gap-3.5 rounded-[20px] p-4 text-left"
                            variants={{ hidden: { opacity: 0, y: 16 }, visible: { opacity: 1, y: 0 } }}
                            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                          >
                            <span className="sp-hero-icon flex h-9 w-9 flex-none items-center justify-center rounded-full">
                              <Icon className="h-4 w-4" />
                            </span>
                            <span className="min-w-0 flex-1">
                              <span className="sp-text flex items-center gap-1.5 text-[14px] font-semibold">
                                {p.title}
                                <ArrowUpRight className="sp-hero-arrow h-3.5 w-3.5 text-[var(--sp-accent-text)]" />
                              </span>
                              <span className="sp-faint mt-1 block text-[12.5px] leading-snug">{p.prompt}</span>
                            </span>
                          </motion.button>
                        );
                      })}
                    </motion.div>
                  </motion.div>
                ) : (
                  messageList
                )}
              </div>
            </div>

            {/* Jump back to the newest line after scrolling up */}
            <AnimatePresence>
              {showJump && !empty && (
                <motion.button
                  type="button"
                  onClick={jumpToLatest}
                  className="sp-conn sp-text absolute bottom-3 left-1/2 z-10 flex items-center gap-1.5 rounded-full px-3.5 py-2 text-[12px] font-medium"
                  initial={{ opacity: 0, y: 8, x: '-50%' }}
                  animate={{ opacity: 1, y: 0, x: '-50%' }}
                  exit={{ opacity: 0, y: 8, x: '-50%' }}
                  transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
                >
                  <ArrowDown className="h-3.5 w-3.5" />
                  {isStreaming ? 'Follow answer' : 'Jump to latest'}
                </motion.button>
              )}
            </AnimatePresence>
            </div>

            {/* Composer */}
            <div className="mt-2 flex flex-col items-center">
              <SpaceComposer
                controls={controls}
                isStreaming={isStreaming}
                onSend={sendMessage}
                onCancel={cancelStream}
              />
              <p className="sp-faint mt-2 hidden text-[11px] sm:block">
                Answers cite your indexed documents. Open a source to verify the exact wording.
              </p>
            </div>
          </div>
        </div>
      </div>

      <SpaceCitationDrawer isOpen={isDrawerOpen} citation={activeCitation} onClose={closeCitationDrawer} />
    </BlackHoleProvider>
  );
}

export default SpaceShell;
