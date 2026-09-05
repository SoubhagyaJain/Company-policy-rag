'use client';

/** SpaceShell — the Ask-view scroll-gate shell. A 220vh runway with a sticky
 *  WebGL hero stage; the glass chrome (sidebar, nav, message column, composer,
 *  status) floats over it. Owns only presentational shell state; all RAG
 *  behavior arrives via props from app/page.tsx (useChatStream / useSessions). */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Moon, Sun, Compass, FileQuestion, ShieldCheck, Zap } from 'lucide-react';

import type { ChatMessageData, ChatSession, Citation, FilterOptions, HealthStatus, ResponseMode } from '../../lib/types';
import { useComposerControls } from '../../hooks/useComposerControls';
import { useSmoothScroll } from '../../hooks/useSmoothScroll';
import { SpaceTabNav } from './SpaceTabNav';
import { SpaceSidebar } from './SpaceSidebar';
import { SpaceComposer } from './SpaceComposer';
import { SpaceMessage } from './SpaceMessage';
import { SpaceCitationDrawer } from './SpaceCitationDrawer';
import { BlackHoleProvider } from './BlackHoleAbsorption';

export type ViewTab = 'chat' | 'documents' | 'observability';

const SUGGESTED_PROMPTS = [
  { title: 'Remote Work & Stipends', prompt: 'What are the rules and eligible stipends for working remotely?', icon: Compass },
  { title: 'PTO & Rollover Policy', prompt: 'How many PTO days can I carry over into the next calendar year?', icon: FileQuestion },
  { title: 'Travel Expense Guidelines', prompt: 'What is the daily meal and hotel reimbursement limit for business travel?', icon: ShieldCheck },
  { title: 'IT Security & Passwords', prompt: 'What is the password rotation policy and MFA requirement for corporate laptops?', icon: Zap },
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
    messages, isStreaming, onSendMessage, onCancelStream,
    openCitation, activeCitation, isDrawerOpen, closeCitationDrawer,
    sessions, activeSessionId, onSelectSession, onNewSession, onDeleteSession, onRenameSession,
    health, activeTab, setActiveTab, isLight, onToggleTheme,
  } = props;

  const scrollRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const controls = useComposerControls();

  // Buttery inertia scrolling for the message pane over the live hero.
  useSmoothScroll(scrollRef);

  // Live black-hole screen point (centre of the hero stage, biased slightly up
  // to the visual hole) — read fresh each time so it tracks resize/scroll.
  const getBlackHolePoint = useCallback(() => {
    const el = stageRef.current;
    if (!el) return { x: window.innerWidth / 2, y: window.innerHeight * 0.44 };
    const r = el.getBoundingClientRect();
    return { x: r.left + r.width * 0.5, y: r.top + r.height * 0.44 };
  }, []);

  const [collapsed, setCollapsed] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia('(max-width: 1024px)');
    const sync = () => setCollapsed(mq.matches);
    sync();
    mq.addEventListener('change', sync);
    return () => mq.removeEventListener('change', sync);
  }, []);

  // Auto-scroll the message pane on new content — but only while the reader is
  // parked near the bottom. If they scroll up during streaming to re-read
  // something, we stop yanking them back down; a brand-new message (they just
  // sent, or the assistant bubble appeared) always snaps to the latest.
  const stickToBottomRef = useRef(true);
  const prevMsgCountRef = useRef(messages.length);

  const handleMessagesScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.clientHeight - el.scrollTop;
    stickToBottomRef.current = distanceFromBottom < 120;
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (messages.length !== prevMsgCountRef.current) {
      prevMsgCountRef.current = messages.length;
      stickToBottomRef.current = true;
    }
    if (stickToBottomRef.current) el.scrollTop = el.scrollHeight;
  }, [messages, isStreaming]);

  const connected = health.status === 'ok' && health.vector_db;
  const empty = messages.length === 0;

  const sendPrompt = (text: string) =>
    onSendMessage(text, controls.buildFilters(), controls.selectedModel, controls.responseMode);

  const messageList = useMemo(
    () =>
      messages.map((m) => (
        <SpaceMessage key={m.id} message={m} onOpenCitation={openCitation} />
      )),
    [messages, openCitation],
  );

  return (
    <BlackHoleProvider getTarget={getBlackHolePoint}>
    <div ref={stageRef} className="relative h-[100dvh] overflow-hidden">
        {/* Veil — darkens the shared hero (rendered at the page root) for text
            readability. The hero itself is no longer mounted per-tab. */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 z-[1]"
          style={{ background: 'radial-gradient(120% 80% at 48% 40%, rgba(4,6,12,0) 42%, rgba(3,5,11,0.55) 100%)', opacity: isLight ? 0 : 1, transition: 'opacity 700ms cubic-bezier(0.22, 1, 0.36, 1)' }}
        />

        {/* Chrome */}
        <div className="relative z-10 flex h-full gap-4 p-4 sm:p-6" style={{ animation: 'chromeReveal 0.8s cubic-bezier(0.22, 1, 0.36, 1) both 0.1s' }}>
          <SpaceSidebar
            collapsed={collapsed}
            onToggleCollapse={() => setCollapsed((v) => !v)}
            sessions={sessions}
            activeSessionId={activeSessionId}
            onSelectSession={onSelectSession}
            onNewSession={onNewSession}
            onDeleteSession={onDeleteSession}
            onRenameSession={onRenameSession}
            modelLabel={controls.selectedModelLabel}
            grounded={connected}
          />

          <div className="flex min-w-0 flex-1 flex-col">
            {/* Top bar */}
            <div className="relative flex items-center justify-center">
              <SpaceTabNav activeTab={activeTab} onChange={setActiveTab} />
              <div className="absolute right-0 flex items-center gap-2">
                <span className="sp-conn sp-mono hidden items-center gap-2 rounded-full px-4 py-2 text-[10px] uppercase tracking-[0.2em] sm:inline-flex">
                  <span
                    className="h-1.5 w-1.5 rounded-full"
                    style={{ background: connected ? 'var(--sp-accent)' : '#e0b45a', animation: 'connPulse 2.4s ease-in-out infinite' }}
                  />
                  {connected ? 'Connected' : 'Offline'}
                </span>
                <button
                  type="button"
                  onClick={onToggleTheme}
                  aria-label={isLight ? 'Use dark theme' : 'Use light theme'}
                  className="sp-ibtn flex h-9 w-9 items-center justify-center rounded-[11px]"
                >
                  {isLight ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
                </button>
              </div>
            </div>

            {/* Messages */}
            <div ref={scrollRef} onScroll={handleMessagesScroll} className="sp-scroll mt-4 flex-1 overflow-y-auto">
              <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 pb-4">
                {empty ? (
                  <motion.div
                    className="flex flex-1 flex-col items-center justify-center gap-6 pt-[10vh] text-center"
                    initial="hidden"
                    animate="visible"
                    variants={{ hidden: {}, visible: { transition: { staggerChildren: 0.09, delayChildren: 0.15 } } }}
                  >
                    <motion.div
                      className="sp-hero-glass rounded-3xl px-8 py-10 sm:px-12 sm:py-12"
                      variants={{ hidden: { opacity: 0, y: 22, filter: 'blur(6px)' }, visible: { opacity: 1, y: 0, filter: 'blur(0px)' } }}
                      transition={{ duration: 0.65, ease: [0.22, 1, 0.36, 1] }}
                    >
                      <h1 className="sp-heading text-[clamp(26px,3vw,38px)] font-semibold tracking-tight">
                        Ask across your grounded corpus
                      </h1>
                      <p className="sp-muted mt-2 text-[14px]">
                        Every answer is retrieved, reranked, and verified against your documents.
                      </p>
                    </motion.div>
                    <motion.div
                      className="grid w-full grid-cols-1 gap-2.5 sm:grid-cols-2"
                      variants={{ hidden: {}, visible: { transition: { staggerChildren: 0.06 } } }}
                    >
                      {SUGGESTED_PROMPTS.map((p) => {
                        const Icon = p.icon;
                        return (
                          <motion.button
                            key={p.title}
                            type="button"
                            onClick={() => sendPrompt(p.prompt)}
                            className="sp-card sp-hero-card sp-text flex items-start gap-3 rounded-2xl p-3.5 text-left transition-[transform,box-shadow] duration-300 ease-out hover:-translate-y-0.5 hover:shadow-lg"
                            variants={{ hidden: { opacity: 0, y: 16, scale: 0.97 }, visible: { opacity: 1, y: 0, scale: 1 } }}
                            transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                          >
                            <Icon className="mt-0.5 h-4 w-4 flex-none text-[var(--sp-accent-text)]" />
                            <span>
                              <span className="block text-[13px] font-semibold">{p.title}</span>
                              <span className="sp-faint block text-[11.5px] leading-snug">{p.prompt}</span>
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

            {/* Composer */}
            <div className="mt-3 flex justify-center">
              <SpaceComposer
                controls={controls}
                isStreaming={isStreaming}
                onSend={onSendMessage}
                onCancel={onCancelStream}
              />
            </div>
          </div>
        </div>
      </div>

      <SpaceCitationDrawer isOpen={isDrawerOpen} citation={activeCitation} onClose={closeCitationDrawer} />
    </BlackHoleProvider>
  );
}

export default SpaceShell;
