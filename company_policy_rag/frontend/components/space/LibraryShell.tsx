'use client';

/** LibraryShell — Space-themed wrapper for the Library (documents) tab.
 *  Shares the persistent hero rendered at the page root (see app/page.tsx); this
 *  shell is transparent so that hero shows through, and adds the veil, shared top
 *  nav, and theme toggle so Library matches the Ask view's glass-over-hero look. */

import { Moon, Sun } from 'lucide-react';
import { SpaceTabNav, type ViewTab } from './SpaceTabNav';

export type { ViewTab };

interface LibraryShellProps {
  activeTab: ViewTab;
  setActiveTab: (t: ViewTab) => void;
  isLight: boolean;
  onToggleTheme: () => void;
  connected: boolean;
  children: React.ReactNode;
}

export function LibraryShell({
  activeTab, setActiveTab, isLight, onToggleTheme, connected, children,
}: LibraryShellProps) {
  return (
    <div className="relative h-[100dvh] overflow-hidden">
      {/* Veil over the shared root hero */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 z-[1]"
        style={{
          background: 'radial-gradient(120% 80% at 48% 40%, rgba(4,6,12,0) 42%, rgba(3,5,11,0.55) 100%)',
          opacity: isLight ? 0 : 1,
          transition: 'opacity 700ms cubic-bezier(0.22, 1, 0.36, 1)',
        }}
      />

      {/* Chrome */}
      <div
        className="relative z-10 flex h-full flex-col p-4 sm:p-6"
        style={{ animation: 'chromeReveal 0.8s cubic-bezier(0.22, 1, 0.36, 1) both 0.1s' }}
      >
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

        {/* Content */}
        <div className="mt-4 flex-1 overflow-hidden rounded-2xl">
          {children}
        </div>
      </div>
    </div>
  );
}

export default LibraryShell;
