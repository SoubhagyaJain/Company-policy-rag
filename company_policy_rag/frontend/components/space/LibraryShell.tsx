'use client';

/** LibraryShell — shared wrapper for the Library (documents), Telemetry and Storage tabs.
 *  Transparent over the persistent AlpineBackdrop rendered at the page root (see
 *  app/page.tsx), which switches to its deeper "focus" veil for these dense
 *  views. Adds the shared top nav, connection status, and theme toggle, and
 *  frames the content in one glass panel so tables and charts never sit
 *  directly on the photograph. */

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
      {/* Chrome */}
      <div
        className="relative z-10 flex h-full flex-col p-3 sm:p-5"
        style={{ animation: 'chromeReveal 0.8s cubic-bezier(0.22, 1, 0.36, 1) both 0.1s' }}
      >
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

        {/* Content */}
        <div className="sp-panel mt-4 flex flex-1 flex-col overflow-hidden rounded-[26px]">
          {children}
        </div>
      </div>
    </div>
  );
}

export default LibraryShell;
