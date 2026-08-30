'use client';

import React from 'react';
import {
  MessageSquare,
  FileText,
  Activity,
  Sun,
  Moon,
  PanelLeft,
  ShieldCheck,
  Zap,
} from 'lucide-react';
import { HealthStatus } from '../lib/types';
import { cn } from '../lib/utils';

export type ViewTab = 'chat' | 'documents' | 'observability';

const NAV_ITEMS: Array<{
  id: ViewTab;
  label: string;
  mobileLabel: string;
  icon: typeof MessageSquare;
}> = [
  { id: 'chat', label: 'Ask', mobileLabel: 'Ask policy', icon: MessageSquare },
  { id: 'documents', label: 'Library', mobileLabel: 'Library', icon: FileText },
  { id: 'observability', label: 'Telemetry', mobileLabel: 'Telemetry', icon: Activity },
];

interface HeaderProps {
  activeTab: ViewTab;
  setActiveTab: (tab: ViewTab) => void;
  isSidebarOpen: boolean;
  setIsSidebarOpen: (open: boolean | ((prev: boolean) => boolean)) => void;
  isDarkMode: boolean;
  setIsDarkMode: (dark: boolean | ((prev: boolean) => boolean)) => void;
  health: HealthStatus;
}

export function Header({
  activeTab,
  setActiveTab,
  isSidebarOpen,
  setIsSidebarOpen,
  isDarkMode,
  setIsDarkMode,
  health,
}: HeaderProps) {
  const isHealthy = health.status === 'ok' && health.vector_db;

  return (
    <header className="relative z-50 w-full border-b border-[#E5E0D8]/80 bg-[#FAF9F5]/88 px-3 py-2 backdrop-blur-2xl transition-colors dark:border-[#2A2925]/80 dark:bg-[#141413]/88 sm:px-4">
      <div className="mx-auto flex max-w-[1440px] items-center justify-between gap-3">
        {/* Left Section: Sidebar Toggle & Branding */}
        <div className="flex min-w-0 items-center gap-2 sm:gap-3">
          <button
            onClick={() => setIsSidebarOpen((prev) => !prev)}
            disabled={activeTab !== 'chat'}
            aria-label={isSidebarOpen ? 'Hide conversation sidebar' : 'Show conversation sidebar'}
            title={isSidebarOpen ? 'Hide Sidebar' : 'Show Sidebar'}
            className={cn(
              'focus-ring rounded-xl p-2 text-charcoal/70 transition-colors hover:bg-cream-200/60 dark:text-cream-200 dark:hover:bg-sand-dark/60',
              activeTab !== 'chat' && 'invisible',
            )}
          >
            <PanelLeft className="w-5 h-5" />
          </button>

          <div className="flex min-w-0 items-center gap-2.5">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-2xl border border-terracotta-500/25 bg-gradient-to-br from-terracotta-500/15 to-amber-500/10 text-terracotta-600 shadow-[inset_0_1px_0_rgba(255,255,255,.7)] dark:from-terracotta-500/20 dark:to-amber-500/10 dark:text-terracotta-400">
              <ShieldCheck className="h-[18px] w-[18px]" />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="truncate font-serif text-[15px] font-semibold tracking-tight text-charcoal dark:text-cream-100 sm:text-base">
                  Company Policy RAG
                </span>
              </div>
              <p className="hidden text-[10px] font-medium tracking-wide text-charcoal-muted dark:text-cream-400 sm:block">
                Trusted answers for every team
              </p>
            </div>
          </div>
        </div>

        {/* Center Section: View Navigation Tabs */}
        <nav aria-label="Primary" className="hidden items-center gap-1 rounded-2xl border border-sand-border/60 bg-cream-100/80 p-1 shadow-inner dark:border-sand-darkBorder/60 dark:bg-sand-dark/80 md:flex">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                aria-current={activeTab === item.id ? 'page' : undefined}
                className={cn(
                  'focus-ring flex items-center gap-2 rounded-xl px-3.5 py-1.5 text-xs font-medium transition-all duration-200',
                  activeTab === item.id
                    ? 'bg-[#FAF9F5] font-semibold text-charcoal shadow-sm dark:bg-[#141413] dark:text-cream-100'
                    : 'text-charcoal-muted hover:text-charcoal dark:text-cream-400 dark:hover:text-cream-200',
                )}
              >
                <Icon className="h-3.5 w-3.5 text-terracotta-600 dark:text-terracotta-500" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Right Section: System Status & Theme Toggle */}
        <div className="flex shrink-0 items-center gap-2 sm:gap-3">
          {/* Health Badge */}
          <div
            title={`Backend: ${isHealthy ? 'Connected' : 'Degraded'}`}
            className="hidden items-center gap-2 rounded-full border border-sand-border/70 bg-cream-100/70 px-2.5 py-1 text-[11px] font-medium dark:border-sand-darkBorder/70 dark:bg-sand-dark/70 lg:flex"
          >
            <Zap className={cn('h-3 w-3', isHealthy ? 'text-emerald-600' : 'text-amber-500')} />
            <span className="font-mono text-charcoal dark:text-cream-200">
              {isHealthy ? 'Systems ready' : 'Connecting'}
            </span>
          </div>

          {/* Theme Toggle */}
          <button
            onClick={() => setIsDarkMode((prev) => !prev)}
            aria-label={isDarkMode ? 'Use light theme' : 'Use dark theme'}
            className="focus-ring rounded-xl p-2 text-charcoal/70 transition-colors hover:bg-cream-200/60 dark:text-cream-200 dark:hover:bg-sand-dark/60"
            title={isDarkMode ? 'Switch to Cream Light Mode' : 'Switch to Cream Dark Mode'}
          >
            {isDarkMode ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-charcoal-light" />}
          </button>
        </div>
      </div>

      <nav aria-label="Primary" className="mx-auto mt-2 grid max-w-md grid-cols-3 gap-1 rounded-2xl border border-sand-border/60 bg-cream-100/75 p-1 shadow-inner dark:border-sand-darkBorder/60 dark:bg-sand-dark/75 md:hidden">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              aria-current={activeTab === item.id ? 'page' : undefined}
              className={cn(
                'focus-ring flex items-center justify-center gap-1.5 rounded-xl px-2 py-1.5 text-[11px] font-medium transition-all',
                activeTab === item.id
                  ? 'bg-[#FAF9F5] font-semibold text-charcoal shadow-sm dark:bg-[#141413] dark:text-cream-100'
                  : 'text-charcoal-muted dark:text-cream-400',
              )}
            >
              <Icon className="h-3.5 w-3.5 text-terracotta-600 dark:text-terracotta-500" />
              <span>{item.mobileLabel}</span>
            </button>
          );
        })}
      </nav>
    </header>
  );
}
