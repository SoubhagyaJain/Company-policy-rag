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
    <header className="sticky top-0 z-30 w-full backdrop-blur-xl bg-[#FAF9F5]/90 dark:bg-[#141413]/90 border-b border-[#E5E0D8]/80 dark:border-[#2A2925]/80 px-4 py-2.5 transition-colors">
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
        {/* Left Section: Sidebar Toggle & Branding */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setIsSidebarOpen((prev) => !prev)}
            title={isSidebarOpen ? 'Hide Sidebar' : 'Show Sidebar'}
            className="p-2 rounded-xl text-charcoal/70 dark:text-cream-200 hover:bg-cream-200/60 dark:hover:bg-sand-dark/60 transition-colors focus:outline-none"
          >
            <PanelLeft className="w-5 h-5" />
          </button>

          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-terracotta-500/10 dark:bg-terracotta-500/20 border border-terracotta-500/30 flex items-center justify-center text-terracotta-600 dark:text-terracotta-500 font-serif font-bold text-lg">
              A
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-serif font-semibold text-charcoal dark:text-cream-100 text-base tracking-tight">
                  Company Policy RAG
                </span>
                <span className="px-2 py-0.5 text-[10px] font-mono font-medium rounded-full bg-terracotta-500/10 dark:bg-terracotta-500/20 text-terracotta-700 dark:text-terracotta-500 border border-terracotta-500/30">
                  Anthropic v15
                </span>
              </div>
              <p className="text-[11px] text-charcoal-muted dark:text-cream-400 font-sans hidden sm:block">
                FastAPI • Hybrid RAG • SSE Streaming
              </p>
            </div>
          </div>
        </div>

        {/* Center Section: View Navigation Tabs */}
        <nav className="flex items-center gap-1 p-1 rounded-2xl bg-cream-100/80 dark:bg-sand-dark/80 border border-sand-border/60 dark:border-sand-darkBorder/60 shadow-inner">
          <button
            onClick={() => setActiveTab('chat')}
            className={cn(
              'flex items-center gap-2 px-3.5 py-1.5 rounded-xl text-xs font-medium transition-all duration-200',
              activeTab === 'chat'
                ? 'bg-[#FAF9F5] dark:bg-[#141413] text-charcoal dark:text-cream-100 shadow-sm font-semibold'
                : 'text-charcoal-muted dark:text-cream-400 hover:text-charcoal dark:hover:text-cream-200'
            )}
          >
            <MessageSquare className="w-3.5 h-3.5 text-terracotta-600 dark:text-terracotta-500" />
            <span>Chat</span>
          </button>

          <button
            onClick={() => setActiveTab('documents')}
            className={cn(
              'flex items-center gap-2 px-3.5 py-1.5 rounded-xl text-xs font-medium transition-all duration-200',
              activeTab === 'documents'
                ? 'bg-[#FAF9F5] dark:bg-[#141413] text-charcoal dark:text-cream-100 shadow-sm font-semibold'
                : 'text-charcoal-muted dark:text-cream-400 hover:text-charcoal dark:hover:text-cream-200'
            )}
          >
            <FileText className="w-3.5 h-3.5 text-terracotta-600 dark:text-terracotta-500" />
            <span>Documents</span>
          </button>

          <button
            onClick={() => setActiveTab('observability')}
            className={cn(
              'flex items-center gap-2 px-3.5 py-1.5 rounded-xl text-xs font-medium transition-all duration-200',
              activeTab === 'observability'
                ? 'bg-[#FAF9F5] dark:bg-[#141413] text-charcoal dark:text-cream-100 shadow-sm font-semibold'
                : 'text-charcoal-muted dark:text-cream-400 hover:text-charcoal dark:hover:text-cream-200'
            )}
          >
            <Activity className="w-3.5 h-3.5 text-terracotta-600 dark:text-terracotta-500" />
            <span>Telemetry</span>
          </button>
        </nav>

        {/* Right Section: System Status & Theme Toggle */}
        <div className="flex items-center gap-3">
          {/* Health Badge */}
          <div
            title={`Backend: ${isHealthy ? 'Connected' : 'Degraded'}`}
            className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-cream-100/70 dark:bg-sand-dark/70 border border-sand-border/70 dark:border-sand-darkBorder/70 text-[11px] font-medium"
          >
            <span
              className={cn(
                'w-2 h-2 rounded-full animate-pulse',
                isHealthy ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' : 'bg-amber-500'
              )}
            />
            <span className="text-charcoal dark:text-cream-200 font-mono hidden md:inline">
              {isHealthy ? 'FastAPI Active' : 'Connecting...'}
            </span>
          </div>

          {/* Theme Toggle */}
          <button
            onClick={() => setIsDarkMode((prev) => !prev)}
            className="p-2 rounded-xl text-charcoal/70 dark:text-cream-200 hover:bg-cream-200/60 dark:hover:bg-sand-dark/60 transition-colors focus:outline-none"
            title={isDarkMode ? 'Switch to Cream Light Mode' : 'Switch to Cream Dark Mode'}
          >
            {isDarkMode ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-charcoal-light" />}
          </button>
        </div>
      </div>
    </header>
  );
}
