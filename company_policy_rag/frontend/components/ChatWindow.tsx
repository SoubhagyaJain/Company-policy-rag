'use client';

import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Square,
  Sparkles,
  Filter,
  Trash2,
  SlidersHorizontal,
  Compass,
  FileQuestion,
  ShieldCheck,
  Zap,
} from 'lucide-react';
import { ChatMessageData, Citation, FilterOptions } from '../lib/types';
import { ChatMessage } from './ChatMessage';

interface ChatWindowProps {
  messages: ChatMessageData[];
  isStreaming: boolean;
  onSendMessage: (content: string, filters?: FilterOptions, model?: string) => void;
  onCancelStream: () => void;
  onClearChat: () => void;
  onOpenCitation: (citation: Citation) => void;
}

const CATEGORY_OPTIONS = [
  'All Categories',
  'HR & Benefits',
  'Operations',
  'IT & Security',
  'Finance',
  'Compliance',
];

const SUGGESTED_PROMPTS = [
  {
    title: 'Remote Work & Telecommuting',
    prompt: 'What are the rules and eligible stipends for working remotely?',
    icon: Compass,
  },
  {
    title: 'PTO & Leave Rollover',
    prompt: 'How many PTO days can I carry over into the next calendar year?',
    icon: FileQuestion,
  },
  {
    title: 'Travel Expense Policy',
    prompt: 'What is the daily meal and hotel reimbursement limit for business travel?',
    icon: ShieldCheck,
  },
  {
    title: 'IT & Security Guidelines',
    prompt: 'What is the password rotation policy and MFA requirement for corporate laptops?',
    icon: Zap,
  },
];

export function ChatWindow({
  messages,
  isStreaming,
  onSendMessage,
  onCancelStream,
  onClearChat,
  onOpenCitation,
}: ChatWindowProps) {
  const [input, setInput] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All Categories');
  const [selectedModel, setSelectedModel] = useState('FastAPI RAG');
  const [showFilters, setShowFilters] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Auto scroll to bottom when messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isStreaming]);

  const handleSend = () => {
    if (!input.trim() || isStreaming) return;
    const categoryFilter =
      selectedCategory !== 'All Categories' ? selectedCategory : undefined;
    onSendMessage(input.trim(), { category: categoryFilter }, selectedModel);
    setInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handlePromptClick = (promptText: string) => {
    onSendMessage(promptText, undefined, selectedModel);
  };

  return (
    <div className="flex-1 flex flex-col h-[calc(100vh-57px)] bg-[#FAF9F5] dark:bg-[#141413] relative overflow-hidden">
      {/* Top Bar inside Chat */}
      <div className="px-4 py-2 border-b border-[#E5E0D8]/60 dark:border-[#2A2925]/60 flex items-center justify-between backdrop-blur-md bg-[#FAF9F5]/70 dark:bg-[#141413]/70 z-10">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowFilters((prev) => !prev)}
            className="flex items-center gap-1.5 px-3 py-1 rounded-xl bg-cream-100 dark:bg-sand-dark border border-sand-border dark:border-sand-darkBorder text-xs text-charcoal dark:text-cream-200 hover:bg-cream-200 transition-colors"
          >
            <Filter className="w-3.5 h-3.5 text-terracotta-600" />
            <span className="font-medium">Category: {selectedCategory}</span>
          </button>

          {showFilters && (
            <div className="flex items-center gap-1 overflow-x-auto py-1">
              {CATEGORY_OPTIONS.map((cat) => (
                <button
                  key={cat}
                  onClick={() => {
                    setSelectedCategory(cat);
                    setShowFilters(false);
                  }}
                  className={`px-2.5 py-0.5 rounded-lg text-xs font-mono transition-colors ${
                    selectedCategory === cat
                      ? 'bg-terracotta-600 text-white font-bold'
                      : 'bg-cream-200 dark:bg-sand-dark text-charcoal dark:text-cream-300 hover:bg-cream-300'
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>
          )}
        </div>

        {messages.length > 0 && (
          <button
            onClick={onClearChat}
            className="flex items-center gap-1 text-xs text-charcoal-muted dark:text-cream-400 hover:text-rose-600 transition-colors"
            title="Clear Chat Messages"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Clear Chat</span>
          </button>
        )}
      </div>

      {/* Message Stream Scroll View */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4 custom-scrollbar">
        {messages.length === 0 ? (
          <div className="max-w-2xl mx-auto py-12 px-4 text-center space-y-6">
            <div className="w-14 h-14 mx-auto rounded-3xl bg-terracotta-500/10 dark:bg-terracotta-500/20 border border-terracotta-500/30 text-terracotta-600 dark:text-terracotta-500 flex items-center justify-center shadow-lg shadow-terracotta-500/10">
              <Sparkles className="w-7 h-7" />
            </div>

            <div className="space-y-2">
              <h2 className="font-serif font-bold text-2xl text-charcoal dark:text-cream-100">
                Company Policy & Guidebook Assistant
              </h2>
              <p className="text-sm text-charcoal-muted dark:text-cream-400 max-w-md mx-auto leading-relaxed">
                Powered by FastAPI, Dense Vector + BM25 Hybrid Retrieval, and BGE Cross-Encoder Reranking.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-4">
              {SUGGESTED_PROMPTS.map((item, idx) => {
                const IconComponent = item.icon;
                return (
                  <button
                    key={idx}
                    onClick={() => handlePromptClick(item.prompt)}
                    className="p-4 text-left rounded-2xl bg-cream-100/70 dark:bg-sand-dark/70 border border-sand-border/80 dark:border-sand-darkBorder/80 hover:border-terracotta-500/40 hover:bg-cream-100 dark:hover:bg-sand-dark transition-all duration-200 group shadow-sm"
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <IconComponent className="w-4 h-4 text-terracotta-600 dark:text-terracotta-400 group-hover:scale-110 transition-transform" />
                      <h3 className="text-xs font-semibold text-charcoal dark:text-cream-100">
                        {item.title}
                      </h3>
                    </div>
                    <p className="text-xs text-charcoal-muted dark:text-cream-400 line-clamp-2 leading-relaxed">
                      "{item.prompt}"
                    </p>
                  </button>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="max-w-4xl mx-auto space-y-4">
            {messages.map((msg) => (
              <ChatMessage key={msg.id} message={msg} onOpenCitation={onOpenCitation} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Box Footer */}
      <div className="p-4 border-t border-[#E5E0D8]/80 dark:border-[#2A2925]/80 backdrop-blur-xl bg-[#FAF9F5]/90 dark:bg-[#141413]/90">
        <div className="max-w-4xl mx-auto space-y-2">
          <div className="relative flex items-end gap-2 p-2 rounded-2xl bg-[#F3F0E6] dark:bg-[#1E1D1A] border border-[#E5E0D8] dark:border-[#2E2C27] focus-within:border-terracotta-500 shadow-inner">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                e.target.style.height = 'auto';
                e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
              }}
              onKeyDown={handleKeyDown}
              placeholder="Ask any question about company policies, benefits, travel, or IT guidelines..."
              rows={1}
              className="flex-1 bg-transparent border-none text-sm text-charcoal dark:text-cream-100 placeholder:text-charcoal-muted dark:placeholder:text-cream-500 focus:outline-none resize-none px-2 py-1.5 custom-scrollbar min-h-[40px] max-h-[160px]"
            />

            {isStreaming ? (
              <button
                onClick={onCancelStream}
                className="p-2.5 rounded-xl bg-amber-600 hover:bg-amber-700 text-white shadow-md transition-all active:scale-95"
                title="Stop Response Generation"
              >
                <Square className="w-4 h-4 fill-white" />
              </button>
            ) : (
              <button
                onClick={handleSend}
                disabled={!input.trim()}
                className="p-2.5 rounded-xl bg-terracotta-600 hover:bg-terracotta-700 disabled:opacity-40 disabled:hover:bg-terracotta-600 text-white shadow-md shadow-terracotta-600/20 transition-all active:scale-95"
                title="Send Message"
              >
                <Send className="w-4 h-4" />
              </button>
            )}
          </div>

          <div className="flex items-center justify-between text-[11px] text-charcoal-muted dark:text-cream-500 px-1">
            <span className="flex items-center gap-1 font-mono">
              <Zap className="w-3 h-3 text-amber-500" /> SSE Streaming Active
            </span>
            <span className="hidden sm:inline">
              Press <kbd className="font-mono bg-cream-200 dark:bg-sand-dark px-1 rounded">Enter</kbd> to send, <kbd className="font-mono bg-cream-200 dark:bg-sand-dark px-1 rounded">Shift + Enter</kbd> for newline
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
