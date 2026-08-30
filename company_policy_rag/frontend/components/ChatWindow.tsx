'use client';

import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send,
  Square,
  Sparkles,
  Filter,
  Trash2,
  Compass,
  FileQuestion,
  ShieldCheck,
  Zap,
  Cpu,
  ChevronDown,
  ArrowDown,
  FileText,
  Layers,
  Search,
  X,
  Check,
  LoaderCircle,
  AlertCircle,
} from 'lucide-react';
import { ChatMessageData, Citation, FilterOptions, DocumentItem, ResponseMode } from '../lib/types';
import { ChatMessage } from './ChatMessage';
import { AmbientKnowledgeField } from './AmbientKnowledgeField';
import { PolicyKnowledgeScene } from './PolicyKnowledgeScene';
import { apiClient } from '../lib/api-client';

interface ChatWindowProps {
  messages: ChatMessageData[];
  isStreaming: boolean;
  onSendMessage: (
    content: string,
    filters?: FilterOptions,
    model?: string,
    responseMode?: ResponseMode
  ) => void;
  onCancelStream: () => void;
  onClearChat: () => void;
  onOpenCitation: (citation: Citation) => void;
}

const DEFAULT_CATEGORIES = [
  'HR & Benefits',
  'Operations',
  'IT & Security',
  'Finance',
  'Legal & Compliance',
  'General',
];

const MODEL_OPTIONS = [
  { id: 'qwen2.5:7b', label: 'Qwen 2.5 7B', desc: 'Fast & balanced (Recommended)' },
  { id: 'llama3.2:3b', label: 'Llama 3.2 3B', desc: 'Ultra-fast compact model' },
  { id: 'gemma4-policy-fast:latest', label: 'Gemma 4 Policy Fast', desc: 'Policy specialized model' },
  { id: 'gemma4:12b', label: 'Gemma 4 12B', desc: 'High capability model' },
];

type ModelOption = {
  id: string;
  label: string;
  desc: string;
  badges?: string[];
};

const RESPONSE_MODE_OPTIONS: Array<{
  id: ResponseMode;
  label: string;
  description: string;
}> = [
  { id: 'compact', label: 'Compact', description: 'Quick answer with essentials.' },
  { id: 'standard', label: 'Standard', description: 'Balanced explanation.' },
  { id: 'detailed', label: 'Detailed', description: 'Deep explanation with examples and broader evidence.' },
];

const SUGGESTED_PROMPTS = [
  {
    title: 'Remote Work & Stipends',
    prompt: 'What are the rules and eligible stipends for working remotely?',
    icon: Compass,
  },
  {
    title: 'PTO & Rollover Policy',
    prompt: 'How many PTO days can I carry over into the next calendar year?',
    icon: FileQuestion,
  },
  {
    title: 'Travel Expense Guidelines',
    prompt: 'What is the daily meal and hotel reimbursement limit for business travel?',
    icon: ShieldCheck,
  },
  {
    title: 'IT Security & Passwords',
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
  const [selectedCategory, setSelectedCategory] = useState<string>('All');
  const [selectedDocId, setSelectedDocId] = useState<string>('All');
  const [availableDocs, setAvailableDocs] = useState<DocumentItem[]>([]);
  const [filterTab, setFilterTab] = useState<'documents' | 'categories'>('documents');
  const [filterSearch, setFilterSearch] = useState('');
  const [modelsList, setModelsList] = useState<ModelOption[]>(MODEL_OPTIONS);
  const [selectedModel, setSelectedModel] = useState('qwen2.5:7b');
  const [pendingModel, setPendingModel] = useState<string | null>(null);
  const [modelSwitchError, setModelSwitchError] = useState<string | null>(null);
  const [responseMode, setResponseMode] = useState<ResponseMode>('standard');
  const [showFilters, setShowFilters] = useState(false);
  const [showModelPicker, setShowModelPicker] = useState(false);
  const [showScrollBottom, setShowScrollBottom] = useState(false);

  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const isUserScrolledUp = useRef<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const filterRef = useRef<HTMLDivElement | null>(null);
  const modelPickerRef = useRef<HTMLDivElement | null>(null);
  const modelSelectionInitializedRef = useRef(false);

  // Load available models from backend API
  const loadModels = useCallback(async () => {
    try {
      const res = await apiClient.getModels();
      if (res && Array.isArray(res.models) && res.models.length > 0) {
        const chatModels = res.models
          .filter((m) => m.type === 'llm' || !m.type.includes('embed'))
          .map((m) => ({
            id: m.id,
            label: m.name || m.id,
            desc: [m.parameter_size, m.quantization, m.family]
              .filter(Boolean)
              .join(' · ') || 'Installed model',
            badges: m.badges || [],
          }));
        if (chatModels.length > 0) {
          setModelsList(chatModels);
          setSelectedModel((currentModel) => {
            if (!modelSelectionInitializedRef.current) {
              modelSelectionInitializedRef.current = true;
              const storedModel = localStorage.getItem('rag_model');
              if (storedModel && chatModels.some((m) => m.id === storedModel)) {
                return storedModel;
              }
              if (res.active_model && chatModels.some((m) => m.id === res.active_model)) {
                return res.active_model;
              }
            }
            return chatModels.some((m) => m.id === currentModel)
              ? currentModel
              : (res.active_model || chatModels[0].id);
          });
        }
      }
    } catch {
      // fallback to MODEL_OPTIONS
    }
  }, []);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  // Load available documents from backend
  const loadDocuments = useCallback(async () => {
    try {
      const docs = await apiClient.getDocuments();
      if (Array.isArray(docs)) {
        setAvailableDocs(docs);
      }
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  // When filter dropdown opens, refresh document list
  useEffect(() => {
    if (showFilters) {
      loadDocuments();
    }
    if (showModelPicker) {
      loadModels();
    }
  }, [showFilters, showModelPicker, loadDocuments, loadModels]);

  // Click-outside listener for dropdowns
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (filterRef.current && !filterRef.current.contains(e.target as Node)) {
        setShowFilters(false);
      }
      if (modelPickerRef.current && !modelPickerRef.current.contains(e.target as Node)) {
        setShowModelPicker(false);
      }
    };

    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setShowFilters(false);
        setShowModelPicker(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleEsc);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEsc);
    };
  }, []);

  useEffect(() => {
    const storedMode = localStorage.getItem('rag_response_mode');
    if (storedMode === 'compact' || storedMode === 'standard' || storedMode === 'detailed') {
      setResponseMode(storedMode);
    }
  }, []);

  const handleSelectResponseMode = (mode: ResponseMode) => {
    setResponseMode(mode);
    localStorage.setItem('rag_response_mode', mode);
  };

  const handleSelectModel = async (modelId: string) => {
    if (pendingModel || modelId === selectedModel) {
      setShowModelPicker(false);
      return;
    }

    const previousModel = selectedModel;
    setPendingModel(modelId);
    setModelSwitchError(null);
    setSelectedModel(modelId);
    try {
      const result = await apiClient.selectModel(modelId);
      setSelectedModel(result.active_model);
      localStorage.setItem('rag_model', result.active_model);
      setShowModelPicker(false);
    } catch (error) {
      setSelectedModel(previousModel);
      setShowModelPicker(false);
      setModelSwitchError(
        error instanceof Error ? error.message : `Unable to switch to ${modelId}`
      );
    } finally {
      setPendingModel(null);
    }
  };


  // Derive unique categories from available docs merged with standard categories
  const categoriesList = useMemo(() => {
    const set = new Set<string>(DEFAULT_CATEGORIES);
    availableDocs.forEach((d) => {
      if (d.category && d.category.trim()) {
        set.add(d.category.trim());
      }
    });
    return Array.from(set);
  }, [availableDocs]);

  // Filtered lists for the search input inside filter popover
  const filteredDocs = useMemo(() => {
    if (!filterSearch.trim()) return availableDocs;
    const q = filterSearch.toLowerCase();
    return availableDocs.filter(
      (d) =>
        d.filename.toLowerCase().includes(q) ||
        (d.category && d.category.toLowerCase().includes(q))
    );
  }, [availableDocs, filterSearch]);

  const filteredCategories = useMemo(() => {
    if (!filterSearch.trim()) return categoriesList;
    const q = filterSearch.toLowerCase();
    return categoriesList.filter((c) => c.toLowerCase().includes(q));
  }, [categoriesList, filterSearch]);

  const selectedDocument = useMemo(
    () => availableDocs.find((doc) => doc.id === selectedDocId),
    [availableDocs, selectedDocId]
  );
  const selectedDoc = selectedDocument?.filename ?? 'All';

  useEffect(() => {
    if (messages.length === 0 && scrollContainerRef.current) {
      isUserScrolledUp.current = false;
      setShowScrollBottom(false);
      const resetScroll = () => {
        if (scrollContainerRef.current) scrollContainerRef.current.scrollTop = 0;
      };
      resetScroll();
      const frame = window.requestAnimationFrame(resetScroll);
      const timer = window.setTimeout(resetScroll, 120);
      return () => {
        window.cancelAnimationFrame(frame);
        window.clearTimeout(timer);
      };
    }
  }, [messages.length]);

  // Check if any filter is active
  const isFilterActive = selectedDocId !== 'All' || selectedCategory !== 'All';

  // Filter label to display in the button
  const filterButtonLabel = useMemo(() => {
    if (selectedDoc !== 'All') {
      return selectedDoc.length > 18 ? selectedDoc.slice(0, 15) + '...' : selectedDoc;
    }
    if (selectedCategory !== 'All') {
      return selectedCategory;
    }
    return 'Filter Documents';
  }, [selectedDoc, selectedCategory]);

  const clearAllFilters = (e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    setSelectedDocId('All');
    setSelectedCategory('All');
  };

  // Scroll detection handler
  const handleScroll = useCallback(() => {
    if (!scrollContainerRef.current) return;
    if (messages.length === 0) {
      isUserScrolledUp.current = false;
      setShowScrollBottom(false);
      return;
    }
    const { scrollTop, scrollHeight, clientHeight } = scrollContainerRef.current;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;

    if (distanceFromBottom > 140) {
      isUserScrolledUp.current = true;
      setShowScrollBottom(true);
    } else {
      isUserScrolledUp.current = false;
      setShowScrollBottom(false);
    }
  }, [messages.length]);

  // Smooth scroll to bottom function
  const scrollToBottom = useCallback((smooth = true) => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTo({
        top: scrollContainerRef.current.scrollHeight,
        behavior: smooth ? 'smooth' : 'auto',
      });
      isUserScrolledUp.current = false;
      setShowScrollBottom(false);
    }
  }, []);

  // Auto-scroll when messages update or during streaming, UNLESS user scrolled up
  useEffect(() => {
    if (!isUserScrolledUp.current && scrollContainerRef.current) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [messages, isStreaming]);

  const handleSend = () => {
    if (!input.trim() || isStreaming) return;

    const filters: FilterOptions = {};
    if (selectedDocument) {
      filters.source_file = selectedDocument.filename;
      filters.document_id = selectedDocument.id;
    }
    if (selectedCategory !== 'All') {
      filters.category = selectedCategory;
    }

    isUserScrolledUp.current = false;
    setShowScrollBottom(false);
    onSendMessage(
      input.trim(),
      Object.keys(filters).length > 0 ? filters : undefined,
      selectedModel,
      responseMode
    );
    setInput('');

    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    setTimeout(() => scrollToBottom(true), 50);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handlePromptClick = (promptText: string) => {
    isUserScrolledUp.current = false;
    const filters: FilterOptions = {};
    if (selectedDocument) {
      filters.source_file = selectedDocument.filename;
      filters.document_id = selectedDocument.id;
    }
    if (selectedCategory !== 'All') filters.category = selectedCategory;

    onSendMessage(
      promptText,
      Object.keys(filters).length > 0 ? filters : undefined,
      selectedModel,
      responseMode
    );
    setTimeout(() => scrollToBottom(true), 50);
  };

  return (
    <div className="flex-1 flex flex-col h-full min-h-0 bg-[#F7F3EA] dark:bg-[#151512] relative overflow-hidden font-sans isolation-isolate">
      <AmbientKnowledgeField />
      {/* Top Bar - Minimalist Anthropic Header */}
      <div className="px-4 py-2.5 border-b border-[#E8E2D5]/60 dark:border-[#302D27]/60 flex items-center justify-between bg-[#FAF8F5]/90 dark:bg-[#161513]/90 z-20">
        <div className="flex items-center gap-2">
          {/* Enhanced Document / Category Filter Pill */}
          <div className="relative" ref={filterRef}>
            <button
              onClick={() => setShowFilters((prev) => !prev)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs transition-all ${
                isFilterActive
                  ? 'bg-terracotta-500/15 dark:bg-terracotta-500/25 border border-terracotta-500/50 text-terracotta-800 dark:text-terracotta-300 font-semibold shadow-xs'
                  : 'bg-[#EFECE2]/80 hover:bg-[#E8E3D5] dark:bg-[#23211E] dark:hover:bg-[#2B2925] border border-[#E0D9CB] dark:border-[#33302B] text-[#38342F] dark:text-[#E2DDD5]'
              }`}
              title="Filter retrieval to specific documents or categories"
            >
              <Filter
                className={`w-3 h-3 ${
                  isFilterActive
                    ? 'text-terracotta-600 dark:text-terracotta-400'
                    : 'text-terracotta-600 dark:text-terracotta-400'
                }`}
              />
              <span className="font-medium text-xs truncate max-w-[160px]">
                {filterButtonLabel}
              </span>
              {isFilterActive ? (
                <span
                  role="button"
                  onClick={clearAllFilters}
                  className="hover:bg-terracotta-500/20 p-0.5 rounded-full ml-0.5 transition-colors"
                  title="Clear filter"
                >
                  <X className="w-3 h-3" />
                </span>
              ) : (
                <ChevronDown className="w-3 h-3 opacity-60 ml-0.5" />
              )}
            </button>

            {/* Filter Dropdown Modal / Popover */}
            {showFilters && (
              <div className="absolute left-0 top-full mt-2 w-80 sm:w-96 rounded-2xl bg-white dark:bg-[#1E1D1A] border border-[#DDD5C5] dark:border-[#33302B] shadow-2xl z-50 p-3 animate-in fade-in zoom-in-95 duration-100 space-y-3">
                {/* Popover Header */}
                <div className="flex items-center justify-between pb-1 border-b border-[#EAE4D8] dark:border-[#2C2A26]">
                  <div className="flex items-center gap-1.5 text-xs font-semibold text-[#2D2A26] dark:text-[#FAF8F5]">
                    <Filter className="w-3.5 h-3.5 text-terracotta-600" />
                    <span>Filter Knowledge Base</span>
                  </div>
                  {isFilterActive && (
                    <button
                      onClick={clearAllFilters}
                      className="text-[11px] text-terracotta-600 hover:text-terracotta-700 dark:text-terracotta-400 font-medium hover:underline flex items-center gap-1"
                    >
                      <X className="w-3 h-3" /> Reset Filter
                    </button>
                  )}
                </div>

                {/* Search Bar */}
                <div className="relative">
                  <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-[#8C867B] dark:text-[#7A756C]" />
                  <input
                    type="text"
                    value={filterSearch}
                    onChange={(e) => setFilterSearch(e.target.value)}
                    placeholder="Search documents or categories..."
                    className="w-full pl-8 pr-3 py-1.5 bg-[#F6F3EB] dark:bg-[#252320] rounded-xl text-xs text-[#2D2A26] dark:text-[#E8E4DD] placeholder:text-[#8C867B] dark:placeholder:text-[#7A756C] border border-[#E5DFD2] dark:border-[#34312C] focus:outline-none focus:border-terracotta-500/60"
                  />
                  {filterSearch && (
                    <button
                      onClick={() => setFilterSearch('')}
                      className="absolute right-2.5 top-2.5 text-[#8C867B] hover:text-[#2D2A26]"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  )}
                </div>

                {/* Segmented Filter Mode Tabs */}
                <div className="flex items-center gap-1 p-1 bg-[#F3EFE6] dark:bg-[#181715] rounded-xl text-xs font-medium">
                  <button
                    onClick={() => setFilterTab('documents')}
                    className={`flex-1 py-1 px-2.5 rounded-lg flex items-center justify-center gap-1.5 transition-all ${
                      filterTab === 'documents'
                        ? 'bg-white dark:bg-[#262420] text-[#2D2A26] dark:text-[#FAF8F5] shadow-xs font-semibold'
                        : 'text-[#6C665C] dark:text-[#9A9386] hover:text-[#2D2A26]'
                    }`}
                  >
                    <FileText className="w-3.5 h-3.5" />
                    <span>Documents ({availableDocs.length})</span>
                  </button>
                  <button
                    onClick={() => setFilterTab('categories')}
                    className={`flex-1 py-1 px-2.5 rounded-lg flex items-center justify-center gap-1.5 transition-all ${
                      filterTab === 'categories'
                        ? 'bg-white dark:bg-[#262420] text-[#2D2A26] dark:text-[#FAF8F5] shadow-xs font-semibold'
                        : 'text-[#6C665C] dark:text-[#9A9386] hover:text-[#2D2A26]'
                    }`}
                  >
                    <Layers className="w-3.5 h-3.5" />
                    <span>Categories ({categoriesList.length})</span>
                  </button>
                </div>

                {/* Scrollable Options List */}
                <div className="max-h-56 overflow-y-auto space-y-1 custom-scrollbar pr-1">
                  {filterTab === 'documents' ? (
                    <>
                      {/* All Documents Option */}
                      <button
                        onClick={() => {
                          setSelectedDocId('All');
                          setShowFilters(false);
                        }}
                        className={`w-full flex items-center justify-between px-3 py-2 rounded-xl text-left text-xs transition-colors ${
                          selectedDocId === 'All'
                            ? 'bg-terracotta-600/10 dark:bg-terracotta-600/20 text-terracotta-800 dark:text-terracotta-300 font-semibold border border-terracotta-500/30'
                            : 'hover:bg-[#F3EFE6] dark:hover:bg-[#262420] text-[#332F2A] dark:text-[#DCD5C9]'
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <Layers className="w-3.5 h-3.5 text-terracotta-600" />
                          <span>All Documents (Unfiltered)</span>
                        </div>
                        {selectedDocId === 'All' && (
                          <Check className="w-3.5 h-3.5 text-terracotta-600" />
                        )}
                      </button>

                      {filteredDocs.length === 0 ? (
                        <div className="text-center py-4 text-xs text-[#8C867B] dark:text-[#7A756C]">
                          No documents match "{filterSearch}"
                        </div>
                      ) : (
                        filteredDocs.map((doc) => (
                          <button
                            key={doc.id || doc.filename}
                            onClick={() => {
                              setSelectedDocId(doc.id);
                              setShowFilters(false);
                            }}
                            className={`w-full flex items-center justify-between px-3 py-2 rounded-xl text-left text-xs transition-colors ${
                              selectedDocId === doc.id
                                ? 'bg-terracotta-600/10 dark:bg-terracotta-600/20 text-terracotta-800 dark:text-terracotta-300 font-semibold border border-terracotta-500/30'
                                : 'hover:bg-[#F3EFE6] dark:hover:bg-[#262420] text-[#332F2A] dark:text-[#DCD5C9]'
                            }`}
                          >
                            <div className="flex items-center gap-2 min-w-0 pr-2">
                              <FileText className="w-3.5 h-3.5 shrink-0 text-terracotta-600" />
                              <div className="truncate">
                                <div className="truncate font-medium">{doc.filename}</div>
                                <div className="text-[10px] text-[#8C867B] dark:text-[#7A756C] flex items-center gap-2">
                                  <span>{doc.category || 'General'}</span>
                                  {doc.chunks_count ? (
                                    <span>• {doc.chunks_count} chunks</span>
                                  ) : null}
                                </div>
                              </div>
                            </div>
                            {selectedDocId === doc.id && (
                              <Check className="w-3.5 h-3.5 shrink-0 text-terracotta-600" />
                            )}
                          </button>
                        ))
                      )}
                    </>
                  ) : (
                    <>
                      {/* All Categories Option */}
                      <button
                        onClick={() => {
                          setSelectedCategory('All');
                          setShowFilters(false);
                        }}
                        className={`w-full flex items-center justify-between px-3 py-2 rounded-xl text-left text-xs transition-colors ${
                          selectedCategory === 'All'
                            ? 'bg-terracotta-600/10 dark:bg-terracotta-600/20 text-terracotta-800 dark:text-terracotta-300 font-semibold border border-terracotta-500/30'
                            : 'hover:bg-[#F3EFE6] dark:hover:bg-[#262420] text-[#332F2A] dark:text-[#DCD5C9]'
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <Layers className="w-3.5 h-3.5 text-terracotta-600" />
                          <span>All Categories</span>
                        </div>
                        {selectedCategory === 'All' && (
                          <Check className="w-3.5 h-3.5 text-terracotta-600" />
                        )}
                      </button>

                      {filteredCategories.map((cat) => (
                        <button
                          key={cat}
                          onClick={() => {
                            setSelectedCategory(cat);
                            setShowFilters(false);
                          }}
                          className={`w-full flex items-center justify-between px-3 py-2 rounded-xl text-left text-xs transition-colors ${
                            selectedCategory === cat
                              ? 'bg-terracotta-600/10 dark:bg-terracotta-600/20 text-terracotta-800 dark:text-terracotta-300 font-semibold border border-terracotta-500/30'
                              : 'hover:bg-[#F3EFE6] dark:hover:bg-[#262420] text-[#332F2A] dark:text-[#DCD5C9]'
                          }`}
                        >
                          <div className="flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-terracotta-600/70" />
                            <span>{cat}</span>
                          </div>
                          {selectedCategory === cat && (
                            <Check className="w-3.5 h-3.5 text-terracotta-600" />
                          )}
                        </button>
                      ))}
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right side: Model Selector & Clear Chat */}
        <div className="flex items-center gap-2">
          {/* Model Switcher */}
          <div className="relative" ref={modelPickerRef}>
            <button
              onClick={() => setShowModelPicker((prev) => !prev)}
              disabled={Boolean(pendingModel)}
              aria-busy={Boolean(pendingModel)}
              aria-label={`Inference model: ${selectedModel}`}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[#EFECE2]/80 hover:bg-[#E8E3D5] dark:bg-[#23211E] dark:hover:bg-[#2B2925] border border-[#E0D9CB] dark:border-[#33302B] text-xs text-[#38342F] dark:text-[#E2DDD5] transition-colors"
            >
              {pendingModel ? (
                <LoaderCircle className="h-3 w-3 animate-spin text-terracotta-600 dark:text-terracotta-400" />
              ) : (
                <Cpu className="w-3 h-3 text-terracotta-600 dark:text-terracotta-400" />
              )}
              <span className="font-medium">
                {pendingModel
                  ? `Switching to ${modelsList.find((m) => m.id === pendingModel)?.label || pendingModel}`
                  : modelsList.find((m) => m.id === selectedModel)?.label || selectedModel}
              </span>
              <ChevronDown className="w-3 h-3 opacity-60" />
            </button>

            {showModelPicker && (
              <div className="absolute right-0 top-full mt-1.5 w-60 rounded-2xl bg-white dark:bg-[#1E1D1A] border border-[#DDD5C5] dark:border-[#33302B] shadow-xl z-50 p-1.5 animate-in fade-in zoom-in-95 duration-100 max-h-72 overflow-y-auto custom-scrollbar">
                <div className="px-2.5 py-1 text-[10px] uppercase font-semibold text-[#8C867B] dark:text-[#7A756C] tracking-wider flex items-center justify-between">
                  <span>Select Inference Model</span>
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                </div>
                {modelsList.map((m) => (
                  <button
                    key={m.id}
                    onClick={() => handleSelectModel(m.id)}
                    disabled={Boolean(pendingModel)}
                    className={`w-full flex items-start gap-2.5 px-3 py-2 rounded-xl text-left transition-colors ${
                      selectedModel === m.id
                        ? 'bg-terracotta-600/10 dark:bg-terracotta-600/20 text-terracotta-700 dark:text-terracotta-400 font-medium'
                        : 'hover:bg-[#F3F0E6] dark:hover:bg-[#282622] text-[#38342F] dark:text-[#E2DDD5]'
                    }`}
                  >
                    <Cpu
                      className={`w-3.5 h-3.5 mt-0.5 shrink-0 ${
                        selectedModel === m.id
                          ? 'text-terracotta-600'
                          : 'text-[#8C867B] dark:text-[#7A756C]'
                      }`}
                    />
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-semibold flex items-center justify-between">
                        <span className="truncate">{m.label}</span>
                        {selectedModel === m.id && (
                          <Check className="w-3 h-3 text-terracotta-600 shrink-0" />
                        )}
                      </div>
                      <div className="text-[10px] text-[#8C867B] dark:text-[#7A756C] truncate">
                        {m.desc}
                      </div>
                      {m.badges && m.badges.length > 0 && (
                        <div className="mt-1 flex flex-wrap gap-1">
                          {m.badges.map((badge) => (
                            <span key={badge} className="rounded-full bg-[#EFE8DC] px-1.5 py-0.5 text-[8px] font-semibold text-[#766D61] dark:bg-[#302D27] dark:text-[#AAA399]">
                              {badge}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            )}
            {modelSwitchError && (
              <div
                role="alert"
                className="absolute right-0 top-full z-50 mt-1.5 flex w-72 items-start gap-2 rounded-xl border border-rose-300 bg-rose-50 px-3 py-2 text-[11px] text-rose-700 shadow-lg dark:border-rose-900/70 dark:bg-rose-950/80 dark:text-rose-300"
              >
                <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <span>{modelSwitchError}</span>
              </div>
            )}
          </div>

          {messages.length > 0 && (
            <button
              onClick={onClearChat}
              className="p-1.5 rounded-full text-[#8C867B] dark:text-[#7A756C] hover:text-rose-600 dark:hover:text-rose-400 hover:bg-[#EFECE2] dark:hover:bg-[#23211E] transition-colors"
              title="Clear current chat"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Main Messages & Welcome View */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        style={{ willChange: 'transform, scroll-position', transform: 'translateZ(0)' }}
        className="relative z-10 flex-1 overflow-y-auto px-4 sm:px-8 py-6 space-y-4 custom-scrollbar"
      >
        {messages.length === 0 ? (
          <div className="welcome-stage max-w-6xl mx-auto py-5 sm:py-8 lg:py-10 animate-in fade-in duration-300">
            <div className="grid items-center gap-7 lg:grid-cols-[1.05fr_0.95fr] lg:gap-10">
              <section className="text-left">
                <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-terracotta-500/25 bg-[#FFF9F1]/70 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-terracotta-700 shadow-sm dark:bg-[#201D18]/75 dark:text-terracotta-400">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 shadow-[0_0_0_4px_rgba(16,185,129,.12)]" />
                  Grounded policy intelligence
                </div>

                <h1 className="max-w-xl font-serif text-4xl font-normal leading-[1.04] tracking-[-0.035em] text-[#211E1A] dark:text-[#FAF8F5] sm:text-5xl lg:text-[3.45rem]">
                  Answers you can trace back to policy.
                </h1>
                <p className="mt-5 max-w-xl text-sm leading-7 text-[#635E54] dark:text-[#AAA397] sm:text-[15px]">
                  Ask a workplace question and get a concise answer grounded in your indexed
                  handbook, benefits, travel, finance, and security documents.
                </p>

                <div className="mt-6 flex flex-wrap gap-2.5">
                  <div className="welcome-proof">
                    <ShieldCheck className="h-3.5 w-3.5 text-emerald-600" />
                    Page-level citations
                  </div>
                  <div className="welcome-proof">
                    <Search className="h-3.5 w-3.5 text-terracotta-600" />
                    Hybrid search
                  </div>
                  <div className="welcome-proof">
                    <FileText className="h-3.5 w-3.5 text-amber-600" />
                    {availableDocs.length > 0
                      ? `${availableDocs.length} indexed ${availableDocs.length === 1 ? 'source' : 'sources'}`
                      : 'Indexed knowledge base'}
                  </div>
                </div>
              </section>

              <PolicyKnowledgeScene />
            </div>

            <div className="mt-8 flex items-end justify-between gap-4 border-t border-[#DDD5C5]/70 pt-5 dark:border-[#35322C]/70">
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-[0.15em] text-terracotta-700 dark:text-terracotta-400">
                  Start with a common question
                </p>
                <p className="mt-1 text-xs text-[#777064] dark:text-[#8F897E]">
                  Select a prompt or write your own below.
                </p>
              </div>
              <Sparkles className="hidden h-5 w-5 text-terracotta-600/50 sm:block" />
            </div>

            <div className="mt-3 grid grid-cols-1 gap-3 text-left sm:grid-cols-2 xl:grid-cols-4">
              {SUGGESTED_PROMPTS.map((item, idx) => {
                const Icon = item.icon;
                return (
                  <button
                    key={idx}
                    onClick={() => handlePromptClick(item.prompt)}
                    className="policy-prompt-card group rounded-2xl p-4 transition-all duration-200 hover:-translate-y-0.5"
                  >
                    <div className="mb-3 flex items-center justify-between">
                      <span className="flex h-8 w-8 items-center justify-center rounded-xl border border-terracotta-500/20 bg-terracotta-500/10 text-terracotta-600 transition-transform group-hover:scale-105 dark:text-terracotta-400">
                        <Icon className="h-4 w-4" />
                      </span>
                      <span className="text-[10px] font-medium text-[#9A9386] transition-colors group-hover:text-terracotta-600 dark:text-[#716C63]">
                        Ask →
                      </span>
                    </div>
                    <h3 className="text-xs font-semibold text-[#2D2A26] dark:text-[#E8E4DD]">
                      {item.title}
                    </h3>
                    <p className="mt-1.5 line-clamp-2 text-[11px] leading-relaxed text-[#6B655B] dark:text-[#A8A295]">
                      {item.prompt}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto space-y-4 pb-4">
            {messages.map((msg) => (
              <ChatMessage key={msg.id} message={msg} onOpenCitation={onOpenCitation} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Floating "Scroll to Bottom" Button */}
      <AnimatePresence>
        {showScrollBottom && (
          <motion.div
            initial={{ opacity: 0, y: 12, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 12, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="absolute bottom-24 right-6 sm:right-12 z-30"
          >
            <button
              onClick={() => scrollToBottom(true)}
              className="flex items-center gap-2 px-3.5 py-2 rounded-full bg-[#FAF8F5] dark:bg-[#252420] text-[#2D2A26] dark:text-[#EAE5DC] border border-[#DDD5C5] dark:border-[#383530] shadow-lg hover:bg-[#F3EFE6] dark:hover:bg-[#2F2D28] text-xs font-medium transition-all active:scale-95 group"
            >
              {isStreaming ? (
                <span className="w-2 h-2 rounded-full bg-terracotta-600 animate-ping" />
              ) : (
                <ArrowDown className="w-3.5 h-3.5 text-terracotta-600 group-hover:translate-y-0.5 transition-transform" />
              )}
              <span>{isStreaming ? 'Streaming response...' : 'Scroll to bottom'}</span>
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Floating Claude-Style Input Container */}
      <div className="p-4 sm:p-5 bg-gradient-to-t from-[#F7F3EA] via-[#F7F3EA]/82 to-transparent dark:from-[#151512] dark:via-[#151512]/84 z-20">
        <div className="max-w-3xl mx-auto space-y-2">
          {/* Main Rounded Input Box */}
          <div className="relative rounded-3xl bg-[#F8F4EB]/94 dark:bg-[#201F1C]/94 border border-white/65 dark:border-[#3A3731]/75 focus-within:border-terracotta-500/70 focus-within:ring-2 focus-within:ring-terracotta-500/15 shadow-[0_16px_50px_rgba(63,48,31,0.11)] p-2 sm:p-3 transition-all">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                e.target.style.height = 'auto';
                e.target.style.height = `${Math.min(e.target.scrollHeight, 180)}px`;
              }}
              onKeyDown={handleKeyDown}
              placeholder={
                isFilterActive
                  ? `Ask about ${selectedDoc !== 'All' ? selectedDoc : selectedCategory}...`
                  : 'Ask about policies, benefits, expenses, travel...'
              }
              rows={1}
              className="w-full bg-transparent border-none text-sm text-[#1E1C1A] dark:text-[#FAF8F5] placeholder:text-[#8E887D] dark:placeholder:text-[#7A756C] focus:outline-none resize-none px-2 py-1 custom-scrollbar min-h-[38px] max-h-[180px] leading-relaxed font-sans"
            />

            <div
              role="radiogroup"
              aria-label="Answer depth"
              className="mx-1 mb-1.5 flex items-center gap-1 overflow-x-auto rounded-xl border border-[#E1D9CB]/80 bg-[#EEE9DE]/65 p-1 dark:border-[#39362F] dark:bg-[#191815]/70"
            >
              <span className="hidden shrink-0 px-1.5 text-[10px] font-semibold uppercase tracking-[0.11em] text-[#7A7468] dark:text-[#8C867B] sm:inline">
                Answer depth
              </span>
              {RESPONSE_MODE_OPTIONS.map((mode) => {
                const selected = responseMode === mode.id;
                return (
                  <button
                    key={mode.id}
                    type="button"
                    role="radio"
                    aria-checked={selected}
                    onClick={() => handleSelectResponseMode(mode.id)}
                    title={mode.description}
                    className={`min-w-0 flex-1 rounded-lg px-2.5 py-1.5 text-[11px] font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-terracotta-500/70 sm:flex-none ${
                      selected
                        ? 'bg-white text-terracotta-800 shadow-sm ring-1 ring-terracotta-500/25 dark:bg-[#302D27] dark:text-terracotta-300'
                        : 'text-[#6F695E] hover:bg-white/60 hover:text-[#2D2A26] dark:text-[#9D978C] dark:hover:bg-[#272520] dark:hover:text-[#EEE9DF]'
                    }`}
                  >
                    {mode.label}
                  </button>
                );
              })}
            </div>

            {/* Bottom Toolbar inside the Input Box */}
            <div className="flex items-center justify-between pt-1 px-1">
              <div className="flex items-center gap-2 text-[11px] text-[#7A7468] dark:text-[#8C867B]">
                <span className="flex items-center gap-1 font-mono text-[10px]">
                  <Zap className="w-3 h-3 text-terracotta-600" /> SSE Grounded
                </span>
                {isFilterActive && (
                  <span className="hidden sm:inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-terracotta-500/15 text-terracotta-800 dark:text-terracotta-300 font-medium text-[10px] border border-terracotta-500/30">
                    Scoped: {filterButtonLabel}
                  </span>
                )}
              </div>

              <div className="flex items-center gap-2">
                {isStreaming ? (
                  <button
                    onClick={onCancelStream}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[#E5DFD0] dark:bg-[#302E29] hover:bg-[#DCD5C4] text-[#2D2A26] dark:text-[#E8E4DD] border border-[#D5CDBC] dark:border-[#3D3A34] text-xs font-medium shadow-xs transition-all active:scale-95"
                    title="Stop generation"
                  >
                    <Square className="w-3 h-3 fill-terracotta-600 text-terracotta-600" />
                    <span>Stop</span>
                  </button>
                ) : (
                  <button
                    onClick={handleSend}
                    disabled={!input.trim()}
                    className="w-8 h-8 rounded-full bg-terracotta-600 hover:bg-terracotta-700 disabled:opacity-30 disabled:hover:bg-terracotta-600 text-white flex items-center justify-center shadow-sm shadow-terracotta-600/30 transition-all active:scale-95"
                    title="Send message"
                  >
                    <Send className="w-3.5 h-3.5 stroke-[2.5]" />
                  </button>
                )}
              </div>
            </div>
          </div>

          <div className="text-center text-[10px] text-[#8C867B] dark:text-[#6C675E] select-none">
            Grounded responses based on indexed company documentation. Press{' '}
            <kbd className="font-mono bg-[#EFECE2] dark:bg-[#201F1C] px-1 py-0.5 rounded border border-[#DDD5C5] dark:border-[#2E2C28]">
              Enter
            </kbd>{' '}
            to send.
          </div>
        </div>
      </div>
    </div>
  );
}

