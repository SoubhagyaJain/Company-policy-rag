'use client';

/**
 * useComposerControls — model switching, answer depth, and document/category
 * filtering for the Space composer. Ported verbatim from the presentational
 * logic in components/ChatWindow.tsx so the backend contract is unchanged:
 *   - GET /api/models + optimistic selectModel() with rollback, persisted to
 *     localStorage 'rag_model'
 *   - response_mode persisted to localStorage 'rag_response_mode'
 *   - FilterOptions built as { document_id, source_file, category } exactly as
 *     ChatWindow.handleSend()
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { apiClient } from '../lib/api-client';
import type { DocumentItem, FilterOptions, ResponseMode } from '../lib/types';

export interface ModelOption {
  id: string;
  label: string;
  desc: string;
  badges?: string[];
}

const MODEL_OPTIONS: ModelOption[] = [
  { id: 'qwen2.5:7b', label: 'Qwen 2.5 7B', desc: 'Fast & balanced (Recommended)' },
  { id: 'llama3.2:3b', label: 'Llama 3.2 3B', desc: 'Ultra-fast compact model' },
  { id: 'gemma4-policy-fast:latest', label: 'Gemma 4 Policy Fast', desc: 'Policy specialized model' },
  { id: 'gemma4:12b', label: 'Gemma 4 12B', desc: 'High capability model' },
];

const DEFAULT_CATEGORIES = [
  'HR & Benefits',
  'Operations',
  'IT & Security',
  'Finance',
  'Legal & Compliance',
  'General',
];

export function useComposerControls() {
  const [modelsList, setModelsList] = useState<ModelOption[]>(MODEL_OPTIONS);
  const [selectedModel, setSelectedModel] = useState('qwen2.5:7b');
  const [pendingModel, setPendingModel] = useState<string | null>(null);
  const [modelSwitchError, setModelSwitchError] = useState<string | null>(null);
  const modelInitRef = useRef(false);

  const [responseMode, setResponseModeState] = useState<ResponseMode>('standard');

  const [availableDocs, setAvailableDocs] = useState<DocumentItem[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string>('All');
  const [selectedCategory, setSelectedCategory] = useState<string>('All');
  const [filterSearch, setFilterSearch] = useState('');

  const loadModels = useCallback(async () => {
    try {
      const res = await apiClient.getModels();
      if (res && Array.isArray(res.models) && res.models.length > 0) {
        const chatModels = res.models
          .filter((m) => m.type === 'llm' || !m.type.includes('embed'))
          .map((m) => ({
            id: m.id,
            label: m.name || m.id,
            desc:
              [m.parameter_size, m.quantization, m.family].filter(Boolean).join(' · ') ||
              'Installed model',
            badges: m.badges || [],
          }));
        if (chatModels.length > 0) {
          setModelsList(chatModels);
          setSelectedModel((current) => {
            if (!modelInitRef.current) {
              modelInitRef.current = true;
              const stored = localStorage.getItem('rag_model');
              if (stored && chatModels.some((m) => m.id === stored)) return stored;
              if (res.active_model && chatModels.some((m) => m.id === res.active_model)) {
                return res.active_model;
              }
            }
            return chatModels.some((m) => m.id === current)
              ? current
              : res.active_model || chatModels[0].id;
          });
        }
      }
    } catch {
      /* keep MODEL_OPTIONS fallback */
    }
  }, []);

  const loadDocuments = useCallback(async () => {
    try {
      const docs = await apiClient.getDocuments();
      if (Array.isArray(docs)) setAvailableDocs(docs);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    loadModels();
    loadDocuments();
    const stored = localStorage.getItem('rag_response_mode');
    if (stored === 'compact' || stored === 'standard' || stored === 'detailed') {
      setResponseModeState(stored);
    }
  }, [loadModels, loadDocuments]);

  const setResponseMode = useCallback((mode: ResponseMode) => {
    setResponseModeState(mode);
    localStorage.setItem('rag_response_mode', mode);
  }, []);

  const selectModel = useCallback(
    async (modelId: string) => {
      if (pendingModel || modelId === selectedModel) return;
      const previous = selectedModel;
      setPendingModel(modelId);
      setModelSwitchError(null);
      setSelectedModel(modelId);
      try {
        const result = await apiClient.selectModel(modelId);
        setSelectedModel(result.active_model);
        localStorage.setItem('rag_model', result.active_model);
      } catch (error) {
        setSelectedModel(previous);
        setModelSwitchError(
          error instanceof Error ? error.message : `Unable to switch to ${modelId}`,
        );
      } finally {
        setPendingModel(null);
      }
    },
    [pendingModel, selectedModel],
  );

  const categoriesList = useMemo(() => {
    const set = new Set<string>(DEFAULT_CATEGORIES);
    availableDocs.forEach((d) => {
      if (d.category && d.category.trim()) set.add(d.category.trim());
    });
    return Array.from(set);
  }, [availableDocs]);

  const filteredDocs = useMemo(() => {
    if (!filterSearch.trim()) return availableDocs;
    const q = filterSearch.toLowerCase();
    return availableDocs.filter(
      (d) =>
        d.filename.toLowerCase().includes(q) ||
        (d.category && d.category.toLowerCase().includes(q)),
    );
  }, [availableDocs, filterSearch]);

  const filteredCategories = useMemo(() => {
    if (!filterSearch.trim()) return categoriesList;
    const q = filterSearch.toLowerCase();
    return categoriesList.filter((c) => c.toLowerCase().includes(q));
  }, [categoriesList, filterSearch]);

  const selectedDocument = useMemo(
    () => availableDocs.find((d) => d.id === selectedDocId),
    [availableDocs, selectedDocId],
  );

  const isFilterActive = selectedDocId !== 'All' || selectedCategory !== 'All';

  const clearFilters = useCallback(() => {
    setSelectedDocId('All');
    setSelectedCategory('All');
  }, []);

  /** Mirrors ChatWindow.handleSend() filter construction. */
  const buildFilters = useCallback((): FilterOptions | undefined => {
    const filters: FilterOptions = {};
    if (selectedDocument) {
      filters.source_file = selectedDocument.filename;
      filters.document_id = selectedDocument.id;
    }
    if (selectedCategory !== 'All') filters.category = selectedCategory;
    return Object.keys(filters).length > 0 ? filters : undefined;
  }, [selectedDocument, selectedCategory]);

  const selectedModelLabel =
    modelsList.find((m) => m.id === selectedModel)?.label || selectedModel;

  return {
    // models
    modelsList,
    selectedModel,
    selectedModelLabel,
    pendingModel,
    modelSwitchError,
    selectModel,
    loadModels,
    // depth
    responseMode,
    setResponseMode,
    // filters
    availableDocs,
    categoriesList,
    filteredDocs,
    filteredCategories,
    filterSearch,
    setFilterSearch,
    selectedDocId,
    setSelectedDocId,
    selectedCategory,
    setSelectedCategory,
    selectedDocument,
    isFilterActive,
    clearFilters,
    buildFilters,
    loadDocuments,
  };
}

export type ComposerControls = ReturnType<typeof useComposerControls>;
