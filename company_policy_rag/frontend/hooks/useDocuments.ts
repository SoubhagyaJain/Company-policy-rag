import { useState, useEffect, useCallback, useRef } from 'react';
import { DocumentItem, IngestionStatusResponse } from '../lib/types';
import { apiClient } from '../lib/api-client';

export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [currentStage, setCurrentStage] = useState<string>('IDLE');
  const [stageMessage, setStageMessage] = useState<string>('');
  const [activeJob, setActiveJob] = useState<IngestionStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [duplicateCount, setDuplicateCount] = useState<number>(0);
  const [deduplicating, setDeduplicating] = useState<boolean>(false);

  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  const fetchDocuments = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [docs, duplicateSummary] = await Promise.all([
        apiClient.getDocuments(),
        apiClient.deduplicateDocuments(true),
      ]);
      setDocuments(Array.isArray(docs) ? docs : []);
      setDuplicateCount(duplicateSummary.duplicates_found);
    } catch (err) {
      console.warn('Failed to fetch documents from backend API', err);
      setDocuments([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, [fetchDocuments]);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  const pollDocumentStatus = useCallback(async (docId: string) => {
    try {
      const statusRes = await apiClient.getDocumentStatus(docId);
      setActiveJob(statusRes);
      setUploadProgress(statusRes.progress);
      setCurrentStage(statusRes.current_stage);
      setStageMessage(
        statusRes.stages?.slice(-1)[0]?.message ||
        `Status: ${statusRes.status} (${statusRes.progress}%)`
      );

      if (statusRes.status === 'READY' || statusRes.status === 'READY_WITH_VISION' || statusRes.status === 'indexed') {
        stopPolling();
        setUploadProgress(100);
        setCurrentStage('READY');
        setStageMessage('Document indexed and ready for RAG.');
        setTimeout(() => {
          setUploading(false);
          setUploadProgress(0);
          setCurrentStage('IDLE');
          setStageMessage('');
          fetchDocuments();
        }, 600);
      } else if (statusRes.status === 'FAILED') {
        stopPolling();
        setUploading(false);
        setError(statusRes.error || `Ingestion failed at stage: ${statusRes.failed_stage || statusRes.current_stage}`);
        fetchDocuments();
      }
    } catch (err) {
      console.warn('Status poll warning:', err);
    }
  }, [fetchDocuments, stopPolling]);

  // Poll the backend for live ingestion progress until READY/FAILED. Large
  // files ingest asynchronously on the server, so this is how the UI tracks
  // real per-stage progress (parse → chunk → embed → index) rather than guessing.
  const startPolling = useCallback((docId: string) => {
    stopPolling();
    let ticks = 0;
    const MAX_TICKS = 1200; // ~20 min ceiling at 1s cadence
    pollDocumentStatus(docId);
    pollingRef.current = setInterval(() => {
      ticks += 1;
      if (ticks > MAX_TICKS) {
        stopPolling();
        setUploading(false);
        setError('Ingestion is taking unusually long. It may still finish — refresh to check.');
        return;
      }
      pollDocumentStatus(docId);
    }, 1000);
  }, [pollDocumentStatus, stopPolling]);

  const uploadDocument = useCallback(
    async (file: File, category = 'General'): Promise<DocumentItem | null> => {
      // Validate file size (<= 100 MB)
      const MAX_SIZE = 100 * 1024 * 1024;
      if (file.size > MAX_SIZE) {
        setError('File size exceeds the 100 MB maximum limit.');
        return null;
      }

      setUploading(true);
      setUploadProgress(5);
      setCurrentStage('UPLOAD');
      setStageMessage(`Uploading ${file.name}...`);
      setError(null);

      try {
        // Server stores the file and returns immediately; heavy ingestion runs
        // asynchronously. We then poll for real per-stage progress.
        const uploadedDoc = await apiClient.uploadDocument(file, category);

        // Surface the document right away (as processing) so it appears in the list.
        setDocuments((prev) => [uploadedDoc, ...prev.filter((d) => d.id !== uploadedDoc.id)]);

        const terminal = ['READY', 'READY_WITH_VISION', 'indexed', 'FAILED'];
        if (terminal.includes(uploadedDoc.status)) {
          // Fast path (tiny files that finished before the response, or errors).
          if (uploadedDoc.status === 'FAILED') {
            setUploading(false);
            setCurrentStage('FAILED');
            setError(uploadedDoc.error || 'Ingestion failed.');
          } else {
            setUploadProgress(100);
            setCurrentStage('READY');
            setStageMessage('Knowledge base updated & READY.');
            setTimeout(() => {
              setUploading(false);
              setUploadProgress(0);
              setCurrentStage('IDLE');
              setStageMessage('');
              fetchDocuments();
            }, 500);
          }
          return uploadedDoc;
        }

        setCurrentStage(uploadedDoc.current_stage || 'QUEUED');
        setStageMessage('Queued for indexing…');
        startPolling(uploadedDoc.id);
        return uploadedDoc;
      } catch (err) {
        setUploading(false);
        setUploadProgress(0);
        setCurrentStage('FAILED');
        const errMsg = err instanceof Error ? err.message : 'Document upload failed';
        setError(errMsg);
        return null;
      }
    },
    [fetchDocuments, startPolling]
  );

  const retryDocument = useCallback(
    async (docId: string): Promise<boolean> => {
      setUploading(true);
      setUploadProgress(10);
      setCurrentStage('RETRYING');
      setStageMessage('Retrying document indexing from saved storage...');
      setError(null);

      try {
        const statusRes = await apiClient.retryDocument(docId);
        setActiveJob(statusRes);

        const terminal = ['READY', 'READY_WITH_VISION', 'indexed', 'FAILED'];
        if (terminal.includes(statusRes.status)) {
          if (statusRes.status === 'FAILED') {
            setUploading(false);
            setError(statusRes.error || 'Retry indexing failed');
            return false;
          }
          setUploadProgress(100);
          setCurrentStage('READY');
          setStageMessage('Document successfully re-indexed and READY.');
          await fetchDocuments();
          setTimeout(() => {
            setUploading(false);
            setUploadProgress(0);
            setCurrentStage('IDLE');
          }, 500);
          return true;
        }

        // Re-indexing runs asynchronously; track live progress.
        setCurrentStage(statusRes.current_stage || 'QUEUED');
        startPolling(docId);
        return true;
      } catch (err) {
        setUploading(false);
        setUploadProgress(0);
        const errMsg = err instanceof Error ? err.message : 'Retry indexing failed';
        setError(errMsg);
        return false;
      }
    },
    [fetchDocuments, startPolling]
  );

  const deleteDocument = useCallback(
    async (docId: string): Promise<boolean> => {
      try {
        await apiClient.deleteDocument(docId);
        setDocuments((prev) => prev.filter((d) => d.id !== docId));
        return true;
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : 'Failed to delete document';
        setError(errMsg);
        return false;
      }
    },
    []
  );

  const removeDuplicates = useCallback(async (): Promise<number> => {
    setDeduplicating(true);
    setError(null);
    try {
      const result = await apiClient.deduplicateDocuments(false);
      setDuplicateCount(0);
      await fetchDocuments();
      return result.duplicates_removed;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to remove duplicate documents');
      return 0;
    } finally {
      setDeduplicating(false);
    }
  }, [fetchDocuments]);

  return {
    documents,
    loading,
    uploading,
    uploadProgress,
    currentStage,
    stageMessage,
    activeJob,
    error,
    duplicateCount,
    deduplicating,
    refreshDocuments: fetchDocuments,
    uploadDocument,
    retryDocument,
    deleteDocument,
    removeDuplicates,
  };
}
