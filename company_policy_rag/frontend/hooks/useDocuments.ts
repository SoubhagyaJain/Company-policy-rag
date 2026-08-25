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

  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  const fetchDocuments = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const docs = await apiClient.getDocuments();
      setDocuments(Array.isArray(docs) ? docs : []);
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
        if (pollingRef.current) {
          clearInterval(pollingRef.current);
          pollingRef.current = null;
        }
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
        if (pollingRef.current) {
          clearInterval(pollingRef.current);
          pollingRef.current = null;
        }
        setUploading(false);
        setError(statusRes.error || `Ingestion failed at stage: ${statusRes.failed_stage || statusRes.current_stage}`);
        fetchDocuments();
      }
    } catch (err) {
      console.warn('Status poll warning:', err);
    }
  }, [fetchDocuments]);

  const uploadDocument = useCallback(
    async (file: File, category = 'General'): Promise<DocumentItem | null> => {
      // Validate file size (<= 100 MB)
      const MAX_SIZE = 100 * 1024 * 1024;
      if (file.size > MAX_SIZE) {
        setError('File size exceeds the 100 MB maximum limit.');
        return null;
      }

      setUploading(true);
      setUploadProgress(10);
      setCurrentStage('UPLOAD');
      setStageMessage(`Uploading ${file.name}...`);
      setError(null);

      // Progressive stage estimator while awaiting network response
      const stages = [
        { progress: 25, stage: 'TEXT_EXTRACTION', message: 'Extracting pages & raw text...' },
        { progress: 40, stage: 'SECTION_DETECTION', message: 'Detecting logical sections & metadata...' },
        { progress: 55, stage: 'CHUNKING', message: 'Applying adaptive document chunking...' },
        { progress: 75, stage: 'EMBEDDINGS', message: 'Generating dense vector embeddings...' },
        { progress: 90, stage: 'VECTOR_INDEX', message: 'Indexing in ChromaDB & BM25...' },
      ];

      let stageIdx = 0;
      const stageInterval = setInterval(() => {
        if (stageIdx < stages.length) {
          const s = stages[stageIdx];
          setUploadProgress(s.progress);
          setCurrentStage(s.stage);
          setStageMessage(s.message);
          stageIdx++;
        }
      }, 500);

      try {
        const uploadedDoc = await apiClient.uploadDocument(file, category);
        clearInterval(stageInterval);

        setUploadProgress(100);
        setCurrentStage('READY');
        setStageMessage('Knowledge base updated & READY.');

        setDocuments((prev) => [uploadedDoc, ...prev.filter((d) => d.id !== uploadedDoc.id)]);

        setTimeout(() => {
          setUploading(false);
          setUploadProgress(0);
          setCurrentStage('IDLE');
          setStageMessage('');
        }, 500);

        return uploadedDoc;
      } catch (err) {
        clearInterval(stageInterval);
        setUploading(false);
        setUploadProgress(0);
        setCurrentStage('FAILED');
        const errMsg = err instanceof Error ? err.message : 'Document upload failed';
        setError(errMsg);
        return null;
      }
    },
    []
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
      } catch (err) {
        setUploading(false);
        setUploadProgress(0);
        const errMsg = err instanceof Error ? err.message : 'Retry indexing failed';
        setError(errMsg);
        return false;
      }
    },
    [fetchDocuments]
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

  return {
    documents,
    loading,
    uploading,
    uploadProgress,
    currentStage,
    stageMessage,
    activeJob,
    error,
    refreshDocuments: fetchDocuments,
    uploadDocument,
    retryDocument,
    deleteDocument,
  };
}
