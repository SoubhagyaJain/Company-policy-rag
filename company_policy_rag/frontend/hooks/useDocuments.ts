import { useState, useEffect, useCallback } from 'react';
import { DocumentItem } from '../lib/types';
import { apiClient } from '../lib/api-client';

export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);

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
      setUploadProgress(15);
      setError(null);

      // Smooth progress indicator
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => (prev < 90 ? prev + 15 : prev));
      }, 200);

      try {
        const uploadedDoc = await apiClient.uploadDocument(file, category);
        clearInterval(progressInterval);
        setUploadProgress(100);

        setDocuments((prev) => [uploadedDoc, ...prev.filter((d) => d.id !== uploadedDoc.id)]);
        setTimeout(() => {
          setUploading(false);
          setUploadProgress(0);
        }, 400);

        return uploadedDoc;
      } catch (err) {
        clearInterval(progressInterval);
        setUploading(false);
        setUploadProgress(0);
        const errMsg = err instanceof Error ? err.message : 'Document upload failed';
        setError(errMsg);
        return null;
      }
    },
    []
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
    error,
    refreshDocuments: fetchDocuments,
    uploadDocument,
    deleteDocument,
  };
}
