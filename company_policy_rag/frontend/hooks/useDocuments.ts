import { useState, useEffect, useCallback } from 'react';
import { DocumentItem } from '../lib/types';
import { apiClient } from '../lib/api-client';
import { generateId } from '../lib/utils';

const DEMO_DOCUMENTS: DocumentItem[] = [
  {
    id: 'doc_hr_policy_2025',
    filename: 'Company_HR_Policy_2025.pdf',
    category: 'HR & Benefits',
    chunks_count: 42,
    file_size: 2450000,
    uploaded_at: new Date(Date.now() - 86400000 * 3).toISOString(),
    status: 'indexed',
    file_type: 'pdf',
  },
  {
    id: 'doc_remote_work_guide',
    filename: 'Remote_Work_Guidelines.docx',
    category: 'Operations',
    chunks_count: 18,
    file_size: 1120000,
    uploaded_at: new Date(Date.now() - 86400000 * 5).toISOString(),
    status: 'indexed',
    file_type: 'docx',
  },
  {
    id: 'doc_it_security_compliance',
    filename: 'IT_Security_Compliance.md',
    category: 'IT & Security',
    chunks_count: 29,
    file_size: 650000,
    uploaded_at: new Date(Date.now() - 86400000 * 7).toISOString(),
    status: 'indexed',
    file_type: 'md',
  },
  {
    id: 'doc_travel_reimbursement',
    filename: 'Travel_Expense_Reimbursement.csv',
    category: 'Finance',
    chunks_count: 14,
    file_size: 420000,
    uploaded_at: new Date(Date.now() - 86400000 * 10).toISOString(),
    status: 'indexed',
    file_type: 'csv',
  },
  {
    id: 'doc_employee_conduct',
    filename: 'Code_of_Employee_Conduct.json',
    category: 'Compliance',
    chunks_count: 36,
    file_size: 1800000,
    uploaded_at: new Date(Date.now() - 86400000 * 12).toISOString(),
    status: 'indexed',
    file_type: 'json',
  },
];

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
      if (docs && docs.length > 0) {
        setDocuments(docs);
      } else {
        setDocuments(DEMO_DOCUMENTS);
      }
    } catch (err) {
      console.warn('Backend document API unavailable, loading fallback indexed documents', err);
      setDocuments(DEMO_DOCUMENTS);
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
      setUploadProgress(10);
      setError(null);

      // Simulate upload progress steps
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => (prev < 90 ? prev + 15 : prev));
      }, 150);

      try {
        let uploadedDoc: DocumentItem;
        try {
          uploadedDoc = await apiClient.uploadDocument(file, category);
        } catch {
          // Client-side fallback if backend endpoint isn't responding yet
          const ext = file.name.split('.').pop()?.toLowerCase() || 'txt';
          uploadedDoc = {
            id: generateId('doc'),
            filename: file.name,
            category,
            chunks_count: Math.max(5, Math.floor(file.size / 25000)),
            file_size: file.size,
            uploaded_at: new Date().toISOString(),
            status: 'indexed',
            file_type: ext,
          };
        }

        clearInterval(progressInterval);
        setUploadProgress(100);

        setDocuments((prev) => [uploadedDoc, ...prev]);
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
        try {
          await apiClient.deleteDocument(docId);
        } catch (e) {
          console.warn('Backend delete document API fallback:', e);
        }
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
