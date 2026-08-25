import pytest
from unittest.mock import MagicMock

from backend.models.api_dto import ChatRequest
from backend.models.chunk import Chunk, ChunkMetadata
from backend.models.rag import (
    Citation,
    QueryCategory,
    QueryClassification,
    RAGResponse,
    RAGTrace,
    RetrievalStrategy,
    ScoredChunk,
    VerificationReport,
)
from backend.rag.scope_resolver import (
    DocumentRetrievalScope,
    DocumentScopeDecision,
    DocumentScopeResolver,
)
from backend.rag.pipeline import RAGPipeline
from backend.services.chat_service import ChatService


# ============================================================================
# 1. Document Scope Resolver Unit Tests
# ============================================================================

def test_scope_resolver_explicit_active_document():
    resolver = DocumentScopeResolver()
    decision = resolver.resolve_scope(
        query="What are the top projects in the doc?",
        active_document_id="doc_ai_projects",
        active_document_name="AI_Projects.pdf",
    )
    assert decision.scope == DocumentRetrievalScope.CURRENT_DOCUMENT
    assert decision.active_document_id == "doc_ai_projects"
    assert decision.active_document_name == "AI_Projects.pdf"
    assert decision.allowed_document_ids == ["doc_ai_projects"]
    assert decision.is_structural_query is True
    assert len(decision.structural_subqueries) > 0


def test_scope_resolver_anaphora_detection():
    resolver = DocumentScopeResolver()
    queries = [
        "What are the main points in this document?",
        "Summarize the uploaded file",
        "Explain key takeaways in this PDF",
        "What is discussed in the current document?",
        "What does this file contain?",
    ]
    for q in queries:
        decision = resolver.resolve_scope(
            query=q,
            active_document_id="doc_123",
            active_document_name="Report.pdf",
        )
        assert decision.scope == DocumentRetrievalScope.CURRENT_DOCUMENT
        assert decision.active_document_id == "doc_123"
        assert decision.allowed_document_ids == ["doc_123"]


def test_scope_resolver_page_and_section_extraction():
    resolver = DocumentScopeResolver()
    decision = resolver.resolve_scope(
        query="What is covered on page 59 of this PDF?",
        active_document_id="doc_handbook",
        active_document_name="Handbook.pdf",
    )
    assert decision.scope == DocumentRetrievalScope.CURRENT_DOCUMENT
    assert decision.page_number == 59

    decision_sec = resolver.resolve_scope(
        query="Explain Section 4.2 in the document",
        active_document_id="doc_handbook",
    )
    assert decision_sec.section_number == "4.2"


def test_scope_resolver_global_scope():
    resolver = DocumentScopeResolver()
    queries = [
        "Search all documents for maternity leave policy",
        "What are the travel reimbursement policies across the company?",
        "What information exists in the entire knowledge base about 401k?",
    ]
    for q in queries:
        decision = resolver.resolve_scope(
            query=q,
            active_document_id="doc_123",
        )
        assert decision.scope == DocumentRetrievalScope.GLOBAL
        assert decision.allowed_document_ids == []


def test_scope_resolver_multi_document_comparison():
    resolver = DocumentScopeResolver()
    decision = resolver.resolve_scope(
        query="Compare the vacation policy and travel policy across these documents",
        selected_document_ids=["doc_vacation", "doc_travel"],
    )
    assert decision.scope == DocumentRetrievalScope.SELECTED_DOCUMENTS
    assert decision.allowed_document_ids == ["doc_vacation", "doc_travel"]


def test_scope_resolver_filename_in_query():
    resolver = DocumentScopeResolver()
    known_docs = {
        "doc_ai": "AI_Guidebook.pdf",
        "doc_hr": "HR_Policy_2025.pdf",
    }
    decision = resolver.resolve_scope(
        query="What does AI_Guidebook.pdf say about neural networks?",
        known_documents=known_docs,
    )
    assert decision.scope == DocumentRetrievalScope.CURRENT_DOCUMENT
    assert decision.active_document_id == "doc_ai"
    assert decision.active_document_name == "AI_Guidebook.pdf"


# ============================================================================
# 2. Critical Bug Reproduction & Hard Cross-Document Isolation Test
# ============================================================================

def _make_chunk(
    chunk_id: str,
    text: str,
    document_id: str,
    source_file: str,
    page_number: int | None = None,
    section_title: str | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            document_id=document_id,
            source_file=source_file,
            file_path=f"/data/docs/{source_file}",
            file_hash=f"hash_{document_id}",
            document_type="pdf",
            chunk_strategy="recursive",
            page_number=page_number,
            section_title=section_title,
        ),
    )


def test_critical_bug_reproduction_no_cross_document_bleed():
    """
    Exact user scenario:
    - User uploaded doc containing AI/ML projects (AI_Projects_Showcase.pdf, id=doc_ai_101)
    - Knowledge base ALSO contains Legal_Studies_Book_v8_XI.pdf (id=doc_legal_999) with FIR info
    - User asks: "What are the top projects in the doc?"
    - System MUST retrieve AI project chunks and NEVER bleed into Legal Studies / FIR.
    """
    # 1. Setup mock chunks
    chunk_ai_1 = _make_chunk(
        chunk_id="chunk_ai_1",
        text="Project 1: Brand Monitoring System. Real-time brand perception analytics using LLMs and sentiment clustering.",
        document_id="doc_ai_101",
        source_file="AI_Projects_Showcase.pdf",
        page_number=1,
        section_title="Top Projects",
    )
    chunk_ai_2 = _make_chunk(
        chunk_id="chunk_ai_2",
        text="Project 2: AI Resume Reviewer. Automated candidate scoring and skill gap identification for hiring pipelines.",
        document_id="doc_ai_101",
        source_file="AI_Projects_Showcase.pdf",
        page_number=2,
        section_title="Top Projects",
    )
    chunk_legal_fir = _make_chunk(
        chunk_id="chunk_legal_fir",
        text="First Information Report (FIR). Under Section 154 of the Code of Criminal Procedure, an FIR is a written document prepared by police.",
        document_id="doc_legal_999",
        source_file="Legal_Studies_Book_v8_XI.pdf",
        page_number=154,
        section_title="Criminal Procedure",
    )

    docstore = {
        "chunk_ai_1": chunk_ai_1,
        "chunk_ai_2": chunk_ai_2,
        "chunk_legal_fir": chunk_legal_fir,
    }

    # 2. Setup mock retriever that returns a mixture of hits (including the erroneous FIR chunk)
    mock_retriever = MagicMock()
    def fake_retrieve(query, dense_top_k=5, bm25_top_k=5, filters=None, rrf_k=60):
        # Simulate vector DB returning both, but respecting document_id filter if passed
        if filters and "document_id" in filters:
            allowed_id = filters["document_id"]
            if allowed_id == "doc_ai_101":
                return [
                    ScoredChunk(chunk=chunk_ai_1, score=0.92),
                    ScoredChunk(chunk=chunk_ai_2, score=0.89),
                ]
            elif allowed_id == "doc_legal_999":
                return [ScoredChunk(chunk=chunk_legal_fir, score=0.50)]
        # If no filter, returns both (with FIR having high accidental score)
        return [
            ScoredChunk(chunk=chunk_legal_fir, score=0.75),
            ScoredChunk(chunk=chunk_ai_1, score=0.70),
            ScoredChunk(chunk=chunk_ai_2, score=0.68),
        ]

    mock_retriever.retrieve.side_effect = fake_retrieve

    # Mock LLM
    mock_llm = MagicMock()
    mock_llm.complete.return_value = (
        "The top projects in the document are:\n"
        "1. Brand Monitoring System - Real-time brand perception analytics using LLMs.\n"
        "2. AI Resume Reviewer - Automated candidate scoring and skill gap identification."
    )

    pipeline = RAGPipeline(
        hybrid_retriever=mock_retriever,
        docstore=docstore,
        llm=mock_llm,
    )

    # 3. Execute query with active_document_id="doc_ai_101"
    response = pipeline.query(
        user_query="What are the top projects in the doc?",
        active_document_id="doc_ai_101",
        active_document_name="AI_Projects_Showcase.pdf",
    )

    # 4. Assertions
    assert response is not None
    assert response.trace is not None
    assert response.trace.query_scope == "current_document"
    assert response.trace.active_document_id == "doc_ai_101"
    assert response.trace.allowed_document_ids == ["doc_ai_101"]

    # Verify context chunks contain 100% doc_ai_101 and 0% Legal Studies / FIR
    for sc in response.context_chunks:
        assert sc.chunk.metadata.document_id == "doc_ai_101"
        assert sc.chunk.metadata.source_file == "AI_Projects_Showcase.pdf"
        assert "FIR" not in sc.chunk.text
        assert "Legal_Studies" not in sc.chunk.metadata.source_file

    assert "First Information Report" not in response.answer
    assert "FIR" not in response.answer
    assert "Brand Monitoring System" in response.answer or "AI Resume Reviewer" in response.answer
    assert response.trace.final_context_documents == ["AI_Projects_Showcase.pdf"]


# ============================================================================
# 3. Hard Cross-Document Validation Layer Rejection Test
# ============================================================================

def test_hard_cross_document_validation_layer_rejects_unrelated_chunks():
    """
    Even if the underlying hybrid retriever fails to respect the filter
    and returns chunks from doc_unrelated, the validation layer MUST reject them.
    """
    chunk_target = _make_chunk(
        chunk_id="chunk_target",
        text="Target document information on AI pipelines.",
        document_id="doc_target",
        source_file="Target.pdf",
    )
    chunk_unrelated = _make_chunk(
        chunk_id="chunk_unrelated",
        text="Unrelated criminal law code chunk.",
        document_id="doc_unrelated",
        source_file="Law.pdf",
    )

    docstore = {"chunk_target": chunk_target, "chunk_unrelated": chunk_unrelated}

    # Retriever rogue behavior: ignores filters and returns both
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        ScoredChunk(chunk=chunk_unrelated, score=0.99),
        ScoredChunk(chunk=chunk_target, score=0.85),
    ]

    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Answer based on Target document."

    pipeline = RAGPipeline(
        hybrid_retriever=mock_retriever,
        docstore=docstore,
        llm=mock_llm,
    )

    response = pipeline.query(
        user_query="Summarize this document",
        active_document_id="doc_target",
        active_document_name="Target.pdf",
    )

    assert response.trace.cross_document_chunks_rejected > 0
    for sc in response.context_chunks:
        assert sc.chunk.metadata.document_id == "doc_target"
    assert response.trace.final_context_documents == ["Target.pdf"]


# ============================================================================
# 4. Same-Document Fallback Tests
# ============================================================================

def test_same_document_fallback_on_missing_page():
    """
    If user asks for a page that has no direct hits, fallback searches whole doc,
    NEVER searching other documents.
    """
    chunk_doc_p1 = _make_chunk(
        chunk_id="chunk_doc_p1",
        text="AI algorithms and data pipelines overview.",
        document_id="doc_ai",
        source_file="AI.pdf",
        page_number=1,
    )
    docstore = {"chunk_doc_p1": chunk_doc_p1}

    mock_retriever = MagicMock()
    def fake_retrieve(query, dense_top_k=5, bm25_top_k=5, filters=None, rrf_k=60):
        if filters and filters.get("page_number") == 99:
            return []  # Page 99 doesn't exist
        return [ScoredChunk(chunk=chunk_doc_p1, score=0.88)]

    mock_retriever.retrieve.side_effect = fake_retrieve
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Found general info in the active doc."

    pipeline = RAGPipeline(
        hybrid_retriever=mock_retriever,
        docstore=docstore,
        llm=mock_llm,
    )

    response = pipeline.query(
        user_query="What does page 99 say in this document?",
        active_document_id="doc_ai",
        active_document_name="AI.pdf",
    )

    assert len(response.context_chunks) > 0
    assert response.context_chunks[0].chunk.metadata.document_id == "doc_ai"


def test_same_document_fallback_clean_abstention_on_zero_hits():
    """
    If the active document has no matching content at all,
    the pipeline returns a clean grounded abstention without searching other docs.
    """
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    pipeline = RAGPipeline(
        hybrid_retriever=mock_retriever,
        docstore={},
        llm=None,
    )

    response = pipeline.query(
        user_query="What is the rocket propulsion formula in this document?",
        active_document_id="doc_ai",
        active_document_name="AI_Projects.pdf",
    )

    assert "could not find" in response.answer.lower()
    assert "AI_Projects.pdf" in response.answer or "active document" in response.answer
    assert response.context_chunks == []


# ============================================================================
# 5. ChatService Session Scope Continuity Test
# ============================================================================

def test_chat_service_session_document_continuity():
    """
    Verify ChatService retains active_document_id across multi-turn sessions.
    """
    mock_pipeline = MagicMock()
    fake_trace = RAGTrace(
        query="Summarize this PDF",
        rewritten_query="Summarize this PDF",
        sub_queries=[],
        query_type="factual",
        routing_confidence=0.9,
        retrieval_strategy="direct_chunk",
        query_scope="current_document",
        active_document_id="doc_session_1",
        active_document_name="Doc1.pdf",
        allowed_document_ids=["doc_session_1"],
        retrieved_candidate_count=1,
        post_rerank_count=1,
        final_context_count=1,
        execution_time_ms=10.0,
        stage_timings_ms={},
        fallback_reason="none",
        faithfulness_checked=True,
        faithfulness_passed=True,
        verification_report=None,
        verification_score=1.0,
        retry_count=0,
        retry_reasons=[],
        cache_hit=False,
        cache_similarity=None,
    )
    mock_pipeline.query.return_value = RAGResponse(
        id="resp_1",
        query="Summarize this PDF",
        answer="Summary of Doc1.",
        citations=[],
        context_chunks=[],
        trace=fake_trace,
        model="qwen2.5:7b",
        token_usage={},
    )
    mock_pipeline.get_active_model.return_value = "qwen2.5:7b"

    mock_telemetry = MagicMock()
    service = ChatService(rag_pipeline=mock_pipeline, telemetry_service=mock_telemetry)

    # Turn 1: Pass active_document_id
    req1 = ChatRequest(
        message="Summarize this PDF",
        session_id="sess_test_100",
        active_document_id="doc_session_1",
        active_document_name="Doc1.pdf",
    )
    res1 = service.execute_query(req1)
    assert res1.active_document_id == "doc_session_1"
    assert res1.document_scope == "current_document"

    # Turn 2: Do NOT pass active_document_id, must inherit from session
    req2 = ChatRequest(
        message="What are the top projects in the doc?",
        session_id="sess_test_100",
    )
    res2 = service.execute_query(req2)
    # Verify pipeline.query was called with active_document_id="doc_session_1"
    call_args = mock_pipeline.query.call_args[1]
    assert call_args["active_document_id"] == "doc_session_1"
    assert call_args["active_document_name"] == "Doc1.pdf"
