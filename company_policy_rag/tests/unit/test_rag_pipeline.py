from __future__ import annotations

import os
import shutil
import tempfile
import pytest

from backend.models.chunk import Chunk, ChunkMetadata, ChunkRole, ContentType
from backend.models.rag import Citation, RAGResponse, RAGTrace, ScoredChunk
from backend.embeddings.embeddings import EmbeddingCache, EmbeddingService, normalize_vector
from backend.embeddings.vector_store import ChromaVectorStore, MetadataFilter
from backend.retrieval.bm25 import BM25SearchIndex, tokenize
from backend.retrieval.vector import DenseVectorRetriever
from backend.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from backend.retrieval.reranker import CrossEncoderReranker, RelativeScoreThresholdPostprocessor
from backend.rag.multi_query import MultiQueryGenerator
from backend.rag.query_rewrite import QueryRewriter
from backend.rag.context_compression import ContextCompressor
from backend.rag.citations import CitationEngine
from backend.rag.pipeline import RAGPipeline


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    meta1 = ChunkMetadata(
        document_id="doc_001",
        source_file="employee_handbook.pdf",
        file_path="data/employee_handbook.pdf",
        file_hash="hash001",
        document_type="pdf",
        category="policy",
        chunk_index=0,
        page_number=3,
        section_title="Resignation & Termination Policy",
        section_path="I. GENERAL > A. Resignation",
        chunk_strategy="recursive",
        node_role=ChunkRole.STANDALONE,
    )
    c1 = Chunk(
        id="chunk_001",
        text="Employees giving notice of resignation must submit two weeks notice in writing to HR. Employment is at-will termination.",
        metadata=meta1,
        embedding=normalize_vector([0.1] * 384),
        token_count=20,
    )

    meta2 = ChunkMetadata(
        document_id="doc_001",
        source_file="employee_handbook.pdf",
        file_path="data/employee_handbook.pdf",
        file_hash="hash001",
        document_type="pdf",
        category="policy",
        chunk_index=1,
        page_number=5,
        section_title="Health Benefits & Insurance",
        section_path="II. BENEFITS > A. Health",
        chunk_strategy="recursive",
        node_role=ChunkRole.CHILD,
        parent_id="chunk_parent_001",
    )
    c2 = Chunk(
        id="chunk_002",
        text="Full-time employees are eligible for medical, dental, and vision health insurance benefits after 30 days of employment.",
        metadata=meta2,
        embedding=normalize_vector([0.8] * 384),
        token_count=18,
    )

    meta3 = ChunkMetadata(
        document_id="doc_002",
        source_file="ai_agents_guidebook.md",
        file_path="data/ai_agents_guidebook.md",
        file_hash="hash002",
        document_type="markdown",
        category="guidebook",
        chunk_index=0,
        page_number=1,
        section_title="Six Building Blocks of AI Agents",
        section_path="Chapter 1 > Building Blocks",
        chunk_strategy="markdown",
        node_role=ChunkRole.STANDALONE,
    )
    c3 = Chunk(
        id="chunk_003",
        text="The six building blocks of AI agents are Role-playing, Focus Tasks, Tools, Cooperation, Guardrails, and Planning and Memory.",
        metadata=meta3,
        embedding=normalize_vector([0.5] * 384),
        token_count=24,
    )

    return [c1, c2, c3]


def test_embedding_service_and_normalization():
    vec = [3.0, 4.0]
    norm = normalize_vector(vec)
    assert norm == [0.6, 0.8]

    service = EmbeddingService(cache_enabled=True, dimension=384)
    emb1 = service.embed_text("Employee resignation notice policy")
    assert len(emb1) == 384

    # Test cache hit
    emb2 = service.embed_text("Employee resignation notice policy")
    assert emb1 == emb2
    assert service.cache is not None and len(service.cache) == 1

    batch_embs = service.embed_chunks(["Text A", "Text B"])
    assert len(batch_embs) == 2
    assert len(batch_embs[0]) == 384


def test_vector_store(sample_chunks: list[Chunk]):
    temp_dir = tempfile.mkdtemp()
    try:
        vstore = ChromaVectorStore(collection_name="test_col", persist_dir=temp_dir)
        vstore.add_chunks(sample_chunks)

        assert vstore.count() == 3

        # Search
        query_emb = sample_chunks[0].embedding or normalize_vector([0.1] * 384)
        results = vstore.search(query_emb, top_k=2)
        assert len(results) >= 1
        assert results[0].chunk.id == "chunk_001"

        # Search with filter
        filtered_results = vstore.search(query_emb, top_k=2, filters={"category": "guidebook"})
        assert len(filtered_results) == 1
        assert filtered_results[0].chunk.id == "chunk_003"

        # Delete by source
        vstore.delete_by_source("employee_handbook.pdf")
        assert vstore.count() == 1

        vstore.clear()
        assert vstore.count() == 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_bm25_search_index(sample_chunks: list[Chunk]):
    temp_dir = tempfile.mkdtemp()
    try:
        bm25 = BM25SearchIndex(storage_dir=temp_dir)
        bm25.build_index(sample_chunks)
        assert len(bm25.entries) == 3

        # Search for resignation
        hits = bm25.search("resignation notice two weeks", top_k=2)
        assert len(hits) >= 1
        assert hits[0].chunk.id == "chunk_001"
        assert hits[0].sparse_score is not None

        # Search with filter
        hits_filtered = bm25.search("building blocks", top_k=2, filters={"category": "guidebook"})
        assert len(hits_filtered) == 1
        assert hits_filtered[0].chunk.id == "chunk_003"

        # Save and reload
        bm25.save()
        bm25_loaded = BM25SearchIndex(storage_dir=temp_dir)
        success = bm25_loaded.load()
        assert success is True
        assert len(bm25_loaded.entries) == 3

        # Remove by source
        bm25.remove_by_source_file("employee_handbook.pdf")
        assert len(bm25.entries) == 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_hybrid_retrieval_rrf(sample_chunks: list[Chunk]):
    # Setup dense & sparse hits
    c1, c2, c3 = sample_chunks[0], sample_chunks[1], sample_chunks[2]

    dense_hits = [
        ScoredChunk(chunk=c1, score=0.9, dense_score=0.9),
        ScoredChunk(chunk=c2, score=0.7, dense_score=0.7),
    ]

    sparse_hits = [
        ScoredChunk(chunk=c2, score=15.0, sparse_score=15.0),
        ScoredChunk(chunk=c3, score=8.0, sparse_score=8.0),
    ]

    fused = reciprocal_rank_fusion([dense_hits, sparse_hits], rrf_k=60)
    assert len(fused) == 3

    # Check RRF scores:
    # c2 is rank 2 in dense (1/62) and rank 1 in sparse (1/61) => score = 1/61 + 1/62 = ~0.0325
    # c1 is rank 1 in dense (1/61) => score = 1/61 = ~0.01639
    # c3 is rank 2 in sparse (1/62) => score = 1/62 = ~0.01612
    assert fused[0].chunk.id == "chunk_002"
    assert fused[0].rank == 1
    assert fused[0].dense_score == 0.7
    assert fused[0].sparse_score == 15.0


def test_reranker_thresholding(sample_chunks: list[Chunk]):
    postprocessor = RelativeScoreThresholdPostprocessor(min_ratio=0.45, min_keep=1)

    c1, c2, c3 = sample_chunks[0], sample_chunks[1], sample_chunks[2]
    candidates = [
        ScoredChunk(chunk=c1, score=10.0, rerank_score=10.0),
        ScoredChunk(chunk=c2, score=5.0, rerank_score=5.0),   # 5.0 >= 10.0 * 0.45 = 4.5 -> KEEP
        ScoredChunk(chunk=c3, score=3.0, rerank_score=3.0),   # 3.0 < 4.5 -> DROP
    ]

    filtered = postprocessor.filter(candidates)
    assert len(filtered) == 2
    assert set(c.chunk.id for c in filtered) == {"chunk_001", "chunk_002"}

    # Test fallback min_keep when all are low
    low_candidates = [
        ScoredChunk(chunk=c3, score=-2.0, rerank_score=-2.0),
    ]
    filtered_low = postprocessor.filter(low_candidates)
    assert len(filtered_low) == 1

    # Test CrossEncoderReranker fallback behavior
    reranker = CrossEncoderReranker(top_n=2, min_ratio=0.45)
    reranked = reranker.rerank("query text", candidates)
    assert len(reranked) <= 2


def test_multi_query_decomposition():
    gen = MultiQueryGenerator()

    queries = gen.generate_subqueries("What are the 6 building blocks of AI agents?")
    assert len(queries) > 1
    assert "Role-playing building block AI agents" in queries or any("building blocks" in q for q in queries)

    guardrail_queries = gen.generate_subqueries("How do guardrails work?")
    assert any("Guardrails" in q for q in guardrail_queries)


def test_query_rewriter():
    rewriter = QueryRewriter(enable_llm_rewrite=False)

    res = rewriter.rewrite("What happens if I quit without notice?")
    assert res.is_comprehensive_list is False
    assert res.inferred_corpus == "policy"
    assert "employment at-will" in res.rewritten_query

    res_comp = rewriter.rewrite("List all six building blocks of AI agents and explain each")
    assert res_comp.is_comprehensive_list is True
    assert res_comp.inferred_corpus == "guidebook"


def test_query_rewriter_with_history():
    # Test fallback pronoun resolution without LLM
    rewriter = QueryRewriter(enable_llm_rewrite=False)
    history = [
        {"role": "user", "content": "What is the employee annual leave policy?"},
        {"role": "assistant", "content": "Employees get 15 days of paid annual leave per year."},
    ]
    res = rewriter.rewrite("Does it apply to part-time employees?", history=history)
    assert "What is the employee annual leave policy?" in res.rewritten_query

    # Test fallback without explicit pronouns in multi-turn follow-up
    res_no_pronoun = rewriter.rewrite("Who is eligible?", history=history)
    assert "What is the employee annual leave policy?" in res_no_pronoun.rewritten_query

    # Test LLM query rewriting with history
    class DummyLLM:
        def complete(self, prompt: str) -> str:
            assert "Conversation History:" in prompt
            assert "Follow-up Question: Does it apply to part-time employees?" in prompt
            return "annual leave policy part-time employee eligibility"

    llm_rewriter = QueryRewriter(enable_llm_rewrite=True, llm=DummyLLM())
    res_llm = llm_rewriter.rewrite("Does it apply to part-time employees?", history=history)
    assert res_llm.rewritten_query == "annual leave policy part-time employee eligibility"


def test_pipeline_query_rewriter_llm_connection(sample_chunks: list[Chunk]):
    # LLM rewriting is opt-in so auxiliary generation cannot silently dominate
    # request latency.
    class DummyLLM:
        def __init__(self):
            self.model = "qwen2.5:7b"

        def complete(self, prompt: str) -> str:
            return "rewritten search query"

    dummy_llm = DummyLLM()
    mock_retriever = HybridRetriever(
        dense_retriever=DenseVectorRetriever(vector_store=None, embedding_service=EmbeddingService()),
        bm25_index=BM25SearchIndex(),
    )
    default_pipeline = RAGPipeline(hybrid_retriever=mock_retriever, llm=dummy_llm)
    assert default_pipeline.query_rewriter.enable_llm_rewrite is False
    assert default_pipeline.query_rewriter.llm is None
    assert default_pipeline.conversation_resolver.llm is None

    pipeline = RAGPipeline(
        hybrid_retriever=mock_retriever,
        llm=dummy_llm,
        query_rewriter=QueryRewriter(enable_llm_rewrite=True),
    )
    assert pipeline.query_rewriter.llm is dummy_llm


def test_history_slicing_twelve_messages():
    # Verify Fix 2: history slicing preserves 12 messages for 6 turns
    from backend.rag.pipeline import _format_history_for_prompt

    history = []
    for i in range(1, 10):  # 9 turns = 18 messages
        history.append({"role": "user", "content": f"User msg {i}"})
        history.append({"role": "assistant", "content": f"Assistant msg {i}"})

    formatted = _format_history_for_prompt(history, max_turns=6)
    # Turn 4 to Turn 9 should be present (6 turns * 2 = 12 messages)
    assert "User msg 4" in formatted
    assert "User msg 9" in formatted
    assert "User msg 3" not in formatted


def test_pipeline_with_history_and_model(sample_chunks: list[Chunk]):
    temp_dir = tempfile.mkdtemp()
    try:
        vstore = ChromaVectorStore(collection_name="rag_mem_test", persist_dir=temp_dir)
        vstore.add_chunks(sample_chunks)
        embed_service = EmbeddingService(cache_enabled=True)
        dense_retriever = DenseVectorRetriever(vector_store=vstore, embedding_service=embed_service)
        bm25 = BM25SearchIndex(storage_dir=os.path.join(temp_dir, "bm25"))
        bm25.build_index(sample_chunks)
        hybrid_retriever = HybridRetriever(dense_retriever=dense_retriever, bm25_index=bm25)

        class DummyLLM:
            def __init__(self):
                self.model = "qwen2.5:7b"
                self.last_prompt = ""

            def complete(self, prompt: str) -> str:
                self.last_prompt = prompt
                return "Part-time employees are not eligible for annual leave [Source 1]."

        dummy_llm = DummyLLM()
        pipeline = RAGPipeline(hybrid_retriever=hybrid_retriever, llm=dummy_llm)

        history = [
            {"role": "user", "content": "What is the employee annual leave policy?"},
            {"role": "assistant", "content": "Employees get 15 days of annual leave."},
        ]

        response = pipeline.query(
            user_query="Does it apply to part-time employees?",
            history=history,
            model="llama3.1:8b",
        )

        assert response.model == "llama3.1:8b"
        # Verify Fix 4: shared singleton dummy_llm.model is NOT mutated
        assert dummy_llm.model == "qwen2.5:7b"
        assert "Recent Conversation History:" in dummy_llm.last_prompt
        assert "User: What is the employee annual leave policy?" in dummy_llm.last_prompt
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)




def test_context_compression_and_parent_expansion(sample_chunks: list[Chunk]):
    compressor = ContextCompressor(enable_parent_expansion=True, max_token_budget=4096)

    # Parent chunk fixture
    parent_chunk = Chunk(
        id="chunk_parent_001",
        text="Full Parent Section: Health Benefits, Insurance, Medical, Dental, and Vision Coverage Details...",
        metadata=ChunkMetadata(
            document_id="doc_001",
            source_file="employee_handbook.pdf",
            file_path="data/employee_handbook.pdf",
            file_hash="hash001",
            document_type="pdf",
            category="policy",
            chunk_index=99,
            node_role=ChunkRole.PARENT,
            chunk_strategy="recursive",
        ),
    )

    docstore = {"chunk_parent_001": parent_chunk}
    child_scored = ScoredChunk(chunk=sample_chunks[1], score=0.85)

    expanded = compressor.expand_to_parents([child_scored], docstore)
    assert len(expanded) == 1
    assert expanded[0].chunk.id == "chunk_parent_001"
    assert "Full Parent Section" in expanded[0].chunk.text

    # Prompt formatting
    formatted = compressor.format_context_for_prompt([child_scored])
    assert "[Source 1] File: employee_handbook.pdf" in formatted


def test_citation_engine(sample_chunks: list[Chunk]):
    engine = CitationEngine()

    answer_with_tags = (
        "Employees must give two weeks notice when resigning [Source 1]. "
        "Health insurance coverage begins after 30 days [Source 2]."
    )

    scored = [
        ScoredChunk(chunk=sample_chunks[0], score=0.9, rank=1),
        ScoredChunk(chunk=sample_chunks[1], score=0.8, rank=2),
    ]

    citations = engine.select_citations(answer_with_tags, scored)
    assert len(citations) == 2
    assert citations[0].source_index == 1
    assert citations[0].source_file == "employee_handbook.pdf"
    assert citations[0].selection_reason == "cited_in_answer"
    assert citations[1].source_index == 2

    # Test fallback when answer has no tags
    answer_no_tags = "Health insurance requires 30 days waiting period."
    citations_fallback = engine.select_citations(answer_no_tags, scored)
    assert len(citations_fallback) >= 1
    assert citations_fallback[0].selection_reason == "score_threshold_fallback"


def test_rag_pipeline_execution(sample_chunks: list[Chunk]):
    temp_dir = tempfile.mkdtemp()
    try:
        # Setup vector store & BM25
        vstore = ChromaVectorStore(collection_name="rag_test", persist_dir=temp_dir)
        vstore.add_chunks(sample_chunks)

        embed_service = EmbeddingService(cache_enabled=True)
        dense_retriever = DenseVectorRetriever(vector_store=vstore, embedding_service=embed_service)

        bm25 = BM25SearchIndex(storage_dir=os.path.join(temp_dir, "bm25"))
        bm25.build_index(sample_chunks)

        hybrid_retriever = HybridRetriever(dense_retriever=dense_retriever, bm25_index=bm25)
        pipeline = RAGPipeline(hybrid_retriever=hybrid_retriever)

        # Run pipeline query
        response = pipeline.query("What is the resignation notice policy?")

        assert isinstance(response, RAGResponse)
        assert response.query == "What is the resignation notice policy?"
        assert len(response.answer) > 0
        assert len(response.citations) >= 1
        assert isinstance(response.trace, RAGTrace)
        assert response.trace.retrieved_candidate_count > 0
        assert "hybrid_retrieval" in response.trace.stage_timings_ms
        assert "reranking" in response.trace.stage_timings_ms
        assert response.trace.execution_time_ms > 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
