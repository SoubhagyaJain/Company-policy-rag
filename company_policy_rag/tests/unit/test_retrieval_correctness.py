"""
Unit test suite for hybrid search retrieval correctness in company_policy_rag.
Tests:
- BM25 + Vector RRF ranking accuracy and score normalization.
- Metadata filtering correctness (filtering by document category, source file, page number).
- Reranker thresholding edge cases (zero matching chunks, low similarity chunks, negative scores, fallbacks).
"""

from __future__ import annotations

import shutil
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from backend.embeddings.embeddings import EmbeddingService, normalize_vector
from backend.embeddings.vector_store import ChromaVectorStore
from backend.models.chunk import Chunk, ChunkMetadata, ChunkRole, ContentType
from backend.models.rag import ScoredChunk
from backend.retrieval.bm25 import BM25SearchIndex
from backend.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from backend.retrieval.reranker import CrossEncoderReranker, RelativeScoreThresholdPostprocessor
from backend.retrieval.vector import DenseVectorRetriever


@pytest.fixture
def test_chunks() -> list[Chunk]:
    """Provide a structured set of chunks across different categories, source files, and page numbers."""
    c1 = Chunk(
        id="chunk_policy_p1_1",
        text="The annual PTO policy grants 20 days of paid time off per calendar year to full-time employees.",
        metadata=ChunkMetadata(
            document_id="doc_pto_01",
            source_file="pto_policy.pdf",
            file_path="data/pto_policy.pdf",
            file_hash="hash_pto_01",
            document_type="pdf",
            category="policy",
            chunk_index=0,
            page_number=1,
            section_title="Paid Time Off",
            chunk_strategy="recursive",
            node_role=ChunkRole.STANDALONE,
        ),
        embedding=normalize_vector([0.9 if i % 2 == 0 else 0.1 for i in range(384)]),
        token_count=18,
    )

    c2 = Chunk(
        id="chunk_policy_p1_2",
        text="Sick leave policy permits up to 10 days of paid sick leave annually with medical certification.",
        metadata=ChunkMetadata(
            document_id="doc_pto_01",
            source_file="pto_policy.pdf",
            file_path="data/pto_policy.pdf",
            file_hash="hash_pto_01",
            document_type="pdf",
            category="policy",
            chunk_index=1,
            page_number=2,
            section_title="Sick Leave",
            chunk_strategy="recursive",
            node_role=ChunkRole.STANDALONE,
        ),
        embedding=normalize_vector([0.8 if i % 2 == 0 else 0.2 for i in range(384)]),
        token_count=16,
    )

    c3 = Chunk(
        id="chunk_handbook_p5",
        text="Remote work guidelines require core hours availability from 10 AM to 4 PM Eastern Time.",
        metadata=ChunkMetadata(
            document_id="doc_handbook_01",
            source_file="employee_handbook.docx",
            file_path="data/employee_handbook.docx",
            file_hash="hash_handbook_01",
            document_type="docx",
            category="handbook",
            chunk_index=0,
            page_number=5,
            section_title="Remote Work",
            chunk_strategy="recursive",
            node_role=ChunkRole.STANDALONE,
        ),
        embedding=normalize_vector([0.3 if i % 2 == 0 else 0.7 for i in range(384)]),
        token_count=15,
    )

    c4 = Chunk(
        id="chunk_tech_p1",
        text="Information security policy mandates password rotation every 90 days and mandatory multi-factor authentication.",
        metadata=ChunkMetadata(
            document_id="doc_sec_01",
            source_file="security_policy.md",
            file_path="data/security_policy.md",
            file_hash="hash_sec_01",
            document_type="markdown",
            category="security",
            chunk_index=0,
            page_number=1,
            section_title="Authentication",
            chunk_strategy="markdown",
            node_role=ChunkRole.STANDALONE,
        ),
        embedding=normalize_vector([0.1 if i % 2 == 0 else 0.9 for i in range(384)]),
        token_count=17,
    )

    return [c1, c2, c3, c4]


# ============================================================================
# 1. BM25 + Vector RRF Ranking Accuracy & Score Normalization
# ============================================================================


def test_rrf_ranking_accuracy_and_fusion_math(test_chunks: list[Chunk]):
    c1, c2, c3, c4 = test_chunks

    dense_list = [
        ScoredChunk(chunk=c1, score=0.95, dense_score=0.95),
        ScoredChunk(chunk=c2, score=0.85, dense_score=0.85),
        ScoredChunk(chunk=c3, score=0.75, dense_score=0.75),
    ]

    sparse_list = [
        ScoredChunk(chunk=c2, score=12.5, sparse_score=12.5),
        ScoredChunk(chunk=c1, score=10.0, sparse_score=10.0),
        ScoredChunk(chunk=c4, score=5.0, sparse_score=5.0),
    ]

    rrf_k = 60
    fused = reciprocal_rank_fusion([dense_list, sparse_list], rrf_k=rrf_k)

    assert len(fused) == 4

    # Calculate expected RRF scores:
    # c1: dense rank 1 (1/(60+1) = 1/61), sparse rank 2 (1/(60+2) = 1/62) => 1/61 + 1/62 = 0.0163934 + 0.0161290 = 0.0325224
    # c2: dense rank 2 (1/(60+2) = 1/62), sparse rank 1 (1/(60+1) = 1/61) => 1/61 + 1/62 = 0.0325224
    # c3: dense rank 3 (1/(60+3) = 1/63) => 1/63 = 0.0158730
    # c4: sparse rank 3 (1/(60+3) = 1/63) => 1/63 = 0.0158730
    expected_c1_rrf = (1.0 / 61.0) + (1.0 / 62.0)
    expected_c2_rrf = (1.0 / 61.0) + (1.0 / 62.0)

    # c1 and c2 tie on RRF score
    top_ids = {fused[0].chunk.id, fused[1].chunk.id}
    assert top_ids == {"chunk_policy_p1_1", "chunk_policy_p1_2"}
    assert pytest.approx(fused[0].score, abs=1e-6) == expected_c1_rrf
    assert pytest.approx(fused[1].score, abs=1e-6) == expected_c2_rrf

    # Verify score attributes preservation
    c1_fused = next(sc for sc in fused if sc.chunk.id == "chunk_policy_p1_1")
    assert c1_fused.dense_score == 0.95
    assert c1_fused.sparse_score == 10.0

    c2_fused = next(sc for sc in fused if sc.chunk.id == "chunk_policy_p1_2")
    assert c2_fused.dense_score == 0.85
    assert c2_fused.sparse_score == 12.5

    # Verify ranks are sequential (1..4)
    ranks = [sc.rank for sc in fused]
    assert ranks == [1, 2, 3, 4]


def test_rrf_score_normalization_and_custom_k(test_chunks: list[Chunk]):
    c1, c2 = test_chunks[0], test_chunks[1]

    dense_list = [ScoredChunk(chunk=c1, score=0.9, dense_score=0.9)]
    sparse_list = [ScoredChunk(chunk=c2, score=8.0, sparse_score=8.0)]

    # Custom rrf_k = 10
    fused_k10 = reciprocal_rank_fusion([dense_list, sparse_list], rrf_k=10)
    assert len(fused_k10) == 2
    # c1: 1 / (10 + 1) = 1/11
    # c2: 1 / (10 + 1) = 1/11
    assert pytest.approx(fused_k10[0].score, abs=1e-6) == 1.0 / 11.0

    # Empty inputs
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_hybrid_retriever_empty_query_and_fallbacks(test_chunks: list[Chunk]):
    temp_dir = tempfile.mkdtemp()
    try:
        vstore = ChromaVectorStore(collection_name="test_fallback", persist_dir=temp_dir)
        vstore.add_chunks(test_chunks)
        emb_svc = EmbeddingService(cache_enabled=False)
        dense_retriever = DenseVectorRetriever(vector_store=vstore, embedding_service=emb_svc)

        bm25 = BM25SearchIndex(storage_dir=f"{temp_dir}/bm25")
        bm25.build_index(test_chunks)

        hybrid = HybridRetriever(dense_retriever=dense_retriever, bm25_index=bm25, rrf_k=60)

        # 1. Blank query returns empty list
        assert hybrid.retrieve("") == []
        assert hybrid.retrieve("   ") == []

        # 2. Dense hits present, BM25 returns 0 hits (query with terms not in BM25 index)
        # Mock bm25.search to return []
        with patch.object(bm25, "search", return_value=[]):
            res_dense_only = hybrid.retrieve("PTO paid time off policy", dense_top_k=2)
            assert len(res_dense_only) == 2
            assert all(sc.rank is not None for sc in res_dense_only)

        # 3. BM25 hits present, Dense returns 0 hits
        with patch.object(dense_retriever, "retrieve", return_value=[]):
            res_bm25_only = hybrid.retrieve("PTO paid time off policy", bm25_top_k=2)
            assert len(res_bm25_only) == 2
            assert all(sc.rank is not None for sc in res_bm25_only)

        # 4. Both return 0 hits
        with patch.object(dense_retriever, "retrieve", return_value=[]), patch.object(bm25, "search", return_value=[]):
            res_none = hybrid.retrieve("Unmatched query")
            assert res_none == []
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# 2. Metadata Filtering Correctness
# ============================================================================


def test_bm25_metadata_filtering_category_source_page(test_chunks: list[Chunk]):
    temp_dir = tempfile.mkdtemp()
    try:
        bm25 = BM25SearchIndex(storage_dir=temp_dir)
        bm25.build_index(test_chunks)

        # Filter by category
        hits_policy = bm25.search("policy leave off work", filters={"category": "policy"})
        assert len(hits_policy) == 2
        assert all(h.chunk.metadata.category == "policy" for h in hits_policy)

        hits_category_list = bm25.search("policy leave off work", filters={"category": ["handbook", "security"]})
        assert len(hits_category_list) == 2
        assert set(h.chunk.metadata.category for h in hits_category_list) == {"handbook", "security"}

        # Filter by source_file
        hits_pto_file = bm25.search("policy time off leave", filters={"source_file": "pto_policy.pdf"})
        assert len(hits_pto_file) == 2
        assert all(h.chunk.metadata.source_file == "pto_policy.pdf" for h in hits_pto_file)

        # Filter by page_number (int)
        hits_page_1 = bm25.search("policy paid time off", filters={"page_number": 1})
        assert len(hits_page_1) == 2  # pto_policy.pdf p.1 and security_policy.md p.1
        assert all(h.chunk.metadata.page_number == 1 for h in hits_page_1)

        hits_page_5 = bm25.search("remote work hours", filters={"page_number": 5})
        assert len(hits_page_5) == 1
        assert hits_page_5[0].chunk.id == "chunk_handbook_p5"

        # Combined filters: category + source_file + page_number
        hits_multi = bm25.search(
            "policy time off",
            filters={"category": "policy", "source_file": "pto_policy.pdf", "page_number": 1},
        )
        assert len(hits_multi) == 1
        assert hits_multi[0].chunk.id == "chunk_policy_p1_1"

        # Non-matching filter returns empty
        hits_nomatch = bm25.search("policy", filters={"category": "non_existent_category"})
        assert hits_nomatch == []
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_vector_store_metadata_filtering(test_chunks: list[Chunk]):
    temp_dir = tempfile.mkdtemp()
    try:
        vstore = ChromaVectorStore(collection_name="test_vstore_filter", persist_dir=temp_dir)
        vstore.add_chunks(test_chunks)

        query_emb = test_chunks[0].embedding or normalize_vector([0.1] * 384)

        # Category filter
        res_cat = vstore.search(query_emb, top_k=10, filters={"category": "security"})
        assert len(res_cat) == 1
        assert res_cat[0].chunk.id == "chunk_tech_p1"

        # Source file filter
        res_source = vstore.search(query_emb, top_k=10, filters={"source_file": "employee_handbook.docx"})
        assert len(res_source) == 1
        assert res_source[0].chunk.id == "chunk_handbook_p5"

        # Page number filter
        res_page = vstore.search(query_emb, top_k=10, filters={"page_number": 2})
        assert len(res_page) == 1
        assert res_page[0].chunk.id == "chunk_policy_p1_2"

        # Combined filter matching nothing
        res_empty = vstore.search(query_emb, top_k=10, filters={"category": "policy", "page_number": 99})
        assert res_empty == []
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_hybrid_retriever_metadata_filtering(test_chunks: list[Chunk]):
    temp_dir = tempfile.mkdtemp()
    try:
        vstore = ChromaVectorStore(collection_name="test_hybrid_filter", persist_dir=temp_dir)
        vstore.add_chunks(test_chunks)
        emb_svc = EmbeddingService(cache_enabled=False)
        dense_retriever = DenseVectorRetriever(vector_store=vstore, embedding_service=emb_svc)

        bm25 = BM25SearchIndex(storage_dir=f"{temp_dir}/bm25")
        bm25.build_index(test_chunks)

        hybrid = HybridRetriever(dense_retriever=dense_retriever, bm25_index=bm25)

        # Hybrid search with metadata filter for category="policy"
        results = hybrid.retrieve("paid time off security hours", filters={"category": "policy"})
        assert len(results) == 2
        assert set(r.chunk.metadata.category for r in results) == {"policy"}

        # Hybrid search with combined filter for source_file and page_number
        results_comb = hybrid.retrieve("paid time off", filters={"source_file": "pto_policy.pdf", "page_number": 2})
        assert len(results_comb) == 1
        assert results_comb[0].chunk.id == "chunk_policy_p1_2"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# 3. Reranker Thresholding Edge Cases
# ============================================================================


def test_relative_score_threshold_postprocessor_zero_candidates():
    postprocessor = RelativeScoreThresholdPostprocessor(min_ratio=0.45, min_keep=1)
    assert postprocessor.filter([]) == []


def test_relative_score_threshold_postprocessor_low_similarity_chunks(test_chunks: list[Chunk]):
    c1, c2, c3, c4 = test_chunks
    postprocessor = RelativeScoreThresholdPostprocessor(min_ratio=0.45, min_keep=1)

    # Top score = 10.0. Threshold = 10.0 * 0.45 = 4.5
    # Chunks: c1 (10.0 - keep), c2 (5.0 - keep), c3 (4.0 - drop), c4 (1.0 - drop)
    candidates = [
        ScoredChunk(chunk=c1, score=10.0, rerank_score=10.0),
        ScoredChunk(chunk=c2, score=5.0, rerank_score=5.0),
        ScoredChunk(chunk=c3, score=4.0, rerank_score=4.0),
        ScoredChunk(chunk=c4, score=1.0, rerank_score=1.0),
    ]

    filtered = postprocessor.filter(candidates)
    assert len(filtered) == 2
    assert [c.chunk.id for c in filtered] == ["chunk_policy_p1_1", "chunk_policy_p1_2"]


def test_relative_score_threshold_postprocessor_negative_scores(test_chunks: list[Chunk]):
    c1, c2, c3 = test_chunks[:3]
    postprocessor = RelativeScoreThresholdPostprocessor(min_ratio=0.45, min_keep=1)

    # Cross-encoder logits can be negative when similarity is low.
    # Top score = -1.5 (<= 0)
    candidates = [
        ScoredChunk(chunk=c1, score=-1.5, rerank_score=-1.5),
        ScoredChunk(chunk=c2, score=-4.0, rerank_score=-4.0),
        ScoredChunk(chunk=c3, score=-8.0, rerank_score=-8.0),
    ]

    filtered = postprocessor.filter(candidates)
    # When top_score <= 0, triggers top_score <= 0 branch and returns top min_keep (1) candidate
    assert len(filtered) == 1
    assert filtered[0].chunk.id == "chunk_policy_p1_1"


def test_relative_score_threshold_postprocessor_min_keep_enforcement(test_chunks: list[Chunk]):
    c1, c2, c3 = test_chunks[:3]
    # min_keep = 2, top_score = 10.0, min_ratio = 0.80 -> threshold = 8.0
    # c1 = 10.0 (keep), c2 = 5.0 (< 8.0), c3 = 2.0 (< 8.0)
    postprocessor = RelativeScoreThresholdPostprocessor(min_ratio=0.80, min_keep=2)

    candidates = [
        ScoredChunk(chunk=c1, score=10.0, rerank_score=10.0),
        ScoredChunk(chunk=c2, score=5.0, rerank_score=5.0),
        ScoredChunk(chunk=c3, score=2.0, rerank_score=2.0),
    ]

    filtered = postprocessor.filter(candidates)
    # Since only 1 chunk >= 8.0, but min_keep=2, postprocessor returns top 2 candidates
    assert len(filtered) == 2
    assert [c.chunk.id for c in filtered] == ["chunk_policy_p1_1", "chunk_policy_p1_2"]


def test_cross_encoder_reranker_fallback_on_model_error(test_chunks: list[Chunk]):
    reranker = CrossEncoderReranker(top_n=3, min_ratio=0.45)
    candidates = [
        ScoredChunk(chunk=sc, score=float(i + 1), rerank_score=None)
        for i, sc in enumerate(test_chunks)
    ]

    # Force model load failure
    with patch.object(reranker, "_init_model") as mock_init:
        reranker._model_loaded = True
        reranker._model = None  # Model unavailable

        result = reranker.rerank("pto policy", candidates)
        assert len(result) <= 3
        # Ranks must be assigned 1..N
        assert [r.rank for r in result] == list(range(1, len(result) + 1))


def test_cross_encoder_reranker_top_n_truncation(test_chunks: list[Chunk]):
    reranker = CrossEncoderReranker(top_n=2, min_ratio=0.1)

    candidates = [
        ScoredChunk(chunk=c, score=1.0)
        for c in test_chunks
    ]

    mock_model = MagicMock()
    # Mock return logits: 10.0, 8.0, 6.0, 4.0
    # top_score = 10.0, min_ratio = 0.1 -> threshold = 1.0. All 4 candidates pass threshold.
    # top_n = 2 -> truncates result to top 2.
    mock_model.predict.return_value = [10.0, 8.0, 6.0, 4.0]

    with patch.object(reranker, "_init_model"), patch.object(reranker, "_model", mock_model):
        reranker._model_loaded = True
        result = reranker.rerank("pto policy", candidates)
        assert len(result) == 2
        assert result[0].rank == 1
        assert result[1].rank == 2
        assert result[0].rerank_score == 10.0
        assert result[1].rerank_score == 8.0


def test_cross_encoder_reranker_real_model_thresholding(test_chunks: list[Chunk]):
    """Test CrossEncoderReranker end-to-end with real or fallback model scoring and postprocessing."""
    reranker = CrossEncoderReranker(top_n=5, min_ratio=0.45)
    candidates = [
        ScoredChunk(chunk=c, score=1.0)
        for c in test_chunks
    ]

    result = reranker.rerank("PTO paid time off policy days per calendar year", candidates)
    assert len(result) >= 1
    # Check that returned chunks are ordered descending by rerank_score / score
    scores = [r.rerank_score if r.rerank_score is not None else r.score for r in result]
    assert sorted(scores, reverse=True) == scores
    # Check ranks are 1..N
    assert [r.rank for r in result] == list(range(1, len(result) + 1))

