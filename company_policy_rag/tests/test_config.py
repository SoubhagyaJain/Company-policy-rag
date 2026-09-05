"""Tests for centralized configuration."""

from __future__ import annotations

from src.config import settings


def test_chunk_size_in_recommended_range() -> None:
    # Default CHUNK_SIZE is 480 (guidebook-tuned); allow env overrides up to 768.
    assert 400 <= settings.chunk_size <= 768


def test_directories_exist() -> None:
    assert settings.policies_dir.exists()
    assert settings.legal_dir.exists()
    assert settings.storage_dir.exists()


def test_folder_document_types() -> None:
    assert settings.folder_document_types["policies"] == "company_policy"
    assert settings.folder_document_types["legal"] == "legal_document"


def test_section_detection_defaults() -> None:
    assert settings.enable_section_detection is True
    assert settings.section_detection_mode in ("standard", "strict", "permissive")


def test_chroma_defaults() -> None:
    assert settings.chroma_collection_name == "company_policies"
    assert settings.chroma_distance_fn == "cosine"
    assert settings.chroma_persist_dir.name == "chroma"


def test_eval_defaults() -> None:
    assert settings.eval_use_llm_judge is True
    assert settings.eval_dataset_path.name == "golden_dataset.json"
    assert settings.eval_results_path.name == "evaluation_results.json"


def test_reranker_defaults() -> None:
    assert settings.enable_reranker is True
    # A bge cross-encoder reranker (base is the CPU default; large is opt-in).
    assert "bge-reranker" in settings.reranker_model
    assert settings.retrieval_candidate_k >= 20
    assert settings.reranker_top_n <= 7
    assert settings.retrieval_candidate_k > settings.reranker_top_n
    assert settings.enable_rerank_score_filter is True
    assert settings.enable_query_rewrite is False


def test_memory_defaults() -> None:
    assert settings.enable_conversation_memory is True
    assert 4 <= settings.memory_window_size <= 6 or settings.memory_window_size == 5
    assert settings.memory_token_limit > 0


def test_grounding_defaults() -> None:
    assert settings.grounding_strictness == "balanced"
    assert settings.response_prompt_version == "v2_balanced"
    assert settings.strict_grounding is False
    assert settings.enable_faithfulness_check is True
    assert settings.faithfulness_guard_mode == "balanced"