"""Tests for retrieval + reranker configuration (no model loading)."""

from __future__ import annotations

from src.config import settings
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.retriever import (
    _PostprocessingRetriever,
    _check_reranker_dependencies,
    _resolve_reranker_device,
    apply_postprocessors,
    get_final_top_k,
    get_initial_top_k,
    get_node_postprocessors,
    get_reranker_install_hints,
    reset_reranker_cache,
)


def setup_function() -> None:
    reset_reranker_cache()


def test_initial_top_k_with_reranker_enabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "enable_reranker", True)
    assert get_initial_top_k() == settings.retrieval_candidate_k


def test_final_top_k_with_reranker_enabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "enable_reranker", True)
    assert get_final_top_k() == settings.reranker_top_n


def test_no_postprocessors_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "enable_reranker", False)
    reset_reranker_cache()
    assert get_node_postprocessors() == []


def test_initial_top_k_without_reranker(monkeypatch) -> None:
    monkeypatch.setattr(settings, "enable_reranker", False)
    assert get_initial_top_k() == settings.similarity_top_k
    assert get_final_top_k() == settings.similarity_top_k


class _KeepTopTwo:
    """Minimal postprocessor stub for wrapper tests."""

    def postprocess_nodes(self, nodes, query_bundle=None):
        return nodes[:2]


def test_postprocessing_retriever_applies_postprocessors() -> None:
    class _StubRetriever:
        def retrieve(self, query):
            return [
                NodeWithScore(node=TextNode(text="a"), score=1.0),
                NodeWithScore(node=TextNode(text="b"), score=0.9),
                NodeWithScore(node=TextNode(text="c"), score=0.8),
            ]

    wrapped = _PostprocessingRetriever(_StubRetriever(), [_KeepTopTwo()])
    result = wrapped.retrieve(QueryBundle(query_str="test"))
    assert len(result) == 2


def test_apply_postprocessors_chain() -> None:
    nodes = [
        NodeWithScore(node=TextNode(text="a"), score=1.0),
        NodeWithScore(node=TextNode(text="b"), score=0.9),
        NodeWithScore(node=TextNode(text="c"), score=0.8),
    ]
    filtered = apply_postprocessors(nodes, QueryBundle(query_str="q"), [_KeepTopTwo()])
    assert len(filtered) == 2


def test_reranker_install_hints_include_torch() -> None:
    hints = get_reranker_install_hints()
    assert "torch" in hints["cpu"]
    assert "cu124" in hints["gpu_cuda124"]


def test_reranker_dependencies_present_in_dev_env() -> None:
    """Sanity check — CI/dev venv should have reranker deps installed."""
    missing = _check_reranker_dependencies()
    assert missing == [], f"Missing reranker deps: {missing}"


def test_resolve_reranker_device_explicit_cpu() -> None:
    assert _resolve_reranker_device("cpu") == "cpu"


def test_resolve_reranker_device_auto_returns_cpu_or_cuda() -> None:
    device = _resolve_reranker_device("auto")
    assert device in ("cpu", "cuda")


def test_retrieval_config_includes_phase2_flags() -> None:
    from src.retriever import get_retrieval_config_summary

    summary = get_retrieval_config_summary()
    assert "enable_hybrid_bm25" in summary
    assert "enable_parent_document_retrieval" in summary
    assert summary["enable_hybrid_bm25"] is True
    assert "enable_code_retrieval_boost" in summary