"""Startup order must keep the generator resident beside the CPU embedder."""

from types import SimpleNamespace


def test_generator_warms_before_cpu_embedder(monkeypatch):
    import backend.api.dependencies as dependencies
    import src.ollama_client as ollama_client
    from backend.api.main import warmup_rag_system
    from src.config import settings

    events = []
    embedding_service = SimpleNamespace(
        _init_model=lambda: events.append("embed_init"),
        embed_text=lambda _text: events.append("embed_warmup"),
    )
    doc_service = SimpleNamespace(
        embedding_service=embedding_service,
        vector_store=SimpleNamespace(count=lambda: 0),
        bm25_index=SimpleNamespace(entries=[]),
    )
    pipeline = SimpleNamespace(
        reranker=SimpleNamespace(_model=None),
        get_active_model=lambda: "qwen3.5:9b",
        llm=None,
        vision_service=SimpleNamespace(vision_model="vision", is_available=lambda: (True, "ready")),
    )
    monkeypatch.setattr(settings, "enable_reranker", False)
    monkeypatch.setattr(dependencies, "get_document_service", lambda: doc_service)
    monkeypatch.setattr(dependencies, "get_rag_pipeline", lambda: pipeline)
    monkeypatch.setattr(dependencies, "get_chat_service", lambda: None)
    monkeypatch.setattr(dependencies, "get_semantic_cache_manager", lambda: None)
    monkeypatch.setattr(
        ollama_client, "preload_model", lambda _model: events.append("generator_warmup")
    )

    warmup_rag_system()

    assert events == ["generator_warmup", "embed_init", "embed_warmup"]
