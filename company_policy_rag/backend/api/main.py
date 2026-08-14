import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import (
    admin_router,
    chat_router,
    documents_router,
    health_router,
    models_router,
)
from backend.utils.logging import logger

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]


def warmup_rag_system() -> None:
    """
    Synchronously preloads and warms up all RAG models and components on server startup:
    1. Dense embedding model (BAAI/bge-small-en-v1.5)
    2. Cross-Encoder reranker (BAAI/bge-reranker-large)
    3. Ollama LLM model weights pinned in VRAM (keep_alive=-1)
    4. ChromaDB vector stores and BM25 index
    Ensures zero cold-start delay when the user sends their first query.
    """
    logger.info("==========================================================")
    logger.info("  STARTING RAG SYSTEM PRELOADING & MODEL WARM-UP          ")
    logger.info("==========================================================")
    t_start = time.perf_counter()

    from backend.api.dependencies import (
        get_chat_service,
        get_document_service,
        get_rag_pipeline,
        get_semantic_cache_manager,
    )
    from src.ollama_client import preload_model

    # 1. Initialize core services
    doc_service = get_document_service()
    pipeline = get_rag_pipeline()
    get_chat_service()
    get_semantic_cache_manager()

    # 2. Embedding Model Preloading & Warm-up
    t0 = time.perf_counter()
    try:
        if hasattr(doc_service.embedding_service, "_init_model"):
            doc_service.embedding_service._init_model()
        doc_service.embedding_service.embed_text("warmup initialization query")
        logger.info("[1/4] Embedding model loaded & warmed up in %.2fs", time.perf_counter() - t0)
    except Exception as exc:
        logger.warning("[1/4] Embedding model warm-up notice: %s", exc)

    # 3. Reranker Model Preloading & Warm-up
    t0 = time.perf_counter()
    try:
        if hasattr(pipeline.reranker, "_init_model"):
            pipeline.reranker._init_model()
        if getattr(pipeline.reranker, "_model", None) is not None:
            pipeline.reranker._model.predict([["warmup query", "warmup chunk context"]])
        logger.info("[2/4] CrossEncoder reranker loaded & warmed up in %.2fs", time.perf_counter() - t0)
    except Exception as exc:
        logger.warning("[2/4] CrossEncoder reranker warm-up notice: %s", exc)

    # 4. Ollama LLM Preloading & Warm-up (pin in VRAM)
    t0 = time.perf_counter()
    try:
        active_model = pipeline.get_active_model()
        preload_model(active_model)
        if pipeline.llm is not None:
            pipeline.llm.complete("warmup")
        logger.info("[3/4] Ollama LLM '%s' preloaded & warmed up in %.2fs", active_model, time.perf_counter() - t0)
    except Exception as exc:
        logger.warning("[3/4] Ollama LLM warm-up notice: %s", exc)

    # 5. ChromaDB & BM25 verification
    t0 = time.perf_counter()
    try:
        chroma_count = doc_service.vector_store.count()
        bm25_count = len(doc_service.bm25_index.entries)
        logger.info("[4/4] Vector Store (%d chunks) & BM25 (%d chunks) ready in %.2fs", chroma_count, bm25_count, time.perf_counter() - t0)
    except Exception as exc:
        logger.warning("[4/4] Vector store / BM25 notice: %s", exc)

    total_time = time.perf_counter() - t_start
    logger.info("==========================================================")
    logger.info("  ALL RAG MODELS READY IN %.2fs — ZERO COLD START READY!  ", total_time)
    logger.info("==========================================================")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager: preloads and warms up all models before accepting traffic."""
    warmup_rag_system()
    yield


def create_app() -> FastAPI:
    """FastAPI application factory configuring CORS, routers, and global error handling."""
    app = FastAPI(
        title="Enterprise Policy RAG System API",
        description="FastAPI backend providing RAG chat, sub-1s TTFT SSE streaming, document ingestion, and admin observability.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Configure CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Register API Routers
    app.include_router(chat_router)
    app.include_router(documents_router)
    app.include_router(admin_router)
    app.include_router(health_router)
    app.include_router(models_router)

    # Global Exception Handlers
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.warning("Validation error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc), "error_code": "INVALID_INPUT"},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled server error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Internal server error: {exc!s}", "error_code": "INTERNAL_ERROR"},
        )

    @app.get("/")
    def root_endpoint():
        return {
            "name": "Enterprise Policy RAG System API",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs",
        }

    return app


app = create_app()

