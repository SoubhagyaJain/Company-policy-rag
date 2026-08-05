from __future__ import annotations

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


def create_app() -> FastAPI:
    """FastAPI application factory configuring CORS, routers, and global error handling."""
    app = FastAPI(
        title="Enterprise Policy RAG System API",
        description="FastAPI backend providing RAG chat, sub-1s TTFT SSE streaming, document ingestion, and admin observability.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
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
            content={"detail": f"Internal server error: {str(exc)}", "error_code": "INTERNAL_ERROR"},
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
