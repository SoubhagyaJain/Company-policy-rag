"""
Celery Application Module for Asynchronous RAG Task Processing.
Configures Redis broker, result backend, serialization, and healthcheck tasks.
"""
from __future__ import annotations

import os
from typing import Any, Dict
from dotenv import load_dotenv

try:
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    Celery = None  # type: ignore[assignment]
    CELERY_AVAILABLE = False

load_dotenv()


def get_redis_url() -> str:
    """Helper to construct or retrieve Redis connection URL for Celery."""
    broker_url = os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_URL")
    if broker_url:
        return broker_url
    
    host = os.getenv("REDIS_HOST", "localhost")
    port = os.getenv("REDIS_PORT", "6379")
    db = os.getenv("REDIS_DB", "0")
    password = os.getenv("REDIS_PASSWORD", "")
    
    if password:
        return f"redis://:{password}@{host}:{port}/{db}"
    return f"redis://{host}:{port}/{db}"


BROKER_URL = get_redis_url()
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", BROKER_URL)

if CELERY_AVAILABLE:
    celery_app = Celery(
        "company_policy_rag",
        broker=BROKER_URL,
        backend=RESULT_BACKEND,
    )

    celery_app.conf.update(
        broker_url=BROKER_URL,
        result_backend=RESULT_BACKEND,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        broker_connection_retry_on_startup=True,
        result_expires=3600,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
    )

    # Optional auto-include for RAG tasks module
    try:
        celery_app.autodiscover_tasks(["backend.tasks"])
    except Exception:
        pass

    # Aliasing for Celery CLI convenience
    app = celery_app

    @celery_app.task(name="ping_task")
    def ping_task() -> str:
        """Lightweight healthcheck test task."""
        return "pong"

    @celery_app.task(name="healthcheck_task")
    def healthcheck_task() -> Dict[str, Any]:
        """Comprehensive Celery worker health check task."""
        return {
            "status": "ok",
            "message": "Celery worker is operational",
            "broker": BROKER_URL.split("@")[-1] if "@" in BROKER_URL else BROKER_URL,
        }

else:
    celery_app = None  # type: ignore[assignment]
    app = None  # type: ignore[assignment]
