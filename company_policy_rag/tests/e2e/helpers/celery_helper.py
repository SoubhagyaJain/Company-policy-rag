"""
Celery Task Execution & Worker Helper Module for E2E Tests.
Provides task inspection, eager/live worker pinging, and execution state checks.
"""
from __future__ import annotations

import contextlib
from typing import Any, Dict, Generator, List, Optional

try:
    from backend.tasks.celery_app import celery_app, healthcheck_task, ping_task
    CELERY_APP_IMPORTED = True
except ImportError:
    celery_app = None  # type: ignore[assignment]
    healthcheck_task = None  # type: ignore[assignment]
    ping_task = None  # type: ignore[assignment]
    CELERY_APP_IMPORTED = False


class CeleryTestHelper:
    """Helper class for interacting with and inspecting Celery worker state in E2E tests."""

    @staticmethod
    def is_celery_available() -> bool:
        """Check if Celery app is imported and available."""
        return CELERY_APP_IMPORTED and celery_app is not None

    @staticmethod
    def get_app() -> Any:
        """Return Celery app instance."""
        if not CeleryTestHelper.is_celery_available():
            raise RuntimeError("Celery application is not available or failed to import.")
        return celery_app

    @staticmethod
    def ping_worker(timeout: float = 2.0) -> bool:
        """
        Check if a live worker is listening on the broker by issuing inspector ping.
        """
        if not CeleryTestHelper.is_celery_available():
            return False
        try:
            inspector = celery_app.control.inspect(timeout=timeout)
            res = inspector.ping()
            return bool(res and len(res) > 0)
        except Exception:
            return False

    @staticmethod
    def run_healthcheck(eager: bool = True) -> Dict[str, Any]:
        """
        Run the worker healthcheck task in eager or async mode and return response dict.
        """
        if not CeleryTestHelper.is_celery_available():
            return {"status": "unavailable", "reason": "Celery not imported"}

        if eager:
            with CeleryTestHelper.eager_mode():
                res = healthcheck_task.apply()
                return res.result
        else:
            async_res = healthcheck_task.delay()
            return async_res.get(timeout=5.0)

    @staticmethod
    def get_task_status(task_id: str) -> Dict[str, Any]:
        """
        Retrieve state, result, and metadata for a task ID using Celery AsyncResult.
        """
        if not CeleryTestHelper.is_celery_available():
            return {"state": "UNKNOWN", "ready": False}

        from celery.result import AsyncResult
        res = AsyncResult(task_id, app=celery_app)
        return {
            "task_id": task_id,
            "state": res.state,
            "status": res.status,
            "ready": res.ready(),
            "successful": res.successful() if res.ready() else False,
            "result": res.result if res.ready() else None,
        }

    @staticmethod
    def get_registered_tasks() -> List[str]:
        """Get list of registered task names in Celery app."""
        if not CeleryTestHelper.is_celery_available():
            return []
        return list(celery_app.tasks.keys())

    @staticmethod
    @contextlib.contextmanager
    def eager_mode() -> Generator[Any, None, None]:
        """
        Context manager to set Celery task_always_eager=True synchronously during a test.
        """
        if not CeleryTestHelper.is_celery_available():
            yield None
            return

        orig_eager = celery_app.conf.task_always_eager
        orig_propagate = celery_app.conf.task_eager_propagates
        try:
            celery_app.conf.task_always_eager = True
            celery_app.conf.task_eager_propagates = True
            yield celery_app
        finally:
            celery_app.conf.task_always_eager = orig_eager
            celery_app.conf.task_eager_propagates = orig_propagate
