"""
Unit tests for Milestone 1: Celery App Initialization & Async Redis Client Setup.
"""
import os
import pytest
from unittest.mock import patch

from backend.tasks.celery_app import (
    celery_app,
    app,
    get_redis_url,
    ping_task,
    healthcheck_task,
    BROKER_URL,
    RESULT_BACKEND,
)
from backend.utils.redis_client import (
    get_redis_connection_url,
    get_redis_pool,
    get_redis_client,
    close_redis_client,
    check_redis_connection,
)


def test_get_redis_url_custom_env():
    """Test get_redis_url with custom environment variables."""
    with patch.dict(os.environ, {"CELERY_BROKER_URL": "redis://custom-host:6380/2"}):
        assert get_redis_url() == "redis://custom-host:6380/2"

    with patch.dict(
        os.environ,
        {
            "CELERY_BROKER_URL": "",
            "REDIS_URL": "",
            "REDIS_HOST": "myredis",
            "REDIS_PORT": "7000",
            "REDIS_DB": "1",
            "REDIS_PASSWORD": "secretpassword",
        },
        clear=False,
    ):
        assert get_redis_url() == "redis://:secretpassword@myredis:7000/1"


def test_celery_app_configuration():
    """Test that celery_app is initialized with expected settings."""
    assert celery_app is not None
    assert app is celery_app
    assert celery_app.main == "company_policy_rag"
    assert celery_app.conf.task_serializer == "json"
    assert celery_app.conf.result_serializer == "json"
    assert celery_app.conf.accept_content == ["json"]
    assert celery_app.conf.broker_connection_retry_on_startup is True


def test_celery_ping_and_healthcheck_tasks():
    """Test direct local execution of ping_task and healthcheck_task."""
    ping_result = ping_task.run()
    assert ping_result == "pong"

    health_result = healthcheck_task.run()
    assert isinstance(health_result, dict)
    assert health_result["status"] == "ok"
    assert health_result["message"] == "Celery worker is operational"
    assert "broker" in health_result


def test_redis_client_connection_url():
    """Test Redis connection URL construction in redis_client module."""
    with patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379/0"}):
        url = get_redis_connection_url()
        assert url.startswith("redis://")


@pytest.mark.asyncio
async def test_redis_client_pool_and_cleanup():
    """Test pool creation and cleanup without error."""
    pool = get_redis_pool()
    assert pool is not None
    await close_redis_client()


@pytest.mark.asyncio
async def test_check_redis_connection_graceful():
    """Test check_redis_connection executes and returns a boolean value."""
    result = await check_redis_connection()
    assert isinstance(result, bool)
    await close_redis_client()
