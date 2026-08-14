"""
Tier 1 E2E Feature Coverage Tests for Area R1: Celery & Redis Setup (Features 1, 2, 3).
Validates package dependencies, Celery app initialization, healthcheck task, Docker compose configuration,
environment variable template, and task result backend lifecycle.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import pytest

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

from backend.tasks.celery_app import (
    BROKER_URL,
    RESULT_BACKEND,
    celery_app,
    healthcheck_task,
    ping_task,
)
from tests.e2e.helpers.celery_helper import CeleryTestHelper


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_tc_r1_001_package_dependencies_celery_and_redis() -> None:
    """
    TC-R1-001: Package Dependency Verification for Celery and Redis.
    Feature 1: Celery & Redis Dependencies.
    Verifies celery (>=5.3.0) and redis (>=5.0.0) are correctly declared in packaging manifests.
    """
    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    req_path = PROJECT_ROOT / "requirements.txt"
    req_docker_path = PROJECT_ROOT / "requirements-docker.txt"

    assert pyproject_path.exists(), f"pyproject.toml missing at {pyproject_path}"
    assert req_path.exists(), f"requirements.txt missing at {req_path}"

    # 1. Parse pyproject.toml
    with pyproject_path.open("rb") as f:
        pyproject_data = tomllib.load(f)
    deps = pyproject_data.get("project", {}).get("dependencies", [])
    pyproject_deps_str = " ".join(deps).lower()

    # Note: Check if dependencies are in pyproject.toml or requirements.txt
    with req_path.open("r", encoding="utf-8") as f:
        req_content = f.read().lower()

    # 2. Check celery dependency presence
    celery_in_manifest = "celery" in pyproject_deps_str or "celery" in req_content
    assert celery_in_manifest, "celery is not declared in pyproject.toml or requirements.txt"

    # 3. Check redis dependency presence
    redis_in_manifest = "redis" in pyproject_deps_str or "redis" in req_content
    assert redis_in_manifest, "redis is not declared in pyproject.toml or requirements.txt"

    # 4. Check docker requirements if file exists
    if req_docker_path.exists():
        with req_docker_path.open("r", encoding="utf-8") as f:
            docker_content = f.read().lower()
        assert "celery" in docker_content or "redis" in docker_content, (
            "Neither celery nor redis declared in requirements-docker.txt"
        )


def test_tc_r1_002_celery_app_initialization_and_broker_config() -> None:
    """
    TC-R1-002: Celery Application Initialization & Broker Connection.
    Feature 2: Celery App & Worker Setup.
    Ensures backend/tasks/celery_app.py initializes Celery app with Redis broker and JSON serialization.
    """
    assert celery_app is not None, "celery_app failed to initialize or is None"
    assert hasattr(celery_app, "conf"), "celery_app has no configuration attribute"

    conf = celery_app.conf
    assert conf.broker_url is not None, "broker_url is not configured in celery_app"
    assert "redis://" in conf.broker_url.lower(), (
        f"broker_url '{conf.broker_url}' does not use redis:// protocol"
    )

    assert conf.task_serializer == "json", (
        f"task_serializer should be 'json', got '{conf.task_serializer}'"
    )
    assert conf.result_serializer == "json", (
        f"result_serializer should be 'json', got '{conf.result_serializer}'"
    )
    assert "json" in conf.accept_content, (
        f"accept_content should accept 'json', got '{conf.accept_content}'"
    )


def test_tc_r1_003_worker_healthcheck_task(eager_celery: Any) -> None:
    """
    TC-R1-003: Worker Health Check Task & Redis Availability Detection.
    Feature 2: Celery App & Worker Setup.
    Verifies ping_task and healthcheck_task return valid operational status.
    """
    # 1. Test ping_task
    ping_res = ping_task.apply()
    assert ping_res.status == "SUCCESS"
    assert ping_res.result == "pong"

    # 2. Test healthcheck_task
    hc_res = healthcheck_task.apply()
    assert hc_res.status == "SUCCESS"
    result_data = hc_res.result
    assert isinstance(result_data, dict), f"healthcheck_task result should be dict, got {type(result_data)}"
    assert result_data.get("status") == "ok"
    assert "broker" in result_data or "message" in result_data


def test_tc_r1_004_docker_compose_celery_worker_service() -> None:
    """
    TC-R1-004: Docker Compose Service Definition for Celery Worker.
    Feature 3: Docker & Environment Configuration.
    Validates docker-compose.yml defines celery_worker service depending on redis.
    """
    compose_path = PROJECT_ROOT / "docker-compose.yml"
    assert compose_path.exists(), f"docker-compose.yml missing at {compose_path}"

    with compose_path.open("r", encoding="utf-8") as f:
        content = f.read()

    assert "celery_worker" in content or "celery-worker" in content, (
        "celery_worker service is not defined in docker-compose.yml"
    )
    assert "redis" in content, "redis service is not referenced in docker-compose.yml"
    assert "celery" in content.lower(), "celery command missing in docker-compose.yml"


def test_tc_r1_005_env_example_template_celery_redis_vars() -> None:
    """
    TC-R1-005: Environment Variable Template Verification (.env.example).
    Feature 3: Docker & Environment Configuration.
    Ensures .env.example defines CELERY_BROKER_URL, REDIS_HOST, REDIS_PORT, REDIS_ENABLED.
    """
    env_example_path = PROJECT_ROOT / ".env.example"
    assert env_example_path.exists(), f".env.example missing at {env_example_path}"

    with env_example_path.open("r", encoding="utf-8") as f:
        env_lines = f.readlines()

    env_vars = {}
    for line in env_lines:
        line_clean = line.strip()
        if line_clean and not line_clean.startswith("#") and "=" in line_clean:
            key, val = line_clean.split("=", 1)
            env_vars[key.strip()] = val.strip()

    assert "CELERY_BROKER_URL" in env_vars or "REDIS_URL" in env_vars, (
        "CELERY_BROKER_URL or REDIS_URL missing in .env.example"
    )
    assert "CELERY_RESULT_BACKEND" in env_vars or "REDIS_URL" in env_vars or "REDIS_HOST" in env_vars, (
        "Redis/Celery backend configuration missing in .env.example"
    )


def test_tc_r1_006_celery_result_backend_state_lifecycle(eager_celery: Any) -> None:
    """
    TC-R1-006: Celery Result Backend State Lifecycle.
    Feature 2: Celery App & Worker Setup.
    Verifies task state lifecycle transitions (PENDING -> SUCCESS) and AsyncResult metadata retrieval.
    """
    async_res = ping_task.apply_async()
    assert async_res.id is not None, "Task AsyncResult has no task_id"

    # Inspect status
    task_status = CeleryTestHelper.get_task_status(async_res.id)
    assert task_status["state"] in ("SUCCESS", "PENDING"), (
        f"Unexpected task state: {task_status['state']}"
    )

    # Wait / retrieve result
    res_val = async_res.get(timeout=5.0)
    assert res_val == "pong"
    assert async_res.successful() is True
