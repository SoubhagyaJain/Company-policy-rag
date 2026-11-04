"""Deployment path contracts; no model downloads or live user storage."""

from pathlib import Path, PurePosixPath

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_compose_mounts_the_configured_library_without_changing_host_location():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    for name in ("backend", "celery_worker"):
        service = compose["services"][name]
        library = PurePosixPath(service["environment"]["APP_STORAGE_DIR"])
        mounts = [tuple(volume.split(":")[:2]) for volume in service["volumes"]]
        covering = [(source, target) for source, target in mounts if library.is_relative_to(target)]
        assert covering, f"{name}: {library} is in the ephemeral container layer"
        assert ("./app/storage", "/app/app/storage") in covering
        assert ("./storage", "/app/storage") in mounts  # retain existing images/telemetry/index data


def test_backend_image_prepares_the_non_root_library_directory():
    dockerfile = (ROOT / "docker" / "Dockerfile.backend").read_text(encoding="utf-8")
    assert "mkdir -p app/storage" in dockerfile


def test_default_network_bindings_are_local_and_redis_is_internal():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    for name in ("backend", "frontend"):
        assert all(port.startswith("127.0.0.1:") for port in compose["services"][name]["ports"])
    assert "ports" not in compose["services"]["redis"]
