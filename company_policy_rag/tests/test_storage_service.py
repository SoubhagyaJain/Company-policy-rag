"""Storage inventory and cleanup, exercised entirely on temporary directories."""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.services import storage_service as module
from backend.services.storage_service import (
    StorageBusyError,
    StorageService,
    UnknownStorageActionError,
)
from backend.services.telemetry_db import TelemetryDB
from backend.utils.redis_cache import RedisCache
from backend.vision.image_asset_manager import ImageAssetManager
from backend.vision.vision_cache import VisionCacheManager

LIVE_DOC = "doc_aaaaaaaaaaaa"
ORPHAN_DOC = "doc_bbbbbbbbbbbb"
MODEL = "Qwen3-VL-2B-Instruct"


class _DocService:
    def __init__(self, root: Path) -> None:
        self.vector_store = SimpleNamespace(persist_dir=root / "chroma")
        self.vector_store.persist_dir.mkdir(parents=True)
        self.image_asset_manager = ImageAssetManager(storage_dir=root / "images")
        self.vision_cache_manager = VisionCacheManager(cache_dir=root / "vision_cache")
        self.storage_dir = root / "uploads"
        self.storage_dir.mkdir()
        (root / "bm25").mkdir()
        self.bm25_index = SimpleNamespace(storage_dir=root / "bm25", entries=[])
        self.embedding_service = SimpleNamespace(model_name="", cache=None)
        self.docstore: dict[str, Any] = {}
        self.known = {LIVE_DOC}
        self.busy = False

    def known_document_ids(self) -> set[str]:
        return set(self.known)

    def has_ingestion_in_flight(self) -> bool:
        return self.busy


@pytest.fixture(autouse=True)
def _no_external_probes(monkeypatch):
    """Keep the tests off Ollama, nvidia-smi and Redis."""
    monkeypatch.setattr(module, "list_loaded_models", lambda **_: [])
    monkeypatch.setattr(module, "list_installed_models", lambda **_: [])
    monkeypatch.setattr(module, "gpu_memory_mb", lambda *_: None)
    cache = RedisCache(enabled=False)
    monkeypatch.setattr(module, "get_redis_cache", lambda: cache)


@pytest.fixture
def env(tmp_path: Path):
    doc_service = _DocService(tmp_path)
    telemetry = SimpleNamespace(db=TelemetryDB(db_path=tmp_path / "telemetry.sqlite3"))
    telemetry.clear = telemetry.db.clear
    logs = tmp_path / "logs"
    logs.mkdir()
    service = StorageService(
        doc_service=doc_service,
        telemetry_service=telemetry,
        logs_dir=logs,
        eval_dirs=[tmp_path / "eval_corpora"],
        sessions_dir=tmp_path / "sessions",
        history_path=tmp_path / "history.jsonl",
        readonly_dirs={},
        model_dirs={},
    )
    return SimpleNamespace(root=tmp_path, docs=doc_service, telemetry=telemetry, logs=logs, service=service)


def _write(path: Path, size: int, age_days: float = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    if age_days:
        stamp = time.time() - age_days * 86400
        os.utime(path, (stamp, stamp))
    return path


def _store(summary: dict[str, Any], store_id: str) -> dict[str, Any]:
    return next(s for s in summary["stores"] if s["id"] == store_id)


def _seed_images_and_cache(env) -> None:
    _write(env.root / "images" / LIVE_DOC / "page_1.jpeg", 1000)
    _write(env.root / "images" / ORPHAN_DOC / "page_1.jpeg", 3000)
    _write(env.root / "images" / ORPHAN_DOC / "page_2.jpeg", 2000)
    _write(env.root / "vision_cache" / f"{LIVE_DOC}_p1_aaaaaaaaaaaaaaaa_{MODEL}.json", 100)
    _write(env.root / "vision_cache" / f"{ORPHAN_DOC}_p1_bbbbbbbbbbbbbbbb_{MODEL}.json", 400)
    _write(env.root / "vision_cache" / f"cccccccccccccccc_{MODEL}.json", 50)  # content-addressed, no owner


def test_summary_reports_sizes_and_what_is_orphaned(env):
    _seed_images_and_cache(env)

    summary = env.service.summary()

    images = _store(summary, "page_images")
    assert images["size_bytes"] == 6000
    assert images["items"] == 2
    assert images["details"]["orphaned"] == 1
    assert images["reclaimable_bytes"] == 5000

    cache = _store(summary, "vision_cache")
    assert cache["size_bytes"] == 550
    assert cache["details"]["orphaned"] == 1
    assert cache["reclaimable_bytes"] == 400

    assert summary["totals"]["reclaimable_bytes"] >= 5400
    assert summary["busy"] is False


def test_remove_orphaned_keeps_everything_a_live_document_owns(env):
    _seed_images_and_cache(env)

    images = env.service.run_action("page_images", "remove_orphaned")
    cache = env.service.run_action("vision_cache", "remove_orphaned")

    assert (images["freed_bytes"], images["removed_items"]) == (5000, 1)
    assert (cache["freed_bytes"], cache["removed_items"]) == (400, 1)
    assert not (env.root / "images" / ORPHAN_DOC).exists()
    assert (env.root / "images" / LIVE_DOC / "page_1.jpeg").is_file()
    remaining = sorted(p.name for p in (env.root / "vision_cache").glob("*.json"))
    assert remaining == [f"cccccccccccccccc_{MODEL}.json", f"{LIVE_DOC}_p1_aaaaaaaaaaaaaaaa_{MODEL}.json"]


def test_delete_refuses_a_path_outside_the_store(env):
    outside = _write(env.root / "uploads" / "precious.pdf", 10)

    with pytest.raises(ValueError, match="Refusing to delete outside"):
        env.service._delete(env.root / "vision_cache", outside)
    with pytest.raises(ValueError, match="Refusing to delete outside"):
        env.service._delete(env.root / "vision_cache", env.root / "vision_cache")

    assert outside.is_file()


def test_age_based_cleanup_keeps_recent_items_and_the_live_log(env):
    old_cache = _write(env.root / "vision_cache" / f"{LIVE_DOC}_p1_aaaaaaaaaaaaaaaa_{MODEL}.json", 100, age_days=40)
    new_cache = _write(env.root / "vision_cache" / f"{LIVE_DOC}_p2_dddddddddddddddd_{MODEL}.json", 100)
    live_log = _write(env.logs / "app.log", 500, age_days=90)
    placeholder = _write(env.logs / ".gitkeep", 1, age_days=400)
    old_report = _write(env.logs / "retrieval_eval" / "run1.json", 700, age_days=90)
    new_report = _write(env.logs / "evaluation_results.json", 300)

    cache = env.service.run_action("vision_cache", "older_than", older_than_days=30)
    logs = env.service.run_action("logs", "older_than", older_than_days=30)

    assert cache["removed_items"] == 1 and not old_cache.exists() and new_cache.is_file()
    assert (logs["freed_bytes"], logs["removed_items"]) == (700, 1)
    assert not old_report.exists() and not old_report.parent.exists()
    assert live_log.is_file() and new_report.is_file()
    # A repository placeholder is not a log, however old it is.
    assert placeholder.is_file()


def test_action_validation(env):
    with pytest.raises(UnknownStorageActionError):
        env.service.run_action("uploads", "delete")
    with pytest.raises(UnknownStorageActionError):
        env.service.run_action("vision_cache", "format_disk")
    with pytest.raises(ValueError, match="older_than_days"):
        env.service.run_action("logs", "older_than")
    with pytest.raises(ValueError, match="older_than_days"):
        env.service.run_action("logs", "older_than", older_than_days=0)
    with pytest.raises(ValueError, match="not loaded"):
        env.service.run_action("ollama_models", "unload", target="qwen2.5:7b")


def test_index_cleanups_wait_for_a_running_ingestion(env):
    _seed_images_and_cache(env)
    env.docs.busy = True

    with pytest.raises(StorageBusyError):
        env.service.run_action("vector_index", "compact")
    with pytest.raises(StorageBusyError):
        env.service.run_action("vision_cache", "clear")

    result = env.service.clean_safe()
    skipped = [r for r in result["results"] if "skipped" in r]
    assert [r["store"] for r in skipped] == ["vector_index"]
    # Orphan removal does not touch what the ingestion writes, so it still runs.
    assert not (env.root / "images" / ORPHAN_DOC).exists()
    assert env.service.summary()["busy"] is True


def test_history_records_actions_and_snapshots_and_stays_bounded(env, monkeypatch):
    _seed_images_and_cache(env)
    env.service.summary()
    env.service.run_action("page_images", "remove_orphaned")

    history = env.service.history()
    assert [a["store"] for a in history["actions"]] == ["page_images"]
    assert history["actions"][0]["freed_bytes"] == 5000
    sizes = [s["sizes"]["page_images"] for s in history["snapshots"]]
    assert sizes == [6000, 1000]

    monkeypatch.setattr(module, "_HISTORY_MAX_LINES", 5)
    monkeypatch.setattr(module, "_HISTORY_KEEP_LINES", 3)
    for _ in range(6):
        env.service.run_action("retrieval_cache", "clear", snapshot=False)
    assert len(env.service.history_path.read_text(encoding="utf-8").splitlines()) <= 5


def test_telemetry_prune_and_compact(env):
    db_path = env.telemetry.db.db_path
    connection = sqlite3.connect(db_path)
    with connection:
        for index in range(400):
            stamp = "2020-01-01T00:00:00+00:00" if index < 300 else "2999-01-01T00:00:00+00:00"
            connection.execute(
                "INSERT INTO cache_events (id, timestamp, cache_type, event_type, latency_ms, key_hash, model_name) "
                "VALUES (?, ?, 'Vision Cache', 'HIT', 1.0, ?, 'm')",
                (f"evt_{index}", stamp, "k" * 400),
            )
    connection.close()

    pruned = env.service.run_action("telemetry_db", "older_than", older_than_days=30)
    assert pruned["removed_items"] == 300
    assert env.telemetry.db.table_counts()["cache_events"] == 100

    before = _store(env.service.summary(), "telemetry_db")
    assert before["details"]["free_pages"] > 0
    env.service.run_action("telemetry_db", "compact")
    after = _store(env.service.summary(), "telemetry_db")
    assert after["details"]["free_pages"] == 0
    assert after["items"] == 100


def test_compacting_the_vector_index_keeps_it_searchable(env):
    from backend.embeddings.vector_store import get_shared_chroma_client

    client = get_shared_chroma_client(env.docs.vector_store.persist_dir)
    collection = client.get_or_create_collection("company_policies", metadata={"hnsw:space": "cosine"})
    ids = [f"chunk_{i}" for i in range(300)]
    collection.add(
        ids=ids,
        embeddings=[[float(i % 7), float(i % 11), 1.0, 0.5] for i in range(300)],
        documents=[f"policy text {i} " + "lorem ipsum " * 40 for i in range(300)],
    )
    collection.delete(ids=ids[10:])

    before = _store(env.service.summary(), "vector_index")
    assert before["items"] == 10
    result = env.service.run_action("vector_index", "compact")
    after = _store(env.service.summary(), "vector_index")

    assert after["details"]["free_pages"] == 0
    assert after["size_bytes"] <= before["size_bytes"]
    assert result["freed_bytes"] == before["size_bytes"] - after["size_bytes"]
    assert collection.count() == 10
    hit = collection.query(query_embeddings=[[3.0, 3.0, 1.0, 0.5]], n_results=1)
    assert hit["ids"][0][0] == "chunk_3"


def test_memory_section_lists_caches_and_loaded_models(env, monkeypatch):
    resident = [
        {"name": "qwen2.5:7b", "size": 5_000, "size_vram": 4_000, "context_length": 4096, "expires_at": "2319-01-01T00:00:00Z"}
    ]
    monkeypatch.setattr(module, "list_loaded_models", lambda **_: list(resident))
    unloaded: list[str] = []

    def fake_unload(name: str) -> bool:
        unloaded.append(name)
        resident.clear()
        return True

    monkeypatch.setattr(module, "unload_model", fake_unload)

    memory = env.service.summary()["memory"]
    models = next(i for i in memory["items"] if i["id"] == "ollama_models")
    assert models["models"] == [
        {"name": "qwen2.5:7b", "size_bytes": 5_000, "vram_bytes": 4_000, "context_length": 4096, "pinned": True}
    ]
    assert {i["id"] for i in memory["items"]} >= {"kv_cache", "retrieval_cache", "vision_model", "conversations"}

    result = env.service.run_action("ollama_models", "unload", target="qwen2.5:7b")
    assert unloaded == ["qwen2.5:7b"]
    assert result["freed_bytes"] == 4_000
    after = next(i for i in env.service.summary()["memory"]["items"] if i["id"] == "ollama_models")
    assert after["value"] == 0


def test_storage_api(env):
    from backend.api.dependencies import get_storage_service
    from backend.api.main import create_app

    _seed_images_and_cache(env)
    app = create_app()
    app.dependency_overrides[get_storage_service] = lambda: env.service
    client = TestClient(app)

    summary = client.get("/api/admin/storage")
    assert summary.status_code == 200
    assert {"stores", "memory", "models", "totals", "busy"} <= summary.json().keys()

    assert client.post("/api/admin/storage/uploads/delete").status_code == 404
    assert client.post("/api/admin/storage/logs/older_than", json={}).status_code == 400

    env.docs.busy = True
    assert client.post("/api/admin/storage/vector_index/compact").status_code == 409
    env.docs.busy = False

    removed = client.post("/api/admin/storage/page_images/remove_orphaned")
    assert removed.status_code == 200
    assert removed.json()["freed_bytes"] == 5000

    cleanup = client.post("/api/admin/storage/cleanup")
    assert cleanup.status_code == 200
    assert cleanup.json()["removed_items"] == 1  # the orphaned vision cache page

    history = client.get("/api/admin/storage/history").json()
    assert [a["store"] for a in history["actions"]][0] == "page_images"
