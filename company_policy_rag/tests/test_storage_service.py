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


# ── Observability console ───────────────────────────────────────────────────


def test_every_store_and_action_is_classified(env):
    summary = env.service.summary()

    for store in summary["stores"]:
        assert store["kinds"], store["id"]
        assert store["info"]["what"] and store["info"]["delete_effect"], store["id"]
    for item in summary["memory"]["items"]:
        assert item["kinds"] and item["info"], item["id"]
    assert {c["id"] for c in summary["caches"]} == {
        "semantic_cache",
        "vision_cache",
        "retrieval_cache",
        "embedding_cache",
        "kv_cache",
        "conversations",
    }

    for spec in env.service._actions.values():
        assert spec.safety in {"SAFE", "REBUILDABLE", "DESTRUCTIVE"}
        assert spec.deletes and spec.rebuild and spec.performance
        # Whatever the one-click cleanup runs must lose nothing.
        assert not spec.safe or spec.safety == "SAFE"
    safety = {key: spec.safety for key, spec in env.service._actions.items()}
    assert safety[("page_images", "remove_orphaned")] == "SAFE"
    assert safety[("semantic_cache", "clear")] == "REBUILDABLE"
    assert safety[("telemetry_db", "clear")] == "DESTRUCTIVE"
    assert safety[("eval_artifacts", "delete")] == "DESTRUCTIVE"


def test_summary_explains_where_storage_goes(env):
    _seed_images_and_cache(env)
    _write(env.root / "uploads" / f"{LIVE_DOC}_handbook.pdf", 9000)

    summary = env.service.summary()

    segments = {s["id"]: s for s in summary["map"]}
    assert segments["images"]["size_bytes"] == 6000
    assert segments["images"]["reclaimable_bytes"] == 5000
    assert segments["documents"]["size_bytes"] == 9000
    assert sum(s["size_bytes"] for s in summary["map"]) == summary["totals"]["storage_bytes"]
    assert summary["totals"]["orphan_bytes"] == 5400
    assert summary["disk"]["free_bytes"] > 0

    checks = {c["id"]: c for c in summary["health"]}
    assert checks["databases"]["ok"] is False  # no Chroma file exists in this fixture
    assert checks["orphans"]["ok"] is False
    assert checks["redis"]["ok"] is False
    assert any(o["id"] == "orphans" for o in summary["observations"])
    # No baseline yet, so growth is unknown rather than zero.
    assert _store(summary, "page_images")["growth"] == {"d1": None, "d7": None, "d30": None}


def test_action_is_audited_with_before_and_after(env):
    _seed_images_and_cache(env)

    entry = env.service.run_action("page_images", "remove_orphaned")

    assert entry["status"] == "success"
    assert (entry["before_bytes"], entry["after_bytes"], entry["freed_bytes"]) == (6000, 1000, 5000)
    assert entry["initiated_by"] == "manual" and entry["safety"] == "SAFE"
    assert entry["duration_ms"] >= 0 and entry["id"].startswith("act_")

    memory_only = env.service.run_action("retrieval_cache", "clear")
    assert memory_only["before_bytes"] is None and memory_only["after_bytes"] is None


def test_failed_action_is_audited_and_releases_the_lock(env, monkeypatch):
    def boom(*_args):
        raise RuntimeError("disk on fire")

    spec = env.service._actions[("vision_cache", "clear")]
    monkeypatch.setitem(
        env.service._actions,
        ("vision_cache", "clear"),
        module.ActionSpec(spec.id, spec.label, spec.description, boom, safety=spec.safety),
    )

    with pytest.raises(RuntimeError, match="disk on fire"):
        env.service.run_action("vision_cache", "clear")

    failed = env.service.history()["actions"][-1]
    assert (failed["status"], failed["error"]) == ("failed", "disk on fire")
    # The lock is released, so the next action is not refused.
    assert env.service.run_action("retrieval_cache", "clear")["status"] == "success"


def test_second_cleanup_is_refused_while_one_runs(env):
    assert env.service._op_lock.acquire(blocking=False)
    try:
        with pytest.raises(StorageBusyError, match="Another storage operation"):
            env.service.run_action("retrieval_cache", "clear")
    finally:
        env.service._op_lock.release()


def test_preview_reports_what_an_age_cutoff_would_remove(env):
    _write(env.root / "vision_cache" / f"{LIVE_DOC}_p1_aaaaaaaaaaaaaaaa_{MODEL}.json", 100, age_days=40)
    _write(env.root / "vision_cache" / f"{LIVE_DOC}_p2_dddddddddddddddd_{MODEL}.json", 300)
    _write(env.logs / "app.log", 500, age_days=90)
    _write(env.logs / "old_report.json", 700, age_days=90)
    connection = sqlite3.connect(env.telemetry.db.db_path)
    with connection:
        for index in range(10):
            stamp = "2020-01-01T00:00:00+00:00" if index < 4 else "2999-01-01T00:00:00+00:00"
            connection.execute(
                "INSERT INTO cache_events (id, timestamp, cache_type, event_type, latency_ms) VALUES (?, ?, 'x', 'HIT', 1.0)",
                (f"evt_{index}", stamp),
            )
    connection.close()

    vision = env.service.preview("vision_cache", "older_than", older_than_days=30)
    assert (vision["affected_items"], vision["estimated_bytes"], vision["exact"]) == (1, 100, True)
    assert vision["safety"] == "REBUILDABLE" and "25 seconds" in vision["impact"]["performance"]

    logs = env.service.preview("logs", "older_than", older_than_days=30)
    assert (logs["affected_items"], logs["estimated_bytes"]) == (1, 700)

    telemetry = env.service.preview("telemetry_db", "older_than", older_than_days=30)
    assert telemetry["affected_items"] == 4 and telemetry["exact"] is False

    # A preview deletes nothing.
    assert len(list((env.root / "vision_cache").glob("*.json"))) == 2
    assert env.telemetry.db.table_counts()["cache_events"] == 10
    with pytest.raises(ValueError, match="older_than_days"):
        env.service.preview("logs", "older_than")
    with pytest.raises(UnknownStorageActionError):
        env.service.preview("uploads", "delete")


def test_cleanup_plan_preselects_only_what_loses_nothing(env):
    _seed_images_and_cache(env)
    _write(env.logs / "old_report.json", 700, age_days=90)

    plan = env.service.cleanup_plan()
    items = {(i["store_id"], i["action_id"]): i for i in plan["items"]}

    assert items[("page_images", "remove_orphaned")]["selected"] is True
    assert items[("page_images", "remove_orphaned")]["estimated_bytes"] == 5000
    assert items[("vision_cache", "clear")]["selected"] is False
    assert items[("logs", "older_than")]["default_days"] == 30
    assert items[("logs", "older_than")]["estimated_bytes"] == 700
    assert items[("vision_cache", "remove_orphaned")]["estimated_bytes"] == 400
    assert plan["safe_reclaimable_bytes"] == sum(i["estimated_bytes"] for i in plan["items"] if i["selected"])
    # Loading and unloading models is a runtime control, not a cleanup.
    assert not any(store == "ollama_models" for store, _ in items)
    assert all(i["selected"] is False for i in plan["items"] if i["safety"] != "SAFE")


def test_selective_cleanup_runs_exactly_the_chosen_items(env):
    _seed_images_and_cache(env)
    old_report = _write(env.logs / "old_report.json", 700, age_days=90)

    result = env.service.clean_safe(
        [
            {"store_id": "page_images", "action_id": "remove_orphaned"},
            {"store_id": "logs", "action_id": "older_than", "older_than_days": 30},
        ]
    )

    assert result["freed_bytes"] == 5700
    assert not old_report.exists()
    assert {a["initiated_by"] for a in env.service.history()["actions"]} == {"cleanup"}
    # The orphaned vision cache page was not selected, so it is still there.
    assert (env.root / "vision_cache" / f"{ORPHAN_DOC}_p1_bbbbbbbbbbbbbbbb_{MODEL}.json").is_file()

    with pytest.raises(UnknownStorageActionError):
        env.service.clean_safe([{"store_id": "uploads", "action_id": "delete"}])
    with pytest.raises(UnknownStorageActionError, match="not a cleanup"):
        env.service.clean_safe([{"store_id": "ollama_models", "action_id": "unload"}])
    with pytest.raises(ValueError, match="older_than_days"):
        env.service.clean_safe([{"store_id": "logs", "action_id": "older_than"}])


def test_documents_report_footprint_and_orphans(env):
    _seed_images_and_cache(env)
    _write(env.root / "images" / LIVE_DOC / "assets.json", 20)
    env.docs.storage_records = lambda: [
        {
            "document_id": LIVE_DOC,
            "filename": "handbook.pdf",
            "file_size_bytes": 9000,
            "chunk_count": 12,
            "pages_count": 3,
            "created_at": "2026-09-01T00:00:00+00:00",
            "status": "READY",
        }
    ]
    orphan_chunk = SimpleNamespace(metadata=SimpleNamespace(document_id=ORPHAN_DOC))
    env.docs.bm25_index.entries = [orphan_chunk, SimpleNamespace(metadata=SimpleNamespace(document_id=LIVE_DOC))]

    report = env.service.documents()

    document = report["documents"][0]
    assert document["images"] == {"count": 1, "bytes": 1020}
    assert document["vision_cache"] == {"entries": 1, "bytes": 100}
    assert (document["chunks"], document["embeddings"]) == (12, 12)
    # No vectors are stored in this fixture, so the index share is unknown, not zero.
    assert document["index_bytes_estimated"] is None
    assert document["total_bytes_estimated"] == 9000 + 1020 + 100
    assert document["last_queried"] is None and document["query_count"] == 0

    orphans = report["orphans"]
    assert orphans["images"] == {
        "count": 1,
        "files": 2,
        "bytes": 5000,
        "items": [{"name": ORPHAN_DOC, "bytes": 5000, "files": 2}],
        "store_id": "page_images",
        "action_id": "remove_orphaned",
        "safety": "SAFE",
    }
    assert (orphans["vision_cache"]["count"], orphans["vision_cache"]["bytes"]) == (1, 400)
    assert orphans["total_bytes"] == 5400
    assert orphans["chunk_references"]["bm25"] == 1
    assert orphans["chunk_references"]["vector"] is None  # only counted on a deep scan


def test_live_attributes_vram_to_what_each_runtime_reports(env, monkeypatch):
    gib = 1024**3
    monkeypatch.setattr(
        module,
        "list_loaded_models",
        lambda **_: [
            {
                "name": "qwen2.5:7b",
                "size": 5 * gib,
                "size_vram": 4 * gib,
                "context_length": 4096,
                "expires_at": "2319-01-01T00:00:00Z",
                "details": {"parameter_size": "7.6B", "quantization_level": "Q4_K_M"},
            }
        ],
    )
    monkeypatch.setattr(
        module, "gpu_memory_mb", lambda *_: {"name": "RTX 4050", "total_mb": 6144.0, "used_mb": 5120.0, "free_mb": 1024.0}
    )
    monkeypatch.setattr(module.runtime, "torch_reserved_bytes", lambda: None)
    env.docs.busy = True

    live = env.service.live()

    model = live["loaded_models"][0]
    assert (model["vram_bytes"], model["ram_bytes"], model["gpu_pct"]) == (4 * gib, gib, 80.0)
    assert model["pinned"] is True and model["quantization"] == "Q4_K_M"
    # Ollama does not say when a model was loaded; that stays unknown.
    assert model["loaded_at"] is None and model["requests"] is None

    segments = {s["id"]: s["bytes"] for s in live["gpu_breakdown"]["segments"]}
    assert segments == {"ollama:qwen2.5:7b": 4 * gib, "other": gib}
    assert live["operations"]["blocks_index_cleanup"] is True
    assert live["caches"]["retrieval_cache"]["hits"] is not None
    assert live["caches"]["kv_cache"]["backend"] == "memory"


def test_live_without_gpu_or_models(env):
    live = env.service.live()

    assert live["gpu"] is None and live["gpu_breakdown"] is None
    assert live["loaded_models"] == []
    assert live["operations"] == {"indexing": [], "cleanup": None, "blocks_index_cleanup": False}


def test_inspect_describes_a_store_and_rejects_unknown_ids(env):
    connection = sqlite3.connect(env.telemetry.db.db_path)
    with connection:
        connection.execute(
            "INSERT INTO cache_events (id, timestamp, cache_type, event_type, latency_ms) "
            "VALUES ('e1', '2026-01-02T00:00:00+00:00', 'x', 'HIT', 1.0)"
        )
    connection.close()

    telemetry = env.service.inspect("telemetry_db")
    events = next(t for t in telemetry["tables"] if t["name"] == "cache_events")
    assert (events["rows"], events["oldest"]) == (1, "2026-01-02T00:00:00+00:00")
    assert telemetry["filesystem"]["path"].endswith("telemetry.sqlite3")
    assert telemetry["filesystem"]["rebuildable"] is False

    cache = env.service.inspect("vision_cache")
    assert cache["filesystem"]["rebuildable"] is True and cache["info"]["rebuild"]
    assert env.service.inspect("kv_cache")["filesystem"] is None

    with pytest.raises(UnknownStorageActionError):
        env.service.inspect("../../etc")


def test_history_window_returns_series_events_and_no_premature_forecast(env):
    _seed_images_and_cache(env)
    env.service.summary()
    env.service.run_action("page_images", "remove_orphaned")

    report = env.service.history(window="7d")

    assert [p["categories"]["images"] for p in report["series"]] == [6000, 1000]
    assert all(p["app_bytes"] >= p["categories"]["images"] for p in report["series"])
    assert [e["kind"] for e in report["events"]] == ["cleanup"]
    assert report["events"][0]["bytes"] == -5000
    # Two snapshots a moment apart cannot support a projection.
    assert report["forecast"] is None
    with pytest.raises(ValueError, match="range"):
        env.service.history(window="1y")


def test_cache_counters_track_hits_and_misses():
    from backend.embeddings.embeddings import EmbeddingCache
    from backend.retrieval.retrieval_cache import RetrievalCache

    retrieval = RetrievalCache()
    assert retrieval.get("q") is None
    retrieval.set("q", [])
    assert retrieval.get("q") == []
    stats = retrieval.stats()
    assert (stats["entries"], stats["hits"], stats["misses"]) == (1, 1, 1)
    assert stats["last_hit_at"] is not None

    embedding = EmbeddingCache()
    assert embedding.get("text") is None
    embedding.set("text", [0.1, 0.2, 0.3])
    assert embedding.get("text") == [0.1, 0.2, 0.3]
    stats = embedding.stats()
    assert (stats["hits"], stats["misses"], stats["vector_dim"]) == (1, 1, 3)

    kv = RedisCache(enabled=False)
    assert kv.get("missing") is None
    kv.set("k", "v")
    assert kv.get("k") == "v"
    stats = kv.stats()
    assert (stats["backend"], stats["keys"], stats["hits"], stats["misses"]) == ("memory", 1, 1, 1)


def test_console_api(env):
    from backend.api.dependencies import get_storage_service
    from backend.api.main import create_app

    _seed_images_and_cache(env)
    app = create_app()
    app.dependency_overrides[get_storage_service] = lambda: env.service
    client = TestClient(app)

    assert client.get("/api/admin/storage/live").json()["operations"]["cleanup"] is None
    assert client.get("/api/admin/storage/documents").json()["orphans"]["total_bytes"] == 5400
    assert client.get("/api/admin/storage?refresh=true").status_code == 200

    assert client.get("/api/admin/storage/vision_cache/inspect").json()["id"] == "vision_cache"
    assert client.get("/api/admin/storage/nope/inspect").status_code == 404

    preview = client.post("/api/admin/storage/page_images/remove_orphaned/preview")
    assert preview.status_code == 200 and preview.json()["estimated_bytes"] == 5000
    assert client.post("/api/admin/storage/logs/older_than/preview", json={}).status_code == 400
    assert client.post("/api/admin/storage/uploads/delete/preview").status_code == 404

    plan = client.get("/api/admin/storage/cleanup/plan").json()
    assert plan["safe_reclaimable_bytes"] >= 5400

    # Paths and arbitrary ids are rejected before they reach the service.
    bad = client.post("/api/admin/storage/cleanup", json={"items": [{"store_id": "../etc", "action_id": "x"}]})
    assert bad.status_code == 422
    unknown = client.post("/api/admin/storage/cleanup", json={"items": [{"store_id": "uploads", "action_id": "delete"}]})
    assert unknown.status_code == 404

    chosen = client.post(
        "/api/admin/storage/cleanup",
        json={"items": [{"store_id": "vision_cache", "action_id": "remove_orphaned"}]},
    )
    assert chosen.status_code == 200 and chosen.json()["freed_bytes"] == 400
    assert (env.root / "images" / ORPHAN_DOC).exists()

    windowed = client.get("/api/admin/storage/history?range=24h").json()
    assert {"series", "events", "forecast", "actions"} <= windowed.keys()
    assert client.get("/api/admin/storage/history?range=1y").status_code == 422
