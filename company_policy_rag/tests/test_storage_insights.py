"""Growth, forecast, events and observations computed from synthetic snapshots."""

from __future__ import annotations

from datetime import UTC, datetime

from backend.services import storage_insights as insights

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC).timestamp()
MIB = 1024 * 1024


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).isoformat()


def _snapshot(age_days: float, models: int | None = None, **sizes: int) -> dict:
    entry = {"type": "snapshot", "ts": _iso(NOW - age_days * insights.DAY), "sizes": sizes}
    if models is not None:
        entry["models"] = models
    return entry


def test_growth_needs_a_baseline_old_enough_for_the_window():
    snapshots = [_snapshot(8, vision_cache=10 * MIB), _snapshot(1.1, vision_cache=40 * MIB)]

    growth = insights.growth(snapshots, {"vision_cache": 100 * MIB, "logs": 5}, NOW)

    assert growth["vision_cache"] == {"d1": 60 * MIB, "d7": 90 * MIB, "d30": None}
    # A store no snapshot mentions has no baseline at all.
    assert growth["logs"] == {"d1": None, "d7": None, "d30": None}


def test_growth_ignores_a_baseline_far_older_than_the_window():
    snapshots = [_snapshot(20, vision_cache=10 * MIB)]

    growth = insights.growth(snapshots, {"vision_cache": 100 * MIB}, NOW)

    # Twenty days ago says nothing about the last 24 hours or the last week.
    assert growth["vision_cache"] == {"d1": None, "d7": None, "d30": None}


def test_total_growth_does_not_compare_against_snapshots_without_models():
    snapshots = [_snapshot(1.2, uploads=100), _snapshot(1.1, models=5000, uploads=100)]

    assert insights.total_growth(snapshots, 150, NOW, include_models=False)["d1"] == 50
    assert insights.total_growth(snapshots, 5300, NOW, include_models=True)["d1"] == 200
    assert insights.total_growth(snapshots[:1], 5300, NOW, include_models=True)["d1"] is None


def test_forecast_needs_three_days_and_enough_points():
    short = [_snapshot(age, uploads=1000) for age in (1.0, 0.75, 0.5, 0.25, 0.0)]
    assert insights.forecast(short, NOW) is None

    sparse = [_snapshot(age, uploads=1000) for age in (5.0, 0.0)]
    assert insights.forecast(sparse, NOW) is None

    # 10 MiB per day for five days, sampled every eight hours.
    steady = [_snapshot(5 - step / 3, uploads=int(step / 3 * 10 * MIB)) for step in range(16)]
    forecast = insights.forecast(steady, NOW, disk_free_bytes=123)

    assert forecast is not None
    assert abs(forecast["bytes_per_day"] - 10 * MIB) < MIB * 0.01
    assert abs(forecast["in_7_days_bytes"] - (forecast["current_bytes"] + 70 * MIB)) < MIB
    assert forecast["disk_free_bytes"] == 123 and forecast["basis_days"] == 5.0


def test_series_groups_stores_into_map_categories():
    snapshots = [_snapshot(1, models=900, vision_cache=10, page_images=20, uploads=5, mystery=1)]

    points = insights.series(snapshots, {"vision_cache": "caches", "page_images": "images", "uploads": "documents"})

    assert points[0]["app_bytes"] == 36
    assert points[0]["models_bytes"] == 900
    assert points[0]["categories"] == {"caches": 10, "images": 20, "documents": 5, "other": 1, "models": 900}


def test_within_thins_evenly_and_keeps_the_latest():
    snapshots = [_snapshot(10 - index / 100, uploads=index) for index in range(1000)]

    picked = insights.within(snapshots, NOW - 5 * insights.DAY, max_points=50)

    assert 50 <= len(picked) <= 51
    assert picked[-1]["sizes"]["uploads"] == 999
    assert all(insights.parse_ts(s["ts"]) >= NOW - 5 * insights.DAY for s in picked)


def test_events_explain_what_changed_storage():
    snapshots = [_snapshot(2.01, uploads=100), _snapshot(1.99, uploads=600)]
    actions = [
        {"ts": _iso(NOW - 1 * insights.DAY), "store": "vector_index", "action": "compact", "freed_bytes": 300, "message": "Vector index compacted."},
        {"ts": _iso(NOW - 1 * insights.DAY), "store": "kv_cache", "action": "clear", "freed_bytes": 0, "message": "Cleared 3 keys."},
        {"ts": _iso(NOW - 0.5 * insights.DAY), "store": "logs", "action": "older_than", "freed_bytes": 9, "status": "failed"},
        # Unloading a model frees VRAM, not disk, so it is not a storage event.
        {"ts": _iso(NOW - 0.4 * insights.DAY), "store": "ollama_models", "action": "unload", "freed_bytes": 4000, "message": "Unloaded."},
    ]
    ingestions = [
        {"timestamp": _iso(NOW - 2 * insights.DAY), "filename": "guide.pdf", "file_size_bytes": 482, "chunks_count": 638, "status": "READY"},
        {"timestamp": _iso(NOW - 2 * insights.DAY), "filename": "broken.pdf", "file_size_bytes": 1, "chunks_count": 0, "status": "FAILED"},
        {"timestamp": _iso(NOW - 40 * insights.DAY), "filename": "ancient.pdf", "file_size_bytes": 1, "chunks_count": 1, "status": "READY"},
    ]
    models = [{"name": "qwen2.5:7b", "modified_at": _iso(NOW - 3 * insights.DAY), "size_bytes": 4700}]

    events = insights.events(snapshots, actions, ingestions, models, since=NOW - 7 * insights.DAY)

    assert [(e["kind"], e["label"]) for e in events] == [
        ("model", "qwen2.5:7b downloaded"),
        ("document", "guide.pdf indexed"),
        ("compaction", "Vector index compacted."),
    ]
    document = events[1]
    assert document["large"] is True and document["detail"] == "638 chunks"
    # Snapshots on both sides of the ingestion show what it actually added.
    assert document["measured_delta_bytes"] == 500
    assert events[2]["bytes"] == -300 and events[2]["measured_delta_bytes"] is None


def _store(store_id: str, label: str, size: int, group: str = "caches", **details) -> dict:
    return {"id": store_id, "label": label, "size_bytes": size, "group": group, "details": details}


def test_observations_state_only_what_was_measured():
    stores = [
        _store("vision_cache", "Vision cache", 44 * MIB),
        _store("logs", "Logs", 12 * MIB),
        _store("vector_index", "Chroma vector database", 900 * MIB, "databases", bloat_pct=31.0, free_bytes=280 * MIB),
        _store("telemetry_db", "SQLite telemetry database", 50 * MIB, "databases", bloat_pct=2.0, free_bytes=MIB),
    ]
    growth = {
        "vision_cache": {"d1": 34 * MIB, "d7": None},
        "logs": {"d1": 6 * MIB, "d7": None},
        "vector_index": {"d1": 0, "d7": 0},
        "telemetry_db": {"d1": None, "d7": None},
    }

    found = {
        o["id"]: o
        for o in insights.observations(
            stores=stores,
            store_growth=growth,
            orphan_bytes=482 * MIB,
            orphan_items=842,
            gpu={"total_mb": 6144.0, "used_mb": 5600.0},
            kv_backend="memory",
        )
    }

    assert found["top_grower"]["text"] == (
        "Vision cache grew 340% (34.0 MB) in the last 24 hours, 85% of all growth in that time."
    )
    assert found["bloat_vector_index"]["level"] == "warn" and "280.0 MB (31%)" in found["bloat_vector_index"]["text"]
    assert "bloat_telemetry_db" not in found
    assert "482.0 MB" in found["orphans"]["text"] and "842 items" in found["orphans"]["text"]
    assert found["vram"]["level"] == "warn" and "91%" in found["vram"]["text"]
    assert "kv_fallback" in found


def test_a_tidy_system_produces_no_observations():
    stores = [_store("vector_index", "Chroma vector database", 10 * MIB, "databases", bloat_pct=0.0, free_bytes=0)]

    found = insights.observations(
        stores=stores,
        store_growth={"vector_index": {"d1": None, "d7": None}},
        orphan_bytes=0,
        orphan_items=0,
        gpu=None,
        kv_backend="redis",
    )

    assert found == []


def test_health_lists_measured_conditions_without_a_score():
    stores = [
        _store("vector_index", "Chroma vector database", 1, "databases", bloat_pct=31.0, page_count=10),
        _store("telemetry_db", "SQLite telemetry database", 1, "databases", bloat_pct=1.0, page_count=10),
    ]

    checks = {
        c["id"]: c
        for c in insights.health(
            stores=stores,
            busy=False,
            kv_backend="redis",
            orphan_bytes=0,
            orphan_chunk_refs=0,
            gpu=None,
            embedding_fallback=False,
        )
    }

    assert all(set(c) == {"id", "ok", "label"} for c in checks.values())
    assert checks["databases"]["ok"] and checks["redis"]["ok"] and checks["chunk_refs"]["ok"]
    assert checks["fragmentation_vector_index"] == {
        "id": "fragmentation_vector_index",
        "ok": False,
        "label": "Chroma vector database fragmentation: 31%",
    }
    assert checks["fragmentation_telemetry_db"]["ok"] is True
    # No GPU was measured, so there is no GPU line.
    assert "vram" not in checks and "embedding" not in checks
