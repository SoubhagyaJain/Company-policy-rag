"""Growth, observations and health derived from storage measurements.

Pure functions over the recorded snapshots and the current summary, so they can be
tested without touching the filesystem. Every statement they produce is backed by
a number passed in; there is no scoring.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

DAY = 86400.0
WINDOWS: dict[str, float] = {"d1": DAY, "d7": 7 * DAY, "d30": 30 * DAY}
_MIB = 1024 * 1024
# A baseline older than this multiple of the window says little about the window.
_BASELINE_SLACK = 1.5
_BLOAT_WARN_PCT = 20.0
_VRAM_WARN_PCT = 90.0
_VRAM_CRITICAL_PCT = 97.0
_FORECAST_MIN_DAYS = 3.0
_FORECAST_MIN_POINTS = 12
_EVENT_DELTA_SLACK = 3600.0
_LARGE_INGESTION_CHUNKS = 500
_LARGE_INGESTION_BYTES = 20 * _MIB
# Stores that live in memory: clearing or unloading them does not change disk use.
_RUNTIME_STORES = frozenset(
    {"ollama_models", "vision_model", "kv_cache", "retrieval_cache", "embedding_cache", "conversations"}
)


def parse_ts(value: Any) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def format_bytes(value: float) -> str:
    size = float(abs(value))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _timed(snapshots: list[dict[str, Any]]) -> list[tuple[float, dict[str, Any]]]:
    timed = [(parse_ts(s.get("ts")), s) for s in snapshots]
    return sorted(((ts, s) for ts, s in timed if ts is not None), key=lambda item: item[0])


def snapshot_total(snapshot: dict[str, Any], include_models: bool = True) -> int:
    total = sum(int(v or 0) for v in (snapshot.get("sizes") or {}).values())
    return total + (int(snapshot.get("models") or 0) if include_models else 0)


def _baseline(timed: list[tuple[float, dict[str, Any]]], now: float, window: float) -> dict[str, Any] | None:
    """The latest snapshot taken at least ``window`` ago, unless it is far older than that."""
    cutoff = now - window
    candidate = None
    for ts, snapshot in timed:
        if ts > cutoff:
            break
        candidate = (ts, snapshot)
    if candidate is None or now - candidate[0] > window * _BASELINE_SLACK:
        return None
    return candidate[1]


def growth(
    snapshots: list[dict[str, Any]],
    current: dict[str, int],
    now: float,
) -> dict[str, dict[str, int | None]]:
    """Bytes each store gained over the last day, week and month; ``None`` without a baseline."""
    timed = _timed(snapshots)
    result: dict[str, dict[str, int | None]] = {store_id: {} for store_id in current}
    for name, window in WINDOWS.items():
        baseline = _baseline(timed, now, window)
        sizes = (baseline or {}).get("sizes") or {}
        for store_id, size in current.items():
            result[store_id][name] = size - int(sizes[store_id]) if baseline is not None and store_id in sizes else None
    return result


def total_growth(
    snapshots: list[dict[str, Any]],
    current_total: int,
    now: float,
    include_models: bool,
) -> dict[str, int | None]:
    timed = _timed(snapshots)
    result: dict[str, int | None] = {}
    for name, window in WINDOWS.items():
        baseline = _baseline(timed, now, window)
        # Model bytes only entered snapshots later; never compare against one without them.
        if baseline is None or (include_models and "models" not in baseline):
            result[name] = None
        else:
            result[name] = current_total - snapshot_total(baseline, include_models)
    return result


def history_span_seconds(snapshots: list[dict[str, Any]], now: float) -> float:
    timed = _timed(snapshots)
    return now - timed[0][0] if timed else 0.0


def within(snapshots: list[dict[str, Any]], since: float | None, max_points: int = 240) -> list[dict[str, Any]]:
    """Snapshots taken since ``since``, oldest first, evenly thinned to ``max_points``."""
    picked = [s for ts, s in _timed(snapshots) if since is None or ts >= since]
    if len(picked) <= max_points:
        return picked
    step = len(picked) / max_points
    thinned = [picked[int(i * step)] for i in range(max_points)]
    if thinned[-1] is not picked[-1]:
        thinned.append(picked[-1])
    return thinned


def series(snapshots: list[dict[str, Any]], category_of: dict[str, str]) -> list[dict[str, Any]]:
    """Chart points: app bytes, model bytes and per-category bytes for each snapshot."""
    points = []
    for snapshot in snapshots:
        categories: dict[str, int] = {}
        for store_id, size in (snapshot.get("sizes") or {}).items():
            category = category_of.get(store_id, "other")
            categories[category] = categories.get(category, 0) + int(size or 0)
        models = snapshot.get("models")
        if models is not None:
            categories["models"] = int(models)
        points.append(
            {
                "ts": snapshot["ts"],
                "app_bytes": snapshot_total(snapshot, include_models=False),
                "models_bytes": int(models) if models is not None else None,
                "categories": categories,
            }
        )
    return points


def forecast(snapshots: list[dict[str, Any]], now: float, disk_free_bytes: int | None = None) -> dict[str, Any] | None:
    """Linear fit of app data over the recorded snapshots; ``None`` without enough history."""
    points = [(ts, snapshot_total(s, include_models=False)) for ts, s in _timed(snapshots)]
    if len(points) < _FORECAST_MIN_POINTS:
        return None
    span = points[-1][0] - points[0][0]
    if span < _FORECAST_MIN_DAYS * DAY:
        return None
    count = len(points)
    mean_t = sum(t for t, _ in points) / count
    mean_y = sum(y for _, y in points) / count
    variance = sum((t - mean_t) ** 2 for t, _ in points)
    if variance == 0:
        return None
    slope = sum((t - mean_t) * (y - mean_y) for t, y in points) / variance
    current = points[-1][1]
    return {
        "basis_days": round(span / DAY, 1),
        "basis_points": count,
        "bytes_per_day": int(slope * DAY),
        "current_bytes": current,
        "in_7_days_bytes": max(0, int(current + slope * 7 * DAY)),
        "in_30_days_bytes": max(0, int(current + slope * 30 * DAY)),
        "disk_free_bytes": disk_free_bytes,
        "method": "Linear fit of app data over the recorded snapshots. Model files are excluded.",
    }


def _delta_around(points: list[tuple[float, int]], ts: float) -> int | None:
    before = [p for p in points if ts - _EVENT_DELTA_SLACK <= p[0] <= ts]
    after = [p for p in points if ts < p[0] <= ts + _EVENT_DELTA_SLACK]
    if not before or not after:
        return None
    return after[0][1] - before[-1][1]


def events(
    snapshots: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    ingestions: list[dict[str, Any]],
    models: list[dict[str, Any]],
    since: float | None = None,
) -> list[dict[str, Any]]:
    """What changed storage: indexed documents, cleanups, compactions and model downloads."""
    points = [(ts, snapshot_total(s, include_models=False)) for ts, s in _timed(snapshots)]
    found: list[dict[str, Any]] = []

    for action in actions:
        if action.get("status") == "failed" or action.get("store") in _RUNTIME_STORES:
            continue
        freed = int(action.get("freed_bytes") or 0)
        compaction = action.get("action") == "compact"
        if not freed and not compaction:
            continue
        found.append(
            {
                "ts": action.get("ts"),
                "kind": "compaction" if compaction else "cleanup",
                "label": action.get("message") or f"{action.get('store')}/{action.get('action')}",
                "bytes": -freed,
                "store_id": action.get("store"),
            }
        )

    for item in ingestions:
        if str(item.get("status", "")).upper() == "FAILED":
            continue
        size = int(item.get("file_size_bytes") or 0)
        chunks = int(item.get("chunks_count") or 0)
        found.append(
            {
                "ts": item.get("timestamp"),
                "kind": "document",
                "label": f"{item.get('filename')} indexed",
                "detail": f"{chunks:,} chunks",
                "bytes": size,
                "large": chunks >= _LARGE_INGESTION_CHUNKS or size >= _LARGE_INGESTION_BYTES,
                "store_id": "uploads",
            }
        )

    for model in models:
        if not model.get("modified_at"):
            continue
        found.append(
            {
                "ts": model["modified_at"],
                "kind": "model",
                "label": f"{model.get('name') or model.get('label')} downloaded",
                "bytes": int(model.get("size_bytes") or 0),
            }
        )

    result = []
    for event in found:
        ts = parse_ts(event.get("ts"))
        if ts is None or (since is not None and ts < since):
            continue
        event["measured_delta_bytes"] = _delta_around(points, ts)
        result.append((ts, event))
    return [event for _ts, event in sorted(result, key=lambda item: item[0])]


def observations(
    *,
    stores: list[dict[str, Any]],
    store_growth: dict[str, dict[str, int | None]],
    orphan_bytes: int,
    orphan_items: int,
    gpu: dict[str, Any] | None,
    kv_backend: str,
) -> list[dict[str, Any]]:
    """Concrete findings worth a line on the page. Empty when there is nothing to say."""
    found: list[dict[str, Any]] = []

    gains = {s["id"]: int(store_growth.get(s["id"], {}).get("d1") or 0) for s in stores}
    positive = {store_id: gain for store_id, gain in gains.items() if gain > 0}
    if positive:
        top_id = max(positive, key=lambda store_id: positive[store_id])
        top_gain = positive[top_id]
        if top_gain >= _MIB:
            store = next(s for s in stores if s["id"] == top_id)
            previous = store["size_bytes"] - top_gain
            text = f"{store['label']} grew {format_bytes(top_gain)} in the last 24 hours"
            if previous > 0 and top_gain >= previous:
                text = f"{store['label']} grew {round(100 * top_gain / previous)}% ({format_bytes(top_gain)}) in the last 24 hours"
            if len(positive) > 1:
                text += f", {round(100 * top_gain / sum(positive.values()))}% of all growth in that time"
            found.append({"id": "top_grower", "level": "info", "text": text + ".", "store_id": top_id})

    for store in stores:
        weekly = store_growth.get(store["id"], {}).get("d7")
        if store["id"] == "telemetry_db" and weekly and weekly / 7 >= 10 * _MIB:
            found.append(
                {
                    "id": "telemetry_rate",
                    "level": "info",
                    "text": f"The telemetry database is growing about {format_bytes(weekly / 7)} per day.",
                    "store_id": store["id"],
                }
            )
        details = store.get("details") or {}
        bloat = float(details.get("bloat_pct") or 0)
        free = int(details.get("free_bytes") or 0)
        if store.get("group") == "databases" and bloat >= _BLOAT_WARN_PCT and free >= _MIB:
            found.append(
                {
                    "id": f"bloat_{store['id']}",
                    "level": "warn",
                    "text": f"{store['label']}: {format_bytes(free)} ({bloat:g}%) is unused space that compaction can release.",
                    "store_id": store["id"],
                }
            )

    if orphan_bytes > 0:
        found.append(
            {
                "id": "orphans",
                "level": "warn",
                "text": f"{format_bytes(orphan_bytes)} of orphaned artifacts in {orphan_items:,} items: "
                "their source documents are no longer in the library.",
                "store_id": "page_images",
            }
        )

    if gpu and gpu.get("total_mb"):
        pct = 100 * gpu["used_mb"] / gpu["total_mb"]
        if pct >= _VRAM_WARN_PCT:
            found.append(
                {
                    "id": "vram",
                    "level": "critical" if pct >= _VRAM_CRITICAL_PCT else "warn",
                    "text": f"GPU memory is {pct:.0f}% used "
                    f"({gpu['used_mb'] / 1024:.1f} of {gpu['total_mb'] / 1024:.1f} GB).",
                    "store_id": "ollama_models",
                }
            )

    if kv_backend != "redis":
        found.append(
            {
                "id": "kv_fallback",
                "level": "info",
                "text": "The application KV cache is using its in-memory fallback, so its entries are lost "
                "when the backend restarts.",
                "store_id": "kv_cache",
            }
        )
    return found


def health(
    *,
    stores: list[dict[str, Any]],
    busy: bool,
    kv_backend: str,
    orphan_bytes: int,
    orphan_chunk_refs: int | None,
    gpu: dict[str, Any] | None,
    embedding_fallback: bool | None,
) -> list[dict[str, Any]]:
    """Measured conditions, each either holding or not. No aggregate score."""
    checks: list[dict[str, Any]] = []
    databases = [s for s in stores if s.get("group") == "databases"]
    unreadable = [s["label"] for s in databases if (s.get("details") or {}).get("page_count") is None]
    checks.append(
        {
            "id": "databases",
            "ok": not unreadable,
            "label": "Databases accessible" if not unreadable else f"Cannot read: {', '.join(unreadable)}",
        }
    )
    checks.append(
        {
            "id": "conflicts",
            "ok": not busy,
            "label": "No active cleanup conflicts" if not busy else "Indexing is running: index cleanups wait for it",
        }
    )
    checks.append(
        {
            "id": "redis",
            "ok": kv_backend == "redis",
            "label": "Redis connected" if kv_backend == "redis" else "Redis not connected: in-memory fallback in use",
        }
    )
    checks.append(
        {
            "id": "orphans",
            "ok": orphan_bytes == 0,
            "label": "No orphaned artifacts" if orphan_bytes == 0 else f"{format_bytes(orphan_bytes)} orphaned artifacts",
        }
    )
    if orphan_chunk_refs is not None:
        checks.append(
            {
                "id": "chunk_refs",
                "ok": orphan_chunk_refs == 0,
                "label": "No orphaned chunk references"
                if orphan_chunk_refs == 0
                else f"{orphan_chunk_refs:,} chunk references without a library document",
            }
        )
    for store in databases:
        details = store.get("details") or {}
        if details.get("bloat_pct") is None:
            continue
        bloat = float(details["bloat_pct"])
        checks.append(
            {
                "id": f"fragmentation_{store['id']}",
                "ok": bloat < _BLOAT_WARN_PCT,
                "label": f"{store['label']} fragmentation: {bloat:g}%",
            }
        )
    if gpu and gpu.get("total_mb"):
        pct = 100 * gpu["used_mb"] / gpu["total_mb"]
        checks.append({"id": "vram", "ok": pct < _VRAM_WARN_PCT, "label": f"GPU VRAM usage: {pct:.0f}%"})
    if embedding_fallback:
        checks.append(
            {
                "id": "embedding",
                "ok": False,
                "label": "Embedding model unavailable: dense retrieval is degraded",
            }
        )
    return checks
