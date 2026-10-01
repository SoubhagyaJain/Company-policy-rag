"""Inventory and cleanup of everything the app persists or keeps in memory.

The Storage tab reads one summary from here and triggers cleanups by store id and
action id. Paths are never taken from the request: every store's root comes from
the running services or settings, and every delete is checked to stay inside it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from backend.embeddings.vector_store import get_shared_chroma_client
from backend.retrieval.retrieval_cache import get_retrieval_cache
from backend.utils.logging import logger
from backend.utils.redis_cache import get_redis_cache
from backend.vision.hf_vision_client import HFVisionClient, gpu_memory_mb
from backend.vision.vision_service import VisionService
from src.config import PROJECT_ROOT, settings
from src.ollama_client import list_installed_models, list_loaded_models, preload_model, unload_model

_DOC_DIR = re.compile(r"doc_[0-9a-f]{12}")
_VISION_CACHE_DOC = re.compile(r"^(doc_[0-9a-f]{12})_p\d+_")
# Files the running process holds open; deleting them under it loses the live log.
_LIVE_LOG_NAMES = frozenset({"app.log"})
_SNAPSHOT_INTERVAL_SECONDS = 15 * 60
_HISTORY_MAX_LINES = 2000
_HISTORY_KEEP_LINES = 1500
_MAX_DAYS = 3650
# Ollama reports a pinned model (keep_alive=-1) with an expiry centuries away.
_PINNED_EXPIRY_YEARS = 50


class StorageBusyError(RuntimeError):
    """A cleanup was requested while an ingestion is writing to the same stores."""


class UnknownStorageActionError(LookupError):
    """The store or action id does not exist."""


@dataclass
class ActionResult:
    freed_bytes: int = 0
    removed_items: int = 0
    message: str = ""


@dataclass(frozen=True)
class ActionSpec:
    id: str
    label: str
    # Shown in the confirmation dialog: what is removed and what it costs to get back.
    description: str
    run: Callable[[int | None, str | None], ActionResult]
    needs_days: bool = False
    needs_target: bool = False
    # Loses nothing the app still uses, so "clean up safe items" may run it.
    safe: bool = False
    blocks_on_ingestion: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "needs_days": self.needs_days,
            "needs_target": self.needs_target,
            "safe": self.safe,
        }


@dataclass
class _DirStats:
    size: int = 0
    files: int = 0
    newest: float = 0.0


def _scan(path: Path) -> _DirStats:
    """Total size, file count and newest mtime under a file or directory."""
    stats = _DirStats()
    if path.is_file():
        info = path.stat()
        return _DirStats(info.st_size, 1, info.st_mtime)
    if not path.is_dir():
        return stats
    pending = [str(path)]
    while pending:
        try:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir():
                            pending.append(entry.path)
                        else:
                            info = entry.stat()
                            stats.size += info.st_size
                            stats.files += 1
                            stats.newest = max(stats.newest, info.st_mtime)
                    except OSError:
                        continue
        except OSError:
            continue
    return stats


def _iso(timestamp: float) -> str | None:
    return datetime.fromtimestamp(timestamp, UTC).isoformat() if timestamp else None


def _is_inside(root: Path, target: Path) -> bool:
    root_resolved, target_resolved = root.resolve(), target.resolve()
    return root_resolved in target_resolved.parents


def _sqlite_stats(db_file: Path, count_tables: tuple[str, ...] = ()) -> dict[str, Any] | None:
    """Page-level health of a SQLite file, read through a read-only connection."""
    if not db_file.is_file():
        return None
    file_bytes = sum(
        candidate.stat().st_size
        for candidate in (db_file, db_file.with_name(db_file.name + "-wal"), db_file.with_name(db_file.name + "-shm"))
        if candidate.is_file()
    )
    try:
        connection = sqlite3.connect(f"{db_file.resolve().as_uri()}?mode=ro", uri=True, timeout=2.0)
    except sqlite3.Error:
        return {"file_bytes": file_bytes}
    try:
        page_size = connection.execute("PRAGMA page_size").fetchone()[0]
        page_count = connection.execute("PRAGMA page_count").fetchone()[0]
        free_pages = connection.execute("PRAGMA freelist_count").fetchone()[0]
        rows: dict[str, int] = {}
        for table in count_tables:
            try:
                rows[table] = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.Error:
                continue
        return {
            "file_bytes": file_bytes,
            "page_size": page_size,
            "page_count": page_count,
            "free_pages": free_pages,
            "free_bytes": free_pages * page_size,
            "bloat_pct": round(100 * free_pages / page_count, 1) if page_count else 0.0,
            "rows": rows,
        }
    except sqlite3.Error as exc:
        logger.debug("Could not read SQLite stats for %s: %s", db_file, exc)
        return {"file_bytes": file_bytes}
    finally:
        connection.close()


class StorageService:
    """Reports what every store holds and runs the cleanup actions offered for it."""

    def __init__(
        self,
        doc_service: Any,
        telemetry_service: Any,
        semantic_cache: Any = None,
        *,
        session_count: Callable[[], int] | None = None,
        clear_sessions: Callable[[], None] | None = None,
        logs_dir: Path | None = None,
        eval_dirs: list[Path] | None = None,
        sessions_dir: Path | None = None,
        history_path: Path | None = None,
        readonly_dirs: dict[str, tuple[str, str, Path]] | None = None,
        model_dirs: dict[str, tuple[str, Path]] | None = None,
    ) -> None:
        self.doc_service = doc_service
        self.telemetry_service = telemetry_service
        self.semantic_cache = semantic_cache
        self._session_count = session_count
        self._clear_sessions = clear_sessions

        # Settings may hold paths relative to the working directory; pin them down once
        # so what is shown, and what a cleanup deletes, cannot shift with the cwd.
        self.vector_dir = Path(doc_service.vector_store.persist_dir).resolve()
        self.images_dir = Path(doc_service.image_asset_manager.storage_dir).resolve()
        self.vision_cache_dir = Path(doc_service.vision_cache_manager.cache_dir).resolve()
        self.uploads_dir = Path(doc_service.storage_dir).resolve()
        self.bm25_dir = Path(doc_service.bm25_index.storage_dir).resolve()
        self.telemetry_file = Path(telemetry_service.db.db_path).resolve()
        self.logs_dir = Path(logs_dir or settings.logs_dir).resolve()
        storage_root = Path(settings.storage_dir).resolve()
        self.eval_dirs = (
            [Path(d).resolve() for d in eval_dirs]
            if eval_dirs is not None
            else [storage_root / "eval_corpora", storage_root / "retrieval_eval"]
        )
        self.sessions_dir = Path(sessions_dir or Path(settings.app_storage_dir) / "sessions").resolve()
        self.history_path = Path(history_path or storage_root / "storage_history.jsonl").resolve()
        self.readonly_dirs = readonly_dirs if readonly_dirs is not None else self._default_readonly_dirs()
        self.model_dirs = model_dirs if model_dirs is not None else self._default_model_dirs()

        self._lock = threading.RLock()
        self._scan_cache: dict[str, tuple[float, _DirStats]] = {}
        self._last_snapshot = 0.0
        self._actions = self._build_actions()

    # ── Defaults ────────────────────────────────────────────────────────────

    def _default_readonly_dirs(self) -> dict[str, tuple[str, str, Path]]:
        dirs = {
            "frontend_build": (
                "Frontend build cache",
                "Next.js dev/build output. Delete it only with the frontend dev server stopped.",
                PROJECT_ROOT / "frontend" / ".next",
            ),
        }
        legacy = Path(settings.chroma_persist_dir).resolve()
        if legacy != self.vector_dir:
            dirs["legacy_index"] = (
                "Legacy vector index",
                "Index used by the older Streamlit pipeline, not by this app.",
                legacy,
            )
        return dirs

    def _default_model_dirs(self) -> dict[str, tuple[str, Path]]:
        hub = Path(os.getenv("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"
        dirs = {"vision_model": (f"Vision model ({settings.vision_model})", Path(settings.vision_model_path))}
        embedding_name = getattr(self.doc_service.embedding_service, "model_name", "")
        for key, label, name in (
            ("embedding_model", "Embedding model", embedding_name),
            ("reranker_model", "Reranker model", settings.reranker_model),
        ):
            if name:
                dirs[key] = (f"{label} ({name})", hub / f"models--{name.replace('/', '--')}")
        return dirs

    # ── Scanning ────────────────────────────────────────────────────────────

    def _stats(self, path: Path, ttl: float = 5.0) -> _DirStats:
        key = str(path)
        now = time.monotonic()
        with self._lock:
            cached = self._scan_cache.get(key)
            if cached and cached[0] > now:
                return cached[1]
        stats = _scan(path)
        with self._lock:
            self._scan_cache[key] = (now + ttl, stats)
        return stats

    def _invalidate(self) -> None:
        with self._lock:
            self._scan_cache.clear()

    def _known_ids(self) -> set[str]:
        return set(self.doc_service.known_document_ids())

    def _orphan_image_dirs(self) -> list[Path]:
        if not self.images_dir.is_dir():
            return []
        known = self._known_ids()
        return [
            child
            for child in self.images_dir.iterdir()
            if child.is_dir() and _DOC_DIR.fullmatch(child.name) and child.name not in known
        ]

    def _orphan_vision_files(self) -> list[Path]:
        if not self.vision_cache_dir.is_dir():
            return []
        known = self._known_ids()
        orphans = []
        for item in self.vision_cache_dir.glob("*.json"):
            match = _VISION_CACHE_DOC.match(item.name)
            if match and match.group(1) not in known:
                orphans.append(item)
        return orphans

    def _old_session_dirs(self) -> list[Path]:
        if not self.sessions_dir.is_dir():
            return []
        active = self.uploads_dir.resolve()
        return [
            child
            for child in self.sessions_dir.iterdir()
            if child.is_dir() and child.resolve() not in active.parents and child.resolve() != active
        ]

    def _collections(self) -> list[dict[str, Any]]:
        try:
            client = get_shared_chroma_client(self.vector_dir)
            names = [c if isinstance(c, str) else c.name for c in client.list_collections()]
            return [{"name": name, "rows": client.get_collection(name).count()} for name in sorted(names)]
        except Exception as exc:
            logger.debug("Could not list Chroma collections: %s", exc)
            return []

    # ── Inventory ───────────────────────────────────────────────────────────

    def _entry(
        self,
        store_id: str,
        *,
        label: str,
        group: str,
        description: str,
        path: Path | None = None,
        size_bytes: int = 0,
        items: int | None = None,
        items_label: str = "files",
        reclaimable_bytes: int = 0,
        last_modified: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": store_id,
            "label": label,
            "group": group,
            "description": description,
            "path": str(path) if path else None,
            "size_bytes": size_bytes,
            "items": items,
            "items_label": items_label,
            "reclaimable_bytes": reclaimable_bytes,
            "last_modified": last_modified,
            "details": details or {},
            "actions": [spec.describe() for (sid, _), spec in self._actions.items() if sid == store_id],
        }

    def _stores(self) -> list[dict[str, Any]]:
        stores: list[dict[str, Any]] = []

        # Databases
        vector = self._stats(self.vector_dir)
        vector_db = _sqlite_stats(self.vector_dir / "chroma.sqlite3", ("embeddings", "embeddings_queue")) or {}
        collections = self._collections()
        stores.append(
            self._entry(
                "vector_index",
                label="Vector index",
                group="databases",
                description="Chroma database holding document embeddings and the semantic answer cache.",
                path=self.vector_dir,
                size_bytes=vector.size,
                items=sum(c["rows"] for c in collections),
                items_label="vectors",
                reclaimable_bytes=vector_db.get("free_bytes", 0),
                last_modified=_iso(vector.newest),
                details={
                    "engine": "Chroma (SQLite + HNSW)",
                    "tables": collections,
                    "queue_rows": vector_db.get("rows", {}).get("embeddings_queue"),
                    **{k: v for k, v in vector_db.items() if k != "rows"},
                },
            )
        )

        telemetry = _sqlite_stats(self.telemetry_file) or {}
        try:
            telemetry_rows = self.telemetry_service.db.table_counts()
        except Exception as exc:
            logger.debug("Could not count telemetry rows: %s", exc)
            telemetry_rows = {}
        stores.append(
            self._entry(
                "telemetry_db",
                label="Telemetry database",
                group="databases",
                description="Query traces, vision and cache events, errors and ingestion history.",
                path=self.telemetry_file,
                size_bytes=telemetry.get("file_bytes", 0),
                items=sum(telemetry_rows.values()),
                items_label="rows",
                reclaimable_bytes=telemetry.get("free_bytes", 0),
                last_modified=_iso(self._stats(self.telemetry_file).newest),
                details={
                    "engine": "SQLite (WAL)",
                    "tables": [{"name": name, "rows": rows} for name, rows in telemetry_rows.items()],
                    **telemetry,
                },
            )
        )

        # Caches and files
        semantic_rows = next(
            (c["rows"] for c in collections if c["name"] == getattr(self.semantic_cache, "collection_name", None)),
            0,
        )
        stores.append(
            self._entry(
                "semantic_cache",
                label="Semantic answer cache",
                group="caches",
                description="Previously generated answers reused for near-identical questions. Stored inside the vector index.",
                items=semantic_rows,
                items_label="answers",
            )
        )

        images = self._stats(self.images_dir)
        image_dirs = [d for d in self.images_dir.iterdir() if d.is_dir()] if self.images_dir.is_dir() else []
        orphan_dirs = self._orphan_image_dirs()
        orphan_names = {d.name for d in orphan_dirs}
        stores.append(
            self._entry(
                "page_images",
                label="Extracted page images",
                group="caches",
                description="Images pulled out of uploaded documents for visual answers.",
                path=self.images_dir,
                size_bytes=images.size,
                items=len(image_dirs),
                items_label="documents",
                reclaimable_bytes=sum(self._stats(d).size for d in orphan_dirs),
                last_modified=_iso(images.newest),
                details={
                    "orphaned": len(orphan_dirs),
                    "largest": sorted(
                        (
                            {"name": d.name, "size_bytes": self._stats(d).size, "orphaned": d.name in orphan_names}
                            for d in image_dirs
                        ),
                        key=lambda row: row["size_bytes"],
                        reverse=True,
                    )[:6],
                },
            )
        )

        vision = self._stats(self.vision_cache_dir)
        orphan_files = self._orphan_vision_files()
        stores.append(
            self._entry(
                "vision_cache",
                label="Vision cache",
                group="caches",
                description="Text the vision model already read from page images. Re-reading a page takes about 25 seconds.",
                path=self.vision_cache_dir,
                size_bytes=vision.size,
                items=vision.files,
                items_label="pages",
                reclaimable_bytes=sum(f.stat().st_size for f in orphan_files if f.is_file()),
                last_modified=_iso(vision.newest),
                details={"orphaned": len(orphan_files)},
            )
        )

        logs = self._stats(self.logs_dir)
        stores.append(
            self._entry(
                "logs",
                label="Logs and evaluation reports",
                group="caches",
                description="Application log and the reports written by evaluation runs.",
                path=self.logs_dir,
                size_bytes=logs.size,
                items=logs.files,
                last_modified=_iso(logs.newest),
                details={"largest": self._largest_children(self.logs_dir)},
            )
        )

        eval_stats = [self._stats(d, ttl=30.0) for d in self.eval_dirs]
        stores.append(
            self._entry(
                "eval_artifacts",
                label="Evaluation indexes",
                group="caches",
                description="Benchmark corpora and indexes built by the evaluation scripts.",
                path=self.eval_dirs[0].parent if self.eval_dirs else None,
                size_bytes=sum(s.size for s in eval_stats),
                items=sum(s.files for s in eval_stats),
                last_modified=_iso(max((s.newest for s in eval_stats), default=0.0)),
                details={
                    "largest": [
                        {"name": d.name, "size_bytes": s.size} for d, s in zip(self.eval_dirs, eval_stats) if s.size
                    ]
                },
            )
        )

        old_sessions = self._old_session_dirs()
        stores.append(
            self._entry(
                "session_libraries",
                label="Old session libraries",
                group="caches",
                description="Isolated libraries left by earlier runs in session mode.",
                path=self.sessions_dir,
                size_bytes=sum(self._stats(d).size for d in old_sessions),
                items=len(old_sessions),
                items_label="sessions",
            )
        )

        # Live data, managed elsewhere
        uploads = self._stats(self.uploads_dir)
        stores.append(
            self._entry(
                "uploads",
                label="Uploaded documents",
                group="library",
                description="Original files. Delete documents from the Library tab.",
                path=self.uploads_dir,
                size_bytes=uploads.size,
                items=uploads.files,
                last_modified=_iso(uploads.newest),
            )
        )
        bm25 = self._stats(self.bm25_dir)
        stores.append(
            self._entry(
                "bm25_index",
                label="Keyword index (BM25)",
                group="library",
                description="Rebuilt automatically from the documents in the library.",
                path=self.bm25_dir,
                size_bytes=bm25.size,
                items=len(getattr(self.doc_service.bm25_index, "entries", []) or []),
                items_label="chunks",
                last_modified=_iso(bm25.newest),
            )
        )
        for store_id, (label, description, path) in self.readonly_dirs.items():
            stats = self._stats(path, ttl=120.0)
            stores.append(
                self._entry(
                    store_id,
                    label=label,
                    group="library",
                    description=description,
                    path=path,
                    size_bytes=stats.size,
                    items=stats.files,
                    last_modified=_iso(stats.newest),
                )
            )
        return stores

    def _largest_children(self, root: Path, limit: int = 6) -> list[dict[str, Any]]:
        if not root.is_dir():
            return []
        rows = [{"name": child.name, "size_bytes": self._stats(child).size} for child in root.iterdir()]
        return sorted(rows, key=lambda row: row["size_bytes"], reverse=True)[:limit]

    def _models(self) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = []
        for store_id, (label, path) in self.model_dirs.items():
            stats = self._stats(path, ttl=600.0)
            if stats.files:
                models.append({"id": store_id, "label": label, "path": str(path), "size_bytes": stats.size})
        for model in list_installed_models(timeout=2.0):
            models.append(
                {
                    "id": f"ollama:{model['name']}",
                    "label": f"Ollama · {model['name']}",
                    "path": None,
                    "size_bytes": int(model.get("size") or 0),
                }
            )
        return models

    def _memory(self) -> dict[str, Any]:
        this_year = time.gmtime().tm_year
        loaded = []
        for model in list_loaded_models(timeout=2.0):
            expiry_year = str(model.get("expires_at") or "")[:4]
            loaded.append(
                {
                    "name": model["name"],
                    "size_bytes": int(model.get("size") or 0),
                    "vram_bytes": int(model.get("size_vram") or 0),
                    "context_length": model.get("context_length"),
                    "pinned": expiry_year.isdigit() and int(expiry_year) - this_year >= _PINNED_EXPIRY_YEARS,
                }
            )

        vision = HFVisionClient.get_instance()
        kv = get_redis_cache().stats()
        embedding_cache = getattr(self.doc_service.embedding_service, "cache", None)

        def actions(item_id: str) -> list[dict[str, Any]]:
            return [spec.describe() for (sid, _), spec in self._actions.items() if sid == item_id]

        items = [
            {
                "id": "ollama_models",
                "label": "Chat models in memory",
                "description": "Models Ollama keeps loaded, including each one's context (KV) cache.",
                "value": len(loaded),
                "unit": "loaded",
                "models": loaded,
                "actions": actions("ollama_models"),
            },
            {
                "id": "vision_model",
                "label": "Vision model",
                "description": "Qwen3-VL weights held by this server once a page image has been read.",
                "value": 1 if vision.is_loaded else 0,
                "unit": f"loaded on {vision.device}" if vision.is_loaded else "not loaded",
                "actions": actions("vision_model") if vision.is_loaded else [],
            },
            {
                "id": "kv_cache",
                "label": "Key-value cache",
                "description": "Cached query responses, embeddings and sessions "
                + ("in Redis." if kv["backend"] == "redis" else "held in memory (Redis is not running)."),
                "value": kv["keys"],
                "unit": "keys",
                "actions": actions("kv_cache"),
            },
            {
                "id": "retrieval_cache",
                "label": "Retrieval cache",
                "description": "Candidate chunks remembered per question for an hour.",
                "value": len(get_retrieval_cache()),
                "unit": "entries",
                "actions": actions("retrieval_cache"),
            },
            {
                "id": "embedding_cache",
                "label": "Embedding cache",
                "description": "Vectors already computed for recently seen text.",
                "value": len(embedding_cache) if embedding_cache is not None else 0,
                "unit": "vectors",
                "actions": actions("embedding_cache"),
            },
            {
                "id": "conversations",
                "label": "Conversation memory",
                "description": "Server-side state of chat sessions used for follow-up questions.",
                "value": self._session_count() if self._session_count else 0,
                "unit": "sessions",
                "actions": actions("conversations") if self._clear_sessions else [],
            },
            {
                "id": "docstore",
                "label": "Chunk store",
                "description": "Document chunks held in memory for fast context assembly.",
                "value": len(self.doc_service.docstore),
                "unit": "chunks",
                "actions": [],
            },
        ]
        return {"gpu": gpu_memory_mb(), "process_rss_bytes": _process_rss(), "items": items}

    def summary(self) -> dict[str, Any]:
        stores = self._stores()
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "busy": bool(self.doc_service.has_ingestion_in_flight()),
            "totals": {
                "disk_bytes": sum(s["size_bytes"] for s in stores),
                "reclaimable_bytes": sum(s["reclaimable_bytes"] for s in stores),
                "stores": len(stores),
                "databases": sum(1 for s in stores if s["group"] == "databases"),
            },
            "stores": stores,
            "memory": self._memory(),
            "models": self._models(),
        }
        self._maybe_snapshot(stores)
        return payload

    # ── History ─────────────────────────────────────────────────────────────

    def _append_history(self, entry: dict[str, Any]) -> None:
        with self._lock:
            try:
                self.history_path.parent.mkdir(parents=True, exist_ok=True)
                with self.history_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry) + "\n")
                lines = self.history_path.read_text(encoding="utf-8").splitlines()
                if len(lines) > _HISTORY_MAX_LINES:
                    self.history_path.write_text("\n".join(lines[-_HISTORY_KEEP_LINES:]) + "\n", encoding="utf-8")
            except OSError as exc:
                logger.warning("Could not write storage history: %s", exc)

    def _maybe_snapshot(self, stores: list[dict[str, Any]], force: bool = False) -> None:
        now = time.time()
        with self._lock:
            if not force and now - self._last_snapshot < _SNAPSHOT_INTERVAL_SECONDS:
                return
            self._last_snapshot = now
        self._append_history(
            {
                "type": "snapshot",
                "ts": datetime.now(UTC).isoformat(),
                "sizes": {s["id"]: s["size_bytes"] for s in stores},
                "items": {s["id"]: s["items"] for s in stores if s["items"] is not None},
            }
        )

    def history(self, limit: int = 500) -> dict[str, list[dict[str, Any]]]:
        entries: list[dict[str, Any]] = []
        with self._lock:
            if self.history_path.is_file():
                for line in self.history_path.read_text(encoding="utf-8").splitlines()[-limit:]:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return {
            "snapshots": [e for e in entries if e.get("type") == "snapshot"],
            "actions": [e for e in entries if e.get("type") == "action"],
        }

    # ── Actions ─────────────────────────────────────────────────────────────

    def _build_actions(self) -> dict[tuple[str, str], ActionSpec]:
        specs = {
            "vector_index": [
                ActionSpec(
                    "compact",
                    "Compact",
                    "Purges the applied write log and rebuilds the database file to release unused pages. "
                    "No documents or answers are removed. Searches pause for a few seconds.",
                    self._compact_vector_index,
                    safe=True,
                    blocks_on_ingestion=True,
                ),
            ],
            "telemetry_db": [
                ActionSpec(
                    "compact",
                    "Compact",
                    "Rebuilds the telemetry file to release unused pages. No records are removed.",
                    self._compact_telemetry,
                    safe=True,
                ),
                ActionSpec(
                    "older_than",
                    "Delete old records",
                    "Deletes telemetry records older than the chosen number of days.",
                    self._prune_telemetry,
                    needs_days=True,
                ),
                ActionSpec(
                    "clear",
                    "Clear all",
                    "Deletes every query trace, event and error record. The Telemetry tab starts empty.",
                    self._clear_telemetry,
                ),
            ],
            "semantic_cache": [
                ActionSpec(
                    "clear",
                    "Clear",
                    "Forgets every cached answer. The next identical question is answered from scratch.",
                    self._clear_semantic_cache,
                    blocks_on_ingestion=True,
                ),
            ],
            "page_images": [
                ActionSpec(
                    "remove_orphaned",
                    "Remove orphaned",
                    "Deletes images that belong to documents no longer in the library. Images of current documents are kept.",
                    self._remove_orphan_images,
                    safe=True,
                ),
            ],
            "vision_cache": [
                ActionSpec(
                    "remove_orphaned",
                    "Remove orphaned",
                    "Deletes cached page readings of documents no longer in the library.",
                    self._remove_orphan_vision,
                    safe=True,
                ),
                ActionSpec(
                    "older_than",
                    "Delete old entries",
                    "Deletes cached page readings older than the chosen number of days. Those pages are re-read on next use.",
                    self._prune_vision_cache,
                    needs_days=True,
                ),
                ActionSpec(
                    "clear",
                    "Clear all",
                    "Deletes every cached page reading, including those of current documents. "
                    "Re-reading takes about 25 seconds per page.",
                    self._clear_vision_cache,
                    blocks_on_ingestion=True,
                ),
            ],
            "logs": [
                ActionSpec(
                    "older_than",
                    "Delete old files",
                    "Deletes log and report files older than the chosen number of days. The live application log is kept.",
                    self._prune_logs,
                    needs_days=True,
                ),
            ],
            "eval_artifacts": [
                ActionSpec(
                    "delete",
                    "Delete",
                    "Deletes the benchmark corpora and indexes. The evaluation scripts rebuild them on their next run, "
                    "which re-embeds the benchmark documents.",
                    self._delete_eval_artifacts,
                ),
            ],
            "session_libraries": [
                ActionSpec(
                    "remove_old",
                    "Remove old",
                    "Deletes the uploads and indexes of earlier session-mode runs. The active library is kept.",
                    self._remove_old_sessions,
                ),
            ],
            "ollama_models": [
                ActionSpec(
                    "unload",
                    "Unload",
                    "Removes the model from memory and frees its VRAM. It reloads on the next question, which takes about 10 seconds.",
                    self._unload_ollama,
                    needs_target=True,
                ),
                ActionSpec(
                    "reload",
                    "Reload",
                    "Loads the model into memory and keeps it there.",
                    self._reload_ollama,
                    needs_target=True,
                ),
            ],
            "vision_model": [
                ActionSpec(
                    "unload",
                    "Unload",
                    "Removes the vision model from memory. It reloads the next time a page image is read.",
                    self._unload_vision,
                    blocks_on_ingestion=True,
                ),
            ],
            "kv_cache": [
                ActionSpec("clear", "Clear", "Forgets cached query responses, embeddings and sessions.", self._clear_kv),
            ],
            "retrieval_cache": [
                ActionSpec("clear", "Clear", "Forgets remembered search results.", self._clear_retrieval_cache),
            ],
            "embedding_cache": [
                ActionSpec("clear", "Clear", "Forgets computed vectors; they are recomputed on demand.", self._clear_embedding_cache),
            ],
            "conversations": [
                ActionSpec(
                    "clear",
                    "Clear",
                    "Forgets server-side conversation state. Follow-up questions in open chats lose their context.",
                    self._clear_conversations,
                ),
            ],
        }
        return {(store_id, spec.id): spec for store_id, group in specs.items() for spec in group}

    def run_action(
        self,
        store_id: str,
        action_id: str,
        older_than_days: int | None = None,
        target: str | None = None,
        *,
        snapshot: bool = True,
    ) -> dict[str, Any]:
        spec = self._actions.get((store_id, action_id))
        if spec is None:
            raise UnknownStorageActionError(f"Unknown storage action '{store_id}/{action_id}'.")
        if spec.needs_days and (older_than_days is None or not 1 <= older_than_days <= _MAX_DAYS):
            raise ValueError(f"older_than_days must be between 1 and {_MAX_DAYS}.")
        if spec.needs_target and not target:
            raise ValueError("This action needs a target.")
        if spec.blocks_on_ingestion and self.doc_service.has_ingestion_in_flight():
            raise StorageBusyError("A document is being indexed. Try again when it finishes.")

        with self._lock:
            result = spec.run(older_than_days, target)
            self._invalidate()
        entry = {
            "type": "action",
            "ts": datetime.now(UTC).isoformat(),
            "store": store_id,
            "action": action_id,
            "freed_bytes": result.freed_bytes,
            "removed_items": result.removed_items,
            "message": result.message,
        }
        self._append_history(entry)
        if snapshot:
            self._maybe_snapshot(self._stores(), force=True)
        logger.info("[STORAGE] %s/%s: %s", store_id, action_id, result.message)
        return entry

    def clean_safe(self) -> dict[str, Any]:
        """Run every action that cannot lose data the app still uses."""
        results = []
        for (store_id, action_id), spec in self._actions.items():
            if not spec.safe:
                continue
            try:
                results.append(self.run_action(store_id, action_id, snapshot=False))
            except StorageBusyError as exc:
                results.append({"store": store_id, "action": action_id, "skipped": str(exc)})
        self._maybe_snapshot(self._stores(), force=True)
        return {
            "freed_bytes": sum(r.get("freed_bytes", 0) for r in results),
            "removed_items": sum(r.get("removed_items", 0) for r in results),
            "results": results,
        }

    def _delete(self, root: Path, target: Path) -> int:
        """Delete a file or directory that must sit inside ``root``; return bytes freed."""
        if not _is_inside(root, target):
            raise ValueError(f"Refusing to delete outside {root}: {target}")
        size = _scan(target).size
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        return size

    @staticmethod
    def _cutoff(days: int | None) -> float:
        return time.time() - (days or 0) * 86400

    def _compact_vector_index(self, _days: int | None, _target: str | None) -> ActionResult:
        from chromadb.db.impl.sqlite import SqliteDB
        from chromadb.ingest.impl.utils import trigger_vector_segments_max_seq_id_migration
        from chromadb.segment import SegmentManager

        before = _scan(self.vector_dir).size
        client = get_shared_chroma_client(self.vector_dir)
        system = client._system
        database = system.instance(SqliteDB)
        # Same steps as `chroma utils vacuum`, on the client the app already holds.
        trigger_vector_segments_max_seq_id_migration(database, system.instance(SegmentManager))
        for collection in client.list_collections():
            name = collection if isinstance(collection, str) else collection.name
            database.purge_log(collection_id=client.get_collection(name).id)
        database.vacuum(timeout=15)
        config = database.config
        config.set_parameter("automatically_purge", True)
        database.set_config(config)
        freed = max(0, before - _scan(self.vector_dir).size)
        return ActionResult(freed, 0, "Vector index compacted.")

    def _compact_telemetry(self, _days: int | None, _target: str | None) -> ActionResult:
        before = (_sqlite_stats(self.telemetry_file) or {}).get("file_bytes", 0)
        self.telemetry_service.db.vacuum()
        after = (_sqlite_stats(self.telemetry_file) or {}).get("file_bytes", 0)
        return ActionResult(max(0, before - after), 0, "Telemetry database compacted.")

    def _prune_telemetry(self, days: int | None, _target: str | None) -> ActionResult:
        cutoff = (datetime.now(UTC) - timedelta(days=days or 0)).isoformat()
        removed = self.telemetry_service.db.prune(cutoff)
        return ActionResult(0, removed, f"Deleted {removed} telemetry records older than {days} days.")

    def _clear_telemetry(self, _days: int | None, _target: str | None) -> ActionResult:
        removed = sum(self.telemetry_service.db.table_counts().values())
        self.telemetry_service.clear()
        return ActionResult(0, removed, f"Cleared {removed} telemetry records.")

    def _clear_semantic_cache(self, _days: int | None, _target: str | None) -> ActionResult:
        if self.semantic_cache is None:
            return ActionResult(message="Semantic cache is not enabled.")
        name = getattr(self.semantic_cache, "collection_name", None)
        removed = next((c["rows"] for c in self._collections() if c["name"] == name), 0)
        self.semantic_cache.clear()
        return ActionResult(0, removed, f"Cleared {removed} cached answers.")

    def _remove_orphan_images(self, _days: int | None, _target: str | None) -> ActionResult:
        freed = removed = 0
        for directory in self._orphan_image_dirs():
            size = _scan(directory).size
            self.doc_service.image_asset_manager.delete_document_assets(directory.name)
            freed += size
            removed += 1
        return ActionResult(freed, removed, f"Removed images of {removed} deleted documents.")

    def _remove_orphan_vision(self, _days: int | None, _target: str | None) -> ActionResult:
        freed = removed = 0
        for item in self._orphan_vision_files():
            freed += self._delete(self.vision_cache_dir, item)
            removed += 1
        return ActionResult(freed, removed, f"Removed {removed} cached pages of deleted documents.")

    def _prune_vision_cache(self, days: int | None, _target: str | None) -> ActionResult:
        cutoff = self._cutoff(days)
        freed = removed = 0
        for item in self.vision_cache_dir.glob("*.json"):
            if item.stat().st_mtime < cutoff:
                freed += self._delete(self.vision_cache_dir, item)
                removed += 1
        return ActionResult(freed, removed, f"Removed {removed} cached pages older than {days} days.")

    def _clear_vision_cache(self, _days: int | None, _target: str | None) -> ActionResult:
        stats = _scan(self.vision_cache_dir)
        self.doc_service.vision_cache_manager.clear()
        return ActionResult(stats.size, stats.files, f"Cleared {stats.files} cached pages.")

    def _prune_logs(self, days: int | None, _target: str | None) -> ActionResult:
        cutoff = self._cutoff(days)
        freed = removed = 0
        for current, _dirs, files in os.walk(self.logs_dir, topdown=False):
            for name in files:
                path = Path(current) / name
                # Dotfiles such as .gitkeep are repository placeholders, not logs.
                if name in _LIVE_LOG_NAMES or name.startswith(".") or path.stat().st_mtime >= cutoff:
                    continue
                freed += self._delete(self.logs_dir, path)
                removed += 1
            folder = Path(current)
            if folder != self.logs_dir and not any(folder.iterdir()):
                folder.rmdir()
        return ActionResult(freed, removed, f"Removed {removed} log files older than {days} days.")

    def _delete_eval_artifacts(self, _days: int | None, _target: str | None) -> ActionResult:
        freed = removed = 0
        for root in self.eval_dirs:
            if not root.is_dir():
                continue
            for child in list(root.iterdir()):
                freed += self._delete(root, child)
                removed += 1
        return ActionResult(freed, removed, "Deleted evaluation corpora and indexes.")

    def _remove_old_sessions(self, _days: int | None, _target: str | None) -> ActionResult:
        freed = removed = 0
        for directory in self._old_session_dirs():
            freed += self._delete(self.sessions_dir, directory)
            removed += 1
        return ActionResult(freed, removed, f"Removed {removed} old session libraries.")

    def _unload_ollama(self, _days: int | None, target: str | None) -> ActionResult:
        loaded = {m["name"]: int(m.get("size_vram") or 0) for m in list_loaded_models(timeout=2.0)}
        if target not in loaded:
            raise ValueError(f"Model '{target}' is not loaded.")
        if not unload_model(str(target)):
            raise RuntimeError(f"Ollama did not unload '{target}'.")
        # Ollama acknowledges first and frees the model a moment later; wait so the
        # summary fetched right after this action already shows it gone.
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and any(m["name"] == target for m in list_loaded_models(timeout=2.0)):
            time.sleep(0.25)
        return ActionResult(loaded[target], 1, f"Unloaded {target} from memory.")

    def _reload_ollama(self, _days: int | None, target: str | None) -> ActionResult:
        installed = {m["name"] for m in list_installed_models(timeout=2.0)}
        if target not in installed:
            raise ValueError(f"Model '{target}' is not installed in Ollama.")
        if not preload_model(str(target), timeout=120.0):
            raise RuntimeError(f"Ollama did not load '{target}'.")
        return ActionResult(0, 1, f"Loaded {target} into memory.")

    def _unload_vision(self, _days: int | None, _target: str | None) -> ActionResult:
        with VisionService._semaphore:
            HFVisionClient.get_instance().unload()
        return ActionResult(0, 1, "Vision model unloaded.")

    def _clear_kv(self, _days: int | None, _target: str | None) -> ActionResult:
        removed = get_redis_cache().clear_app_keys()
        return ActionResult(0, removed, f"Cleared {removed} cached keys.")

    def _clear_retrieval_cache(self, _days: int | None, _target: str | None) -> ActionResult:
        cache = get_retrieval_cache()
        removed = len(cache)
        cache.clear()
        return ActionResult(0, removed, f"Cleared {removed} remembered searches.")

    def _clear_embedding_cache(self, _days: int | None, _target: str | None) -> ActionResult:
        cache = getattr(self.doc_service.embedding_service, "cache", None)
        removed = len(cache) if cache is not None else 0
        if cache is not None:
            cache.clear()
        return ActionResult(0, removed, f"Cleared {removed} cached vectors.")

    def _clear_conversations(self, _days: int | None, _target: str | None) -> ActionResult:
        removed = self._session_count() if self._session_count else 0
        if self._clear_sessions:
            self._clear_sessions()
        return ActionResult(0, removed, f"Cleared {removed} conversation sessions.")


def _process_rss() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None
