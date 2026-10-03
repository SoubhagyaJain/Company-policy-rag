"""Inventory and cleanup of everything the app persists or keeps in memory.

The Storage tab reads its summary from here and triggers cleanups by store id and
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
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from backend.embeddings.embeddings import OLLAMA_EMBED_NUM_CTX
from backend.embeddings.vector_store import get_shared_chroma_client
from backend.retrieval.retrieval_cache import get_retrieval_cache
from backend.services import storage_catalog as catalog
from backend.services import storage_insights as insights
from backend.services import storage_runtime as runtime
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
# Thirty days of 15-minute snapshots plus the audit entries written between them.
_HISTORY_MAX_LINES = 6000
_HISTORY_KEEP_LINES = 5000
_MAX_DAYS = 3650
# Directory sizes are remembered this long; an action or a deep scan drops them.
_SCAN_TTL_SECONDS = 60.0
_PROBE_TTL_SECONDS = 1.0
_GPU_TTL_SECONDS = 2.0
# An unreachable Ollama costs the full timeout; do not pay it on every poll.
_OLLAMA_DOWN_TTL_SECONDS = 30.0
_TELEMETRY_TTL_SECONDS = 30.0
_PLAN_DEFAULT_DAYS = 30
_AUDIT_LIMIT = 500
_CHART_POINTS = 240
_RANGES: dict[str, float] = {"24h": insights.DAY, "7d": 7 * insights.DAY, "30d": 30 * insights.DAY}

SAFE = "SAFE"
REBUILDABLE = "REBUILDABLE"
DESTRUCTIVE = "DESTRUCTIVE"

_LABELS: dict[str, str] = {
    "vector_index": "Chroma vector database",
    "telemetry_db": "SQLite telemetry database",
    "semantic_cache": "Semantic answer cache",
    "page_images": "Extracted page images",
    "vision_cache": "Vision cache",
    "logs": "Logs and evaluation reports",
    "eval_artifacts": "Evaluation indexes",
    "session_libraries": "Old session libraries",
    "uploads": "Uploaded documents",
    "bm25_index": "Keyword index (BM25)",
    "ollama_models": "Ollama models in memory",
    "vision_model": "Vision model",
    "kv_cache": "Application KV cache",
    "retrieval_cache": "Retrieval cache",
    "embedding_cache": "Embedding cache",
    "conversations": "Conversation memory",
    "docstore": "Chunk store",
}

_MODEL_DIR_ROLES: dict[str, tuple[str, str]] = {
    "vision_model": ("Vision", "Transformers"),
    "embedding_model": ("Embedding", "Sentence Transformers"),
    "reranker_model": ("Reranker", "Sentence Transformers"),
}


class StorageBusyError(RuntimeError):
    """A cleanup was requested while an ingestion or another cleanup holds the same stores."""


class UnknownStorageActionError(LookupError):
    """The store or action id does not exist."""


@dataclass
class ActionResult:
    freed_bytes: int = 0
    removed_items: int = 0
    message: str = ""


@dataclass(frozen=True)
class Estimate:
    """What an action would remove. ``None`` means the figure cannot be measured."""

    items: int | None = None
    size: int | None = None
    exact: bool = True
    note: str = ""


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
    # SAFE keeps everything the app uses, REBUILDABLE is regenerated on demand,
    # DESTRUCTIVE cannot be brought back.
    safety: str = DESTRUCTIVE
    deletes: str = ""
    rebuild: str = ""
    performance: str = ""
    estimate: Callable[[int | None], Estimate] | None = None
    # Offered in the cleanup center. Loading and unloading models is not a cleanup.
    cleanup: bool = True

    def describe(self, busy: bool = False) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "needs_days": self.needs_days,
            "needs_target": self.needs_target,
            "safe": self.safe,
            "safety": self.safety,
            "impact": {"deletes": self.deletes, "rebuild": self.rebuild, "performance": self.performance},
            "blocked_reason": (
                f"{self.label} is unavailable while document indexing is active."
                if self.blocks_on_ingestion and busy
                else None
            ),
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


def _iso(timestamp: float | None) -> str | None:
    return datetime.fromtimestamp(timestamp, UTC).isoformat() if timestamp else None


def _is_inside(root: Path, target: Path) -> bool:
    root_resolved, target_resolved = root.resolve(), target.resolve()
    return root_resolved in target_resolved.parents


def _sqlite_stats(db_file: Path, count_tables: tuple[str, ...] = ()) -> dict[str, Any] | None:
    """Page-level health of a SQLite file, read through a read-only connection."""
    if not db_file.is_file():
        return None
    wal_file = db_file.with_name(db_file.name + "-wal")
    wal_bytes = wal_file.stat().st_size if wal_file.is_file() else 0
    file_bytes = sum(
        candidate.stat().st_size
        for candidate in (db_file, wal_file, db_file.with_name(db_file.name + "-shm"))
        if candidate.is_file()
    )
    try:
        connection = sqlite3.connect(f"{db_file.resolve().as_uri()}?mode=ro", uri=True, timeout=2.0)
    except sqlite3.Error:
        return {"file_bytes": file_bytes, "wal_bytes": wal_bytes}
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
            "wal_bytes": wal_bytes,
            "page_size": page_size,
            "page_count": page_count,
            "free_pages": free_pages,
            "free_bytes": free_pages * page_size,
            "bloat_pct": round(100 * free_pages / page_count, 1) if page_count else 0.0,
            "rows": rows,
        }
    except sqlite3.Error as exc:
        logger.debug("Could not read SQLite stats for %s: %s", db_file, exc)
        return {"file_bytes": file_bytes, "wal_bytes": wal_bytes}
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
        # Short-lived results of probes and derived indexes, keyed by name.
        self._derived: dict[str, tuple[float, Any]] = {}
        self._last_snapshot = 0.0
        self._last_models_bytes: int | None = None
        self._history_lines: int | None = None
        self._vector_orphans: dict[str, Any] | None = None
        # One cleanup at a time; a second request is refused instead of queued.
        self._op_lock = threading.Lock()
        self._current_op: dict[str, Any] | None = None
        self._snapshot_thread: threading.Thread | None = None
        self._snapshot_stop = threading.Event()
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
        if embedding_name and "/" not in embedding_name:
            embedding_name = ""
        for key, label, name in (
            ("embedding_model", "Embedding model", embedding_name),
            ("reranker_model", "Reranker model", settings.reranker_model),
        ):
            if name:
                dirs[key] = (f"{label} ({name})", hub / f"models--{name.replace('/', '--')}")
        return dirs

    # ── Scanning ────────────────────────────────────────────────────────────

    def _stats(self, path: Path, ttl: float = _SCAN_TTL_SECONDS) -> _DirStats:
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

    def _cached(self, key: str, ttl: float, compute: Callable[[], Any]) -> Any:
        now = time.monotonic()
        with self._lock:
            hit = self._derived.get(key)
            if hit and hit[0] > now:
                return hit[1]
        value = compute()
        with self._lock:
            self._derived[key] = (time.monotonic() + ttl, value)
        return value

    def _invalidate(self) -> None:
        with self._lock:
            self._scan_cache.clear()
            self._derived.clear()

    def _known_ids(self) -> set[str]:
        return set(self.doc_service.known_document_ids())

    def _busy(self) -> bool:
        return bool(self.doc_service.has_ingestion_in_flight())

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

    def _document_collection_name(self) -> str:
        return str(getattr(self.doc_service.vector_store, "collection_name", settings.chroma_collection_name))

    def _collection_dim(self, name: str) -> int | None:
        """Width of the vectors in a collection, read from one stored row."""
        try:
            collection = get_shared_chroma_client(self.vector_dir).get_collection(name)
            sample = collection.get(limit=1, include=["embeddings"])
            embeddings = sample.get("embeddings")
            if embeddings is None or len(embeddings) == 0:
                return None
            return len(embeddings[0])
        except Exception as exc:
            logger.debug("Could not read the vector width of %s: %s", name, exc)
            return None

    def _vector_dim(self) -> int | None:
        return self._collection_dim(self._document_collection_name())

    def _vision_cache_index(self) -> dict[str, Any]:
        """Entries and bytes of the vision cache per owning document, in one directory pass."""

        def build() -> dict[str, Any]:
            by_doc: dict[str, list[int]] = {}
            unowned = [0, 0]
            oldest = newest = 0.0
            if self.vision_cache_dir.is_dir():
                with os.scandir(self.vision_cache_dir) as entries:
                    for entry in entries:
                        if not entry.name.endswith(".json"):
                            continue
                        try:
                            info = entry.stat()
                        except OSError:
                            continue
                        match = _VISION_CACHE_DOC.match(entry.name)
                        bucket = by_doc.setdefault(match.group(1), [0, 0]) if match else unowned
                        bucket[0] += 1
                        bucket[1] += info.st_size
                        oldest = info.st_mtime if not oldest else min(oldest, info.st_mtime)
                        newest = max(newest, info.st_mtime)
            return {"by_doc": by_doc, "unowned": unowned, "oldest": oldest, "newest": newest}

        return self._cached("vision_cache_index", _SCAN_TTL_SECONDS, build)

    def _orphan_chunk_refs(self) -> dict[str, int]:
        """Chunks held in memory whose document is no longer in the library."""
        known = self._known_ids()

        def count(chunks: Any) -> int:
            total = 0
            for chunk in chunks:
                doc_id = getattr(getattr(chunk, "metadata", None), "document_id", None)
                if doc_id and doc_id not in known:
                    total += 1
            return total

        docstore = self.doc_service.docstore
        return {
            "bm25": count(list(getattr(self.doc_service.bm25_index, "entries", []) or [])),
            "docstore": count(list(docstore.values()) if hasattr(docstore, "values") else []),
        }

    def _scan_vector_orphans(self) -> dict[str, Any]:
        """Count vector rows whose document is gone. Reads every row's metadata, so only on request."""
        known = self._known_ids()
        resolve = getattr(self.doc_service, "_restored_document_identity", None)
        orphaned = 0
        try:
            collection = get_shared_chroma_client(self.vector_dir).get_collection(self._document_collection_name())
            for meta in collection.get(include=["metadatas"]).get("metadatas") or []:
                meta = meta or {}
                doc_id = resolve(meta)[0] if callable(resolve) else str(meta.get("document_id") or "")
                if doc_id and doc_id not in known:
                    orphaned += 1
        except Exception as exc:
            logger.debug("Could not scan vector rows for orphans: %s", exc)
            return {"count": None, "checked_at": datetime.now(UTC).isoformat(), "error": str(exc)}
        return {"count": orphaned, "checked_at": datetime.now(UTC).isoformat()}

    def _telemetry(self, method: str, *args: Any, default: Any = None) -> Any:
        """Call a read-only telemetry helper; an unavailable database yields ``default``."""
        target = getattr(getattr(self.telemetry_service, "db", None), method, None)
        if not callable(target):
            return default
        try:
            return target(*args)
        except Exception as exc:
            logger.debug("Telemetry %s failed: %s", method, exc)
            return default

    # ── Inventory ───────────────────────────────────────────────────────────

    def _describe_actions(self, store_id: str, busy: bool) -> list[dict[str, Any]]:
        return [spec.describe(busy) for (sid, _), spec in self._actions.items() if sid == store_id]

    def _entry(
        self,
        store_id: str,
        *,
        group: str,
        description: str,
        label: str | None = None,
        path: Path | None = None,
        size_bytes: int = 0,
        items: int | None = None,
        items_label: str = "files",
        file_count: int | None = None,
        reclaimable_bytes: int = 0,
        last_modified: str | None = None,
        details: dict[str, Any] | None = None,
        busy: bool = False,
    ) -> dict[str, Any]:
        info = catalog.STORES.get(store_id)
        return {
            "id": store_id,
            "label": label or _LABELS.get(store_id, store_id),
            "group": group,
            "description": description,
            "path": str(path) if path else None,
            "size_bytes": size_bytes,
            "items": items,
            "items_label": items_label,
            "file_count": file_count,
            "reclaimable_bytes": reclaimable_bytes,
            "last_modified": last_modified,
            "details": details or {},
            "kinds": list(info.kinds) if info else [],
            "category": info.category if info else "other",
            "info": info.describe() if info else None,
            "actions": self._describe_actions(store_id, busy),
        }

    def _stores(self) -> list[dict[str, Any]]:
        stores: list[dict[str, Any]] = []
        busy = self._busy()

        # Databases
        vector = self._stats(self.vector_dir)
        vector_db = _sqlite_stats(self.vector_dir / "chroma.sqlite3", ("embeddings", "embeddings_queue")) or {}
        collections = self._collections()
        document_rows = next((c["rows"] for c in collections if c["name"] == self._document_collection_name()), None)
        sqlite_bytes = vector_db.get("file_bytes", 0)
        stores.append(
            self._entry(
                "vector_index",
                group="databases",
                description="Chroma database holding document embeddings and the semantic answer cache.",
                path=self.vector_dir,
                size_bytes=vector.size,
                items=sum(c["rows"] for c in collections),
                items_label="vectors",
                file_count=vector.files,
                reclaimable_bytes=vector_db.get("free_bytes", 0),
                last_modified=_iso(vector.newest),
                details={
                    "engine": "Chroma (SQLite + HNSW)",
                    "tables": collections,
                    "queue_rows": vector_db.get("rows", {}).get("embeddings_queue"),
                    "embedding_rows": vector_db.get("rows", {}).get("embeddings"),
                    "chunk_rows": document_rows,
                    "sqlite_bytes": sqlite_bytes,
                    "hnsw_bytes": max(0, vector.size - sqlite_bytes),
                    **{k: v for k, v in vector_db.items() if k != "rows"},
                },
                busy=busy,
            )
        )

        telemetry = _sqlite_stats(self.telemetry_file) or {}
        telemetry_rows = self._telemetry("table_counts", default={})
        bounds = self._telemetry("time_bounds", default={})
        oldest = [b["oldest"] for b in bounds.values() if b.get("oldest")]
        newest = [b["newest"] for b in bounds.values() if b.get("newest")]
        telemetry_scan = self._stats(self.telemetry_file)
        stores.append(
            self._entry(
                "telemetry_db",
                group="databases",
                description="Query traces, vision and cache events, errors and ingestion history.",
                path=self.telemetry_file,
                size_bytes=telemetry.get("file_bytes", 0),
                items=sum(telemetry_rows.values()),
                items_label="rows",
                file_count=1 if telemetry else 0,
                reclaimable_bytes=telemetry.get("free_bytes", 0),
                last_modified=_iso(telemetry_scan.newest),
                details={
                    "engine": "SQLite (WAL)",
                    "tables": [{"name": name, "rows": rows} for name, rows in telemetry_rows.items()],
                    "oldest_record": min(oldest) if oldest else None,
                    "newest_record": max(newest) if newest else None,
                    **telemetry,
                },
                busy=busy,
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
                group="caches",
                description="Previously generated answers reused for near-identical questions. Stored inside the vector database.",
                items=semantic_rows,
                items_label="answers",
                busy=busy,
            )
        )

        images = self._stats(self.images_dir)
        image_dirs = [d for d in self.images_dir.iterdir() if d.is_dir()] if self.images_dir.is_dir() else []
        orphan_dirs = self._orphan_image_dirs()
        orphan_names = {d.name for d in orphan_dirs}
        stores.append(
            self._entry(
                "page_images",
                group="caches",
                description="Images pulled out of uploaded documents for visual answers.",
                path=self.images_dir,
                size_bytes=images.size,
                items=len(image_dirs),
                items_label="documents",
                file_count=images.files,
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
                busy=busy,
            )
        )

        vision = self._stats(self.vision_cache_dir)
        orphan_files = self._orphan_vision_files()
        stores.append(
            self._entry(
                "vision_cache",
                group="caches",
                description="Text the vision model already read from page images. Re-reading a page takes about 25 seconds.",
                path=self.vision_cache_dir,
                size_bytes=vision.size,
                items=vision.files,
                items_label="pages",
                file_count=vision.files,
                reclaimable_bytes=sum(f.stat().st_size for f in orphan_files if f.is_file()),
                last_modified=_iso(vision.newest),
                details={"orphaned": len(orphan_files)},
                busy=busy,
            )
        )

        logs = self._stats(self.logs_dir)
        stores.append(
            self._entry(
                "logs",
                group="caches",
                description="Application log and the reports written by evaluation runs.",
                path=self.logs_dir,
                size_bytes=logs.size,
                items=logs.files,
                file_count=logs.files,
                last_modified=_iso(logs.newest),
                details={"largest": self._largest_children(self.logs_dir)},
                busy=busy,
            )
        )

        eval_stats = [self._stats(d) for d in self.eval_dirs]
        stores.append(
            self._entry(
                "eval_artifacts",
                group="caches",
                description="Benchmark corpora and indexes built by the evaluation scripts.",
                path=self.eval_dirs[0].parent if self.eval_dirs else None,
                size_bytes=sum(s.size for s in eval_stats),
                items=sum(s.files for s in eval_stats),
                file_count=sum(s.files for s in eval_stats),
                last_modified=_iso(max((s.newest for s in eval_stats), default=0.0)),
                details={
                    "largest": [
                        {"name": d.name, "size_bytes": s.size} for d, s in zip(self.eval_dirs, eval_stats) if s.size
                    ]
                },
                busy=busy,
            )
        )

        old_sessions = self._old_session_dirs()
        session_stats = [self._stats(d) for d in old_sessions]
        stores.append(
            self._entry(
                "session_libraries",
                group="caches",
                description="Isolated libraries left by earlier runs in session mode.",
                path=self.sessions_dir,
                size_bytes=sum(s.size for s in session_stats),
                items=len(old_sessions),
                items_label="sessions",
                file_count=sum(s.files for s in session_stats),
                last_modified=_iso(max((s.newest for s in session_stats), default=0.0)),
                busy=busy,
            )
        )

        # Live data, managed elsewhere
        uploads = self._stats(self.uploads_dir)
        stores.append(
            self._entry(
                "uploads",
                group="library",
                description="Original files. Delete documents from the Library tab.",
                path=self.uploads_dir,
                size_bytes=uploads.size,
                items=uploads.files,
                file_count=uploads.files,
                last_modified=_iso(uploads.newest),
                busy=busy,
            )
        )
        bm25 = self._stats(self.bm25_dir)
        stores.append(
            self._entry(
                "bm25_index",
                group="library",
                description="Chunks and tokens for keyword retrieval, kept in step with the library.",
                path=self.bm25_dir,
                size_bytes=bm25.size,
                items=len(getattr(self.doc_service.bm25_index, "entries", []) or []),
                items_label="chunks",
                file_count=bm25.files,
                last_modified=_iso(bm25.newest),
                busy=busy,
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
                    file_count=stats.files,
                    last_modified=_iso(stats.newest),
                    busy=busy,
                )
            )
        return stores

    def _largest_children(self, root: Path, limit: int = 6) -> list[dict[str, Any]]:
        if not root.is_dir():
            return []
        rows = [{"name": child.name, "size_bytes": self._stats(child).size} for child in root.iterdir()]
        return sorted(rows, key=lambda row: row["size_bytes"], reverse=True)[:limit]

    # ── Runtime ─────────────────────────────────────────────────────────────

    def _gpu(self) -> dict[str, Any] | None:
        return self._cached("gpu", _GPU_TTL_SECONDS, lambda: gpu_memory_mb())

    def _loaded_raw(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        with self._lock:
            hit = self._derived.get("ollama_ps")
            if hit and hit[0] > now:
                return hit[1]
        started = time.monotonic()
        raw = list_loaded_models(timeout=2.0)
        unreachable = not raw and time.monotonic() - started > 1.0
        ttl = _OLLAMA_DOWN_TTL_SECONDS if unreachable else _PROBE_TTL_SECONDS
        with self._lock:
            self._derived["ollama_ps"] = (time.monotonic() + ttl, raw)
        return raw

    def _installed_raw(self) -> list[dict[str, Any]]:
        return self._cached("ollama_tags", _SCAN_TTL_SECONDS, lambda: list_installed_models(timeout=2.0))

    def _request_stats(self) -> dict[str, dict[str, Any]]:
        return self._cached(
            "model_request_stats",
            _TELEMETRY_TTL_SECONDS,
            lambda: self._telemetry("model_request_stats", default={}),
        )

    def _in_process_models(self) -> list[dict[str, Any]]:
        try:
            return runtime.in_process_models(self.doc_service.embedding_service)
        except Exception as exc:
            logger.debug("Could not inspect in-process models: %s", exc)
            return []

    def _runtime(self) -> dict[str, Any]:
        loaded = runtime.ollama_loaded(self._loaded_raw(), self._request_stats())
        gpu = self._gpu()
        return {
            "ram": runtime.system_memory(),
            "gpu": gpu,
            "gpu_breakdown": runtime.gpu_breakdown(gpu, loaded, runtime.torch_reserved_bytes()),
            "loaded_models": loaded,
            "in_process_models": self._in_process_models(),
        }

    def _cache_counters(self) -> dict[str, dict[str, Any]]:
        """Entry counts and lookup counters of the in-memory caches. Cheap enough to poll."""
        embedding_cache = getattr(self.doc_service.embedding_service, "cache", None)
        embedding_stats = embedding_cache.stats() if hasattr(embedding_cache, "stats") else {}
        retrieval = get_retrieval_cache()
        return {
            "kv_cache": get_redis_cache().stats(),
            "retrieval_cache": retrieval.stats(),
            "embedding_cache": {
                "entries": len(embedding_cache) if embedding_cache is not None else 0,
                **{k: v for k, v in embedding_stats.items() if k != "entries"},
            },
            "conversations": {"entries": self._session_count() if self._session_count else 0},
            "docstore": {"entries": len(self.doc_service.docstore)},
        }

    def _memory(self, rt: dict[str, Any], counters: dict[str, dict[str, Any]]) -> dict[str, Any]:
        busy = self._busy()
        vision = HFVisionClient.get_instance()
        kv = counters["kv_cache"]

        def item(item_id: str, **fields: Any) -> dict[str, Any]:
            info = catalog.RUNTIME.get(item_id)
            return {
                "id": item_id,
                "label": _LABELS[item_id],
                "kinds": list(info.kinds) if info else [],
                "info": info.describe() if info else None,
                **fields,
            }

        items = [
            item(
                "ollama_models",
                description="Chat and embedding models the Ollama server keeps loaded, including their working memory.",
                value=len(rt["loaded_models"]),
                unit="loaded",
                models=[
                    {
                        "name": m["name"],
                        "size_bytes": m["size_bytes"],
                        "vram_bytes": m["vram_bytes"],
                        "context_length": m["context_length"],
                        "pinned": m["pinned"],
                    }
                    for m in rt["loaded_models"]
                ],
                actions=self._describe_actions("ollama_models", busy),
            ),
            item(
                "vision_model",
                description="Qwen3-VL weights held by this server once a page image has been read.",
                value=1 if vision.is_loaded else 0,
                unit=f"loaded on {vision.device}" if vision.is_loaded else "not loaded",
                actions=self._describe_actions("vision_model", busy) if vision.is_loaded else [],
            ),
            item(
                "kv_cache",
                description="Cached query responses, embeddings and sessions "
                + ("in Redis." if kv["backend"] == "redis" else "held in memory (Redis is not running)."),
                value=kv["keys"],
                unit="keys",
                actions=self._describe_actions("kv_cache", busy),
            ),
            item(
                "retrieval_cache",
                description="Candidate chunks remembered per question for an hour.",
                value=counters["retrieval_cache"]["entries"],
                unit="entries",
                actions=self._describe_actions("retrieval_cache", busy),
            ),
            item(
                "embedding_cache",
                description="Vectors already computed for recently seen text.",
                value=counters["embedding_cache"]["entries"],
                unit="vectors",
                actions=self._describe_actions("embedding_cache", busy),
            ),
            item(
                "conversations",
                description="Server-side state of chat sessions used for follow-up questions.",
                value=counters["conversations"]["entries"],
                unit="sessions",
                actions=self._describe_actions("conversations", busy) if self._clear_sessions else [],
            ),
            item(
                "docstore",
                description="Document chunks held in memory for fast context assembly.",
                value=counters["docstore"]["entries"],
                unit="chunks",
                actions=[],
            ),
        ]
        ram = rt["ram"]
        return {
            "gpu": rt["gpu"],
            "process_rss_bytes": ram["process_rss_bytes"],
            "items": items,
            "system_total_bytes": ram["system_total_bytes"],
            "system_available_bytes": ram["system_available_bytes"],
            "gpu_breakdown": rt["gpu_breakdown"],
            "loaded_models": rt["loaded_models"],
            "in_process_models": rt["in_process_models"],
        }

    def _models(self, rt: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        loaded_names = {m["name"] for m in (rt or {}).get("loaded_models", [])}
        loaded_purposes = {m["purpose"] for m in (rt or {}).get("in_process_models", [])}
        request_stats = self._request_stats() if rt is not None else {}
        models: list[dict[str, Any]] = []
        for store_id, (label, path) in self.model_dirs.items():
            stats = self._stats(path, ttl=600.0)
            if not stats.files:
                continue
            purpose, backend = _MODEL_DIR_ROLES.get(store_id, ("Model", "Transformers"))
            models.append(
                {
                    "id": store_id,
                    "label": label,
                    "name": label,
                    "path": str(path),
                    "size_bytes": stats.size,
                    "purpose": purpose,
                    "runtime": backend,
                    "loaded": purpose in loaded_purposes,
                    "modified_at": _iso(stats.newest),
                    "last_used": None,
                }
            )
        for model in self._installed_raw():
            name = str(model["name"])
            details = model.get("details") or {}
            models.append(
                {
                    "id": f"ollama:{name}",
                    "label": f"Ollama · {name}",
                    "name": name,
                    "path": None,
                    "size_bytes": int(model.get("size") or 0),
                    "purpose": runtime.model_purpose(name, details.get("families")),
                    "runtime": "Ollama",
                    "loaded": name in loaded_names,
                    "modified_at": model.get("modified_at"),
                    "last_used": (request_stats.get(name) or {}).get("last"),
                    "parameter_size": details.get("parameter_size"),
                    "quantization": details.get("quantization_level"),
                }
            )
        return models

    def _caches(self, stores: list[dict[str, Any]], counters: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        """One descriptor per cache: size, entries and whether keeping it pays off.

        A figure the runtime does not record is left as ``None``; the page says so
        instead of showing a guess.
        """
        by_id = {s["id"]: s for s in stores}
        busy = self._busy()
        since_start = "since the backend started"

        def rate(hits: int | None, misses: int | None) -> float | None:
            total = (hits or 0) + (misses or 0)
            return round(hits / total, 4) if hits is not None and total else None

        def base(cache_id: str, **fields: Any) -> dict[str, Any]:
            info = catalog.STORES.get(cache_id) or catalog.RUNTIME.get(cache_id)
            row = {
                "id": cache_id,
                "label": _LABELS[cache_id],
                "kinds": list(info.kinds) if info else [],
                "info": info.describe() if info else None,
                "location": "disk",
                "entries": 0,
                "entries_label": "entries",
                "size_bytes": None,
                "size_estimated": False,
                "size_note": None,
                "reclaimable_bytes": 0,
                "hits": None,
                "misses": None,
                "hit_rate": None,
                "hit_rate_window": None,
                "avg_saved_ms": None,
                "last_hit_at": None,
                "oldest_entry": None,
                "newest_entry": None,
                "ttl_seconds": None,
                "max_entries": None,
                "backend": None,
                "rebuild_cost": info.rebuild if info else None,
                "actions": self._describe_actions(cache_id, busy),
            }
            row.update(fields)
            return row

        semantic = self._cached(
            "semantic_value", _TELEMETRY_TTL_SECONDS, lambda: self._telemetry("semantic_cache_value", "7d", default={})
        )
        semantic_saved = None
        if semantic.get("avg_hit_ms") is not None and semantic.get("avg_miss_ms") is not None:
            semantic_saved = max(0.0, float(semantic["avg_miss_ms"]) - float(semantic["avg_hit_ms"]))
        lookups = int(semantic.get("lookups") or 0)

        vision_events = self._cached(
            "vision_cache_events",
            _TELEMETRY_TTL_SECONDS,
            lambda: self._telemetry("cache_event_stats", "Vision Cache", "7d", default={}),
        )
        vision_hit = vision_events.get("HIT") or {}
        # A SET is a page the model had to read because nothing was cached.
        vision_set = vision_events.get("SET") or {}
        vision_saved = None
        if vision_hit.get("avg_ms") is not None and vision_set.get("avg_ms") is not None:
            vision_saved = max(0.0, float(vision_set["avg_ms"]) - float(vision_hit["avg_ms"]))
        vision_index = self._vision_cache_index()

        kv = counters["kv_cache"]
        retrieval = counters["retrieval_cache"]
        embedding = counters["embedding_cache"]
        embedding_dim = embedding.get("vector_dim")

        return [
            base(
                "semantic_cache",
                entries=by_id["semantic_cache"]["items"] or 0,
                entries_label="answers",
                size_note="Stored inside the vector database; not measured separately.",
                hits=int(semantic.get("hits") or 0) if lookups else None,
                misses=lookups - int(semantic.get("hits") or 0) if lookups else None,
                hit_rate=round(int(semantic.get("hits") or 0) / lookups, 4) if lookups else None,
                hit_rate_window="last 7 days, from query traces",
                avg_saved_ms=semantic_saved,
                last_hit_at=semantic.get("last_hit"),
            ),
            base(
                "vision_cache",
                entries=by_id["vision_cache"]["items"] or 0,
                entries_label="pages",
                size_bytes=by_id["vision_cache"]["size_bytes"],
                reclaimable_bytes=by_id["vision_cache"]["reclaimable_bytes"],
                hits=vision_hit.get("count", 0) if vision_events else None,
                misses=vision_set.get("count", 0) if vision_events else None,
                hit_rate=rate(vision_hit.get("count", 0), vision_set.get("count", 0)) if vision_events else None,
                hit_rate_window="last 7 days, from cache events",
                avg_saved_ms=vision_saved,
                last_hit_at=vision_hit.get("last"),
                oldest_entry=_iso(vision_index["oldest"]),
                newest_entry=_iso(vision_index["newest"]),
            ),
            base(
                "retrieval_cache",
                location="memory",
                entries=retrieval["entries"],
                size_note="Held in backend RAM; not currently measured.",
                hits=retrieval["hits"],
                misses=retrieval["misses"],
                hit_rate=rate(retrieval["hits"], retrieval["misses"]),
                hit_rate_window=since_start,
                last_hit_at=_iso(retrieval["last_hit_at"]),
                ttl_seconds=retrieval["ttl_seconds"],
                max_entries=retrieval["max_entries"],
            ),
            base(
                "embedding_cache",
                location="memory",
                entries=embedding["entries"],
                entries_label="vectors",
                # A Python list of floats costs a pointer and a float object per element.
                size_bytes=embedding["entries"] * (embedding_dim * 32 + 120) if embedding_dim else None,
                size_estimated=True,
                size_note="Estimated from entries × vector width; held in backend RAM."
                if embedding_dim
                else "Held in backend RAM; not currently measured.",
                hits=embedding.get("hits"),
                misses=embedding.get("misses"),
                hit_rate=rate(embedding.get("hits"), embedding.get("misses")),
                hit_rate_window=since_start,
                last_hit_at=_iso(embedding.get("last_hit_at")),
                max_entries=embedding.get("max_entries"),
            ),
            base(
                "kv_cache",
                location="redis" if kv["backend"] == "redis" else "memory",
                entries=kv["keys"],
                entries_label="keys",
                size_note="Held by Redis; not measured from here."
                if kv["backend"] == "redis"
                else "Held in backend RAM; not currently measured.",
                hits=kv["hits"],
                misses=kv["misses"],
                hit_rate=rate(kv["hits"], kv["misses"]),
                hit_rate_window=since_start,
                last_hit_at=_iso(kv["last_hit_at"]),
                ttl_seconds=kv["default_ttl_seconds"],
                backend=kv["backend"],
            ),
            base(
                "conversations",
                location="memory",
                entries=counters["conversations"]["entries"],
                entries_label="sessions",
                size_note="Held in backend RAM; not currently measured.",
                actions=self._describe_actions("conversations", busy) if self._clear_sessions else [],
            ),
        ]

    def _operations(self) -> dict[str, Any]:
        jobs_fn = getattr(self.doc_service, "in_flight_jobs", None)
        jobs = jobs_fn() if callable(jobs_fn) else []
        indexing = [
            {
                "document_id": job.document_id,
                "filename": job.filename,
                "status": job.status,
                "stage": job.current_stage,
                "progress": job.progress,
                "pages_processed": job.pages_processed,
                "pages_total": job.pages_total,
                "chunks_created": job.chunks_created,
                "chunks_indexed": job.chunks_indexed,
                "vision_status": job.vision_status,
                "vision_pages_processed": job.vision_pages_processed,
                "vision_pages_total": job.vision_pages_total,
                "updated_at": job.updated_at,
                "stages": [{"stage": s.stage, "status": s.status} for s in job.stages],
            }
            for job in jobs
        ]
        with self._lock:
            current = dict(self._current_op) if self._current_op else None
        return {"indexing": indexing, "cleanup": current, "blocks_index_cleanup": self._busy()}

    def live(self) -> dict[str, Any]:
        """RAM, VRAM, loaded models, cache counters and running operations. Scans no directories."""
        rt = self._runtime()
        counters = self._cache_counters()
        ram = rt["ram"]
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "busy": self._busy(),
            "ram": ram,
            "gpu": rt["gpu"],
            "gpu_breakdown": rt["gpu_breakdown"],
            "loaded_models": rt["loaded_models"],
            "in_process_models": rt["in_process_models"],
            "vision_loaded": HFVisionClient.get_instance().is_loaded,
            "caches": {
                cache_id: {
                    "entries": stats.get("entries", stats.get("keys", 0)),
                    "hits": stats.get("hits"),
                    "misses": stats.get("misses"),
                    "last_hit_at": _iso(stats.get("last_hit_at")),
                    "backend": stats.get("backend"),
                }
                for cache_id, stats in counters.items()
            },
            "operations": self._operations(),
        }

    # ── Summary ─────────────────────────────────────────────────────────────

    def _disk(self) -> dict[str, Any] | None:
        try:
            usage = shutil.disk_usage(self.uploads_dir if self.uploads_dir.exists() else self.vector_dir)
            return {"total_bytes": usage.total, "free_bytes": usage.free, "volume": self.uploads_dir.anchor}
        except OSError:
            return None

    def _storage_map(
        self,
        stores: list[dict[str, Any]],
        models: list[dict[str, Any]],
        snapshots: list[dict[str, Any]],
        now: float,
    ) -> list[dict[str, Any]]:
        """Category totals for the composition bar, largest members first."""
        segments: dict[str, dict[str, Any]] = {
            cat_id: {
                "id": cat_id,
                "label": label,
                "size_bytes": 0,
                "reclaimable_bytes": 0,
                "growth_d7": None,
                "members": [],
            }
            for cat_id, label in catalog.CATEGORIES
        }
        for store in stores:
            segment = segments[store["category"]]
            segment["size_bytes"] += store["size_bytes"]
            segment["reclaimable_bytes"] += store["reclaimable_bytes"]
            weekly = store.get("growth", {}).get("d7")
            if weekly is not None:
                segment["growth_d7"] = (segment["growth_d7"] or 0) + weekly
            segment["members"].append(
                {"id": store["id"], "label": store["label"], "size_bytes": store["size_bytes"], "kind": "store"}
            )
        model_segment = segments["models"]
        for model in models:
            model_segment["size_bytes"] += model["size_bytes"]
            model_segment["members"].append(
                {"id": model["id"], "label": model["label"], "size_bytes": model["size_bytes"], "kind": "model"}
            )
        with_models = [s for s in snapshots if "models" in s]
        model_baseline = insights.total_growth(
            [{"ts": s["ts"], "sizes": {}, "models": s["models"]} for s in with_models],
            model_segment["size_bytes"],
            now,
            include_models=True,
        )
        model_segment["growth_d7"] = model_baseline.get("d7")
        for segment in segments.values():
            segment["members"].sort(key=lambda member: member["size_bytes"], reverse=True)
        return list(segments.values())

    def summary(self, refresh: bool = False) -> dict[str, Any]:
        if refresh:
            self._invalidate()
        now = time.time()
        stores = self._stores()
        rt = self._runtime()
        counters = self._cache_counters()
        models = self._models(rt)
        history = self._read_history()
        snapshots = [e for e in history if e.get("type") == "snapshot"]
        actions = [e for e in history if e.get("type") == "action"]

        store_growth = insights.growth(snapshots, {s["id"]: s["size_bytes"] for s in stores}, now)
        for store in stores:
            store["growth"] = store_growth.get(store["id"], {})
            if store["group"] == "databases":
                store["details"]["last_compaction"] = next(
                    (
                        a.get("ts")
                        for a in reversed(actions)
                        if a.get("store") == store["id"] and a.get("action") == "compact" and a.get("status") != "failed"
                    ),
                    None,
                )

        disk_bytes = sum(s["size_bytes"] for s in stores)
        reclaimable = sum(s["reclaimable_bytes"] for s in stores)
        models_bytes = sum(m["size_bytes"] for m in models)
        self._last_models_bytes = models_bytes
        by_id = {s["id"]: s for s in stores}
        orphan_bytes = by_id["page_images"]["reclaimable_bytes"] + by_id["vision_cache"]["reclaimable_bytes"]
        orphan_items = int(by_id["page_images"]["details"].get("orphaned") or 0) + int(
            by_id["vision_cache"]["details"].get("orphaned") or 0
        )
        chunk_refs = self._orphan_chunk_refs()
        kv_backend = counters["kv_cache"]["backend"]
        busy = self._busy()

        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "busy": busy,
            "totals": {
                "disk_bytes": disk_bytes,
                "reclaimable_bytes": reclaimable,
                "stores": len(stores),
                "databases": sum(1 for s in stores if s["group"] == "databases"),
                "models_bytes": models_bytes,
                "storage_bytes": disk_bytes + models_bytes,
                "growth": insights.total_growth(snapshots, disk_bytes, now, include_models=False),
                "history_span_seconds": insights.history_span_seconds(snapshots, now),
                "orphan_bytes": orphan_bytes,
                "orphan_items": orphan_items,
            },
            "disk": self._disk(),
            "stores": stores,
            "memory": self._memory(rt, counters),
            "models": models,
            "caches": self._caches(stores, counters),
            "map": self._storage_map(stores, models, snapshots, now),
            "observations": insights.observations(
                stores=stores,
                store_growth=store_growth,
                orphan_bytes=orphan_bytes,
                orphan_items=orphan_items,
                gpu=rt["gpu"],
                kv_backend=kv_backend,
            ),
            "health": insights.health(
                stores=stores,
                busy=busy,
                kv_backend=kv_backend,
                orphan_bytes=orphan_bytes,
                orphan_chunk_refs=chunk_refs["bm25"] + chunk_refs["docstore"],
                gpu=rt["gpu"],
                embedding_fallback=getattr(self.doc_service.embedding_service, "is_using_fallback", None),
            ),
            "operations": self._operations(),
            "hierarchy": catalog.HIERARCHY,
            "lifecycle": catalog.LIFECYCLE,
            "memory_flow": catalog.MEMORY_FLOW,
            "snapshot_interval_seconds": _SNAPSHOT_INTERVAL_SECONDS,
        }
        self._maybe_snapshot(stores, models_bytes=models_bytes)
        return payload

    # ── Documents ───────────────────────────────────────────────────────────

    def documents(self, deep: bool = False) -> dict[str, Any]:
        """Storage footprint of every document, and the artifacts whose document is gone."""
        records_fn = getattr(self.doc_service, "storage_records", None)
        records = records_fn() if callable(records_fn) else []
        vision_index = self._vision_cache_index()
        known = self._known_ids()
        dim = self._vector_dim()

        query_stats = self._cached(
            "document_query_stats",
            _TELEMETRY_TTL_SECONDS,
            lambda: self._telemetry("document_query_stats", default=[]),
        )

        rows = []
        for record in records:
            doc_id = record["document_id"]
            image_bytes = image_files = 0
            if _DOC_DIR.fullmatch(doc_id):
                image_dir = self.images_dir / doc_id
                image_stats = self._stats(image_dir)
                image_bytes = image_stats.size
                # The asset index sits next to the images it describes.
                image_files = max(0, image_stats.files - (1 if (image_dir / "assets.json").is_file() else 0))
            vision_entries, vision_bytes = vision_index["by_doc"].get(doc_id, [0, 0])
            chunks = record["chunk_count"]
            index_bytes = chunks * dim * 4 if dim else None
            matches = [
                q
                for q in query_stats
                if q.get("document_id") == doc_id or (record["filename"] and q.get("document_name") == record["filename"])
            ]
            rows.append(
                {
                    "document_id": doc_id,
                    "filename": record["filename"],
                    "status": record["status"],
                    "storage_state": record.get("storage_state", "HEALTHY"),
                    "created_at": record["created_at"],
                    "pages": record["pages_count"],
                    "file_bytes": record["file_size_bytes"],
                    "chunks": chunks,
                    "embeddings": chunks,
                    "images": {"count": image_files, "bytes": image_bytes},
                    "vision_cache": {"entries": vision_entries, "bytes": vision_bytes},
                    "index_bytes_estimated": index_bytes,
                    "total_bytes_estimated": record["file_size_bytes"] + image_bytes + vision_bytes + (index_bytes or 0),
                    "query_count": sum(q["count"] for q in matches),
                    "last_queried": max((q["last"] for q in matches if q.get("last")), default=None),
                }
            )

        orphan_dirs = self._orphan_image_dirs()
        orphan_image_items = sorted(
            (
                {"name": d.name, "bytes": self._stats(d).size, "files": self._stats(d).files}
                for d in orphan_dirs
            ),
            key=lambda row: row["bytes"],
            reverse=True,
        )
        orphan_vision_items = sorted(
            (
                {"name": doc_id, "entries": entries, "bytes": size}
                for doc_id, (entries, size) in vision_index["by_doc"].items()
                if doc_id not in known
            ),
            key=lambda row: row["bytes"],
            reverse=True,
        )
        old_sessions = self._old_session_dirs()
        session_items = [{"name": d.name, "bytes": self._stats(d).size, "files": self._stats(d).files} for d in old_sessions]
        if deep:
            self._vector_orphans = self._scan_vector_orphans()
        chunk_refs = self._orphan_chunk_refs()

        image_total = sum(row["bytes"] for row in orphan_image_items)
        vision_total = sum(row["bytes"] for row in orphan_vision_items)
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "vector_dim": dim,
            "documents": rows,
            "notes": {
                "index_bytes": "Estimated as chunks × vector width × 4 bytes. Chroma does not report bytes per document."
                if dim
                else "Not available: no vectors are stored yet.",
                "last_queried": "Counts questions that were scoped to this document. "
                "Library-wide questions are not attributed to a document.",
            },
            "orphans": {
                "total_bytes": image_total + vision_total,
                "reason": "Their source documents are no longer in the library.",
                "images": {
                    "count": len(orphan_image_items),
                    "files": sum(row["files"] for row in orphan_image_items),
                    "bytes": image_total,
                    "items": orphan_image_items[:50],
                    "store_id": "page_images",
                    "action_id": "remove_orphaned",
                    "safety": SAFE,
                },
                "vision_cache": {
                    "count": sum(row["entries"] for row in orphan_vision_items),
                    "bytes": vision_total,
                    "items": orphan_vision_items[:50],
                    "store_id": "vision_cache",
                    "action_id": "remove_orphaned",
                    "safety": SAFE,
                },
                "sessions": {
                    "count": len(session_items),
                    "bytes": sum(row["bytes"] for row in session_items),
                    "items": session_items[:50],
                    "store_id": "session_libraries",
                    "action_id": "remove_old",
                    "safety": DESTRUCTIVE,
                },
                "chunk_references": {
                    "bm25": chunk_refs["bm25"],
                    "docstore": chunk_refs["docstore"],
                    "vector": (self._vector_orphans or {}).get("count"),
                    "vector_checked_at": (self._vector_orphans or {}).get("checked_at"),
                    "report_only": True,
                },
            },
        }

    # ── Inspect ─────────────────────────────────────────────────────────────

    def _filesystem_details(self, store: dict[str, Any]) -> dict[str, Any]:
        info = store.get("info") or {}
        created = accessed = None
        if store["path"]:
            try:
                stat = Path(store["path"]).stat()
                # st_ctime is the creation time on Windows only.
                created = _iso(stat.st_ctime) if os.name == "nt" else None
                accessed = _iso(stat.st_atime)
            except OSError:
                pass
        return {
            "logical_name": store["id"],
            "path": store["path"],
            "engine": store["details"].get("engine") or ("Files" if store["path"] else None),
            "kinds": store.get("kinds", []),
            "size_bytes": store["size_bytes"],
            "file_count": store.get("file_count"),
            "created": created,
            "last_modified": store["last_modified"],
            "last_accessed": accessed,
            "rebuildable": catalog.REBUILDABLE in store.get("kinds", []),
            "cleanup_policy": info.get("cleanup_policy"),
        }

    def _inspect_vector(self, store: dict[str, Any]) -> dict[str, Any]:
        document_name = self._document_collection_name()
        semantic_name = getattr(self.semantic_cache, "collection_name", None)
        collections = []
        for collection in store["details"].get("tables", []):
            name = collection["name"]
            dim = self._vector_dim() if name == document_name else self._collection_dim(name)
            role = "Document chunks" if name == document_name else "Semantic answer cache" if name == semantic_name else "Other"
            collections.append(
                {
                    "name": name,
                    "role": role,
                    "rows": collection["rows"],
                    "vector_dim": dim,
                    "embedding_bytes_estimated": collection["rows"] * dim * 4 if dim else None,
                }
            )
        records_fn = getattr(self.doc_service, "storage_records", None)
        records = records_fn() if callable(records_fn) else []
        dated = sorted((r for r in records if r.get("created_at")), key=lambda r: r["created_at"])
        docstore = self.doc_service.docstore
        chunks = list(docstore.values()) if hasattr(docstore, "values") else []
        text_lengths = [len(getattr(chunk, "text", "") or "") for chunk in chunks]
        return {
            "collections": collections,
            "documents": {
                "count": len(records),
                "chunks": sum(r["chunk_count"] for r in records),
                "avg_chunk_chars": round(sum(text_lengths) / len(text_lengths)) if text_lengths else None,
                "largest": [
                    {"document_id": r["document_id"], "filename": r["filename"], "chunks": r["chunk_count"]}
                    for r in sorted(records, key=lambda r: r["chunk_count"], reverse=True)[:8]
                ],
                "oldest": {"filename": dated[0]["filename"], "created_at": dated[0]["created_at"]} if dated else None,
                "newest": {"filename": dated[-1]["filename"], "created_at": dated[-1]["created_at"]} if dated else None,
            },
            "sqlite": {
                key: store["details"].get(key)
                for key in ("file_bytes", "wal_bytes", "page_size", "page_count", "free_pages", "free_bytes", "bloat_pct")
            },
            "segments": {"sqlite_bytes": store["details"].get("sqlite_bytes"), "hnsw_bytes": store["details"].get("hnsw_bytes")},
        }

    def _inspect_semantic(self) -> dict[str, Any]:
        collection = getattr(self.semantic_cache, "_collection", None)
        if collection is None:
            return {"entries": None, "note": "The semantic cache collection is not available."}
        metadatas = collection.get(include=["metadatas"]).get("metadatas") or []
        stamps = [float(m["timestamp"]) for m in metadatas if m and m.get("timestamp") is not None]
        models: dict[str, int] = {}
        for meta in metadatas:
            name = str((meta or {}).get("model") or "unspecified")
            models[name] = models.get(name, 0) + 1
        current = None
        resolver = getattr(self.semantic_cache, "_resolve_kb_version", None)
        if callable(resolver):
            current = resolver(None)
        return {
            "entries": len(metadatas),
            "oldest_entry": _iso(min(stamps)) if stamps else None,
            "newest_entry": _iso(max(stamps)) if stamps else None,
            "by_model": [{"model": name, "entries": count} for name, count in sorted(models.items())],
            # An entry written for another library version is never served again.
            "for_current_library": sum(1 for m in metadatas if (m or {}).get("kb_version") == current)
            if current is not None
            else None,
        }

    def inspect(self, store_id: str) -> dict[str, Any]:
        """Everything known about one store, for the detail drawer."""
        runtime_info = catalog.RUNTIME.get(store_id)
        if runtime_info is not None and store_id not in catalog.STORES:
            return {"id": store_id, "label": _LABELS.get(store_id, store_id), "info": runtime_info.describe(), "filesystem": None}

        store = next((s for s in self._stores() if s["id"] == store_id), None)
        if store is None:
            raise UnknownStorageActionError(f"Unknown store '{store_id}'.")
        payload: dict[str, Any] = {
            "id": store_id,
            "label": store["label"],
            "info": store["info"],
            "filesystem": self._filesystem_details(store),
            "largest": self._largest_children(Path(store["path"]), limit=10)
            if store["path"] and Path(store["path"]).is_dir()
            else [],
        }
        try:
            if store_id == "vector_index":
                payload.update(self._inspect_vector(store))
            elif store_id == "telemetry_db":
                bounds = self._telemetry("time_bounds", default={})
                payload["tables"] = [
                    {
                        "name": table["name"],
                        "rows": table["rows"],
                        "oldest": (bounds.get(table["name"]) or {}).get("oldest"),
                        "newest": (bounds.get(table["name"]) or {}).get("newest"),
                    }
                    for table in store["details"].get("tables", [])
                ]
                payload["sqlite"] = {
                    key: store["details"].get(key)
                    for key in ("file_bytes", "wal_bytes", "page_size", "page_count", "free_pages", "free_bytes", "bloat_pct")
                }
            elif store_id == "semantic_cache":
                payload["semantic"] = self._inspect_semantic()
        except Exception as exc:
            logger.warning("Storage inspection of %s failed: %s", store_id, exc)
            payload["error"] = f"Inspection failed: {exc}"
        return payload

    # ── History ─────────────────────────────────────────────────────────────

    def _read_history(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        with self._lock:
            if self.history_path.is_file():
                for line in self.history_path.read_text(encoding="utf-8").splitlines():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries

    def _append_history(self, entry: dict[str, Any]) -> None:
        with self._lock:
            try:
                self.history_path.parent.mkdir(parents=True, exist_ok=True)
                if self._history_lines is None:
                    self._history_lines = (
                        len(self.history_path.read_text(encoding="utf-8").splitlines()) if self.history_path.is_file() else 0
                    )
                with self.history_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry) + "\n")
                self._history_lines += 1
                if self._history_lines > _HISTORY_MAX_LINES:
                    kept = self.history_path.read_text(encoding="utf-8").splitlines()[-_HISTORY_KEEP_LINES:]
                    self.history_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
                    self._history_lines = len(kept)
            except OSError as exc:
                logger.warning("Could not write storage history: %s", exc)

    def _maybe_snapshot(self, stores: list[dict[str, Any]], force: bool = False, models_bytes: int | None = None) -> None:
        now = time.time()
        with self._lock:
            if not force and now - self._last_snapshot < _SNAPSHOT_INTERVAL_SECONDS:
                return
            self._last_snapshot = now
        entry: dict[str, Any] = {
            "type": "snapshot",
            "ts": datetime.now(UTC).isoformat(),
            "sizes": {s["id"]: s["size_bytes"] for s in stores},
            "items": {s["id"]: s["items"] for s in stores if s["items"] is not None},
        }
        if models_bytes is None:
            models_bytes = self._last_models_bytes
        if models_bytes is not None:
            entry["models"] = models_bytes
        self._append_history(entry)

    def start_snapshot_loop(self, interval: float = _SNAPSHOT_INTERVAL_SECONDS) -> None:
        """Record size snapshots in the background so growth is tracked while the page is closed."""
        with self._lock:
            if self._snapshot_thread is not None:
                return
            self._snapshot_stop.clear()
            self._snapshot_thread = threading.Thread(
                target=self._snapshot_loop, args=(interval,), name="storage-snapshots", daemon=True
            )
        self._snapshot_thread.start()

    def stop_snapshot_loop(self) -> None:
        self._snapshot_stop.set()
        with self._lock:
            self._snapshot_thread = None

    def _snapshot_loop(self, interval: float) -> None:
        # Let start-up finish before the first directory scan.
        delay = min(30.0, interval)
        while not self._snapshot_stop.wait(delay):
            delay = interval
            try:
                models_bytes = sum(m["size_bytes"] for m in self._models())
                self._last_models_bytes = models_bytes
                self._maybe_snapshot(self._stores(), models_bytes=models_bytes)
            except Exception as exc:
                logger.warning("Storage snapshot failed: %s", exc)

    def _ingestion_events(self) -> list[dict[str, Any]]:
        traces = self._telemetry("get_ingestion_traces", 200, default=[])
        return [
            {
                "timestamp": trace.created_at,
                "filename": trace.filename,
                "file_size_bytes": trace.file_size_bytes,
                "chunks_count": trace.chunks_count,
                "status": trace.status,
            }
            for trace in traces
        ]

    def history(self, limit: int = 500, window: str | None = None) -> dict[str, Any]:
        """Recorded snapshots and the audit log; with ``window`` also chart series, events and a forecast."""
        entries = self._read_history()
        if window is None:
            entries = entries[-limit:]
            return {
                "snapshots": [e for e in entries if e.get("type") == "snapshot"],
                "actions": [e for e in entries if e.get("type") == "action"],
            }
        if window not in _RANGES:
            raise ValueError(f"range must be one of {', '.join(_RANGES)}.")

        now = time.time()
        since = now - _RANGES[window]
        snapshots = [e for e in entries if e.get("type") == "snapshot"]
        actions = [e for e in entries if e.get("type") == "action"]
        in_window = insights.within(snapshots, since, _CHART_POINTS)
        category_of = {store_id: info.category for store_id, info in catalog.STORES.items()}
        disk = self._disk()
        model_events = [
            {"name": m["name"], "modified_at": m.get("modified_at"), "size_bytes": int(m.get("size") or 0)}
            for m in self._installed_raw()
        ]
        return {
            "range": window,
            "snapshots": in_window,
            "actions": actions[-_AUDIT_LIMIT:],
            "series": insights.series(in_window, category_of),
            "events": insights.events(snapshots, actions, self._ingestion_events(), model_events, since),
            "forecast": insights.forecast(snapshots, now, disk["free_bytes"] if disk else None),
            "span_seconds": insights.history_span_seconds(snapshots, now),
            "snapshot_interval_seconds": _SNAPSHOT_INTERVAL_SECONDS,
        }

    # ── Actions ─────────────────────────────────────────────────────────────

    def _build_actions(self) -> dict[tuple[str, str], ActionSpec]:
        nothing_to_rebuild = "Nothing to rebuild."
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
                    safety=SAFE,
                    deletes="Unused pages and the applied write log. No documents, vectors or cached answers.",
                    rebuild=nothing_to_rebuild,
                    performance="Searches pause for a few seconds while the file is rewritten.",
                    estimate=lambda _days: self._estimate_free_pages(self.vector_dir / "chroma.sqlite3"),
                ),
            ],
            "telemetry_db": [
                ActionSpec(
                    "compact",
                    "Compact",
                    "Rebuilds the telemetry file to release unused pages. No records are removed.",
                    self._compact_telemetry,
                    safe=True,
                    safety=SAFE,
                    deletes="Unused pages in the telemetry file. No records.",
                    rebuild=nothing_to_rebuild,
                    performance="Telemetry writes wait a moment while the file is rewritten.",
                    estimate=lambda _days: self._estimate_free_pages(self.telemetry_file),
                ),
                ActionSpec(
                    "older_than",
                    "Delete old records",
                    "Deletes telemetry records older than the chosen number of days.",
                    self._prune_telemetry,
                    needs_days=True,
                    safety=DESTRUCTIVE,
                    deletes="Query traces, events, errors and ingestion history older than the cutoff.",
                    rebuild="Cannot be regenerated.",
                    performance="None. Run Compact afterwards to return the space to the disk.",
                    estimate=self._estimate_telemetry_prune,
                ),
                ActionSpec(
                    "clear",
                    "Clear all",
                    "Deletes every query trace, event and error record. The Telemetry tab starts empty.",
                    self._clear_telemetry,
                    safety=DESTRUCTIVE,
                    deletes="Every query trace, event, error and ingestion record.",
                    rebuild="Cannot be regenerated. The Telemetry tab starts empty.",
                    performance="None. Hit rates and last-queried times on this page start over.",
                    estimate=lambda _days: self._estimate_telemetry_prune(None),
                ),
            ],
            "semantic_cache": [
                ActionSpec(
                    "clear",
                    "Clear",
                    "Forgets every cached answer. The next identical question is answered from scratch.",
                    self._clear_semantic_cache,
                    blocks_on_ingestion=True,
                    safety=REBUILDABLE,
                    deletes="Every cached answer and its citations.",
                    rebuild="Refills automatically as questions are answered.",
                    performance="Repeated questions run full retrieval and generation once instead of returning instantly.",
                    estimate=self._estimate_semantic_cache,
                ),
            ],
            "page_images": [
                ActionSpec(
                    "remove_orphaned",
                    "Remove orphaned",
                    "Deletes images that belong to documents no longer in the library. Images of current documents are kept.",
                    self._remove_orphan_images,
                    safe=True,
                    safety=SAFE,
                    deletes="Image folders of documents that are no longer in the library.",
                    rebuild=nothing_to_rebuild,
                    performance="None.",
                    estimate=lambda _days: self._estimate_paths(self._orphan_image_dirs()),
                ),
            ],
            "vision_cache": [
                ActionSpec(
                    "remove_orphaned",
                    "Remove orphaned",
                    "Deletes cached page readings of documents no longer in the library.",
                    self._remove_orphan_vision,
                    safe=True,
                    safety=SAFE,
                    deletes="Cached page readings of documents that are no longer in the library.",
                    rebuild=nothing_to_rebuild,
                    performance="None.",
                    estimate=lambda _days: self._estimate_paths(self._orphan_vision_files()),
                ),
                ActionSpec(
                    "older_than",
                    "Delete old entries",
                    "Deletes cached page readings older than the chosen number of days. Those pages are re-read on next use.",
                    self._prune_vision_cache,
                    needs_days=True,
                    safety=REBUILDABLE,
                    deletes="Cached page readings older than the cutoff, including those of current documents.",
                    rebuild="Regenerated when a page is needed again.",
                    performance="Each of those pages is read again by the vision model, about 25 seconds per page.",
                    estimate=lambda days: self._estimate_paths(self._vision_files_older_than(days)),
                ),
                ActionSpec(
                    "clear",
                    "Clear all",
                    "Deletes every cached page reading, including those of current documents. "
                    "Re-reading takes about 25 seconds per page.",
                    self._clear_vision_cache,
                    blocks_on_ingestion=True,
                    safety=REBUILDABLE,
                    deletes="Every cached page reading, including those of current documents.",
                    rebuild="Regenerated when a page is needed again.",
                    performance="Every visual page is read again by the vision model, about 25 seconds per page.",
                    estimate=lambda _days: self._estimate_dir(self.vision_cache_dir),
                ),
            ],
            "logs": [
                ActionSpec(
                    "older_than",
                    "Delete old files",
                    "Deletes log and report files older than the chosen number of days. The live application log is kept.",
                    self._prune_logs,
                    needs_days=True,
                    safety=SAFE,
                    deletes="Log and report files older than the cutoff. The live application log is kept.",
                    rebuild="Logs cannot be regenerated. Evaluation reports reappear when an evaluation is run again.",
                    performance="None.",
                    estimate=lambda days: self._estimate_paths(self._log_files_older_than(days)),
                ),
            ],
            "eval_artifacts": [
                ActionSpec(
                    "delete",
                    "Delete",
                    "Deletes the benchmark corpora and indexes. The evaluation scripts rebuild them on their next run, "
                    "which re-embeds the benchmark documents.",
                    self._delete_eval_artifacts,
                    safety=DESTRUCTIVE,
                    deletes="Benchmark corpora and the indexes built from them.",
                    rebuild="Rebuilt by the next evaluation run.",
                    performance="The next evaluation run re-embeds the benchmark documents before it starts.",
                    estimate=lambda _days: self._estimate_roots(self.eval_dirs),
                ),
            ],
            "session_libraries": [
                ActionSpec(
                    "remove_old",
                    "Remove old",
                    "Deletes the uploads and indexes of earlier session-mode runs. The active library is kept.",
                    self._remove_old_sessions,
                    safety=DESTRUCTIVE,
                    deletes="Uploads and indexes of earlier session-mode runs. The active library is kept.",
                    rebuild="Cannot be regenerated: these folders contain the documents uploaded in those sessions.",
                    performance="None.",
                    estimate=lambda _days: self._estimate_paths(self._old_session_dirs()),
                ),
            ],
            "ollama_models": [
                ActionSpec(
                    "unload",
                    "Unload",
                    "Removes the model from memory and frees its VRAM. It reloads on the next request for that model.",
                    self._unload_ollama,
                    needs_target=True,
                    safety=REBUILDABLE,
                    deletes="The model's weights and working buffers from memory. The model file stays on disk.",
                    rebuild="Reloaded from disk on the next request for that model.",
                    performance="The next request waits for the model to load.",
                    cleanup=False,
                ),
                ActionSpec(
                    "reload",
                    "Reload",
                    "Loads the model into memory and keeps it there.",
                    self._reload_ollama,
                    needs_target=True,
                    safety=SAFE,
                    deletes="Nothing.",
                    rebuild=nothing_to_rebuild,
                    performance="Uses VRAM until the model is unloaded.",
                    cleanup=False,
                ),
            ],
            "vision_model": [
                ActionSpec(
                    "unload",
                    "Unload",
                    "Removes the vision model from memory. It reloads the next time a page image is read.",
                    self._unload_vision,
                    blocks_on_ingestion=True,
                    safety=REBUILDABLE,
                    deletes="The vision model's weights from memory. The model directory stays on disk.",
                    rebuild="Reloaded from disk the next time a page image is read.",
                    performance="The next page reading waits for the model to load.",
                    cleanup=False,
                ),
            ],
            "kv_cache": [
                ActionSpec(
                    "clear",
                    "Clear",
                    "Forgets cached query responses, embeddings and sessions.",
                    self._clear_kv,
                    safety=REBUILDABLE,
                    deletes="This app's cached keys: query responses, embeddings and sessions.",
                    rebuild="Refills automatically as the app is used.",
                    performance="Cached lookups are recomputed once.",
                    estimate=lambda _days: Estimate(get_redis_cache().stats()["keys"], None),
                ),
            ],
            "retrieval_cache": [
                ActionSpec(
                    "clear",
                    "Clear",
                    "Forgets remembered search results.",
                    self._clear_retrieval_cache,
                    safety=REBUILDABLE,
                    deletes="Remembered candidate chunks for recent questions.",
                    rebuild="Refills automatically as questions are asked.",
                    performance="The next occurrence of each question runs retrieval again.",
                    estimate=lambda _days: Estimate(len(get_retrieval_cache()), None),
                ),
            ],
            "embedding_cache": [
                ActionSpec(
                    "clear",
                    "Clear",
                    "Forgets computed vectors; they are recomputed on demand.",
                    self._clear_embedding_cache,
                    safety=REBUILDABLE,
                    deletes="Cached embedding vectors.",
                    rebuild="Regenerated automatically when text is embedded again.",
                    performance="The first retrieval after clearing may be slower.",
                    estimate=lambda _days: Estimate(self._embedding_cache_len(), None),
                ),
            ],
            "conversations": [
                ActionSpec(
                    "clear",
                    "Clear",
                    "Forgets server-side conversation state. Follow-up questions in open chats lose their context.",
                    self._clear_conversations,
                    safety=DESTRUCTIVE,
                    deletes="Server-side state of every chat session.",
                    rebuild="Cannot be regenerated. New state forms as chats continue.",
                    performance="Follow-up questions in open chats lose their context.",
                    estimate=lambda _days: Estimate(self._session_count() if self._session_count else 0, None),
                ),
            ],
        }
        return {(store_id, spec.id): spec for store_id, group in specs.items() for spec in group}

    def _spec(self, store_id: str, action_id: str) -> ActionSpec:
        spec = self._actions.get((store_id, action_id))
        if spec is None:
            raise UnknownStorageActionError(f"Unknown storage action '{store_id}/{action_id}'.")
        return spec

    @staticmethod
    def _check_days(spec: ActionSpec, older_than_days: int | None) -> None:
        if spec.needs_days and (older_than_days is None or not 1 <= older_than_days <= _MAX_DAYS):
            raise ValueError(f"older_than_days must be between 1 and {_MAX_DAYS}.")

    def _measure(self, store_id: str) -> int | None:
        """Bytes a store occupies on disk right now, or ``None`` for stores that live in memory."""
        if store_id == "telemetry_db":
            return (_sqlite_stats(self.telemetry_file) or {}).get("file_bytes", 0)
        if store_id == "session_libraries":
            return sum(_scan(d).size for d in self._old_session_dirs())
        roots = {
            "vector_index": [self.vector_dir],
            "page_images": [self.images_dir],
            "vision_cache": [self.vision_cache_dir],
            "logs": [self.logs_dir],
            "eval_artifacts": self.eval_dirs,
        }.get(store_id)
        return sum(_scan(root).size for root in roots) if roots else None

    def run_action(
        self,
        store_id: str,
        action_id: str,
        older_than_days: int | None = None,
        target: str | None = None,
        *,
        snapshot: bool = True,
        initiated_by: str = "manual",
    ) -> dict[str, Any]:
        spec = self._spec(store_id, action_id)
        self._check_days(spec, older_than_days)
        if spec.needs_target and not target:
            raise ValueError("This action needs a target.")
        if spec.blocks_on_ingestion and self._busy():
            raise StorageBusyError("A document is being indexed. Try again when it finishes.")
        if not self._op_lock.acquire(blocking=False):
            raise StorageBusyError("Another storage operation is still running. Try again when it finishes.")

        entry: dict[str, Any] = {
            "type": "action",
            "id": f"act_{uuid.uuid4().hex[:10]}",
            "ts": datetime.now(UTC).isoformat(),
            "store": store_id,
            "action": action_id,
            "target": target,
            "older_than_days": older_than_days if spec.needs_days else None,
            "initiated_by": initiated_by,
            "safety": spec.safety,
        }
        started = time.perf_counter()
        try:
            with self._lock:
                self._current_op = {"store": store_id, "action": action_id, "target": target, "started_at": entry["ts"]}
            before = self._measure(store_id)
            try:
                result = spec.run(older_than_days, target)
            except Exception as exc:
                self._invalidate()
                entry.update(
                    freed_bytes=0,
                    removed_items=0,
                    message=f"{spec.label} failed.",
                    before_bytes=before,
                    after_bytes=self._measure(store_id),
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                    status="failed",
                    error=str(exc),
                )
                self._append_history(entry)
                raise
            self._invalidate()
            entry.update(
                freed_bytes=result.freed_bytes,
                removed_items=result.removed_items,
                message=result.message,
                before_bytes=before,
                after_bytes=self._measure(store_id),
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
                status="success",
            )
        finally:
            with self._lock:
                self._current_op = None
            self._op_lock.release()

        self._append_history(entry)
        if snapshot:
            self._maybe_snapshot(self._stores(), force=True)
        logger.info("[STORAGE] %s/%s: %s", store_id, action_id, result.message)
        return entry

    def preview(self, store_id: str, action_id: str, older_than_days: int | None = None) -> dict[str, Any]:
        """What an action would remove, without running it."""
        spec = self._spec(store_id, action_id)
        self._check_days(spec, older_than_days)
        estimate = self._estimate(spec, older_than_days)
        return {
            "store": store_id,
            "action": action_id,
            "older_than_days": older_than_days if spec.needs_days else None,
            "affected_items": estimate.items,
            "estimated_bytes": estimate.size,
            "exact": estimate.exact,
            "note": estimate.note,
            **{k: v for k, v in spec.describe(self._busy()).items() if k in ("safety", "impact", "blocked_reason")},
        }

    def cleanup_plan(self) -> dict[str, Any]:
        """Every cleanup that can be offered, with what it would reclaim and what it costs."""
        busy = self._busy()
        items = []
        for (store_id, action_id), spec in self._actions.items():
            if not spec.cleanup or (store_id == "conversations" and not self._clear_sessions):
                continue
            days = _PLAN_DEFAULT_DAYS if spec.needs_days else None
            estimate = self._estimate(spec, days)
            described = spec.describe(busy)
            has_effect = bool(estimate.size) or bool(estimate.items)
            items.append(
                {
                    "store_id": store_id,
                    "action_id": action_id,
                    "store_label": _LABELS.get(store_id, store_id),
                    "label": spec.label,
                    "safety": spec.safety,
                    "impact": described["impact"],
                    "blocked_reason": described["blocked_reason"],
                    "needs_days": spec.needs_days,
                    "default_days": days,
                    "affected_items": estimate.items,
                    "estimated_bytes": estimate.size,
                    "exact": estimate.exact,
                    "note": estimate.note,
                    "has_effect": has_effect,
                    # Only what loses nothing is preselected.
                    "selected": spec.safe and bool(estimate.size) and not described["blocked_reason"],
                }
            )
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "busy": busy,
            "items": items,
            "safe_reclaimable_bytes": sum(i["estimated_bytes"] or 0 for i in items if i["selected"]),
        }

    def clean_safe(self, items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Run the chosen cleanups, or every one that cannot lose data the app still uses."""
        if items is None:
            selected = [(store_id, action_id, None) for (store_id, action_id), spec in self._actions.items() if spec.safe]
        else:
            selected = []
            for item in items:
                spec = self._spec(str(item.get("store_id")), str(item.get("action_id")))
                if not spec.cleanup:
                    raise UnknownStorageActionError(
                        f"'{item.get('store_id')}/{item.get('action_id')}' is not a cleanup action."
                    )
                self._check_days(spec, item.get("older_than_days"))
                selected.append((str(item["store_id"]), str(item["action_id"]), item.get("older_than_days")))

        started = time.perf_counter()
        results = []
        for store_id, action_id, days in selected:
            try:
                results.append(self.run_action(store_id, action_id, days, snapshot=False, initiated_by="cleanup"))
            except StorageBusyError as exc:
                results.append({"store": store_id, "action": action_id, "skipped": str(exc)})
            except Exception as exc:
                logger.warning("[STORAGE] cleanup step %s/%s failed: %s", store_id, action_id, exc)
                results.append({"store": store_id, "action": action_id, "failed": str(exc)})
        self._maybe_snapshot(self._stores(), force=True)
        return {
            "freed_bytes": sum(r.get("freed_bytes", 0) for r in results),
            "removed_items": sum(r.get("removed_items", 0) for r in results),
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "results": results,
        }

    # ── Estimates ───────────────────────────────────────────────────────────

    def _estimate(self, spec: ActionSpec, days: int | None) -> Estimate:
        if spec.estimate is None:
            return Estimate(None, None, exact=False)
        try:
            return spec.estimate(days)
        except Exception as exc:
            logger.debug("Could not estimate %s: %s", spec.id, exc)
            return Estimate(None, None, exact=False, note="Could not be estimated.")

    @staticmethod
    def _estimate_free_pages(db_file: Path) -> Estimate:
        stats = _sqlite_stats(db_file) or {}
        return Estimate(
            None,
            stats.get("free_bytes", 0),
            exact=False,
            note="Unused pages reported by SQLite. The file shrinks by about this much.",
        )

    def _estimate_telemetry_prune(self, days: int | None) -> Estimate:
        counts = self._telemetry("table_counts", default={})
        total = sum(counts.values())
        if days is None:
            affected = total
        else:
            cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
            affected = self._telemetry("count_older_than", cutoff, default=0)
        stats = _sqlite_stats(self.telemetry_file) or {}
        used = max(0, stats.get("file_bytes", 0) - stats.get("free_bytes", 0))
        return Estimate(
            affected,
            int(used * affected / total) if total else 0,
            exact=False,
            note="Estimated from the share of rows. The space is freed inside the file; Compact returns it to the disk.",
        )

    def _estimate_semantic_cache(self, _days: int | None) -> Estimate:
        name = getattr(self.semantic_cache, "collection_name", None)
        rows = next((c["rows"] for c in self._collections() if c["name"] == name), 0)
        return Estimate(
            rows,
            None,
            exact=False,
            note="Stored inside the vector database. Compacting it afterwards releases the space.",
        )

    @staticmethod
    def _estimate_paths(paths: list[Path]) -> Estimate:
        return Estimate(len(paths), sum(_scan(path).size for path in paths))

    @staticmethod
    def _estimate_dir(root: Path) -> Estimate:
        stats = _scan(root)
        return Estimate(stats.files, stats.size)

    @staticmethod
    def _estimate_roots(roots: list[Path]) -> Estimate:
        children = [child for root in roots if root.is_dir() for child in root.iterdir()]
        return Estimate(len(children), sum(_scan(child).size for child in children))

    def _embedding_cache_len(self) -> int:
        cache = getattr(self.doc_service.embedding_service, "cache", None)
        return len(cache) if cache is not None else 0

    def _vision_files_older_than(self, days: int | None) -> list[Path]:
        cutoff = self._cutoff(days)
        return [item for item in self.vision_cache_dir.glob("*.json") if item.stat().st_mtime < cutoff]

    def _log_files_older_than(self, days: int | None) -> list[Path]:
        cutoff = self._cutoff(days)
        found = []
        for current, _dirs, files in os.walk(self.logs_dir):
            for name in files:
                path = Path(current) / name
                # Dotfiles such as .gitkeep are repository placeholders, not logs.
                if name in _LIVE_LOG_NAMES or name.startswith(".") or path.stat().st_mtime >= cutoff:
                    continue
                found.append(path)
        return found

    # ── Action bodies ───────────────────────────────────────────────────────

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
        freed = removed = 0
        for item in self._vision_files_older_than(days):
            freed += self._delete(self.vision_cache_dir, item)
            removed += 1
        return ActionResult(freed, removed, f"Removed {removed} cached pages older than {days} days.")

    def _clear_vision_cache(self, _days: int | None, _target: str | None) -> ActionResult:
        stats = _scan(self.vision_cache_dir)
        self.doc_service.vision_cache_manager.clear()
        return ActionResult(stats.size, stats.files, f"Cleared {stats.files} cached pages.")

    def _prune_logs(self, days: int | None, _target: str | None) -> ActionResult:
        freed = removed = 0
        for path in self._log_files_older_than(days):
            freed += self._delete(self.logs_dir, path)
            removed += 1
        for current, _dirs, _files in os.walk(self.logs_dir, topdown=False):
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
        if runtime.model_purpose(str(target)) == "Embedding":
            try:
                from ollama import Client

                Client(host=settings.ollama_base_url, timeout=120.0).embed(
                    model=str(target),
                    input=[],
                    options={"num_ctx": OLLAMA_EMBED_NUM_CTX, "num_gpu": 0},
                    keep_alive=-1,
                )
            except Exception as exc:
                raise RuntimeError(f"Ollama did not load '{target}'.") from exc
        elif not preload_model(str(target), timeout=120.0):
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
