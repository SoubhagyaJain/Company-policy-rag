"""Durable library identity and ingestion state, separate from disposable telemetry.

SQLite transactions fence job claims/publication. A library-scoped OS file lock
also serializes workers across processes because Chroma/BM25 cannot fence leases.
Recovery deliberately replays parsing from the verified original, not partial
intermediate artifacts. No catalog operation removes source or index files.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class CatalogConflict(RuntimeError):
    """A job is owned elsewhere or its persisted state changed."""


class DocumentCatalog:
    def __init__(self, path: Path, *, clock=time.time) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        with self.connection() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise RuntimeError(f"Unsupported document catalog schema: {version}")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    source_path TEXT,
                    file_hash TEXT NOT NULL,
                    state TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    options_json TEXT NOT NULL,
                    expected_chunks INTEGER,
                    current_job_id TEXT,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(document_id),
                    state TEXT NOT NULL,
                    status_json TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL,
                    lease_owner TEXT,
                    lease_until REAL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    number INTEGER NOT NULL,
                    owner TEXT NOT NULL,
                    state TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    finished_at REAL,
                    error TEXT,
                    PRIMARY KEY(job_id, number)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    job_id TEXT,
                    state TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                PRAGMA user_version=1;
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            yield db
        finally:
            db.close()

    @contextmanager
    def transaction(self):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    @contextmanager
    def worker_lock(self):
        """Nonblocking process-death-safe lock for the single library writer."""
        with self.path.with_suffix(".worker.lock").open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            acquired = False
            try:
                if os.name == "nt":
                    import msvcrt
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        acquired = True
                    except OSError:
                        pass
                else:
                    import fcntl
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                    except BlockingIOError:
                        pass
                yield acquired
            finally:
                if acquired:
                    if os.name == "nt":
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(handle, fcntl.LOCK_UN)

    def _event(self, db, document_id, job_id, state, detail=""):
        db.execute("INSERT INTO events(document_id,job_id,state,detail,created_at) VALUES(?,?,?,?,?)",
                   (document_id, job_id, state, detail, self.clock()))

    @staticmethod
    def _document(row) -> dict[str, Any]:
        result = dict(row)
        result["record"] = json.loads(result.pop("record_json"))
        result["options"] = json.loads(result.pop("options_json"))
        return result

    def documents(self) -> list[dict[str, Any]]:
        with self.connection() as db:
            return [self._document(row) for row in db.execute("SELECT * FROM documents")]

    def get(self, document_id: str) -> dict[str, Any] | None:
        with self.connection() as db:
            row = db.execute("SELECT * FROM documents WHERE document_id=?", (document_id,)).fetchone()
            return self._document(row) if row else None

    def job(self, document_id: str) -> dict[str, Any] | None:
        with self.connection() as db:
            row = db.execute("SELECT j.* FROM jobs j JOIN documents d ON d.current_job_id=j.job_id "
                             "WHERE d.document_id=?", (document_id,)).fetchone()
            if not row:
                return None
            result = dict(row)
            result["status"] = json.loads(result.pop("status_json"))
            return result

    def searchable(self, document_id: str | None) -> bool:
        # Unknown IDs can belong to preserved legacy CLI indexes. Migration marks
        # them explicitly; new API ingestion always registers before indexing.
        row = self.get(document_id) if document_id else None
        return row is None or row["state"] in {"READY", "LEGACY"}

    def revision(self) -> str:
        with self.connection() as db:
            return str(db.execute("SELECT COALESCE(MAX(id),0) FROM events").fetchone()[0])

    def import_legacy(self, record: dict, source: Path | None, options: dict) -> None:
        document_id = record["document_id"]
        state = "DELETING" if record.get("status") == "DELETING" else (
            "LEGACY" if record.get("chunk_count", 0) else "FAILED")
        if state == "LEGACY":
            record = {**record, "status": "PARTIALLY_INDEXED", "current_stage": "LEGACY_REVIEW",
                      "error": "Legacy index completeness is unverified; existing evidence is retained."}
        with self.transaction() as db:
            cursor = db.execute("INSERT OR IGNORE INTO documents VALUES(?,?,?,?,?,?,?,?,?)",
                                (document_id, str(source.resolve()) if source else None,
                                 record.get("file_hash", ""), state, json.dumps(record),
                                 json.dumps(options), None, None, self.clock()))
            if cursor.rowcount:
                self._event(db, document_id, None, state, "Imported without changing files or indexes")

    def enqueue(self, record: dict, source: Path, options: dict, status: dict, *, max_attempts=3) -> None:
        document_id, job_id = record["document_id"], status["job_id"]
        with self.transaction() as db:
            prior = db.execute("SELECT state FROM documents WHERE document_id=?", (document_id,)).fetchone()
            if prior and prior[0] not in {"FAILED"}:
                raise CatalogConflict("Document is not eligible for a new ingestion attempt.")
            db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(document_id) DO UPDATE SET "
                       "state=excluded.state,record_json=excluded.record_json,options_json=excluded.options_json,"
                       "expected_chunks=NULL,current_job_id=excluded.current_job_id,updated_at=excluded.updated_at",
                       (document_id, str(source.resolve()), record["file_hash"], "QUEUED", json.dumps(record),
                        json.dumps(options), None, job_id, self.clock()))
            db.execute("INSERT INTO jobs(job_id,document_id,state,status_json,max_attempts,updated_at) "
                       "VALUES(?,?,?,?,?,?)", (job_id, document_id, "QUEUED", json.dumps(status), max_attempts, self.clock()))
            self._event(db, document_id, job_id, "QUEUED")

    def claim(self, document_id: str, owner: str, lease_seconds: float) -> dict | None:
        with self.transaction() as db:
            row = db.execute("SELECT j.* FROM jobs j JOIN documents d ON d.current_job_id=j.job_id "
                             "WHERE d.document_id=? AND d.state IN ('QUEUED','RUNNING')", (document_id,)).fetchone()
            if not row or row["state"] not in {"QUEUED", "RUNNING"}:
                return None
            if row["state"] == "RUNNING" and row["lease_until"] > self.clock():
                return None
            if row["state"] == "RUNNING":
                db.execute("UPDATE attempts SET state='INTERRUPTED',finished_at=?,error='Worker lease expired' "
                           "WHERE job_id=? AND number=?", (self.clock(), row["job_id"], row["attempt_count"]))
            if row["attempt_count"] >= row["max_attempts"]:
                status = json.loads(row["status_json"])
                status.update(status="FAILED", text_ready=False, can_retry=True,
                              error="Automatic recovery limit reached. Review and retry explicitly.")
                self._finish(db, document_id, row["job_id"], "FAILED", status)
                return None
            number = row["attempt_count"] + 1
            db.execute("UPDATE jobs SET state='RUNNING',attempt_count=?,lease_owner=?,lease_until=?,updated_at=? "
                       "WHERE job_id=?", (number, owner, self.clock() + lease_seconds, self.clock(), row["job_id"]))
            db.execute("UPDATE documents SET state='RUNNING',updated_at=? WHERE document_id=?", (self.clock(), document_id))
            db.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?,NULL,NULL)",
                       (row["job_id"], number, owner, "RUNNING", "SOURCE_VALIDATION", self.clock()))
            self._event(db, document_id, row["job_id"], "RUNNING", f"Attempt {number}")
            return {"job_id": row["job_id"], "attempt": number}

    def renew(self, job_id: str, owner: str, lease_seconds: float) -> None:
        with self.transaction() as db:
            cursor = db.execute("UPDATE jobs SET lease_until=?,updated_at=? WHERE job_id=? AND lease_owner=? "
                                "AND state='RUNNING' AND lease_until>?",
                                (self.clock() + lease_seconds, self.clock(), job_id, owner, self.clock()))
            if not cursor.rowcount:
                raise CatalogConflict("Ingestion lease is no longer owned by this worker.")

    def checkpoint(self, document_id: str, owner: str, status: dict, expected_chunks: int | None = None) -> None:
        with self.transaction() as db:
            job = db.execute("SELECT * FROM jobs WHERE job_id=?", (status["job_id"],)).fetchone()
            if not job or job["lease_owner"] != owner or job["state"] != "RUNNING" or job["lease_until"] <= self.clock():
                raise CatalogConflict("Cannot checkpoint an unowned ingestion job.")
            db.execute("UPDATE jobs SET status_json=?,updated_at=? WHERE job_id=?",
                       (json.dumps(status), self.clock(), status["job_id"]))
            db.execute("UPDATE attempts SET stage=? WHERE job_id=? AND number=?",
                       (status["current_stage"], status["job_id"], job["attempt_count"]))
            if expected_chunks is not None:
                db.execute("UPDATE documents SET expected_chunks=? WHERE document_id=?", (expected_chunks, document_id))

    def _finish(self, db, document_id, job_id, state, status):
        db.execute("UPDATE jobs SET state=?,status_json=?,lease_owner=NULL,lease_until=NULL,updated_at=? WHERE job_id=?",
                   (state, json.dumps(status), self.clock(), job_id))
        db.execute("UPDATE documents SET state=?,updated_at=? WHERE document_id=?", (state, self.clock(), document_id))
        db.execute("UPDATE attempts SET state=?,finished_at=?,error=? WHERE job_id=? AND state='RUNNING'",
                   (state, self.clock(), status.get("error"), job_id))
        self._event(db, document_id, job_id, state, status.get("error") or "")

    def finish(self, document_id: str, owner: str, status: dict, record: dict | None = None) -> None:
        with self.transaction() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (status["job_id"],)).fetchone()
            if not row or row["lease_owner"] != owner or row["state"] != "RUNNING":
                raise CatalogConflict("Cannot publish an unowned ingestion job.")
            if status["status"] == "READY" and row["lease_until"] <= self.clock():
                raise CatalogConflict("Cannot publish after the ingestion lease expired.")
            if record is not None:
                db.execute("UPDATE documents SET record_json=? WHERE document_id=?", (json.dumps(record), document_id))
            self._finish(db, document_id, status["job_id"], status["status"], status)

    def deletion(self, document_id: str, *, complete: bool = False) -> None:
        state = "DELETED" if complete else "DELETING"
        with self.transaction() as db:
            db.execute("UPDATE documents SET state=?,updated_at=? WHERE document_id=?", (state, self.clock(), document_id))
            db.execute("UPDATE jobs SET state=?,lease_owner=NULL,lease_until=NULL WHERE document_id=?", (state, document_id))
            self._event(db, document_id, None, state)

    def invalidate(self, document_id: str, error: str) -> None:
        with self.transaction() as db:
            row = db.execute("SELECT current_job_id FROM documents WHERE document_id=?", (document_id,)).fetchone()
            if row and row[0]:
                job = db.execute("SELECT status_json FROM jobs WHERE job_id=?", (row[0],)).fetchone()
                status = json.loads(job[0])
                status.update(status="FAILED", text_ready=False, can_retry=True, error=error)
                self._finish(db, document_id, row[0], "FAILED", status)

    def backup(self, destination: Path) -> None:
        """Consistent backup including WAL; refuse to replace an existing backup."""
        with Path(destination).open("xb"):
            pass
        with self.connection() as source, sqlite3.connect(destination) as target:
            source.backup(target)
