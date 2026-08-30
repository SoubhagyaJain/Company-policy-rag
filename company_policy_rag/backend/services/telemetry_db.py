from __future__ import annotations

import json
import queue
import sqlite3
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from backend.models.telemetry_models import (
    CacheTypeMetrics,
    DocumentIngestionTrace,
    ErrorIncident,
    IngestionStageTelemetry,
    MemoryResolutionEvent,
    QueryTraceRecord,
    SeverityLevel,
    TimeSeriesPoint,
    VisionFailureRecord,
)
from backend.utils.logging import logger


class TelemetryDB:
    """
    Thread-safe SQLite persistent telemetry repository with async write-behind buffering.
    Ensures zero query-time disk write latency while maintaining complete telemetry durability
    across server reboots, worker restarts, and multi-process deployments.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            base_dir = Path("storage")
            base_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = base_dir / "telemetry.sqlite3"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._write_generation = 0
        self._deleted_trace_identifiers: set[str] = set()
        self._write_queue: queue.Queue[tuple[int, str, tuple[Any, ...]]] = queue.Queue(maxsize=10000)
        self._stop_event = threading.Event()
        self._init_schema()

        # Start non-blocking background writer thread
        self._writer_thread = threading.Thread(target=self._background_writer, daemon=True)
        self._writer_thread.start()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS query_traces (
                            trace_id TEXT PRIMARY KEY,
                            request_id TEXT NOT NULL,
                            conversation_id TEXT,
                            document_id TEXT,
                            timestamp TEXT NOT NULL,
                            original_query TEXT NOT NULL,
                            resolved_query TEXT,
                            rewritten_query TEXT,
                            query_type TEXT NOT NULL,
                            routing_confidence REAL,
                            retrieval_strategy TEXT,
                            retrieval_required INTEGER NOT NULL,
                            conversational_bypass INTEGER NOT NULL,
                            evidence_required INTEGER NOT NULL,
                            candidate_count INTEGER NOT NULL,
                            post_rerank_count INTEGER NOT NULL,
                            final_chunk_count INTEGER NOT NULL,
                            anchor_section TEXT,
                            section_expansion_used INTEGER NOT NULL,
                            vision_used INTEGER NOT NULL,
                            vision_model TEXT,
                            vision_cache_status TEXT,
                            evidence_text_count INTEGER NOT NULL,
                            evidence_code_count INTEGER NOT NULL,
                            evidence_diagram_count INTEGER NOT NULL,
                            evidence_table_count INTEGER NOT NULL,
                            faithfulness_passed INTEGER NOT NULL,
                            verification_score REAL,
                            retry_count INTEGER NOT NULL,
                            cache_hit INTEGER NOT NULL,
                            cache_similarity REAL,
                            execution_time_ms REAL NOT NULL,
                            ttft_ms REAL,
                            tokens_per_second REAL,
                            prompt_tokens INTEGER NOT NULL,
                            completion_tokens INTEGER NOT NULL,
                            total_tokens INTEGER NOT NULL,
                            generation_model TEXT NOT NULL,
                            query_scope TEXT,
                            active_document_name TEXT,
                            error TEXT,
                            raw_trace_json TEXT NOT NULL
                        );
                    """)
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_qt_ts ON query_traces(timestamp);")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_qt_req ON query_traces(request_id);")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_qt_doc ON query_traces(document_id);")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_qt_conv ON query_traces(conversation_id);")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_qt_type ON query_traces(query_type);")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_qt_model ON query_traces(generation_model);")

                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS vision_events (
                            id TEXT PRIMARY KEY,
                            timestamp TEXT NOT NULL,
                            request_id TEXT,
                            document_id TEXT,
                            page_number INTEGER,
                            visual_type TEXT,
                            status TEXT NOT NULL,
                            duration_ms REAL NOT NULL,
                            model_name TEXT NOT NULL,
                            message TEXT
                        );
                    """)
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_ve_ts ON vision_events(timestamp);")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_ve_status ON vision_events(status);")

                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS memory_events (
                            id TEXT PRIMARY KEY,
                            timestamp TEXT NOT NULL,
                            session_id TEXT,
                            user_query TEXT NOT NULL,
                            resolved_query TEXT NOT NULL,
                            referent_found TEXT,
                            resolution_status TEXT NOT NULL,
                            latency_ms REAL NOT NULL
                        );
                    """)
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_me_ts ON memory_events(timestamp);")

                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS cache_events (
                            id TEXT PRIMARY KEY,
                            timestamp TEXT NOT NULL,
                            cache_type TEXT NOT NULL,
                            event_type TEXT NOT NULL,
                            latency_ms REAL NOT NULL,
                            key_hash TEXT,
                            model_name TEXT
                        );
                    """)
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_ce_ts ON cache_events(timestamp);")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_ce_type ON cache_events(cache_type, event_type);")

                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS error_incidents (
                            incident_id TEXT PRIMARY KEY,
                            timestamp TEXT NOT NULL,
                            request_id TEXT,
                            document_id TEXT,
                            conversation_id TEXT,
                            component TEXT NOT NULL,
                            severity TEXT NOT NULL,
                            message TEXT NOT NULL,
                            duration_ms REAL,
                            retry_count INTEGER NOT NULL DEFAULT 0,
                            stack_trace TEXT,
                            details_json TEXT
                        );
                    """)
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_ei_ts ON error_incidents(timestamp);")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_ei_comp ON error_incidents(component);")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_ei_sev ON error_incidents(severity);")

                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS ingestion_events (
                            document_id TEXT PRIMARY KEY,
                            timestamp TEXT NOT NULL,
                            filename TEXT NOT NULL,
                            category TEXT NOT NULL,
                            file_size_bytes INTEGER NOT NULL,
                            status TEXT NOT NULL,
                            current_stage TEXT NOT NULL,
                            pages_count INTEGER NOT NULL,
                            chunks_count INTEGER NOT NULL,
                            visual_assets_count INTEGER NOT NULL,
                            vision_success_count INTEGER NOT NULL,
                            vision_failed_count INTEGER NOT NULL,
                            total_duration_ms REAL NOT NULL,
                            error TEXT,
                            raw_stages_json TEXT NOT NULL
                        );
                    """)
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_ie_ts ON ingestion_events(timestamp);")
            finally:
                conn.close()

    def _background_writer(self) -> None:
        """Process queued SQL insert statements in batch mode."""
        while not self._stop_event.is_set():
            try:
                item = self._write_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            batch = [item]
            while len(batch) < 100:
                try:
                    batch.append(self._write_queue.get_nowait())
                except queue.Empty:
                    break

            try:
                with self._lock:
                    conn = self._get_connection()
                    with conn:
                        for generation, sql, params in batch:
                            if generation != self._write_generation:
                                continue
                            if (
                                "query_traces" in sql
                                and len(params) >= 2
                                and (str(params[0]) in self._deleted_trace_identifiers or str(params[1]) in self._deleted_trace_identifiers)
                            ):
                                continue
                            conn.execute(sql, params)
                    conn.close()
            except Exception as exc:
                logger.error("TelemetryDB background batch write error: %s", exc)

    def record_query_trace(self, trace: QueryTraceRecord) -> None:
        """Enqueue QueryTraceRecord for persistent SQLite storage."""
        with self._lock:
            self._deleted_trace_identifiers.discard(trace.trace_id)
            self._deleted_trace_identifiers.discard(trace.request_id)
        sql = """
            INSERT OR REPLACE INTO query_traces (
                trace_id, request_id, conversation_id, document_id, timestamp,
                original_query, resolved_query, rewritten_query, query_type, routing_confidence,
                retrieval_strategy, retrieval_required, conversational_bypass, evidence_required,
                candidate_count, post_rerank_count, final_chunk_count, anchor_section,
                section_expansion_used, vision_used, vision_model, vision_cache_status,
                evidence_text_count, evidence_code_count, evidence_diagram_count, evidence_table_count,
                faithfulness_passed, verification_score, retry_count, cache_hit, cache_similarity,
                execution_time_ms, ttft_ms, tokens_per_second, prompt_tokens, completion_tokens,
                total_tokens, generation_model, query_scope, active_document_name, error, raw_trace_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            trace.trace_id,
            trace.request_id,
            trace.conversation_id,
            trace.document_id,
            trace.timestamp,
            trace.original_query,
            trace.resolved_query,
            trace.rewritten_query,
            trace.query_type,
            trace.routing_confidence,
            trace.retrieval_strategy,
            1 if trace.retrieval_required else 0,
            1 if trace.conversational_bypass else 0,
            1 if trace.evidence_required else 0,
            trace.candidate_count,
            trace.post_rerank_count,
            trace.final_chunk_count,
            trace.anchor_section,
            1 if trace.section_expansion_used else 0,
            1 if trace.vision_used else 0,
            trace.vision_model,
            trace.vision_cache_status,
            trace.evidence_text_count,
            trace.evidence_code_count,
            trace.evidence_diagram_count,
            trace.evidence_table_count,
            1 if trace.faithfulness_passed else 0,
            trace.verification_score,
            trace.retry_count,
            1 if trace.cache_hit else 0,
            trace.cache_similarity,
            trace.execution_time_ms,
            trace.ttft_ms,
            trace.tokens_per_second,
            trace.prompt_tokens,
            trace.completion_tokens,
            trace.total_tokens,
            trace.generation_model,
            trace.query_scope,
            trace.active_document_name,
            trace.error,
            trace.model_dump_json(),
        )
        try:
            self._write_queue.put_nowait((self._write_generation, sql, params))
        except queue.Full:
            logger.warning("TelemetryDB write queue full, dropping record.")

    def record_vision_event(
        self,
        document_id: str | None,
        page_number: int | None,
        visual_type: str,
        status: str,
        duration_ms: float,
        model_name: str = "Qwen3-VL-2B-Instruct",
        request_id: str | None = None,
        message: str | None = None,
    ) -> None:
        event_id = f"ve_{uuid.uuid4().hex[:8]}"
        sql = """
            INSERT INTO vision_events (
                id, timestamp, request_id, document_id, page_number,
                visual_type, status, duration_ms, model_name, message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            event_id,
            datetime.now(UTC).isoformat(),
            request_id,
            document_id,
            page_number,
            visual_type,
            status,
            duration_ms,
            model_name,
            message or "",
        )
        try:
            self._write_queue.put_nowait((self._write_generation, sql, params))
        except queue.Full:
            pass

    def record_memory_event(
        self,
        session_id: str | None,
        user_query: str,
        resolved_query: str,
        referent_found: str | None,
        resolution_status: str,
        latency_ms: float,
    ) -> None:
        event_id = f"me_{uuid.uuid4().hex[:8]}"
        sql = """
            INSERT INTO memory_events (
                id, timestamp, session_id, user_query, resolved_query,
                referent_found, resolution_status, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            event_id,
            datetime.now(UTC).isoformat(),
            session_id,
            user_query,
            resolved_query,
            referent_found,
            resolution_status,
            latency_ms,
        )
        try:
            self._write_queue.put_nowait((self._write_generation, sql, params))
        except queue.Full:
            pass

    def record_cache_event(
        self,
        cache_type: str,
        event_type: str,
        latency_ms: float,
        key_hash: str | None = None,
        model_name: str | None = None,
    ) -> None:
        event_id = f"ce_{uuid.uuid4().hex[:8]}"
        sql = """
            INSERT INTO cache_events (
                id, timestamp, cache_type, event_type, latency_ms, key_hash, model_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            event_id,
            datetime.now(UTC).isoformat(),
            cache_type,
            event_type,
            latency_ms,
            key_hash,
            model_name,
        )
        try:
            self._write_queue.put_nowait((self._write_generation, sql, params))
        except queue.Full:
            pass

    def record_error_incident(self, incident: ErrorIncident) -> None:
        sql = """
            INSERT INTO error_incidents (
                incident_id, timestamp, request_id, document_id, conversation_id,
                component, severity, message, duration_ms, retry_count, stack_trace, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            incident.incident_id,
            incident.timestamp,
            incident.request_id,
            incident.document_id,
            incident.conversation_id,
            incident.component,
            incident.severity.value,
            incident.message,
            incident.duration_ms,
            incident.retry_count,
            incident.stack_trace,
            json.dumps(incident.details),
        )
        try:
            self._write_queue.put_nowait((self._write_generation, sql, params))
        except queue.Full:
            pass

    def record_ingestion_event(self, trace: DocumentIngestionTrace) -> None:
        sql = """
            INSERT OR REPLACE INTO ingestion_events (
                document_id, timestamp, filename, category, file_size_bytes,
                status, current_stage, pages_count, chunks_count, visual_assets_count,
                vision_success_count, vision_failed_count, total_duration_ms, error, raw_stages_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            trace.document_id,
            trace.created_at,
            trace.filename,
            trace.category,
            trace.file_size_bytes,
            trace.status,
            trace.current_stage,
            trace.pages_count,
            trace.chunks_count,
            trace.visual_assets_count,
            trace.vision_success_count,
            trace.vision_failed_count,
            trace.total_duration_ms,
            trace.error,
            json.dumps([s.model_dump() for s in trace.stages]),
        )
        try:
            self._write_queue.put_nowait((self._write_generation, sql, params))
        except queue.Full:
            pass

    def _parse_time_range(self, time_range: str) -> str:
        now = datetime.now(UTC)
        tr = (time_range or "24h").strip().lower()
        if tr in ("live", "3s"):
            delta = timedelta(seconds=3)
        elif tr.endswith("s") and tr[:-1].isdigit():
            delta = timedelta(seconds=int(tr[:-1]))
        elif tr.endswith("m") and tr[:-1].isdigit():
            delta = timedelta(minutes=int(tr[:-1]))
        elif tr.endswith("h") and tr[:-1].isdigit():
            delta = timedelta(hours=int(tr[:-1]))
        elif tr.endswith("d") and tr[:-1].isdigit():
            delta = timedelta(days=int(tr[:-1]))
        else:
            delta = timedelta(hours=24)

        return (now - delta).isoformat()

    def get_filtered_traces(
        self,
        time_range: str = "24h",
        document_id: str | None = None,
        conversation_id: str | None = None,
        intent: str | None = None,
        model: str | None = None,
        status: str | None = None,
        grounding: str | None = None,
        vision: str | None = None,
        cache: str | None = None,
        has_error: bool | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> tuple[list[QueryTraceRecord], int]:
        cutoff_iso = self._parse_time_range(time_range)
        conditions = ["timestamp >= ?"]
        params: list[Any] = [cutoff_iso]

        if document_id:
            conditions.append("(document_id = ? OR active_document_name = ?)")
            params.extend([document_id, document_id])
        if conversation_id:
            conditions.append("conversation_id = ?")
            params.append(conversation_id)
        if intent:
            conditions.append("LOWER(query_type) = ?")
            params.append(intent.lower())
        if model:
            conditions.append("LOWER(generation_model) LIKE ?")
            params.append(f"%{model.lower()}%")
        if status:
            if status.lower() == "success":
                conditions.append("error IS NULL")
            elif status.lower() == "error":
                conditions.append("error IS NOT NULL")
        if grounding:
            if grounding.lower() == "grounded":
                conditions.append("faithfulness_passed = 1 AND conversational_bypass = 0")
            elif grounding.lower() == "bypass":
                conditions.append("conversational_bypass = 1")
            elif grounding.lower() == "failed":
                conditions.append("faithfulness_passed = 0")
        if vision:
            if vision.lower() in ("used", "true", "1"):
                conditions.append("vision_used = 1")
            elif vision.lower() in ("unused", "false", "0"):
                conditions.append("vision_used = 0")
        if cache:
            if cache.lower() in ("hit", "true", "1"):
                conditions.append("cache_hit = 1")
            elif cache.lower() in ("miss", "false", "0"):
                conditions.append("cache_hit = 0")
        if has_error is True:
            conditions.append("error IS NOT NULL")
        elif has_error is False:
            conditions.append("error IS NULL")

        where_clause = " AND ".join(conditions)
        conn = self._get_connection()
        try:
            count_cur = conn.execute(
                f"SELECT COUNT(*) FROM query_traces WHERE {where_clause}", tuple(params)
            )
            total_count = count_cur.fetchone()[0]

            query_sql = f"""
                SELECT raw_trace_json FROM query_traces
                WHERE {where_clause}
                ORDER BY timestamp DESC
            """
            query_params = list(params)
            if limit is not None:
                query_sql += " LIMIT ? OFFSET ?"
                query_params.extend([limit, offset])
            cur = conn.execute(query_sql, tuple(query_params))
            rows = cur.fetchall()

            traces: list[QueryTraceRecord] = []
            for r in rows:
                try:
                    data = json.loads(r["raw_trace_json"])
                    traces.append(QueryTraceRecord.model_validate(data))
                except Exception:
                    pass
            return traces, total_count
        finally:
            conn.close()

    def get_trace_by_id_or_request_id(self, identifier: str) -> QueryTraceRecord | None:
        conn = self._get_connection()
        try:
            cur = conn.execute(
                "SELECT raw_trace_json FROM query_traces WHERE trace_id = ? OR request_id = ? LIMIT 1",
                (identifier, identifier),
            )
            row = cur.fetchone()
            if row:
                data = json.loads(row["raw_trace_json"])
                return QueryTraceRecord.model_validate(data)
            return None
        finally:
            conn.close()

    def get_recent_incidents(
        self,
        time_range: str = "24h",
        component: str | None = None,
        severity: str | None = None,
        limit: int = 50,
    ) -> list[ErrorIncident]:
        cutoff_iso = self._parse_time_range(time_range)
        conditions = ["timestamp >= ?"]
        params: list[Any] = [cutoff_iso]

        if component:
            conditions.append("LOWER(component) = ?")
            params.append(component.lower())
        if severity:
            conditions.append("LOWER(severity) = ?")
            params.append(severity.lower())

        where_clause = " AND ".join(conditions)
        conn = self._get_connection()
        try:
            cur = conn.execute(
                f"""
                SELECT * FROM error_incidents
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                tuple(params + [limit]),
            )
            rows = cur.fetchall()
            incidents: list[ErrorIncident] = []
            for r in rows:
                try:
                    details = json.loads(r["details_json"]) if r["details_json"] else {}
                except Exception:
                    details = {}
                incidents.append(
                    ErrorIncident(
                        incident_id=r["incident_id"],
                        timestamp=r["timestamp"],
                        request_id=r["request_id"],
                        document_id=r["document_id"],
                        conversation_id=r["conversation_id"],
                        component=r["component"],
                        severity=SeverityLevel(r["severity"])
                        if r["severity"] in SeverityLevel._value2member_map_
                        else SeverityLevel.ERROR,
                        message=r["message"],
                        duration_ms=r["duration_ms"],
                        retry_count=r["retry_count"],
                        stack_trace=r["stack_trace"],
                        details=details,
                    )
                )
            return incidents
        finally:
            conn.close()

    def get_vision_failures(self, time_range: str = "24h", limit: int = 20) -> list[VisionFailureRecord]:
        cutoff_iso = self._parse_time_range(time_range)
        conn = self._get_connection()
        try:
            cur = conn.execute(
                """
                SELECT * FROM vision_events
                WHERE timestamp >= ? AND status != 'SUCCESS' AND status != 'CACHE_HIT'
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (cutoff_iso, limit),
            )
            rows = cur.fetchall()
            failures: list[VisionFailureRecord] = []
            for r in rows:
                failures.append(
                    VisionFailureRecord(
                        id=r["id"],
                        timestamp=r["timestamp"],
                        document_id=r["document_id"],
                        page_number=r["page_number"],
                        visual_type=r["visual_type"] or "unknown",
                        error_type=r["status"],
                        duration_ms=r["duration_ms"],
                        request_id=r["request_id"],
                        message=r["message"] or "",
                    )
                )
            return failures
        finally:
            conn.close()

    def get_recent_resolutions(self, time_range: str = "24h", limit: int = 20) -> list[MemoryResolutionEvent]:
        cutoff_iso = self._parse_time_range(time_range)
        conn = self._get_connection()
        try:
            cur = conn.execute(
                """
                SELECT * FROM memory_events
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (cutoff_iso, limit),
            )
            rows = cur.fetchall()
            events: list[MemoryResolutionEvent] = []
            for r in rows:
                events.append(
                    MemoryResolutionEvent(
                        id=r["id"],
                        timestamp=r["timestamp"],
                        session_id=r["session_id"],
                        user_query=r["user_query"],
                        resolved_query=r["resolved_query"],
                        referent_found=r["referent_found"],
                        resolution_status=r["resolution_status"],
                        latency_ms=r["latency_ms"],
                    )
                )
            return events
        finally:
            conn.close()

    def get_ingestion_traces(self, limit: int = 20) -> list[DocumentIngestionTrace]:
        conn = self._get_connection()
        try:
            cur = conn.execute(
                """
                SELECT * FROM ingestion_events
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()
            traces: list[DocumentIngestionTrace] = []
            for r in rows:
                try:
                    stages_data = json.loads(r["raw_stages_json"]) if r["raw_stages_json"] else []
                    stages = [IngestionStageTelemetry(**s) for s in stages_data]
                except Exception:
                    stages = []
                traces.append(
                    DocumentIngestionTrace(
                        document_id=r["document_id"],
                        filename=r["filename"],
                        category=r["category"],
                        file_size_bytes=r["file_size_bytes"],
                        status=r["status"],
                        current_stage=r["current_stage"],
                        pages_count=r["pages_count"],
                        chunks_count=r["chunks_count"],
                        sections_count=0,
                        visual_assets_count=r["visual_assets_count"],
                        vision_success_count=r["vision_success_count"],
                        vision_failed_count=r["vision_failed_count"],
                        created_at=r["timestamp"],
                        total_duration_ms=r["total_duration_ms"],
                        error=r["error"],
                        stages=stages,
                    )
                )
            return traces
        finally:
            conn.close()

    def compute_cache_metrics(self, time_range: str = "24h") -> dict[str, CacheTypeMetrics]:
        cutoff_iso = self._parse_time_range(time_range)
        conn = self._get_connection()
        caches: dict[str, CacheTypeMetrics] = {
            "Semantic Cache": CacheTypeMetrics(name="Semantic Cache"),
            "Embedding Cache": CacheTypeMetrics(name="Embedding Cache"),
            "Retrieval Cache": CacheTypeMetrics(name="Retrieval Cache"),
            "Vision Cache": CacheTypeMetrics(name="Vision Cache"),
            "Negative Vision Cache": CacheTypeMetrics(name="Negative Vision Cache"),
        }
        try:
            cur = conn.execute(
                """
                SELECT cache_type, event_type, COUNT(*) as cnt, AVG(latency_ms) as avg_lat
                FROM cache_events
                WHERE timestamp >= ?
                GROUP BY cache_type, event_type
                """,
                (cutoff_iso,),
            )
            for row in cur.fetchall():
                c_name = row["cache_type"]
                if c_name not in caches:
                    caches[c_name] = CacheTypeMetrics(name=c_name)
                cm = caches[c_name]
                evt = row["event_type"].upper()
                cnt = int(row["cnt"])
                avg_lat = round(float(row["avg_lat"] or 0.0), 2)
                if evt == "HIT":
                    cm.hits += cnt
                    cm.avg_hit_latency_ms = avg_lat
                elif evt == "MISS":
                    cm.misses += cnt
                    cm.avg_miss_latency_ms = avg_lat
                elif evt == "EVICT":
                    cm.evictions += cnt

            # Compute hit rates
            for cm in caches.values():
                total = cm.hits + cm.misses
                if total > 0:
                    cm.hit_rate = round(cm.hits / total, 4)
            return caches
        finally:
            conn.close()

    def compute_aggregates(self, time_range: str = "24h", document_id: str | None = None) -> dict[str, Any]:
        cutoff_iso = self._parse_time_range(time_range)
        conn = self._get_connection()
        try:
            cur = conn.execute(
                """
                SELECT
                    COUNT(*) as total_queries,
                    AVG(execution_time_ms) as avg_lat,
                    AVG(ttft_ms) as avg_ttft,
                    AVG(tokens_per_second) as avg_tps,
                    AVG(prompt_tokens) as avg_prompt_tok,
                    AVG(completion_tokens) as avg_comp_tok,
                    SUM(prompt_tokens) as sum_prompt_tok,
                    SUM(completion_tokens) as sum_comp_tok,
                    SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as err_count,
                    AVG(candidate_count) as avg_candidates,
                    AVG(final_chunk_count) as avg_final_chunks,
                    AVG(evidence_text_count) as avg_text_cnt,
                    AVG(evidence_code_count) as avg_code_cnt,
                    AVG(evidence_diagram_count) as avg_diag_cnt,
                    AVG(evidence_table_count) as avg_tab_cnt,
                    AVG(CASE WHEN conversational_bypass = 1 THEN 1.0 ELSE verification_score END) as avg_ver_score,
                    SUM(CASE WHEN retrieval_required = 1 AND candidate_count > 0 THEN 1 ELSE 0 END) as hit_count,
                    SUM(CASE WHEN retrieval_required = 1 THEN 1 ELSE 0 END) as ret_req_count,
                    SUM(CASE WHEN section_expansion_used = 1 THEN 1 ELSE 0 END) as exp_count,
                    SUM(CASE WHEN vision_used = 1 THEN 1 ELSE 0 END) as vis_count
                FROM query_traces
                WHERE timestamp >= ?
                """,
                (cutoff_iso,),
            )
            row = cur.fetchone()

            # Latency percentiles
            lat_cur = conn.execute(
                "SELECT execution_time_ms FROM query_traces WHERE timestamp >= ? ORDER BY execution_time_ms ASC",
                (cutoff_iso,),
            )
            all_lats = [r[0] for r in lat_cur.fetchall()]

            p50_lat = None
            p95_lat = None
            p99_lat = None
            if all_lats:
                n = len(all_lats)
                p50_lat = round(all_lats[int(n * 0.50)], 2)
                p95_lat = round(all_lats[min(n - 1, int(n * 0.95))], 2)
                p99_lat = round(all_lats[min(n - 1, int(n * 0.99))], 2)

            # Vision specific metrics
            vis_conditions = ["timestamp >= ?"]
            vis_params: list[Any] = [cutoff_iso]
            if document_id:
                vis_conditions.append("document_id = ?")
                vis_params.append(document_id)
            vis_where = " AND ".join(vis_conditions)
            vis_cur = conn.execute(
                f"""
                SELECT
                    COUNT(*) as reqs,
                    SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as succ,
                    SUM(CASE WHEN status = 'TIMEOUT' THEN 1 ELSE 0 END) as tout,
                    SUM(CASE WHEN status = 'CACHE_HIT' THEN 1 ELSE 0 END) as chit,
                    SUM(CASE WHEN status NOT IN ('SUCCESS', 'CACHE_HIT') THEN 1 ELSE 0 END) as failures,
                    COUNT(DISTINCT CASE
                        WHEN page_number IS NOT NULL
                            THEN COALESCE(document_id, '') || ':' || CAST(page_number AS TEXT)
                        ELSE id
                    END) as detected_pages,
                    SUM(CASE WHEN LOWER(visual_type) LIKE '%diagram%' OR LOWER(visual_type) LIKE '%architecture%' THEN 1 ELSE 0 END) as diagrams,
                    SUM(CASE WHEN LOWER(visual_type) LIKE '%code%' OR LOWER(visual_type) LIKE '%screenshot%' THEN 1 ELSE 0 END) as code_screenshots,
                    SUM(CASE WHEN LOWER(visual_type) LIKE '%table%' THEN 1 ELSE 0 END) as tables,
                    AVG(duration_ms) as avg_lat
                FROM vision_events
                WHERE {vis_where}
                """,
                tuple(vis_params),
            )
            vis_row = vis_cur.fetchone()

            # Vision percentiles
            vis_lat_cur = conn.execute(
                f"SELECT duration_ms FROM vision_events WHERE {vis_where} ORDER BY duration_ms ASC",
                tuple(vis_params),
            )
            vis_lats = [r[0] for r in vis_lat_cur.fetchall()]
            vis_p95 = round(vis_lats[min(len(vis_lats) - 1, int(len(vis_lats) * 0.95))], 2) if vis_lats else None

            # Memory aggregates
            mem_cur = conn.execute(
                """
                SELECT
                    COUNT(DISTINCT session_id) as act_sess,
                    COUNT(*) as total_mem_events,
                    SUM(CASE WHEN resolution_status = 'SUCCESS' THEN 1 ELSE 0 END) as succ_res,
                    SUM(CASE WHEN referent_found IS NOT NULL AND TRIM(referent_found) != '' THEN 1 ELSE 0 END) as hit_count,
                    AVG(latency_ms) as avg_mem_lat
                FROM memory_events
                WHERE timestamp >= ?
                """,
                (cutoff_iso,),
            )
            mem_row = mem_cur.fetchone()

            return {
                "total_queries": row["total_queries"] or 0,
                "avg_latency_ms": round(row["avg_lat"], 2) if row["avg_lat"] is not None else None,
                "p50_latency_ms": p50_lat,
                "p95_latency_ms": p95_lat,
                "p99_latency_ms": p99_lat,
                "avg_ttft_ms": round(row["avg_ttft"], 2) if row["avg_ttft"] is not None else None,
                "avg_tokens_per_second": round(row["avg_tps"], 1) if row["avg_tps"] is not None else None,
                "avg_prompt_tokens": round(row["avg_prompt_tok"], 1) if row["avg_prompt_tok"] is not None else 0.0,
                "avg_completion_tokens": round(row["avg_comp_tok"], 1) if row["avg_comp_tok"] is not None else 0.0,
                "sum_prompt_tokens": row["sum_prompt_tok"] or 0,
                "sum_completion_tokens": row["sum_comp_tok"] or 0,
                "error_rate": round(row["err_count"] / max(1, row["total_queries"]), 4) if row["total_queries"] else 0.0,
                "avg_candidates": round(row["avg_candidates"], 1) if row["avg_candidates"] is not None else 0.0,
                "avg_final_chunks": round(row["avg_final_chunks"], 1) if row["avg_final_chunks"] is not None else 0.0,
                "avg_verification_score": round(row["avg_ver_score"], 4) if row["avg_ver_score"] is not None else 0.0,
                "hit_rate": round(row["hit_count"] / max(1, row["ret_req_count"]), 4) if row["ret_req_count"] else 0.0,
                "vision_reqs": vis_row["reqs"] or 0,
                "vision_success": vis_row["succ"] or 0,
                "vision_timeouts": vis_row["tout"] or 0,
                "vision_cache_hits": vis_row["chit"] or 0,
                "vision_failures": vis_row["failures"] or 0,
                "visual_pages_detected": vis_row["detected_pages"] or 0,
                "vision_diagrams": vis_row["diagrams"] or 0,
                "vision_code_screenshots": vis_row["code_screenshots"] or 0,
                "vision_tables": vis_row["tables"] or 0,
                "vision_avg_lat": round(vis_row["avg_lat"], 2) if vis_row["avg_lat"] is not None else None,
                "vision_p95_lat": vis_p95,
                "active_sessions": mem_row["act_sess"] or 0,
                "memory_events_count": mem_row["total_mem_events"] or 0,
                "memory_hit_rate": round(mem_row["hit_count"] / mem_row["total_mem_events"], 4) if mem_row["total_mem_events"] else None,
                "memory_resolution_rate": round(mem_row["succ_res"] / mem_row["total_mem_events"], 4) if mem_row["total_mem_events"] else None,
                "avg_memory_latency_ms": round(mem_row["avg_mem_lat"], 2) if mem_row["avg_mem_lat"] is not None else None,
            }
        finally:
            conn.close()

    def compute_time_series(self, time_range: str = "24h", bucket_count: int = 12) -> list[TimeSeriesPoint]:
        cutoff_iso = self._parse_time_range(time_range)
        conn = self._get_connection()
        points: list[TimeSeriesPoint] = []
        try:
            # Query traces in time window
            cur = conn.execute(
                """
                SELECT timestamp, execution_time_ms, prompt_tokens, completion_tokens,
                       final_chunk_count, verification_score, error
                FROM query_traces
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
                """,
                (cutoff_iso,),
            )
            rows = cur.fetchall()
            if not rows:
                return []

            # Group into time buckets
            start_dt = datetime.fromisoformat(cutoff_iso)
            end_dt = datetime.now(UTC)
            total_sec = max(1.0, (end_dt - start_dt).total_seconds())
            bucket_sec = total_sec / bucket_count

            buckets: list[list[Any]] = [[] for _ in range(bucket_count)]
            for r in rows:
                try:
                    dt = datetime.fromisoformat(r["timestamp"])
                    idx = int((dt - start_dt).total_seconds() / bucket_sec)
                    idx = min(bucket_count - 1, max(0, idx))
                    buckets[idx].append(r)
                except Exception:
                    pass

            for i, b_rows in enumerate(buckets):
                b_time = (start_dt + timedelta(seconds=i * bucket_sec)).isoformat()
                if not b_rows:
                    points.append(TimeSeriesPoint(timestamp=b_time))
                    continue

                lats = sorted([r["execution_time_ms"] for r in b_rows])
                p50 = round(lats[int(len(lats) * 0.50)], 2)
                p95 = round(lats[min(len(lats) - 1, int(len(lats) * 0.95))], 2)
                p99 = round(lats[min(len(lats) - 1, int(len(lats) * 0.99))], 2)
                p_toks = sum(r["prompt_tokens"] for r in b_rows)
                c_toks = sum(r["completion_tokens"] for r in b_rows)
                avg_chunks = round(sum(r["final_chunk_count"] for r in b_rows) / len(b_rows), 1)
                scores = [r["verification_score"] for r in b_rows if r["verification_score"] is not None]
                avg_rerank = round(sum(scores) / len(scores), 3) if scores else 0.0
                errs = sum(1 for r in b_rows if r["error"] is not None)

                points.append(
                    TimeSeriesPoint(
                        timestamp=b_time,
                        latency_p50_ms=p50,
                        latency_p95_ms=p95,
                        latency_p99_ms=p99,
                        requests_count=len(b_rows),
                        prompt_tokens=p_toks,
                        completion_tokens=c_toks,
                        avg_chunks=avg_chunks,
                        avg_rerank_score=avg_rerank,
                        errors_count=errs,
                    )
                )
            return points
        finally:
            conn.close()

    def delete_query_trace(self, identifier: str) -> bool:
        """Delete a single query trace by trace_id or request_id."""
        with self._lock:
            self._deleted_trace_identifiers.add(identifier)
            conn = self._get_connection()
            try:
                with conn:
                    cursor = conn.execute(
                        "DELETE FROM query_traces WHERE trace_id = ? OR request_id = ?;",
                        (identifier, identifier),
                    )
                    return cursor.rowcount > 0
            finally:
                conn.close()

    def clear(self) -> None:
        """Purge all telemetry database records."""
        with self._lock:
            self._write_generation += 1
            self._deleted_trace_identifiers.clear()
            while True:
                try:
                    self._write_queue.get_nowait()
                except queue.Empty:
                    break
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute("DELETE FROM query_traces;")
                    conn.execute("DELETE FROM vision_events;")
                    conn.execute("DELETE FROM memory_events;")
                    conn.execute("DELETE FROM cache_events;")
                    conn.execute("DELETE FROM error_incidents;")
                    conn.execute("DELETE FROM ingestion_events;")
            finally:
                conn.close()
