from __future__ import annotations

import collections
import statistics
import threading

from backend.models.api_dto import ObservabilityMetrics, TraceSummary
from backend.models.rag import RAGResponse, ScoredChunk


class TelemetryService:
    """
    In-memory circular buffer (up to 1,000 traces) tracking query counts, token totals,
    latency metrics, stage timings, vector similarity scores, BM25 scores, RRF scores,
    rerank logits, and sources used.
    """

    def __init__(self, max_traces: int = 1000) -> None:
        self.max_traces = max_traces
        self._traces: collections.deque[TraceSummary] = collections.deque(maxlen=max_traces)
        self._lock = threading.Lock()

        # Aggregated metrics counters
        self._total_queries: int = 0
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._all_latencies: collections.deque[float] = collections.deque(maxlen=max_traces)
        self._all_ttfts: collections.deque[float] = collections.deque(maxlen=max_traces)
        self._all_similarity_scores: collections.deque[float] = collections.deque(maxlen=max_traces)
        self._all_rerank_scores: collections.deque[float] = collections.deque(maxlen=max_traces)

    def record_trace(self, trace: TraceSummary) -> TraceSummary:
        """Record a pre-assembled TraceSummary object into the circular buffer."""
        with self._lock:
            self._traces.appendleft(trace)
            self._total_queries += 1

            # Accumulate tokens
            p_tok = trace.token_usage.get("prompt_tokens", 0)
            c_tok = trace.token_usage.get("completion_tokens", 0)
            self._prompt_tokens += p_tok
            self._completion_tokens += c_tok

            # Latencies
            self._all_latencies.append(trace.execution_time_ms)
            if trace.ttft_ms is not None:
                self._all_ttfts.append(trace.ttft_ms)

            # Scores
            if trace.similarity_scores:
                self._all_similarity_scores.extend(trace.similarity_scores)
            if trace.rerank_scores:
                self._all_rerank_scores.extend(trace.rerank_scores)

        return trace

    def record_from_rag_response(
        self,
        rag_response: RAGResponse,
        ttft_ms: float | None = None,
    ) -> TraceSummary:
        """Extract telemetry data from a RAGResponse and record in trace store."""
        t = rag_response.trace
        context_chunks: list[ScoredChunk] = rag_response.context_chunks or []

        similarity_scores = [sc.dense_score for sc in context_chunks if sc.dense_score is not None]
        bm25_scores = [sc.sparse_score for sc in context_chunks if sc.sparse_score is not None]
        rrf_scores = [sc.score for sc in context_chunks if sc.score is not None]
        rerank_scores = [sc.rerank_score for sc in context_chunks if sc.rerank_score is not None]

        sources_set = set()
        for citation in rag_response.citations:
            if citation.source_file:
                sources_set.add(citation.source_file)
        for sc in context_chunks:
            if sc.chunk and sc.chunk.metadata and sc.chunk.metadata.source_file:
                sources_set.add(sc.chunk.metadata.source_file)

        trace_summary = TraceSummary(
            query=rag_response.query,
            rewritten_query=t.rewritten_query if t else None,
            sub_queries=t.sub_queries if t else [],
            query_type=t.query_type if t else None,
            routing_confidence=t.routing_confidence if t else None,
            retrieval_strategy=t.retrieval_strategy if t else None,
            inferred_filters=t.inferred_filters if t else {},
            applied_filters=t.applied_filters if t else {},
            filter_relaxed=t.filter_relaxed if t else False,
            candidate_count=t.retrieved_candidate_count if t else len(context_chunks),
            post_rerank_count=t.post_rerank_count if t else len(context_chunks),
            final_context_count=t.final_context_count if t else len(context_chunks),
            execution_time_ms=t.execution_time_ms if t else rag_response.latency_ms,
            ttft_ms=ttft_ms,
            stage_timings=t.stage_timings_ms if t else {},
            similarity_scores=similarity_scores,
            rerank_scores=rerank_scores,
            bm25_scores=bm25_scores,
            rrf_scores=rrf_scores,
            sources_used=sorted(list(sources_set)),
            token_usage=rag_response.token_usage or {},
            faithfulness_passed=t.faithfulness_passed if t else True,
            verification=t.verification_report if t else None,
            verification_score=t.verification_score if t else None,
            retry_count=t.retry_count if t else 0,
            retry_reasons=t.retry_reasons if t else [],
        )

        return self.record_trace(trace_summary)

    def get_metrics(
        self,
        recent_limit: int = 20,
        active_documents: int = 0,
        indexed_chunks: int = 0,
    ) -> ObservabilityMetrics:
        """Compute aggregated telemetry metrics and return summary overview."""
        with self._lock:
            total_q = self._total_queries
            avg_lat = round(statistics.mean(self._all_latencies), 2) if self._all_latencies else 0.0

            if self._all_latencies:
                sorted_lat = sorted(self._all_latencies)
                p95_idx = int(len(sorted_lat) * 0.95)
                p95_lat = round(sorted_lat[min(p95_idx, len(sorted_lat) - 1)], 2)
            else:
                p95_lat = 0.0

            avg_ttft = round(statistics.mean(self._all_ttfts), 2) if self._all_ttfts else 0.0

            sim_avg = round(statistics.mean(self._all_similarity_scores), 4) if self._all_similarity_scores else 0.0
            rerank_avg = round(statistics.mean(self._all_rerank_scores), 4) if self._all_rerank_scores else 0.0

            recent_list = list(self._traces)[:recent_limit]

            tot_prompt = self._prompt_tokens
            tot_compl = self._completion_tokens

            return ObservabilityMetrics(
                total_queries=total_q,
                avg_latency_ms=avg_lat,
                avg_ttft_ms=avg_ttft,
                p95_latency_ms=p95_lat,
                token_usage={
                    "prompt_tokens": tot_prompt,
                    "completion_tokens": tot_compl,
                    "total_tokens": tot_prompt + tot_compl,
                },
                score_distributions={
                    "similarity_avg": sim_avg,
                    "rerank_avg": rerank_avg,
                },
                active_documents=active_documents,
                indexed_chunks=indexed_chunks,
                recent_traces=recent_list,
            )

    def get_recent_traces(self, limit: int = 50, offset: int = 0) -> list[TraceSummary]:
        """Return slices of stored traces from newest to oldest."""
        with self._lock:
            all_traces = list(self._traces)
            return all_traces[offset : offset + limit]

    def get_trace_by_id(self, trace_id: str) -> TraceSummary | None:
        """Find a specific trace by trace_id."""
        with self._lock:
            for t in self._traces:
                if t.trace_id == trace_id:
                    return t
            return None

    def clear(self) -> None:
        """Reset telemetry buffer (useful for unit test isolation)."""
        with self._lock:
            self._traces.clear()
            self._total_queries = 0
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._all_latencies.clear()
            self._all_ttfts.clear()
            self._all_similarity_scores.clear()
            self._all_rerank_scores.clear()
