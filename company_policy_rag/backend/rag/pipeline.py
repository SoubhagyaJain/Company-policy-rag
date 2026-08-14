from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Generator
from typing import Any

from backend.models.chunk import Chunk
from backend.models.rag import (
    Citation,
    QueryCategory,
    QueryClassification,
    RAGResponse,
    RAGTrace,
    RetrievalStrategy,
    ScoredChunk,
    VerificationReport,
)
from backend.rag.citations import CitationEngine
from backend.rag.context_compression import ContextCompressor
from backend.rag.filter_extractor import QueryMetadataInferer
from backend.rag.multi_query import MultiQueryGenerator
from backend.rag.query_rewrite import QueryRewriter
from backend.rag.query_router import QueryRouter
from backend.rag.retry_engine import RetryEngine
from backend.rag.semantic_cache import SemanticCacheManager
from backend.rag.verifier import SelfReflectionVerifier
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.reranker import CrossEncoderReranker
from backend.utils.logging import logger
from src.config import settings
from src.ollama_client import preload_model, unload_model

_llm_lock = threading.Lock()


class ModelManager:
    """
    Lightweight, stateless model router modeled after Antigravity.
    Manages active model pointers and non-blocking background preloading
    without thread starvation, write-lock bottlenecks, or reader deadlocks.
    """

    def __init__(self, initial_model: str):
        self.current_model = initial_model

    def set_model(self, model_name: str) -> None:
        """Update active model and optionally trigger non-blocking background preload."""
        self.current_model = model_name

        def _bg_preload():
            try:
                preload_model(model_name)
            except Exception as e:
                logger.warning("Background preload for %s failed: %s", model_name, e)

        try:
            threading.Thread(target=_bg_preload, daemon=True).start()
        except Exception:
            pass


class _LLMProxy:
    """Per-request thread-safe wrapper overriding the model attribute for shared LLM instances."""

    def __init__(self, target_llm: Any, target_model: str) -> None:
        self._target_llm = target_llm
        self.model = target_model

    def complete(self, prompt: str, **kwargs: Any) -> Any:
        with _llm_lock:
            old_model = getattr(self._target_llm, "model", None)
            try:
                if hasattr(self._target_llm, "model"):
                    self._target_llm.model = self.model
                return self._target_llm.complete(prompt, **kwargs)
            finally:
                if hasattr(self._target_llm, "model") and old_model is not None:
                    self._target_llm.model = old_model

    def stream_complete(self, prompt: str, **kwargs: Any) -> Any:
        with _llm_lock:
            old_model = getattr(self._target_llm, "model", None)
            try:
                if hasattr(self._target_llm, "model"):
                    self._target_llm.model = self.model
                gen = self._target_llm.stream_complete(prompt, **kwargs)
                first = None
                try:
                    first = next(gen)
                except StopIteration:
                    pass
            finally:
                if hasattr(self._target_llm, "model") and old_model is not None:
                    self._target_llm.model = old_model

        def wrapper():
            if first is not None:
                yield first
            yield from gen

        return wrapper()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target_llm, name)


GROUNDED_SYSTEM_PROMPT = """You are an accurate, helpful AI assistant.
Your instructions:
1. Answer the user's question using the provided document sources below and relevant conversation history.
2. For EVERY key claim or fact in your response, cite the source number using exact bracket format [Source N] (e.g. [Source 1], [Source 2]).
3. If the provided sources do not contain sufficient information to answer the question, state clearly: "I am unable to answer based on the provided documents."
4. Do not invent, hallucinate, or extrapolate facts beyond the sources.
{refinement_directive}
Document Sources:
{context_text}

{history_text}User Question: {query}
Answer:"""


def _format_history_for_prompt(history: list[dict[str, Any]] | None, max_turns: int = 6) -> str:
    if not history:
        return ""
    recent = history[-(max_turns * 2) :]
    lines = ["Recent Conversation History:"]
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = str(msg.get("content", "")).strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines) + "\n\n"


class RAGPipeline:
    """
    Master end-to-end RAG Pipeline orchestrating query rewrite, multi-query decomposition,
    hybrid dense+sparse search, cross-encoder reranking, parent context expansion,
    LLM grounded synthesis, structured citation extraction, self-reflection verification,
    autonomous retry loops, semantic caching, and telemetry traces.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker | None = None,
        query_rewriter: QueryRewriter | None = None,
        query_router: QueryRouter | None = None,
        multi_query_gen: MultiQueryGenerator | None = None,
        compressor: ContextCompressor | None = None,
        citation_engine: CitationEngine | None = None,
        docstore: dict[str, Chunk] | None = None,
        llm: Any | None = None,
        semantic_cache: SemanticCacheManager | None = None,
        verifier: SelfReflectionVerifier | None = None,
        retry_engine: RetryEngine | None = None,
        filter_inferer: QueryMetadataInferer | None = None,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.query_router = query_router or QueryRouter()
        self.multi_query_gen = multi_query_gen or MultiQueryGenerator()
        self.compressor = compressor or ContextCompressor()
        self.citation_engine = citation_engine or CitationEngine()
        self.docstore = docstore or {}
        self.llm = llm
        self.semantic_cache = semantic_cache
        self.verifier = verifier or SelfReflectionVerifier(llm=self.llm)
        self.retry_engine = retry_engine or RetryEngine()
        self.filter_inferer = filter_inferer or QueryMetadataInferer()
        self.model_manager = ModelManager(
            initial_model=getattr(self.llm, "model", None) or "qwen2.5:7b"
        )

        if self.query_rewriter.llm is None and self.llm is not None:
            self.query_rewriter.llm = self.llm
        if self.query_router.llm is None and self.llm is not None:
            self.query_router.llm = self.llm
        if self.verifier.llm is None and self.llm is not None:
            self.verifier.llm = self.llm

    def set_active_model(self, model: str) -> str:
        """Switch the backend pipeline to a new model without blocking."""
        self.model_manager.set_model(model)
        return model

    def get_active_model(self) -> str:
        """Return the currently configured generation model."""
        return self.model_manager.current_model

    def _queue_cache_write(
        self,
        user_query: str,
        answer: str,
        citations: list[Any],
        kb_version: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Queue asynchronous non-blocking background cache write."""
        if not self.semantic_cache:
            return

        def _async_put():
            try:
                self.semantic_cache.put(
                    query=user_query,
                    answer=answer,
                    citations=citations,
                    kb_version=kb_version,
                    model_name=model_name,
                )
            except Exception as exc:
                logger.warning("Background cache write error: %s", exc)

        try:
            thread = threading.Thread(target=_async_put, daemon=True)
            thread.start()
        except Exception as exc:
            logger.warning("Failed to start background cache write thread: %s", exc)

    def _get_effective_llm(self, model: str | None) -> tuple[Any | None, str]:
        """Return a per-request thread-safe LLM instance and model name."""
        base_model = self.model_manager.current_model or "qwen2.5:7b"
        selected_model = (model or base_model).strip()
        if not selected_model:
            selected_model = "qwen2.5:7b"

        if self.llm is None:
            return None, selected_model

        if not hasattr(self, "_llm_instance_cache"):
            self._llm_instance_cache: dict[str, Any] = {}

        if selected_model not in self._llm_instance_cache:
            if hasattr(self.llm, "model") and getattr(self.llm, "model") == selected_model:
                self._llm_instance_cache[selected_model] = self.llm
            else:
                try:
                    from llama_index.llms.ollama import Ollama

                    base_url = getattr(self.llm, "base_url", "http://localhost:11434")
                    temperature = getattr(self.llm, "temperature", 0.1)
                    request_timeout = getattr(self.llm, "request_timeout", 120.0)
                    self._llm_instance_cache[selected_model] = Ollama(
                        base_url=base_url,
                        model=selected_model,
                        temperature=temperature,
                        request_timeout=request_timeout,
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not instantiate Ollama for model %s: %s", selected_model, exc
                    )
                    self._llm_instance_cache[selected_model] = self.llm

        return self._llm_instance_cache[selected_model], selected_model

    def query(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> RAGResponse:
        """Execute end-to-end RAG pipeline and return structured RAGResponse with trace telemetry."""
        return self._query_internal(user_query, filters, history, model)

    def _query_internal(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> RAGResponse:
        total_start = time.perf_counter()
        stage_timings: dict[str, float] = {}

        # 0a. Query Classification & Dynamic Strategy Selection
        t0 = time.perf_counter()
        classification = self.query_router.classify(user_query, history=history)
        strategy = classification.strategy
        stage_timings["query_routing"] = round((time.perf_counter() - t0) * 1000, 2)

        req_llm, selected_model = self._get_effective_llm(model)

        # Conversational / Greeting intent check: bypass vector DB and fake citations
        if classification.category == QueryCategory.CONVERSATIONAL or self.query_rewriter.is_conversational(
            user_query
        ):
            greeting_answer = (
                "Hello! How can I assist you today? Feel free to ask any questions regarding company policies, "
                "employee benefits, travel expenses, code of conduct, or any uploaded documentation."
            )
            total_elapsed = round((time.perf_counter() - total_start) * 1000, 2)
            stage_timings["conversational_bypass"] = total_elapsed
            trace = RAGTrace(
                query=user_query,
                rewritten_query=None,
                sub_queries=[],
                query_type=classification.category.value,
                routing_confidence=classification.confidence,
                retrieval_strategy=strategy.name,
                retrieved_candidate_count=0,
                post_rerank_count=0,
                final_context_count=0,
                execution_time_ms=total_elapsed,
                stage_timings_ms=stage_timings,
                fallback_reason="conversational_greeting",
                faithfulness_checked=True,
                faithfulness_passed=True,
                verification_report=None,
                verification_score=1.0,
                retry_count=0,
                retry_reasons=[],
                cache_hit=False,
                cache_similarity=None,
            )
            return RAGResponse(
                id=f"resp_{uuid.uuid4().hex[:12]}",
                query=user_query,
                answer=greeting_answer,
                citations=[],
                context_chunks=[],
                trace=trace,
                model=selected_model,
                token_usage={
                    "prompt_tokens": 0,
                    "completion_tokens": len(greeting_answer.split()),
                },
            )

        # 0. Pre-rewrite Cache Lookup
        cache_enabled = (
            getattr(self.semantic_cache.settings, "semantic_cache_enabled", True)
            if (self.semantic_cache and hasattr(self.semantic_cache, "settings"))
            else True
        )
        if cache_enabled and self.semantic_cache is not None:
            t0 = time.perf_counter()
            cached_res = self.semantic_cache.get(user_query, model_name=selected_model)
            stage_timings["cache_lookup"] = round((time.perf_counter() - t0) * 1000, 2)
            if cached_res is not None:
                total_elapsed = round((time.perf_counter() - total_start) * 1000, 2)
                trace = RAGTrace(
                    query=user_query,
                    rewritten_query=None,
                    sub_queries=[],
                    query_type=classification.category.value,
                    routing_confidence=classification.confidence,
                    retrieval_strategy=strategy.name,
                    retrieved_candidate_count=0,
                    post_rerank_count=0,
                    final_context_count=0,
                    execution_time_ms=total_elapsed,
                    stage_timings_ms=stage_timings,
                    fallback_reason="none",
                    faithfulness_checked=True,
                    faithfulness_passed=True,
                    verification_report=None,
                    verification_score=1.0,
                    retry_count=0,
                    retry_reasons=[],
                    cache_hit=True,
                    cache_similarity=cached_res.similarity_score,
                )
                return RAGResponse(
                    id=f"resp_{uuid.uuid4().hex[:12]}",
                    query=user_query,
                    answer=cached_res.answer,
                    citations=cached_res.citations,
                    context_chunks=[],
                    trace=trace,
                    model=model or "semantic_cache",
                    token_usage={
                        "prompt_tokens": 0,
                        "completion_tokens": len(cached_res.answer.split()),
                    },
                )

        # 1. Query Rewrite
        t0 = time.perf_counter()
        rewrite_res = self.query_rewriter.rewrite(user_query, history=history, llm=req_llm)
        stage_timings["query_rewrite"] = round((time.perf_counter() - t0) * 1000, 2)

        # 1b. Query-Time Metadata Filter Inference
        inferred_filters: dict[str, Any] = {}
        applied_filters: dict[str, Any] = {}
        filter_relaxed = False
        enable_filtering = getattr(settings, "enable_query_metadata_filtering", True)
        if enable_filtering and self.filter_inferer is not None:
            t0 = time.perf_counter()
            inferred_filters = self.filter_inferer.infer_filters(
                query=user_query, history=history, explicit_filters=filters,
            )
            stage_timings["filter_inference"] = round((time.perf_counter() - t0) * 1000, 2)
            if inferred_filters:
                applied_filters = {**inferred_filters}
                logger.info("Inferred metadata filters: %s", applied_filters)
        elif filters:
            applied_filters = {**filters}

        # Autonomous Retry Loop Execution
        enable_verification = getattr(settings, "enable_answer_verification", True)
        max_retries = self.retry_engine.max_retries if self.retry_engine else 2
        current_strategy = strategy.model_copy(deep=True)
        attempt = 0

        best_answer = ""
        best_citations: list[Citation] = []
        best_context_chunks: list[ScoredChunk] = []
        best_candidate_chunks: list[ScoredChunk] = []
        best_reranked_chunks: list[ScoredChunk] = []
        best_report: VerificationReport | None = None
        best_score = -1.0

        retry_reasons: list[str] = []
        prompt_refinement = ""
        sub_queries: list[str] = [rewrite_res.rewritten_query]
        formatted_context = ""

        while attempt <= max_retries:
            prefix = f"_att{attempt}" if attempt > 0 else ""

            # 2. Multi-Query Generation
            t0 = time.perf_counter()
            if current_strategy.enable_multi_query or rewrite_res.is_comprehensive_list:
                sub_queries = self.multi_query_gen.generate_subqueries(rewrite_res.rewritten_query)
            else:
                sub_queries = [rewrite_res.rewritten_query]
            stage_timings[f"multi_query{prefix}"] = round((time.perf_counter() - t0) * 1000, 2)

            # 3. Hybrid Search (Dense + BM25 with RRF) across sub-queries
            t0 = time.perf_counter()
            search_filters = applied_filters if applied_filters else None
            candidate_map: dict[str, ScoredChunk] = {}
            for sq in sub_queries:
                hits = self.hybrid_retriever.retrieve(
                    sq,
                    dense_top_k=current_strategy.dense_top_k,
                    bm25_top_k=current_strategy.bm25_top_k,
                    filters=search_filters,
                    rrf_k=current_strategy.rrf_k,
                )
                for sc in hits:
                    cid = sc.chunk.id
                    if cid not in candidate_map or (sc.score or 0.0) > (
                        candidate_map[cid].score or 0.0
                    ):
                        candidate_map[cid] = sc
            candidate_chunks = list(candidate_map.values())

            # Filter relaxation fallback: retry without filters if zero results
            enable_relaxation = getattr(settings, "enable_filter_fallback_relaxation", True)
            if not candidate_chunks and search_filters and enable_relaxation:
                logger.warning(
                    "Filtered retrieval returned 0 candidates. Relaxing filters and retrying."
                )
                filter_relaxed = True
                applied_filters = {}
                candidate_map = {}
                for sq in sub_queries:
                    hits = self.hybrid_retriever.retrieve(
                        sq,
                        dense_top_k=current_strategy.dense_top_k,
                        bm25_top_k=current_strategy.bm25_top_k,
                        filters=None,
                        rrf_k=current_strategy.rrf_k,
                    )
                    for sc in hits:
                        cid = sc.chunk.id
                        if cid not in candidate_map or (sc.score or 0.0) > (
                            candidate_map[cid].score or 0.0
                        ):
                            candidate_map[cid] = sc
                candidate_chunks = list(candidate_map.values())
                stage_timings[f"filter_relaxation{prefix}"] = round(
                    (time.perf_counter() - t0) * 1000, 2
                )

            candidate_chunks.sort(key=lambda x: x.score or 0.0, reverse=True)
            candidate_pool_limit = max(len(candidate_chunks), current_strategy.rerank_top_n * 3, 30)
            candidate_chunks = candidate_chunks[:candidate_pool_limit]
            stage_timings[f"hybrid_retrieval{prefix}"] = round((time.perf_counter() - t0) * 1000, 2)

            # Zero retrieved chunks edge case
            if not candidate_chunks:
                unanswerable_text = "I am unable to answer based on the provided documents."
                report = VerificationReport(
                    faithfulness=1.0,
                    completeness=1.0,
                    citation_coverage=1.0,
                    coherence=1.0,
                    composite_score=1.0,
                    passed=True,
                    retry_count=attempt,
                )
                best_answer = unanswerable_text
                best_citations = []
                best_context_chunks = []
                best_candidate_chunks = []
                best_reranked_chunks = []
                best_report = report
                break

            # 4. Cross-Encoder Reranking & Relative Score Thresholding
            t0 = time.perf_counter()
            reranked_chunks = self.reranker.rerank(
                rewrite_res.rewritten_query,
                candidate_chunks,
                top_n=current_strategy.rerank_top_n,
                min_ratio=current_strategy.min_score_ratio,
            )
            stage_timings[f"reranking{prefix}"] = round((time.perf_counter() - t0) * 1000, 2)

            # 5. Parent Context Expansion & Formatting
            t0 = time.perf_counter()
            expanded_chunks = self.compressor.expand_to_parents(
                reranked_chunks,
                self.docstore,
                enable_expansion=current_strategy.enable_parent_expansion,
            )
            formatted_context = self.compressor.format_context_for_prompt(expanded_chunks)
            stage_timings[f"context_expansion{prefix}"] = round(
                (time.perf_counter() - t0) * 1000, 2
            )

            # 6. LLM Grounded Answer Synthesis
            t0 = time.perf_counter()
            answer_text = ""

            if req_llm is not None:
                try:
                    history_text = _format_history_for_prompt(history)
                    refinement_str = (
                        f"\nRefinement Instructions:\n{prompt_refinement}\n"
                        if prompt_refinement
                        else ""
                    )
                    prompt = GROUNDED_SYSTEM_PROMPT.format(
                        refinement_directive=refinement_str,
                        context_text=formatted_context,
                        history_text=history_text,
                        query=user_query,
                    )
                    try:
                        raw_answer = str(
                            req_llm.complete(prompt, temperature=current_strategy.temperature)
                        ).strip()
                    except TypeError:
                        raw_answer = str(req_llm.complete(prompt)).strip()
                    answer_text = raw_answer
                except Exception as exc:
                    logger.warning("LLM synthesis error (%s). Using fallback synthesis.", exc)
                    answer_text = self._fallback_synthesis(user_query, expanded_chunks)
            else:
                answer_text = self._fallback_synthesis(user_query, expanded_chunks)

            stage_timings[f"llm_synthesis{prefix}"] = round((time.perf_counter() - t0) * 1000, 2)

            # 7. Verifiable Citation Extraction
            t0 = time.perf_counter()
            citations = self.citation_engine.select_citations(
                answer_text=answer_text,
                generation_chunks=expanded_chunks,
                user_query=user_query,
            )
            stage_timings[f"citation_extraction{prefix}"] = round(
                (time.perf_counter() - t0) * 1000, 2
            )

            # 8. Post-Generation Verification
            t0 = time.perf_counter()
            if enable_verification and self.verifier is not None:
                report = self.verifier.verify(
                    query=user_query,
                    answer=answer_text,
                    context_chunks=expanded_chunks,
                    citations=citations,
                    llm=req_llm,
                )
                report.retry_count = attempt
            else:
                report = VerificationReport(
                    faithfulness=1.0,
                    completeness=1.0,
                    citation_coverage=1.0,
                    coherence=1.0,
                    composite_score=1.0,
                    passed=True,
                    retry_count=attempt,
                )
            stage_timings[f"verification{prefix}"] = round((time.perf_counter() - t0) * 1000, 2)

            # Update best candidate
            if report.composite_score > best_score or best_report is None:
                best_score = report.composite_score
                best_answer = answer_text
                best_citations = citations
                best_context_chunks = expanded_chunks
                best_candidate_chunks = candidate_chunks
                best_reranked_chunks = reranked_chunks
                best_report = report

            if report.passed:
                best_answer = answer_text
                best_citations = citations
                best_context_chunks = expanded_chunks
                best_candidate_chunks = candidate_chunks
                best_reranked_chunks = reranked_chunks
                best_report = report
                break

            if attempt >= max_retries or not self.retry_engine.should_retry(attempt, report):
                logger.warning(
                    "Retry hard cap reached (%d retries). Gracefully falling back to best response.",
                    attempt,
                )
                break

            # Prepare retry parameters for next iteration
            if report.critique:
                retry_reasons.append(report.critique)
            elif report.missing_aspects or report.unsupported_claims:
                parts = []
                if report.missing_aspects:
                    parts.append(f"Missing: {', '.join(report.missing_aspects)}")
                if report.unsupported_claims:
                    parts.append(f"Unsupported: {', '.join(report.unsupported_claims)}")
                retry_reasons.append("; ".join(parts))
            else:
                retry_reasons.append(
                    f"Verification score {report.composite_score:.2f} below threshold."
                )

            current_strategy, prompt_refinement = self.retry_engine.prepare_retry(
                attempt=attempt,
                report=report,
                strategy=current_strategy,
                query=user_query,
            )
            attempt += 1

        total_elapsed = round((time.perf_counter() - total_start) * 1000, 2)

        fallback_reason = "none"
        if req_llm is None:
            fallback_reason = "llm_offline_fallback"
        elif best_report is not None and not best_report.passed:
            fallback_reason = "retry_exhausted_fallback"

        trace = RAGTrace(
            query=user_query,
            rewritten_query=rewrite_res.rewritten_query,
            sub_queries=sub_queries,
            query_type=classification.category.value,
            routing_confidence=classification.confidence,
            retrieval_strategy=current_strategy.name if current_strategy else strategy.name,
            inferred_filters=inferred_filters,
            applied_filters=applied_filters,
            filter_relaxed=filter_relaxed,
            retrieved_candidate_count=len(best_candidate_chunks),
            post_rerank_count=len(best_reranked_chunks),
            final_context_count=len(best_context_chunks),
            execution_time_ms=total_elapsed,
            stage_timings_ms=stage_timings,
            fallback_reason=fallback_reason,
            faithfulness_checked=enable_verification,
            faithfulness_passed=best_report.passed if best_report else True,
            verification_report=best_report.model_dump() if best_report else None,
            verification_score=best_report.composite_score if best_report else None,
            retry_count=attempt,
            retry_reasons=retry_reasons,
            cache_hit=False,
            cache_similarity=None,
        )

        if (
            best_citations
            and len(best_citations) > 0
            and best_answer
            and (best_report is None or best_report.passed)
        ):
            self._queue_cache_write(
                user_query, best_answer, best_citations, model_name=selected_model
            )

        return RAGResponse(
            id=f"resp_{uuid.uuid4().hex[:12]}",
            query=user_query,
            answer=best_answer,
            citations=best_citations,
            context_chunks=best_context_chunks,
            trace=trace,
            model=selected_model,
            token_usage={
                "prompt_tokens": len(formatted_context.split()) if best_context_chunks else 0,
                "completion_tokens": len(best_answer.split()),
            },
        )

    def _fallback_synthesis(self, user_query: str, context_chunks: list[ScoredChunk]) -> str:
        """Deterministic grounded response fallback when LLM service is offline."""
        if not context_chunks:
            return "I am unable to answer based on the provided documents."

        paragraphs: list[str] = []
        for idx, sc in enumerate(context_chunks, start=1):
            text_snippet = sc.chunk.text.strip()
            # Pick first 2 sentences
            sentences = [s.strip() for s in text_snippet.split(".") if s.strip()]
            snippet = ". ".join(sentences[:2]) + "." if sentences else text_snippet
            paragraphs.append(f"{snippet} [Source {idx}]")

        return (
            f"Based on the official documentation regarding '{user_query}':\n\n"
            + "\n\n".join(paragraphs)
        )

    async def stream_query(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        model: str | None = None,
        cancel_token: Any = None,
    ):
        """
        Streaming RAG pipeline: performs pre-rewrite cache lookup, or runs retrieval
        synchronously and yields real-time LLM tokens via llm.stream_complete().

        Yields dicts with 'type' key:
          - {'type': 'retrieval_done', 'stage_timings': {...}, 'candidate_count': int, ...}
          - {'type': 'token', 'content': str}
          - {'type': 'done', 'answer': str, 'citations': [...], 'trace': RAGTrace, ...}
        """
        from starlette.concurrency import iterate_in_threadpool

        async for chunk in iterate_in_threadpool(
            self._stream_query_internal(user_query, filters, history, model, cancel_token)
        ):
            yield chunk

    def _stream_query_internal(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        model: str | None = None,
        cancel_token: Any = None,
    ) -> Generator[dict[str, Any], None, None]:
        total_start = time.perf_counter()
        stage_timings: dict[str, float] = {}

        # 0a. Query Classification & Dynamic Strategy Selection
        t0 = time.perf_counter()
        classification = self.query_router.classify(user_query, history=history)
        strategy = classification.strategy
        stage_timings["query_routing"] = round((time.perf_counter() - t0) * 1000, 2)

        req_llm, selected_model = self._get_effective_llm(model)

        # Conversational / Greeting intent check: bypass vector DB and fake citations
        if classification.category == QueryCategory.CONVERSATIONAL or self.query_rewriter.is_conversational(
            user_query
        ):
            stage_timings["conversational_bypass"] = 0.5
            yield {
                "type": "retrieval_done",
                "stage_timings": stage_timings,
                "candidate_count": 0,
                "reranked_count": 0,
                "context_count": 0,
                "cache_hit": False,
            }

            greeting_text = (
                "Hello! How can I assist you today? Feel free to ask any questions regarding company policies, "
                "employee benefits, travel expenses, code of conduct, or any uploaded documentation."
            )
            words = greeting_text.split(" ")
            for i, word in enumerate(words):
                if cancel_token and cancel_token.is_set():
                    return
                chunk_text = word + (" " if i < len(words) - 1 else "")
                yield {"type": "token", "content": chunk_text}

            total_elapsed = round((time.perf_counter() - total_start) * 1000, 2)
            trace = RAGTrace(
                query=user_query,
                rewritten_query=None,
                sub_queries=[],
                query_type=classification.category.value,
                routing_confidence=classification.confidence,
                retrieval_strategy=strategy.name,
                retrieved_candidate_count=0,
                post_rerank_count=0,
                final_context_count=0,
                execution_time_ms=total_elapsed,
                stage_timings_ms=stage_timings,
                fallback_reason="conversational_greeting",
                faithfulness_checked=True,
                faithfulness_passed=True,
                verification_report=None,
                verification_score=1.0,
                retry_count=0,
                retry_reasons=[],
                cache_hit=False,
                cache_similarity=None,
            )
            yield {
                "type": "done",
                "answer": greeting_text,
                "citations": [],
                "context_chunks": [],
                "trace": trace,
                "model": selected_model,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": len(words)},
                "total_elapsed_ms": total_elapsed,
                "cache_hit": False,
            }
            return

        # 0. Pre-rewrite Cache Lookup
        cache_enabled = (
            getattr(self.semantic_cache.settings, "semantic_cache_enabled", True)
            if (self.semantic_cache and hasattr(self.semantic_cache, "settings"))
            else True
        )
        if cache_enabled and self.semantic_cache is not None:
            t0 = time.perf_counter()
            cached_res = self.semantic_cache.get(user_query, model_name=selected_model)
            stage_timings["cache_lookup"] = round((time.perf_counter() - t0) * 1000, 2)
            if cached_res is not None:
                yield {
                    "type": "retrieval_done",
                    "stage_timings": stage_timings,
                    "candidate_count": 0,
                    "reranked_count": 0,
                    "context_count": 0,
                    "cache_hit": True,
                }

                words = cached_res.answer.split(" ")
                for i, word in enumerate(words):
                    chunk_text = word + (" " if i < len(words) - 1 else "")
                    yield {"type": "token", "content": chunk_text}

                total_elapsed = round((time.perf_counter() - total_start) * 1000, 2)
                trace = RAGTrace(
                    query=user_query,
                    rewritten_query=None,
                    sub_queries=[],
                    query_type=classification.category.value,
                    routing_confidence=classification.confidence,
                    retrieval_strategy=strategy.name,
                    retrieved_candidate_count=0,
                    post_rerank_count=0,
                    final_context_count=0,
                    execution_time_ms=total_elapsed,
                    stage_timings_ms=stage_timings,
                    fallback_reason="none",
                    faithfulness_checked=True,
                    faithfulness_passed=True,
                    verification_report=None,
                    verification_score=1.0,
                    retry_count=0,
                    retry_reasons=[],
                    cache_hit=True,
                    cache_similarity=cached_res.similarity_score,
                )
                yield {
                    "type": "done",
                    "answer": cached_res.answer,
                    "citations": cached_res.citations,
                    "context_chunks": [],
                    "trace": trace,
                    "model": model or "semantic_cache",
                    "token_usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": len(cached_res.answer.split()),
                    },
                    "total_elapsed_ms": total_elapsed,
                    "cache_hit": True,
                }
                return

        # 1. Query Rewrite
        t0 = time.perf_counter()
        logger.info(f"Starting query rewrite for: {user_query}")
        rewrite_res = self.query_rewriter.rewrite(user_query, history=history, llm=req_llm)
        stage_timings["query_rewrite"] = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            f"Query rewrite complete in {stage_timings['query_rewrite']}ms. Rewritten as: {rewrite_res.rewritten_query}"
        )

        # 1b. Query-Time Metadata Filter Inference
        inferred_filters: dict[str, Any] = {}
        applied_filters: dict[str, Any] = {}
        filter_relaxed = False
        enable_filtering = getattr(settings, "enable_query_metadata_filtering", True)
        if enable_filtering and self.filter_inferer is not None:
            t0 = time.perf_counter()
            inferred_filters = self.filter_inferer.infer_filters(
                query=user_query, history=history, explicit_filters=filters,
            )
            stage_timings["filter_inference"] = round((time.perf_counter() - t0) * 1000, 2)
            if inferred_filters:
                applied_filters = {**inferred_filters}
                logger.info("Inferred metadata filters (stream): %s", applied_filters)
        elif filters:
            applied_filters = {**filters}

        # Autonomous Retry Loop Execution
        enable_verification = getattr(settings, "enable_answer_verification", True)
        max_retries = self.retry_engine.max_retries if self.retry_engine else 2
        current_strategy = strategy.model_copy(deep=True)
        attempt = 0

        best_answer = ""
        best_citations: list[Citation] = []
        best_context_chunks: list[ScoredChunk] = []
        best_candidate_chunks: list[ScoredChunk] = []
        best_reranked_chunks: list[ScoredChunk] = []
        best_report: VerificationReport | None = None
        best_score = -1.0

        retry_reasons: list[str] = []
        prompt_refinement = ""
        sub_queries: list[str] = [rewrite_res.rewritten_query]
        formatted_context = ""

        while attempt <= max_retries:
            prefix = f"_att{attempt}" if attempt > 0 else ""

            # 2. Multi-Query Generation
            t0 = time.perf_counter()
            if current_strategy.enable_multi_query or rewrite_res.is_comprehensive_list:
                sub_queries = self.multi_query_gen.generate_subqueries(rewrite_res.rewritten_query)
            else:
                sub_queries = [rewrite_res.rewritten_query]
            stage_timings[f"multi_query{prefix}"] = round((time.perf_counter() - t0) * 1000, 2)

            # 3. Hybrid Search
            t0 = time.perf_counter()
            search_filters = applied_filters if applied_filters else None
            candidate_map: dict[str, ScoredChunk] = {}
            for sq in sub_queries:
                hits = self.hybrid_retriever.retrieve(
                    sq,
                    dense_top_k=current_strategy.dense_top_k,
                    bm25_top_k=current_strategy.bm25_top_k,
                    filters=search_filters,
                    rrf_k=current_strategy.rrf_k,
                )
                for sc in hits:
                    cid = sc.chunk.id
                    if cid not in candidate_map or (sc.score or 0.0) > (
                        candidate_map[cid].score or 0.0
                    ):
                        candidate_map[cid] = sc
            candidate_chunks = list(candidate_map.values())

            # Filter relaxation fallback: retry without filters if zero results
            enable_relaxation = getattr(settings, "enable_filter_fallback_relaxation", True)
            if not candidate_chunks and search_filters and enable_relaxation:
                logger.warning(
                    "Filtered retrieval returned 0 candidates (stream). Relaxing filters and retrying."
                )
                filter_relaxed = True
                applied_filters = {}
                candidate_map = {}
                for sq in sub_queries:
                    hits = self.hybrid_retriever.retrieve(
                        sq,
                        dense_top_k=current_strategy.dense_top_k,
                        bm25_top_k=current_strategy.bm25_top_k,
                        filters=None,
                        rrf_k=current_strategy.rrf_k,
                    )
                    for sc in hits:
                        cid = sc.chunk.id
                        if cid not in candidate_map or (sc.score or 0.0) > (
                            candidate_map[cid].score or 0.0
                        ):
                            candidate_map[cid] = sc
                candidate_chunks = list(candidate_map.values())
                stage_timings[f"filter_relaxation{prefix}"] = round(
                    (time.perf_counter() - t0) * 1000, 2
                )

            candidate_chunks.sort(key=lambda x: x.score or 0.0, reverse=True)
            candidate_pool_limit = max(len(candidate_chunks), current_strategy.rerank_top_n * 3, 30)
            candidate_chunks = candidate_chunks[:candidate_pool_limit]
            stage_timings[f"hybrid_retrieval{prefix}"] = round((time.perf_counter() - t0) * 1000, 2)

            if not candidate_chunks:
                unanswerable_text = "I am unable to answer based on the provided documents."
                report = VerificationReport(
                    faithfulness=1.0,
                    completeness=1.0,
                    citation_coverage=1.0,
                    coherence=1.0,
                    composite_score=1.0,
                    passed=True,
                    retry_count=attempt,
                )
                best_answer = unanswerable_text
                best_citations = []
                best_context_chunks = []
                best_candidate_chunks = []
                best_reranked_chunks = []
                best_report = report
                break

            # 4. Cross-Encoder Reranking
            t0 = time.perf_counter()
            reranked_chunks = self.reranker.rerank(
                rewrite_res.rewritten_query,
                candidate_chunks,
                top_n=current_strategy.rerank_top_n,
                min_ratio=current_strategy.min_score_ratio,
            )
            stage_timings[f"reranking{prefix}"] = round((time.perf_counter() - t0) * 1000, 2)

            # 5. Parent Context Expansion
            t0 = time.perf_counter()
            expanded_chunks = self.compressor.expand_to_parents(
                reranked_chunks,
                self.docstore,
                enable_expansion=current_strategy.enable_parent_expansion,
            )
            formatted_context = self.compressor.format_context_for_prompt(expanded_chunks)
            stage_timings[f"context_expansion{prefix}"] = round(
                (time.perf_counter() - t0) * 1000, 2
            )

            # 6. LLM Grounded Answer Synthesis
            t0 = time.perf_counter()
            answer_text = ""

            if req_llm is not None:
                try:
                    history_text = _format_history_for_prompt(history)
                    refinement_str = (
                        f"\nRefinement Instructions:\n{prompt_refinement}\n"
                        if prompt_refinement
                        else ""
                    )
                    prompt = GROUNDED_SYSTEM_PROMPT.format(
                        refinement_directive=refinement_str,
                        context_text=formatted_context,
                        history_text=history_text,
                        query=user_query,
                    )
                    try:
                        raw_answer = str(
                            req_llm.complete(prompt, temperature=current_strategy.temperature)
                        ).strip()
                    except TypeError:
                        raw_answer = str(req_llm.complete(prompt)).strip()
                    answer_text = raw_answer
                except Exception as exc:
                    logger.warning("LLM synthesis error (%s). Using fallback synthesis.", exc)
                    answer_text = self._fallback_synthesis(user_query, expanded_chunks)
            else:
                answer_text = self._fallback_synthesis(user_query, expanded_chunks)

            stage_timings[f"llm_synthesis{prefix}"] = round((time.perf_counter() - t0) * 1000, 2)

            # 7. Verifiable Citation Extraction
            t0 = time.perf_counter()
            citations = self.citation_engine.select_citations(
                answer_text=answer_text,
                generation_chunks=expanded_chunks,
                user_query=user_query,
            )
            stage_timings[f"citation_extraction{prefix}"] = round(
                (time.perf_counter() - t0) * 1000, 2
            )

            # 8. Post-Generation Verification
            t0 = time.perf_counter()
            if enable_verification and self.verifier is not None:
                report = self.verifier.verify(
                    query=user_query,
                    answer=answer_text,
                    context_chunks=expanded_chunks,
                    citations=citations,
                    llm=req_llm,
                )
                report.retry_count = attempt
            else:
                report = VerificationReport(
                    faithfulness=1.0,
                    completeness=1.0,
                    citation_coverage=1.0,
                    coherence=1.0,
                    composite_score=1.0,
                    passed=True,
                    retry_count=attempt,
                )
            stage_timings[f"verification{prefix}"] = round((time.perf_counter() - t0) * 1000, 2)

            # Update best candidate
            if report.composite_score > best_score or best_report is None:
                best_score = report.composite_score
                best_answer = answer_text
                best_citations = citations
                best_context_chunks = expanded_chunks
                best_candidate_chunks = candidate_chunks
                best_reranked_chunks = reranked_chunks
                best_report = report

            if report.passed:
                best_answer = answer_text
                best_citations = citations
                best_context_chunks = expanded_chunks
                best_candidate_chunks = candidate_chunks
                best_reranked_chunks = reranked_chunks
                best_report = report
                break

            if attempt >= max_retries or not self.retry_engine.should_retry(attempt, report):
                logger.warning(
                    "Retry hard cap reached in stream (%d retries). Gracefully falling back.",
                    attempt,
                )
                break

            # Prepare retry parameters for next iteration
            if report.critique:
                retry_reasons.append(report.critique)
            elif report.missing_aspects or report.unsupported_claims:
                parts = []
                if report.missing_aspects:
                    parts.append(f"Missing: {', '.join(report.missing_aspects)}")
                if report.unsupported_claims:
                    parts.append(f"Unsupported: {', '.join(report.unsupported_claims)}")
                retry_reasons.append("; ".join(parts))
            else:
                retry_reasons.append(
                    f"Verification score {report.composite_score:.2f} below threshold."
                )

            current_strategy, prompt_refinement = self.retry_engine.prepare_retry(
                attempt=attempt,
                report=report,
                strategy=current_strategy,
                query=user_query,
            )
            attempt += 1

        # Signal retrieval_done
        yield {
            "type": "retrieval_done",
            "stage_timings": stage_timings,
            "candidate_count": len(best_candidate_chunks),
            "reranked_count": len(best_reranked_chunks),
            "context_count": len(best_context_chunks),
            "cache_hit": False,
        }

        # Stream tokens of the verified final response
        words = best_answer.split(" ")
        for i, word in enumerate(words):
            if cancel_token and cancel_token.is_set():
                logger.info("Stream cancelled by client during token stream.")
                return
            chunk_text = word + (" " if i < len(words) - 1 else "")
            yield {"type": "token", "content": chunk_text}

        if cancel_token and cancel_token.is_set():
            logger.info("Stream aborted before completion, skipping done payload.")
            return

        fallback_reason = "none"
        if req_llm is None:
            fallback_reason = "llm_offline_fallback"
        elif best_report is not None and not best_report.passed:
            fallback_reason = "retry_exhausted_fallback"

        total_elapsed = round((time.perf_counter() - total_start) * 1000, 2)

        trace = RAGTrace(
            query=user_query,
            rewritten_query=rewrite_res.rewritten_query,
            sub_queries=sub_queries,
            query_type=classification.category.value,
            routing_confidence=classification.confidence,
            retrieval_strategy=current_strategy.name if current_strategy else strategy.name,
            inferred_filters=inferred_filters,
            applied_filters=applied_filters,
            filter_relaxed=filter_relaxed,
            retrieved_candidate_count=len(best_candidate_chunks),
            post_rerank_count=len(best_reranked_chunks),
            final_context_count=len(best_context_chunks),
            execution_time_ms=total_elapsed,
            stage_timings_ms=stage_timings,
            fallback_reason=fallback_reason,
            faithfulness_checked=enable_verification,
            faithfulness_passed=best_report.passed if best_report else True,
            verification_report=best_report.model_dump() if best_report else None,
            verification_score=best_report.composite_score if best_report else None,
            retry_count=attempt,
            retry_reasons=retry_reasons,
            cache_hit=False,
            cache_similarity=None,
        )

        if (
            best_citations
            and len(best_citations) > 0
            and best_answer
            and (best_report is None or best_report.passed)
        ):
            self._queue_cache_write(
                user_query, best_answer, best_citations, model_name=selected_model
            )

        yield {
            "type": "done",
            "answer": best_answer,
            "citations": best_citations,
            "context_chunks": best_context_chunks,
            "trace": trace,
            "model": selected_model,
            "token_usage": {
                "prompt_tokens": len(formatted_context.split()) if best_context_chunks else 0,
                "completion_tokens": len(best_answer.split()),
            },
            "total_elapsed_ms": total_elapsed,
            "cache_hit": False,
        }
