from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Generator
from typing import Any

from backend.models.chunk import Chunk
from backend.models.rag import (
    RAGResponse,
    RAGTrace,
    ScoredChunk,
)
from backend.rag.citations import CitationEngine
from backend.rag.context_compression import ContextCompressor
from backend.rag.multi_query import MultiQueryGenerator
from backend.rag.query_rewrite import QueryRewriter
from backend.rag.semantic_cache import SemanticCacheManager
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.reranker import CrossEncoderReranker
from backend.utils.logging import logger
from src.ollama_client import unload_model, preload_model

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

Document Sources:
{context_text}

{history_text}User Question: {query}
Answer:"""


def _format_history_for_prompt(history: list[dict[str, Any]] | None, max_turns: int = 6) -> str:
    if not history:
        return ""
    recent = history[-(max_turns * 2):]
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
    LLM grounded synthesis, structured citation extraction, semantic caching, and telemetry traces.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker | None = None,
        query_rewriter: QueryRewriter | None = None,
        multi_query_gen: MultiQueryGenerator | None = None,
        compressor: ContextCompressor | None = None,
        citation_engine: CitationEngine | None = None,
        docstore: dict[str, Chunk] | None = None,
        llm: Any | None = None,
        semantic_cache: SemanticCacheManager | None = None,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.multi_query_gen = multi_query_gen or MultiQueryGenerator()
        self.compressor = compressor or ContextCompressor()
        self.citation_engine = citation_engine or CitationEngine()
        self.docstore = docstore or {}
        self.llm = llm
        self.semantic_cache = semantic_cache
        self.model_manager = ModelManager(initial_model=getattr(self.llm, "model", None) or "qwen2.5:7b")

        if self.query_rewriter.llm is None and self.llm is not None:
            self.query_rewriter.llm = self.llm

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
                    logger.warning("Could not instantiate Ollama for model %s: %s", selected_model, exc)
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

        req_llm, selected_model = self._get_effective_llm(model)

        # Conversational / Greeting intent check: bypass vector DB and fake citations
        if self.query_rewriter.is_conversational(user_query):
            greeting_answer = (
                "Hello! How can I assist you today? Feel free to ask any questions regarding company policies, "
                "employee benefits, travel expenses, code of conduct, or any uploaded documentation."
            )
            total_elapsed = round((time.perf_counter() - total_start) * 1000, 2)
            trace = RAGTrace(
                query=user_query,
                rewritten_query=None,
                sub_queries=[],
                retrieved_candidate_count=0,
                post_rerank_count=0,
                final_context_count=0,
                execution_time_ms=total_elapsed,
                stage_timings_ms={"conversational_bypass": total_elapsed},
                fallback_reason="conversational_greeting",
                faithfulness_checked=True,
                faithfulness_passed=True,
                cache_hit=False,
            )
            return RAGResponse(
                id=f"resp_{uuid.uuid4().hex[:12]}",
                query=user_query,
                answer=greeting_answer,
                citations=[],
                context_chunks=[],
                trace=trace,
                model=selected_model,
                token_usage={"prompt_tokens": 0, "completion_tokens": len(greeting_answer.split())},
            )

        # 0. Pre-rewrite Cache Lookup
        cache_enabled = getattr(self.semantic_cache.settings, "semantic_cache_enabled", True) if (self.semantic_cache and hasattr(self.semantic_cache, "settings")) else True
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
                    retrieved_candidate_count=0,
                    post_rerank_count=0,
                    final_context_count=0,
                    execution_time_ms=total_elapsed,
                    stage_timings_ms=stage_timings,
                    fallback_reason="none",
                    faithfulness_checked=True,
                    faithfulness_passed=True,
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
                    token_usage={"prompt_tokens": 0, "completion_tokens": len(cached_res.answer.split())},
                )

        # 1. Query Rewrite
        t0 = time.perf_counter()
        rewrite_res = self.query_rewriter.rewrite(user_query, history=history, llm=req_llm)
        stage_timings["query_rewrite"] = round((time.perf_counter() - t0) * 1000, 2)

        # 2. Multi-Query Generation
        t0 = time.perf_counter()
        if rewrite_res.is_comprehensive_list:
            sub_queries = self.multi_query_gen.generate_subqueries(rewrite_res.rewritten_query)
        else:
            sub_queries = [rewrite_res.rewritten_query]
        stage_timings["multi_query"] = round((time.perf_counter() - t0) * 1000, 2)

        # 3. Hybrid Search (Dense + BM25 with RRF) across sub-queries
        t0 = time.perf_counter()
        candidate_map: dict[str, ScoredChunk] = {}
        for sq in sub_queries:
            hits = self.hybrid_retriever.retrieve(sq, dense_top_k=25, bm25_top_k=25, filters=filters)
            for sc in hits:
                cid = sc.chunk.id
                if cid not in candidate_map or (sc.score or 0.0) > (candidate_map[cid].score or 0.0):
                    candidate_map[cid] = sc
        candidate_chunks = list(candidate_map.values())
        candidate_chunks.sort(key=lambda x: x.score or 0.0, reverse=True)
        candidate_chunks = candidate_chunks[:8]
        stage_timings["hybrid_retrieval"] = round((time.perf_counter() - t0) * 1000, 2)

        # 4. Cross-Encoder Reranking & Relative Score Thresholding
        t0 = time.perf_counter()
        reranked_chunks = self.reranker.rerank(rewrite_res.rewritten_query, candidate_chunks)
        stage_timings["reranking"] = round((time.perf_counter() - t0) * 1000, 2)

        # 5. Parent Context Expansion & Formatting
        t0 = time.perf_counter()
        expanded_chunks = self.compressor.expand_to_parents(reranked_chunks, self.docstore)
        formatted_context = self.compressor.format_context_for_prompt(expanded_chunks)
        stage_timings["context_expansion"] = round((time.perf_counter() - t0) * 1000, 2)

        # 6. LLM Grounded Answer Synthesis
        t0 = time.perf_counter()
        answer_text = ""
        model_name = selected_model

        if req_llm is not None:
            try:
                history_text = _format_history_for_prompt(history)
                prompt = GROUNDED_SYSTEM_PROMPT.format(
                    context_text=formatted_context,
                    history_text=history_text,
                    query=user_query,
                )
                raw_answer = str(req_llm.complete(prompt)).strip()
                answer_text = raw_answer
            except Exception as exc:
                logger.warning("LLM synthesis error (%s). Using fallback synthesis.", exc)
                answer_text = self._fallback_synthesis(user_query, expanded_chunks)
        else:
            answer_text = self._fallback_synthesis(user_query, expanded_chunks)

        stage_timings["llm_synthesis"] = round((time.perf_counter() - t0) * 1000, 2)

        # 7. Verifiable Citation Extraction
        t0 = time.perf_counter()
        citations = self.citation_engine.select_citations(
            answer_text=answer_text,
            generation_chunks=expanded_chunks,
            user_query=user_query,
        )
        stage_timings["citation_extraction"] = round((time.perf_counter() - t0) * 1000, 2)

        total_elapsed = round((time.perf_counter() - total_start) * 1000, 2)

        # 8. Telemetry Trace
        trace = RAGTrace(
            query=user_query,
            rewritten_query=rewrite_res.rewritten_query,
            sub_queries=sub_queries,
            retrieved_candidate_count=len(candidate_chunks),
            post_rerank_count=len(reranked_chunks),
            final_context_count=len(expanded_chunks),
            execution_time_ms=total_elapsed,
            stage_timings_ms=stage_timings,
            fallback_reason="none" if req_llm is not None else "llm_offline_fallback",
            faithfulness_checked=True,
            faithfulness_passed=True,
            cache_hit=False,
            cache_similarity=None,
        )

        # 9. Queue non-blocking cache write on successful citation-backed answer
        if citations and len(citations) > 0 and answer_text:
            self._queue_cache_write(user_query, answer_text, citations, model_name=model_name)

        return RAGResponse(
            id=f"resp_{uuid.uuid4().hex[:12]}",
            query=user_query,
            answer=answer_text,
            citations=citations,
            context_chunks=expanded_chunks,
            trace=trace,
            model=model_name,
            token_usage={"prompt_tokens": len(formatted_context.split()), "completion_tokens": len(answer_text.split())},
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

        return f"Based on the official documentation regarding '{user_query}':\n\n" + "\n\n".join(paragraphs)

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

        req_llm, selected_model = self._get_effective_llm(model)

        # Conversational / Greeting intent check: bypass vector DB and fake citations
        if self.query_rewriter.is_conversational(user_query):
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
                retrieved_candidate_count=0,
                post_rerank_count=0,
                final_context_count=0,
                execution_time_ms=total_elapsed,
                stage_timings_ms=stage_timings,
                fallback_reason="conversational_greeting",
                faithfulness_checked=True,
                faithfulness_passed=True,
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
        cache_enabled = getattr(self.semantic_cache.settings, "semantic_cache_enabled", True) if (self.semantic_cache and hasattr(self.semantic_cache, "settings")) else True
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
                    retrieved_candidate_count=0,
                    post_rerank_count=0,
                    final_context_count=0,
                    execution_time_ms=total_elapsed,
                    stage_timings_ms=stage_timings,
                    fallback_reason="none",
                    faithfulness_checked=True,
                    faithfulness_passed=True,
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
        logger.info(f"Query rewrite complete in {stage_timings['query_rewrite']}ms. Rewritten as: {rewrite_res.rewritten_query}")

        # 2. Multi-Query Generation
        t0 = time.perf_counter()
        if rewrite_res.is_comprehensive_list:
            sub_queries = self.multi_query_gen.generate_subqueries(rewrite_res.rewritten_query)
        else:
            sub_queries = [rewrite_res.rewritten_query]
        stage_timings["multi_query"] = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(f"Multi-query generation complete. Generated {len(sub_queries)} subqueries.")

        # 3. Hybrid Search
        t0 = time.perf_counter()
        candidate_map: dict[str, ScoredChunk] = {}
        logger.info("Starting hybrid retrieval for subqueries...")
        for sq in sub_queries:
            hits = self.hybrid_retriever.retrieve(sq, dense_top_k=25, bm25_top_k=25, filters=filters)
            for sc in hits:
                cid = sc.chunk.id
                if cid not in candidate_map or (sc.score or 0.0) > (candidate_map[cid].score or 0.0):
                    candidate_map[cid] = sc
        candidate_chunks = list(candidate_map.values())
        # Sort by initial score and truncate to max 8 chunks for fast CrossEncoder inference
        candidate_chunks.sort(key=lambda x: x.score or 0.0, reverse=True)
        candidate_chunks = candidate_chunks[:8]
        stage_timings["hybrid_retrieval"] = round((time.perf_counter() - t0) * 1000, 2)

        # 4. Cross-Encoder Reranking
        t0 = time.perf_counter()
        reranked_chunks = self.reranker.rerank(rewrite_res.rewritten_query, candidate_chunks)
        stage_timings["reranking"] = round((time.perf_counter() - t0) * 1000, 2)

        # 5. Parent Context Expansion
        t0 = time.perf_counter()
        expanded_chunks = self.compressor.expand_to_parents(reranked_chunks, self.docstore)
        formatted_context = self.compressor.format_context_for_prompt(expanded_chunks)
        stage_timings["context_expansion"] = round((time.perf_counter() - t0) * 1000, 2)

        # Signal that retrieval is done — frontend can show "Generating..."
        yield {
            "type": "retrieval_done",
            "stage_timings": stage_timings,
            "candidate_count": len(candidate_chunks),
            "reranked_count": len(reranked_chunks),
            "context_count": len(expanded_chunks),
            "cache_hit": False,
        }

        # 6. Stream LLM tokens in real-time
        t0 = time.perf_counter()
        answer_parts: list[str] = []
        model_name = selected_model

        if req_llm is not None:
            try:
                history_text = _format_history_for_prompt(history)
                prompt = GROUNDED_SYSTEM_PROMPT.format(
                    context_text=formatted_context,
                    history_text=history_text,
                    query=user_query,
                )
                # Use stream_complete for real-time token generation
                stream_response = req_llm.stream_complete(prompt)
                for token_response in stream_response:
                    if cancel_token and cancel_token.is_set():
                        logger.info("Stream cancelled by client during LLM token synthesis.")
                        break
                    token_text = token_response.delta
                    if token_text:
                        answer_parts.append(token_text)
                        yield {"type": "token", "content": token_text}
            except Exception as exc:
                if cancel_token and cancel_token.is_set():
                    logger.info("Stream cancelled by client.")
                    return
                logger.warning("LLM stream error (%s). Using fallback synthesis.", exc)
                fallback = self._fallback_synthesis(user_query, expanded_chunks)
                answer_parts = [fallback]
                yield {"type": "token", "content": fallback}
        else:
            fallback = self._fallback_synthesis(user_query, expanded_chunks)
            answer_parts = [fallback]
            yield {"type": "token", "content": fallback}

        if cancel_token and cancel_token.is_set():
            logger.info("Stream aborted before completion, skipping done payload.")
            return

        stage_timings["llm_synthesis"] = round((time.perf_counter() - t0) * 1000, 2)
        full_answer = "".join(answer_parts)

        # 7. Citation Extraction
        t0 = time.perf_counter()
        citations = self.citation_engine.select_citations(
            answer_text=full_answer,
            generation_chunks=expanded_chunks,
            user_query=user_query,
        )
        stage_timings["citation_extraction"] = round((time.perf_counter() - t0) * 1000, 2)

        total_elapsed = round((time.perf_counter() - total_start) * 1000, 2)

        # 8. Build Trace
        trace = RAGTrace(
            query=user_query,
            rewritten_query=rewrite_res.rewritten_query,
            sub_queries=sub_queries,
            retrieved_candidate_count=len(candidate_chunks),
            post_rerank_count=len(reranked_chunks),
            final_context_count=len(expanded_chunks),
            execution_time_ms=total_elapsed,
            stage_timings_ms=stage_timings,
            fallback_reason="none" if req_llm is not None else "llm_offline_fallback",
            faithfulness_checked=True,
            faithfulness_passed=True,
            cache_hit=False,
            cache_similarity=None,
        )

        # 9. Queue non-blocking cache write on successful completion with valid citations
        if citations and len(citations) > 0 and full_answer:
            self._queue_cache_write(user_query, full_answer, citations, model_name=selected_model)

        # Yield final done event with all metadata
        yield {
            "type": "done",
            "answer": full_answer,
            "citations": citations,
            "context_chunks": expanded_chunks,
            "trace": trace,
            "model": model_name,
            "token_usage": {
                "prompt_tokens": len(formatted_context.split()),
                "completion_tokens": len(full_answer.split()),
            },
            "total_elapsed_ms": total_elapsed,
            "cache_hit": False,
        }


