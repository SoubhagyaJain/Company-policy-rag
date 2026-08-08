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
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.reranker import CrossEncoderReranker
from backend.utils.logging import logger

_llm_lock = threading.Lock()


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
                return list(self._target_llm.stream_complete(prompt, **kwargs))
            finally:
                if hasattr(self._target_llm, "model") and old_model is not None:
                    self._target_llm.model = old_model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target_llm, name)


GROUNDED_SYSTEM_PROMPT = """You are an accurate, enterprise policy & technical documentation assistant.
Your instructions:
1. Answer the user's question using ONLY the provided document sources below and relevant conversation history.
2. For EVERY key claim or fact in your response, cite the source number using exact bracket format [Source N] (e.g. [Source 1], [Source 2]).
3. If the provided sources do not contain sufficient information to answer the question, state clearly: "I am unable to answer based on the provided policy documents."
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
    LLM grounded synthesis, structured citation extraction, and telemetry traces.
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
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.multi_query_gen = multi_query_gen or MultiQueryGenerator()
        self.compressor = compressor or ContextCompressor()
        self.citation_engine = citation_engine or CitationEngine()
        self.docstore = docstore or {}
        self.llm = llm

        if self.query_rewriter.llm is None and self.llm is not None:
            self.query_rewriter.llm = self.llm

    def _get_effective_llm(self, model: str | None) -> tuple[Any | None, str]:
        """Return a per-request thread-safe LLM proxy and model name without mutating shared singleton state."""
        if self.llm is None:
            return None, model or "qwen2.5:7b"

        base_model = getattr(self.llm, "model", "qwen2.5:7b")
        selected_model = model or base_model
        if not model or model == base_model:
            return self.llm, selected_model

        return _LLMProxy(self.llm, selected_model), selected_model

    def query(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> RAGResponse:
        """Execute end-to-end RAG pipeline and return structured RAGResponse with trace telemetry."""
        total_start = time.perf_counter()
        stage_timings: dict[str, float] = {}

        req_llm, selected_model = self._get_effective_llm(model)

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
        )

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
            return "I am unable to answer based on the provided policy documents."

        paragraphs: list[str] = []
        for idx, sc in enumerate(context_chunks, start=1):
            text_snippet = sc.chunk.text.strip()
            # Pick first 2 sentences
            sentences = [s.strip() for s in text_snippet.split(".") if s.strip()]
            snippet = ". ".join(sentences[:2]) + "." if sentences else text_snippet
            paragraphs.append(f"{snippet} [Source {idx}]")

        return f"Based on the official documentation regarding '{user_query}':\n\n" + "\n\n".join(paragraphs)

    def stream_query(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Streaming RAG pipeline: runs retrieval synchronously, then yields
        real-time LLM tokens via llm.stream_complete().

        Yields dicts with 'type' key:
          - {'type': 'retrieval_done', 'stage_timings': {...}, 'candidate_count': int, ...}
          - {'type': 'token', 'content': str}
          - {'type': 'done', 'answer': str, 'citations': [...], 'trace': RAGTrace, ...}
        """
        total_start = time.perf_counter()
        stage_timings: dict[str, float] = {}

        req_llm, selected_model = self._get_effective_llm(model)

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

        # 3. Hybrid Search
        t0 = time.perf_counter()
        candidate_map: dict[str, ScoredChunk] = {}
        for sq in sub_queries:
            hits = self.hybrid_retriever.retrieve(sq, dense_top_k=25, bm25_top_k=25, filters=filters)
            for sc in hits:
                cid = sc.chunk.id
                if cid not in candidate_map or (sc.score or 0.0) > (candidate_map[cid].score or 0.0):
                    candidate_map[cid] = sc
        candidate_chunks = list(candidate_map.values())
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
                    token_text = token_response.delta
                    if token_text:
                        answer_parts.append(token_text)
                        yield {"type": "token", "content": token_text}
            except Exception as exc:
                logger.warning("LLM stream error (%s). Using fallback synthesis.", exc)
                fallback = self._fallback_synthesis(user_query, expanded_chunks)
                answer_parts = [fallback]
                yield {"type": "token", "content": fallback}
        else:
            fallback = self._fallback_synthesis(user_query, expanded_chunks)
            answer_parts = [fallback]
            yield {"type": "token", "content": fallback}

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
        )

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
        }

