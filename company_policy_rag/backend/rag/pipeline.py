from __future__ import annotations

import re
import threading
import time
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

from backend.models.chunk import Chunk, ChunkMetadata, ContentType
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
from backend.rag.evidence_gate import EvidenceSufficiencyGate, EvidenceSufficiencyResult
from backend.rag.filter_extractor import QueryMetadataInferer
from backend.rag.multi_query import MultiQueryGenerator
from backend.rag.query_rewrite import QueryRewriter
from backend.rag.query_router import QueryRouter
from backend.rag.retry_engine import RetryEngine
from backend.rag.scope_resolver import (
    DocumentRetrievalScope,
    DocumentScopeDecision,
    DocumentScopeResolver,
)
from backend.rag.semantic_cache import SemanticCacheManager
from backend.rag.verifier import SelfReflectionVerifier
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.reranker import CrossEncoderReranker
from backend.retrieval.retrieval_cache import get_retrieval_cache
from backend.utils.logging import logger
from backend.vision.vision_service import VisionService
from src.config import settings
from src.ollama_client import preload_model, unload_model


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
        old_model = getattr(self._target_llm, "model", None)
        try:
            if hasattr(self._target_llm, "model") and self.model:
                self._target_llm.model = self.model
            return self._target_llm.complete(prompt, **kwargs)
        finally:
            if hasattr(self._target_llm, "model") and old_model is not None:
                self._target_llm.model = old_model

    def stream_complete(self, prompt: str, **kwargs: Any) -> Any:
        old_model = getattr(self._target_llm, "model", None)
        try:
            if hasattr(self._target_llm, "model") and self.model:
                self._target_llm.model = self.model
            return self._target_llm.stream_complete(prompt, **kwargs)
        finally:
            if hasattr(self._target_llm, "model") and old_model is not None:
                self._target_llm.model = old_model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target_llm, name)


GROUNDED_SYSTEM_PROMPT = """You are a document-faithful AI assistant.
Your absolute source of truth is the RETRIEVED CONTEXT below.

Core Non-Negotiable Rules:
1. Grounding Fidelity: Answer using ONLY the retrieved context. Do not invent facts, code, or assumptions.
2. Code Fidelity:
   - If the implementation code is present in the retrieved context, provide it exactly as written.
   - If the implementation code is NOT present in the retrieved context, DO NOT fabricate placeholder code (such as 'def func(): pass'). Instead, explicitly state: "The retrieved document content does not contain the implementation code."
3. Absence Handling: If the specific requested information is not in the context, explicitly say:
   "I could not find this information in the provided document."
4. Citations: Cite sources using [Source N] tags for every substantive claim and code block.
{mode_instructions}
{refinement_directive}
RETRIEVED CONTEXT:
{context_text}

{history_text}USER QUESTION: {query}
ANSWER:"""

EXACT_MODE_INSTRUCTIONS = """Mode: EXACT EXTRACTION
- Extract and present the exact text, tables, headings, and code from the document with maximum source fidelity.
- Do not paraphrase or add external commentary unless requested.
- Preserve original variable names, function names, and code syntax exactly."""

EXPLAIN_MODE_INSTRUCTIONS = """Mode: EXPLAIN
- First present the relevant document excerpts and code faithfully.
- Then provide a structured, grounded explanation of how it works."""

IMPLEMENT_MODE_INSTRUCTIONS = """Mode: IMPLEMENTATION
- Present the step-by-step implementation strictly based on the retrieved code, agent/task configurations, and parameters from the document.
- Include the exact code provided in the document. Do not substitute generic or fabricated code."""


def _detect_fidelity_mode(query: str) -> str:
    q_lower = query.lower()
    if any(k in q_lower for k in ("show exactly", "what is written", "give me the exact", "copy from document", "exact code", "show me the code", "give me the code")):
        return "exact"
    elif any(k in q_lower for k in ("explain", "how does this work", "teach me", "walk me through", "why")):
        return "explain"
    elif any(k in q_lower for k in ("how can i make", "how do i build", "how to make", "how to implement", "how is", "implementation", "create", "build")):
        return "implement"
    return "grounded"


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


def _log_rag_trace(trace: RAGTrace) -> None:
    """Output comprehensive Phase 14 structured observability telemetry trace."""
    lines = [
        "\n══════════════════════════════════════════════════════════════════════",
        "[RAG TRACE]",
        f"QUERY:                    {trace.query}",
        f"INTENT:                   {trace.query_type or 'factual'}",
        f"DOCUMENT SCOPE:           {trace.query_scope or 'global'}",
        f"ANCHOR SECTION:           {trace.anchor_section or 'General'}",
        f"CANDIDATES RETRIEVED:     {trace.retrieved_candidate_count}",
        f"POST-RERANK COUNT:        {trace.post_rerank_count}",
        f"FINAL RETRIEVED EVIDENCE: TEXT={trace.evidence_text_count} CODE={trace.evidence_code_count} DIAGRAM={trace.evidence_diagram_count} TABLE={trace.evidence_table_count}",
        f"SECTION EXPANSION:        {'YES' if trace.section_expansion else 'NO'}",
        f"ADJACENT PAGE CHECK:      {'YES' if trace.adjacent_page_check else 'NO'}",
        f"VISION FALLBACK:          {'YES' if trace.vision_fallback else 'NO'}",
        f"VISION MODEL:             {trace.vision_model or 'qwen2.5vl:7b'}",
        f"VISION CACHE:             {trace.vision_cache_status or 'N/A'}",
        f"EVIDENCE SUFFICIENCY:     {'PASS' if trace.evidence_sufficiency_passed else 'FAIL'}",
        f"GENERATION MODEL:         {trace.generation_model or 'qwen2.5:7b'}",
        f"GROUNDING VALIDATION:     {'PASS' if trace.grounding_validation_passed else 'FAIL'}",
        f"TOTAL LATENCY:            {trace.execution_time_ms:.2f} ms",
        "══════════════════════════════════════════════════════════════════════\n",
    ]
    logger.info("\n".join(lines))


class RAGPipeline:
    """
    Master end-to-end RAG Pipeline orchestrating query routing, multi-query decomposition,
    hybrid dense+sparse search, cross-encoder reranking, logical section expansion,
    cross-page continuation, high-fidelity multimodal vision extraction, pre-generation
    evidence sufficiency gating, and grounded synthesis with Qwen 2.5 7B.
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
        scope_resolver: DocumentScopeResolver | None = None,
        vision_service: VisionService | None = None,
        evidence_gate: EvidenceSufficiencyGate | None = None,
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
        self.scope_resolver = scope_resolver or DocumentScopeResolver()
        self.vision_service = vision_service or VisionService()
        self.evidence_gate = evidence_gate or EvidenceSufficiencyGate()

        raw_llm_name = getattr(self.llm, "model", None)
        if isinstance(raw_llm_name, str) and raw_llm_name.strip():
            default_llm_name = raw_llm_name.strip()
        else:
            default_llm_name = getattr(settings, "llm_model", "qwen2.5:7b")
        self.model_manager = ModelManager(initial_model=str(default_llm_name))

        if self.query_rewriter.llm is None and self.llm is not None:
            self.query_rewriter.llm = self.llm
        if self.query_router.llm is None and self.llm is not None:
            self.query_router.llm = self.llm
        if self.verifier.llm is None and self.llm is not None:
            self.verifier.llm = self.llm

    def _resolve_document_file_path(self, meta: ChunkMetadata) -> Path | None:
        """Find the real document file path on disk across known storage and data folders."""
        candidates = []
        if getattr(meta, "file_path", None):
            candidates.append(Path(meta.file_path))
        if getattr(meta, "source_file", None):
            s_name = Path(meta.source_file).name
            candidates.extend([
                Path("storage/documents") / s_name,
                Path("storage/uploads") / s_name,
                Path("app/storage/uploads") / s_name,
                Path("company_policy_rag/app/storage/uploads") / s_name,
                Path("data/legal") / s_name,
                Path("data/policies") / s_name,
                Path("company_policy_rag/data/legal") / s_name,
                Path("company_policy_rag/data/policies") / s_name,
            ])
            # Also search with doc_id prefix
            if getattr(meta, "document_id", None):
                candidates.extend([
                    Path("storage/documents") / f"{meta.document_id}_{s_name}",
                    Path("storage/uploads") / f"{meta.document_id}_{s_name}",
                    Path("app/storage/uploads") / f"{meta.document_id}_{s_name}",
                    Path("company_policy_rag/app/storage/uploads") / f"{meta.document_id}_{s_name}",
                ])

        for p in candidates:
            if p.is_file():
                return p
        return None

    def _apply_cross_page_vision_fallback_if_needed(
        self,
        chunks: list[ScoredChunk],
        user_query: str,
        intent: QueryCategory | str,
    ) -> tuple[list[ScoredChunk], dict[str, Any]]:
        """
        Evaluate evidence sufficiency before LLM generation.
        If code or visual information is missing for an implementation/code query,
        inspect the anchor section and adjacent continuation pages (Page N, N+1, N+2)
        using qwen2.5vl:7b, cache results, and pack the extracted code into context.
        """
        raw_vm = getattr(self.vision_service, "vision_model", "qwen2.5vl:7b")
        telemetry: dict[str, Any] = {
            "section_expansion": False,
            "adjacent_page_check": False,
            "vision_fallback": False,
            "vision_model": str(raw_vm) if (isinstance(raw_vm, str) and raw_vm.strip()) else "qwen2.5vl:7b",
            "vision_cache_status": "N/A",
            "evidence_sufficiency_passed": True,
            "anchor_section": None,
        }

        if not getattr(settings, "enable_lazy_vision_fallback", True) or not getattr(settings, "vision_enabled", True):
            return chunks, telemetry

        if not chunks:
            return chunks, telemetry

        # 1. Evaluate Evidence Sufficiency
        gate_res = self.evidence_gate.evaluate(query=user_query, intent=intent, candidate_chunks=chunks)
        telemetry["evidence_sufficiency_passed"] = gate_res.is_sufficient
        if gate_res.anchor_chunk:
            telemetry["anchor_section"] = gate_res.anchor_chunk.chunk.metadata.section_title or gate_res.anchor_chunk.chunk.metadata.section_path

        if gate_res.is_sufficient:
            return chunks, telemetry

        # 2. Evidence is insufficient -> Trigger Cross-Page Adjacent Inspection
        logger.info(
            "Evidence sufficiency check failed for query '%s' (intent=%s). Inspecting pages %s",
            user_query,
            intent,
            gate_res.pages_to_inspect,
        )
        telemetry["section_expansion"] = True
        telemetry["adjacent_page_check"] = True
        telemetry["vision_fallback"] = True

        anchor = gate_res.anchor_chunk or chunks[0]
        meta = anchor.chunk.metadata
        resolved_path = self._resolve_document_file_path(meta)
        if not resolved_path:
            logger.warning("Could not find physical PDF file on disk for chunk %s (%s)", anchor.chunk.id, meta.source_file)
            return chunks, telemetry

        cue = gate_res.detected_continuation_cues[0] if gate_res.detected_continuation_cues else None
        anchor_title = meta.section_title or meta.section_path or "Section Implementation"

        # 3. Extract visuals across target page range (prioritize anchor page, stop early on first success)
        visual_chunks = []
        cache_hits = 0
        cache_misses = 0

        target_pages = list(gate_res.pages_to_inspect)
        if meta.page_number and meta.page_number in target_pages:
            target_pages.remove(meta.page_number)
            target_pages.insert(0, meta.page_number)

        # At query-time, inspect at most 2 pages and break on first successful extraction
        for p_num in target_pages[:2]:
            extracted = self.vision_service.process_pdf_page_visuals(
                pdf_path=resolved_path,
                page_number=p_num,
                page_text=anchor.chunk.text if p_num == meta.page_number else "",
                document_id=meta.document_id,
                section_title=anchor_title,
                continuation_cue=cue,
                is_query_time=True,
            )
            for vc in extracted:
                visual_chunks.append((p_num, vc))
            if visual_chunks:
                break

        if not visual_chunks:
            return chunks, telemetry

        telemetry["vision_cache_status"] = "HIT" if cache_hits > 0 and cache_misses == 0 else "MISS"

        # 4. Wrap extracted visual chunks into ScoredChunk objects
        new_scored: list[ScoredChunk] = []
        for p_num, vc in visual_chunks:
            c_id = f"chunk_lazy_vis_{meta.document_id}_p{p_num}_{vc.image_hash[:8]}"
            ct_enum = (
                ContentType.CODE
                if vc.content_type == "code"
                else ContentType.TABLE
                if vc.content_type == "table"
                else ContentType.PROSE
            )
            img_url = f"/api/documents/{meta.document_id}/images/{vc.image_hash}" if meta.document_id and vc.image_hash else None
            meta_dict = meta.model_dump()
            meta_dict["content_type"] = ct_enum
            meta_dict["page_number"] = p_num
            meta_dict["page_label"] = vc.page_label or str(p_num)
            meta_dict["internal_page_index"] = vc.internal_page_index
            meta_dict["section_title"] = anchor_title
            meta_dict["has_code"] = vc.content_type == "code"
            meta_dict["has_tables"] = vc.content_type == "table"
            meta_dict["image_assets"] = [
                {
                    "asset_url": img_url,
                    "image_hash": vc.image_hash,
                    "page_number": p_num,
                    "page_label": vc.page_label,
                }
            ] if img_url else []
            meta_dict["extra"] = {
                **(meta.extra or {}),
                "is_visual_extraction": True,
                "visual_type": vc.visual_type,
                "image_hash": vc.image_hash,
                "image_url": img_url,
                "content_type": vc.content_type,
                "raw_code": vc.raw_code,
                "continuation_from_page": meta.page_number if p_num != meta.page_number else None,
            }
            lazy_chunk = Chunk(
                id=c_id,
                text=vc.text,
                metadata=ChunkMetadata(**meta_dict),
                token_count=len(vc.text.split()),
            )
            if self.docstore is not None:
                self.docstore[c_id] = lazy_chunk

            new_scored.append(
                ScoredChunk(
                    chunk=lazy_chunk,
                    score=0.96,
                    rerank_score=0.96,
                    dense_score=0.96,
                    sparse_score=0.96,
                )
            )

        # Merge newly extracted visual code chunks ahead of prose
        combined = new_scored + chunks
        telemetry["evidence_sufficiency_passed"] = True
        return combined, telemetry

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
        fallback_model = str(getattr(settings, "llm_model", "qwen2.5:7b"))
        curr = self.model_manager.current_model
        base_model = curr if isinstance(curr, str) and curr else fallback_model
        raw_model = (model or "").strip() if isinstance(model, str) else ""
        if not raw_model or raw_model.lower() in ("default", "fastapi rag", "none"):
            selected_model = base_model
        else:
            selected_model = raw_model

        if self.llm is None:
            return None, str(selected_model)

        if not hasattr(self, "_llm_instance_cache"):
            self._llm_instance_cache: dict[str, Any] = {}

        if selected_model not in self._llm_instance_cache:
            llm_model_attr = getattr(self.llm, "model", None)
            if llm_model_attr == selected_model or not isinstance(llm_model_attr, str):
                self._llm_instance_cache[selected_model] = self.llm
            else:
                self._llm_instance_cache[selected_model] = _LLMProxy(self.llm, selected_model)

        return self._llm_instance_cache[selected_model], selected_model

    def _retrieve_hybrid_hits(
        self,
        query: str,
        dense_top_k: int,
        bm25_top_k: int,
        filters: dict[str, Any] | None,
        rrf_k: int = 60,
    ) -> list[ScoredChunk]:
        """Robust hybrid retrieval supporting mock and production hybrid retrievers."""
        try:
            return self.hybrid_retriever.retrieve(
                query,
                dense_top_k=dense_top_k,
                bm25_top_k=bm25_top_k,
                filters=filters,
                rrf_k=rrf_k,
            )
        except TypeError:
            return self.hybrid_retriever.retrieve(
                query,
                dense_top_k=dense_top_k,
                bm25_top_k=bm25_top_k,
                filters=filters,
            )

    def query(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        model: str | None = None,
        active_document_id: str | None = None,
        active_document_name: str | None = None,
        selected_document_ids: list[str] | None = None,
        document_scope: str | None = None,
    ) -> RAGResponse:
        """Execute end-to-end document-faithful RAG pipeline."""
        return self._query_internal(
            user_query=user_query,
            filters=filters,
            history=history,
            model=model,
            active_document_id=active_document_id,
            active_document_name=active_document_name,
            selected_document_ids=selected_document_ids,
            document_scope=document_scope,
        )

    def _query_internal(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        model: str | None = None,
        active_document_id: str | None = None,
        active_document_name: str | None = None,
        selected_document_ids: list[str] | None = None,
        document_scope: str | None = None,
    ) -> RAGResponse:
        total_start = time.perf_counter()
        stage_timings: dict[str, float] = {}

        # 0a. Query Classification & Intent Selection
        t0 = time.perf_counter()
        classification = self.query_router.classify(user_query, history=history)
        strategy = classification.strategy
        fidelity_mode = _detect_fidelity_mode(user_query)
        stage_timings["query_routing"] = round((time.perf_counter() - t0) * 1000, 2)

        req_llm, selected_model = self._get_effective_llm(model)

        # Conversational / Greeting intent check
        if classification.category == QueryCategory.CONVERSATIONAL or self.query_rewriter.is_conversational(
            user_query
        ):
            greeting_answer = (
                "Hello! How can I assist you today? Feel free to ask any questions regarding company policies, "
                "AI agent architectures, code implementations, or any uploaded documentation."
            )
            total_elapsed = round((time.perf_counter() - total_start) * 1000, 2)
            trace = RAGTrace(
                query=user_query,
                rewritten_query=None,
                sub_queries=[],
                query_type=classification.category.value,
                routing_confidence=classification.confidence,
                retrieval_strategy="conversational_bypass",
                query_scope="global",
                retrieved_candidate_count=0,
                post_rerank_count=0,
                final_context_count=0,
                execution_time_ms=total_elapsed,
                stage_timings_ms={"conversational_bypass": total_elapsed},
                fallback_reason="conversational_greeting",
                faithfulness_checked=False,
                faithfulness_passed=True,
                verification_report=None,
                verification_score=None,
                retry_count=0,
                retry_reasons=[],
                cache_hit=False,
                cache_similarity=None,
                anchor_section="Conversational Bypass",
                evidence_sufficiency_passed=True,
                generation_model=selected_model,
                grounding_validation_passed=True,
            )
            _log_rag_trace(trace)
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

        # 0b. Pre-rewrite Cache Lookup
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
                    generation_model=selected_model,
                    evidence_sufficiency_passed=True,
                    grounding_validation_passed=True,
                )
                _log_rag_trace(trace)
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

        # 1. Scope Resolution
        known_docs: dict[str, str] = {}
        if self.docstore:
            for c in self.docstore.values():
                if c.metadata and c.metadata.document_id and c.metadata.source_file:
                    known_docs[c.metadata.document_id] = c.metadata.source_file

        if filters:
            if "document_id" in filters and not active_document_id:
                if isinstance(filters["document_id"], list):
                    selected_document_ids = filters["document_id"]
                else:
                    active_document_id = str(filters["document_id"])
            if "source_file" in filters and not active_document_name:
                active_document_name = str(filters["source_file"])

        scope_decision = self.scope_resolver.resolve_scope(
            query=user_query,
            active_document_id=active_document_id,
            active_document_name=active_document_name,
            selected_document_ids=selected_document_ids,
            explicit_scope=document_scope,
            known_documents=known_docs,
        )

        # 1a. Query Rewrite
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
        elif filters:
            applied_filters = {**filters}

        if scope_decision.scope == DocumentRetrievalScope.CURRENT_DOCUMENT:
            if scope_decision.active_document_id:
                applied_filters["document_id"] = scope_decision.active_document_id
            elif scope_decision.active_document_name:
                applied_filters["source_file"] = scope_decision.active_document_name
            if scope_decision.page_number is not None:
                applied_filters["page_number"] = scope_decision.page_number
            if scope_decision.section_number is not None:
                applied_filters["section_number"] = scope_decision.section_number
        elif scope_decision.scope == DocumentRetrievalScope.SELECTED_DOCUMENTS:
            if scope_decision.allowed_document_ids:
                if len(scope_decision.allowed_document_ids) == 1:
                    applied_filters["document_id"] = scope_decision.allowed_document_ids[0]
                else:
                    applied_filters["document_id"] = scope_decision.allowed_document_ids

        # Fast path check for high-confidence factual questions
        is_fast_path = (
            classification.category == QueryCategory.FACTUAL
            and classification.confidence >= 0.85
            and not (history and len(history) > 0 and self.query_rewriter._is_followup_query(user_query))
            and not scope_decision.is_structural_query
        )

        enable_verification = getattr(settings, "enable_answer_verification", True)
        max_retries = 0 if is_fast_path else (self.retry_engine.max_retries if self.retry_engine else 2)

        current_strategy = strategy.model_copy(deep=True)
        if is_fast_path:
            current_strategy.enable_multi_query = False

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
        cross_document_count = 0
        telemetry_extra: dict[str, Any] = {}

        while attempt <= max_retries:
            prefix = f"_att{attempt}" if attempt > 0 else ""

            # 2. Multi-Query Generation & Structural Query Expansion
            t0 = time.perf_counter()
            if not is_fast_path and scope_decision.is_structural_query:
                sub_queries = scope_decision.structural_subqueries
            elif not is_fast_path and (current_strategy.enable_multi_query or rewrite_res.is_comprehensive_list):
                sub_queries = self.multi_query_gen.generate_subqueries(rewrite_res.rewritten_query)
            else:
                sub_queries = [rewrite_res.rewritten_query]
            stage_timings[f"multi_query{prefix}"] = round((time.perf_counter() - t0) * 1000, 2)

            # 3. Hybrid Search (Dense + BM25 with RRF)
            t0 = time.perf_counter()
            search_filters = applied_filters if applied_filters else None
            retrieval_cache = get_retrieval_cache()
            candidate_chunks = []
            cache_hit_retrieval = False

            if getattr(settings, "retrieval_cache_enabled", True) and len(sub_queries) == 1:
                cached_cands = retrieval_cache.get(
                    sub_queries[0], filters=search_filters, top_k=current_strategy.dense_top_k
                )
                if cached_cands:
                    candidate_chunks = cached_cands
                    cache_hit_retrieval = True

            if not candidate_chunks:
                candidate_map: dict[str, ScoredChunk] = {}
                for sq in sub_queries:
                    hits = self._retrieve_hybrid_hits(
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

                # Filter relaxation fallback if 0 results
                if not candidate_chunks and search_filters and getattr(settings, "enable_filter_fallback_relaxation", True):
                    relaxed_filters = None
                    if "document_id" in search_filters:
                        relaxed_filters = {"document_id": search_filters["document_id"]}
                    elif "source_file" in search_filters:
                        relaxed_filters = {"source_file": search_filters["source_file"]}
                    filter_relaxed = True
                    applied_filters = relaxed_filters or {}
                    candidate_map = {}
                    for sq in sub_queries:
                        hits = self._retrieve_hybrid_hits(
                            sq,
                            dense_top_k=current_strategy.dense_top_k,
                            bm25_top_k=current_strategy.bm25_top_k,
                            filters=relaxed_filters,
                            rrf_k=current_strategy.rrf_k,
                        )
                        for sc in hits:
                            cid = sc.chunk.id
                            if cid not in candidate_map or (sc.score or 0.0) > (
                                candidate_map[cid].score or 0.0
                            ):
                                candidate_map[cid] = sc
                    candidate_chunks = list(candidate_map.values())

                # Enforce hard document scope validation
                if scope_decision.scope == DocumentRetrievalScope.CURRENT_DOCUMENT:
                    valid_cands = []
                    for sc in candidate_chunks:
                        d_id = sc.chunk.metadata.document_id
                        s_name = sc.chunk.metadata.source_file
                        is_match = False
                        if scope_decision.active_document_id and d_id == scope_decision.active_document_id:
                            is_match = True
                        elif scope_decision.active_document_name and s_name == scope_decision.active_document_name:
                            is_match = True
                        if is_match:
                            valid_cands.append(sc)
                        else:
                            cross_document_count += 1
                    candidate_chunks = valid_cands
                elif scope_decision.scope == DocumentRetrievalScope.SELECTED_DOCUMENTS and scope_decision.allowed_document_ids:
                    valid_cands = []
                    for sc in candidate_chunks:
                        d_id = sc.chunk.metadata.document_id
                        if d_id in scope_decision.allowed_document_ids:
                            valid_cands.append(sc)
                        else:
                            cross_document_count += 1
                    candidate_chunks = valid_cands

                candidate_chunks.sort(key=lambda x: x.score or 0.0, reverse=True)
                candidate_pool_limit = max(len(candidate_chunks), current_strategy.rerank_top_n * 3, 15)
                candidate_chunks = candidate_chunks[:candidate_pool_limit]

                if getattr(settings, "retrieval_cache_enabled", True) and len(sub_queries) == 1 and candidate_chunks:
                    retrieval_cache.set(
                        sub_queries[0],
                        candidate_chunks,
                        filters=search_filters,
                        top_k=current_strategy.dense_top_k,
                        ttl=getattr(settings, "retrieval_cache_ttl_seconds", 3600),
                    )

            stage_timings[f"hybrid_retrieval{prefix}"] = 0.1 if cache_hit_retrieval else round((time.perf_counter() - t0) * 1000, 2)

            if not candidate_chunks:
                if scope_decision.active_document_name:
                    unanswerable_text = f"I could not find this information in the active document '{scope_decision.active_document_name}'."
                elif scope_decision.active_document_id:
                    unanswerable_text = f"I could not find this information in the active document '{scope_decision.active_document_id}'."
                else:
                    unanswerable_text = "I could not find this information in the provided document."

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
                enable_expansion=False if is_fast_path else current_strategy.enable_parent_expansion,
            )

            # 5b. Evidence Sufficiency Gate & Cross-Page Vision Fallback (Phases 3, 4, 8)
            expanded_chunks, telemetry_extra = self._apply_cross_page_vision_fallback_if_needed(
                expanded_chunks,
                user_query=user_query,
                intent=classification.category,
            )

            # 5c. Complementary Chunk Packing
            expanded_chunks = self.compressor.pack_complementary_chunks(
                expanded_chunks, user_query, max_chunks=current_strategy.rerank_top_n
            )
            formatted_context = self.compressor.format_context_for_prompt(expanded_chunks)
            stage_timings[f"context_expansion{prefix}"] = round((time.perf_counter() - t0) * 1000, 2)

            # 6. LLM Grounded Answer Synthesis
            t0 = time.perf_counter()
            mode_prompt_str = (
                EXACT_MODE_INSTRUCTIONS
                if fidelity_mode == "exact"
                else EXPLAIN_MODE_INSTRUCTIONS
                if fidelity_mode == "explain"
                else IMPLEMENT_MODE_INSTRUCTIONS
                if fidelity_mode == "implement"
                else ""
            )

            history_text = _format_history_for_prompt(history)
            refinement_str = f"\nRefinement Instructions:\n{prompt_refinement}\n" if prompt_refinement else ""
            prompt = GROUNDED_SYSTEM_PROMPT.format(
                mode_instructions=mode_prompt_str,
                refinement_directive=refinement_str,
                context_text=formatted_context,
                history_text=history_text,
                query=user_query,
            )

            max_tokens = getattr(settings, "max_new_tokens_complex", 1024)
            if classification.category == QueryCategory.FACTUAL:
                max_tokens = getattr(settings, "max_new_tokens_factual", 256)
            elif classification.category in (QueryCategory.PROCEDURAL, QueryCategory.IMPLEMENTATION, QueryCategory.CODE):
                max_tokens = getattr(settings, "max_new_tokens_technical", 768)

            if req_llm is not None:
                try:
                    try:
                        raw_answer = str(
                            req_llm.complete(
                                prompt,
                                temperature=current_strategy.temperature,
                                max_new_tokens=max_tokens,
                            )
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
            stage_timings[f"citation_extraction{prefix}"] = round((time.perf_counter() - t0) * 1000, 2)

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

            if report.composite_score > best_score or best_report is None:
                best_score = report.composite_score
                best_answer = answer_text
                best_citations = citations
                best_context_chunks = expanded_chunks
                best_candidate_chunks = candidate_chunks
                best_reranked_chunks = reranked_chunks
                best_report = report

            if report.passed or is_fast_path:
                best_answer = answer_text
                best_citations = citations
                best_context_chunks = expanded_chunks
                best_candidate_chunks = candidate_chunks
                best_reranked_chunks = reranked_chunks
                best_report = report
                break

            if attempt >= max_retries or not self.retry_engine.should_retry(attempt, report):
                break

            current_strategy, prompt_refinement = self.retry_engine.prepare_retry(
                attempt=attempt,
                report=report,
                strategy=current_strategy,
                query=user_query,
            )
            attempt += 1

        total_elapsed = round((time.perf_counter() - total_start) * 1000, 2)
        final_context_documents = list({sc.chunk.metadata.source_file for sc in best_context_chunks if sc.chunk.metadata and sc.chunk.metadata.source_file})

        # Count evidence types
        text_cnt = sum(1 for sc in best_context_chunks if str(sc.chunk.metadata.content_type).lower() in ("prose", "contenttype.prose"))
        code_cnt = sum(1 for sc in best_context_chunks if str(sc.chunk.metadata.content_type).lower() in ("code", "contenttype.code") or "```" in sc.chunk.text)
        diag_cnt = sum(1 for sc in best_context_chunks if "diagram" in str(sc.chunk.metadata.content_type).lower() or sc.chunk.metadata.extra.get("visual_type") == "diagram_architecture")
        tab_cnt = sum(1 for sc in best_context_chunks if "table" in str(sc.chunk.metadata.content_type).lower() or sc.chunk.metadata.extra.get("visual_type") == "table_data")

        fallback_reason = "none"
        if req_llm is None:
            fallback_reason = "llm_offline_fallback"
        elif best_report is not None and not best_report.passed:
            fallback_reason = "retry_exhausted_fallback"

        anchor_sec = telemetry_extra.get("anchor_section")
        if not anchor_sec and best_context_chunks:
            anchor_sec = best_context_chunks[0].chunk.metadata.section_title or best_context_chunks[0].chunk.metadata.section_path

        trace = RAGTrace(
            query=user_query,
            rewritten_query=rewrite_res.rewritten_query,
            sub_queries=sub_queries,
            query_type=classification.category.value,
            routing_confidence=classification.confidence,
            retrieval_strategy=current_strategy.name if current_strategy else strategy.name,
            query_scope=scope_decision.scope.value,
            active_document_id=scope_decision.active_document_id,
            active_document_name=scope_decision.active_document_name,
            allowed_document_ids=scope_decision.allowed_document_ids,
            cross_document_chunks_rejected=cross_document_count,
            final_context_documents=final_context_documents,
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
            anchor_section=anchor_sec,
            evidence_text_count=text_cnt,
            evidence_code_count=code_cnt,
            evidence_diagram_count=diag_cnt,
            evidence_table_count=tab_cnt,
            section_expansion=telemetry_extra.get("section_expansion", False),
            adjacent_page_check=telemetry_extra.get("adjacent_page_check", False),
            vision_fallback=telemetry_extra.get("vision_fallback", False),
            vision_model=str(telemetry_extra.get("vision_model", "qwen2.5vl:7b")) if isinstance(telemetry_extra.get("vision_model"), str) else "qwen2.5vl:7b",
            vision_cache_status=str(telemetry_extra.get("vision_cache_status", "N/A")) if isinstance(telemetry_extra.get("vision_cache_status"), str) else "N/A",
            evidence_sufficiency_passed=telemetry_extra.get("evidence_sufficiency_passed", True),
            generation_model=selected_model,
            grounding_validation_passed=best_report.passed if best_report else True,
        )

        _log_rag_trace(trace)

        if best_citations and len(best_citations) > 0 and best_answer and (best_report is None or best_report.passed):
            self._queue_cache_write(user_query, best_answer, best_citations, model_name=selected_model)

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
            paragraphs.append(f"{text_snippet} [Source {idx}]")

        return f"Based on the official document regarding '{user_query}':\n\n" + "\n\n".join(paragraphs)

    async def stream_query(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        model: str | None = None,
        active_document_id: str | None = None,
        active_document_name: str | None = None,
        selected_document_ids: list[str] | None = None,
        document_scope: str | None = None,
        cancel_token: Any = None,
    ):
        """Streaming RAG pipeline yielding SSE token events and telemetry."""
        from starlette.concurrency import iterate_in_threadpool

        async for chunk in iterate_in_threadpool(
            self._stream_query_internal(
                user_query=user_query,
                filters=filters,
                history=history,
                model=model,
                active_document_id=active_document_id,
                active_document_name=active_document_name,
                selected_document_ids=selected_document_ids,
                document_scope=document_scope,
                cancel_token=cancel_token,
            )
        ):
            yield chunk

    def _stream_query_internal(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        model: str | None = None,
        active_document_id: str | None = None,
        active_document_name: str | None = None,
        selected_document_ids: list[str] | None = None,
        document_scope: str | None = None,
        cancel_token: Any = None,
    ) -> Generator[dict[str, Any], None, None]:
        # Reuse robust query execution for full consistency
        resp = self.query(
            user_query=user_query,
            filters=filters,
            history=history,
            model=model,
            active_document_id=active_document_id,
            active_document_name=active_document_name,
            selected_document_ids=selected_document_ids,
            document_scope=document_scope,
        )

        yield {
            "type": "retrieval_done",
            "stage_timings": resp.trace.stage_timings_ms,
            "candidate_count": resp.trace.retrieved_candidate_count,
            "reranked_count": resp.trace.post_rerank_count,
            "context_count": resp.trace.final_context_count,
            "cache_hit": resp.trace.cache_hit,
        }

        # Stream words smoothly
        words = resp.answer.split(" ")
        for i, word in enumerate(words):
            if cancel_token and cancel_token.is_set():
                return
            chunk_text = word + (" " if i < len(words) - 1 else "")
            yield {"type": "token", "content": chunk_text}

        yield {
            "type": "done",
            "answer": resp.answer,
            "citations": resp.citations,
            "context_chunks": resp.context_chunks,
            "trace": resp.trace,
            "model": resp.model,
            "token_usage": resp.token_usage,
            "total_elapsed_ms": resp.trace.execution_time_ms,
            "cache_hit": resp.trace.cache_hit,
        }
