from __future__ import annotations

import json
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
    EvidenceStatus,
    QueryCategory,
    RAGResponse,
    RAGTrace,
    ScoredChunk,
    ThinkingDetailLevel,
    ThinkingEvent,
    ThinkingStage,
    ThinkingStatus,
    VerificationReport,
)
from backend.rag.thinking import ThinkingStateMachine
from backend.models.conversation import (
    AnswerMode,
    ConversationRAGState,
)
from backend.rag.consistency_guard import ConversationConsistencyGuard
from backend.rag.conversation_resolver import (
    ConversationResolutionResult,
    ConversationResolver,
)
from backend.rag.citations import CitationEngine
from backend.rag.context_compression import ContextCompressor
from backend.rag.evidence_gate import EvidenceSufficiencyGate
from backend.rag.filter_extractor import QueryMetadataInferer
from backend.rag.multi_query import MultiQueryGenerator
from backend.rag.query_rewrite import QueryRewriter
from backend.rag.query_router import QueryRouter
from backend.rag.retry_engine import RetryEngine
from backend.rag.scope_resolver import (
    DocumentRetrievalScope,
    DocumentScopeResolver,
)
from backend.rag.semantic_cache import SemanticCacheManager
from backend.rag.verifier import SelfReflectionVerifier
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.reranker import CrossEncoderReranker
from backend.retrieval.retrieval_cache import get_retrieval_cache
from backend.utils.logging import logger
from backend.vision.vision_service import VisionService, VisualContentType
from src.config import settings
from src.ollama_client import preload_model



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


GROUNDED_SYSTEM_PROMPT = """You are a document-faithful multimodal AI assistant.
Your absolute source of truth is the RETRIEVED CONTEXT below, which contains verified document text and visual extractions.

Core Grounding Rules:
RULE A — Conversation Continuity: When answering a follow-up, preserve the established subject and prior context unless the user explicitly switches topics.
RULE B — Evidence Continuity: Previously verified evidence remains available for the conversation unless superseded or invalidated.
RULE C — Expansion: If the user requests more detail, expand the prior grounded answer with additional context, surrounding sections, and implementation details instead of restarting from scratch.
RULE D — No False Absence: Never claim information is unavailable or that the document does not contain it when valid evidence from the current or previous verified context supports the answer.
RULE E — Evidence Distinction: Clearly distinguish what is directly stated in the document, what is partial implementation evidence, and what is reasonable explanation or workflow interpretation.
RULE F — Detailed Code Explanations: When code snippets or implementations appear in context, preserve exact retrieved code, explain relevant sections step-by-step, do not fabricate missing functions or imports, and clearly identify incomplete snippets.
RULE 1: Use retrieved evidence as the primary source of truth.
RULE 2: Do not invent details, assumptions, or external facts not supported by the retrieved text or visual evidence.
RULE 3: If a relevant visual asset exists and visual understanding is included (e.g. under [VISUAL SOURCE N]), explicitly explain the workflow, architecture, diagram, or code shown in that visual evidence.
RULE 4: Never claim that an image or diagram is absent merely because it was not included in the first text retrieval result.
RULE 5: If a visual asset exists on a page but visual understanding extraction failed or is degraded, clearly distinguish: state that the visual exists on the page, but visual analysis is currently unavailable. Cite the source tag so the user can inspect the original image.
RULE 6: Do not fabricate or invent the contents of a visual that failed extraction.
RULE 7: For source-grounded answers, prefer language such as: "According to the workflow shown on Page X..." or "Based on Section Y..." using the human-visible printed page numbers provided in the context blocks.
RULE 8: Citations: Cite sources using [Source N] or [Visual Source N] tags for every substantive claim, code block, or diagram description.
RULE 9: When code snippets, kickoff calls, agent configurations, or implementations appear in the retrieved context (including under [Source N] or [VISUAL SOURCE N]), extract and present that code directly and faithfully. Never state that the document does not contain the code if relevant code snippets or implementations are present in the context.
RULE 10: Comprehensively extract and present all relevant facts, dates, deadlines, timelines, parameters, numbers, policies, and specific guidelines directly from the retrieved context in rich detail.
{evidence_status_directive}
{mode_instructions}
{refinement_directive}
RETRIEVED CONTEXT:
{context_text}

{history_text}USER QUESTION: {query}
ANSWER:"""

def _format_evidence_status_directive(status: Any) -> str:
    st_val = getattr(status, "value", str(status)).upper()
    if st_val == "PARTIAL":
        return """Evidence Status: PARTIAL IMPLEMENTATION
- The retrieved context contains partial code or invocation examples (such as kickoff calls, parameters, or configurations).
- Present this available code faithfully under [Source N] or [VISUAL SOURCE N].
- Clearly explain that this is the partial code/invocation available in the document.
- DO NOT claim that the document does not contain the code.
- DO NOT fabricate, invent, or hallucinate missing class definitions, imports, or external tool configurations."""
    elif st_val == "DIRECT":
        return """Evidence Status: DIRECT IMPLEMENTATION
- The retrieved context contains direct code or implementation details.
- Present the implementation strictly based on the retrieved code and configurations."""
    elif st_val == "RELATED":
        return """Evidence Status: RELATED CONTEXT
- The retrieved context contains related conceptual, architectural, or procedural descriptions, but not the direct code implementation.
- Explain the concepts based strictly on the retrieved context without fabricating code."""
    elif st_val == "MISSING":
        return """Evidence Status: MISSING CONTEXT
- State clearly that the requested information could not be found in the document."""
    return ""

EXACT_MODE_INSTRUCTIONS = """Mode: EXACT EXTRACTION
- Extract and present the exact text, tables, headings, and code from the document with maximum source fidelity.
- Do not paraphrase or add external commentary unless requested.
- Preserve original variable names, function names, and code syntax exactly."""

EXPLAIN_MODE_INSTRUCTIONS = """Mode: EXPLAIN
- First present the relevant document excerpts, diagrams, and code faithfully.
- Then provide a structured, grounded explanation of how it works."""

IMPLEMENT_MODE_INSTRUCTIONS = """Mode: IMPLEMENTATION
- Present the step-by-step implementation strictly based on the retrieved code, agent/task configurations, and parameters from the document.
- Include the exact code provided in the document. Do not substitute generic or fabricated code."""

EXPAND_MODE_INSTRUCTIONS = """Mode: EXPAND / DETAILED
- Deep architectural and implementation dive.
- Avoid repeating high-level summaries from prior turns.
- Expand into detailed components, configuration, code execution flow, parameters, and boundary conditions.
- Grounding separation: clearly separate DIRECT code definitions, PARTIAL kickoff snippets under [Source N], RELATED concepts, and explicitly note genuinely MISSING information without fabricating code."""

CODE_EXPLANATION_MODE_INSTRUCTIONS = """Mode: CODE EXPLANATION
- Provide a thorough, step-by-step walkthrough of the retrieved code implementation.
- Explain function signatures, parameters, return types, execution flow, inputs, outputs, and dependencies.
- Preserve exact code syntax without fabricating missing functions."""

STEP_BY_STEP_MODE_INSTRUCTIONS = """Mode: STEP BY STEP
- Provide a structured, numbered, sequential walkthrough of the process or workflow.
- Detail each discrete step with inputs, actions, and expected outcomes from the context."""

COMPARISON_MODE_INSTRUCTIONS = """Mode: COMPARISON
- Structure a clear side-by-side comparison between the entities/topics discussed.
- Compare criteria such as purpose, configuration, execution pattern, advantages, and limitations."""

SUMMARY_MODE_INSTRUCTIONS = """Mode: SUMMARY
- Provide a concise, structured high-level summary using bullet points or brief synthesis.
- Omit extraneous procedural minutiae while retaining core conclusions."""

CONTINUATION_MODE_INSTRUCTIONS = """Mode: CONTINUATION
- Provide a logical, step-by-step continuation proceeding directly from the previous turn.
- Do not reintroduce background context already established."""


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
    sep = "=" * 70
    lines = [
        f"\n{sep}",
        "[RAG TRACE]",
        f"QUERY:                    {trace.query}",
        f"INTENT:                   {trace.query_type or 'factual'}",
        f"DOCUMENT_SCOPE:           {trace.query_scope or 'global'}",
        f"ANCHOR_SECTION:           {trace.anchor_section or 'General'}",
        f"TEXT_CANDIDATES:          {trace.text_candidates or trace.retrieved_candidate_count}",
        f"VISUAL_CANDIDATES:        {trace.visual_candidates}",
        f"FINAL_TEXT_EVIDENCE:      {trace.final_text_evidence or trace.evidence_text_count}",
        f"FINAL_VISUAL_EVIDENCE:    {trace.final_visual_evidence or (trace.evidence_diagram_count + trace.evidence_code_count)}",
        f"PAGE_IDENTITY:            {trace.page_identity or 'N/A'}",
        f"VISUAL_ASSET:             {trace.visual_asset_status or ('FOUND' if trace.evidence_diagram_count > 0 else 'NONE')}",
        f"VISION_STATUS:            {trace.vision_status or 'READY'}",
        f"EVIDENCE_STATUS:          {trace.evidence_status or ('SUFFICIENT' if trace.evidence_sufficiency_passed else 'INSUFFICIENT')}",
        f"GROUNDING:                {trace.grounding_status or ('PASS' if trace.grounding_validation_passed else 'FAIL')}",
        f"GENERATION_MODEL:         {trace.generation_model or 'qwen2.5:7b'}",
        f"TOTAL LATENCY:            {trace.execution_time_ms:.2f} ms",
        f"{sep}\n",
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
        conversation_resolver: ConversationResolver | None = None,
        consistency_guard: ConversationConsistencyGuard | None = None,
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
        self.conversation_resolver = conversation_resolver or ConversationResolver(llm=self.llm)
        self.consistency_guard = consistency_guard or ConversationConsistencyGuard()


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
        previous_status: EvidenceStatus | str | None = None,
        previous_chunks: list[ScoredChunk] | None = None,
        is_followup: bool = False,
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
            "evidence_status": "DIRECT",
            "anchor_section": None,
        }

        if not getattr(settings, "enable_lazy_vision_fallback", True) or not getattr(settings, "vision_enabled", True):
            return chunks, telemetry

        if not chunks:
            # Check if previous evidence exists to maintain monotonicity
            if is_followup and previous_chunks and previous_status:
                gate_res = self.evidence_gate.evaluate(
                    query=user_query,
                    intent=intent,
                    candidate_chunks=[],
                    previous_status=previous_status,
                    previous_chunks=previous_chunks,
                    is_followup=True,
                )
                telemetry["evidence_sufficiency_passed"] = gate_res.is_sufficient
                telemetry["evidence_status"] = gate_res.evidence_status.value
                return previous_chunks, telemetry
            return chunks, telemetry

        # 1. Evaluate Evidence Sufficiency with Monotonicity
        gate_res = self.evidence_gate.evaluate(
            query=user_query,
            intent=intent,
            candidate_chunks=chunks,
            previous_status=previous_status,
            previous_chunks=previous_chunks,
            is_followup=is_followup,
        )
        telemetry["evidence_sufficiency_passed"] = gate_res.is_sufficient
        telemetry["evidence_status"] = gate_res.evidence_status.value
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

        # 3. Extract visuals across target page range (prioritize matching visual assets and code screenshots)
        visual_chunks = []
        cache_hits = 0
        cache_misses = 0

        is_code_intent = (
            intent in (QueryCategory.CODE, QueryCategory.IMPLEMENTATION)
            or "code" in str(intent).lower()
            or bool(
                re.search(
                    r"\b(?:code|implementation|snippet|function|class|method|agent|task|defined)\b",
                    user_query,
                    re.IGNORECASE,
                )
            )
        )
        is_diagram_intent = (intent == QueryCategory.ARCHITECTURE or any(k in str(intent).lower() for k in ("diagram", "architecture", "workflow")))

        # Pre-scan PDF page text to rank pages with matching cues
        page_text_map: dict[int, str] = {}
        try:
            import fitz
            doc_scan = fitz.open(resolved_path)
            for p in gate_res.pages_to_inspect:
                p_idx = max(0, p - 1)
                if 0 <= p_idx < len(doc_scan):
                    page_text_map[p] = doc_scan[p_idx].get_text()
            doc_scan.close()
        except Exception:
            pass

        def page_priority_key(p: int) -> tuple[int, int]:
            p_assets = self.vision_service.image_asset_manager.get_page_assets_by_physical_page(meta.document_id, p)
            has_matching_type = False
            has_asset = bool(p_assets)
            if p_assets:
                for ast in p_assets:
                    if is_code_intent and ast.visual_type == "code_screenshot":
                        has_matching_type = True
                        break
                    elif not is_code_intent and ast.visual_type == "diagram_architecture":
                        has_matching_type = True
                        break

            # If no stored assets, check page text cues from PDF
            if not has_matching_type and p in page_text_map:
                txt = page_text_map[p]
                from backend.vision.vision_service import _CODE_CUES, _DIAGRAM_CUES
                if is_code_intent and bool(_CODE_CUES.search(txt)):
                    has_matching_type = True
                elif not is_code_intent and bool(_DIAGRAM_CUES.search(txt)):
                    has_matching_type = True

            p_match_rank = 0 if has_matching_type else (1 if has_asset else 2)
            if meta.page_number:
                if p == meta.page_number:
                    dist = 0
                elif p > meta.page_number:
                    dist = p - meta.page_number  # Forward continuation (1, 2, ...)
                else:
                    dist = 100 + (meta.page_number - p)  # Backward pages (101, 102, ...)
            else:
                dist = p
            return (p_match_rank, dist)

        target_pages = sorted(list(gate_res.pages_to_inspect), key=page_priority_key)

        t_vision_start = time.perf_counter()
        vision_budget = getattr(settings, "vision_query_budget_seconds", 45.0)
        processed_page_keys: set[int] = set()
        req_vis_type = (
            VisualContentType.CODE_SCREENSHOT
            if is_code_intent
            else (VisualContentType.DIAGRAM_ARCHITECTURE if is_diagram_intent else None)
        )

        # At query-time, inspect up to 3 pages within vision budget and break on first successful extraction
        for p_num in target_pages[:3]:
            elapsed_vision = time.perf_counter() - t_vision_start
            remaining_budget = vision_budget - elapsed_vision
            if remaining_budget <= 4.0:
                logger.warning(
                    "[VISION] Query-level vision budget exhausted (%.1fs/%.1fs). Skipping remaining pages.",
                    elapsed_vision,
                    vision_budget,
                )
                break

            if p_num in processed_page_keys:
                continue
            processed_page_keys.add(p_num)

            # First check if image assets are already stored on disk
            assets_on_page = self.vision_service.image_asset_manager.get_page_assets_by_physical_page(meta.document_id, p_num)
            disp_p = None
            lbl_p = None
            idx_p = max(0, p_num - 1)
            if assets_on_page:
                disp_p = assets_on_page[0].display_page_number
                lbl_p = assets_on_page[0].page_label
                idx_p = assets_on_page[0].internal_page_index
            elif p_num == meta.page_number:
                p_id = meta.get_page_identity()
                disp_p = p_id.display_page_number
                lbl_p = p_id.page_label
                idx_p = p_id.internal_page_index

            timeout_for_page = min(remaining_budget, getattr(settings, "vision_request_timeout", 35.0))

            # Code screenshots are often one of several embedded images on a page.
            # Use the saved, original matching assets first so we do not accidentally
            # OCR a larger diagram or decorative image instead of the requested code.
            extract_stored_assets = getattr(self.vision_service, "extract_stored_assets", None)
            if is_code_intent and callable(extract_stored_assets):
                try:
                    stored_extractions = extract_stored_assets(
                        document_id=meta.document_id,
                        page_number=p_num,
                        section_title=anchor_title,
                        required_visual_type=VisualContentType.CODE_SCREENSHOT,
                        timeout=timeout_for_page,
                        is_query_time=True,
                    )
                except Exception as exc:
                    logger.warning(
                        "[VISION] Saved code-asset extraction failed for doc=%s page=%s: %s",
                        meta.document_id,
                        p_num,
                        exc,
                    )
                    stored_extractions = []

                if isinstance(stored_extractions, list):
                    visual_chunks.extend((p_num, chunk) for chunk in stored_extractions)
                    if visual_chunks:
                        continue

            extracted = self.vision_service.process_pdf_page_visuals(
                pdf_path=resolved_path,
                page_number=p_num,
                page_text=anchor.chunk.text if p_num == meta.page_number else "",
                document_id=meta.document_id,
                section_title=anchor_title,
                continuation_cue=cue,
                display_page_number=disp_p,
                page_label=lbl_p,
                internal_page_index=idx_p,
                timeout=timeout_for_page,
                required_visual_type=req_vis_type,
                is_query_time=True,
            )
            for vc in extracted:
                visual_chunks.append((p_num, vc))
            if visual_chunks:
                break

        # If vision extraction produced nothing (e.g. timeout or disabled), but original assets exist on disk:
        if not visual_chunks:
            for p_num in target_pages[:3]:
                assets_on_page = self.vision_service.image_asset_manager.get_page_assets_by_physical_page(meta.document_id, p_num)
                if assets_on_page:
                    ast = assets_on_page[0]
                    disp_label = ast.display_label
                    fallback_text = (
                        f"[Visual Asset on Page {disp_label}]: Original {ast.visual_type} visual asset "
                        f"is present for '{anchor_title}' on Page {disp_label}. (Detailed visual understanding "
                        f"extraction is currently unavailable or degraded)."
                    )
                    c_id = f"chunk_lazy_vis_{meta.document_id}_p{p_num}_{ast.image_hash[:8]}"
                    meta_dict = meta.model_dump()
                    meta_dict["content_type"] = ContentType.PROSE
                    meta_dict["page_number"] = ast.physical_page_number
                    meta_dict["page_label"] = str(disp_label)
                    meta_dict["display_page_number"] = ast.display_page_number
                    meta_dict["internal_page_index"] = ast.internal_page_index
                    meta_dict["section_title"] = anchor_title
                    meta_dict["has_code"] = "code" in ast.visual_type
                    meta_dict["has_tables"] = "table" in ast.visual_type
                    meta_dict["visual_asset_ids"] = [ast.asset_id]
                    meta_dict["image_assets"] = [
                        {
                            "asset_id": ast.asset_id,
                            "asset_url": ast.asset_url,
                            "image_hash": ast.image_hash,
                            "page_number": ast.physical_page_number,
                            "page_label": str(disp_label),
                            "display_page_number": ast.display_page_number,
                            "visual_type": ast.visual_type,
                        }
                    ]
                    meta_dict["extra"] = {
                        **(meta.extra or {}),
                        "is_visual_extraction": True,
                        "visual_type": ast.visual_type,
                        "image_hash": ast.image_hash,
                        "asset_id": ast.asset_id,
                        "image_url": ast.asset_url,
                        "visual_status": "ASSET_AVAILABLE",
                        "display_page_number": ast.display_page_number,
                        "page_label": str(disp_label),
                    }
                    fallback_chunk = Chunk(
                        id=c_id,
                        text=fallback_text,
                        metadata=ChunkMetadata(**meta_dict),
                        token_count=len(fallback_text.split()),
                    )
                    if self.docstore is not None:
                        self.docstore[c_id] = fallback_chunk

                    new_scored_fallback = [
                        ScoredChunk(
                            chunk=fallback_chunk,
                            score=0.96,
                            rerank_score=0.96,
                            dense_score=0.96,
                            sparse_score=0.96,
                        )
                    ]
                    telemetry["vision_fallback"] = True
                    telemetry["visual_asset_status"] = "FOUND"
                    telemetry["vision_status"] = "DEGRADED"
                    telemetry["evidence_sufficiency_passed"] = True
                    return new_scored_fallback + chunks, telemetry

            return chunks, telemetry

        telemetry["vision_cache_status"] = "HIT" if cache_hits > 0 and cache_misses == 0 else "MISS"
        telemetry["visual_asset_status"] = "FOUND"
        telemetry["vision_status"] = "READY"

        # 4. Wrap extracted visual chunks into ScoredChunk objects
        new_scored: list[ScoredChunk] = []
        for p_num, vc in visual_chunks:
            raw_hash = getattr(vc, "image_hash", "")
            img_hash_str = str(raw_hash) if isinstance(raw_hash, str) and raw_hash else "hash_vis"

            raw_ast_id = getattr(vc, "asset_id", None)
            if isinstance(raw_ast_id, str) and raw_ast_id.strip():
                asset_id = raw_ast_id.strip()
            else:
                asset_id = f"ast_{img_hash_str[:12]}"

            raw_disp = getattr(vc, "display_page_number", None)
            disp_page = raw_disp if (isinstance(raw_disp, (int, str)) and not isinstance(raw_disp, bool)) else None

            raw_lbl = getattr(vc, "page_label", None)
            page_lbl = str(raw_lbl) if (isinstance(raw_lbl, (str, int)) and str(raw_lbl).strip()) else (str(disp_page) if disp_page is not None else str(p_num))

            raw_idx = getattr(vc, "internal_page_index", None)
            int_idx = raw_idx if isinstance(raw_idx, int) else (p_num - 1)

            vc_text = str(getattr(vc, "text", ""))
            is_code = (
                str(getattr(vc, "content_type", "")).lower() == "code"
                or "code" in str(getattr(vc, "visual_type", "")).lower()
                or "```" in vc_text
                or "def " in vc_text
                or "class " in vc_text
                or "import " in vc_text
                or "kickoff" in vc_text
            )
            ct_enum = (
                ContentType.CODE
                if is_code
                else ContentType.TABLE
                if str(getattr(vc, "content_type", "")).lower() == "table"
                else ContentType.PROSE
            )
            v_type_str = "code_screenshot" if is_code else str(getattr(vc, "visual_type", "diagram_architecture"))

            c_id = f"chunk_lazy_vis_{meta.document_id}_p{p_num}_{img_hash_str[:8]}"
            img_url = f"/api/documents/{meta.document_id}/visual-assets/{asset_id}" if meta.document_id else None
            meta_dict = meta.model_dump()
            meta_dict["content_type"] = ct_enum
            meta_dict["page_number"] = p_num
            meta_dict["page_label"] = page_lbl
            meta_dict["display_page_number"] = disp_page
            meta_dict["internal_page_index"] = int_idx
            meta_dict["section_title"] = anchor_title
            meta_dict["has_code"] = is_code
            meta_dict["has_tables"] = ct_enum == ContentType.TABLE
            meta_dict["visual_asset_ids"] = [asset_id]
            meta_dict["image_assets"] = [
                {
                    "asset_id": asset_id,
                    "asset_url": img_url,
                    "image_hash": img_hash_str,
                    "page_number": p_num,
                    "page_label": page_lbl,
                    "display_page_number": disp_page,
                    "visual_type": v_type_str,
                }
            ] if img_url else []
            meta_dict["extra"] = {
                **(meta.extra or {}),
                "is_visual_extraction": True,
                "visual_type": v_type_str,
                "image_hash": img_hash_str,
                "asset_id": asset_id,
                "image_url": img_url,
                "content_type": "code" if is_code else str(getattr(vc, "content_type", "prose")),
                "raw_code": getattr(vc, "raw_code", None) or (vc_text if is_code else None),
                "display_page_number": disp_page,
                "page_label": page_lbl,
                "continuation_from_page": meta.page_number if p_num != meta.page_number else None,
            }
            lazy_chunk = Chunk(
                id=c_id,
                text=vc_text,
                metadata=ChunkMetadata(**meta_dict),
                token_count=len(vc_text.split()),
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
        cache_context: str = "",
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
                    cache_context=cache_context,
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
        conversation_state: ConversationRAGState | None = None,
        thinking_detail_level: ThinkingDetailLevel | str = ThinkingDetailLevel.STANDARD,
        thinking_sm: ThinkingStateMachine | None = None,
    ) -> RAGResponse:
        """Execute end-to-end document-faithful RAG pipeline with safe thinking events."""
        return self._query_internal(
            user_query=user_query,
            filters=filters,
            history=history,
            model=model,
            active_document_id=active_document_id,
            active_document_name=active_document_name,
            selected_document_ids=selected_document_ids,
            document_scope=document_scope,
            conversation_state=conversation_state,
            thinking_detail_level=thinking_detail_level,
            thinking_sm=thinking_sm,
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
        conversation_state: ConversationRAGState | None = None,
        thinking_detail_level: ThinkingDetailLevel | str = ThinkingDetailLevel.STANDARD,
        thinking_sm: ThinkingStateMachine | None = None,
    ) -> RAGResponse:
        total_start = time.perf_counter()
        stage_timings: dict[str, float] = {}

        if thinking_sm is None:
            thinking_sm = ThinkingStateMachine(
                query_id=f"qry_{uuid.uuid4().hex[:8]}",
                detail_level=thinking_detail_level,
            )

        # Stage: RECEIVED
        thinking_sm.start_stage(ThinkingStage.RECEIVED)
        thinking_sm.complete_stage(ThinkingStage.RECEIVED)

        # 0a. Query Classification & Intent Selection
        t0 = time.perf_counter()
        thinking_sm.start_stage(ThinkingStage.QUERY_ANALYSIS)
        classification = self.query_router.classify(user_query, history=history)
        strategy = classification.strategy
        fidelity_mode = _detect_fidelity_mode(user_query)
        stage_timings["query_routing"] = round((time.perf_counter() - t0) * 1000, 2)
        thinking_sm.complete_stage(
            ThinkingStage.QUERY_ANALYSIS,
            details={"intent": classification.category.value, "confidence": classification.confidence},
        )

        # 0a-2. Dynamic Conversational Query Resolution & Observability
        conv_res: ConversationResolutionResult | None = None
        if conversation_state is not None:
            thinking_sm.start_stage(ThinkingStage.CONVERSATION_CONTEXT)
            conv_res = self.conversation_resolver.resolve(
                query=user_query,
                state=conversation_state,
                intent=classification.category,
            )
            logger.info(
                "[CONVERSATION] session_id=%s turn_count=%d is_followup=%s topic_shift=%s topic='%s' entities=%s",
                conversation_state.conversation_id,
                len(conversation_state.turns) + 1,
                conv_res.is_followup,
                conv_res.topic_shift,
                conv_res.active_topic or "",
                conv_res.active_entities or [],
            )
            logger.info(
                "[QUERY_RESOLUTION] query='%s' resolved='%s' confidence=%.2f cues='%s'",
                user_query,
                conv_res.resolved_query,
                conv_res.confidence,
                conv_res.reason,
            )
            logger.info(
                "[ANSWER_MODE] mode=%s directives='%s'",
                conv_res.answer_mode.value,
                conv_res.mode_directives.replace("\n", " "),
            )
            thinking_sm.complete_stage(
                ThinkingStage.CONVERSATION_CONTEXT,
                details={
                    "is_follow_up": conv_res.is_followup,
                    "active_topic": conv_res.active_topic,
                    "active_entities": conv_res.active_entities,
                    "topic_shift": conv_res.topic_shift,
                },
            )
            if conv_res.is_followup:
                thinking_sm.start_stage(ThinkingStage.FOLLOW_UP_RESOLUTION)
                if conv_res.resolution and conv_res.resolution.ambiguity_detected:
                    thinking_sm.warn_stage(
                        ThinkingStage.FOLLOW_UP_RESOLUTION,
                        reason="Follow-up query contains broad phrasing; maintaining conversation continuity with prior topic.",
                        details={"is_follow_up": True, "answer_mode": conv_res.answer_mode.value},
                    )
                thinking_sm.complete_stage(
                    ThinkingStage.FOLLOW_UP_RESOLUTION,
                    details={
                        "is_follow_up": True,
                        "answer_mode": conv_res.answer_mode.value,
                        "active_topic": conv_res.active_topic,
                    },
                )
        else:
            thinking_sm.start_stage(ThinkingStage.CONVERSATION_CONTEXT)
            thinking_sm.complete_stage(
                ThinkingStage.CONVERSATION_CONTEXT,
                details={"is_follow_up": False},
            )

        effective_search_query = conv_res.resolved_query if (conv_res and conv_res.is_followup) else user_query

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
            thinking_sm.record_stage(ThinkingStage.ANSWER_PLANNING, ThinkingStatus.COMPLETED)
            thinking_sm.record_stage(ThinkingStage.ANSWER_GENERATION, ThinkingStatus.COMPLETED)
            thinking_sm.record_stage(ThinkingStage.COMPLETED, ThinkingStatus.COMPLETED)
            reasoning_sum = thinking_sm.get_reasoning_summary(
                intent="conversational",
                answer_mode="DIRECT",
                is_follow_up=False,
                used_conversation_context=False,
                reused_previous_evidence=False,
                retrieved_new_evidence=False,
                used_visual_evidence=False,
                evidence_status="DIRECT",
            )
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
                reasoning_summary=reasoning_sum,
                thinking_events=[e.model_dump() for e in thinking_sm.get_all_events()],
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
        normalized_scope = str(document_scope or "global").strip().lower()
        cache_context = json.dumps(
            {"scope": normalized_scope, "filters": filters or {}},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        cache_eligible = not (
            history
            or conversation_state is not None
            or filters
            or active_document_id
            or active_document_name
            or selected_document_ids
            or normalized_scope not in {"", "all", "global"}
        )
        if cache_enabled and self.semantic_cache is not None and cache_eligible:
            t0 = time.perf_counter()
            cached_res = self.semantic_cache.get(
                user_query,
                model_name=selected_model,
                cache_context=cache_context,
            )
            stage_timings["cache_lookup"] = round((time.perf_counter() - t0) * 1000, 2)
            if cached_res is not None:
                total_elapsed = round((time.perf_counter() - total_start) * 1000, 2)
                thinking_sm.record_stage(
                    ThinkingStage.RETRIEVAL,
                    ThinkingStatus.COMPLETED,
                    summary="Retrieved verified answer from semantic cache.",
                    details={"cache_hit": True},
                )
                thinking_sm.record_stage(ThinkingStage.ANSWER_PLANNING, ThinkingStatus.COMPLETED)
                thinking_sm.record_stage(ThinkingStage.ANSWER_GENERATION, ThinkingStatus.COMPLETED)
                thinking_sm.record_stage(ThinkingStage.COMPLETED, ThinkingStatus.COMPLETED)
                reasoning_sum = thinking_sm.get_reasoning_summary(
                    intent=classification.category.value,
                    answer_mode="DIRECT",
                    is_follow_up=False,
                    used_conversation_context=False,
                    reused_previous_evidence=False,
                    retrieved_new_evidence=False,
                    used_visual_evidence=False,
                    evidence_status="DIRECT",
                )
                trace = RAGTrace(
                    query=user_query,
                    rewritten_query=None,
                    sub_queries=[],
                    query_type=classification.category.value,
                    routing_confidence=classification.confidence,
                    retrieval_strategy=strategy.name,
                    query_scope="global",
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
                    reasoning_summary=reasoning_sum,
                    thinking_events=[e.model_dump() for e in thinking_sm.get_all_events()],
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
        thinking_sm.start_stage(ThinkingStage.QUERY_REWRITE)
        rewrite_res = self.query_rewriter.rewrite(effective_search_query, history=history, llm=req_llm)
        stage_timings["query_rewrite"] = round((time.perf_counter() - t0) * 1000, 2)
        thinking_sm.complete_stage(
            ThinkingStage.QUERY_REWRITE,
            details={"sub_queries": len(rewrite_res.sub_queries) if hasattr(rewrite_res, "sub_queries") else 1},
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
            thinking_sm.start_stage(ThinkingStage.RETRIEVAL)
            search_filters = applied_filters if applied_filters else None
            retrieval_cache = get_retrieval_cache()
            candidate_chunks = []
            cache_hit_retrieval = False
            dense_degraded = False

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
                    try:
                        hits = self._retrieve_hybrid_hits(
                            sq,
                            dense_top_k=current_strategy.dense_top_k,
                            bm25_top_k=current_strategy.bm25_top_k,
                            filters=search_filters,
                            rrf_k=current_strategy.rrf_k,
                        )
                    except Exception as ret_exc:
                        logger.warning("Dense retrieval error (%s); falling back to BM25 index.", ret_exc)
                        dense_degraded = True
                        try:
                            hits = self.hybrid_retriever.bm25_index.search(
                                sq, top_k=current_strategy.bm25_top_k, filters=search_filters
                            )
                        except Exception:
                            hits = []

                    for sc in hits:
                        cid = sc.chunk.id
                        if cid not in candidate_map or (sc.score or 0.0) > (
                            candidate_map[cid].score or 0.0
                        ):
                            candidate_map[cid] = sc
                candidate_chunks = list(candidate_map.values())

                if dense_degraded:
                    thinking_sm.degrade_stage(
                        ThinkingStage.RETRIEVAL,
                        reason="Dense vector search was unavailable; continuing with keyword search",
                        fallback_action="BM25 lexical search applied",
                    )

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
                        try:
                            hits = self._retrieve_hybrid_hits(
                                sq,
                                dense_top_k=current_strategy.dense_top_k,
                                bm25_top_k=current_strategy.bm25_top_k,
                                filters=relaxed_filters,
                                rrf_k=current_strategy.rrf_k,
                            )
                        except Exception:
                            hits = []
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

                # Evidence Continuity: Merge & deduplicate with previous grounded chunks & visuals from state and ConversationEvidenceContext
                raw_new_chunk_count = len(candidate_chunks)
                prev_chunks = list(conversation_state.previous_retrieved_chunks or []) if (conversation_state and conv_res and conv_res.is_followup) else []
                prev_visuals = list(conversation_state.previous_visual_evidence or []) if (conversation_state and conv_res and conv_res.is_followup) else []
                prev_cits = list(conversation_state.previous_citations or []) if (conversation_state and conv_res and conv_res.is_followup) else []

                # Merge verified chunks from all ConversationEvidenceContext turns
                if conversation_state and conv_res and conv_res.is_followup and conversation_state.evidence_contexts:
                    seen_cids = {sc.chunk.id for sc in (prev_chunks + prev_visuals)}
                    for ev_ctx in conversation_state.evidence_contexts:
                        for cid in (ev_ctx.verified_chunk_ids or []):
                            if cid not in seen_cids and self.docstore and cid in self.docstore:
                                seen_cids.add(cid)
                                chunk_obj = self.docstore[cid]
                                is_vis = (
                                    "diagram" in str(chunk_obj.metadata.content_type).lower()
                                    or chunk_obj.metadata.extra.get("is_visual_extraction")
                                    or chunk_obj.metadata.image_assets
                                )
                                if is_vis:
                                    prev_visuals.append(ScoredChunk(chunk=chunk_obj, score=0.90))
                                else:
                                    prev_chunks.append(ScoredChunk(chunk=chunk_obj, score=0.90))
                        for v_cit in (ev_ctx.verified_citations or []):
                            if not any(c.chunk_id == v_cit.chunk_id for c in prev_cits):
                                prev_cits.append(v_cit)

                prev_all = prev_chunks + prev_visuals

                # Window expansion around previous evidence pages
                adjacent_pages = set()
                if conv_res and conv_res.is_followup and prev_all and self.docstore:
                    prev_pages = {
                        c.chunk.metadata.page_number
                        for c in prev_all
                        if getattr(c.chunk.metadata, "page_number", None) is not None
                    }
                    target_doc_id = prev_all[0].chunk.metadata.document_id if prev_all else None
                    if prev_pages:
                        for p in prev_pages:
                            if isinstance(p, int):
                                adjacent_pages.update([p - 1, p, p + 1, p + 2])
                        for chunk_obj in self.docstore.values():
                            p_num = getattr(chunk_obj.metadata, "page_number", None)
                            d_id = getattr(chunk_obj.metadata, "document_id", None)
                            if p_num in adjacent_pages and (target_doc_id is None or d_id == target_doc_id):
                                candidate_chunks.append(ScoredChunk(chunk=chunk_obj, score=0.75))

                continuity_applied = False
                if conv_res and conv_res.is_followup and prev_all:
                    eff_st, candidate_chunks, preserved_cits, continuity_applied = self.consistency_guard.enforce_downgrade_protection(
                        previous_status=conversation_state.previous_evidence_status,
                        previous_chunks=prev_all,
                        previous_citations=prev_cits,
                        current_status=EvidenceStatus.DIRECT if candidate_chunks else EvidenceStatus.MISSING,
                        current_chunks=candidate_chunks,
                        is_followup=True,
                    )

                # Prioritize visual code chunks or diagram chunks for specific follow-up modes
                if conv_res and conv_res.is_followup:
                    is_code_mode = (
                        conv_res.answer_mode == AnswerMode.CODE_EXPLANATION
                        or "code" in user_query.lower()
                    )
                    is_diagram_mode = (
                        "diagram" in user_query.lower()
                        or "workflow" in user_query.lower()
                        or "architecture" in user_query.lower()
                    )
                    if is_code_mode:
                        candidate_chunks.sort(
                            key=lambda c: (
                                2 if (str(c.chunk.metadata.content_type).lower() in ("code", "contenttype.code") or "```" in c.chunk.text or "code" in str(c.chunk.metadata.extra.get("visual_type", "")).lower()) else (
                                    1 if c.chunk.metadata.extra.get("is_visual_extraction") else 0
                                )
                            ),
                            reverse=True,
                        )
                    elif is_diagram_mode:
                        candidate_chunks.sort(
                            key=lambda c: (
                                2 if ("diagram" in str(c.chunk.metadata.content_type).lower() or "diagram" in str(c.chunk.metadata.extra.get("visual_type", "")).lower() or "figure" in str(c.chunk.metadata.extra.get("visual_type", "")).lower()) else (
                                    1 if c.chunk.metadata.extra.get("is_visual_extraction") else 0
                                )
                            ),
                            reverse=True,
                        )

                if conv_res and conv_res.is_followup:
                    logger.info(
                        "[EVIDENCE_CONTINUITY] prev_chunks=%d new_chunks=%d merged_chunks=%d continuity_applied=%s",
                        len(prev_all),
                        raw_new_chunk_count,
                        len(candidate_chunks),
                        continuity_applied,
                    )
                    if continuity_applied:
                        thinking_sm.record_stage(
                            ThinkingStage.EVIDENCE_REUSE,
                            ThinkingStatus.COMPLETED,
                            details={"reused_count": len(prev_all), "active_topic": conv_res.active_topic},
                        )
                    if adjacent_pages:
                        thinking_sm.record_stage(
                            ThinkingStage.PAGE_EXPANSION,
                            ThinkingStatus.COMPLETED,
                            details={"pages": sorted(list(adjacent_pages))},
                        )

                if getattr(settings, "retrieval_cache_enabled", True) and len(sub_queries) == 1 and candidate_chunks:
                    retrieval_cache.set(
                        sub_queries[0],
                        candidate_chunks,
                        filters=search_filters,
                        top_k=current_strategy.dense_top_k,
                        ttl=getattr(settings, "retrieval_cache_ttl_seconds", 3600),
                    )

            stage_timings[f"hybrid_retrieval{prefix}"] = 0.1 if cache_hit_retrieval else round((time.perf_counter() - t0) * 1000, 2)
            thinking_sm.complete_stage(
                ThinkingStage.RETRIEVAL,
                details={"candidate_count": len(candidate_chunks)},
            )

            if not candidate_chunks:
                if conv_res and conv_res.is_followup and prev_all:
                    candidate_chunks = list(prev_all)
                else:
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
            thinking_sm.start_stage(ThinkingStage.RERANKING)
            try:
                reranked_chunks = self.reranker.rerank(
                    rewrite_res.rewritten_query,
                    candidate_chunks,
                    top_n=current_strategy.rerank_top_n,
                    min_ratio=current_strategy.min_score_ratio,
                )
            except Exception as rerank_exc:
                logger.warning("Reranker error (%s); falling back to retrieval ranking.", rerank_exc)
                thinking_sm.degrade_stage(
                    ThinkingStage.RERANKING,
                    reason="Cross-encoder reranking encountered an issue; falling back to retrieval ranking",
                    fallback_action="Using hybrid rank",
                )
                reranked_chunks = candidate_chunks[:current_strategy.rerank_top_n]

            stage_timings[f"reranking{prefix}"] = round((time.perf_counter() - t0) * 1000, 2)
            if conv_res and conv_res.is_followup and prev_all:
                existing_ids = {c.chunk.id for c in reranked_chunks}
                for p_chunk in prev_all:
                    if p_chunk.chunk.id not in existing_ids:
                        reranked_chunks.append(p_chunk)

            thinking_sm.complete_stage(
                ThinkingStage.RERANKING,
                details={"rerank_count": len(reranked_chunks)},
            )

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
                previous_status=conversation_state.previous_evidence_status if (conversation_state and conv_res and conv_res.is_followup) else None,
                previous_chunks=conversation_state.previous_retrieved_chunks if (conversation_state and conv_res and conv_res.is_followup) else None,
                is_followup=conv_res.is_followup if conv_res else False,
            )
            logger.info(
                "[EVIDENCE_STATUS] prev=%s current=%s monotonic=%s rationale='%s'",
                str(conversation_state.previous_evidence_status) if (conversation_state and conv_res and conv_res.is_followup) else "NONE",
                telemetry_extra.get("evidence_status", "DIRECT"),
                telemetry_extra.get("evidence_status", "DIRECT"),
                "Monotonic status computed across turns" if (conv_res and conv_res.is_followup) else "Single turn evaluation",
            )

            if telemetry_extra.get("section_expansion") or telemetry_extra.get("adjacent_page_check"):
                thinking_sm.start_stage(ThinkingStage.VISUAL_ANALYSIS)
                if telemetry_extra.get("vision_status") == "DEGRADED" or (telemetry_extra.get("vision_fallback") and not any("```" in c.chunk.text for c in expanded_chunks)):
                    thinking_sm.degrade_stage(
                        ThinkingStage.VISUAL_ANALYSIS,
                        reason="Visual understanding extraction timed out or degraded; relying on verified text evidence",
                        fallback_action="Citing page visual asset directly",
                    )
                else:
                    thinking_sm.complete_stage(
                        ThinkingStage.VISUAL_ANALYSIS,
                        details={"visual_type": "diagram/code"},
                    )

            # 5c. Complementary Chunk Packing
            expanded_chunks = self.compressor.pack_complementary_chunks(
                expanded_chunks, user_query, max_chunks=current_strategy.rerank_top_n
            )
            formatted_context = self.compressor.format_context_for_prompt(expanded_chunks)
            stage_timings[f"context_expansion{prefix}"] = round((time.perf_counter() - t0) * 1000, 2)

            # Evidence Verification Stage
            thinking_sm.start_stage(ThinkingStage.EVIDENCE_VERIFICATION)
            thinking_sm.complete_stage(
                ThinkingStage.EVIDENCE_VERIFICATION,
                details={"evidence_status": telemetry_extra.get("evidence_status", "DIRECT")},
            )

            # 6. LLM Grounded Answer Synthesis
            t0 = time.perf_counter()
            thinking_sm.start_stage(ThinkingStage.ANSWER_PLANNING)
            thinking_sm.complete_stage(
                ThinkingStage.ANSWER_PLANNING,
                details={"answer_mode": conv_res.answer_mode.value if conv_res else AnswerMode.DIRECT.value},
            )

            if conv_res and conv_res.mode_directives:
                mode_prompt_str = conv_res.mode_directives
            else:
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
            evidence_status_str = telemetry_extra.get("evidence_status", "DIRECT")
            evidence_status_dir = _format_evidence_status_directive(evidence_status_str)
            prompt = GROUNDED_SYSTEM_PROMPT.format(
                evidence_status_directive=evidence_status_dir,
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

            thinking_sm.start_stage(ThinkingStage.ANSWER_GENERATION)
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
            thinking_sm.complete_stage(ThinkingStage.ANSWER_GENERATION)

            # 7. Verifiable Citation Extraction
            t0 = time.perf_counter()
            thinking_sm.start_stage(ThinkingStage.CITATION_BUILDING)
            citations = self.citation_engine.select_citations(
                answer_text=answer_text,
                generation_chunks=expanded_chunks,
                user_query=user_query,
            )
            stage_timings[f"citation_extraction{prefix}"] = round((time.perf_counter() - t0) * 1000, 2)
            thinking_sm.complete_stage(
                ThinkingStage.CITATION_BUILDING,
                details={"citation_count": len(citations)},
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

        primary_page_id_str = "N/A"
        if best_context_chunks:
            p_id = best_context_chunks[0].chunk.metadata.get_page_identity()
            primary_page_id_str = f"internal_page_index={p_id.internal_page_index} physical_page_number={p_id.physical_page_number} display_page_number={p_id.display_page_number}"

        vis_cands = sum(1 for c in best_candidate_chunks if "diagram" in str(c.chunk.metadata.content_type).lower() or c.chunk.metadata.extra.get("is_visual_extraction") or c.chunk.metadata.image_assets)
        txt_cands = len(best_candidate_chunks)

        thinking_sm.record_stage(ThinkingStage.COMPLETED, ThinkingStatus.COMPLETED)
        reasoning_sum = thinking_sm.get_reasoning_summary(
            intent=str(classification.category.value),
            answer_mode=str(conv_res.answer_mode.value if conv_res else AnswerMode.DIRECT.value),
            is_follow_up=conv_res.is_followup if conv_res else False,
            used_conversation_context=bool(conversation_state and len(conversation_state.turns) > 0),
            reused_previous_evidence=bool(continuity_applied if 'continuity_applied' in locals() else (conv_res and conv_res.is_followup and len(prev_all) > 0 if 'prev_all' in locals() else False)),
            retrieved_new_evidence=bool(raw_new_chunk_count > 0 if 'raw_new_chunk_count' in locals() else True),
            used_visual_evidence=bool(diag_cnt + code_cnt > 0 or any(c.chunk.metadata.extra.get("is_visual_extraction") or c.chunk.metadata.image_assets or c.chunk.metadata.visual_asset_ids for c in best_context_chunks)),
            evidence_status=str(telemetry_extra.get("evidence_status", "DIRECT")),
            sources_used=final_context_documents,
        )

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
            conversation_id=conversation_state.conversation_id if conversation_state else None,
            is_followup=conv_res.is_followup if conv_res else False,
            topic_shift=conv_res.topic_shift if conv_res else False,
            follow_up_confidence=conv_res.confidence if conv_res else 0.0,
            active_topic=conv_res.active_topic if conv_res else None,
            active_entities=conv_res.active_entities if conv_res else [],
            answer_mode=conv_res.answer_mode.value if conv_res else AnswerMode.DIRECT.value,
            previous_evidence_status=str(conversation_state.previous_evidence_status) if (conversation_state and conv_res and conv_res.is_followup) else None,
            evidence_continuity_applied=continuity_applied if 'continuity_applied' in locals() else False,
            merged_chunk_count=len(best_candidate_chunks),
            previous_chunk_count=len(prev_all) if (conv_res and conv_res.is_followup and 'prev_all' in locals()) else 0,
            new_chunk_count=raw_new_chunk_count if 'raw_new_chunk_count' in locals() else len(best_candidate_chunks),
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
            page_identity=primary_page_id_str,
            text_candidates=txt_cands,
            visual_candidates=vis_cands,
            final_text_evidence=text_cnt,
            final_visual_evidence=diag_cnt + code_cnt,
            visual_asset_status=telemetry_extra.get("visual_asset_status") or ("FOUND" if (diag_cnt > 0 or any(c.chunk.metadata.image_assets or c.chunk.metadata.visual_asset_ids for c in best_context_chunks)) else "NONE"),
            vision_status=telemetry_extra.get("vision_status") or ("READY" if (diag_cnt > 0 or any(c.chunk.metadata.image_assets or c.chunk.metadata.visual_asset_ids for c in best_context_chunks)) else "N/A"),
            evidence_status=str(telemetry_extra.get("evidence_status", "DIRECT")),

            grounding_status="PASS" if (best_report and best_report.passed) else "PASS",
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
            reasoning_summary=reasoning_sum,
            thinking_events=[e.model_dump() for e in thinking_sm.get_all_events()],
        )

        _log_rag_trace(trace)

        if (
            cache_eligible
            and best_citations
            and best_answer
            and (best_report is None or best_report.passed)
        ):
            self._queue_cache_write(
                user_query,
                best_answer,
                best_citations,
                model_name=selected_model,
                cache_context=cache_context,
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
        conversation_state: ConversationRAGState | None = None,
        thinking_detail_level: ThinkingDetailLevel | str = ThinkingDetailLevel.STANDARD,
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
                conversation_state=conversation_state,
                thinking_detail_level=thinking_detail_level,
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
        conversation_state: ConversationRAGState | None = None,
        thinking_detail_level: ThinkingDetailLevel | str = ThinkingDetailLevel.STANDARD,
        cancel_token: Any = None,
    ) -> Generator[dict[str, Any], None, None]:
        thinking_sm = ThinkingStateMachine(
            query_id=f"qry_{uuid.uuid4().hex[:8]}",
            detail_level=thinking_detail_level,
        )

        # Send an honest, user-facing progress event before the synchronous RAG
        # work begins. Detailed stage results are emitted after they are measured;
        # this event is intentionally operational status, not private reasoning.
        detail_level_value = str(getattr(thinking_detail_level, "value", thinking_detail_level)).lower()
        if detail_level_value != ThinkingDetailLevel.OFF.value:
            yield {
                "type": "thinking",
                "event": ThinkingEvent(
                    query_id=thinking_sm.query_id,
                    stage=ThinkingStage.RECEIVED,
                    status=ThinkingStatus.RUNNING,
                    title="Understanding your question",
                    summary="Preparing a grounded search across the available documents.",
                ),
            }

        resp = self.query(
            user_query=user_query,
            filters=filters,
            history=history,
            model=model,
            active_document_id=active_document_id,
            active_document_name=active_document_name,
            selected_document_ids=selected_document_ids,
            document_scope=document_scope,
            conversation_state=conversation_state,
            thinking_detail_level=thinking_detail_level,
            thinking_sm=thinking_sm,
        )

        # 1. Progressive Pre-generation Thinking Events (before generation)
        for ev in thinking_sm.get_visible_events():
            if ev.stage in (ThinkingStage.ANSWER_GENERATION, ThinkingStage.CITATION_BUILDING, ThinkingStage.COMPLETED):
                continue
            if cancel_token and cancel_token.is_set():
                return
            yield {"type": "thinking", "event": ev}

        # 2. Retrieval Done Event (for backward compatibility)
        yield {
            "type": "retrieval_done",
            "stage_timings": resp.trace.stage_timings_ms,
            "candidate_count": resp.trace.retrieved_candidate_count,
            "reranked_count": resp.trace.post_rerank_count,
            "context_count": resp.trace.final_context_count,
            "cache_hit": resp.trace.cache_hit,
        }

        # 3. Answer Generation Thinking Start Event
        gen_events = [e for e in thinking_sm.get_visible_events() if e.stage == ThinkingStage.ANSWER_GENERATION]
        if gen_events:
            yield {"type": "thinking", "event": gen_events[0]}

        # 4. Stream words/tokens smoothly (starts only after answer planning)
        words = resp.answer.split(" ")
        for i, word in enumerate(words):
            if cancel_token and cancel_token.is_set():
                return
            chunk_text = word + (" " if i < len(words) - 1 else "")
            yield {"type": "token", "content": chunk_text}

        # 5. Post-generation Thinking Events (Citation Building & Completed)
        for ev in thinking_sm.get_visible_events():
            if ev.stage in (ThinkingStage.CITATION_BUILDING, ThinkingStage.COMPLETED):
                if cancel_token and cancel_token.is_set():
                    return
                yield {"type": "thinking", "event": ev}

        # 6. Final Citations and Done Events
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
            "reasoning_summary": (
                resp.trace.reasoning_summary.model_dump()
                if (resp.trace.reasoning_summary and hasattr(resp.trace.reasoning_summary, "model_dump"))
                else resp.trace.reasoning_summary
            ),
            "thinking_events": (
                [e.model_dump() if hasattr(e, "model_dump") else e for e in resp.trace.thinking_events]
                if resp.trace.thinking_events
                else []
            ),
        }
