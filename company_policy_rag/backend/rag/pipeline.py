from __future__ import annotations

import copy
import json
import queue
import re
import threading
import time
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any, Callable

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
from backend.rag.multi_query import MultiQueryGenerator, decompose_multi_part
from backend.rag.query_context import QueryContext
from backend.rag.policy_reliability import (
    ClauseSelection,
    GoverningClauseSelector,
    allowed_derived_facts,
    bind_source_indices,
    enforce_deterministic_calculations,
    expand_policy_queries,
    extract_query_facts,
    format_multipart_policy_decision_context,
    format_policy_decision_context,
    validate_policy_answer,
)
from backend.rag.query_rewrite import QueryRewriter
from backend.rag.query_router import QueryRouter
from backend.rag.retry_engine import RetryEngine
from backend.rag.response_modes import (
    ResponseMode,
    get_response_mode_config,
)
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
        self._current_model = initial_model
        self._lock = threading.RLock()

    @property
    def current_model(self) -> str:
        with self._lock:
            return self._current_model

    def set_model(self, model_name: str) -> None:
        """Update active model and optionally trigger non-blocking background preload."""
        normalized = str(model_name or "").strip()
        if not normalized:
            raise ValueError("Model selection cannot be empty.")
        with self._lock:
            if normalized == self._current_model:
                return
            self._current_model = normalized

        def _bg_preload():
            try:
                preload_model(normalized)
            except Exception as e:
                logger.warning("Background preload for %s failed: %s", normalized, e)

        try:
            threading.Thread(target=_bg_preload, daemon=True).start()
        except Exception:
            pass


class _LLMProxy:
    """Locked fallback wrapper for LLM clients that cannot be copied per model."""

    def __init__(self, target_llm: Any, target_model: str, lock: threading.RLock) -> None:
        self._target_llm = target_llm
        self.model = target_model
        self._lock = lock

    def complete(self, prompt: str, **kwargs: Any) -> Any:
        with self._lock:
            old_model = getattr(self._target_llm, "model", None)
            try:
                if hasattr(self._target_llm, "model") and self.model:
                    self._target_llm.model = self.model
                return self._target_llm.complete(prompt, **kwargs)
            finally:
                if hasattr(self._target_llm, "model") and old_model is not None:
                    self._target_llm.model = old_model

    def stream_complete(self, prompt: str, **kwargs: Any) -> Any:
        with self._lock:
            old_model = getattr(self._target_llm, "model", None)
            try:
                if hasattr(self._target_llm, "model") and self.model:
                    self._target_llm.model = self.model
                # Keep the selected model for the generator's full lifetime.
                yield from self._target_llm.stream_complete(prompt, **kwargs)
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
RULE F2 — Code Formatting (MANDATORY): Reproduce EVERY code snippet inside a fenced markdown code block that opens with three backticks and the correct language tag (```python, ```typescript, ```javascript, ```bash, ```json, ```yaml, ```sql, ```html, ...) and closes with three backticks on its own line. Copy the code CHARACTER-FOR-CHARACTER from the retrieved context — preserve exact indentation, line breaks, blank lines, quotes, and symbols. Never reflow multi-line code into a paragraph, never merge lines, and never place multi-line code in inline single-backtick spans. If the language is unknown, still fence it with plain triple backticks. Put explanatory prose OUTSIDE the code fence, never inside it.
RULE 1: Use retrieved evidence as the primary source of truth.
RULE 2: Do not invent details, assumptions, or external facts not supported by the retrieved text or visual evidence.
RULE 3: If a relevant visual asset exists and visual understanding is included (e.g. under [VISUAL SOURCE N]), explicitly explain the workflow, architecture, diagram, or code shown in that visual evidence.
RULE 4: Never claim that an image or diagram is absent merely because it was not included in the first text retrieval result.
RULE 5: If a visual asset exists on a page but visual understanding extraction failed or is degraded, clearly distinguish: state that the visual exists on the page, but visual analysis is currently unavailable. Cite the source tag so the user can inspect the original image.
RULE 6: Do not fabricate or invent the contents of a visual that failed extraction.
RULE 7: For source-grounded answers, prefer language such as: "According to the workflow shown on Page X..." or "Based on Section Y..." using the human-visible printed page numbers provided in the context blocks.
RULE 8: Citations: Cite sources using [Source N] or [Visual Source N] tags for every substantive claim, code block, or diagram description.
RULE 9: When code snippets, kickoff calls, agent configurations, or implementations appear in the retrieved context (including under [Source N] or [VISUAL SOURCE N]), extract and present that code directly and faithfully, verbatim, inside a fenced ```language code block (see RULE F2). Never state that the document does not contain the code if relevant code snippets or implementations are present in the context.
RULE 10: Include only the retrieved facts needed to answer the exact question. Do not dump adjacent context, generic background, or implementation details the user did not request.
RULE 11: Lead with the direct answer. For non-trivial questions, organize the rest under short descriptive headings and use bullets or numbered steps only when they improve clarity.
RULE 12: Do not repeat the question or add a generic preamble. Keep simple factual answers concise; use a compact summary followed by supporting details for broader questions.
RULE 13: If sources disagree or the evidence is incomplete, state the uncertainty explicitly instead of blending conflicting facts.
RULE 14: Match the requested depth. By default answer in 2-4 short sentences or at most 4 compact bullets. Do not add a recap or conclusion. Give a long walkthrough or code only when the user explicitly asks for detail, steps, or code.
{evidence_status_directive}
{mode_instructions}
{refinement_directive}
RETRIEVED CONTEXT:
{context_text}

{history_text}USER QUESTION: {query}
ANSWER:"""

GENERAL_CHAT_PROMPT = """You are a helpful conversational assistant in General chat mode.
Do not search, cite, or claim to rely on the user's document repository in this mode.
Use general knowledge and the recent conversation below when it is relevant.
Lead with the answer, avoid repeating the question, and use short headings or bullets only when they improve clarity.
If the user asks for document-specific facts, explain that they should switch to Document search mode.

RESPONSE MODE:
{response_mode_instructions}

{history_text}USER QUESTION: {query}
ANSWER:"""

def _format_evidence_status_directive(status: Any) -> str:
    st_val = getattr(status, "value", str(status)).upper()
    if st_val == "PARTIAL":
        return """Evidence Status: PARTIAL
- The retrieved context does not contain every detail required for a complete answer.
- State only what the retrieved text or extracted visual explicitly supports.
- If a list, diagram, table, code block, or workflow is referenced but not extracted, say that those details are not available in the retrieved evidence.
- Never fill missing labels, steps, tools, APIs, code, or facts from model memory."""
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
- Give the shortest useful implementation outline supported by the retrieved evidence.
- Include code only when the user explicitly requests code and the document actually provides it.
- Do not substitute generic or fabricated steps, tools, APIs, or code."""

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


def _is_high_risk_query(query: str) -> bool:
    """True when an answer must be verified before the user sees it.

    Targets the failure modes where a wrong or hallucinated figure/rule is most
    costly: explicit amounts or times, entitlement/calculation questions, and
    queries that match a known policy topic profile. These answers are buffered
    and verified (with a retry budget) instead of streamed token-by-token.
    Ordinary factual/lookup questions stay on the live-streaming path.
    """
    try:
        facts = extract_query_facts(query)
    except Exception:
        return False
    return bool(
        facts.amounts
        or facts.times
        or facts.topic
        or facts.intent == "calculation_or_entitlement"
    )


_EXPLICIT_DETAIL_PATTERN = re.compile(
    r"\b(?:in detail|detailed|step[- ]by[- ]step|walk me through|deep dive|"
    r"comprehensive|thorough|exhaustive|all details|show me the code|"
    r"give me the code|source code|code example)\b",
    re.IGNORECASE,
)


def _select_answer_token_budget(
    category: QueryCategory,
    answer_mode: AnswerMode | str | None,
    query: str,
) -> int:
    """Choose a concise default budget and expand only on explicit request."""
    if category == QueryCategory.FACTUAL:
        base = int(getattr(settings, "max_new_tokens_factual", 256))
    elif category in (QueryCategory.PROCEDURAL, QueryCategory.IMPLEMENTATION, QueryCategory.CODE):
        base = int(getattr(settings, "max_new_tokens_technical", 512))
    else:
        base = int(getattr(settings, "max_new_tokens_complex", 1024))

    mode = str(getattr(answer_mode, "value", answer_mode) or "DIRECT").upper()
    expansive_modes = {"EXPAND", "DETAILED", "CODE_EXPLANATION", "STEP_BY_STEP"}
    if mode in expansive_modes or _EXPLICIT_DETAIL_PATTERN.search(query or ""):
        return base

    concise_limit = max(64, int(getattr(settings, "max_new_tokens_direct", 256)))
    return min(base, concise_limit)


def _enforce_direct_answer_length(answer: str, max_words: int = 100) -> str:
    """Keep direct answers compact when a model backend ignores token limits."""
    word_matches = list(re.finditer(r"\S+", answer))
    if len(word_matches) <= max_words:
        return answer.strip()

    prefix = answer[: word_matches[max_words - 1].end()]
    min_boundary = max(40, int(len(prefix) * 0.65))
    sentence_end = max(prefix.rfind("."), prefix.rfind("!"), prefix.rfind("?"))
    if sentence_end >= min_boundary:
        return prefix[: sentence_end + 1].strip()
    return prefix.rstrip(" ,;:-") + "…"


_DEGRADED_ANSWER_MARKERS = (
    "visual labels could not be read reliably",
    "visual analysis is currently unavailable",
    "visual understanding extraction is currently unavailable or degraded",
    "i can't list them without guessing",
    "i cannot list them without guessing",
)


def _is_degraded_or_abstention_answer(answer: str | None) -> bool:
    """Identify incomplete fallback answers that must never become durable cache hits."""
    normalized = " ".join(str(answer or "").casefold().split())
    return bool(normalized) and any(marker in normalized for marker in _DEGRADED_ANSWER_MARKERS)


def _requested_enumeration_count(query: str | None) -> int | None:
    """Read an explicit requested list size without confusing item follow-ups for lists."""
    match = re.search(
        r"\b(?P<count>\d{1,2}|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(?:\w+[ -]?){0,3}(?:tech\w*|methods?|steps?|types?|ways?|items?)\b",
        str(query or ""),
        re.IGNORECASE,
    )
    if not match:
        return None
    raw_count = match.group("count").casefold()
    number_words = {
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    return int(raw_count) if raw_count.isdigit() else number_words.get(raw_count)


def _extract_requested_numbered_list(
    query: str,
    chunks: list[ScoredChunk],
) -> list[str] | None:
    """Extract an exact 1..N list from retrieved continuation text."""
    expected = _requested_enumeration_count(query)
    if expected is None or expected < 2:
        return None

    ordered = sorted(
        chunks,
        key=lambda sc: (
            sc.chunk.metadata.document_id,
            sc.chunk.metadata.page_number or 0,
            sc.chunk.id,
        ),
    )
    combined = "\n".join(sc.chunk.text for sc in ordered)
    found: dict[int, str] = {}
    for raw_index, raw_label in re.findall(
        r"(?m)^\s*(\d{1,2})\s*[.)]\s*([^\r\n]+)",
        combined,
    ):
        index = int(raw_index)
        label = raw_label.strip().strip(" -*:.;")
        if 1 <= index <= expected and label:
            found.setdefault(index, label)

    if not all(index in found for index in range(1, expected + 1)):
        return None
    return [found[index] for index in range(1, expected + 1)]


def _enumeration_preamble(query: str, count: int) -> str:
    """Derive a grounded lead-in for a deterministically extracted list from the
    query itself — e.g. "What are the five LLM fine-tuning techniques?" ->
    "The five LLM fine-tuning techniques are:". Never fabricates a subject: if
    the enumerated noun phrase cannot be parsed, falls back to a neutral lead-in.

    Root-cause fix: the pipeline previously hardcoded the fine-tuning phrasing for
    every extracted list, so any other "list the N X" question emitted a false
    subject sentence.
    """
    phrase_match = re.search(
        r"\b(?P<phrase>(?:\d{1,2}|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(?:[\w-]+[ -]?){0,3}(?:tech\w*|methods?|steps?|types?|ways?|items?))\b",
        str(query or ""),
        re.IGNORECASE,
    )
    if phrase_match:
        phrase = re.sub(r"\s+", " ", phrase_match.group("phrase")).strip()
        return f"The {phrase} are:"
    return f"Based on the retrieved document, the {count} items are:"


def _answer_matches_requested_enumeration(query: str, answer: str | None) -> bool:
    """Reject cached list answers that contain too few or too many numbered items."""
    expected = _requested_enumeration_count(query)
    if expected is None:
        return True
    indices = [
        int(value)
        for value in re.findall(r"(?m)^\s*(\d{1,2})\s*[.)]\s*\S", str(answer or ""))
    ]
    return sorted(set(indices)) == list(range(1, expected + 1))


def _is_cacheable_grounded_answer(
    answer: str | None,
    *,
    has_citations: bool,
    verifier_passed: bool,
    evidence_sufficiency_passed: bool,
    vision_status: str | None = None,
    requires_visual_abstention: bool = False,
) -> bool:
    """Apply the final quality gate before a generated answer enters semantic cache."""
    return bool(
        answer
        and has_citations
        and verifier_passed
        and evidence_sufficiency_passed
        and not requires_visual_abstention
        and str(vision_status or "").upper() != "DEGRADED"
        and not _is_degraded_or_abstention_answer(answer)
    )


def _format_history_for_prompt(
    history: list[dict[str, Any]] | None,
    max_turns: int = 6,
    max_chars: int = 12000,
) -> str:
    if not history:
        return ""
    recent = history[-(max_turns * 2) :]
    formatted: list[tuple[str, str]] = []
    for msg in recent:
        raw_role = str(msg.get("role", "")).lower()
        if raw_role not in {"user", "assistant"}:
            continue
        role = "User" if raw_role == "user" else "Assistant"
        content = str(msg.get("content", "")).strip()
        if content:
            # History supports reference resolution and continuity, but a long
            # prior answer must not become a second, unverified RAG context.
            if raw_role == "assistant" and len(content) > 600:
                content = content[:600].rstrip() + "…"
            formatted.append((role, content))

    # Keep the newest useful turns within a predictable prompt budget. This
    # prevents old, very long answers from slowing every later request.
    kept: list[str] = []
    used = 0
    for role, content in reversed(formatted):
        line = f"{role}: {content}"
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(line) > remaining:
            line = line[:remaining].rstrip() + "…"
        kept.append(line)
        used += len(line) + 1
    kept.reverse()
    if not kept:
        return ""
    lines = ["Recent Conversation History:", *kept]
    return "\n".join(lines) + "\n\n"


def _log_rag_trace(trace: RAGTrace) -> None:
    """Output comprehensive Phase 14 structured observability telemetry trace."""
    sep = "=" * 70
    lines = [
        f"\n{sep}",
        "[RAG TRACE]",
        f"QUERY:                    {trace.query}",
        f"INTENT:                   {trace.query_type or 'factual'}",
        f"RESPONSE_MODE:            {trace.response_mode}",
        f"RETRIEVAL_TOP_K:          {trace.retrieval_top_k}",
        f"RERANK_TOP_K:             {trace.rerank_top_k}",
        f"CONTEXT_TOKENS:           {trace.context_tokens}",
        f"GENERATION_MAX_TOKENS:    {trace.generation_max_tokens}",
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
        governing_clause_selector: GoverningClauseSelector | None = None,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.query_rewriter = query_rewriter or QueryRewriter(
            enable_llm_rewrite=bool(getattr(settings, "enable_query_rewrite", False))
        )
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
        self.conversation_resolver = conversation_resolver or ConversationResolver(
            llm=self.llm if bool(getattr(settings, "enable_query_rewrite", False)) else None
        )
        self.consistency_guard = consistency_guard or ConversationConsistencyGuard()
        self.governing_clause_selector = governing_clause_selector or GoverningClauseSelector()


        raw_llm_name = getattr(self.llm, "model", None)
        if isinstance(raw_llm_name, str) and raw_llm_name.strip():
            default_llm_name = raw_llm_name.strip()
        else:
            default_llm_name = getattr(settings, "llm_model", "qwen2.5:7b")
        self.model_manager = ModelManager(initial_model=str(default_llm_name))
        self._llm_instance_cache: dict[str, Any] = {}
        self._llm_cache_lock = threading.RLock()
        self._shared_llm_lock = threading.RLock()

        if (
            self.query_rewriter.enable_llm_rewrite
            and self.query_rewriter.llm is None
            and self.llm is not None
        ):
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

    def _expand_adjacent_text_evidence(
        self,
        chunks: list[ScoredChunk],
        *,
        document_id: str,
        pages_to_inspect: list[int],
        anchor_section: str | None,
    ) -> list[ScoredChunk]:
        """Add indexed continuation-page text before paying for visual inference."""
        if not self.docstore or not document_id or not pages_to_inspect:
            return chunks

        existing_ids = {sc.chunk.id for sc in chunks}
        wanted_pages = set(pages_to_inspect)
        section_key = " ".join(str(anchor_section or "").casefold().split())
        candidates: list[Chunk] = []
        for chunk in self.docstore.values():
            meta = chunk.metadata
            if chunk.id in existing_ids or meta.document_id != document_id:
                continue
            if meta.page_number not in wanted_pages:
                continue
            extra = meta.extra or {}
            if extra.get("is_visual_extraction"):
                continue
            candidate_section = " ".join(
                str(meta.section_title or meta.section_path or "").casefold().split()
            )
            if section_key and candidate_section and candidate_section != section_key:
                continue
            if not chunk.text or not chunk.text.strip():
                continue
            candidates.append(chunk)

        candidates.sort(key=lambda chunk: (chunk.metadata.page_number or 0, chunk.id))
        additions = [
            ScoredChunk(
                chunk=chunk,
                score=0.94,
                rerank_score=0.94,
                dense_score=0.94,
                sparse_score=0.94,
            )
            for chunk in candidates
        ]
        return chunks + additions

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
        using Qwen3-VL-2B-Instruct, cache results, and pack the extracted code into context.
        """
        raw_vm = getattr(self.vision_service, "vision_model", "Qwen3-VL-2B-Instruct")
        telemetry: dict[str, Any] = {
            "section_expansion": False,
            "adjacent_page_check": False,
            "vision_fallback": False,
            "vision_model": str(raw_vm) if (isinstance(raw_vm, str) and raw_vm.strip()) else "Qwen3-VL-2B-Instruct",
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
        telemetry["missing_evidence_types"] = list(gate_res.missing_evidence_types)
        if gate_res.anchor_chunk:
            telemetry["anchor_section"] = gate_res.anchor_chunk.chunk.metadata.section_title or gate_res.anchor_chunk.chunk.metadata.section_path

        if gate_res.is_sufficient:
            return chunks, telemetry

        initial_anchor = gate_res.anchor_chunk or chunks[0]
        initial_meta = initial_anchor.chunk.metadata
        expanded_chunks = self._expand_adjacent_text_evidence(
            chunks,
            document_id=initial_meta.document_id,
            pages_to_inspect=gate_res.pages_to_inspect,
            anchor_section=initial_meta.section_title or initial_meta.section_path,
        )
        if len(expanded_chunks) > len(chunks):
            telemetry["section_expansion"] = True
            telemetry["adjacent_page_check"] = True
            telemetry["text_continuation_expansion"] = len(expanded_chunks) - len(chunks)
            gate_res = self.evidence_gate.evaluate(
                query=user_query,
                intent=intent,
                candidate_chunks=expanded_chunks,
                previous_status=previous_status,
                previous_chunks=previous_chunks,
                is_followup=is_followup,
            )
            telemetry["evidence_sufficiency_passed"] = gate_res.is_sufficient
            telemetry["evidence_status"] = gate_res.evidence_status.value
            telemetry["missing_evidence_types"] = list(gate_res.missing_evidence_types)
            if gate_res.is_sufficient:
                logger.info(
                    "Resolved incomplete evidence from %d indexed continuation chunk(s); skipping vision.",
                    len(expanded_chunks) - len(chunks),
                )
                return expanded_chunks, telemetry
            chunks = expanded_chunks

        anchor = gate_res.anchor_chunk or chunks[0]
        meta = anchor.chunk.metadata
        cue = gate_res.detected_continuation_cues[0] if gate_res.detected_continuation_cues else None
        anchor_title = meta.section_title or meta.section_path or "Section Implementation"

        def page_assets(p_num: int) -> list[Any]:
            """Read indexed assets defensively, including older manager adapters."""
            manager = self.vision_service.image_asset_manager
            exact_getter = getattr(manager, "get_page_assets_by_physical_page", None)
            if callable(exact_getter):
                try:
                    result = exact_getter(meta.document_id, p_num)
                    if isinstance(result, list):
                        return result
                except Exception:
                    pass
            legacy_getter = getattr(manager, "get_page_assets", None)
            if callable(legacy_getter):
                try:
                    result = legacy_getter(meta.document_id, p_num)
                    return result if isinstance(result, list) else []
                except Exception:
                    pass
            return []

        # Missing code text alone is not proof that an image must be scanned.
        # Only pay the VLM cost when retrieval exposes a real visual signal or
        # the user explicitly asks about visual content. This keeps generic
        # questions such as "how can I build a voice RAG agent?" on text RAG.
        explicit_visual_request = bool(
            re.search(
                r"\b(?:image|screenshot|diagram|figure|flowchart|chart|table|visual|scan(?:ned)?)\b",
                user_query,
                re.IGNORECASE,
            )
        )
        textual_continuation_signal = any(
            re.search(
                r"\b(?:here(?:'s| is) how it(?:'s| is) done|shown below|depicted below|illustrated below|"
                r"see (?:the )?(?:next|following) page|continued on (?:the )?next page)\b",
                scored.chunk.text,
                re.IGNORECASE,
            )
            for scored in chunks
        )
        indexed_visual_available = gate_res.visual_asset_available or any(
            page_assets(p_num) for p_num in gate_res.pages_to_inspect
        )
        explicit_visual_code_request = bool(
            re.search(
                r"\b(?:extract|transcribe|read|explain|show)\b.{0,40}"
                r"\b(?:code|snippet)\b.{0,40}\b(?:image|screenshot|figure|page)\b|"
                r"\b(?:code|snippet)\b.{0,40}\b(?:image|screenshot|figure|page)\b",
                user_query,
                re.IGNORECASE,
            )
        )
        should_run_vision = bool(
            gate_res.detected_continuation_cues
            or textual_continuation_signal
            or explicit_visual_request
            or (indexed_visual_available and explicit_visual_code_request)
        )
        if not should_run_vision:
            telemetry["vision_status"] = "SKIPPED_NO_VISUAL_SIGNAL"
            telemetry["vision_cache_status"] = "SKIPPED"
            logger.info(
                "Skipping lazy vision for query '%s': evidence is incomplete, "
                "but no visual asset, continuation cue, or explicit visual request exists.",
                user_query,
            )
            return chunks, telemetry

        # 2. Evidence is insufficient and visual evidence is plausible -> inspect pages.
        logger.info(
            "Evidence sufficiency check failed for query '%s' (intent=%s). Inspecting pages %s",
            user_query,
            intent,
            gate_res.pages_to_inspect,
        )
        telemetry["section_expansion"] = True
        telemetry["adjacent_page_check"] = True
        telemetry["vision_fallback"] = True

        resolved_path = self._resolve_document_file_path(meta)
        if not resolved_path:
            telemetry["vision_status"] = "SKIPPED_DOCUMENT_UNAVAILABLE"
            logger.warning("Could not find physical PDF file on disk for chunk %s (%s)", anchor.chunk.id, meta.source_file)
            return chunks, telemetry

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
            p_assets = page_assets(p)
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

        # Inspect the concrete service type so dynamically-created mock
        # attributes do not look like a configured readiness hook.
        query_time_available = getattr(type(self.vision_service), "is_query_time_available", None)
        can_run_live_vision = True
        if callable(query_time_available):
            try:
                can_run_live_vision, unavailable_reason = query_time_available(self.vision_service)
            except Exception as exc:
                can_run_live_vision = False
                unavailable_reason = str(exc)
            if not can_run_live_vision:
                telemetry["vision_status"] = "DEGRADED"
                telemetry["vision_cache_status"] = "SKIPPED"
                telemetry["vision_unavailable_reason"] = unavailable_reason
                logger.warning("[VISION] Skipping live query-time vision: %s", unavailable_reason)

        t_vision_start = time.perf_counter()
        vision_budget = max(
            1.0,
            float(getattr(settings, "vision_query_budget_seconds", 40.0)),
        )
        max_vision_pages = max(
            1,
            int(getattr(settings, "vision_query_max_pages", 2)),
        )
        processed_page_keys: set[int] = set()
        req_vis_type = (
            VisualContentType.CODE_SCREENSHOT
            if is_code_intent
            else (VisualContentType.DIAGRAM_ARCHITECTURE if is_diagram_intent else None)
        )

        # Inspect only the best-ranked pages and stop on the first success.
        for p_num in target_pages[:max_vision_pages] if can_run_live_vision else []:
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
            assets_on_page = page_assets(p_num)
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

            timeout_for_page = min(
                remaining_budget,
                max(1.0, float(getattr(settings, "vision_request_timeout", 30.0))),
            )

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
            for p_num in target_pages[:max_vision_pages]:
                assets_on_page = page_assets(p_num)
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
                    telemetry["evidence_sufficiency_passed"] = False
                    if "referenced_visual_content" in gate_res.missing_evidence_types:
                        telemetry["requires_visual_abstention"] = True
                    return new_scored_fallback + chunks, telemetry

            telemetry["vision_status"] = "DEGRADED"
            telemetry["evidence_sufficiency_passed"] = False
            if "referenced_visual_content" in gate_res.missing_evidence_types:
                telemetry["requires_visual_abstention"] = True
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

        with self._llm_cache_lock:
            if selected_model not in self._llm_instance_cache:
                llm_model_attr = getattr(self.llm, "model", None)
                if llm_model_attr == selected_model or not isinstance(llm_model_attr, str):
                    self._llm_instance_cache[selected_model] = self.llm
                else:
                    try:
                        isolated_llm = copy.copy(self.llm)
                        isolated_llm.model = selected_model
                    except Exception:
                        isolated_llm = _LLMProxy(
                            self.llm,
                            selected_model,
                            self._shared_llm_lock,
                        )
                    self._llm_instance_cache[selected_model] = isolated_llm

            return self._llm_instance_cache[selected_model], selected_model

    def _rerank_for_parts(
        self,
        parts: list[str],
        candidates: list[ScoredChunk],
        top_n: int,
        min_ratio: float,
    ) -> list[ScoredChunk]:
        """
        Rerank once per question part and interleave the winners.

        The cross-encoder only scores a bounded slice of the candidate pool and
        ranks it against a single query string. For a message asking several
        things that silently drops every part after the first, because those
        chunks rank low against the combined query or never get scored at all.
        Giving each part its own pass and round-robin merging the results keeps
        evidence for every part in the final context.
        """
        if not candidates:
            return []
        if len(parts) < 2:
            return self.reranker.rerank(parts[0], candidates, top_n=top_n, min_ratio=min_ratio)

        # Each part must be able to win seats, but no part may crowd out the rest.
        per_part_quota = max(2, top_n // len(parts))
        ranked_per_part: list[list[ScoredChunk]] = []
        for part in parts:
            try:
                ranked_per_part.append(
                    self.reranker.rerank(part, candidates, top_n=per_part_quota, min_ratio=min_ratio)
                )
            except Exception as exc:
                logger.warning("Per-part rerank failed for %r: %s", part, exc)

        merged: list[ScoredChunk] = []
        seen: set[str] = set()
        for tier in range(per_part_quota):
            for ranked in ranked_per_part:
                if tier >= len(ranked):
                    continue
                chunk = ranked[tier]
                if chunk.chunk.id not in seen:
                    seen.add(chunk.chunk.id)
                    merged.append(chunk)
        if not merged:
            return self.reranker.rerank(parts[0], candidates, top_n=top_n, min_ratio=min_ratio)
        return merged[:top_n]

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

    @staticmethod
    def _split_control_filters(
        filters: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, str]:
        """Separate UI routing controls from metadata sent to retrievers."""
        metadata_filters = dict(filters or {})
        raw_mode = metadata_filters.pop("chat_mode", metadata_filters.pop("mode", "documents"))
        chat_mode = str(raw_mode or "documents").strip().lower()
        if chat_mode not in {"general", "documents"}:
            chat_mode = "documents"
        return (metadata_filters or None), chat_mode

    def _resolve_page_number_filter(
        self,
        requested_page: int,
        *,
        active_document_id: str | None,
        active_document_name: str | None,
        allowed_document_ids: list[str] | None = None,
    ) -> int | list[int] | None:
        """Map a printed page reference to the physical PDF sheet(s) in the index."""
        if not self.docstore:
            return requested_page

        matched_physical_pages: set[int] = set()
        for chunk in self.docstore.values():
            meta = chunk.metadata
            if active_document_id and meta.document_id != active_document_id:
                continue
            if active_document_name and meta.source_file != active_document_name:
                continue
            if allowed_document_ids and meta.document_id not in allowed_document_ids:
                continue
            page_id = meta.get_page_identity()
            if page_id.matches_display(requested_page) or (
                page_id.display_page_number is None
                and page_id.matches_physical_page(requested_page)
            ):
                matched_physical_pages.add(page_id.physical_page_number)

        if not matched_physical_pages:
            return None
        resolved = sorted(matched_physical_pages)
        return resolved[0] if len(resolved) == 1 else resolved

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
        response_mode: ResponseMode = "standard",
        thinking_detail_level: ThinkingDetailLevel | str = ThinkingDetailLevel.STANDARD,
        thinking_sm: ThinkingStateMachine | None = None,
        stream_callback: Callable[[str], None] | None = None,
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
            response_mode=response_mode,
            thinking_detail_level=thinking_detail_level,
            thinking_sm=thinking_sm,
            stream_callback=stream_callback,
        )

    def _stage_scope_and_rewrite(self, ctx: QueryContext) -> None:
        """Resolve document scope, rewrite the query, and infer metadata filters onto ctx."""
        # 1. Scope Resolution
        known_docs: dict[str, str] = {}
        if self.docstore:
            for c in self.docstore.values():
                if c.metadata and c.metadata.document_id and c.metadata.source_file:
                    known_docs[c.metadata.document_id] = c.metadata.source_file
        ctx.known_docs = known_docs

        if ctx.filters:
            if "document_id" in ctx.filters and not ctx.active_document_id:
                if isinstance(ctx.filters["document_id"], list):
                    ctx.selected_document_ids = ctx.filters["document_id"]
                else:
                    ctx.active_document_id = str(ctx.filters["document_id"])
            if "source_file" in ctx.filters and not ctx.active_document_name:
                ctx.active_document_name = str(ctx.filters["source_file"])

        scope_decision = self.scope_resolver.resolve_scope(
            query=ctx.user_query,
            active_document_id=ctx.active_document_id,
            active_document_name=ctx.active_document_name,
            selected_document_ids=ctx.selected_document_ids,
            explicit_scope=ctx.document_scope,
            known_documents=known_docs,
        )
        ctx.scope_decision = scope_decision

        # 1a. Query Rewrite
        t0 = time.perf_counter()
        ctx.thinking_sm.start_stage(ThinkingStage.QUERY_REWRITE)
        rewrite_res = self.query_rewriter.rewrite(
            ctx.effective_search_query,
            # ConversationResolver has already produced a standalone query.
            # Passing history here would invoke a second LLM rewrite for the
            # same turn. Retain the fallback path for callers without state.
            history=ctx.history if (ctx.is_history_followup and ctx.conv_res is None) else None,
            llm=ctx.req_llm,
        )
        ctx.rewrite_res = rewrite_res
        ctx.stage_timings["query_rewrite"] = round((time.perf_counter() - t0) * 1000, 2)
        ctx.thinking_sm.complete_stage(
            ThinkingStage.QUERY_REWRITE,
            details={"sub_queries": len(rewrite_res.sub_queries) if hasattr(rewrite_res, "sub_queries") else 1},
        )

        # 1b. Query-Time Metadata Filter Inference
        inferred_filters: dict[str, Any] = {}
        applied_filters: dict[str, Any] = {}
        ctx.filter_relaxed = False
        enable_filtering = getattr(settings, "enable_query_metadata_filtering", True)
        if enable_filtering and self.filter_inferer is not None:
            t0 = time.perf_counter()
            inferred_filters = self.filter_inferer.infer_filters(
                query=ctx.user_query,
                # Metadata inferred from old turns can silently hide the right
                # facts after a topic change. Inherit it only for true follow-ups.
                history=ctx.history if ctx.is_history_followup else None,
                explicit_filters=ctx.filters,
            )
            ctx.stage_timings["filter_inference"] = round((time.perf_counter() - t0) * 1000, 2)
            if inferred_filters:
                applied_filters = {**inferred_filters}
        elif ctx.filters:
            applied_filters = {**ctx.filters}

        if scope_decision.scope == DocumentRetrievalScope.CURRENT_DOCUMENT:
            if scope_decision.active_document_id:
                applied_filters["document_id"] = scope_decision.active_document_id
            elif scope_decision.active_document_name:
                applied_filters["source_file"] = scope_decision.active_document_name
            if scope_decision.page_number is not None:
                resolved_page_filter = self._resolve_page_number_filter(
                    scope_decision.page_number,
                    active_document_id=scope_decision.active_document_id,
                    active_document_name=scope_decision.active_document_name,
                    allowed_document_ids=scope_decision.allowed_document_ids,
                )
                # User references are printed/display pages. Resolve them to the
                # physical sheet stored in Chroma/BM25; an unresolved page stays
                # impossible rather than relaxing to unrelated document pages.
                applied_filters["page_number"] = (
                    resolved_page_filter
                    if resolved_page_filter is not None
                    else 0
                )
            if scope_decision.section_number is not None:
                applied_filters["section_number"] = scope_decision.section_number
        elif scope_decision.scope == DocumentRetrievalScope.SELECTED_DOCUMENTS:
            if scope_decision.allowed_document_ids:
                if len(scope_decision.allowed_document_ids) == 1:
                    applied_filters["document_id"] = scope_decision.allowed_document_ids[0]
                else:
                    applied_filters["document_id"] = scope_decision.allowed_document_ids

        ctx.inferred_filters = inferred_filters
        ctx.applied_filters = applied_filters

    def _stage_plan(self, ctx: QueryContext) -> None:
        """Decompose multi-part questions; decide fast-path, retry budget, and strategy."""
        # A message that asks several things needs one retrieval per part, so it
        # can never take the single-shot fast path.
        ctx.question_parts = decompose_multi_part(ctx.user_query)

        # Fast path check for high-confidence factual questions
        ctx.is_fast_path = (
            ctx.response_mode != "detailed"
            and ctx.classification.category == QueryCategory.FACTUAL
            and ctx.classification.confidence >= 0.85
            and not (ctx.history and len(ctx.history) > 0 and self.query_rewriter._is_followup_query(ctx.user_query))
            and not ctx.scope_decision.is_structural_query
            and not ctx.question_parts
        )

        ctx.enable_verification = getattr(settings, "enable_answer_verification", True)

        # High-risk policy/numeric answers are buffered and verified before the
        # user sees them, so a failed check can still retry. Low-risk answers on
        # the streaming path emit live and therefore cannot be replaced by a
        # later retry (never stream multiple competing answers for one request).
        ctx.is_high_risk = _is_high_risk_query(ctx.user_query)
        streaming = ctx.stream_callback is not None
        ctx.stream_live = streaming and not ctx.is_high_risk
        retry_budget = self.retry_engine.max_retries if self.retry_engine else 2
        if ctx.is_fast_path or ctx.stream_live:
            ctx.max_retries = 0
        else:
            # Non-streaming requests and buffered high-risk streaming requests
            # both keep the full retry budget.
            ctx.max_retries = retry_budget

        ctx.current_strategy = ctx.response_mode_config.apply_to(ctx.strategy)
        if ctx.is_fast_path:
            ctx.current_strategy.enable_multi_query = False

    def _stage_retrieve(self, ctx: QueryContext, prefix: str) -> None:
        """Build sub-queries and run one attempt's hybrid retrieval onto ``ctx``.

        Expands the query (structural / multi-part / policy sub-queries), runs
        cached-or-hybrid retrieval with a BM25 fallback, relaxes only soft
        filters on an empty result (hard document scope is never dropped),
        enforces document-scope validity, trims the candidate pool, and applies
        follow-up code/diagram prioritisation. Writes ``sub_queries``,
        ``candidate_chunks``, ``applied_filters``, ``filter_relaxed``,
        ``cross_document_count`` and ``raw_new_chunk_count`` back onto ``ctx``.
        """
        thinking_sm = ctx.thinking_sm
        current_strategy = ctx.current_strategy
        scope_decision = ctx.scope_decision
        rewrite_res = ctx.rewrite_res
        conv_res = ctx.conv_res

        # 2. Multi-Query Generation & Structural Query Expansion
        t0 = time.perf_counter()
        if not ctx.is_fast_path and scope_decision.is_structural_query:
            sub_queries = scope_decision.structural_subqueries
        elif not ctx.is_fast_path and (current_strategy.enable_multi_query or rewrite_res.is_comprehensive_list):
            sub_queries = self.multi_query_gen.generate_subqueries(rewrite_res.rewritten_query)
        else:
            sub_queries = [rewrite_res.rewritten_query]
        # Every part of a multi-part message retrieves on its own terms, even
        # when the strategy would otherwise run a single query.
        for part in ctx.question_parts:
            if part not in sub_queries:
                sub_queries.append(part)
        if not ctx.is_fast_path:
            for policy_query in expand_policy_queries(ctx.user_query):
                if policy_query not in sub_queries:
                    sub_queries.append(policy_query)
            sub_queries = sub_queries[:8]
        ctx.sub_queries = sub_queries
        ctx.stage_timings[f"multi_query{prefix}"] = round((time.perf_counter() - t0) * 1000, 2)

        # 3. Hybrid Search (Dense + BM25 with RRF)
        t0 = time.perf_counter()
        thinking_sm.start_stage(ThinkingStage.RETRIEVAL)
        search_filters = ctx.applied_filters if ctx.applied_filters else None
        retrieval_cache = get_retrieval_cache()
        candidate_chunks: list[ScoredChunk] = []
        cache_hit_retrieval = False
        dense_degraded = False

        if (
            getattr(settings, "retrieval_cache_enabled", True)
            and len(sub_queries) == 1
            and not (conv_res and conv_res.is_followup)
        ):
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

            # Filter relaxation fallback if 0 results. Document, page, and
            # section constraints are hard user scope and must never be
            # dropped, otherwise an invalid page can return unrelated facts.
            if not candidate_chunks and search_filters and getattr(settings, "enable_filter_fallback_relaxation", True):
                hard_filter_keys = {"document_id", "source_file", "page_number", "section_number"}
                relaxed_filters = {
                    key: value
                    for key, value in search_filters.items()
                    if key in hard_filter_keys
                }
                relaxed_filters = relaxed_filters or None
                ctx.filter_relaxed = relaxed_filters != search_filters
                ctx.applied_filters = relaxed_filters or {}
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
                        ctx.cross_document_count += 1
                candidate_chunks = valid_cands
            elif scope_decision.scope == DocumentRetrievalScope.SELECTED_DOCUMENTS and scope_decision.allowed_document_ids:
                valid_cands = []
                for sc in candidate_chunks:
                    d_id = sc.chunk.metadata.document_id
                    if d_id in scope_decision.allowed_document_ids:
                        valid_cands.append(sc)
                    else:
                        ctx.cross_document_count += 1
                candidate_chunks = valid_cands

            candidate_chunks.sort(key=lambda x: x.score or 0.0, reverse=True)
            candidate_pool_limit = max(len(candidate_chunks), current_strategy.rerank_top_n * 3, 15)
            candidate_chunks = candidate_chunks[:candidate_pool_limit]

            # Prioritize visual code chunks or diagram chunks for specific follow-up modes
            if conv_res and conv_res.is_followup:
                is_code_mode = (
                    conv_res.answer_mode == AnswerMode.CODE_EXPLANATION
                    or "code" in ctx.user_query.lower()
                )
                is_diagram_mode = (
                    "diagram" in ctx.user_query.lower()
                    or "workflow" in ctx.user_query.lower()
                    or "architecture" in ctx.user_query.lower()
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

            if (
                getattr(settings, "retrieval_cache_enabled", True)
                and len(sub_queries) == 1
                and candidate_chunks
                and not (conv_res and conv_res.is_followup)
            ):
                retrieval_cache.set(
                    sub_queries[0],
                    candidate_chunks,
                    filters=search_filters,
                    top_k=current_strategy.dense_top_k,
                    ttl=getattr(settings, "retrieval_cache_ttl_seconds", 3600),
                )

        ctx.raw_new_chunk_count = len(candidate_chunks)
        ctx.candidate_chunks = candidate_chunks
        ctx.stage_timings[f"hybrid_retrieval{prefix}"] = 0.1 if cache_hit_retrieval else round((time.perf_counter() - t0) * 1000, 2)
        thinking_sm.complete_stage(
            ThinkingStage.RETRIEVAL,
            details={"candidate_count": len(candidate_chunks)},
        )

    def _stage_rerank_and_context(self, ctx: QueryContext, prefix: str) -> None:
        """Rerank, select the governing clause, expand parents, run the vision
        fallback, pack to the token budget, and format the final prompt context.

        Reads ``ctx.candidate_chunks`` and writes ``reranked_chunks``,
        ``expanded_chunks``, ``policy_selection``, ``telemetry_extra``,
        ``context_tokens`` and ``formatted_context`` back onto ``ctx``.
        """
        thinking_sm = ctx.thinking_sm
        current_strategy = ctx.current_strategy
        candidate_chunks = ctx.candidate_chunks
        user_query = ctx.user_query
        response_mode_config = ctx.response_mode_config
        conv_res = ctx.conv_res

        # 4. Cross-Encoder Reranking
        t0 = time.perf_counter()
        thinking_sm.start_stage(ThinkingStage.RERANKING)
        try:
            reranked_chunks = self._rerank_for_parts(
                ctx.question_parts or [ctx.rewrite_res.rewritten_query],
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

        ctx.stage_timings[f"reranking{prefix}"] = round((time.perf_counter() - t0) * 1000, 2)

        thinking_sm.complete_stage(
            ThinkingStage.RERANKING,
            details={"rerank_count": len(reranked_chunks)},
        )

        # Cross-encoders rank semantic relatedness. Policy QA additionally
        # needs a deterministic decision about which clause directly
        # governs the user's exact actors, conditions, and thresholds.
        t0 = time.perf_counter()
        policy_selection = self.governing_clause_selector.select(
            user_query,
            reranked_chunks,
            candidate_pool=candidate_chunks,
        )
        selected_context = self.governing_clause_selector.order_for_context(
            policy_selection,
            max_chunks=max(current_strategy.rerank_top_n, 5),
        )
        if selected_context:
            reranked_chunks = selected_context
        ctx.stage_timings[f"governing_clause_selection{prefix}"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )

        # 5. Parent Context Expansion
        t0 = time.perf_counter()
        try:
            expanded_chunks = self.compressor.expand_to_parents(
                reranked_chunks,
                self.docstore,
                enable_expansion=False if ctx.is_fast_path else current_strategy.enable_parent_expansion,
            )
        except TypeError:
            # Preserve compatibility with lightweight/custom compressors
            # that implement the original two-argument protocol.
            expanded_chunks = self.compressor.expand_to_parents(
                reranked_chunks,
                self.docstore,
            )

        # 5b. Evidence Sufficiency Gate & Cross-Page Vision Fallback (Phases 3, 4, 8)
        expanded_chunks, telemetry_extra = self._apply_cross_page_vision_fallback_if_needed(
            expanded_chunks,
            user_query=user_query,
            intent=ctx.classification.category,
            previous_status=None,
            previous_chunks=None,
            is_followup=conv_res.is_followup if conv_res else False,
        )
        logger.info(
            "[EVIDENCE_STATUS] prev=%s current=%s monotonic=%s rationale='%s'",
            str(ctx.conversation_state.previous_evidence_status) if (ctx.conversation_state and conv_res and conv_res.is_followup) else "NONE",
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
        if hasattr(self.compressor, "pack_complementary_chunks"):
            expanded_chunks = self.compressor.pack_complementary_chunks(
                expanded_chunks, user_query, max_chunks=current_strategy.rerank_top_n
            )
        else:
            expanded_chunks = expanded_chunks[: current_strategy.rerank_top_n]
        if hasattr(self.compressor, "pack_to_token_budget"):
            expanded_chunks, context_tokens = self.compressor.pack_to_token_budget(
                expanded_chunks,
                response_mode_config.max_context_tokens,
            )
        else:
            packed_chunks: list[ScoredChunk] = []
            context_tokens = 0
            for sc in expanded_chunks:
                estimated = max(1, int(len(sc.chunk.text.split()) * 1.3) + 24)
                if packed_chunks and context_tokens + estimated > response_mode_config.max_context_tokens:
                    break
                packed_chunks.append(sc)
                context_tokens += estimated
            expanded_chunks = packed_chunks

        # Packing may reorder or remove evidence. Recompute the structured
        # rule view and bind each rule to the final Source N indices.
        policy_selection = self.governing_clause_selector.select(
            user_query,
            expanded_chunks,
        )
        bind_source_indices(policy_selection, expanded_chunks)
        try:
            formatted_context = self.compressor.format_context_for_prompt(
                expanded_chunks,
                max_token_budget=response_mode_config.max_context_tokens,
            )
        except TypeError:
            formatted_context = self.compressor.format_context_for_prompt(expanded_chunks)

        # For a multi-part question, give the generator each part's OWN governing
        # rule in a separate labeled block so rules, conditions, and thresholds
        # from unrelated parts are not merged into one blended answer. The single
        # global `policy_selection` above is kept for validation, deterministic
        # enforcement, and trace.
        if len(ctx.question_parts) >= 2:
            part_selections: list[tuple[str, ClauseSelection]] = []
            for part in ctx.question_parts:
                part_sel = self.governing_clause_selector.select(part, expanded_chunks)
                bind_source_indices(part_sel, expanded_chunks)
                part_selections.append((part, part_sel))
            policy_block = format_multipart_policy_decision_context(part_selections)
        else:
            policy_block = format_policy_decision_context(policy_selection)
        formatted_context = f"{policy_block}\n\n{formatted_context}"
        ctx.stage_timings[f"context_expansion{prefix}"] = round((time.perf_counter() - t0) * 1000, 2)

        # Evidence Verification Stage
        thinking_sm.start_stage(ThinkingStage.EVIDENCE_VERIFICATION)
        thinking_sm.complete_stage(
            ThinkingStage.EVIDENCE_VERIFICATION,
            details={"evidence_status": telemetry_extra.get("evidence_status", "DIRECT")},
        )

        ctx.reranked_chunks = reranked_chunks
        ctx.expanded_chunks = expanded_chunks
        ctx.policy_selection = policy_selection
        ctx.telemetry_extra = telemetry_extra
        ctx.context_tokens = context_tokens
        ctx.formatted_context = formatted_context

    def _stage_generate(self, ctx: QueryContext, prefix: str) -> None:
        """Build the grounded prompt and synthesize the answer for one attempt.

        Handles deterministic short-circuits (exact enumerated lists, visual
        abstention), streamed vs buffered LLM synthesis (gated by
        ``ctx.stream_live``), and deterministic-calculation enforcement. Writes
        ``ctx.answer_text``.
        """
        thinking_sm = ctx.thinking_sm
        conv_res = ctx.conv_res
        response_mode_config = ctx.response_mode_config
        current_strategy = ctx.current_strategy
        user_query = ctx.user_query
        expanded_chunks = ctx.expanded_chunks
        policy_selection = ctx.policy_selection
        telemetry_extra = ctx.telemetry_extra
        req_llm = ctx.req_llm
        stream_callback = ctx.stream_callback

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
                if ctx.fidelity_mode == "exact"
                else EXPLAIN_MODE_INSTRUCTIONS
                if ctx.fidelity_mode == "explain"
                else IMPLEMENT_MODE_INSTRUCTIONS
                if ctx.fidelity_mode == "implement"
                else ""
            )
        mode_prompt_str = (
            f"{mode_prompt_str}\n\nREQUESTED RESPONSE DEPTH:\n"
            f"{response_mode_config.prompt_instructions}"
        ).strip()

        # The DIRECT mode directive asks for one short answer, which silently
        # drops the later parts of a multi-part message. State the parts
        # explicitly and require one answer per part.
        if ctx.question_parts:
            numbered_parts = "\n".join(
                f"  {idx}. {part}" for idx, part in enumerate(ctx.question_parts, start=1)
            )
            mode_prompt_str = (
                f"{mode_prompt_str}\n\nMULTI-PART QUESTION — the user asked "
                f"{len(ctx.question_parts)} distinct things:\n{numbered_parts}\n"
                "- Answer EVERY part, in the order asked, under its own heading or bullet.\n"
                "- This overrides any brevity instruction above: brevity applies per part, "
                "never as a reason to omit a part.\n"
                "- If the context supports only some parts, answer those and state "
                "explicitly which parts the documents do not cover."
            ).strip()

        # Prevent generation context leakage: only inject history if it's a true follow-up.
        effective_history = ctx.history if ctx.is_history_followup else None
        history_text = _format_history_for_prompt(effective_history)
        refinement_str = f"\nRefinement Instructions:\n{ctx.prompt_refinement}\n" if ctx.prompt_refinement else ""
        evidence_status_str = telemetry_extra.get("evidence_status", "DIRECT")
        evidence_status_dir = _format_evidence_status_directive(evidence_status_str)
        prompt = GROUNDED_SYSTEM_PROMPT.format(
            evidence_status_directive=evidence_status_dir,
            mode_instructions=mode_prompt_str,
            refinement_directive=refinement_str,
            context_text=ctx.formatted_context,
            history_text=history_text,
            query=user_query,
        )

        max_tokens = response_mode_config.max_output_tokens

        thinking_sm.start_stage(ThinkingStage.ANSWER_GENERATION)
        exact_numbered_list = _extract_requested_numbered_list(user_query, expanded_chunks)
        if exact_numbered_list:
            answer_text = _enumeration_preamble(user_query, len(exact_numbered_list)) + "\n\n" + "\n".join(
                f"{index}. {label}"
                for index, label in enumerate(exact_numbered_list, start=1)
            )
            if ctx.stream_live:
                stream_callback(answer_text)
        elif telemetry_extra.get("requires_visual_abstention"):
            page_label = "the cited page"
            if expanded_chunks:
                identity = expanded_chunks[0].chunk.metadata.get_page_identity()
                page_label = f"Page {identity.page_label}"
            answer_text = (
                f"The retrieved text says the requested details are shown in a visual on {page_label}, "
                "but the visual labels could not be read reliably. I can’t list them without guessing."
            )
            if ctx.stream_live:
                stream_callback(answer_text)
        elif req_llm is not None:
            try:
                deterministic_policy_answer = bool(
                    policy_selection.calculations or policy_selection.missing_inputs
                )
                if (
                    ctx.stream_live
                    and hasattr(req_llm, "stream_complete")
                    and not deterministic_policy_answer
                ):
                    try:
                        completion_stream = req_llm.stream_complete(
                            prompt,
                            temperature=current_strategy.temperature,
                            max_new_tokens=max_tokens,
                        )
                    except TypeError:
                        completion_stream = req_llm.stream_complete(prompt)
                    answer_parts: list[str] = []
                    for part in completion_stream:
                        delta = getattr(part, "delta", None)
                        if delta is None:
                            delta = getattr(part, "text", None)
                        if delta is None:
                            delta = str(part)
                        delta = str(delta)
                        if not delta:
                            continue
                        answer_parts.append(delta)
                        stream_callback(delta)
                    raw_answer = "".join(answer_parts).strip()
                else:
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

        answer_text = enforce_deterministic_calculations(answer_text, policy_selection)
        if (
            ctx.stream_live
            and (policy_selection.calculations or policy_selection.missing_inputs)
        ):
            stream_callback(answer_text)

        ctx.stage_timings[f"llm_synthesis{prefix}"] = round((time.perf_counter() - t0) * 1000, 2)
        thinking_sm.complete_stage(ThinkingStage.ANSWER_GENERATION)
        ctx.answer_text = answer_text

    def _stage_classify_and_resolve(self, ctx: QueryContext) -> None:
        """Classify the query, resolve conversational context, set follow-up flags on ctx."""
        # 0a. Query Classification & Intent Selection
        t0 = time.perf_counter()
        ctx.thinking_sm.start_stage(ThinkingStage.QUERY_ANALYSIS)
        classification = self.query_router.classify(ctx.user_query, history=ctx.history)
        ctx.classification = classification
        ctx.strategy = classification.strategy
        ctx.fidelity_mode = _detect_fidelity_mode(ctx.user_query)
        ctx.stage_timings["query_routing"] = round((time.perf_counter() - t0) * 1000, 2)
        ctx.thinking_sm.complete_stage(
            ThinkingStage.QUERY_ANALYSIS,
            details={"intent": classification.category.value, "confidence": classification.confidence},
        )

        # 0a-2. Dynamic Conversational Query Resolution & Observability
        conv_res: ConversationResolutionResult | None = None
        if ctx.conversation_state is not None and ctx.chat_mode != "general":
            ctx.thinking_sm.start_stage(ThinkingStage.CONVERSATION_CONTEXT)
            conv_res = self.conversation_resolver.resolve(
                query=ctx.user_query,
                state=ctx.conversation_state,
                intent=classification.category,
            )
            logger.info(
                "[CONVERSATION] session_id=%s turn_count=%d is_followup=%s topic_shift=%s topic='%s' entities=%s",
                ctx.conversation_state.conversation_id,
                len(ctx.conversation_state.turns) + 1,
                conv_res.is_followup,
                conv_res.topic_shift,
                conv_res.active_topic or "",
                conv_res.active_entities or [],
            )
            logger.info(
                "[QUERY_RESOLUTION] query='%s' resolved='%s' confidence=%.2f cues='%s'",
                ctx.user_query,
                conv_res.resolved_query,
                conv_res.confidence,
                conv_res.reason,
            )
            logger.info(
                "[ANSWER_MODE] mode=%s directives='%s'",
                conv_res.answer_mode.value,
                conv_res.mode_directives.replace("\n", " "),
            )
            ctx.thinking_sm.complete_stage(
                ThinkingStage.CONVERSATION_CONTEXT,
                details={
                    "is_follow_up": conv_res.is_followup,
                    "active_topic": conv_res.active_topic,
                    "active_entities": conv_res.active_entities,
                    "topic_shift": conv_res.topic_shift,
                },
            )
            if conv_res.is_followup:
                ctx.thinking_sm.start_stage(ThinkingStage.FOLLOW_UP_RESOLUTION)
                if conv_res.resolution and conv_res.resolution.ambiguity_detected:
                    ctx.thinking_sm.warn_stage(
                        ThinkingStage.FOLLOW_UP_RESOLUTION,
                        reason="Follow-up query contains broad phrasing; maintaining conversation continuity with prior topic.",
                        details={"is_follow_up": True, "answer_mode": conv_res.answer_mode.value},
                    )
                ctx.thinking_sm.complete_stage(
                    ThinkingStage.FOLLOW_UP_RESOLUTION,
                    details={
                        "is_follow_up": True,
                        "answer_mode": conv_res.answer_mode.value,
                        "active_topic": conv_res.active_topic,
                    },
                )
        else:
            ctx.thinking_sm.start_stage(ThinkingStage.CONVERSATION_CONTEXT)
            ctx.thinking_sm.complete_stage(
                ThinkingStage.CONVERSATION_CONTEXT,
                details={"is_follow_up": False},
            )
        ctx.conv_res = conv_res

        ctx.is_history_followup = bool(
            (conv_res and conv_res.is_followup)
            or (
                conv_res is None
                and ctx.history
                and self.query_rewriter._is_followup_query(ctx.user_query)
            )
        )
        ctx.effective_search_query = (
            conv_res.resolved_query if (conv_res and conv_res.is_followup) else ctx.user_query
        )

    def _try_general_chat(self, ctx: QueryContext) -> RAGResponse | None:
        """Retrieval-free General chat bypass. Returns a response, or None to continue."""
        if ctx.chat_mode != "general":
            return None

        t0 = time.perf_counter()
        history_text = _format_history_for_prompt(
            ctx.history,
            max_turns=max(1, int(getattr(settings, "memory_window_size", 5))),
        )
        prompt = GENERAL_CHAT_PROMPT.format(
            response_mode_instructions=ctx.response_mode_config.prompt_instructions,
            history_text=history_text,
            query=ctx.user_query,
        )
        ctx.thinking_sm.record_stage(ThinkingStage.ANSWER_PLANNING, ThinkingStatus.COMPLETED)
        ctx.thinking_sm.start_stage(ThinkingStage.ANSWER_GENERATION)
        if ctx.req_llm is not None:
            try:
                try:
                    answer_text = str(
                        ctx.req_llm.complete(
                            prompt,
                            temperature=0.6,
                            max_new_tokens=ctx.response_mode_config.max_output_tokens,
                        )
                    ).strip()
                except TypeError:
                    answer_text = str(ctx.req_llm.complete(prompt)).strip()
            except Exception as exc:
                logger.warning("General chat generation failed: %s", exc)
                answer_text = "General chat is selected, but the language model is currently unavailable. Please try again shortly."
        else:
            answer_text = "General chat is selected, but the language model is currently unavailable. Please try again shortly."
        ctx.thinking_sm.complete_stage(ThinkingStage.ANSWER_GENERATION)
        ctx.thinking_sm.record_stage(ThinkingStage.COMPLETED, ThinkingStatus.COMPLETED)
        ctx.stage_timings["general_chat_generation"] = round((time.perf_counter() - t0) * 1000, 2)
        total_elapsed = round((time.perf_counter() - ctx.total_start) * 1000, 2)
        reasoning_sum = ctx.thinking_sm.get_reasoning_summary(
            intent="conversational",
            answer_mode="DIRECT",
            is_follow_up=bool(ctx.history),
            used_conversation_context=bool(ctx.history),
            reused_previous_evidence=False,
            retrieved_new_evidence=False,
            used_visual_evidence=False,
            evidence_status="DIRECT",
        )
        trace = RAGTrace(
            query=ctx.user_query,
            rewritten_query=None,
            sub_queries=[],
            query_type=QueryCategory.CONVERSATIONAL.value,
            routing_confidence=1.0,
            retrieval_strategy="general_chat_bypass",
            query_scope="general",
            retrieved_candidate_count=0,
            post_rerank_count=0,
            final_context_count=0,
            response_mode=ctx.response_mode,
            retrieval_top_k=0,
            rerank_top_k=0,
            context_tokens=0,
            generation_max_tokens=ctx.response_mode_config.max_output_tokens,
            execution_time_ms=total_elapsed,
            stage_timings_ms=ctx.stage_timings,
            fallback_reason="general_chat_mode",
            faithfulness_checked=False,
            faithfulness_passed=True,
            verification_report=None,
            verification_score=None,
            retry_count=0,
            retry_reasons=[],
            cache_hit=False,
            cache_similarity=None,
            evidence_sufficiency_passed=True,
            generation_model=ctx.selected_model,
            grounding_validation_passed=True,
            reasoning_summary=reasoning_sum,
            thinking_events=[e.model_dump() for e in ctx.thinking_sm.get_all_events()],
        )
        _log_rag_trace(trace)
        return RAGResponse(
            id=f"resp_{uuid.uuid4().hex[:12]}",
            query=ctx.user_query,
            answer=answer_text,
            citations=[],
            context_chunks=[],
            trace=trace,
            model=ctx.selected_model,
            token_usage={"prompt_tokens": len(prompt.split()), "completion_tokens": len(answer_text.split())},
        )

    def _try_conversational_greeting(self, ctx: QueryContext) -> RAGResponse | None:
        """Greeting/conversational bypass. Returns a response, or None to continue."""
        if not (
            ctx.classification.category == QueryCategory.CONVERSATIONAL
            or self.query_rewriter.is_conversational(ctx.user_query)
        ):
            return None

        greeting_answer = (
            "Hello! How can I assist you today? Feel free to ask any questions regarding company policies, "
            "AI agent architectures, code implementations, or any uploaded documentation."
        )
        total_elapsed = round((time.perf_counter() - ctx.total_start) * 1000, 2)
        ctx.thinking_sm.record_stage(ThinkingStage.ANSWER_PLANNING, ThinkingStatus.COMPLETED)
        ctx.thinking_sm.record_stage(ThinkingStage.ANSWER_GENERATION, ThinkingStatus.COMPLETED)
        ctx.thinking_sm.record_stage(ThinkingStage.COMPLETED, ThinkingStatus.COMPLETED)
        reasoning_sum = ctx.thinking_sm.get_reasoning_summary(
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
            query=ctx.user_query,
            rewritten_query=None,
            sub_queries=[],
            query_type=ctx.classification.category.value,
            routing_confidence=ctx.classification.confidence,
            retrieval_strategy="conversational_bypass",
            query_scope="global",
            retrieved_candidate_count=0,
            post_rerank_count=0,
            final_context_count=0,
            response_mode=ctx.response_mode,
            retrieval_top_k=0,
            rerank_top_k=0,
            context_tokens=0,
            generation_max_tokens=0,
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
            generation_model=ctx.selected_model,
            grounding_validation_passed=True,
            reasoning_summary=reasoning_sum,
            thinking_events=[e.model_dump() for e in ctx.thinking_sm.get_all_events()],
        )
        _log_rag_trace(trace)
        return RAGResponse(
            id=f"resp_{uuid.uuid4().hex[:12]}",
            query=ctx.user_query,
            answer=greeting_answer,
            citations=[],
            context_chunks=[],
            trace=trace,
            model=ctx.selected_model,
            token_usage={"prompt_tokens": 0, "completion_tokens": len(greeting_answer.split())},
        )

    def _try_semantic_cache_hit(
        self,
        ctx: QueryContext,
        *,
        cache_enabled: bool,
        cache_read_eligible: bool,
    ) -> RAGResponse | None:
        """Pre-rewrite semantic-cache lookup. Returns a cached response, or None to continue."""
        if not (cache_enabled and self.semantic_cache is not None and cache_read_eligible):
            return None

        t0 = time.perf_counter()
        cached_res = self.semantic_cache.get(
            ctx.user_query,
            model_name=ctx.selected_model,
            cache_context=ctx.cache_context,
        )
        ctx.stage_timings["cache_lookup"] = round((time.perf_counter() - t0) * 1000, 2)
        if cached_res is not None and (
            _is_degraded_or_abstention_answer(cached_res.answer)
            or not _answer_matches_requested_enumeration(ctx.user_query, cached_res.answer)
        ):
            logger.info("Ignoring incomplete semantic-cache answer and continuing with fresh retrieval.")
            cached_res = None
        if cached_res is None:
            return None

        total_elapsed = round((time.perf_counter() - ctx.total_start) * 1000, 2)
        ctx.thinking_sm.record_stage(
            ThinkingStage.RETRIEVAL,
            ThinkingStatus.COMPLETED,
            summary="Retrieved verified answer from semantic cache.",
            details={"cache_hit": True},
        )
        ctx.thinking_sm.record_stage(ThinkingStage.ANSWER_PLANNING, ThinkingStatus.COMPLETED)
        ctx.thinking_sm.record_stage(ThinkingStage.ANSWER_GENERATION, ThinkingStatus.COMPLETED)
        ctx.thinking_sm.record_stage(ThinkingStage.COMPLETED, ThinkingStatus.COMPLETED)
        reasoning_sum = ctx.thinking_sm.get_reasoning_summary(
            intent=ctx.classification.category.value,
            answer_mode="DIRECT",
            is_follow_up=False,
            used_conversation_context=False,
            reused_previous_evidence=False,
            retrieved_new_evidence=False,
            used_visual_evidence=False,
            evidence_status="DIRECT",
        )
        trace = RAGTrace(
            query=ctx.user_query,
            rewritten_query=None,
            sub_queries=[],
            query_type=ctx.classification.category.value,
            routing_confidence=ctx.classification.confidence,
            retrieval_strategy=ctx.strategy.name,
            query_scope="global",
            retrieved_candidate_count=0,
            post_rerank_count=0,
            final_context_count=0,
            response_mode=ctx.response_mode,
            retrieval_top_k=ctx.response_mode_config.retrieval_top_k,
            rerank_top_k=ctx.response_mode_config.rerank_top_k,
            context_tokens=0,
            generation_max_tokens=ctx.response_mode_config.max_output_tokens,
            execution_time_ms=total_elapsed,
            stage_timings_ms=ctx.stage_timings,
            fallback_reason="none",
            faithfulness_checked=True,
            faithfulness_passed=True,
            verification_report=None,
            verification_score=1.0,
            retry_count=0,
            retry_reasons=[],
            cache_hit=True,
            cache_similarity=cached_res.similarity_score,
            generation_model=ctx.selected_model,
            evidence_sufficiency_passed=True,
            grounding_validation_passed=True,
            reasoning_summary=reasoning_sum,
            thinking_events=[e.model_dump() for e in ctx.thinking_sm.get_all_events()],
        )
        _log_rag_trace(trace)
        return RAGResponse(
            id=f"resp_{uuid.uuid4().hex[:12]}",
            query=ctx.user_query,
            answer=cached_res.answer,
            citations=cached_res.citations,
            context_chunks=[],
            trace=trace,
            model=ctx.model or "semantic_cache",
            token_usage={"prompt_tokens": 0, "completion_tokens": len(cached_res.answer.split())},
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
        response_mode: ResponseMode = "standard",
        thinking_detail_level: ThinkingDetailLevel | str = ThinkingDetailLevel.STANDARD,
        thinking_sm: ThinkingStateMachine | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> RAGResponse:
        total_start = time.perf_counter()
        stage_timings: dict[str, float] = {}
        filters, chat_mode = self._split_control_filters(filters)
        response_mode_config = get_response_mode_config(response_mode)

        if thinking_sm is None:
            thinking_sm = ThinkingStateMachine(
                query_id=f"qry_{uuid.uuid4().hex[:8]}",
                detail_level=thinking_detail_level,
            )

        # Stage: RECEIVED
        thinking_sm.start_stage(ThinkingStage.RECEIVED)
        thinking_sm.complete_stage(ThinkingStage.RECEIVED)

        # State shared across the extracted pipeline stages. Built once here from
        # the request inputs; each stage reads and extends it in place.
        ctx = QueryContext(
            user_query=user_query,
            filters=filters,
            chat_mode=chat_mode,
            history=history,
            model=model,
            active_document_id=active_document_id,
            active_document_name=active_document_name,
            selected_document_ids=selected_document_ids,
            document_scope=document_scope,
            conversation_state=conversation_state,
            response_mode=response_mode,
            thinking_detail_level=thinking_detail_level,
            thinking_sm=thinking_sm,
            stream_callback=stream_callback,
            total_start=total_start,
            stage_timings=stage_timings,
            response_mode_config=response_mode_config,
        )

        # Query classification, conversational resolution, and follow-up flags.
        self._stage_classify_and_resolve(ctx)

        ctx.req_llm, ctx.selected_model = self._get_effective_llm(model)

        # Bridge: the not-yet-extracted remainder of this method still uses these
        # as locals. Each binding is removed as the stage that reads it is extracted.
        classification = ctx.classification
        strategy = ctx.strategy
        fidelity_mode = ctx.fidelity_mode
        conv_res = ctx.conv_res
        is_history_followup = ctx.is_history_followup
        effective_search_query = ctx.effective_search_query
        req_llm = ctx.req_llm
        selected_model = ctx.selected_model

        # Explicit General chat mode is a true retrieval bypass, not a document
        # category filter. It uses only same-mode history supplied by ChatService.
        general_chat_response = self._try_general_chat(ctx)
        if general_chat_response is not None:
            return general_chat_response

        # Conversational / Greeting intent check
        greeting_response = self._try_conversational_greeting(ctx)
        if greeting_response is not None:
            return greeting_response

        # 0b. Pre-rewrite Cache Lookup
        cache_enabled = (
            getattr(self.semantic_cache.settings, "semantic_cache_enabled", True)
            if (self.semantic_cache and hasattr(self.semantic_cache, "settings"))
            else True
        )
        normalized_scope = str(document_scope or "global").strip().lower()
        cache_context = json.dumps(
            {
                "policy": "precision_v8",
                "scope": normalized_scope,
                "filters": filters or {},
                "response_mode": response_mode,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        is_exact_repeat = bool(
            conversation_state
            and conversation_state.last_user_query
            and " ".join(user_query.casefold().split())
            == " ".join(conversation_state.last_user_query.casefold().split())
        )
        if conv_res:
            is_followup_intent = conv_res.is_followup
        else:
            # Fallback if conversation resolver didn't run: assume followup if history exists
            is_followup_intent = bool(history and len(history) > 0 and not is_exact_repeat)

        cache_eligible = not (
            is_followup_intent
            or filters
            or active_document_id
            or active_document_name
            or selected_document_ids
            or normalized_scope not in {"", "all", "global"}
        )
        cache_read_eligible = cache_eligible and not is_exact_repeat
        # Mirror eligibility onto ctx so the later cache-write path can reuse it.
        ctx.cache_context = cache_context
        ctx.cache_eligible = cache_eligible
        cache_hit_response = self._try_semantic_cache_hit(
            ctx, cache_enabled=cache_enabled, cache_read_eligible=cache_read_eligible
        )
        if cache_hit_response is not None:
            return cache_hit_response

        # Document-scope resolution, query rewrite, and metadata-filter inference.
        self._stage_scope_and_rewrite(ctx)
        scope_decision = ctx.scope_decision
        rewrite_res = ctx.rewrite_res
        inferred_filters = ctx.inferred_filters
        applied_filters = ctx.applied_filters
        filter_relaxed = ctx.filter_relaxed
        active_document_id = ctx.active_document_id
        active_document_name = ctx.active_document_name
        selected_document_ids = ctx.selected_document_ids

        # Multi-part decomposition, fast-path decision, retry budget, and strategy.
        self._stage_plan(ctx)
        question_parts = ctx.question_parts
        is_fast_path = ctx.is_fast_path
        enable_verification = ctx.enable_verification
        max_retries = ctx.max_retries
        current_strategy = ctx.current_strategy

        attempt = 0
        best_answer = ""
        best_citations: list[Citation] = []
        best_context_chunks: list[ScoredChunk] = []
        best_context_tokens = 0
        best_candidate_chunks: list[ScoredChunk] = []
        best_reranked_chunks: list[ScoredChunk] = []
        best_report: VerificationReport | None = None
        best_policy_selection: ClauseSelection | None = None
        best_score = -1.0
        retry_reasons: list[str] = []
        prompt_refinement = ""
        sub_queries: list[str] = [rewrite_res.rewritten_query]
        formatted_context = ""
        context_tokens = 0
        cross_document_count = 0
        telemetry_extra: dict[str, Any] = {}
        policy_selection: ClauseSelection | None = None

        while attempt <= max_retries:
            prefix = f"_att{attempt}" if attempt > 0 else ""

            self._stage_retrieve(ctx, prefix)
            candidate_chunks = ctx.candidate_chunks
            sub_queries = ctx.sub_queries
            applied_filters = ctx.applied_filters
            filter_relaxed = ctx.filter_relaxed
            cross_document_count = ctx.cross_document_count

            if not candidate_chunks:
                if scope_decision.page_number is not None and scope_decision.active_document_name:
                    unanswerable_text = (
                        f"I could not find a page labeled {scope_decision.page_number} "
                        f"in the active document '{scope_decision.active_document_name}'."
                    )
                elif scope_decision.page_number is not None and scope_decision.active_document_id:
                    unanswerable_text = (
                        f"I could not find a page labeled {scope_decision.page_number} "
                        f"in the active document '{scope_decision.active_document_id}'."
                    )
                elif scope_decision.active_document_name:
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

            self._stage_rerank_and_context(ctx, prefix)
            reranked_chunks = ctx.reranked_chunks
            expanded_chunks = ctx.expanded_chunks
            policy_selection = ctx.policy_selection
            telemetry_extra = ctx.telemetry_extra
            context_tokens = ctx.context_tokens
            formatted_context = ctx.formatted_context

            self._stage_generate(ctx, prefix)
            answer_text = ctx.answer_text

            # 7. Verifiable Citation Extraction
            t0 = time.perf_counter()
            thinking_sm.start_stage(ThinkingStage.CITATION_BUILDING)
            citations = self.citation_engine.select_citations(
                answer_text=answer_text,
                generation_chunks=expanded_chunks,
                user_query=user_query,
                max_citations=response_mode_config.max_citations,
            )
            if telemetry_extra.get("requires_visual_abstention"):
                # The deterministic abstention makes one claim only: the
                # requested labels live in the referenced page visual. Keep
                # only that primary page instead of attaching nearby chunks.
                citations = citations[:1]
            stage_timings[f"citation_extraction{prefix}"] = round((time.perf_counter() - t0) * 1000, 2)
            thinking_sm.complete_stage(
                ThinkingStage.CITATION_BUILDING,
                details={"citation_count": len(citations)},
            )

            # 8. Post-Generation Verification
            t0 = time.perf_counter()
            if enable_verification and self.verifier is not None:
                # High-risk (policy / numeric) answers are already buffered before
                # the user sees them (see _stage_plan), so an LLM claim-support
                # audit here costs no perceived latency and catches hallucinations
                # the lexical heuristic cannot. Ordinary factual answers keep the
                # fast heuristic-only path.
                use_llm_judge = bool(
                    ctx.is_high_risk
                    and req_llm is not None
                    and getattr(settings, "enable_llm_faithfulness_verification", True)
                )
                report = self.verifier.verify(
                    query=user_query,
                    answer=answer_text,
                    context_chunks=expanded_chunks,
                    citations=citations,
                    llm=req_llm,
                    allowed_derived_facts=allowed_derived_facts(policy_selection),
                    use_llm_judge=use_llm_judge,
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
            policy_validation = validate_policy_answer(answer_text, policy_selection)
            report.incorrect_numbers = list(policy_validation.incorrect_numbers)
            report.missed_conditions = list(policy_validation.missed_conditions)
            report.missed_exceptions = list(policy_validation.missed_exceptions)
            if policy_validation.unsupported_numbers:
                report.unsupported_claims.extend(
                    f"Unsupported numerical claim: {item}"
                    for item in policy_validation.unsupported_numbers
                )
            if not policy_validation.passed:
                report.passed = False
                report.overall_grounded = False
            stage_timings[f"verification{prefix}"] = round((time.perf_counter() - t0) * 1000, 2)

            if report.composite_score > best_score or best_report is None:
                best_score = report.composite_score
                best_answer = answer_text
                best_citations = citations
                best_context_chunks = expanded_chunks
                best_context_tokens = context_tokens
                best_candidate_chunks = candidate_chunks
                best_reranked_chunks = reranked_chunks
                best_report = report
                best_policy_selection = policy_selection

            if report.passed or is_fast_path:
                best_answer = answer_text
                best_citations = citations
                best_context_chunks = expanded_chunks
                best_context_tokens = context_tokens
                best_candidate_chunks = candidate_chunks
                best_reranked_chunks = reranked_chunks
                best_report = report
                best_policy_selection = policy_selection
                break

            if attempt >= max_retries or not self.retry_engine.should_retry(attempt, report):
                break

            current_strategy, prompt_refinement = self.retry_engine.prepare_retry(
                attempt=attempt,
                report=report,
                strategy=current_strategy,
                query=user_query,
            )
            current_strategy = response_mode_config.apply_to(current_strategy)
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
            reused_previous_evidence=ctx.continuity_applied,
            retrieved_new_evidence=bool(ctx.raw_new_chunk_count > 0),
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
            evidence_continuity_applied=ctx.continuity_applied,
            merged_chunk_count=len(best_candidate_chunks),
            previous_chunk_count=len(ctx.prev_all),
            new_chunk_count=ctx.raw_new_chunk_count,
            retrieved_candidate_count=len(best_candidate_chunks),
            post_rerank_count=len(best_reranked_chunks),
            final_context_count=len(best_context_chunks),
            response_mode=response_mode,
            retrieval_top_k=current_strategy.dense_top_k,
            rerank_top_k=current_strategy.rerank_top_n,
            context_tokens=best_context_tokens,
            generation_max_tokens=response_mode_config.max_output_tokens,
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
            governing_clause_confidence=(
                best_policy_selection.confidence if best_policy_selection else None
            ),
            selected_primary_clause=(
                best_policy_selection.to_trace_dict()["primary_rules"][0]
                if best_policy_selection and best_policy_selection.primary_rules
                else None
            ),
            selected_exceptions=(
                best_policy_selection.to_trace_dict()["exceptions"]
                if best_policy_selection
                else []
            ),
            selected_definitions=(
                best_policy_selection.to_trace_dict()["definitions"]
                if best_policy_selection
                else []
            ),
            structured_rules=(
                best_policy_selection.to_trace_dict()["structured_rules"]
                if best_policy_selection
                else []
            ),
            deterministic_calculations=(
                best_policy_selection.to_trace_dict()["calculations"]
                if best_policy_selection
                else []
            ),
            missing_required_inputs=(
                best_policy_selection.missing_inputs if best_policy_selection else []
            ),
            anchor_section=anchor_sec,
            page_identity=primary_page_id_str,
            text_candidates=txt_cands,
            visual_candidates=vis_cands,
            final_text_evidence=text_cnt,
            final_visual_evidence=diag_cnt + code_cnt,
            visual_asset_status=telemetry_extra.get("visual_asset_status") or ("FOUND" if (diag_cnt > 0 or any(c.chunk.metadata.image_assets or c.chunk.metadata.visual_asset_ids for c in best_context_chunks)) else "NONE"),
            vision_status=telemetry_extra.get("vision_status") or ("READY" if (diag_cnt > 0 or any(c.chunk.metadata.image_assets or c.chunk.metadata.visual_asset_ids for c in best_context_chunks)) else "N/A"),
            evidence_status=str(telemetry_extra.get("evidence_status", "DIRECT")),

            grounding_status="PASS" if (best_report is None or best_report.passed) else "FAIL",
            evidence_text_count=text_cnt,
            evidence_code_count=code_cnt,
            evidence_diagram_count=diag_cnt,
            evidence_table_count=tab_cnt,
            section_expansion=telemetry_extra.get("section_expansion", False),
            adjacent_page_check=telemetry_extra.get("adjacent_page_check", False),
            vision_fallback=telemetry_extra.get("vision_fallback", False),
            vision_model=str(telemetry_extra.get("vision_model", "Qwen3-VL-2B-Instruct")) if isinstance(telemetry_extra.get("vision_model"), str) else "Qwen3-VL-2B-Instruct",
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
            and _answer_matches_requested_enumeration(user_query, best_answer)
            and _is_cacheable_grounded_answer(
            best_answer,
            has_citations=bool(best_citations),
            verifier_passed=(best_report is None or best_report.passed),
            evidence_sufficiency_passed=bool(
                telemetry_extra.get("evidence_sufficiency_passed", True)
            ),
            vision_status=telemetry_extra.get("vision_status"),
            requires_visual_abstention=bool(
                telemetry_extra.get("requires_visual_abstention", False)
            ),
            )
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
        response_mode: ResponseMode = "standard",
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
                response_mode=response_mode,
                thinking_detail_level=thinking_detail_level,
                cancel_token=cancel_token,
            )
        ):
            yield chunk

    def _stream_general_chat_response(
        self,
        *,
        user_query: str,
        history: list[dict[str, Any]] | None,
        model: str | None,
        thinking_sm: ThinkingStateMachine,
        response_mode: ResponseMode,
        cancel_token: Any = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Use the LLM's real stream for retrieval-free General chat mode."""
        req_llm, selected_model = self._get_effective_llm(model)
        response_mode_config = get_response_mode_config(response_mode)
        history_text = _format_history_for_prompt(
            history,
            max_turns=max(1, int(getattr(settings, "memory_window_size", 5))),
        )
        prompt = GENERAL_CHAT_PROMPT.format(
            response_mode_instructions=response_mode_config.prompt_instructions,
            history_text=history_text,
            query=user_query,
        )

        thinking_sm.record_stage(ThinkingStage.ANSWER_PLANNING, ThinkingStatus.COMPLETED)
        thinking_sm.start_stage(ThinkingStage.ANSWER_GENERATION)
        yield {
            "type": "retrieval_done",
            "stage_timings": {},
            "candidate_count": 0,
            "reranked_count": 0,
            "context_count": 0,
            "cache_hit": False,
        }
        generation_events = [
            event
            for event in thinking_sm.get_visible_events()
            if event.stage == ThinkingStage.ANSWER_GENERATION
        ]
        if generation_events:
            yield {"type": "thinking", "event": generation_events[-1]}

        t_generation = time.perf_counter()
        answer_parts: list[str] = []
        streamed = False
        if req_llm is not None and hasattr(req_llm, "stream_complete"):
            try:
                try:
                    completion_stream = req_llm.stream_complete(
                        prompt,
                        temperature=0.6,
                        max_new_tokens=response_mode_config.max_output_tokens,
                    )
                except TypeError:
                    completion_stream = req_llm.stream_complete(prompt)
                for part in completion_stream:
                    if cancel_token and cancel_token.is_set():
                        return
                    delta = getattr(part, "delta", None)
                    if delta is None:
                        delta = getattr(part, "text", None)
                    if delta is None:
                        delta = str(part)
                    delta = str(delta)
                    if not delta:
                        continue
                    answer_parts.append(delta)
                    streamed = True
                    yield {"type": "token", "content": delta}
            except Exception as exc:
                logger.warning("General chat streaming failed: %s", exc)

        if not streamed:
            if req_llm is not None:
                try:
                    try:
                        fallback_answer = str(
                            req_llm.complete(
                                prompt,
                                temperature=0.6,
                                max_new_tokens=response_mode_config.max_output_tokens,
                            )
                        ).strip()
                    except TypeError:
                        fallback_answer = str(req_llm.complete(prompt)).strip()
                except Exception as exc:
                    logger.warning("General chat generation failed: %s", exc)
                    fallback_answer = ""
            else:
                fallback_answer = ""

            if not fallback_answer:
                fallback_answer = (
                    "General chat is selected, but the language model is currently "
                    "unavailable. Please try again shortly."
                )
            answer_parts = [fallback_answer]
            words = fallback_answer.split(" ")
            for index, word in enumerate(words):
                if cancel_token and cancel_token.is_set():
                    return
                yield {
                    "type": "token",
                    "content": word + (" " if index < len(words) - 1 else ""),
                }

        answer_text = "".join(answer_parts).strip()
        generation_ms = round((time.perf_counter() - t_generation) * 1000, 2)
        thinking_sm.complete_stage(ThinkingStage.ANSWER_GENERATION)
        thinking_sm.record_stage(ThinkingStage.COMPLETED, ThinkingStatus.COMPLETED)
        reasoning_sum = thinking_sm.get_reasoning_summary(
            intent="conversational",
            answer_mode="DIRECT",
            is_follow_up=bool(history),
            used_conversation_context=bool(history),
            reused_previous_evidence=False,
            retrieved_new_evidence=False,
            used_visual_evidence=False,
            evidence_status="DIRECT",
        )
        trace = RAGTrace(
            query=user_query,
            query_type=QueryCategory.CONVERSATIONAL.value,
            routing_confidence=1.0,
            retrieval_strategy="general_chat_bypass",
            query_scope="general",
            retrieved_candidate_count=0,
            post_rerank_count=0,
            final_context_count=0,
            response_mode=response_mode,
            retrieval_top_k=0,
            rerank_top_k=0,
            context_tokens=0,
            generation_max_tokens=response_mode_config.max_output_tokens,
            execution_time_ms=generation_ms,
            stage_timings_ms={"general_chat_generation": generation_ms},
            fallback_reason="general_chat_mode",
            faithfulness_checked=False,
            faithfulness_passed=True,
            verification_score=None,
            retry_count=0,
            cache_hit=False,
            evidence_sufficiency_passed=True,
            generation_model=selected_model,
            grounding_validation_passed=True,
            reasoning_summary=reasoning_sum,
            thinking_events=[event.model_dump() for event in thinking_sm.get_all_events()],
        )
        _log_rag_trace(trace)
        yield {
            "type": "done",
            "answer": answer_text,
            "citations": [],
            "context_chunks": [],
            "trace": trace,
            "model": selected_model,
            "token_usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(answer_text.split()),
            },
            "total_elapsed_ms": generation_ms,
            "cache_hit": False,
            "reasoning_summary": reasoning_sum.model_dump(),
            "thinking_events": trace.thinking_events,
        }

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
        response_mode: ResponseMode = "standard",
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

        _, chat_mode = self._split_control_filters(filters)
        if chat_mode == "general":
            yield from self._stream_general_chat_response(
                user_query=user_query,
                history=history,
                model=model,
                thinking_sm=thinking_sm,
                response_mode=response_mode,
                cancel_token=cancel_token,
            )
            return

        token_queue: queue.Queue[str] = queue.Queue()
        result_box: dict[str, Any] = {}
        worker_done = threading.Event()

        def run_query() -> None:
            try:
                result_box["response"] = self.query(
                    user_query=user_query,
                    filters=filters,
                    history=history,
                    model=model,
                    active_document_id=active_document_id,
                    active_document_name=active_document_name,
                    selected_document_ids=selected_document_ids,
                    document_scope=document_scope,
                    conversation_state=conversation_state,
                    response_mode=response_mode,
                    thinking_detail_level=thinking_detail_level,
                    thinking_sm=thinking_sm,
                    stream_callback=token_queue.put,
                )
            except BaseException as exc:
                result_box["error"] = exc
            finally:
                worker_done.set()

        worker = threading.Thread(target=run_query, daemon=True, name="rag-stream-worker")
        worker.start()

        streamed_answer = False
        retrieval_event_sent = False
        while not worker_done.is_set() or not token_queue.empty():
            if cancel_token and cancel_token.is_set():
                return
            try:
                token_text = token_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if not retrieval_event_sent:
                # The first model delta proves retrieval and context building
                # have completed, even though final timings are not known yet.
                yield {
                    "type": "retrieval_done",
                    "stage_timings": {},
                    "candidate_count": 0,
                    "reranked_count": 0,
                    "context_count": 0,
                    "cache_hit": False,
                }
                retrieval_event_sent = True
            streamed_answer = True
            yield {"type": "token", "content": token_text}

        if "error" in result_box:
            raise result_box["error"]
        resp = result_box["response"]

        # 1. Progressive Pre-generation Thinking Events (before generation)
        for ev in thinking_sm.get_visible_events():
            if ev.stage in (ThinkingStage.ANSWER_GENERATION, ThinkingStage.CITATION_BUILDING, ThinkingStage.COMPLETED):
                continue
            if cancel_token and cancel_token.is_set():
                return
            yield {"type": "thinking", "event": ev}

        # 2. Retrieval Done Event (for backward compatibility)
        if not retrieval_event_sent:
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
        if not streamed_answer:
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
