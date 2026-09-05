"""
Single source of truth for all hyperparameters and paths.

Production rationale:
- Centralized config prevents drift between indexing, agent, and chat app.
- Pydantic Settings allows .env overrides without code changes (12-factor).
- Chunk size 640 tokens balances legal/policy recall (sections stay intact)
  with embedding model context limits (nomic-embed-text ≈ 8192, but smaller
  chunks improve retrieval precision for clause-level questions).
"""

from __future__ import annotations

import os

# Disable Chroma anonymized telemetry before any chromadb import.
# Avoids posthog version mismatch noise: "capture() takes 1 positional argument..."
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_project_root() -> Path:
    """Resolve runtime project root for dev, pip install, and Docker layouts."""
    env_root = os.environ.get("POLICY_RAG_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    dev_root = Path(__file__).resolve().parent.parent
    if (dev_root / "pyproject.toml").is_file() or (dev_root / "data").is_dir():
        return dev_root

    return Path.cwd().resolve()


# Project root: one level above src/ in dev; cwd or POLICY_RAG_ROOT when pip-installed
PROJECT_ROOT = _resolve_project_root()


class Settings(BaseSettings):
    """All tunable parameters for indexing, retrieval, and generation."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Paths ──────────────────────────────────────────────────────────────
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    policies_dir: Path = Field(default=PROJECT_ROOT / "data" / "policies")
    legal_dir: Path = Field(default=PROJECT_ROOT / "data" / "legal")
    raw_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    storage_dir: Path = Field(default=PROJECT_ROOT / "storage")
    pdf_images_dir: Path = Field(default=PROJECT_ROOT / "storage" / "images")
    logs_dir: Path = Field(default=PROJECT_ROOT / "logs")

    # ── ChromaDB vector store ──────────────────────────────────────────────
    # Chroma replaces SimpleVectorStore: persistent storage, metadata filtering,
    # and incremental indexing without full rebuilds.
    chroma_persist_dir: Path = Field(default=PROJECT_ROOT / "storage" / "chroma")
    chroma_collection_name: str = Field(
        default="company_policies", alias="CHROMA_COLLECTION_NAME"
    )
    # Distance function for HNSW index: cosine works well with normalized embeddings
    chroma_distance_fn: Literal["cosine", "l2", "ip"] = Field(
        default="cosine", alias="CHROMA_DISTANCE_FN"
    )

    # ── Semantic Cache ─────────────────────────────────────────────────────
    semantic_cache_enabled: bool = Field(
        default=True, alias="SEMANTIC_CACHE_ENABLED"
    )
    semantic_cache_threshold: float = Field(
        default=0.95, alias="SEMANTIC_CACHE_THRESHOLD"
    )
    semantic_cache_collection_name: str = Field(
        default="semantic_cache", alias="SEMANTIC_CACHE_COLLECTION_NAME"
    )

    @property
    def SEMANTIC_CACHE_ENABLED(self) -> bool:
        return self.semantic_cache_enabled

    @property
    def SEMANTIC_CACHE_THRESHOLD(self) -> float:
        return self.semantic_cache_threshold

    @property
    def SEMANTIC_CACHE_COLLECTION_NAME(self) -> str:
        return self.semantic_cache_collection_name

    # ── Agentic Intelligence Layer: Feature Flags ───────────────────────────
    enable_query_routing: bool = Field(default=True, alias="ENABLE_QUERY_ROUTING")
    enable_answer_verification: bool = Field(default=True, alias="ENABLE_ANSWER_VERIFICATION")
    enable_metadata_extraction: bool = Field(default=True, alias="ENABLE_METADATA_EXTRACTION")
    enable_query_metadata_filtering: bool = Field(default=True, alias="ENABLE_QUERY_METADATA_FILTERING")

    @property
    def ENABLE_QUERY_ROUTING(self) -> bool:
        return self.enable_query_routing

    @property
    def ENABLE_ANSWER_VERIFICATION(self) -> bool:
        return self.enable_answer_verification

    @property
    def ENABLE_METADATA_EXTRACTION(self) -> bool:
        return self.enable_metadata_extraction

    @property
    def ENABLE_QUERY_METADATA_FILTERING(self) -> bool:
        return self.enable_query_metadata_filtering

    # ── Dynamic Metadata Extraction & Filtering ─────────────────────────────
    metadata_extraction_mode: Literal["heuristic", "llm", "hybrid"] = Field(
        default="heuristic", alias="METADATA_EXTRACTION_MODE"
    )
    metadata_extractor_model: str = Field(default="qwen2.5:7b", alias="METADATA_EXTRACTOR_MODEL")
    metadata_filter_fallback_relaxation: bool = Field(
        default=True, alias="METADATA_FILTER_FALLBACK_RELAXATION"
    )
    enable_filter_fallback_relaxation: bool = Field(
        default=True, alias="ENABLE_FILTER_FALLBACK_RELAXATION"
    )
    metadata_filter_min_confidence: float = Field(
        default=0.60, alias="METADATA_FILTER_MIN_CONFIDENCE"
    )
    metadata_confidence_threshold: float = Field(
        default=0.60, alias="METADATA_CONFIDENCE_THRESHOLD"
    )
    metadata_max_entities_per_chunk: int = Field(
        default=20, alias="METADATA_MAX_ENTITIES_PER_CHUNK"
    )

    @property
    def METADATA_EXTRACTION_MODE(self) -> str:
        return self.metadata_extraction_mode

    # ── Query Routing Thresholds ────────────────────────────────────────────
    query_router_confidence_threshold: float = Field(
        default=0.70, alias="QUERY_ROUTER_CONFIDENCE_THRESHOLD"
    )
    enable_conversational_bypass: bool = Field(
        default=True, alias="ENABLE_CONVERSATIONAL_BYPASS"
    )

    # ── Self-Reflection & Answer Verification Thresholds ────────────────────
    verification_faithfulness_threshold: float = Field(
        default=0.75, alias="VERIFICATION_FAITHFULNESS_THRESHOLD"
    )
    verification_completeness_threshold: float = Field(
        default=0.70, alias="VERIFICATION_COMPLETENESS_THRESHOLD"
    )
    verification_citation_threshold: float = Field(
        default=0.60, alias="VERIFICATION_CITATION_THRESHOLD"
    )
    verification_coherence_threshold: float = Field(
        default=0.70, alias="VERIFICATION_COHERENCE_THRESHOLD"
    )
    verification_composite_threshold: float = Field(
        default=0.70, alias="VERIFICATION_COMPOSITE_THRESHOLD"
    )
    verification_max_retries: int = Field(
        default=2, alias="VERIFICATION_MAX_RETRIES"
    )
    # LLM-backed faithfulness auditing. The heuristic verifier only measures
    # lexical overlap; an LLM judge actually checks whether each claim is
    # supported by the retrieved context. Runs only for high-risk (policy /
    # numeric) answers, which are already buffered before the user sees them,
    # so it adds no latency to ordinary factual streaming. It can only make the
    # verdict stricter (catch hallucinations), never inflate a weak answer, and
    # falls back to the heuristic on any LLM or parse failure.
    enable_llm_faithfulness_verification: bool = Field(
        default=True, alias="ENABLE_LLM_FAITHFULNESS_VERIFICATION"
    )
    llm_verification_max_context_chars: int = Field(
        default=6000, alias="LLM_VERIFICATION_MAX_CONTEXT_CHARS"
    )


    # ── Ollama / Models ────────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    text_model: str = Field(default="qwen2.5:7b", alias="TEXT_MODEL")
    llm_model: str = Field(default="qwen2.5:7b", alias="OLLAMA_LLM_MODEL")
    embed_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBED_MODEL")

    # ── Vision Model & Document Understanding ──────────────────────────────
    vision_model: str = Field(default="Qwen3-VL-2B-Instruct", alias="VISION_MODEL")
    vision_model_path: Path = Field(
        default=Path.home() / "Qwen3-VL-2B-Instruct",
        alias="VISION_MODEL_PATH",
    )
    vision_enabled: bool = Field(default=True, alias="VISION_ENABLED")
    vision_cache_dir: Path = Field(default=PROJECT_ROOT / "storage" / "vision_cache")
    images_storage_dir: Path = Field(default=PROJECT_ROOT / "storage" / "images")
    vision_dpi: int = Field(default=150, alias="VISION_DPI")
    vision_inference_max_dimension: int = Field(default=768, alias="VISION_INFERENCE_MAX_DIMENSION")
    vision_num_ctx: int = Field(default=4096, alias="VISION_NUM_CTX")
    # Retrieval needs concise factual descriptions, not long-form visual prose.
    # This protects the query-time budget on partly CPU-offloaded local models.
    vision_num_predict: int = Field(default=160, alias="VISION_NUM_PREDICT")
    vision_max_ingestion_retries: int = Field(default=0, alias="VISION_MAX_INGESTION_RETRIES")
    vision_max_lazy_retries: int = Field(default=0, alias="VISION_MAX_LAZY_RETRIES")
    vision_timeout_seconds: float = Field(default=35.0, alias="VISION_TIMEOUT_SECONDS")
    enable_lazy_vision_fallback: bool = Field(default=True, alias="ENABLE_LAZY_VISION_FALLBACK")
    vision_request_timeout: float = Field(default=30.0, alias="VISION_REQUEST_TIMEOUT")
    vision_query_budget_seconds: float = Field(default=40.0, alias="VISION_QUERY_BUDGET_SECONDS")
    vision_query_max_pages: int = Field(default=2, alias="VISION_QUERY_MAX_PAGES")
    vision_min_gpu_free_gb: float = Field(default=2.0, alias="VISION_MIN_GPU_FREE_GB")
    # CPU-only Qwen3-VL generation can take several minutes and cannot be
    # cancelled safely in-process. Query-time vision therefore requires a GPU
    # by default; cached visual understanding remains available on every host.
    vision_allow_cpu_query_time: bool = Field(default=False, alias="VISION_ALLOW_CPU_QUERY_TIME")

    @property
    def VISION_MODEL(self) -> str:
        return self.vision_model

    @property
    def VISION_ENABLED(self) -> bool:
        return self.vision_enabled

    @property
    def TEXT_MODEL(self) -> str:
        return self.text_model or self.llm_model

    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    # 300s tolerates slow CPU-bound local generation; the API composition root
    # reads this rather than defining its own default.
    llm_request_timeout: float = Field(default=300.0, alias="LLM_REQUEST_TIMEOUT")
    # 4K keeps Qwen2.5 7B mostly GPU-resident on common 6 GB cards. The
    # model's 32K native default creates an oversized KV cache and CPU offload.
    llm_context_window: int = Field(default=4096, alias="LLM_CONTEXT_WINDOW")

    # ── PDF parsing ────────────────────────────────────────────────────────
    # Marker improves layout/code extraction on technical PDFs; PDFReader fallback always available.
    enable_marker_pdf: bool = Field(default=False, alias="ENABLE_MARKER_PDF")
    pdf_parser: Literal["auto", "marker", "pypdf"] = Field(default="auto", alias="PDF_PARSER")
    marker_device: str = Field(default="auto", alias="MARKER_DEVICE")

    # ── Chunking (tokens) ──────────────────────────────────────────────────
    # Hierarchical: parents in docstore (wide context), children embedded in Chroma (precision).
    enable_hierarchical_chunking: bool = Field(
        default=True, alias="ENABLE_HIERARCHICAL_CHUNKING"
    )
    parent_chunk_size: int = Field(default=2000, alias="PARENT_CHUNK_SIZE")
    # Child size smaller than legacy 640 — better code/diagram retrieval precision.
    chunk_size: int = Field(default=480, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")
    docstore_dir: Path = Field(default=PROJECT_ROOT / "storage" / "docstore")

    # ── Diagram captions (indexed as embeddable nodes) ─────────────────────
    enable_diagram_captions: bool = Field(default=True, alias="ENABLE_DIAGRAM_CAPTIONS")
    enable_caption_llm: bool = Field(default=False, alias="ENABLE_CAPTION_LLM")
    caption_model: str = Field(default="qwen2.5:7b", alias="CAPTION_MODEL")

    # ── Section detection ──────────────────────────────────────────────────
    # Rich structural metadata is among the highest-ROI improvements for
    # legal/policy RAG — enables better citations, filtering, and reranking.
    enable_section_detection: bool = Field(default=True, alias="ENABLE_SECTION_DETECTION")
    # standard = balanced; strict = formal patterns only; permissive = + ALL CAPS
    section_detection_mode: Literal["standard", "strict", "permissive"] = Field(
        default="standard", alias="SECTION_DETECTION_MODE"
    )
    # Lines to scan at the top of each PDF page for section headers
    section_page_scan_lines: int = Field(default=25)

    # ── Retrieval ──────────────────────────────────────────────────────────
    similarity_top_k: int = Field(default=5, alias="SIMILARITY_TOP_K")
    # Over-retrieve for reranker pool. 15 is optimal for Qwen 2.5 7B (high recall without excess overhead).
    retrieval_candidate_k: int = Field(default=15, alias="RETRIEVAL_CANDIDATE_K")

    # ── Hybrid BM25 + dense (Phase 2) ──────────────────────────────────────
    enable_hybrid_bm25: bool = Field(default=True, alias="ENABLE_HYBRID_BM25")
    bm25_top_k: int = Field(default=15, alias="BM25_TOP_K")
    hybrid_rrf_k: int = Field(default=60, alias="HYBRID_RRF_K")
    bm25_storage_dir: Path = Field(default=PROJECT_ROOT / "storage" / "bm25")

    # ── Parent-document retrieval (Phase 2) ──────────────────────────────
    enable_parent_document_retrieval: bool = Field(
        default=True, alias="ENABLE_PARENT_DOCUMENT_RETRIEVAL"
    )

    # ── Reranker (post-retrieval precision) ────────────────────────────────
    # Single source of truth for the reranker; the API composition root
    # (backend/api/dependencies.py) reads these from here instead of its own
    # os.getenv defaults, so the live pipeline and the eval/legacy stack agree.
    # Base is the latency-safe default for CPU inference; set
    # RERANKER_MODEL=BAAI/bge-reranker-large for higher precision on dense legal
    # text at a few seconds/query. Defaults match the project's .env.
    enable_reranker: bool = Field(default=True, alias="ENABLE_RERANKER")
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base", alias="RERANKER_MODEL"
    )
    reranker_top_n: int = Field(default=5, alias="RERANKER_TOP_N")
    reranker_batch_size: int = Field(default=32, alias="RERANKER_BATCH_SIZE")
    reranker_device: str = Field(default="cpu", alias="RERANKER_DEVICE")
    # Drop chunks scoring below this fraction of the top reranker score
    enable_rerank_score_filter: bool = Field(default=True, alias="ENABLE_RERANK_SCORE_FILTER")
    rerank_min_score_ratio: float = Field(default=0.40, alias="RERANK_MIN_SCORE_RATIO")
    rerank_min_keep: int = Field(default=3, alias="RERANK_MIN_KEEP")

    # ── Conditional Reranking & Retrieval Caching (Qwen 2.5 7B) ──────────
    enable_conditional_reranking: bool = Field(default=True, alias="ENABLE_CONDITIONAL_RERANKING")
    conditional_reranker_threshold: float = Field(default=0.85, alias="CONDITIONAL_RERANKER_THRESHOLD")
    retrieval_cache_enabled: bool = Field(default=True, alias="RETRIEVAL_CACHE_ENABLED")
    retrieval_cache_ttl_seconds: int = Field(default=3600, alias="RETRIEVAL_CACHE_TTL_SECONDS")
    # Max concurrent sub-query retrievals. Sub-queries are independent, so they
    # run in a thread pool; each also fans dense/BM25 out concurrently. Keep this
    # modest to avoid oversubscribing CPU cores during embedding.
    retrieval_max_workers: int = Field(default=4, alias="RETRIEVAL_MAX_WORKERS")

    # ── Query rewrite (pre-retrieval) ──────────────────────────────────────
    # Disabled by default for fast single-turn; conditional for multi-turn follow-ups
    enable_query_rewrite: bool = Field(default=False, alias="ENABLE_QUERY_REWRITE")
    # LLM-based multi-query decomposition. When on and an LLM is available, one
    # LLM call splits comprehensive/list questions into focused sub-queries
    # (generalizes to any corpus); the keyword-table heuristic remains the
    # fallback. Only runs where multi-query is already enabled (not fast-path).
    enable_llm_multi_query: bool = Field(default=True, alias="ENABLE_LLM_MULTI_QUERY")

    # ── Dynamic Output Limits (Qwen 2.5 7B) ────────────────────────────────
    max_new_tokens_direct: int = Field(default=128, alias="MAX_NEW_TOKENS_DIRECT")
    max_new_tokens_factual: int = Field(default=256, alias="MAX_NEW_TOKENS_FACTUAL")
    max_new_tokens_technical: int = Field(default=384, alias="MAX_NEW_TOKENS_TECHNICAL")
    max_new_tokens_complex: int = Field(default=512, alias="MAX_NEW_TOKENS_COMPLEX")

    # ── Generation / faithfulness grounding ──────────────────────────────────
    # balanced (default): helpful synthesis + partial answers; strict: max faithfulness
    grounding_strictness: Literal["strict", "balanced"] = Field(
        default="balanced", alias="GROUNDING_STRICTNESS"
    )
    response_prompt_version: Literal["v1_standard", "v2_strict", "v2_balanced", "v3_qwen_compact"] = Field(
        default="v3_qwen_compact", alias="RESPONSE_PROMPT_VERSION"
    )
    # Legacy override — true forces strict mode regardless of GROUNDING_STRICTNESS
    strict_grounding: bool = Field(default=False, alias="STRICT_GROUNDING")
    enable_faithfulness_check: bool = Field(default=True, alias="ENABLE_FAITHFULNESS_CHECK")
    # strict guard: reject unless fully supported; balanced: reject only clear hallucinations
    faithfulness_guard_mode: Literal["strict", "balanced", "off"] = Field(
        default="balanced", alias="FAITHFULNESS_GUARD_MODE"
    )
    # keep: preserve answer on guard reject (relevancy-safe); trim: remove unsupported claims;
    # abstain: replace with insufficient-info (strict-audit only — collapses relevancy)
    faithfulness_guard_reject_action: Literal["keep", "trim", "abstain"] = Field(
        default="keep", alias="FAITHFULNESS_GUARD_REJECT_ACTION"
    )

    # ── Code validation (Phase 3) ──────────────────────────────────────────
    enable_code_validation: bool = Field(default=True, alias="ENABLE_CODE_VALIDATION")
    enable_code_self_correction: bool = Field(
        default=True, alias="ENABLE_CODE_SELF_CORRECTION"
    )
    code_self_correction_max_retries: int = Field(
        default=1, alias="CODE_SELF_CORRECTION_MAX_RETRIES"
    )
    code_validation_use_heuristic: bool = Field(
        default=True, alias="CODE_VALIDATION_USE_HEURISTIC"
    )
    code_validation_trigger_mode: Literal["answer_only", "answer_or_context"] = Field(
        default="answer_only", alias="CODE_VALIDATION_TRIGGER_MODE"
    )
    code_validation_fail_mode: Literal["strip_code", "fallback"] = Field(
        default="strip_code", alias="CODE_VALIDATION_FAIL_MODE"
    )
    code_validation_judge_mode: Literal["balanced", "strict"] = Field(
        default="balanced", alias="CODE_VALIDATION_JUDGE_MODE"
    )
    code_validation_heuristic_min_ratio: float = Field(
        default=0.8, alias="CODE_VALIDATION_HEURISTIC_MIN_RATIO"
    )

    # ── Corpus-scoped retrieval ────────────────────────────────────────────
    enable_corpus_scoped_retrieval: bool = Field(
        default=True, alias="ENABLE_CORPUS_SCOPED_RETRIEVAL"
    )

    # ── Agent ──────────────────────────────────────────────────────────────
    agent_max_iterations: int = Field(default=8)
    agent_verbose: bool = Field(default=True)

    # ── Conversation memory (short-term, per session) ──────────────────────
    # Enables natural follow-ups ("What about part-time?") without external stores.
    enable_conversation_memory: bool = Field(default=True, alias="ENABLE_CONVERSATION_MEMORY")
    memory_window_size: int = Field(default=5, alias="MEMORY_WINDOW_SIZE")  # turns (user+assistant pairs)
    memory_token_limit: int = Field(default=3000, alias="MEMORY_TOKEN_LIMIT")

    # ── Logging ────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )

    # ── Chat UI ────────────────────────────────────────────────────────────
    chainlit_port: int = Field(default=8000, alias="CHAINLIT_PORT")

    # ── Citation display (chat UI) ─────────────────────────────────────────
    # Citations are critical for trust in policy/legal RAG — keep configurable
    # so UX can evolve (inline vs sidebar, excerpts on/off) without code changes.
    show_citations: bool = Field(default=True, alias="SHOW_CITATIONS")
    citation_max_sources: int = Field(default=6, alias="CITATION_MAX_SOURCES")
    citation_show_excerpts: bool = Field(default=True, alias="CITATION_SHOW_EXCERPTS")
    citation_show_relevance_score: bool = Field(default=False, alias="CITATION_SHOW_SCORE")
    citation_dedupe: bool = Field(default=True, alias="CITATION_DEDUPE")
    # section_first: "II. GENERAL > 5.2 Vacation (p.14)"
    # document_first: "Employee Handbook.pdf — Section 5.2 (Page 14)"
    citation_format: Literal["section_first", "document_first"] = Field(
        default="section_first", alias="CITATION_FORMAT"
    )
    # Minimum reranker score (fraction of top chunk) to show a source when the
    # answer has no explicit [Source N] tags. Higher = fewer, more precise citations.
    citation_min_relevance_ratio: float = Field(
        default=0.55, alias="CITATION_MIN_RELEVANCE_RATIO"
    )
    # Log retrieved / filtered / displayed chunks for citation debugging.
    enable_citation_pipeline_logging: bool = Field(
        default=True, alias="ENABLE_CITATION_PIPELINE_LOGGING"
    )

    # ── Evaluation ───────────────────────────────────────────────────────
    # Golden-set eval is the quality gate before chunking / retrieval changes.
    eval_dataset_path: Path = Field(default=PROJECT_ROOT / "data" / "eval" / "golden_dataset.json")
    eval_guidebook_dataset_path: Path = Field(
        default=PROJECT_ROOT / "golden_dataset_guidebook.json",
        alias="EVAL_GUIDEBOOK_DATASET_PATH",
    )
    eval_corpus: Literal["all", "policy", "guidebook"] = Field(
        default="all", alias="EVAL_CORPUS"
    )
    eval_results_path: Path = Field(default=PROJECT_ROOT / "logs" / "evaluation_results.json")
    eval_llm_model: str = Field(default="qwen2.5:7b", alias="EVAL_LLM_MODEL")
    eval_max_samples: int = Field(default=0, alias="EVAL_MAX_SAMPLES")  # 0 = all cases
    eval_use_llm_judge: bool = Field(default=True, alias="EVAL_USE_LLM_JUDGE")

    # ── CI smoke eval gate (retrieval-only) ────────────────────────────────
    ci_smoke_dataset_path: Path = Field(
        default=PROJECT_ROOT / "data" / "eval" / "golden_subset_ci_smoke.json",
        alias="CI_SMOKE_DATASET_PATH",
    )
    ci_smoke_baseline_path: Path = Field(
        default=PROJECT_ROOT / "data" / "eval" / "ci_smoke_baseline.json",
        alias="CI_SMOKE_BASELINE_PATH",
    )
    ci_smoke_min_hit_rate: float = Field(default=0.75, alias="CI_SMOKE_MIN_HIT_RATE")
    ci_smoke_min_context_precision: float = Field(
        default=0.50, alias="CI_SMOKE_MIN_CONTEXT_PRECISION"
    )
    ci_smoke_min_context_recall: float = Field(
        default=0.55, alias="CI_SMOKE_MIN_CONTEXT_RECALL"
    )

    # ── PDF image extraction (citation visuals) ────────────────────────────
    enable_pdf_images: bool = Field(default=True, alias="ENABLE_PDF_IMAGES")
    pdf_image_min_px: int = Field(default=80, alias="PDF_IMAGE_MIN_PX")
    pdf_page_thumb_dpi: int = Field(default=120, alias="PDF_PAGE_THUMB_DPI")
    citation_max_page_images: int = Field(default=4, alias="CITATION_MAX_PAGE_IMAGES")

    # ── Comprehensive list retrieval ───────────────────────────────────────
    enable_comprehensive_retrieval: bool = Field(
        default=True, alias="ENABLE_COMPREHENSIVE_RETRIEVAL"
    )
    comprehensive_reranker_top_n: int = Field(
        default=12, alias="COMPREHENSIVE_RERANKER_TOP_N"
    )
    comprehensive_max_subqueries: int = Field(default=12, alias="COMPREHENSIVE_MAX_SUBQUERIES")

    # ── Code/tool retrieval boost ─────────────────────────────────────────
    enable_code_retrieval_boost: bool = Field(
        default=True, alias="ENABLE_CODE_RETRIEVAL_BOOST"
    )
    code_chunk_inject_min: int = Field(default=1, alias="CODE_CHUNK_INJECT_MIN")
    code_boost_score_multiplier: float = Field(
        default=1.2, alias="CODE_BOOST_SCORE_MULTIPLIER"
    )

    # ── Response language ──────────────────────────────────────────────────
    # english = always English (recommended for qwen2.5); match_query = follow user language
    response_language: Literal["english", "chinese", "match_query"] = Field(
        default="english", alias="RESPONSE_LANGUAGE"
    )

    # ── Document taxonomy ──────────────────────────────────────────────────
    # Maps top-level data subfolders → metadata document_type values
    folder_document_types: dict[str, str] = Field(
        default_factory=lambda: {
            "policies": "company_policy",
            "legal": "legal_document",
            "raw": "raw_backup",
        }
    )

    def ensure_directories(self) -> None:
        """Create runtime directories if they do not exist."""
        for path in (
            self.data_dir,
            self.policies_dir,
            self.legal_dir,
            self.raw_dir,
            self.storage_dir,
            self.pdf_images_dir,
            self.chroma_persist_dir,
            self.docstore_dir,
            self.bm25_storage_dir,
            self.logs_dir,
            self.eval_dataset_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
