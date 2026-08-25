# -*- coding: utf-8 -*-
"""
Complete Data Repository of 100 Technical Interview Questions & Answers with Code Snippets
"""

ALL_MODULES = [
    {
        "id": "mod1",
        "title": "Module 1: High-Level Architecture & System Design",
        "badge": "Q01–Q10",
        "questions": [
            {
                "num": "Q01",
                "level": "1",
                "level_text": "L1 Fundamental",
                "q": "Walk me through the complete end-to-end lifecycle of a query from HTTP request to SSE stream completion.",
                "short": "The query hits POST /api/chat/stream, where ChatService retrieves session history from TTLCache. QueryRouter classifies intent (~0.5ms). Semantic cache probes ChromaDB (cosine >= 0.95). On miss, query rewrite resolves pronouns and metadata filters are inferred. Parallel dense vector + BM25 search execute and fuse via RRF (k=60). Top 30 chunks are scored by bge-reranker-large on CUDA with relative threshold filtering, then expanded to parent documents. Ollama generates grounded tokens while 4D Verifier evaluates composite quality. If composite < 0.70, RetryEngine re-executes. Verified tokens stream via SSE.",
                "deep": "1. Entry: FastAPI validates Pydantic ChatRequest at `backend/api/routes/chat.py` with an asyncio.Event cancel token.\n2. Session: ChatService retrieves conversation history from thread-safe TTLCache (1000 sessions, 24h TTL).\n3. Intent: QueryRouter runs compiled regex to classify into 5 categories, returning dynamic RetrievalStrategy hyperparameters.\n4. Cache: SemanticCacheManager embeds query, probes ChromaDB semantic_cache. Cosine >= 0.95 exits early with simulated streaming.\n5. Rewrite & Filter: QueryRewriter resolves pronouns; FilterExtractor extracts department and policy tags.\n6. Hybrid Retrieval: ChromaDB HNSW (bge-small-en-v1.5) + BM25 retrieve candidates, merged via Reciprocal Rank Fusion: Score(d) = Σ 1/(60+rank).\n7. Rerank: CrossEncoderReranker (bge-reranker-large) re-scores top 30 chunks; RelativeScoreThreshold drops items below top_score * 0.45.\n8. Expand: ContextCompressor swaps 480-token children for 2000-token parent documents from docstore.\n9. Synthesize & Verify: Ollama streams tokens with GROUNDED_SYSTEM_PROMPT. SelfReflectionVerifier computes 4D score (35% Faithfulness, 30% Completeness, 20% Citation, 15% Coherence). If score < 0.70, RetryEngine refines retrieval and re-prompts.\n10. Delivery: EventSource response streams to client; background thread writes answer to semantic cache.",
                "code": "backend/rag/pipeline.py:259 (_query_internal), backend/api/routes/chat.py:45",
                "file": "backend/rag/pipeline.py",
                "lang": "python",
                "snippet": """async def _query_internal(self, query: str, session_history: List[ChatMessage]) -> RAGResponse:
    # Step 1: Classify intent & assign dynamic retrieval strategy
    intent = self.router.classify(query)
    strategy = intent.strategy

    # Step 2: Probe semantic cache (cosine distance <= 0.05)
    cached = await self.semantic_cache.get(query)
    if cached and cached.similarity >= 0.95:
        return cached.to_response()

    # Step 3: Multi-turn query rewrite & metadata filter extraction
    rewritten_query = self.rewriter.rewrite(query, session_history)
    inferred_filters = self.filter_extractor.extract(rewritten_query)

    # Step 4: Hybrid retrieval (Dense + BM25) with RRF fusion
    candidates = await self.retriever.retrieve(
        query=rewritten_query, strategy=strategy, filters=inferred_filters
    )

    # Step 5: Cross-Encoder reranking & relative thresholding
    reranked_chunks = self.reranker.rerank(rewritten_query, candidates, top_n=strategy.rerank_top_n)
    filtered_chunks = self.threshold_postprocessor.process(reranked_chunks)

    # Step 6: Context expansion (480-tok child -> 2000-tok parent section)
    expanded_context = self.compressor.expand_context(filtered_chunks)

    # Step 7: Grounded LLM Generation & 4D Verification Loop
    answer, verifier_report = await self._generate_and_verify(
        query=rewritten_query, context=expanded_context, strategy=strategy
    )
    return RAGResponse(answer=answer, citations=verifier_report.citations, trace=self.get_trace())"""
            },
            {
                "num": "Q02",
                "level": "2",
                "level_text": "L2 Architecture",
                "q": "What is the high-level architecture of this system and why is it structured as an Agentic RAG rather than naive RAG?",
                "short": "Naive RAG follows a rigid, brittle pipeline (Query -> Embed -> Vector DB -> LLM) which fails on complex compliance policies due to semantic drift, lost clause context, and unchecked hallucinations. Our Agentic architecture introduces intelligent routing, dynamic parameter tuning, semantic caching, filter inference, cross-encoder thresholding, and a closed-loop 4D verification + retry engine.",
                "deep": "In enterprise policy compliance, accuracy must be absolute. Naive RAG fails in three distinct ways:\n1. Semantic Blindness: Vector search alone misses exact keywords like 'POL-402' or '$1,500'. Hybrid search fixes this.\n2. Context Truncation: Fixed chunks split conditional clauses across boundaries. Child-parent hierarchical chunking ensures full clause retention.\n3. Open-Loop Hallucination: Standard RAG has no mechanism to verify if the LLM's answer is factually grounded. Our 4D Self-Reflection Verifier acts as an automated heuristic evaluator that checks numerical claims, citations, and completeness, triggering autonomous parameter adjustments (broadening search, tightening reranking) via RetryEngine if quality falls below 0.70.",
                "code": "backend/rag/pipeline.py:135 (RAGPipeline.__init__), backend/rag/verifier.py:210",
                "file": "backend/rag/pipeline.py",
                "lang": "python",
                "snippet": """class RAGPipeline:
    def __init__(
        self,
        router: QueryRouter,
        semantic_cache: SemanticCacheManager,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        compressor: ContextCompressor,
        verifier: SelfReflectionVerifier,
        retry_engine: RetryEngine,
        llm: _LLMProxy,
    ):
        self.router = router
        self.semantic_cache = semantic_cache
        self.retriever = retriever
        self.reranker = reranker
        self.compressor = compressor
        self.verifier = verifier
        self.retry_engine = retry_engine
        self.llm = llm"""
            },
            {
                "num": "Q03",
                "level": "2",
                "level_text": "L2 Integration",
                "q": "How does the FastAPI backend interface with Ollama, ChromaDB, and the Next.js frontend?",
                "short": "FastAPI acts as the central asynchronous orchestrator. It receives SSE requests from Next.js, queries ChromaDB via local PersistentClient for vector embeddings and semantic caching, invokes rank-bm25 for lexical search, runs cross-encoders on local PyTorch CUDA, and streams token completions from Ollama via async HTTP client.",
                "deep": "1. Next.js to FastAPI: Communicates over HTTP SSE (POST /api/chat/stream) using Fetch EventSource parser with custom session headers.\n2. FastAPI to ChromaDB: Uses `chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)` for local SQLite/HNSW vector storage with embedded bge-small embeddings.\n3. FastAPI to Ollama: Connects via httpx/Ollama API (`http://localhost:11434/api/generate`) with stream=True, consuming ndjson chunks and relaying them to the client SSE stream.\n4. FastAPI to Cross-Encoder: Direct in-process PyTorch model loaded on CUDA device with singleton thread locks.",
                "code": "backend/api/routes/chat.py, backend/rag/pipeline.py:720 (stream_query)",
                "file": "backend/api/routes/chat.py",
                "lang": "python",
                "snippet": """@router.post("/stream")
async def stream_chat(
    request: Request,
    payload: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    cancel_token = asyncio.Event()

    async def event_generator():
        async for chunk in chat_service.stream_query(
            message=payload.message,
            session_id=payload.session_id,
            model=payload.model,
            cancel_token=cancel_token
        ):
            if await request.is_disconnected():
                cancel_token.set()
                break
            yield f"data: {chunk.model_dump_json()}\\n\\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )"""
            },
            {
                "num": "Q04",
                "level": "2",
                "level_text": "L2 State & Concurrency",
                "q": "What is the role of ChatService and how does it manage conversation session state?",
                "short": "ChatService manages session lifecycle, TTL-based memory eviction, history truncation, and coordinates query execution between REST controllers and the RAGPipeline.",
                "deep": "ChatService uses cachetools `TTLCache(maxsize=1000, ttl=86400)` to store conversation history per session_id for 24 hours. A threading `Lock` guards cache read/write operations against race conditions during concurrent requests. When a query arrives, ChatService retrieves the last 5 turns (max 3000 tokens), passes them to QueryRewriter for coreference resolution, executes the pipeline, appends user query and verified assistant response to the session history, and records query telemetry in an audit log.",
                "code": "backend/services/chat_service.py:45 (ChatService), backend/services/chat_service.py:110",
                "file": "backend/services/chat_service.py",
                "lang": "python",
                "snippet": """class ChatService:
    def __init__(self, pipeline: RAGPipeline, max_sessions: int = 1000, ttl: int = 86400):
        self.pipeline = pipeline
        self._sessions: TTLCache[str, List[ChatMessage]] = TTLCache(maxsize=max_sessions, ttl=ttl)
        self._lock = threading.Lock()

    def get_history(self, session_id: str, max_turns: int = 5) -> List[ChatMessage]:
        with self._lock:
            history = self._sessions.get(session_id, [])
            return history[-max_turns * 2:]"""
            },
            {
                "num": "Q05",
                "level": "2",
                "level_text": "L2 Protocol Design",
                "q": "Why did you choose Server-Sent Events (SSE) over WebSockets or polling for response streaming?",
                "short": "SSE provides native unidirectional text streaming over standard HTTP/1.1 and HTTP/2 with built-in reconnection, minimal protocol overhead, full firewall/proxy compatibility, and simple browser EventSource integration — perfectly matching LLM token generation dynamics.",
                "deep": "1. Unidirectional vs Bidirectional: LLM query-response is strictly client request -> server stream. WebSockets introduce unnecessary bidirectional TCP framing overhead, stateful socket server complexity, and connection keep-alive management.\n2. Proxy & Firewall Friendly: SSE runs over standard HTTPS (port 443) with `Content-Type: text/event-stream` and `Transfer-Encoding: chunked`, traversing corporate firewalls and ALB/Nginx proxies without special WebSocket upgrade handshakes.\n3. Built-in Client Reconnection: The EventSource standard supports automatic reconnection and event IDs.\n4. Low Latency TTFT: Tokens are flushed immediately upon generation with zero buffer delay.",
                "code": "backend/api/routes/chat.py:55 (StreamingResponse text/event-stream)",
                "file": "backend/api/routes/chat.py",
                "lang": "python",
                "snippet": """return StreamingResponse(
    event_generator(),
    media_type="text/event-stream",
    headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Content-Encoding": "none"
    }
)"""
            },
            {
                "num": "Q06",
                "level": "3",
                "level_text": "L3 Edge Cases",
                "q": "How does the system handle client disconnects during active generation?",
                "short": "FastAPI's Request object exposes `is_disconnected()`. An `asyncio.Event` cancel token is checked in the streaming token generator loop; if disconnected, generation terminates immediately to prevent wasting GPU VRAM and CPU cycles.",
                "deep": "In `backend/api/routes/chat.py`, the SSE generator loop periodically polls `await request.is_disconnected()`. If True, it signals `cancel_token.set()`. The underlying Ollama streaming consumer checks `cancel_token.is_set()` before processing each token chunk; when set, it closes the HTTP response stream and aborts synthesis. This prevents orphaned LLM generation tasks from consuming GPU memory and thread capacity when users navigate away or close browser tabs.",
                "code": "backend/api/routes/chat.py:68, backend/rag/pipeline.py:750",
                "file": "backend/rag/pipeline.py",
                "lang": "python",
                "snippet": """async def _stream_llm_tokens(self, prompt: str, cancel_token: asyncio.Event):
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", f"{self.ollama_url}/api/generate", json={"prompt": prompt}) as response:
            async for line in response.aiter_lines():
                if cancel_token.is_set():
                    logger.info("Client disconnected. Aborting generation loop.")
                    break
                if line:
                    chunk = json.loads(line)
                    yield chunk.get("response", "")"""
            },
            {
                "num": "Q07",
                "level": "3",
                "level_text": "L3 Concurrency Safety",
                "q": "How does _LLMProxy prevent race conditions when multiple concurrent requests select different LLM models?",
                "short": "In async environments, mutating a shared LLM instance's model attribute causes race conditions where concurrent requests get responses from the wrong model. `_LLMProxy` encapsulates per-request model routing with thread-safe instantiation and caching.",
                "deep": "If request A requests `qwen2.5` and request B requests `llama3.1` concurrently, setting `self.llm.model = model_name` on a shared singleton leads to a race condition during `await` suspension. `_LLMProxy` wraps LLM access: it maintains an internal model instance dictionary guarded by `threading.Lock()`. When a request executes, `_LLMProxy` provides a thread-isolated view of the requested model without modifying shared state, ensuring 100% deterministic model selection under high concurrency.",
                "code": "backend/rag/pipeline.py:64 (_LLMProxy class)",
                "file": "backend/rag/pipeline.py",
                "lang": "python",
                "snippet": """class _LLMProxy:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self._instances: Dict[str, OllamaClient] = {}
        self._lock = threading.Lock()

    def get_client(self, model_name: str) -> OllamaClient:
        with self._lock:
            if model_name not in self._instances:
                self._instances[model_name] = OllamaClient(
                    base_url=self.base_url, model=model_name
                )
            return self._instances[model_name]"""
            },
            {
                "num": "Q08",
                "level": "3",
                "level_text": "L3 Latency Profiling",
                "q": "What are the primary latency bottlenecks across the 10 pipeline stages, and what are their typical runtimes?",
                "short": "Total latency for cache miss ranges from 800ms to 2.2s. The primary bottlenecks are LLM Token Generation (~600–1500ms) and Cross-Encoder Reranking (~85ms). All routing, cache probing, and verification run in under 20ms.",
                "deep": "1. Query Routing: ~0.5ms (compiled regex)\n2. Semantic Cache Probe: ~8ms (embedding + ChromaDB distance lookup)\n3. Query Rewrite & Filter Extraction: ~15ms (history formatting + regex extraction)\n4. Hybrid Retrieval: ~40ms (parallel ChromaDB HNSW + BM25 lookup)\n5. RRF Fusion: ~1.5ms (pure algorithmic rank calculation)\n6. Cross-Encoder Reranking: ~85ms (PyTorch CUDA batch inference across 30 pairs)\n7. Parent Chunk Expansion: ~5ms (docstore in-memory / SQLite map)\n8. LLM Time to First Token (TTFT): ~250–450ms; Total Generation: ~600–1500ms\n9. 4D Heuristic Verification: ~2ms (regex checks, token overlap, math validation)\n10. Semantic Cache Write: 0ms (executed in detached background daemon thread)",
                "code": "backend/rag/pipeline.py:310 (timing logs and RAGTrace)",
                "file": "backend/rag/pipeline.py",
                "lang": "python",
                "snippet": """trace = {
    "router_time_ms": round((t1 - t0) * 1000, 2),        # ~0.5ms
    "cache_probe_time_ms": round((t2 - t1) * 1000, 2),   # ~8.2ms
    "retrieval_time_ms": round((t3 - t2) * 1000, 2),     # ~41.5ms
    "reranking_time_ms": round((t4 - t3) * 1000, 2),     # ~85.3ms (CUDA)
    "expansion_time_ms": round((t5 - t4) * 1000, 2),     # ~4.8ms
    "generation_time_ms": round((t6 - t5) * 1000, 2),    # ~850.0ms
    "verification_time_ms": round((t7 - t6) * 1000, 2),  # ~1.8ms
    "total_latency_ms": round((t7 - t0) * 1000, 2)       # ~992.1ms
}"""
            },
            {
                "num": "Q09",
                "level": "2",
                "level_text": "L2 Project Layout",
                "q": "How is the codebase structured between company_policy_rag/src/ and company_policy_rag/backend/?",
                "short": "`src/` contains core ML engines, fine-tuning scripts, GGUF export tools, and global Pydantic Settings. `backend/` contains the web service layer: FastAPI REST routes, Celery background tasks, session services, and API DTOs.",
                "deep": "The architecture separates domain logic from service delivery:\n- `company_policy_rag/src/config.py`: Single Source of Truth Settings defining 60+ hyperparameters.\n- `company_policy_rag/src/finetuning/`: Offline ML pipelines (trainer, dataset loader, merger, GGUF exporter, Ollama registrar).\n- `company_policy_rag/backend/rag/`: The runtime RAG pipeline, verifier, retry engine, router, and cache manager.\n- `company_policy_rag/backend/retrieval/`: Hybrid dense + sparse retrievers and cross-encoder rerankers.\n- `company_policy_rag/backend/api/` & `backend/services/`: FastAPI routers, SSE endpoints, and session management.\nThis ensures ML training scripts can be run standalone without spinning up web server dependencies.",
                "code": "company_policy_rag/src/config.py, company_policy_rag/backend/main.py",
                "file": "company_policy_rag/src/config.py",
                "lang": "python",
                "snippet": """class Settings(BaseSettings):
    CHUNK_SIZE: int = 480
    CHUNK_OVERLAP: int = 64
    PARENT_CHUNK_SIZE: int = 2000
    DENSE_TOP_K: int = 15
    BM25_TOP_K: int = 15
    RERANK_TOP_N: int = 4
    MIN_SCORE_RATIO: float = 0.45
    RRF_K: int = 60
    VERIFIER_PASS_THRESHOLD: float = 0.70
    MIN_FAITHFULNESS_SCORE: float = 0.65"""
            },
            {
                "num": "Q10",
                "level": "3",
                "level_text": "L3 Fault Tolerance",
                "q": "If the Ollama service crashes or becomes unreachable during execution, how does the system recover?",
                "short": "The pipeline catches connection and timeout exceptions from Ollama and automatically triggers `_fallback_synthesis()`, generating a deterministic, extractive summary directly from the top reranked chunks with citations.",
                "deep": "In `backend/rag/pipeline.py`, Ollama calls are wrapped in `try/except (httpx.ConnectError, httpx.TimeoutException, Exception)`. When triggered, `_fallback_synthesis()` takes over: it extracts the most salient sentences (first 2 sentences of each verified reranked chunk), structures them into bullet points, appends source citations `[Source N]`, and returns a guaranteed 100% factually grounded answer with a system warning in `RAGTrace`. This guarantees that system outages never cause complete request failures or unhandled 500 errors.",
                "code": "backend/rag/pipeline.py:512 (_fallback_synthesis)",
                "file": "backend/rag/pipeline.py",
                "lang": "python",
                "snippet": """def _fallback_synthesis(self, chunks: List[ScoredChunk]) -> str:
    logger.warning("LLM unreachable. Initiating deterministic extractive fallback synthesis.")
    bullet_points = []
    for idx, chunk in enumerate(chunks[:3], 1):
        sentences = re.split(r'(?<=[.!?])\\s+', chunk.text.strip())[:2]
        extracted_text = " ".join(sentences)
        bullet_points.append(f"• {extracted_text} [Source {idx}]")
    
    return (
        "*(Direct Source Extract — LLM Service Offline)*\\n\\n"
        + "\\n".join(bullet_points)
    )"""
            }
        ]
    }
]
