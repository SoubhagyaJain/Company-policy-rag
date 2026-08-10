import re

with open("backend/rag/pipeline.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Imports
content = content.replace(
    "from src.ollama_client import stop_ollama_model",
    "from src.ollama_client import unload_model, preload_model"
)

# 2. AsyncModelManager definition
manager_code = """
class AsyncModelManager:
    \"\"\"
    Background worker that manages unloading and preloading models via a thread queue.
    Uses an active_readers counter and is_switching boolean gate to implement a non-blocking
    Reader-Writer lock. This ensures we never unload a model while active streams are reading from it.
    \"\"\"
    def __init__(self, initial_model: str):
        self.current_model = initial_model
        import queue
        self.queue: queue.Queue = queue.Queue()
        self.is_switching = False
        self.active_readers = 0
        self.reader_lock = threading.Lock()
        
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _worker(self):
        \"\"\"Background daemon processing model switch requests with Debounce.\"\"\"
        import queue
        while True:
            try:
                target_model = self.queue.get()
            except Exception:
                continue
            
            # DEBOUNCE: Drain the queue if user clicked rapidly, only process the latest
            try:
                while True:
                    target_model = self.queue.get_nowait()
                    self.queue.task_done()
            except queue.Empty:
                pass
                
            if target_model == self.current_model:
                self.queue.task_done()
                continue
                
            # Engage the is_switching gate to prevent new readers from starting (Writer Starvation)
            self.is_switching = True
            
            # Acquire Write Lock: wait until all active streams (readers) finish
            while True:
                with self.reader_lock:
                    if self.active_readers == 0:
                        break
                time.sleep(0.1)
                
            logger.info(f"Unloading '{self.current_model}' and preloading '{target_model}'")
            try:
                # Synchronous HTTP calls to Ollama
                unload_model(self.current_model)
                preload_model(target_model)
                self.current_model = target_model
                logger.info(f"Successfully switched active model to '{target_model}'")
            except Exception as e:
                logger.error(f"Error during model switch: {e}")
            finally:
                # Disengage the switching gate so waiting streams can proceed
                self.is_switching = False
                self.queue.task_done()

"""

content = content.replace("class _LLMProxy:\n", manager_code + "class _LLMProxy:\n")

# 3. RAGPipeline __init__
init_old = "self.active_model = getattr(self.llm, \"model\", None) or \"qwen2.5:7b\""
init_new = "self.model_manager = AsyncModelManager(initial_model=getattr(self.llm, \"model\", None) or \"qwen2.5:7b\")"
content = content.replace(init_old, init_new)

# 4. set_active_model
set_old = """    def set_active_model(self, model: str) -> str:
        \"\"\"Switch the global runtime model used by the backend pipeline.

        Best-effort: stop the prior Ollama model over the local RPC endpoint before
        re-pointing the LLM object at the new default. This is safe for the current API;
        failures are logged and downgraded so the new model can still be selected.
        \"\"\"
        model = (model or "").strip()
        if not model:
            raise ValueError("Model name cannot be empty.")

        previous_model = self.active_model or getattr(self.llm, "model", None) or "qwen2.5:7b"
        if previous_model and previous_model != model:
            try:
                base_url = getattr(self.llm, "base_url", None) if self.llm is not None else None
                stopped = stop_ollama_model(previous_model, base_url=base_url)
                if not stopped:
                    logger.warning("Previous Ollama model '%s' was not stopped cleanly; forcing runtime switch anyway.", previous_model)
            except Exception as exc:
                logger.warning("Previous model shutdown failed for '%s': %s", previous_model, exc)

        self.active_model = model
        if self.llm is not None:
            try:
                if hasattr(self.llm, "model"):
                    self.llm.model = model
                else:
                    logger.warning("LLM provider does not expose a .model attribute; active_model=%s", model)
            except Exception as exc:
                logger.warning("Unable to patch LLM provider model attribute: %s", exc)
                raise
        logger.info("Backend pipeline model switched to %s", model)
        return model"""

set_new = """    def set_active_model(self, model: str) -> str:
        \"\"\"Switch the backend pipeline to a new model.\"\"\"
        self.model_manager.queue.put(model)
        return model"""

content = content.replace(set_old, set_new)

# 5. get_active_model
get_old = """    def get_active_model(self) -> str:
        \"\"\"Return the currently configured generation model.\"\"\"
        if self.llm is not None:
            current = getattr(self.llm, "model", None)
            if current:
                self.active_model = str(current)
        return self.active_model or "qwen2.5:7b\""""

get_new = """    def get_active_model(self) -> str:
        \"\"\"Return the currently configured generation model.\"\"\"
        return self.model_manager.current_model"""

content = content.replace(get_old, get_new)

# 6. _get_effective_llm
eff_old = """        if self.llm is None:
            return None, model or "qwen2.5:7b"

        base_model = getattr(self.llm, "model", "qwen2.5:7b")"""

eff_new = """        if self.llm is None:
            return None, model or self.model_manager.current_model

        base_model = self.model_manager.current_model"""

content = content.replace(eff_old, eff_new)

# 7. queue cache write
cache_old = "self._queue_cache_write(user_query, answer_text, citations, model_name=selected_model)"
cache_new = "self._queue_cache_write(user_query, answer_text, citations, model_name=model_name)"
content = content.replace(cache_old, cache_new)

# 8. query
query_old = """    def query(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> RAGResponse:
        \"\"\"Execute end-to-end RAG pipeline and return structured RAGResponse with trace telemetry.\"\"\"
        total_start = time.perf_counter()
        stage_timings: dict[str, float] = {}

        req_llm, selected_model = self._get_effective_llm(model)"""

query_new = """    def query(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> RAGResponse:
        \"\"\"Execute end-to-end RAG pipeline and return structured RAGResponse with trace telemetry.\"\"\"
        while self.model_manager.is_switching:
            time.sleep(0.1)
            
        with self.model_manager.reader_lock:
            self.model_manager.active_readers += 1
            
        try:
            return self._query_internal(user_query, filters, history, model)
        finally:
            with self.model_manager.reader_lock:
                self.model_manager.active_readers -= 1

    def _query_internal(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> RAGResponse:
        total_start = time.perf_counter()
        stage_timings: dict[str, float] = {}

        req_llm, selected_model = self._get_effective_llm(model)"""

content = content.replace(query_old, query_new)

# 9. stream_query
stream_old = """    def stream_query(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        \"\"\"
        Streaming RAG pipeline: performs pre-rewrite cache lookup, or runs retrieval
        synchronously and yields real-time LLM tokens via llm.stream_complete().

        Yields dicts with 'type' key:
          - {'type': 'retrieval_done', 'stage_timings': {...}, 'candidate_count': int, ...}
          - {'type': 'token', 'content': str}
          - {'type': 'done', 'answer': str, 'citations': [...], 'trace': RAGTrace, ...}
        \"\"\"
        total_start = time.perf_counter()
        stage_timings: dict[str, float] = {}

        req_llm, selected_model = self._get_effective_llm(model)"""

stream_new = """    async def stream_query(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        model: str | None = None,
        cancel_token: Any = None,
    ):
        \"\"\"
        Streaming RAG pipeline: performs pre-rewrite cache lookup, or runs retrieval
        synchronously and yields real-time LLM tokens via llm.stream_complete().

        Yields dicts with 'type' key:
          - {'type': 'retrieval_done', 'stage_timings': {...}, 'candidate_count': int, ...}
          - {'type': 'token', 'content': str}
          - {'type': 'done', 'answer': str, 'citations': [...], 'trace': RAGTrace, ...}
        \"\"\"
        import asyncio
        while self.model_manager.is_switching:
            await asyncio.sleep(0.1)
            
        with self.model_manager.reader_lock:
            self.model_manager.active_readers += 1
            
        try:
            # yield from does not work in async generator, so we iterate
            for chunk in self._stream_query_internal(user_query, filters, history, model):
                yield chunk
        finally:
            with self.model_manager.reader_lock:
                self.model_manager.active_readers -= 1

    def _stream_query_internal(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        total_start = time.perf_counter()
        stage_timings: dict[str, float] = {}

        req_llm, selected_model = self._get_effective_llm(model)"""

content = content.replace(stream_old, stream_new)


with open("backend/rag/pipeline.py", "w", encoding="utf-8") as f:
    f.write(content)
