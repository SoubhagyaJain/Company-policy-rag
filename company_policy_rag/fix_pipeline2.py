with open("backend/rag/pipeline.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. AsyncModelManager
content = content.replace("""class AsyncModelManager:
    \"\"\"
    Background worker that manages unloading and preloading models via an async queue.
    Uses RWLocks to ensure we never unload a model while active streams are reading from it.
    \"\"\"
    def __init__(self, initial_model: str):
        self.current_model = initial_model
        self.queue: asyncio.Queue = asyncio.Queue()
        self.rwlock = aiorwlock.RWLock()
        self.is_switching = False
        
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._worker())

    async def _worker(self):
        \"\"\"Background daemon processing model switch requests with Debounce.\"\"\"
        while True:
            target_model = await self.queue.get()
            
            # DEBOUNCE: Drain the queue if user clicked rapidly, only process the latest
            while not self.queue.empty():
                target_model = self.queue.get_nowait()
                self.queue.task_done()
                
            if target_model == self.current_model:
                self.queue.task_done()
                continue
                
            # Engage the is_switching gate to prevent Read Lock starvation
            self.is_switching = True
            
            # Acquire Write Lock: wait until all active streams (Read Locks) finish
            async with self.rwlock.writer_lock:
                logger.info(f"Unloading '{self.current_model}' and preloading '{target_model}'")
                try:
                    # Run synchronous HTTP calls to Ollama in a threadpool so we don't block the async event loop
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, unload_model, self.current_model)
                    await loop.run_in_executor(None, preload_model, target_model)
                    self.current_model = target_model
                    logger.info(f"Successfully switched active model to '{target_model}'")
                except Exception as e:
                    logger.error(f"Error during async model switch: {e}")
                
            # Disengage the switching gate so waiting streams can proceed
            self.is_switching = False
            self.queue.task_done()""", """class AsyncModelManager:
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
                self.queue.task_done()""")

# 2. set_active_model put_nowait to put
content = content.replace("self.model_manager.queue.put_nowait(model)", "self.model_manager.queue.put(model)")

# 3. query
old_query = """    def query(
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

new_query = """    def query(
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

content = content.replace(old_query, new_query)

# 4. stream_query
old_stream = """    async def stream_query(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        model: str | None = None,
        cancel_token: asyncio.Event | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
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

new_stream = """    async def stream_query(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        model: str | None = None,
        cancel_token: asyncio.Event | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        \"\"\"
        Streaming RAG pipeline: performs pre-rewrite cache lookup, or runs retrieval
        synchronously and yields real-time LLM tokens via llm.stream_complete().

        Yields dicts with 'type' key:
          - {'type': 'retrieval_done', 'stage_timings': {...}, 'candidate_count': int, ...}
          - {'type': 'token', 'content': str}
          - {'type': 'done', 'answer': str, 'citations': [...], 'trace': RAGTrace, ...}
        \"\"\"
        while self.model_manager.is_switching:
            await asyncio.sleep(0.1)
            
        with self.model_manager.reader_lock:
            self.model_manager.active_readers += 1
            
        try:
            async for chunk in self._stream_query_internal(user_query, filters, history, model, cancel_token):
                yield chunk
        finally:
            with self.model_manager.reader_lock:
                self.model_manager.active_readers -= 1

    async def _stream_query_internal(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        model: str | None = None,
        cancel_token: asyncio.Event | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        total_start = time.perf_counter()
        stage_timings: dict[str, float] = {}

        req_llm, selected_model = self._get_effective_llm(model)"""
content = content.replace(old_stream, new_stream)

# 5. Remove async with rwlock from stream_query
old_lock = """        # 6. LLM Grounded Answer Synthesis (Streaming)
        async with self.model_manager.rwlock.reader_lock:
            t0 = time.perf_counter()"""

new_lock = """        # 6. LLM Grounded Answer Synthesis (Streaming)
        t0 = time.perf_counter()"""
content = content.replace(old_lock, new_lock)

with open("backend/rag/pipeline.py", "w", encoding="utf-8") as f:
    f.write(content)
