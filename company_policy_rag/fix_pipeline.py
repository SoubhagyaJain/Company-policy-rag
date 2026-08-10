import re

with open("backend/rag/pipeline.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Replace AsyncModelManager
old_manager = """class AsyncModelManager:
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
            self.queue.task_done()"""

new_manager = """class AsyncModelManager:
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
                self.queue.task_done()"""
content = content.replace(old_manager, new_manager)

# 2. Update set_active_model
old_set_active = """    def set_active_model(self, model: str) -> str:
        \"\"\"Switch the backend pipeline to a new model.\"\"\"
        self.model_manager.queue.put_nowait(model)
        return model"""

new_set_active = """    def set_active_model(self, model: str) -> str:
        \"\"\"Switch the backend pipeline to a new model.\"\"\"
        self.model_manager.queue.put(model)
        return model"""
content = content.replace(old_set_active, new_set_active)

# 3. Update stream_query (remove async with rwlock)
old_stream = """        # 0. Pre-rewrite Cache Lookup
        cache_enabled = getattr(self.semantic_cache.settings, "semantic_cache_enabled", True) if (self.semantic_cache and hasattr(self.semantic_cache, "settings")) else True
        if cache_enabled and self.semantic_cache is not None:"""

new_stream = """        while self.model_manager.is_switching:
            await asyncio.sleep(0.1)
            
        with self.model_manager.reader_lock:
            self.model_manager.active_readers += 1
            
        try:
            async for chunk in self._stream_query_internal(user_query, filters, history, selected_model, req_llm, cancel_token):
                yield chunk
        finally:
            with self.model_manager.reader_lock:
                self.model_manager.active_readers -= 1

    async def _stream_query_internal(
        self,
        user_query: str,
        filters: dict[str, Any] | None,
        history: list[dict[str, Any]] | None,
        selected_model: str,
        req_llm: Any,
        cancel_token: asyncio.Event | None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        total_start = time.perf_counter()
        stage_timings: dict[str, float] = {}

        # 0. Pre-rewrite Cache Lookup
        cache_enabled = getattr(self.semantic_cache.settings, "semantic_cache_enabled", True) if (self.semantic_cache and hasattr(self.semantic_cache, "settings")) else True
        if cache_enabled and self.semantic_cache is not None:"""
content = content.replace(old_stream, new_stream)

# 4. Remove rwlock from stream_query
old_rwlock = """        async with self.model_manager.rwlock.reader_lock:
            # 6. LLM Grounded Answer Synthesis (Streaming)"""
new_rwlock = """        # 6. LLM Grounded Answer Synthesis (Streaming)"""
content = content.replace(old_rwlock, new_rwlock)

# 5. Fix query (synchronous lock check)
old_query = """        req_llm, selected_model = self._get_effective_llm(model)

        # 0. Pre-rewrite Cache Lookup"""

new_query = """        while self.model_manager.is_switching:
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
        filters: dict[str, Any] | None,
        history: list[dict[str, Any]] | None,
        model: str | None,
    ) -> RAGResponse:
        total_start = time.perf_counter()
        stage_timings: dict[str, float] = {}
        req_llm, selected_model = self._get_effective_llm(model)

        # 0. Pre-rewrite Cache Lookup"""
content = content.replace(old_query, new_query)

with open("backend/rag/pipeline.py", "w", encoding="utf-8") as f:
    f.write(content)
