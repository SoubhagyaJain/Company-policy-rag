# Project: Celery & Redis Async RAG Offloading and SSE Streaming

## Architecture
The system offloads heavy RAG processing (retrieval, cross-encoder reranking, LLM token generation) from the main FastAPI web server process to an asynchronous Celery worker process backed by Redis.

```
[ Frontend Chat UI ] ──(HTTP POST)──> [ FastAPI Server (/api/chat/stream) ]
                                              │ (1. Enqueue Task)
                                              ▼
                                      [ Redis Broker ]
                                              │ (2. Pick up Task)
                                              ▼
                                    [ Celery RAG Worker ]
                                              │ (3. Execute RAG Pipeline)
                                              │ (4. Publish SSE Chunks)
                                              ▼
                                    [ Redis Pub/Sub Channel ]
                                              │ (5. Subscribe & Stream)
                                              ▼
[ Frontend Chat UI ] <──(SSE Stream)── [ FastAPI Server (/api/chat/stream) ]
```

- **Broker & Cache**: Redis on `redis://localhost:6379/0` (or `rag-redis` in Docker).
- **Worker**: Celery background worker running `backend/tasks/celery_app.py` executing `RAGPipeline.stream_query()`.
- **Publisher**: Worker publishes JSON-encoded SSE event dicts to channel `rag:stream:{task_id}`.
- **Subscriber**: FastAPI `/api/chat/stream` subscribes asynchronously to `rag:stream:{task_id}` via `redis.asyncio` client and yields SSE formatted text to the client.

## Code Layout
- `company_policy_rag/backend/tasks/celery_app.py` — Celery app initialization, Redis broker configuration.
- `company_policy_rag/backend/tasks/rag_tasks.py` — Celery background tasks (`run_rag_task`, `stream_rag_task`).
- `company_policy_rag/backend/utils/redis_client.py` — Shared async Redis connection pool and Pub/Sub helpers.
- `company_policy_rag/backend/api/routes/chat.py` — Endpoint `/api/chat/stream` updated for async task dispatch & Redis Pub/Sub SSE streaming.
- `company_policy_rag/backend/services/chat_service.py` — Service layer integration with Celery tasks.
- `company_policy_rag/pyproject.toml` & `requirements.txt` — Dependency declarations (`celery`, `redis`).
- `company_policy_rag/docker-compose.yml` & `.env.example` — Celery worker container service & Redis config.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Celery & Redis Dependencies | Add `celery` and `redis` to `pyproject.toml`, `requirements.txt`, and `requirements-docker.txt` | M1 | R1 |
| 2 | Celery App & Worker Setup | Implement `backend/tasks/celery_app.py` with Redis broker URL configuration and healthcheck | M1 | R1 |
| 3 | Docker & Environment Configuration | Configure `celery_worker` service in `docker-compose.yml` and add `CELERY_BROKER_URL` in `.env.example` | M1 | R1 |
| 4 | Asynchronous RAG Task Offloading | Wrap `RAGPipeline` / `ChatService` execution into Celery background tasks in `backend/tasks/rag_tasks.py` | M2 | R2 |
| 5 | Non-Blocking FastAPI Enqueuing | Update `/api/chat` and `/api/chat/stream` to enqueue background Celery task without blocking the web request lifecycle | M2 | R2 |
| 6 | Redis Pub/Sub Token Publishing | Celery task publishes token chunks and SSE events (`start`, `retrieval`, `chunk`, `citation`, `trace`, `done`) to channel `rag:stream:{task_id}` | M3 | R3 |
| 7 | Redis Pub/Sub SSE Receiver | Update FastAPI `/api/chat/stream` to subscribe to `rag:stream:{task_id}` and stream Server-Sent Events to the client | M3 | R3 |
| 8 | Frontend Compatibility & SSE Schema | Ensure exact preservation of SSE payload structure and event sequence required by existing frontend and test suite | M3 | R3 |
| 9 | Task Error Handling & Fallbacks | Handle worker disconnects, Redis timeouts, task failure events, and return error SSE payload gracefully | M3 | R3 |
| 10| Full E2E & Hardened Testing | Pass 100% of E2E test suite (Tiers 1-4) and complete Tier 5 white-box adversarial coverage hardening | M-Final | Final |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Celery & Redis Setup | Features 1, 2, 3: Celery app init, Redis broker setup, dependency updates, Docker config | none | PLANNED |
| M2 | Async Task Offloading | Features 4, 5: Celery background tasks for RAG processing, FastAPI async task dispatch | M1 | PLANNED |
| M3 | Redis Pub/Sub SSE Streaming | Features 6, 7, 8, 9: Redis Pub/Sub publishing, FastAPI SSE stream subscription, error handling | M2 | PLANNED |
| M-Final | E2E Testing & Hardening | Feature 10: Complete 100% E2E test suite pass (Tiers 1-4) + Tier 5 adversarial hardening | M3 | PLANNED |

## Interface Contracts

### Celery Task Signature
`stream_rag_task(task_id: str, query: str, session_id: str, model_name: str | None = None, kb_version: str | None = None)`
- **Return**: `dict` containing final response summary (`task_id`, `status`, `citations`, `answer_length`).
- **Side Effect**: Publishes JSON payloads to Redis Pub/Sub channel `rag:stream:{task_id}`.

### Redis Pub/Sub Channel Schema
Channel Name: `rag:stream:{task_id}`
Message Format: JSON string representing SSE event object:
```json
{
  "event": "chunk", // "start" | "retrieval" | "chunk" | "citation" | "trace" | "done" | "error"
  "data": {
    "text": "token text",
    "finish_reason": null
  }
}
```

### FastAPI SSE Endpoint
URL: `/api/chat/stream`
HTTP Method: POST
Request Body: `ChatRequest` (`query`, `session_id`, `model_name`, `kb_version`)
Response: `StreamingResponse(sse_generator(), media_type="text/event-stream")`
Headers: `Cache-Control: no-cache`, `Connection: keep-alive`, `X-Accel-Buffering: no`
