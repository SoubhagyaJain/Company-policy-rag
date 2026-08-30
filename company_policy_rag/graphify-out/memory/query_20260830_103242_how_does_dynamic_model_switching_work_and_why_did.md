---
type: "query"
date: "2026-08-30T10:32:42.697970+00:00"
question: "How does dynamic model switching work and why did the selector revert?"
contributor: "graphify"
source_nodes: ["frontend/components/ChatWindow.tsx", "frontend/lib/api-client.ts", "backend/api/routes/models.py", "backend/services/chat_service.py", "backend/rag/pipeline.py"]
---

# Q: How does dynamic model switching work and why did the selector revert?

## Answer

The frontend model loader depended on selectedModel, so an optimistic selection retriggered GET /api/models before POST /api/models/select finished and restored the stale active model. The fix makes initial model loading stable, awaits backend acknowledgement, rolls back with a visible error on failure, records the actual response model, and uses isolated cached LLM client copies per model with a lock-protected fallback. Requests that omit model now use the active backend model, while explicit per-request choices remain supported.

## Source Nodes

- frontend/components/ChatWindow.tsx
- frontend/lib/api-client.ts
- backend/api/routes/models.py
- backend/services/chat_service.py
- backend/rag/pipeline.py