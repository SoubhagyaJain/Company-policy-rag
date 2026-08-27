---
type: "query"
date: "2026-08-27T02:54:17.040528+00:00"
question: "What causes the maximum update depth loop between useChatStream updateProgress, ApiClient.handleStreamEvent, and streamChat?"
contributor: "graphify"
source_nodes: ["useChatStream()", "ApiClient", ".handleStreamEvent()", ".streamChat()"]
---

# Q: What causes the maximum update depth loop between useChatStream updateProgress, ApiClient.handleStreamEvent, and streamChat?

## Answer

A burst of SSE token events called onChunk synchronously. Each token performed both a redundant answer_generation progress update and a content update, while page.tsx persisted every intermediate messages snapshot. The repair batches token text into 32 ms updates, emits answer-generation progress once, returns unchanged state for duplicate progress events, and debounces session persistence by 180 ms.

## Source Nodes

- useChatStream()
- ApiClient
- .handleStreamEvent()
- .streamChat()