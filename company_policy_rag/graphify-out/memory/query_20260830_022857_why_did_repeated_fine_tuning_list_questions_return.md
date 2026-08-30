---
type: "query"
date: "2026-08-30T02:28:57.580240+00:00"
question: "Why did repeated fine-tuning-list questions return the same degraded visual abstention?"
contributor: "graphify"
source_nodes: ["backend/rag/pipeline.py", "backend/rag/evidence_gate.py", "backend/services/document_service.py"]
---

# Q: Why did repeated fine-tuning-list questions return the same degraded visual abstention?

## Answer

The evidence gate found a visual pointer, but after restart the persisted Chroma chunks were not restored into docstore/BM25, continuation depth stopped before item 5, CPU query-time vision was disabled, and degraded answers were cache-eligible. Fixed startup chunk restoration, indexed adjacent-page expansion through three pages, complete numbered-list detection/extraction, exact-repeat cache bypass, legacy incomplete-cache rejection, and degraded-answer cache exclusion. Live verification returns LoRA, LoRA-FA, VeRA, Delta-LoRA, and LoRA+ twice and serves only the correct answer cross-session.

## Source Nodes

- backend/rag/pipeline.py
- backend/rag/evidence_gate.py
- backend/services/document_service.py