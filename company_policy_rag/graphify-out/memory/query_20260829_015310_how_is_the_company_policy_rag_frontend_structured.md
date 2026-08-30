---
type: "query"
date: "2026-08-29T01:53:10.248327+00:00"
question: "How is the company policy RAG frontend structured, and which components control navigation, chat, documents, and visual styling?"
contributor: "graphify"
source_nodes: ["app/page.tsx", "Header", "SessionSidebar", "ChatWindow", "DocumentsView.tsx", "AmbientKnowledgeField", "PolicyKnowledgeScene"]
---

# Q: How is the company policy RAG frontend structured, and which components control navigation, chat, documents, and visual styling?

## Answer

The Next.js page component orchestrates the Header, SessionSidebar, ChatWindow, DocumentsView, and AdminView. Header owns top-level navigation and theme controls; SessionSidebar owns conversation discovery and actions; ChatWindow owns retrieval filters, model selection, prompts, streaming messages, and the input composer; AmbientKnowledgeField and the global stylesheet supply the visual canvas. The refresh adds PolicyKnowledgeScene as a lazy Three.js enhancement for the empty state and keeps the existing SVG/CSS field as a fallback.

## Source Nodes

- app/page.tsx
- Header
- SessionSidebar
- ChatWindow
- DocumentsView.tsx
- AmbientKnowledgeField
- PolicyKnowledgeScene