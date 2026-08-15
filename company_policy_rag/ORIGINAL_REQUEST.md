# Original User Request

## 2026-08-15T02:36:11Z

Add full-featured agentic intelligence visual indicators to an existing production Next.js 16 / React 19 RAG chatbot frontend. The backend already emits all agentic telemetry via SSE trace and done events — the frontend needs to surface this data as polished, production-quality UI components that match the existing Anthropic-inspired editorial aesthetic (warm cream, sand, terracotta accents, serif typography, Framer Motion animations).

Working directory: c:\Users\jains\OneDrive\Desktop\Rag-chatbot\company_policy_rag
Integrity mode: demo

## Context

The backend FastAPI service emits SSE events during chat streaming. The event: trace payload already contains all agentic fields but the frontend's mapTrace() in lib/api-client.ts currently drops them, and lib/types.ts has no TypeScript types for them. The existing QueryTrace type only tracks basic retrieval metrics (chunks, rerank score, latency, tokens). The styling is 100% Tailwind CSS with framer-motion animations and lucide-react icons. Dark mode is supported via dark: classes.

### SSE Trace Payload (already emitted by backend, currently dropped by frontend)
The event: trace SSE payload includes these unused agentic fields:
- query_type: string — one of factual, comparison, enumeration, procedural, conversational
- routing_confidence: float — 0.0 to 1.0
- retrieval_strategy: string — strategy name selected by the router
- inferred_filters: dict — metadata filters auto-detected from the query (e.g. {department: "HR", topic: "benefits"})
- applied_filters: dict — filters actually used in retrieval (may differ if relaxed)
- filter_relaxed: boolean — true if filters were dropped due to zero results
- verification_score: float — composite verification score (0.0 to 1.0)
- verification: dict — full report with faithfulness, completeness, citation_coverage, coherence, composite_score, passed, critique, missing_aspects, unsupported_claims
- faithfulness_passed: boolean
- retry_count: int — number of verification retry cycles (0–2)
- retry_reasons: list[str] — why each retry was triggered
- cache_hit: boolean
- cache_similarity: float | null

### Files That Need Changes
- frontend/lib/types.ts — extend QueryTrace with agentic fields
- frontend/lib/api-client.ts — update mapTrace() to preserve agentic fields
- frontend/components/ChatMessage.tsx — add badges, verification indicators, filter tags to the existing trace banner
- frontend/components/AdminView.tsx — extend the trace table with agentic columns
- frontend/hooks/useChatStream.ts — may need minor updates to pass new trace fields through

## Requirements

### R1. Query Classification Badge
Display the detected query type and router confidence as a compact badge in each AI response's existing collapsible trace banner (the "Thought for X ms" section in ChatMessage.tsx). The badge should use distinct colors per query type and show the confidence percentage on hover.

### R2. Self-Reflection Verification Indicators
Show a verification status indicator on each AI response: a composite score pill (pass = green, fail = amber) next to the existing trace banner header. When the trace banner is expanded, show 4 dimension scores (faithfulness, completeness, citation coverage, coherence) as mini progress bars with numeric labels. If retries occurred (retry_count > 0), show a retry count badge with tooltip listing the retry reasons.

### R3. Metadata Filter Tags
When inferred metadata filters are present (inferred_filters is non-empty), display them as styled tag chips inside the expanded trace banner. If filter_relaxed is true, show a subtle warning indicator explaining that filtered search returned zero results and fell back to unfiltered retrieval. Also display a cache_hit badge when the response came from semantic cache.

### R4. AdminView Trace Table Enhancement
Extend the existing Recent Query Traces table in AdminView.tsx to include new columns for Query Type (with color chip), Verification Score (with pass/fail coloring), and Filter Status (showing active filters or "none"). The expandable row detail should show the full verification breakdown and retry history.

## Acceptance Criteria

### Visual Indicators
- [ ] Query type badge renders inside the existing trace banner on every AI response with the correct classification from SSE trace data
- [ ] Verification composite score pill shows pass (green tones) / fail (warm amber tones) matching the existing Anthropic palette
- [ ] Expanding the trace banner shows 4 verification dimension bars with numeric scores (faithfulness, completeness, citation coverage, coherence)
- [ ] Retry count badge with tooltip appears when retry_count > 0
- [ ] Metadata filter tag chips render inside the expanded trace banner when inferred_filters is non-empty
- [ ] Filter relaxation warning indicator appears when filter_relaxed is true
- [ ] Cache hit badge appears when cache_hit is true

### Integration
- [ ] QueryTrace TypeScript type is extended with all agentic fields
- [ ] mapTrace() in api-client.ts preserves agentic fields from SSE trace payloads
- [ ] All data is consumed from existing SSE events — no backend changes needed
- [ ] Existing chat functionality (streaming, model switching, sessions, citations, copy button) is not broken
- [ ] The frontend builds without TypeScript errors (npm run build succeeds)

### Quality
- [ ] New components use the existing Tailwind color palette (cream, sand, terracotta, charcoal) and dark mode classes
- [ ] Framer Motion animations are used for expandable sections and badge appearances
- [ ] lucide-react icons are used consistently with existing icon usage
- [ ] All new UI elements are responsive on mobile viewports
