# Project: Agentic Intelligence Visual Indicators

## Architecture
- **Framework**: Next.js 16 / React 19 Frontend with Tailwind CSS, Framer Motion animations, Lucide React icons.
- **Backend Service**: FastAPI backend emitting Server-Sent Events (SSE) including `event: trace`, `event: done`, `event: retrieval`.
- **Data Flow**:
  1. Backend emits SSE events during chat streaming with full agentic telemetry in `trace` and `done` payloads.
  2. Frontend `api-client.ts` parses SSE events in `streamChat()` and deserializes traces via `mapTrace()`.
  3. `useChatStream.ts` stores the mapped `QueryTrace` on the assistant message object and stream state.
  4. `ChatMessage.tsx` renders the collapsible trace banner with R1 query badge, R2 verification pill & 4 progress bars & retry badge, R3 filter tag chips & relaxation warning & cache hit badge.
  5. `AdminView.tsx` displays recent traces in an interactive table with query type, verification score, filter status columns, and expandable row breakdown.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | Extended TypeScript Models | Define `VerificationReport` and extend `QueryTrace` in `frontend/lib/types.ts` with all 13 agentic telemetry fields. | M1 | ORIGINAL_REQUEST §Context, §Integration |
| F2 | SSE Trace Deserialization & Mapping | Update `mapTrace()` and `getObservability()` in `frontend/lib/api-client.ts` to map and preserve all agentic telemetry fields without data loss. | M1 | ORIGINAL_REQUEST §Context, §Integration |
| F3 | R1 Query Classification Badge | Display query type (`factual`, `comparison`, `enumeration`, `procedural`, `conversational`) with distinct Anthropic palette color chip and routing confidence percentage on hover in `ChatMessage.tsx`. | M2 | ORIGINAL_REQUEST §R1 |
| F4 | R2 Verification Composite Score Pill | Display pass (emerald/green) vs fail (warm amber) composite score pill in trace header row in `ChatMessage.tsx`. | M2 | ORIGINAL_REQUEST §R2 |
| F5 | R2 Verification 4-Dimension Progress Bars | Render 4 mini progress bars (faithfulness, completeness, citation coverage, coherence) with numeric labels inside expanded trace banner in `ChatMessage.tsx`. | M2 | ORIGINAL_REQUEST §R2 |
| F6 | R2 Verification Retry Indicator | Display retry count badge with hover tooltip listing retry reasons when `retry_count > 0` in `ChatMessage.tsx`. | M2 | ORIGINAL_REQUEST §R2 |
| F7 | R3 Metadata Filter Tag Chips | Display key-value tag chips inside expanded trace banner when `inferred_filters` is non-empty in `ChatMessage.tsx`. | M2 | ORIGINAL_REQUEST §R3 |
| F8 | R3 Filter Relaxation Warning | Display subtle warning callout explaining filter fallback when `filter_relaxed === true` in `ChatMessage.tsx`. | M2 | ORIGINAL_REQUEST §R3 |
| F9 | R3 Semantic Cache Hit Badge | Display cache hit badge (and similarity score if present) when `cache_hit === true` in `ChatMessage.tsx`. | M2 | ORIGINAL_REQUEST §R3 |
| F10 | R4 AdminView Trace Table Columns | Add Query Type chip, Verification Score pill, and Filter Status columns to Recent Query Traces table in `AdminView.tsx`. | M3 | ORIGINAL_REQUEST §R4 |
| F11 | R4 AdminView Expandable Detail Row | Expandable row detail in `AdminView.tsx` showing full verification breakdown (4 dimensions, critique, unsupported claims, missing aspects), retry history, and filter parameters. | M3 | ORIGINAL_REQUEST §R4 |
| F12 | Mobile Responsiveness & Dark Mode | Ensure all new indicators and badges adapt cleanly to mobile viewports and support dark mode with Anthropic color palette. | M2, M3 | ORIGINAL_REQUEST §Quality |
| F13 | Full E2E & Production Build Verification | 100% test pass on E2E test suites (Tiers 1-4) and clean Next.js build (`npm run build`). | M4 | ORIGINAL_REQUEST §Acceptance Criteria |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Telemetry Ingestion & Type System | Extend `types.ts` (`QueryTrace`, `VerificationReport`), update `api-client.ts` (`mapTrace`, `getObservability`), verify `useChatStream.ts`. | None | DONE |
| M2 | ChatMessage Agentic Indicators | Implement R1, R2, R3 in `ChatMessage.tsx` (badges, pills, 4-bar breakdown, retries, filter tags, relaxation warning, cache hit, dark mode, Framer Motion). | M1 | DONE |
| M3 | AdminView Observability Enhancements | Implement R4 in `AdminView.tsx` (new table columns, responsive layout, expanded row details for verification and filters). | M1 | DONE |
| M4 | E2E Integration & Verification | Execute E2E testing suite (Tiers 1-4), verify build (`npm run build`), review, challenger validation, and forensic audit. | M2, M3 | DONE |

## Interface Contracts
### `frontend/lib/types.ts` ↔ `frontend/lib/api-client.ts`
```typescript
export interface VerificationReport {
  faithfulness: number;
  completeness: number;
  citation_coverage: number;
  coherence: number;
  composite_score: number;
  passed: boolean;
  critique?: string | null;
  missing_aspects?: string[];
  unsupported_claims?: string[];
  retry_count?: number;
}

export type QueryCategory = 'factual' | 'comparison' | 'enumeration' | 'procedural' | 'conversational';

export interface QueryTrace {
  trace_id: string;
  timestamp: string;
  original_query: string;
  query_rewritten?: string;
  expanded_queries?: string[];
  total_chunks_retrieved: number;
  top_rerank_score: number;
  rerank_latency_ms: number;
  total_latency_ms: number;
  prompt_tokens: number;
  completion_tokens: number;
  model: string;
  // Agentic Telemetry Fields
  query_type?: string;
  routing_confidence?: number;
  retrieval_strategy?: string;
  inferred_filters?: Record<string, any>;
  applied_filters?: Record<string, any>;
  filter_relaxed?: boolean;
  verification_score?: number;
  verification?: VerificationReport | null;
  faithfulness_passed?: boolean;
  retry_count?: number;
  retry_reasons?: string[];
  cache_hit?: boolean;
  cache_similarity?: number | null;
}
```

## Code Layout
- `frontend/lib/types.ts` — TypeScript definitions (owned by M1)
- `frontend/lib/api-client.ts` — API client & SSE parsing (owned by M1)
- `frontend/hooks/useChatStream.ts` — Stream state management (shared/verified in M1)
- `frontend/components/ChatMessage.tsx` — Chat message rendering & trace banner (owned by M2)
- `frontend/components/AdminView.tsx` — Observability dashboard & trace table (owned by M3)
- `frontend/e2e/` or `tests/` — E2E test suites (owned by E2E Testing Track / M4)
