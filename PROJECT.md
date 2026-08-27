# Project: Production-Grade Thinking & Conversation-Aware Follow-Up System

## Architecture
The system delivers an enterprise-grade multi-turn RAG conversation and safe thinking experience for `company_policy_rag`.

```
[User / Client]
  │
  ▼
[FastAPI Endpoints: /api/chat, /api/chat/stream]
  │
  ├── [ChatService & ConversationStateManager]
  │     ├── Multi-turn session history
  │     └── ConversationEvidenceContext (verified chunks, citations, visual assets)
  │
  ├── [ConversationResolver & FollowUpResolver] (Phases 2 & 4)
  │     ├── Layer 1: Deterministic conversational cues
  │     ├── Layer 2: Turn structure analysis
  │     ├── Layer 3: Entity/topic extraction & subject resolution
  │     └── AnswerMode & ExpansionPlan determination
  │
  ├── [ThinkingStateMachine] (Phases 6, 7, 8, 13, 14)
  │     ├── Deterministic stage lifecycle (received → follow_up → retrieval → evidence → visual → plan → stream)
  │     ├── Safe summaries (zero hidden reasoning, zero extra LLM calls)
  │     └── Real-time SSE event emission (event: thinking, event: token, event: citation, event: complete)
  │
  ├── [Hybrid Retrieval & Expansion Engine] (Phases 3, 10, 14)
  │     ├── Dense Vector + BM25 + Cross-Encoder Reranker
  │     ├── Conversation consistency & downgrade protection (ConversationConsistencyGuard)
  │     ├── Page & Section expansion around previous evidence
  │     └── Multimodal ImageAssetManager & VisionCacheManager (code screenshots, diagrams)
  │
  ├── [Grounded LLM Generator] (Phase 12)
  │     └── Grounded prompt with Rules A-F (Continuity, Expansion, No False Absence, Code fidelity)
  │
  └── [Frontend UI & Streamlit] (Phase 11)
        ├── Next.js & Aether: ThinkingPanel, ThinkingStep, useThinkingStream, types/thinking.ts
        └── Streamlit: st.status live progress & history expander
```

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Architecture Audit & Mapping | Map complete request lifecycle and verify integration points | Survey (Done) | Phase 1 |
| 2 | FollowUpResolver & Resolution Model | Generic detection of follow-ups ("tell me about it in detail", "explain this code", etc.) | M1 | Phase 2 |
| 3 | ConversationEvidenceContext | Cross-turn evidence preservation, chunk/citation reuse, window expansion | M1 | Phase 3 |
| 4 | AnswerMode & ExpansionPolicy | Explicit answer modes (DETAILED, CODE_EXPLANATION, etc.) and non-shrinking expansion policy | M1 | Phase 4 |
| 5 | ConversationConsistencyGuard | Monotonic downgrade protection preventing "I could not find this information" contradictions | M1 | Phase 5 |
| 6 | Generation Prompt Hardening | Rules A-F for grounded generation in system prompt | M1 | Phase 12 |
| 7 | ThinkingEvent Data Model & State Machine | ThinkingStage, ThinkingStatus, ThinkingDetailLevel, ThinkingEvent, ThinkingStateMachine | M2 | Phase 6 |
| 8 | Safe Deterministic Summaries | Milestone summaries generated deterministically without exposing CoT or adding LLM latency | M2 | Phase 7 |
| 9 | SSE Streaming Extension | Strict event ordering (thinking -> token -> citation -> complete), backward compatibility | M2 | Phase 8 |
| 10 | Performance Tracking | Zero added LLM latency, timing measurements across all stages | M2 | Phase 13 |
| 11 | Graceful Failure Degradation | Dense->BM25 fallback, reranker bypass, vision timeout handling, ambiguity handling | M2 | Phase 14 |
| 12 | RAGTrace ReasoningSummary | Safe structured reasoning metadata added to RAGTrace for telemetry | M3 | Phase 9 |
| 13 | Multimodal Integration | Code screenshot reuse, diagram context expansion, vision timeout resilience | M3 | Phase 10 |
| 14 | Premium Frontend Thinking UI | ThinkingPanel, ThinkingStep, useThinkingStream in Next.js and Aether, accessibility, themes | M4 | Phase 11 |
| 15 | Streamlit Thinking UI | Live st.status milestone tracking and turn history reasoning expanders in Streamlit | M4 | Phase 11 |
| 16 | 20-Scenario E2E Test Suite | Comprehensive opaque-box test suite across Tiers 1-4 covering all required scenarios | E2E Track | Phase 15 |
| 17 | Adversarial Hardening (Tier 5) | White-box challenger verification and gap closure | M5 | Phase 15 |
| 18 | Multi-Turn E2E Verification & Final Report | Live multi-turn evaluation ("What is implementation code..." -> "tell me about it in detail") | M5 | Phase 16 |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Testing Track | Test harness & 20-scenario suite (Tiers 1-4), TEST_INFRA.md, TEST_READY.md | Survey | READY |
| M1 | Core Conversation & Evidence Continuity | FollowUpResolver, ConversationEvidenceContext, AnswerMode, ExpansionPlan, ConsistencyGuard, Rules A-F | Survey | DONE |
| M2 | Safe Thinking State Machine & SSE Streaming | ThinkingEvent, ThinkingStateMachine, SSE stream emission, degradation paths, performance tracking | M1 | DONE |
| M3 | Multimodal & RAG Trace Integration | Code screenshot reuse, diagram expansion, vision degradation resilience, ReasoningSummary in RAGTrace | M1, M2 | DONE |
| M4 | Premium Frontend & Streamlit UI | ThinkingPanel, ThinkingStep, useThinkingStream, types/thinking.ts, Streamlit st.status | M2, M3 | DONE |
| M5 | 100% E2E Test Pass & Adversarial Hardening | Pass all 20 scenarios, Tier 5 adversarial hardening, multi-turn E2E verification, final report | E2E, M1, M2, M3, M4 | IN_PROGRESS |

---

## Interface Contracts

### M1 ↔ M2: FollowUpResolution & ConversationEvidenceContext
- `FollowUpResolution`:
  - `is_follow_up: bool`
  - `confidence: float`
  - `resolved_query: str`
  - `primary_subject: Optional[str]`
  - `referenced_answer_id: Optional[str]`
  - `answer_mode: AnswerMode`
  - `expansion_requested: bool`
  - `requested_detail_level: ThinkingDetailLevel`
  - `preserve_previous_evidence: bool`
  - `evidence_continuity_ids: list[str]`
  - `ambiguity_detected: bool`
- `ConversationEvidenceContext`:
  - `session_id: str`
  - `turn_id: str`
  - `verified_chunks: list[RetrievedChunk]`
  - `verified_citations: list[Citation]`
  - `visual_asset_ids: list[str]`
  - `evidence_status: EvidenceStatus`
  - `answer_mode: AnswerMode`
- `ThinkingStateMachine.record_stage(stage: ThinkingStage, status: ThinkingStatus, ...)` consumes resolution and evidence context to construct deterministic summaries.

### M2 ↔ M4: SSE Protocol
- `event: thinking` -> `data: {"id": str, "query_id": str, "stage": str, "status": str, "title": str, "summary": str, "details": Optional[dict], "duration_ms": Optional[float]}`
- `event: token` / `event: chunk` -> `data: {"content": str}`
- `event: citation` -> `data: {"citation": dict}`
- `event: trace` -> `data: {"trace": dict}`
- `event: complete` / `event: done` -> `data: {"turn_id": str, "reasoning_summary": dict, ...}`

---

## Code Layout
- Backend Models:
  - `backend/models/conversation.py` (FollowUpResolution, ConversationEvidenceContext, AnswerMode, ExpansionPlan)
  - `backend/models/rag.py` (ThinkingStage, ThinkingStatus, ThinkingDetailLevel, ThinkingEvent, ReasoningSummary, RAGTrace)
- Backend RAG Engine:
  - `backend/rag/conversation_resolver.py` (FollowUpResolver)
  - `backend/rag/consistency_guard.py` (ConversationConsistencyGuard)
  - `backend/rag/thinking.py` (ThinkingStateMachine, safe deterministic generators)
  - `backend/rag/pipeline.py` (Grounded system prompt Rules A-F, streaming query integration, evidence expansion)
  - `backend/services/chat_service.py` (SSE event generator, session evidence context manager)
- Backend Vision:
  - `backend/vision/image_asset_manager.py`, `backend/vision/vision_service.py` (Multi-turn screenshot/diagram reuse, cached visual extractions)
- Frontend (Next.js & Aether):
  - `frontend/types/thinking.ts` & `frontend/aether/src/types/thinking.ts`
  - `frontend/hooks/useThinkingStream.ts` & `frontend/aether/src/hooks/useThinkingStream.ts`
  - `frontend/components/ThinkingPanel.tsx`, `ThinkingStep.tsx` & `frontend/aether/src/components/ThinkingPanel.tsx`, `ThinkingStep.tsx`
  - `frontend/lib/api-client.ts` & `frontend/aether/src/api/client.ts`
- Streamlit UI:
  - `app/ui/components/chat.py`
- Tests:
  - `tests/test_thinking_conversation_scenarios.py` (20-scenario suite)
  - `tests/test_evidence_continuity.py`
  - `tests/test_thinking_events_and_degradation.py`
