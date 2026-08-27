# TEST_INFRA: Multi-Turn Conversation & Safe Thinking System

## 1. Test Philosophy & Principles
- **Opaque-Box Requirement Derivation**: All test specifications are derived directly from user requirements defined in `ORIGINAL_REQUEST.md` (Phases 1–16) and `PROJECT.md`. Tests validate external contracts, observable behavior, and data schemas rather than private internal implementation details.
- **Progressive Testability & Interface Contracts**: Validates both current and modular pipeline abstractions (`FollowUpResolver`, `ConversationEvidenceContext`, `AnswerMode`, `ConversationConsistencyGuard`, `ThinkingStateMachine`, `ThinkingEvent`, `SSE Stream Protocol`, `RAGTrace`).
- **Strict Safety & Zero Chain-of-Thought Exposure**: Asserts that internal chain-of-thought, classifier prompts, retrieval templates, secrets, vector IDs, embeddings, and raw scoring formulas are NEVER leaked in thinking events, trace summaries, or client responses.
- **Deterministic State Verification**: Verifies that thinking events, evidence statuses (DIRECT, PARTIAL, RELATED, MISSING), answer modes (DIRECT, DETAILED, EXPAND, SUMMARY, CONTINUE), and follow-up resolutions are generated deterministically from pipeline state without unneeded extra LLM calls.
- **Monotonic Evidence Preservation**: Enforces that valid previously verified grounded evidence is never erased or downgraded when a subsequent follow-up retrieval returns weak or missing chunks.

---

## 2. 4-Tier Testing Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 4-TIER TEST HARNESS                                    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TIER 1: Feature Coverage (Scenarios 1-10)                                              │
│ • S01: Normal factual query & grounding                                                │
│ • S02: Code implementation query & faithful extraction                                 │
│ • S03: Architecture/workflow diagram query & visual asset verification                 │
│ • S04: Structured table query & numerical matrix extraction                             │
│ • S05: Follow-up: "tell me more" (AnswerMode.EXPAND / DETAILED, topic continuity)      │
│ • S06: Follow-up: "tell me about it in detail" (Subject resolution & query expansion)   │
│ • S07: Follow-up: "explain this code" (Snippet retention & line-by-line explanation)   │
│ • S08: Follow-up after visual evidence (Diagram reuse & neighboring context expansion) │
│ • S09: Topic switch (Clean topic/chunk reset & zero state contamination)               │
│ • S10: Ambiguous pronoun resolution ("it", "that", "their" resolution)                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TIER 2: Boundary, Degradation & Safety (Scenarios 11-18)                              │
│ • S11: DIRECT evidence + weak retrieval (Monotone retention & no false absence)        │
│ • S12: PARTIAL evidence + MISSING retrieval (Monotone retention of partial facts)      │
│ • S13: Dense retrieval failure (Graceful BM25 sparse fallback & degraded event)        │
│ • S14: Vision timeout (Fallback to text evidence, cached extraction reuse, warning)    │
│ • S15: SSE thinking event ordering (thinking -> token -> citation -> trace -> done)   │
│ • S16: Thinking detail level OFF (Zero thinking events emitted)                        │
│ • S17: COMPACT filtering (Only milestone stage summaries emitted)                      │
│ • S18: DETAILED safe metrics (Duration, source counts, candidate stats, NO CoT leakage)│
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TIER 3: Cross-Feature Combinations & Isolation (Scenario 19 + Interactions)            │
│ • S19: Multi-session concurrency & complete conversation state isolation               │
│ • S19b: Cross-turn Multimodal + Code follow-up pipeline interaction                   │
│ • S19c: Degradation + Follow-up monotonic consistency interaction                      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TIER 4: Real-World E2E Workloads & Critical Regression (Scenario 20 + Critical Test)   │
│ • S20: Backward compatibility with existing REST and SSE streaming endpoints          │
│ • CRITICAL REGRESSION TEST: Turn 1 (Hotel Search Agent code) -> Turn 2 ("tell me about │
│   it in detail") with 7 mandatory assertions (is_follow_up=True, resolved subject,     │
│   evidence reuse, window expansion, DETAILED mode, no false absence, citations).       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Feature Inventory & Scenario Traceability Matrix

| # | Scenario Name | Target Feature / Module | Tier | Primary Assertions |
|---|---------------|-------------------------|:----:|-------------------|
| 1 | Normal factual query | `ConversationResolver`, `RAGPipeline` | 1 | `is_followup == False`, topic extracted, direct text citations `[Source N]`, grounded answer |
| 2 | Code implementation query | `EvidenceSufficiencyGate`, `RAGPipeline` | 1 | Code intent detected, syntax/kickoff snippets preserved, evidence status `DIRECT`/`PARTIAL` |
| 3 | Diagram query | `VisionService`, `RAGPipeline` | 1 | Architecture visual asset identified, `[VISUAL SOURCE N]` citations generated, workflow explained |
| 4 | Table query | `ChunkMetadata`, `RAGPipeline` | 1 | Tabular markdown/data extracted, numbers and column metrics faithfully represented |
| 5 | Follow-up: "tell me more" | `ConversationResolver` | 1 | `is_followup == True`, `answer_mode in (EXPAND, DETAILED)`, active topic preserved |
| 6 | Follow-up: "tell me about it in detail" | `ConversationResolver`, `RAGPipeline` | 1 | `is_followup == True`, referent "it" resolved to prior subject, expanded query synthesized |
| 7 | Follow-up: "explain this code" | `ConversationResolver`, `GroundedPrompt` | 1 | "this code" mapped to prior code snippet, explanation directives applied, code unbroken |
| 8 | Follow-up after visual evidence | `ImageAssetManager`, `VisionCacheManager` | 1 | Turn 1 diagram reused in Turn 2, zero redundant vision inference, adjacent context inspected |
| 9 | Topic switch | `ConversationResolver`, `ConversationStateManager` | 1 | `is_followup == False`, `topic_shift == True`, previous chunks cleared from active set |
| 10 | Ambiguous pronoun reference | `ConversationResolver` | 1 | Pronouns ("their", "those") resolved against active entities without crash or hallucination |
| 11 | DIRECT evidence + weak retrieval | `ConversationConsistencyGuard`, `EvidenceGate` | 2 | Monotonicity preserves `DIRECT` status, prior chunks retained, answer does not claim "not found" |
| 12 | PARTIAL evidence + MISSING retrieval | `ConversationConsistencyGuard`, `EvidenceGate` | 2 | Monotonicity preserves `PARTIAL` status, retains partial snippets, notes no new evidence |
| 13 | Dense retrieval failure | `HybridRetriever`, `ThinkingStateMachine` | 2 | Graceful BM25 fallback, degraded thinking event emitted, grounded synthesis succeeds |
| 14 | Vision timeout | `VisionService`, `ThinkingStateMachine` | 2 | Text evidence preserved, warning event emitted, pipeline does not crash |
| 15 | SSE thinking event ordering | `ChatService`, `SSEGenerator` | 2 | Strict ordering: `start` -> `thinking` -> `chunk`/`token` -> `citation` -> `trace` -> `done` |
| 16 | Thinking detail level OFF | `ThinkingStateMachine`, `ChatService` | 2 | Zero `thinking` events emitted, standard token streaming and completion unaffected |
| 17 | COMPACT filtering | `ThinkingStateMachine` | 2 | Only milestone stages (Understanding, Context, Retrieval, Verification, Planning) emitted |
| 18 | DETAILED safe metrics | `ThinkingStateMachine`, `ReasoningSummary` | 2 | Safe candidate counts, durations, source counts present; zero CoT / secrets leaked |
| 19 | Conversation session isolation | `ConversationStateManager` | 3 | Parallel sessions maintain 100% separate topics, entities, chunks, and citations |
| 20 | API regression tests | `APIRoutes` (`/api/chat`, `/api/chat/stream`) | 4 | Backward-compatible responses, correct HTTP status codes, session deletion support |
| 21 | Critical Multi-Turn Regression Test | Full Multi-Turn RAG Stack | 4 | Complete 2-turn verification: code query -> "tell me about it in detail" meeting all 7 criteria |

---

## 4. Safety & Anti-Hallucination Verification

Every test case in the suite asserts:
1. **Zero CoT Exposure**: No private thoughts, model scratchpads, or `First I thought...` strings in user payloads.
2. **Zero Internal Prompt Leakage**: System prompts, routing templates, and classifier instructions remain strictly internal.
3. **No False Absence**: The system never responds with "I could not find this information" when valid evidence exists in current or prior turns.
4. **No Code Fabrication**: Incomplete or partial code snippets are marked as partial rather than invented.

---

## 5. Test Commands & Verification Instructions

### Run the Complete 4-Tier Test Suite:
```bash
pytest tests/test_thinking_conversation_scenarios.py -v
```

### Run Specific Tiers:
```bash
# Run Tier 1 (Feature Coverage: Scenarios 1-10)
pytest tests/test_thinking_conversation_scenarios.py -k "tier1" -v

# Run Tier 2 (Boundary & Safety: Scenarios 11-18)
pytest tests/test_thinking_conversation_scenarios.py -k "tier2" -v

# Run Tier 3 (Cross-Feature & Isolation: Scenario 19)
pytest tests/test_thinking_conversation_scenarios.py -k "tier3" -v

# Run Tier 4 (E2E & Critical Regression: Scenario 20 + Critical Test)
pytest tests/test_thinking_conversation_scenarios.py -k "tier4" -v
```

### Run Full Regression Suite:
```bash
pytest tests/ -v
```
