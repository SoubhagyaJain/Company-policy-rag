# Original User Request

## 2026-08-25T16:16:10Z

Implement a production-grade Conversation-Aware RAG layer for multi-turn conversational retrieval, multimodal evidence grounding, dynamic query resolution, evidence continuity, and monotonicity.

Working directory: c:\Users\jains\OneDrive\Desktop\Rag-chatbot\company_policy_rag
Integrity mode: development

## Requirements

### R1. Conversation State & Isolation
Implement an isolated, thread-safe `ConversationRAGState` and `ConversationStateManager` tracking `conversation_id`, `active_topic`, `active_entities`, `last_resolved_query`, `previous_intent`, `previous_answer_mode`, `previous_evidence_status`, `previous_retrieved_chunks`, `previous_visual_evidence`, `previous_citations`, and `last_answer`. Prevent global mutable state and evidence leakage across sessions.

### R2. Dynamic Query Resolution & Topic Continuity
Detect follow-up queries and referential pronouns (it, that, this, the above, explain further, tell about it in detail). Detect topic shifts versus continuations dynamically. Resolve follow-up queries into standalone search queries incorporating topic entities and expansion intent without hardcoded document specifics.

### R3. Evidence Continuity & Monotonicity
Merge, deduplicate, and rerank previous grounded chunks + new chunks + visual evidence. Enforce evidence status monotonicity so `DIRECT` or `PARTIAL` evidence is never downgraded to `MISSING` on follow-up turns when valid context exists.

### R4. Answer Modes & Grounded Synthesis Rules
Implement answer modes (`DIRECT`, `DETAILED`, `EXPAND`, `SUMMARY`, `CONTINUE`). In `EXPAND`/`DETAILED` mode, avoid repeating previous answers, explain available code and architecture in detail, and clearly distinguish directly documented facts, partial implementation, related context, and genuinely missing information.

### R5. Observability & Testing
Emit structured logs (`[CONVERSATION]`, `[QUERY_RESOLUTION]`, `[EVIDENCE_CONTINUITY]`, `[EVIDENCE_STATUS]`, `[ANSWER_MODE]`) and implement 12 comprehensive automated tests covering all follow-up, topic-shift, and evidence continuity scenarios.

## Acceptance Criteria

### Follow-Up Query Resolution
- [ ] Resolves "it", "that", "this", and phrases like "tell about it in detail" to the active topic.
- [ ] Detects genuine new topics and clears the active topic.

### Evidence Continuity & Monotonicity
- [ ] Retains valid evidence from Turn 1 when Turn 2 retrieval is weak.
- [ ] Never downgrades `PARTIAL` or `DIRECT` evidence to `MISSING` on follow-ups.
- [ ] Reuses previous visual evidence without forced re-inference.

### State & Isolation
- [ ] Zero evidence leakage between distinct conversation IDs.
- [ ] Thread-safe session eviction and clearing.

### Answer Quality & Grounding
- [ ] In `EXPAND` mode, expands grounded explanation without fabricating code or claiming evidence is absent.
- [ ] 100% pass rate across automated test suite.
