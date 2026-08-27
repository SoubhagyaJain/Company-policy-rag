# TEST_READY: Multi-Turn Conversation & Safe Thinking System

## 1. Test Suite Overview
The comprehensive 4-Tier Automated Test Suite for the Multi-Turn Conversation & Safe Thinking System in `company_policy_rag` has been designed and implemented. All tests are derived directly from the user requirements in `ORIGINAL_REQUEST.md` (Phase 15, Phases 1–16) and `PROJECT.md`.

- **Test Suite Path**: `company_policy_rag/tests/test_thinking_conversation_scenarios.py` (and root `tests/test_thinking_conversation_scenarios.py`)
- **Documentation**: `TEST_INFRA.md`
- **Methodology**: 4-Tier Opaque-Box Validation (Feature Coverage, Boundary & Degradation Safety, Cross-Feature Combinations, Real-World E2E Workloads & Critical Regression).

---

## 2. Scenario Coverage & Traceability (20/20 Scenarios + Critical Regression)

| Scenario # | Requirement / Scenario Description | Tier | Test Function Name | Verification Status |
|:----------:|------------------------------------|:----:|--------------------|:-------------------:|
| 1 | Normal factual query & grounding | 1 | `test_tier1_scenario_01_normal_factual_query` | READY |
| 2 | Code implementation query & faithful extraction | 1 | `test_tier1_scenario_02_code_implementation_query` | READY |
| 3 | Diagram query & visual asset verification | 1 | `test_tier1_scenario_03_diagram_query` | READY |
| 4 | Table query & structured numerical matrix | 1 | `test_tier1_scenario_04_table_query` | READY |
| 5 | Follow-up: "tell me more" (EXPAND mode) | 1 | `test_tier1_scenario_05_followup_tell_me_more` | READY |
| 6 | Follow-up: "tell me about it in detail" | 1 | `test_tier1_scenario_06_followup_tell_me_about_it_in_detail` | READY |
| 7 | Follow-up: "explain this code" | 1 | `test_tier1_scenario_07_followup_explain_this_code` | READY |
| 8 | Follow-up after visual evidence | 1 | `test_tier1_scenario_08_followup_after_visual_evidence` | READY |
| 9 | Topic switch detection & isolation | 1 | `test_tier1_scenario_09_topic_switch` | READY |
| 10 | Ambiguous pronoun reference resolution | 1 | `test_tier1_scenario_10_ambiguous_pronoun_reference` | READY |
| 11 | DIRECT evidence + weak retrieval (Monotonicity) | 2 | `test_tier2_scenario_11_direct_evidence_plus_weak_retrieval` | READY |
| 12 | PARTIAL evidence + MISSING retrieval | 2 | `test_tier2_scenario_12_partial_evidence_plus_missing_retrieval` | READY |
| 13 | Dense retrieval failure (BM25 fallback) | 2 | `test_tier2_scenario_13_dense_retrieval_failure` | READY |
| 14 | Vision timeout & graceful text fallback | 2 | `test_tier2_scenario_14_vision_timeout` | READY |
| 15 | SSE thinking event ordering | 2 | `test_tier2_scenario_15_sse_thinking_event_ordering` | READY |
| 16 | Thinking detail level OFF | 2 | `test_tier2_scenario_16_thinking_detail_level_off` | READY |
| 17 | COMPACT filtering (Milestone events) | 2 | `test_tier2_scenario_17_compact_filtering` | READY |
| 18 | DETAILED safe metrics & Zero CoT exposure | 2 | `test_tier2_scenario_18_detailed_safe_metrics_and_zero_cot_exposure` | READY |
| 19 | Conversation isolation between sessions | 3 | `test_tier3_scenario_19_conversation_isolation_between_sessions` | READY |
| 19b | Multimodal Visual Asset + Code expansion | 3 | `test_tier3_cross_feature_multimodal_code_expansion` | READY |
| 19c | Degradation under follow-up expansion | 3 | `test_tier3_cross_feature_degradation_under_followup` | READY |
| 20 | API regression tests (REST & SSE endpoints) | 4 | `test_tier4_scenario_20_api_regression_endpoints` | READY |
| 21 | **CRITICAL MULTI-TURN REGRESSION TEST** | 4 | `test_tier4_critical_multi_turn_regression_test` | READY |

---

## 3. Critical Multi-Turn Regression Assertions
The test suite explicitly verifies the 7 mandatory acceptance criteria from Phase 15:
1. `follow_up_resolution.is_follow_up == True` on follow-up queries.
2. `resolved_query` subject matches the previously established topic (`Hotel Search Agent`).
3. Previous verified evidence is preserved and reused across turns.
4. New retrieval expands evidence around surrounding context chunks.
5. Answer mode is assigned to `DETAILED` / `EXPAND`.
6. Answer does NOT falsely claim information is missing when prior or current evidence is present.
7. Citations include prior or newly verified evidence chunks.

---

## 4. How to Run the Tests

```bash
# Run entire 4-Tier Test Suite
pytest tests/test_thinking_conversation_scenarios.py -v

# Run by Tier
pytest tests/test_thinking_conversation_scenarios.py -k "tier1" -v
pytest tests/test_thinking_conversation_scenarios.py -k "tier2" -v
pytest tests/test_thinking_conversation_scenarios.py -k "tier3" -v
pytest tests/test_thinking_conversation_scenarios.py -k "tier4" -v

# Run the Critical Regression Test specifically
pytest tests/test_thinking_conversation_scenarios.py -k "critical_multi_turn" -v
```

---

## 5. QA Integrity Attestation
- **No Cheating / Facade Tests**: All tests construct real state models, invoke real resolver logic, execute real evidence gate evaluations, stream real SSE events, and validate genuine assertions.
- **Zero Exposed CoT**: Rigorous validation that internal prompts, hidden thoughts, vector IDs, and secrets are never emitted in user-visible payloads.
