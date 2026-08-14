# E2E Test Infra: Company Policy RAG Agentic Layer

## Test Philosophy
- Opaque-box, requirement-driven. No dependency on implementation design.
- Methodology: Category-Partition + Boundary Value Analysis (BVA) + Pairwise Combinatorial + Real-World Workload Testing.

## Feature Inventory
| # | Feature | Source (requirement) | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|---------|---------------------|:------:|:------:|:------:|:------:|
| 1 | Query Classification (Factual, Comparison, Enum, Procedural, Conversational) | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 2 | Strategy Selection & Retrieval Parameter Variation | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 3 | Conversational Bypass (No DB/Retrieval for Greetings) | ORIGINAL_REQUEST §R1 | 3 | 3 | ✓ | ✓ |
| 4 | Post-Generation Verification (Faithfulness, Completeness, Citations, Coherence) | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 5 | Autonomous Retry Loop (Adjusted Parameters on Failure) | ORIGINAL_REQUEST §R2 | 4 | 4 | ✓ | ✓ |
| 6 | Retry Hard Cap & Fallback (Max 2 Retries, No Infinite Loops) | ORIGINAL_REQUEST §R2 | 3 | 3 | ✓ | ✓ |
| 7 | Ingestion Dynamic Metadata Extraction (Dept, Policy ID, Dates, Entities, Tags) | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 8 | Query-Time Metadata Inference & Pre-Filtering | ORIGINAL_REQUEST §R3 | 4 | 4 | ✓ | ✓ |
| 9 | Filter Fallback Relaxation (Relax filter if 0 results) | ORIGINAL_REQUEST §R3 | 3 | 3 | ✓ | ✓ |
| 10 | SSE Stream Trace Telemetry & UI Badge/Indicator Surfacing | ORIGINAL_REQUEST §R4 | 4 | 4 | ✓ | ✓ |
| 11 | Non-Regression (Hybrid RRF, Rerank, Parent Expansion, Semantic Cache, Memory) | ORIGINAL_REQUEST §R4 | 5 | 5 | ✓ | ✓ |

## Test Architecture
- Test Runner: `pytest company_policy_rag/tests/e2e/test_e2e_agentic_layer.py -v`
- In-process ASGI transport via `httpx.AsyncClient` with `create_app()`
- SSE Event stream parser in `tests/e2e/helpers/sse_client.py`
- Directory layout: `company_policy_rag/tests/e2e/`

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | IT Remote Security Onboarding Flow | Query Routing + Metadata Inference + Reranking + Citations | High |
| 2 | HR Benefits vs Legal Code Comparison | Comparison Strategy + Multi-Query + Verification + Citations | High |
| 3 | Financial Expense Policy Limit Lookups | Factual Strategy + Entity Extraction + Ingestion Metadata | Medium |
| 4 | PTO & Leave Escalation with Retry Loop | Verification Failure Simulation + Autonomous Retry + Fallback | High |
| 5 | Conversational Smalltalk & Follow-up Multi-turn | Conversational Bypass + Session Memory + Pronoun Resolution | Medium |
| 6 | Non-Existent Policy Query with Filter Relaxation | Filter Inference + Fallback Relaxation + Grounded Synthesis | High |

## Coverage Thresholds
- Tier 1: ≥5 per major feature (21 feature coverage tests)
- Tier 2: ≥5 boundary & corner tests (20 boundary tests)
- Tier 3: Pairwise cross-feature interactions (10 combination tests)
- Tier 4: Realistic enterprise application workflows (6 multi-turn scenarios)
- **Total: 57 E2E tests + 270+ Unit/Integration tests**
