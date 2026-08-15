# E2E Test Infra: Agentic Intelligence Visual Indicators

## Test Philosophy
- **Requirement-Driven & Opaque-Box**: Validate that all visual indicators (query classification, verification scores, dimension breakdowns, retries, filter chips, relaxation warnings, cache hits) render accurately and faithfully across desktop and mobile, light and dark themes.
- **Methodology**: Category-Partition, Boundary Value Analysis (BVA), Pairwise Feature Interactions, and Real-World Workload Testing.

## Feature Inventory & Test Coverage Mapping
| # | Feature | Source | Tier 1 (Feature) | Tier 2 (Boundary) | Tier 3 (Pairwise) | Tier 4 (Workload) |
|---|---------|--------|:----------------:|:-----------------:|:-----------------:|:-----------------:|
| F1 | Type System Integrity | ORIGINAL_REQUEST §Integration | 5 | 5 | ✓ | ✓ |
| F2 | SSE Deserialization | ORIGINAL_REQUEST §Context, §Integration | 5 | 5 | ✓ | ✓ |
| F3 | R1 Query Classification Badge | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| F4 | R2 Composite Score Pill | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| F5 | R2 4-Dimension Progress Bars | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| F6 | R2 Retry Indicator & Tooltip | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| F7 | R3 Metadata Filter Tags | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| F8 | R3 Filter Relaxation Warning | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| F9 | R3 Semantic Cache Hit Badge | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| F10 | R4 AdminView Table Columns | ORIGINAL_REQUEST §R4 | 5 | 5 | ✓ | ✓ |
| F11 | R4 AdminView Expandable Detail | ORIGINAL_REQUEST §R4 | 5 | 5 | ✓ | ✓ |

## Test Architecture
- **E2E & Component Test Runner**: Integration and E2E validation scripts verifying:
  1. Next.js build compilation with full TypeScript type checks (`npm run build`).
  2. SSE stream payload ingestion, mapping, and state propagation.
  3. Frontend rendering of all badges, pills, progress bars, tooltips, filter chips, and table columns.
  4. Light / Dark mode class adherence and responsive layout styles.
- **Pass/Fail Semantics**: All test suites must pass with exit code 0 and 0 runtime/type errors.

## Coverage Goals
- **Tier 1 (Feature Coverage)**: >= 55 test cases across 11 features.
- **Tier 2 (Boundary & Corner)**: >= 55 test cases (nulls, zero retries, missing dimensions, extreme latency, long queries, relaxed filters, empty tags).
- **Tier 3 (Cross-Feature Combinations)**: Pairwise combinations (e.g. Cache Hit + Filter Relaxed, Retry > 0 + Low Verification Score, Conversational + No Chunks).
- **Tier 4 (Real-World Scenarios)**: >= 6 comprehensive realistic workloads (Factual HR query, Comparison query across departments, Procedural multi-step with retries, Enumeration with relaxed filters, Conversational greeting, Admin observability audit).
