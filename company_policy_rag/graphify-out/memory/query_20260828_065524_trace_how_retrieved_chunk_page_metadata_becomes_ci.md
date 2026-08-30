---
type: "query"
date: "2026-08-28T06:55:24.480083+00:00"
question: "Trace how retrieved chunk page metadata becomes citation page numbers and identify why answers and page numbers can be wrong for arbitrary queries."
contributor: "graphify"
outcome: "useful"
source_nodes: ["company_policy_rag_backend_models_chunk_chunkmetadata", "company_policy_rag_backend_ingestion_chunkers_base_basechunker", "company_policy_rag_backend_rag_evidence_gate_evidencesufficiencygate", "company_policy_rag_backend_rag_pipeline_ragpipeline"]
---

# Q: Trace how retrieved chunk page metadata becomes citation page numbers and identify why answers and page numbers can be wrong for arbitrary queries.

## Answer

Root causes fixed: chunking dropped canonical page identity and visual asset fields; citation identity ignored preserved extra metadata; document filtering sent filenames without stable IDs and the document registry disappeared on restart; evidence gating treated 'depicted below' as complete text and the verifier allowed unsupported list completion. The implementation now preserves display page labels/assets, scopes retrieval by document ID, restores the document registry, detects missing referenced visuals, abstains instead of guessing when CPU vision is unavailable, deduplicates the abstention citation, and enforces concise direct answers. Exact UI validation used AI Engineering Guidebook (2).pdf and now reports one source at printed Page 70 in about 1.3 seconds.

## Outcome

- Signal: useful

## Source Nodes

- company_policy_rag_backend_models_chunk_chunkmetadata
- company_policy_rag_backend_ingestion_chunkers_base_basechunker
- company_policy_rag_backend_rag_evidence_gate_evidencesufficiencygate
- company_policy_rag_backend_rag_pipeline_ragpipeline