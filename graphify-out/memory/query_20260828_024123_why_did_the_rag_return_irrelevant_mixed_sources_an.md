---
type: "query"
date: "2026-08-28T02:41:23.766949+00:00"
question: "Why did the RAG return irrelevant mixed sources and an overly detailed answer for how can i make voice rag agent?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["company_policy_rag_backend_rag_query_rewrite_queryrewriter", "company_policy_rag_backend_rag_context_compression_contextcompressor", "company_policy_rag_backend_rag_citations_citationengine", "company_policy_rag_backend_rag_pipeline_ragpipeline", "company_policy_rag_backend_rag_semantic_cache_semanticcachemanager"]
---

# Q: Why did the RAG return irrelevant mixed sources and an overly detailed answer for how can i make voice rag agent?

## Answer

The query rewriter broadened voice RAG into generic Agentic RAG/vector DB terms, duplicated document passages survived retrieval and citation selection, the generation prompt encouraged rich implementation detail, and semantic cache could replay the stale answer. The fix preserves voice intent, deduplicates context and citations, caps default direct-answer length, and versions the cache context.

## Outcome

- Signal: useful

## Source Nodes

- company_policy_rag_backend_rag_query_rewrite_queryrewriter
- company_policy_rag_backend_rag_context_compression_contextcompressor
- company_policy_rag_backend_rag_citations_citationengine
- company_policy_rag_backend_rag_pipeline_ragpipeline
- company_policy_rag_backend_rag_semantic_cache_semanticcachemanager