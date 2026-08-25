# -*- coding: utf-8 -*-
"""
Generate the full index.html with all 15 core sections and all 100 deep technical interview Q&As
"""
import os
import json

HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Enterprise Policy RAG — 100 Interview Defense Playbook</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Caveat:wght@400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="style.css">
</head>
<body>

<!-- ═══ SCROLL PROGRESS ═══ -->
<div class="nav-progress" style="width:0%"></div>

<!-- ═══ TOP STICKY NAVBAR ═══ -->
<header class="top-navbar">
  <div class="navbar-container">
    <a href="#hero" class="navbar-brand">
      <span class="brand-dot"></span>
      <span class="brand-text">Enterprise Policy RAG</span>
      <span class="brand-tag">Playbook</span>
    </a>

    <nav class="navbar-links">
      <a href="#hero" class="nav-link">Overview</a>
      <a href="#architecture" class="nav-link">Architecture</a>
      <a href="#execution" class="nav-link">Execution</a>
      <a href="#dataflow" class="nav-link">Data Flow</a>
      <a href="#components" class="nav-link">Components</a>
      <a href="#decisions" class="nav-link">Decisions</a>
      <a href="#failures" class="nav-link">Failures</a>
      <a href="#interview" class="nav-link nav-link--highlight">100 Q&A</a>
      <a href="#scale" class="nav-link">Scale</a>
      <a href="#revision" class="nav-link">Revision</a>
    </nav>

    <div class="navbar-actions">
      <a href="#interview" class="navbar-btn--search" title="Jump to Q&A Search">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" style="margin-right:2px"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        <span class="search-hint">Search 100 Q&A</span>
      </a>

      <button class="theme-toggle-btn" id="themeToggle" aria-label="Toggle light/dark theme" title="Toggle theme">
        <span class="theme-icon sun-icon">☀️</span>
        <span class="theme-icon moon-icon">🌙</span>
      </button>

      <button class="mobile-menu-btn" id="mobileMenuBtn" aria-label="Toggle navigation menu">
        <span></span>
        <span></span>
        <span></span>
      </button>
    </div>
  </div>

  <!-- Mobile Drawer Menu -->
  <div class="mobile-drawer" id="mobileDrawer">
    <a href="#hero" class="mobile-nav-link">Overview</a>
    <a href="#architecture" class="mobile-nav-link">Architecture</a>
    <a href="#execution" class="mobile-nav-link">Execution Path</a>
    <a href="#dataflow" class="mobile-nav-link">Data Flow</a>
    <a href="#components" class="mobile-nav-link">Components</a>
    <a href="#decisions" class="mobile-nav-link">Design Decisions</a>
    <a href="#testing" class="mobile-nav-link">Testing Contracts</a>
    <a href="#failures" class="mobile-nav-link">Failure Scenarios</a>
    <a href="#interview" class="mobile-nav-link mobile-nav-link--highlight">Top 100 Q&A Explorer</a>
    <a href="#defense" class="mobile-nav-link">Code Defense</a>
    <a href="#scale" class="mobile-nav-link">Scalability</a>
    <a href="#gaps" class="mobile-nav-link">Knowledge Gaps</a>
    <a href="#revision" class="mobile-nav-link">10-Min Revision</a>
  </div>
</header>

<!-- ═══ SIDE NAVIGATION ═══ -->
<nav class="nav-sidebar">
  <div class="nav-dot" data-target="hero" data-label="Overview"></div>
  <div class="nav-dot" data-target="architecture" data-label="Architecture"></div>
  <div class="nav-dot" data-target="execution" data-label="Execution"></div>
  <div class="nav-dot" data-target="dataflow" data-label="Data Flow"></div>
  <div class="nav-dot" data-target="components" data-label="Components"></div>
  <div class="nav-dot" data-target="decisions" data-label="Decisions"></div>
  <div class="nav-dot" data-target="testing" data-label="Testing"></div>
  <div class="nav-dot" data-target="failures" data-label="Failures"></div>
  <div class="nav-dot" data-target="interview" data-label="100 Q&A"></div>
  <div class="nav-dot" data-target="defense" data-label="Code Defense"></div>
  <div class="nav-dot" data-target="scale" data-label="Scale"></div>
  <div class="nav-dot" data-target="gaps" data-label="Gaps"></div>
  <div class="nav-dot" data-target="revision" data-label="Revision"></div>
</nav>

<div class="page-wrapper">

<!-- ═══════════════════════════════════════════════════════════
     SECTION 01 — HERO: THE PROJECT IN 30 SECONDS
     ═══════════════════════════════════════════════════════════ -->
<section class="hero section" id="hero">
  <div class="hero-eyebrow">Project Interview Playbook</div>
  <h1 class="hero-title">Enterprise Policy<br>RAG AI Assistant</h1>
  <p class="hero-thesis">A local-first Retrieval-Augmented Generation platform that eliminates LLM hallucinations on high-stakes compliance policies through hybrid retrieval, cross-encoder reranking, and autonomous self-reflection verification — with an end-to-end QLoRA fine-tuning pipeline.</p>

  <!-- Hand-drawn illustration: Engineer studying system map -->
  <svg class="illustration illustration--hero" viewBox="0 0 320 200" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="80" cy="60" r="14" stroke="#C4653A" stroke-width="1.5" fill="none"/>
    <path d="M80 74 L80 120 M80 88 L60 105 M80 88 L100 105 M80 120 L65 155 M80 120 L95 155" stroke="#C4653A" stroke-width="1.5" stroke-linecap="round"/>
    <rect x="140" y="30" width="60" height="24" rx="4" stroke="#B0A898" stroke-width="1" fill="none"/>
    <rect x="220" y="30" width="60" height="24" rx="4" stroke="#B0A898" stroke-width="1" fill="none"/>
    <rect x="140" y="75" width="60" height="24" rx="4" stroke="#C4653A" stroke-width="1.5" fill="none" stroke-dasharray="3 2"/>
    <rect x="220" y="75" width="60" height="24" rx="4" stroke="#B0A898" stroke-width="1" fill="none"/>
    <rect x="180" y="120" width="60" height="24" rx="4" stroke="#B0A898" stroke-width="1" fill="none"/>
    <line x1="170" y1="54" x2="170" y2="75" stroke="#D0C8BC" stroke-width="1"/>
    <line x1="250" y1="54" x2="250" y2="75" stroke="#D0C8BC" stroke-width="1"/>
    <line x1="170" y1="99" x2="210" y2="120" stroke="#D0C8BC" stroke-width="1"/>
    <line x1="250" y1="99" x2="210" y2="120" stroke="#D0C8BC" stroke-width="1"/>
    <path d="M100 65 Q 120 50 138 42" stroke="#C4653A" stroke-width="1" stroke-dasharray="4 3" fill="none" marker-end="url(#arrowhead)"/>
    <defs><marker id="arrowhead" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto"><polygon points="0 0, 6 2, 0 4" fill="#C4653A"/></marker></defs>
    <text x="145" y="170" font-family="Caveat, cursive" font-size="14" fill="#C4653A">"I can see the whole system now."</text>
  </svg>

  <!-- Depth Layers -->
  <div class="depth-layers">
    <div class="depth-layer active" data-depth="10s">
      <div class="depth-layer-time">10 Seconds — The Recruiter Version</div>
      <div class="depth-layer-label">What is this?</div>
      <div class="depth-layer-content">
        <p>I built a production-ready RAG assistant for enterprise compliance policies that combines hybrid search, cross-encoder reranking, and an autonomous self-reflection loop to guarantee zero-hallucination answers with verified source citations.</p>
      </div>
    </div>

    <div class="depth-layer" data-depth="30s">
      <div class="depth-layer-time">30 Seconds — Technical Overview</div>
      <div class="depth-layer-label">How does it work?</div>
      <div class="depth-layer-content">
        <p>A local-first enterprise Policy RAG system built on FastAPI, ChromaDB, and Ollama. It uses hierarchical child-parent chunking, dense + BM25 hybrid retrieval merged via Reciprocal Rank Fusion, and a CUDA-accelerated BGE cross-encoder reranker. An Agentic layer classifies query intent, infers metadata filters, and runs a 4-dimensional self-reflection verification engine that autonomously retries failed generations up to two cycles. Includes a full QLoRA fine-tuning and GGUF export pipeline for local Qwen 2.5 model deployment.</p>
      </div>
    </div>

    <div class="depth-layer" data-depth="2min">
      <div class="depth-layer-time">2 Minutes — The Complete Story</div>
      <div class="depth-layer-label">The full project defense</div>
      <div class="depth-layer-content">
        <p>The system addresses the fundamental failure modes of standard RAG on complex policy and legal texts: semantic drift, loss of clause context, and subtle numerical hallucinations.</p>
        <p style="margin-top:0.75rem">On the <strong>ingestion</strong> side, documents undergo format-specific parsing and section-aware hierarchical chunking — storing 480-token child nodes in ChromaDB and linking them to 2,000-token parent contexts in an indexed docstore, with automated ISO date and department metadata extraction.</p>
        <p style="margin-top:0.75rem">At <strong>query time</strong>, an intelligent regex router classifies queries into five intent categories, dynamically tuning retrieval hyperparameters. After passing a cosine-similarity semantic cache (≥ 0.95), the query is rewritten for conversational context and metadata filters are inferred. We execute parallel dense vector search and BM25 lexical search, fusing them via Reciprocal Rank Fusion (k=60).</p>
        <p style="margin-top:0.75rem">The top 30 candidates pass to a GPU-accelerated <code>bge-reranker-large</code> cross-encoder, filtered using an adaptive relative score threshold, and expanded to their parent context. The generated response passes through a <strong>4D Self-Reflection Verifier</strong> measuring Faithfulness (35%), Completeness (30%), Citation Validity (20%), and Coherence (15%). Below the 0.70 composite threshold, an autonomous retry engine adjusts retrieval breadth and re-prompts the LLM.</p>
        <p style="margin-top:0.75rem">The system supports sub-second TTFT via Server-Sent Events, client disconnect cancellation, and features a full <strong>PEFT QLoRA fine-tuning engine</strong> targeting all 7 linear projection layers of Qwen 2.5 Coder 7B, exporting directly to quantized GGUF for Ollama serving.</p>
      </div>
    </div>
  </div>
</section>

<!-- ═══════════════════════════════════════════════════════════
     SECTION 02 — SYSTEM ARCHITECTURE
     ═══════════════════════════════════════════════════════════ -->
<section class="section" id="architecture">
  <div class="section-number">02</div>
  <div class="section-eyebrow">System Architecture</div>
  <h2 class="section-title">Before We Enter the Code</h2>
  <p class="section-subtitle">The entire system in one view. Hover any node to see what it does, what enters, and what leaves.</p>

  <span class="annotation">Start here →</span>

  <div class="arch-container">
    <div class="arch-flow">
      <div class="arch-node arch-node--primary">
        <div class="arch-node-title">🌐 User Browser (Next.js 16)</div>
        <div class="arch-node-desc">SSE Stream / POST /api/chat/stream</div>
        <div class="arch-node-detail">Input: User's natural language query + session ID + selected model.<br>Output: Real-time SSE token stream with citations and telemetry.</div>
      </div>

      <div class="arch-arrow">
        <div class="arch-arrow-label">HTTP POST + SSE</div>
        <svg viewBox="0 0 14 24"><line x1="7" y1="0" x2="7" y2="18" stroke-linecap="round"/><polygon points="3,16 7,22 11,16" fill="#B0A898"/></svg>
      </div>

      <div class="arch-node arch-node--brain">
        <div class="arch-node-title">⚡ FastAPI Gateway + ChatService</div>
        <div class="arch-node-desc">Session memory (TTLCache 24h) · Thread-safe model routing</div>
        <div class="arch-node-detail">Manages conversation history, validates requests, routes to RAGPipeline. Detects client disconnects via asyncio cancel tokens.</div>
      </div>

      <div class="arch-arrow">
        <div class="arch-arrow-label">Pipeline Entry</div>
        <svg viewBox="0 0 14 24"><line x1="7" y1="0" x2="7" y2="18" stroke-linecap="round"/><polygon points="3,16 7,22 11,16" fill="#B0A898"/></svg>
      </div>

      <div class="arch-node" style="border-color:var(--accent-tradeoff)">
        <div class="arch-node-title">🧭 Query Router (5-Type Intent Classifier)</div>
        <div class="arch-node-desc">factual · comparison · enumeration · procedural · conversational</div>
        <div class="arch-node-detail">Regex-based classifier (~0.5ms). Maps each intent to a specific RetrievalStrategy with tuned dense_top_k, bm25_top_k, rerank_top_n, and temperature.</div>
      </div>

      <div class="arch-arrow">
        <svg viewBox="0 0 14 24"><line x1="7" y1="0" x2="7" y2="18" stroke-linecap="round"/><polygon points="3,16 7,22 11,16" fill="#B0A898"/></svg>
      </div>

      <div class="arch-node">
        <div class="arch-node-title">💾 Semantic Cache (ChromaDB · cosine ≥ 0.95)</div>
        <div class="arch-node-desc">Sub-100ms cache hits bypass entire pipeline</div>
        <div class="arch-node-detail">Embeds query, probes dedicated ChromaDB collection. On hit: simulates SSE token streaming of cached answer. On miss: continues to retrieval.</div>
      </div>

      <div class="arch-arrow">
        <div class="arch-arrow-label">Cache Miss</div>
        <svg viewBox="0 0 14 24"><line x1="7" y1="0" x2="7" y2="18" stroke-linecap="round"/><polygon points="3,16 7,22 11,16" fill="#B0A898"/></svg>
      </div>

      <div class="arch-node">
        <div class="arch-node-title">✏️ Query Rewriter + Metadata Filter Inferer</div>
        <div class="arch-node-desc">Context-aware pronoun resolution · Department/Policy ID extraction</div>
        <div class="arch-node-detail">Rewrites follow-up queries ("Are there exceptions for it?" → "Exceptions for maternity leave policy"). Infers ChromaDB metadata filters (department=HR).</div>
      </div>

      <div class="arch-arrow">
        <svg viewBox="0 0 14 24"><line x1="7" y1="0" x2="7" y2="18" stroke-linecap="round"/><polygon points="3,16 7,22 11,16" fill="#B0A898"/></svg>
      </div>

      <div class="arch-parallel">
        <div class="arch-node">
          <div class="arch-node-title">🔍 Dense Vector Search</div>
          <div class="arch-node-desc">ChromaDB HNSW · bge-small-en-v1.5</div>
          <div class="arch-node-detail">384-dim embeddings. Cosine distance. Returns Top 10–30 semantically similar chunks. Supports metadata pre-filtering.</div>
        </div>
        <div class="arch-node">
          <div class="arch-node-title">📝 BM25 Lexical Search</div>
          <div class="arch-node-desc">rank-bm25 · Exact keyword matching</div>
          <div class="arch-node-detail">Captures exact policy codes (POL-402), financial thresholds ($1,500), and acronyms (COBRA, FMLA) that dense embeddings miss.</div>
        </div>
      </div>

      <div class="arch-merge-label">↓ Reciprocal Rank Fusion (k=60) ↓</div>

      <div class="arch-node" style="border-color:var(--accent-core)">
        <div class="arch-node-title">🎯 CrossEncoder Reranker (bge-reranker-large · CUDA)</div>
        <div class="arch-node-desc">Deep cross-attention scoring + Relative Score Threshold Filter</div>
        <div class="arch-node-detail">Scores each (query, chunk) pair through all 24 transformer layers. Drops chunks below top_score × min_ratio (0.45). Adds ~85ms GPU latency.</div>
      </div>

      <div class="arch-arrow">
        <div class="arch-arrow-label">Parent Expansion</div>
        <svg viewBox="0 0 14 24"><line x1="7" y1="0" x2="7" y2="18" stroke-linecap="round"/><polygon points="3,16 7,22 11,16" fill="#B0A898"/></svg>
      </div>

      <div class="arch-node">
        <div class="arch-node-title">📄 Context Compressor + LLM Generation (Ollama)</div>
        <div class="arch-node-desc">480-tok child → 2000-tok parent expansion · Grounded prompt synthesis</div>
        <div class="arch-node-detail">Expands surviving child chunks to full parent documents from docstore. Formats [Source N] context blocks. Streams tokens via Ollama.</div>
      </div>

      <div class="arch-arrow">
        <svg viewBox="0 0 14 24"><line x1="7" y1="0" x2="7" y2="18" stroke-linecap="round"/><polygon points="3,16 7,22 11,16" fill="#B0A898"/></svg>
      </div>

      <div class="arch-node" style="border-color:var(--accent-failure);border-width:2px">
        <div class="arch-node-title">🛡️ 4D Self-Reflection Verifier</div>
        <div class="arch-node-desc">35% Faith · 30% Comp · 20% Cit · 15% Coh → Composite ≥ 0.70</div>
        <div class="arch-node-detail">Heuristic evaluator checking numerical grounding, query entity coverage, [Source N] tag validity, and grammatical structure. If failed: triggers RetryEngine (max 2 cycles).</div>
      </div>

      <div class="arch-arrow">
        <div class="arch-arrow-label">Verified</div>
        <svg viewBox="0 0 14 24"><line x1="7" y1="0" x2="7" y2="18" stroke-linecap="round"/><polygon points="3,16 7,22 11,16" fill="#B0A898"/></svg>
      </div>

      <div class="arch-node arch-node--primary">
        <div class="arch-node-title">✅ SSE Token Stream + Citations + RAGTrace</div>
        <div class="arch-node-desc">Sub-1s TTFT · Async background cache write</div>
      </div>
    </div>
  </div>

  <span class="annotation--memory annotation" style="transform:rotate(-0.5deg)">This is the brain. Memorize this flow.</span>
</section>

<!-- ═══════════════════════════════════════════════════════════
     SECTION 03 — EXECUTION STORY
     ═══════════════════════════════════════════════════════════ -->
<section class="section" id="execution">
  <div class="section-number">03</div>
  <div class="section-eyebrow">Execution Path</div>
  <h2 class="section-title">Where Execution Actually Begins</h2>
  <p class="section-subtitle">Trace the complete journey of a user query from HTTP request to the first rendered token. Click any step to expand.</p>

  <div class="exec-timeline">
    <div class="exec-step">
      <div class="exec-step-number">Step 01</div>
      <div class="exec-step-title">HTTP Request Arrives</div>
      <div class="exec-step-body">Client sends <code>POST /api/chat/stream</code> with JSON payload: <code>{message, session_id, model, filters}</code>.</div>
      <span class="exec-step-file">backend/api/routes/chat.py</span>
      <div class="exec-step-detail">
        <p>FastAPI validates the request body via Pydantic <code>ChatRequest</code> model. An <code>asyncio.Event</code> cancel token is initialized for client disconnect detection. The request is handed to <code>ChatService.stream_query()</code>.</p>
      </div>
    </div>

    <div class="exec-step">
      <div class="exec-step-number">Step 02</div>
      <div class="exec-step-title">Session History Loaded</div>
      <div class="exec-step-body">Conversation history is retrieved from thread-safe <code>TTLCache</code> (1000 sessions, 24h TTL).</div>
      <span class="exec-step-file">backend/services/chat_service.py</span>
      <span class="exec-step-timing">~1ms</span>
      <div class="exec-step-detail">
        <p>Thread lock acquired. Previous user/assistant turns fetched for pronoun resolution and context-aware rewriting. Limited to last 5 turns × 3000 tokens.</p>
      </div>
    </div>

    <div class="exec-step">
      <div class="exec-step-number">Step 03</div>
      <div class="exec-step-title">Intent Classification & Strategy Selection</div>
      <div class="exec-step-body">QueryRouter classifies query into 5 categories using compiled regex patterns. Returns a tuned <code>RetrievalStrategy</code>.</div>
      <span class="exec-step-file">backend/rag/query_router.py</span>
      <span class="exec-step-timing">~0.5ms</span>
      <div class="exec-step-detail">
        <p><strong>Factual</strong>: top_k=10, rerank_top_n=4, min_ratio=0.45<br>
        <strong>Comparison</strong>: top_k=25, rerank_top_n=10, multi_query=True<br>
        <strong>Enumeration</strong>: top_k=30, rerank_top_n=12, exhaustive<br>
        <strong>Procedural</strong>: top_k=15, parent_expansion=True<br>
        <strong>Conversational</strong>: Bypasses DB entirely, streams greeting</p>
      </div>
    </div>

    <div class="exec-step">
      <div class="exec-step-number">Step 04</div>
      <div class="exec-step-title">Semantic Cache Probe</div>
      <div class="exec-step-body">Query is embedded and checked against ChromaDB <code>semantic_cache</code> collection. If cosine similarity ≥ 0.95: cache hit, pipeline exits early.</div>
      <span class="exec-step-file">backend/rag/semantic_cache.py</span>
      <span class="exec-step-timing">~8ms</span>
      <div class="exec-step-detail">
        <p>Cached results include the answer string, structured citations, and similarity score. On cache hit, the answer is word-split and streamed as simulated SSE tokens for UI consistency.</p>
      </div>
    </div>

    <div class="exec-step">
      <div class="exec-step-number">Step 05</div>
      <div class="exec-step-title">Query Rewrite & Metadata Filter Inference</div>
      <div class="exec-step-body">Pronouns are resolved using conversation history. Department names, policy IDs, and topic tags are extracted from the query.</div>
      <span class="exec-step-file">backend/rag/query_rewrite.py · filter_extractor.py</span>
      <span class="exec-step-timing">~15ms</span>
      <div class="exec-step-detail">
        <p><em>Example</em>: "Are there any exceptions for it?" → "Exceptions for maternal bereavement leave policy".<br>
        Filter inference: "IT department laptop policy" → <code>filters={"department": "IT"}</code>. Disambiguates English pronoun "it" vs "IT department" using keyword context.</p>
      </div>
    </div>

    <div class="exec-step">
      <div class="exec-step-number">Step 06</div>
      <div class="exec-step-title">Hybrid Retrieval & RRF Fusion</div>
      <div class="exec-step-body">Parallel dense vector (ChromaDB HNSW) and sparse BM25 search. Results merged via Reciprocal Rank Fusion: <code>Score(d) = Σ 1/(k+rank)</code>.</div>
      <span class="exec-step-file">backend/retrieval/hybrid.py</span>
      <span class="exec-step-timing">~40ms</span>
      <div class="exec-step-detail">
        <p>Dense captures semantic meaning. BM25 captures exact policy codes and numbers. RRF fuses ranks without requiring score normalization. If metadata filters return 0 results → <strong>Filter Relaxation Fallback</strong>: automatically drops filters and retries.</p>
      </div>
    </div>

    <div class="exec-step">
      <div class="exec-step-number">Step 07</div>
      <div class="exec-step-title">Cross-Encoder Reranking & Threshold Filtering</div>
      <div class="exec-step-body"><code>bge-reranker-large</code> scores 30 (query, chunk) pairs on CUDA GPU. <code>RelativeScoreThreshold</code>: chunks below <code>top_score × 0.45</code> are discarded.</div>
      <span class="exec-step-file">backend/retrieval/reranker.py</span>
      <span class="exec-step-timing">~85ms</span>
      <div class="exec-step-detail">
        <p>Cross-encoder provides deep query-document interaction via all 24 transformer attention layers. Example: top logit = 8.0, threshold = 3.6. Chunks scoring 2.1 are dropped. Minimum keep = 1 chunk.</p>
      </div>
    </div>

    <div class="exec-step">
      <div class="exec-step-number">Step 08</div>
      <div class="exec-step-title">Parent Context Expansion</div>
      <div class="exec-step-body">480-token child chunks are replaced by their 2000-token parent documents from the docstore.</div>
      <span class="exec-step-file">backend/rag/context_compression.py</span>
      <span class="exec-step-timing">~5ms</span>
      <div class="exec-step-detail">
        <p>Parent documents contain the full policy section including preambles, exceptions, and surrounding clauses. This resolves the "chunk size dilemma": embed small chunks for precision, generate from large chunks for completeness.</p>
      </div>
    </div>

    <div class="exec-step">
      <div class="exec-step-number">Step 09</div>
      <div class="exec-step-title">Grounded LLM Generation & Verification Loop</div>
      <div class="exec-step-body">Ollama generates tokens with strict grounding prompt. 4D Verifier evaluates. If composite &lt; 0.70: RetryEngine adjusts parameters and re-queries (max 2 retries).</div>
      <span class="exec-step-file">backend/rag/pipeline.py · verifier.py · retry_engine.py</span>
      <span class="exec-step-timing">~600–1800ms</span>
      <div class="exec-step-detail">
        <p><strong>Composite</strong> = 0.35 × Faithfulness + 0.30 × Completeness + 0.20 × Citations + 0.15 × Coherence.<br>
        On failure: RetryEngine broadens top_k (+10), tightens min_score_ratio (+0.10), injects dimension-specific prompt refinements, and re-executes from Step 06.</p>
      </div>
    </div>

    <div class="exec-step">
      <div class="exec-step-number">Step 10</div>
      <div class="exec-step-title">SSE Token Stream & Cache Write</div>
      <div class="exec-step-body">Verified tokens stream to the browser as SSE events. Background daemon thread writes the validated answer to the semantic cache.</div>
      <span class="exec-step-file">backend/api/routes/chat.py</span>
      <span class="exec-step-timing">TTFT &lt; 1s</span>
      <div class="exec-step-detail">
        <p>Event format: <code>data: {"type":"token","content":"..."}</code>. Final event: <code>data: {"type":"done","answer":"...","citations":[...],"trace":{...}}</code>. If user closes tab: <code>cancel_token.set()</code> halts GPU inference immediately.</p>
      </div>
    </div>
  </div>
</section>

<!-- ═══════════════════════════════════════════════════════════
     SECTION 04 — FOLLOW THE DATA
     ═══════════════════════════════════════════════════════════ -->
<section class="section" id="dataflow">
  <div class="section-number">04</div>
  <div class="section-eyebrow">Data Flow</div>
  <h2 class="section-title">Don't Memorize Functions.<br>Follow the Data.</h2>
  <p class="section-subtitle">Trace every transformation from raw file upload to verified SSE token stream.</p>

  <div class="data-journey">
    <div class="data-stage">
      <div class="data-stage-number">1</div>
      <div class="data-stage-content">
        <div class="data-stage-label">Source</div>
        <div class="data-stage-title">Raw File Upload</div>
        <div class="data-stage-desc">User uploads PDF, DOCX, TXT, MD, HTML, CSV, or JSON (up to 100MB) via <code>POST /api/documents</code>.</div>
        <div class="data-sample">Content-Type: multipart/form-data
file: "Employee_Handbook_2024.pdf" (4.2MB)
category: "hr_policies"</div>
      </div>
    </div>

    <div class="data-transform-arrow">↓ <span style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted)">LoaderFactory</span></div>

    <div class="data-stage">
      <div class="data-stage-number">2</div>
      <div class="data-stage-content">
        <div class="data-stage-label">Parsed</div>
        <div class="data-stage-title">Document Object with Raw Text</div>
        <div class="data-stage-desc">Format-specific loader extracts text, tables, page boundaries, and headers.</div>
        <div class="data-sample">Document {
  content: "Section 4.1 — Vacation Policy\nFull-time employees...",
  metadata: { source_file: "Employee_Handbook_2024.pdf",
               page_number: 14, file_type: "pdf" }
}</div>
      </div>
    </div>

    <div class="data-transform-arrow">↓ <span style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted)">MetadataExtractor</span></div>

    <div class="data-stage">
      <div class="data-stage-number">3</div>
      <div class="data-stage-content">
        <div class="data-stage-label">Enriched</div>
        <div class="data-stage-title">Document + Extracted Compliance Metadata</div>
        <div class="data-stage-desc">Regex scans extract department, effective date, policy ID, key entities, and topic tags.</div>
        <div class="data-sample">metadata: {
  department: "HR",
  effective_date: "2024-01-15",
  policy_id: "POL-2024-08",
  key_entities: ["full-time", "$1,500", "30 days"],
  topic_tags: ["vacation", "time-off", "rollover"]
}</div>
      </div>
    </div>

    <div class="data-transform-arrow">↓ <span style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted)">AdaptiveChunker</span></div>

    <div class="data-stage">
      <div class="data-stage-number">4</div>
      <div class="data-stage-content">
        <div class="data-stage-label">Chunked</div>
        <div class="data-stage-title">Hierarchical Child-Parent Chunk Pairs</div>
        <div class="data-stage-desc">480-token child chunks (64 overlap) embedded in ChromaDB. 2000-token parent docs stored in docstore. Linked by <code>parent_id</code>.</div>
        <div class="data-sample">Child Chunk { id: "chk_a1b2c3", parent_id: "par_x7y8z9",
  text: "Full-time employees receive 15 days of paid...",
  token_count: 480 }

Parent Doc  { id: "par_x7y8z9",
  text: "Section 4.1 — Vacation Policy\n...[full 2000 tokens]...",
  token_count: 2000 }</div>
      </div>
    </div>

    <div class="data-transform-arrow">↓ <span style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted)">EmbeddingService (bge-small-en-v1.5)</span></div>

    <div class="data-stage">
      <div class="data-stage-number">5</div>
      <div class="data-stage-content">
        <div class="data-stage-label">Indexed</div>
        <div class="data-stage-title">384-dim Vectors + BM25 Inverted Index</div>
        <div class="data-stage-desc">Child chunks become dense vectors in ChromaDB HNSW and tokenized entries in BM25. All metadata is flattened and indexed.</div>
        <div class="data-sample">ChromaDB: [0.0312, -0.0841, 0.1205, ...] (384 dims)
  + metadata: {department: "HR", policy_id: "POL-2024-08"}
BM25 Index: {"vacation": [chk_a1b2c3, chk_d4e5f6, ...]}</div>
      </div>
    </div>

    <div class="data-transform-arrow">↓ <span style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted)">HybridRetriever → RRF → CrossEncoder</span></div>

    <div class="data-stage">
      <div class="data-stage-number">6</div>
      <div class="data-stage-content">
        <div class="data-stage-label">Retrieved & Ranked</div>
        <div class="data-stage-title">ScoredChunk Objects (with rerank logits)</div>
        <div class="data-stage-desc">Candidates ranked by fusion score, then re-scored by cross-encoder, filtered by relative threshold.</div>
        <div class="data-sample">ScoredChunk {
  chunk: { id: "chk_a1b2c3", text: "Full-time employees..." },
  rrf_score: 0.0325,
  rerank_score: 7.84,   ← Deep cross-attention logit
  rank: 1
}</div>
      </div>
    </div>

    <div class="data-transform-arrow">↓ <span style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted)">ContextCompressor → LLM → Verifier</span></div>

    <div class="data-stage">
      <div class="data-stage-number">7</div>
      <div class="data-stage-content">
        <div class="data-stage-label">Output</div>
        <div class="data-stage-title">RAGResponse + RAGTrace + SSE Token Stream</div>
        <div class="data-stage-desc">Verified answer with structured citations, 4D verification report, and full execution telemetry.</div>
        <div class="data-sample">RAGResponse {
  answer: "Full-time employees receive 15 paid vacation... [Source 1]",
  citations: [{ source_file: "Employee_Handbook.pdf", section: "4.1" }],
  trace: { execution_time_ms: 847, verification_score: 0.91,
           cache_hit: false, retry_count: 0 }
}</div>
      </div>
    </div>
  </div>
</section>

<!-- ═══════════════════════════════════════════════════════════
     SECTION 05 — THE PEOPLE INSIDE THE MACHINE
     ═══════════════════════════════════════════════════════════ -->
<section class="section" id="components">
  <div class="section-number">05</div>
  <div class="section-eyebrow">Core Components</div>
  <h2 class="section-title">The People Inside the Machine</h2>
  <p class="section-subtitle">Each component has a specific role, specific inputs, and specific outputs. Know them like teammates.</p>

  <div class="component-grid">
    <div class="component-card component-card--orchestrator">
      <div class="component-metaphor">🎼 The Conductor</div>
      <div class="component-name">RAGPipeline</div>
      <div class="component-role">Master orchestrator coordinating every stage from query classification through verification and retry. Manages model routing, cache writes, and streaming.</div>
      <div class="component-io">
        <div><div class="component-io-label">Receives</div><div class="component-io-value">User query, session history, model name, filters</div></div>
        <div><div class="component-io-label">Produces</div><div class="component-io-value">RAGResponse with answer, citations, trace telemetry</div></div>
      </div>
      <div style="margin-top:0.75rem"><span class="exec-step-file">backend/rag/pipeline.py</span> <span style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted)">1,235 lines</span></div>
    </div>

    <div class="component-card">
      <div class="component-metaphor">🧭 The Dispatcher</div>
      <div class="component-name">QueryRouter</div>
      <div class="component-role">5-type intent classifier using compiled regex patterns. Maps each category to dynamic retrieval parameters in &lt;0.5ms.</div>
      <div class="component-io">
        <div><div class="component-io-label">Receives</div><div class="component-io-value">Raw query string + optional history</div></div>
        <div><div class="component-io-label">Produces</div><div class="component-io-value">QueryClassification (category, confidence, strategy)</div></div>
      </div>
    </div>

    <div class="component-card">
      <div class="component-metaphor">🔎 The Researcher</div>
      <div class="component-name">HybridRetriever</div>
      <div class="component-role">Executes parallel dense vector and BM25 keyword searches, fuses results via Reciprocal Rank Fusion (k=60).</div>
      <div class="component-io">
        <div><div class="component-io-label">Receives</div><div class="component-io-value">Query, top-k params, metadata filters</div></div>
        <div><div class="component-io-label">Produces</div><div class="component-io-value">Ranked list of ScoredChunk objects</div></div>
      </div>
    </div>

    <div class="component-card">
      <div class="component-metaphor">🎯 The Sharpshooter</div>
      <div class="component-name">CrossEncoderReranker</div>
      <div class="component-role">Re-scores candidates via deep cross-attention (bge-reranker-large on CUDA). Applies adaptive relative score threshold filtering.</div>
      <div class="component-io">
        <div><div class="component-io-label">Receives</div><div class="component-io-value">Query + 30 candidate chunks</div></div>
        <div><div class="component-io-label">Produces</div><div class="component-io-value">Top-N chunks surviving threshold filter</div></div>
      </div>
    </div>

    <div class="component-card">
      <div class="component-metaphor">🛡️ The Bouncer</div>
      <div class="component-name">SelfReflectionVerifier</div>
      <div class="component-role">Evaluates answer across 4 dimensions before letting it reach the user. Catches numerical hallucinations, missing aspects, and uncited claims.</div>
      <div class="component-io">
        <div><div class="component-io-label">Receives</div><div class="component-io-value">Query, answer, context chunks, citations</div></div>
        <div><div class="component-io-label">Produces</div><div class="component-io-value">VerificationReport (scores, passed, critique)</div></div>
      </div>
    </div>

    <div class="component-card">
      <div class="component-metaphor">🔄 The Adjuster</div>
      <div class="component-name">RetryEngine</div>
      <div class="component-role">On verification failure, dynamically broadens retrieval, tightens grounding, and injects dimension-specific prompt refinements. Hard cap: 2 retries.</div>
      <div class="component-io">
        <div><div class="component-io-label">Receives</div><div class="component-io-value">Attempt number, VerificationReport, current strategy</div></div>
        <div><div class="component-io-label">Produces</div><div class="component-io-value">Adjusted RetrievalStrategy + prompt refinement</div></div>
      </div>
    </div>
  </div>
</section>

<!-- ═══════════════════════════════════════════════════════════
     SECTION 06 — DESIGN DECISIONS & TRADEOFFS
     ═══════════════════════════════════════════════════════════ -->
<section class="section" id="decisions">
  <div class="section-number">06</div>
  <div class="section-eyebrow">Design Decisions</div>
  <h2 class="section-title">Every Architecture Is a Tradeoff</h2>
  <p class="section-subtitle">Why was it built this way, and what was sacrificed?</p>

  <div class="mt-2xl">
    <div class="decision-card">
      <div class="decision-label">Decision 01</div>
      <div class="decision-title">Hybrid Dense + BM25 with RRF instead of pure vector search</div>
      <div class="decision-flow">
        <div class="decision-row"><div class="decision-row-label">Why</div><div class="decision-row-value">Pure dense embeddings miss exact keyword matches like policy IDs (POL-402), dollar amounts ($1,500), and legal acronyms (COBRA, FMLA).</div></div>
        <div class="decision-row"><div class="decision-row-label">Alternative</div><div class="decision-row-value">Convex score combination: α × Dense + (1-α) × BM25.</div></div>
        <div class="decision-row"><div class="decision-row-label">Why Not</div><div class="decision-row-value">Dense cosine scores [0,1] and BM25 scores [0,∞) have incompatible distributions. Min-Max normalization is unstable with outlier queries.</div></div>
      </div>
      <div class="tradeoff-bar">
        <div class="tradeoff-gain"><div class="tradeoff-label">We gain</div>Robust rank-based fusion immune to score distribution drift</div>
        <div class="tradeoff-lose"><div class="tradeoff-label">We lose</div>In-memory BM25 index scales linearly with corpus size</div>
      </div>
    </div>

    <div class="decision-card">
      <div class="decision-label">Decision 02</div>
      <div class="decision-title">Cross-Encoder Reranker (bge-reranker-large) on GPU</div>
      <div class="decision-flow">
        <div class="decision-row"><div class="decision-row-label">Why</div><div class="decision-row-value">Bi-encoders embed query and document independently — no cross-token interaction. Cross-encoders pass the concatenated pair through all attention layers for deep relevance scoring.</div></div>
        <div class="decision-row"><div class="decision-row-label">Alternative</div><div class="decision-row-value">Cohere Rerank API or lightweight MiniLM cross-encoder.</div></div>
        <div class="decision-row"><div class="decision-row-label">Why Not</div><div class="decision-row-value">Cohere API violates our zero data egress requirement. MiniLM underperforms on dense legal text.</div></div>
      </div>
      <div class="tradeoff-bar">
        <div class="tradeoff-gain"><div class="tradeoff-label">We gain</div>Superior context precision (0.62 → 0.89) on policy queries</div>
        <div class="tradeoff-lose"><div class="tradeoff-label">We lose</div>~85ms GPU latency per query vs ~5ms for bi-encoder scoring</div>
      </div>
    </div>

    <div class="decision-card">
      <div class="decision-label">Decision 03</div>
      <div class="decision-title">Heuristic 4D Verifier instead of LLM-as-a-Judge</div>
      <div class="decision-flow">
        <div class="decision-row"><div class="decision-row-label">Why</div><div class="decision-row-value">Heuristic verification (regex, token overlap, numerical checks) executes in &lt;2ms. LLM-as-judge adds 800ms–1500ms per verification.</div></div>
        <div class="decision-row"><div class="decision-row-label">Alternative</div><div class="decision-row-value">Run a second LLM call to judge every response's faithfulness.</div></div>
        <div class="decision-row"><div class="decision-row-label">Why Not</div><div class="decision-row-value">Doubles inference latency and VRAM requirements. Unacceptable for sub-1s TTFT targets on local hardware.</div></div>
      </div>
      <div class="tradeoff-bar">
        <div class="tradeoff-gain"><div class="tradeoff-label">We gain</div>Near-zero verification latency; catches 95% of numerical hallucinations</div>
        <div class="tradeoff-lose"><div class="tradeoff-label">We lose</div>Cannot detect semantic contradictions or nuanced logical errors</div>
      </div>
    </div>

    <div class="decision-card">
      <div class="decision-label">Decision 04</div>
      <div class="decision-title">Hierarchical Chunking (480-tok child / 2000-tok parent)</div>
      <div class="decision-flow">
        <div class="decision-row"><div class="decision-row-label">Why</div><div class="decision-row-value">Small chunks optimize embedding precision. Large parent chunks give the LLM complete clause context including exceptions and prerequisites.</div></div>
        <div class="decision-row"><div class="decision-row-label">Alternative</div><div class="decision-row-value">Embed 2000-token chunks directly or use fixed 500-token chunks without parents.</div></div>
        <div class="decision-row"><div class="decision-row-label">Why Not</div><div class="decision-row-value">Large chunk embeddings suffer from vector dilution. Fixed small chunks truncate legal clauses causing hallucinated conditions.</div></div>
      </div>
      <div class="tradeoff-bar">
        <div class="tradeoff-gain"><div class="tradeoff-label">We gain</div>Best of both worlds: precise retrieval + complete generation context</div>
        <div class="tradeoff-lose"><div class="tradeoff-label">We lose</div>Additional complexity of managing a separate docstore + parent_id linkage</div>
      </div>
    </div>

    <div class="decision-card">
      <div class="decision-label">Decision 05</div>
      <div class="decision-title">100% Local Inference (Ollama) vs Cloud APIs</div>
      <div class="decision-flow">
        <div class="decision-row"><div class="decision-row-label">Why</div><div class="decision-row-value">Enterprise compliance documents contain sensitive PII, compensation data, and proprietary IP. Zero data leaves the internal network.</div></div>
        <div class="decision-row"><div class="decision-row-label">Alternative</div><div class="decision-row-value">OpenAI/Anthropic API for higher quality generation at lower engineering effort.</div></div>
        <div class="decision-row"><div class="decision-row-label">Why Not</div><div class="decision-row-value">Regulatory risk (GDPR, HIPAA, SOC2). Recurring per-token costs. Network dependency.</div></div>
      </div>
      <div class="tradeoff-bar">
        <div class="tradeoff-gain"><div class="tradeoff-label">We gain</div>Data sovereignty, zero API costs, offline operability</div>
        <div class="tradeoff-lose"><div class="tradeoff-label">We lose</div>Peak model quality (GPT-4 level reasoning) and higher inference latency on consumer GPU</div>
      </div>
    </div>
  </div>
</section>

<!-- ═══════════════════════════════════════════════════════════
     SECTION 07 — TESTING CONTRACTS
     ═══════════════════════════════════════════════════════════ -->
<section class="section" id="testing">
  <div class="section-number">07</div>
  <div class="section-eyebrow">Testing & Reliability</div>
  <h2 class="section-title">What Does This System Promise?</h2>
  <p class="section-subtitle">Each test is a behavioral contract. Here's what the project guarantees — and what it doesn't.</p>

  <div class="mt-2xl" style="display:flex;flex-direction:column;gap:var(--space-xl)">
    <div class="decision-card" style="border-left:3px solid var(--accent-followup)">
      <div class="decision-label" style="color:var(--accent-followup)">Tier 1 — Feature Tests</div>
      <div class="decision-title" style="font-size:var(--text-xl)">Core functionality behaves correctly</div>
      <div class="decision-flow" style="font-size:var(--text-sm);color:var(--text-secondary)">
        <p>✓ Chunkers produce token counts within configured bounds (480 child / 2000 parent)</p>
        <p>✓ Loaders extract text from all 7 supported formats (PDF, DOCX, TXT, MD, HTML, CSV, JSON)</p>
        <p>✓ QueryRouter classifies greetings as CONVERSATIONAL in &lt;0.5ms</p>
        <p>✓ Semantic cache returns hits for cosine similarity ≥ 0.95</p>
        <p>✓ RRF fusion produces deterministic ranked output with k=60</p>
      </div>
    </div>

    <div class="decision-card" style="border-left:3px solid var(--accent-tradeoff)">
      <div class="decision-label" style="color:var(--accent-tradeoff)">Tier 2 — Boundary Tests</div>
      <div class="decision-title" style="font-size:var(--text-xl)">Edge cases are handled safely</div>
      <div class="decision-flow" style="font-size:var(--text-sm);color:var(--text-secondary)">
        <p>✓ Empty query → 400 Bad Request (not an unhandled 500 crash)</p>
        <p>✓ Zero retrieved chunks → Returns grounded abstention message with zero hallucination</p>
        <p>✓ 100MB file upload → validated against MAX_FILE_SIZE_BYTES before processing</p>
        <p>✓ Unicode/CJK text → processed without character encoding errors</p>
        <p>✓ Negative rerank scores → min_keep=1 prevents completely empty context</p>
      </div>
    </div>

    <div class="decision-card" style="border-left:3px solid var(--accent-failure)">
      <div class="decision-label" style="color:var(--accent-failure)">Honest Gaps — Not Currently Tested</div>
      <div class="decision-title" style="font-size:var(--text-xl)">What we don't yet guarantee</div>
      <div class="decision-flow" style="font-size:var(--text-sm);color:var(--text-secondary)">
        <p>✗ Multi-node distributed ChromaDB clustering is not implemented</p>
        <p>✗ OCR for scanned image-only PDFs requires external Tesseract integration</p>
        <p>✗ No load testing for >50 concurrent streaming requests on single GPU</p>
        <p>✗ No automated regression for LLM output quality after model weights update</p>
      </div>
    </div>
  </div>
</section>

<!-- ═══════════════════════════════════════════════════════════
     SECTION 08 — BREAK THE SYSTEM
     ═══════════════════════════════════════════════════════════ -->
<section class="section" id="failures">
  <div class="section-number">08</div>
  <div class="section-eyebrow">Failure Modes</div>
  <h2 class="section-title">What Happens When Things Go Wrong?</h2>
  <p class="section-subtitle">Real failure scenarios from the actual codebase. Not generic error lists.</p>

  <div class="mt-xl">
    <div class="failure-card">
      <div class="failure-title">⚠ LLM Hallucinates a Dollar Amount</div>
      <div class="failure-flow">
        <div class="failure-step"><div class="failure-step-label">Normal</div><div class="failure-step-text">LLM generates: "Reimbursement is up to $5,000"</div></div>
        <div class="failure-step"><div class="failure-step-label">Detection</div><div class="failure-step-text">SelfReflectionVerifier extracts "$5,000" via <code>_NUMERICAL_REGEX</code>. Token not found in context chunks.</div></div>
        <div class="failure-step"><div class="failure-step-label">Scoring</div><div class="failure-step-text">Faithfulness capped at 0.35. Composite drops below 0.70. Verification fails.</div></div>
        <div class="failure-step"><div class="failure-step-label">Recovery</div><div class="failure-step-text">RetryEngine increments <code>min_score_ratio += 0.10</code>, decreases temperature, injects prompt: "Remove unsupported claims: $5,000".</div></div>
        <div class="failure-step"><div class="failure-step-label">Outcome</div><div class="failure-step-text"><span class="failure-handled">Handled</span> — Second generation adheres to context. Max 2 retry cycles with graceful best-effort fallback.</div></div>
      </div>
    </div>

    <div class="failure-card">
      <div class="failure-title">⚠ Ollama LLM Becomes Unreachable</div>
      <div class="failure-flow">
        <div class="failure-step"><div class="failure-step-label">Normal</div><div class="failure-step-text">Pipeline calls <code>req_llm.complete(prompt)</code></div></div>
        <div class="failure-step"><div class="failure-step-label">Detection</div><div class="failure-step-text">Connection refused or timeout exception caught in <code>pipeline.py</code></div></div>
        <div class="failure-step"><div class="failure-step-label">Recovery</div><div class="failure-step-text"><code>_fallback_synthesis()</code> activates — extracts first 2 sentences from each retrieved chunk with [Source N] tags.</div></div>
        <div class="failure-step"><div class="failure-step-label">Outcome</div><div class="failure-step-text"><span class="failure-handled">Handled</span> — Returns 100% grounded deterministic answer from retrieved chunks. Zero hallucination risk.</div></div>
      </div>
    </div>

    <div class="failure-card">
      <div class="failure-title">⚠ Metadata Filter Returns Zero Candidates</div>
      <div class="failure-flow">
        <div class="failure-step"><div class="failure-step-label">Normal</div><div class="failure-step-text">Query filtered by <code>department="Legal"</code> (incorrectly inferred)</div></div>
        <div class="failure-step"><div class="failure-step-label">Detection</div><div class="failure-step-text"><code>len(candidate_chunks) == 0</code> after filtered retrieval</div></div>
        <div class="failure-step"><div class="failure-step-label">Recovery</div><div class="failure-step-text">Filter Relaxation Fallback: drops all metadata filters, re-executes unfiltered hybrid search.</div></div>
        <div class="failure-step"><div class="failure-step-label">Outcome</div><div class="failure-step-text"><span class="failure-handled">Handled</span> — Prevents empty responses from over-aggressive filtering.</div></div>
      </div>
    </div>

    <div class="failure-card">
      <div class="failure-title">⚠ Client Disconnects Mid-Generation</div>
      <div class="failure-flow">
        <div class="failure-step"><div class="failure-step-label">Normal</div><div class="failure-step-text">User closes browser tab during LLM token streaming</div></div>
        <div class="failure-step"><div class="failure-step-label">Detection</div><div class="failure-step-text"><code>await request.is_disconnected()</code> returns True in SSE generator</div></div>
        <div class="failure-step"><div class="failure-step-label">Recovery</div><div class="failure-step-text"><code>cancel_token.set()</code> halts generation loops. GPU inference aborted.</div></div>
        <div class="failure-step"><div class="failure-step-label">Outcome</div><div class="failure-step-text"><span class="failure-handled">Handled</span> — Prevents GPU VRAM waste on abandoned requests.</div></div>
      </div>
    </div>

    <div class="failure-card">
      <div class="failure-title">⚠ Concurrent Model Switching Race Condition</div>
      <div class="failure-flow">
        <div class="failure-step"><div class="failure-step-label">Normal</div><div class="failure-step-text">User A requests <code>qwen2.5</code>, User B requests <code>llama3.1</code> simultaneously</div></div>
        <div class="failure-step"><div class="failure-step-label">Risk</div><div class="failure-step-text">Mutating shared <code>llm.model</code> attribute causes User A to receive output from User B's model</div></div>
        <div class="failure-step"><div class="failure-step-label">Prevention</div><div class="failure-step-text"><code>_LLMProxy</code> wraps each request with per-thread model isolation via <code>_llm_lock</code> + instance cache.</div></div>
        <div class="failure-step"><div class="failure-step-label">Outcome</div><div class="failure-step-text"><span class="failure-handled">Handled</span> — Thread-safe model routing without duplicate memory allocation.</div></div>
      </div>
    </div>
  </div>
</section>
"""

# Let's import the full 100 Q&As from a generator function in python
from generate_100_questions import get_all_modules_html

HTML_BODY_QA = get_all_modules_html()

HTML_TAIL = """
<!-- ═══════════════════════════════════════════════════════════
     SECTION 10 — CAN YOU DEFEND THE CODE?
     ═══════════════════════════════════════════════════════════ -->
<section class="section" id="defense">
  <div class="section-number">10</div>
  <div class="section-eyebrow">AI-Generated Code Stress Test</div>
  <h2 class="section-title">Can You Defend the Code?</h2>
  <p class="section-subtitle">Questions designed to expose shallow understanding. If you can't answer these, study the code before the interview.</p>

  <span class="annotation--warning annotation" style="transform:rotate(-1deg)">If you can't explain this, learn it before the interview.</span>

  <div class="mt-2xl">
    <div class="question-block">
      <div class="question-header">
        <span class="question-level question-level--4">Trap</span>
        <span class="question-text">Why does pipeline.py have a class called _LLMProxy? Why not just set llm.model directly?</span>
        <span class="question-toggle">+</span>
      </div>
      <div class="question-answer">
        <div class="answer-content">
          <div class="answer-section">
            <div class="answer-label">What You Must Understand</div>
            <div class="answer-text">In an async web server handling concurrent requests, directly mutating <span class="answer-code">self.llm.model = new_model</span> creates a race condition. Request A sets model to "llama3.1", awaits an async operation, Request B sets model to "mistral" — when A resumes, it generates with the wrong model. <span class="answer-code">_LLMProxy</span> wraps the LLM per-request with <span class="answer-code">_llm_lock</span> thread locks, guaranteeing model isolation without allocating duplicate model memory.</div>
          </div>
        </div>
      </div>
    </div>

    <div class="question-block">
      <div class="question-header">
        <span class="question-level question-level--4">Trap</span>
        <span class="question-text">Your verifier checks for "$5,000" using regex. What happens if the LLM writes "five thousand dollars" instead?</span>
        <span class="question-toggle">+</span>
      </div>
      <div class="question-answer">
        <div class="answer-content">
          <div class="answer-section">
            <div class="answer-label">Honest Answer</div>
            <div class="answer-text">The numerical regex <span class="answer-code">_NUMERICAL_REGEX</span> only catches formatted numbers ($5,000, 80%, 14 days). Natural language numbers ("five thousand") bypass the check. This is an honest gap. To fix it: add a number-word normalization step that converts English number words to digits before comparison, or use an NER-based entity matcher.</div>
          </div>
        </div>
      </div>
    </div>

    <div class="question-block">
      <div class="question-header">
        <span class="question-level question-level--4">Trap</span>
        <span class="question-text">What happens if you remove the RetryEngine entirely? Would the system still work?</span>
        <span class="question-toggle">+</span>
      </div>
      <div class="question-answer">
        <div class="answer-content">
          <div class="answer-section">
            <div class="answer-label">What You Must Understand</div>
            <div class="answer-text">Yes, it would still work. The retry loop is wrapped in <span class="answer-code">while attempt <= max_retries</span>. Without the RetryEngine, verification would still run, but failed answers would be returned as-is rather than corrected. The system degrades gracefully — you lose autonomous quality recovery but not core functionality. The RetryEngine is an optimization layer, not a structural dependency.</div>
          </div>
        </div>
      </div>
    </div>

    <div class="question-block">
      <div class="question-header">
        <span class="question-level question-level--4">Trap</span>
        <span class="question-text">You have both company_policy_rag/src/ and company_policy_rag/backend/. Did you combine two different projects?</span>
        <span class="question-toggle">+</span>
      </div>
      <div class="question-answer">
        <div class="answer-content">
          <div class="answer-section">
            <div class="answer-label">Strong Answer</div>
            <div class="answer-text"><span class="answer-code">src/</span> is the foundational engine layer: core Pydantic Settings, Ollama client, the fine-tuning pipeline, and deep-learning toolchains. <span class="answer-code">backend/</span> is the production FastAPI service layer wrapping those engines with async REST routes, Celery tasks, Pydantic DTOs, and session management. Both reference the unified <span class="answer-code">Settings</span> singleton in <span class="answer-code">src/config.py</span> as the single source of truth. It's an intentional layered architecture, not project duplication.</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ═══════════════════════════════════════════════════════════
     SECTION 11 — SCALE THE PROJECT
     ═══════════════════════════════════════════════════════════ -->
<section class="section" id="scale">
  <div class="section-number">11</div>
  <div class="section-eyebrow">Scalability</div>
  <h2 class="section-title">Scale the Project</h2>
  <p class="section-subtitle">Honest assessment: current limits → what breaks → how to evolve.</p>

  <div class="mt-2xl" style="display:flex;flex-direction:column;gap:var(--space-xl)">
    <div class="decision-card">
      <div class="decision-label">Current → 10× Users</div>
      <div class="decision-title">Single-Node Desktop → Departmental Production</div>
      <div class="decision-flow">
        <div class="decision-row"><div class="decision-row-label">Limit</div><div class="decision-row-value">~5–10 concurrent streaming requests before GPU VRAM queueing</div></div>
        <div class="decision-row"><div class="decision-row-label">Breaks</div><div class="decision-row-value">In-memory TTLCache lost on restart. Single Uvicorn process cannot saturate GPU.</div></div>
        <div class="decision-row"><div class="decision-row-label">Fix</div><div class="decision-row-value">Deploy Ollama on dedicated GPU server. 4 Uvicorn workers behind Nginx. Persist sessions in Redis.</div></div>
        <div class="decision-row"><div class="decision-row-label">Tradeoff</div><div class="decision-row-value">Adds operational infrastructure complexity. Redis becomes a new failure dependency.</div></div>
      </div>
    </div>

    <div class="decision-card">
      <div class="decision-label">10× → 100× Users</div>
      <div class="decision-title">Departmental → Enterprise-Wide Cluster</div>
      <div class="decision-flow">
        <div class="decision-row"><div class="decision-row-label">Limit</div><div class="decision-row-value">ChromaDB SQLite: single-writer lock contention. In-memory BM25 doesn't shard.</div></div>
        <div class="decision-row"><div class="decision-row-label">Breaks</div><div class="decision-row-value">Vector search latency degrades beyond 1M documents. LLM throughput caps at ~50 tokens/sec.</div></div>
        <div class="decision-row"><div class="decision-row-label">Fix</div><div class="decision-row-value">Migrate to distributed Qdrant/Milvus + Elasticsearch. Deploy vLLM fleet with continuous batching on A100/H100 GPUs.</div></div>
        <div class="decision-row"><div class="decision-row-label">Tradeoff</div><div class="decision-row-value">Significant engineering cost. Requires Kubernetes orchestration, observability stack (Prometheus + Grafana), and distributed caching.</div></div>
      </div>
    </div>
  </div>
</section>

<!-- ═══════════════════════════════════════════════════════════
     SECTION 12 — KNOWLEDGE GAP MAP
     ═══════════════════════════════════════════════════════════ -->
<section class="section" id="gaps">
  <div class="section-number">12</div>
  <div class="section-eyebrow">Knowledge Gaps</div>
  <h2 class="section-title">What Do I Still Need to Learn?</h2>
  <p class="section-subtitle">Check concepts as you master them. Progress is saved locally. <span class="gap-counter mono text-accent text-sm"></span></p>

  <div class="mt-2xl">
    <div class="gap-section">
      <div class="gap-level gap-level--must">🟢 Must Know — Not ready without these</div>
      <div class="gap-item"><input type="checkbox" class="gap-checkbox" data-gap-id="rrf"><div><div class="gap-concept">Reciprocal Rank Fusion (RRF) Formula & Intuition</div><div class="gap-reason">Used in backend/retrieval/hybrid.py. Must explain Score(d) = Σ 1/(k+rank) and why k=60.</div></div></div>
      <div class="gap-item"><input type="checkbox" class="gap-checkbox" data-gap-id="hier-chunk"><div><div class="gap-concept">Hierarchical Chunking: Child vs Parent Purpose</div><div class="gap-reason">480-tok child for embedding precision. 2000-tok parent for LLM context completeness. Must explain the Chunk Size Dilemma.</div></div></div>
      <div class="gap-item"><input type="checkbox" class="gap-checkbox" data-gap-id="4d-verifier"><div><div class="gap-concept">4D Verification Composite Formula & Pass Gates</div><div class="gap-reason">Composite = 0.35×Faith + 0.30×Comp + 0.20×Cit + 0.15×Coh. Pass requires composite ≥ 0.70 AND faith ≥ 0.65 AND comp ≥ 0.50 AND cit ≥ 0.50.</div></div></div>
      <div class="gap-item"><input type="checkbox" class="gap-checkbox" data-gap-id="grounding"><div><div class="gap-concept">Why strict grounding prevents hallucinations</div><div class="gap-reason">The system prompt enforces: cite [Source N] for every claim, abstain if sources insufficient. Verifier catches unsourced claims.</div></div></div>
      <div class="gap-item"><input type="checkbox" class="gap-checkbox" data-gap-id="sse"><div><div class="gap-concept">SSE Streaming Architecture & TTFT Optimization</div><div class="gap-reason">Sub-1s TTFT via async streaming. Must explain SSE event format, client disconnect detection, and cancel token pattern.</div></div></div>
    </div>

    <div class="gap-section">
      <div class="gap-level gap-level--strong">🟡 Strong Candidate — Impress the interviewer</div>
      <div class="gap-item"><input type="checkbox" class="gap-checkbox" data-gap-id="cross-vs-bi"><div><div class="gap-concept">Cross-Encoder vs Bi-Encoder Mechanics</div><div class="gap-reason">Bi-encoders embed independently (O(1) scoring). Cross-encoders process query+doc jointly through all attention layers (O(N) per candidate, deeper relevance).</div></div></div>
      <div class="gap-item"><input type="checkbox" class="gap-checkbox" data-gap-id="qlora"><div><div class="gap-concept">QLoRA: 4-bit NF4 Quantization + Low-Rank Adapters</div><div class="gap-reason">h = W₀x + (α/r)(B·A)x. Base frozen in NF4. Trainable A, B at r=16. Reduces VRAM from 28GB to ~5.5GB.</div></div></div>
      <div class="gap-item"><input type="checkbox" class="gap-checkbox" data-gap-id="bm25"><div><div class="gap-concept">BM25 Scoring Formula (TF-IDF with length normalization)</div><div class="gap-reason">Must explain why BM25 captures exact keywords that dense embeddings miss (policy codes, dollar amounts, acronyms).</div></div></div>
      <div class="gap-item"><input type="checkbox" class="gap-checkbox" data-gap-id="filter-relax"><div><div class="gap-concept">Filter Relaxation Fallback Pattern</div><div class="gap-reason">Over-aggressive metadata filters → 0 results → system auto-drops filters and retries. Prevents empty responses.</div></div></div>
    </div>

    <div class="gap-section">
      <div class="gap-level gap-level--deep">🔴 Deep Expert — Senior / Staff level questions</div>
      <div class="gap-item"><input type="checkbox" class="gap-checkbox" data-gap-id="gguf"><div><div class="gap-concept">GGUF v3 Binary Structure & Quantization Types</div><div class="gap-reason">Magic bytes 0x46554747, version 3, tensor alignment. Q4_K_M uses different quantization scales for attention vs feedforward layers.</div></div></div>
      <div class="gap-item"><input type="checkbox" class="gap-checkbox" data-gap-id="completion-mask"><div><div class="gap-concept">Completion-Only Loss Masking in SFTTrainer</div><div class="gap-reason">DataCollatorForCompletionOnlyLM sets prompt labels to -100. Gradients only on assistant tokens. Prevents memorizing prompt phrasing.</div></div></div>
      <div class="gap-item"><input type="checkbox" class="gap-checkbox" data-gap-id="hnsw"><div><div class="gap-concept">HNSW Graph Index Internals in ChromaDB</div><div class="gap-reason">Hierarchical Navigable Small World graphs. M parameter controls edge connectivity. ef_construction vs ef_search tradeoffs.</div></div></div>
    </div>
  </div>
</section>

<!-- ═══════════════════════════════════════════════════════════
     SECTION 13 — 10-MINUTE REVISION MODE
     ═══════════════════════════════════════════════════════════ -->
<section class="section" id="revision">
  <div class="section-number">13</div>
  <div class="section-eyebrow">Quick Revision</div>
  <h2 class="section-title">Before the Interview</h2>
  <p class="section-subtitle">Scan this entire section in 10 minutes. This is your final memory map.</p>

  <div class="revision-section">
    <div class="revision-grid">
      <div class="revision-block">
        <div class="revision-block-title">The Project Story</div>
        <div class="revision-flow">
          <div class="revision-flow-item"><span class="revision-flow-arrow">→</span> Problem: Naive RAG hallucinates on compliance policies</div>
          <div class="revision-flow-item"><span class="revision-flow-arrow">→</span> Solution: Hybrid retrieval + Cross-encoder + 4D Verification</div>
          <div class="revision-flow-item"><span class="revision-flow-arrow">→</span> How: Dense+BM25 → RRF → Rerank → Parent Expand → Verify → Retry</div>
          <div class="revision-flow-item"><span class="revision-flow-arrow">→</span> Result: Zero-hallucination grounded answers with source citations</div>
        </div>
      </div>

      <div class="revision-block">
        <div class="revision-block-title">The Architecture</div>
        <div class="revision-flow">
          <div class="revision-flow-item"><span class="revision-flow-arrow">→</span> Entry: Next.js → FastAPI SSE endpoint</div>
          <div class="revision-flow-item"><span class="revision-flow-arrow">→</span> Route: QueryRouter (5-type) → SemanticCache</div>
          <div class="revision-flow-item"><span class="revision-flow-arrow">→</span> Retrieve: ChromaDB + BM25 → RRF(k=60)</div>
          <div class="revision-flow-item"><span class="revision-flow-arrow">→</span> Rank: bge-reranker-large (CUDA) → threshold filter</div>
          <div class="revision-flow-item"><span class="revision-flow-arrow">→</span> Generate: Ollama → 4D Verifier → RetryEngine</div>
          <div class="revision-flow-item"><span class="revision-flow-arrow">→</span> Output: SSE token stream + Citations + RAGTrace</div>
        </div>
      </div>

      <div class="revision-block">
        <div class="revision-block-title">5 Critical Decisions</div>
        <div class="revision-flow">
          <div class="revision-flow-item"><span class="revision-flow-arrow">1.</span> Hybrid RRF → rank-based fusion without score normalization</div>
          <div class="revision-flow-item"><span class="revision-flow-arrow">2.</span> Cross-encoder → +85ms latency, precision 0.62→0.89</div>
          <div class="revision-flow-item"><span class="revision-flow-arrow">3.</span> Heuristic verifier → &lt;2ms vs 800ms LLM-as-judge</div>
          <div class="revision-flow-item"><span class="revision-flow-arrow">4.</span> Child-parent chunking → precise retrieval + complete context</div>
          <div class="revision-flow-item"><span class="revision-flow-arrow">5.</span> Local Ollama → zero data egress, full privacy</div>
        </div>
      </div>

      <div class="revision-block">
        <div class="revision-block-title">5 Failure Stories</div>
        <div class="revision-flow">
          <div class="revision-flow-item"><span class="revision-flow-arrow">1.</span> Hallucinated $$ → Verifier catches → Retry fixes</div>
          <div class="revision-flow-item"><span class="revision-flow-arrow">2.</span> LLM offline → _fallback_synthesis() extracts chunks</div>
          <div class="revision-flow-item"><span class="revision-flow-arrow">3.</span> Bad filter → 0 results → Filter relaxation fallback</div>
          <div class="revision-flow-item"><span class="revision-flow-arrow">4.</span> Client disconnect → cancel_token halts GPU</div>
          <div class="revision-flow-item"><span class="revision-flow-arrow">5.</span> Model race condition → _LLMProxy + thread lock</div>
        </div>
      </div>

      <div class="revision-block" style="grid-column:1/-1">
        <div class="revision-block-title">5 Most Dangerous Follow-ups</div>
        <div class="revision-qa">
          <div class="revision-q">1. "Why RRF instead of weighted score combination?"</div>
          <div class="revision-a">Cosine [0,1] vs BM25 [0,∞) — incompatible distributions. RRF uses ordinal ranks only. No normalization needed.</div>
        </div>
        <div class="revision-qa" style="margin-top:0.75rem">
          <div class="revision-q">2. "Why not use an LLM to verify every response?"</div>
          <div class="revision-a">800ms–1500ms per verification. Heuristic runs in &lt;2ms. LLM judge reserved for offline CI evaluation.</div>
        </div>
        <div class="revision-qa" style="margin-top:0.75rem">
          <div class="revision-q">3. "What's the biggest bottleneck at scale?"</div>
          <div class="revision-a">GPU cross-encoder reranking and single-stream LLM generation on consumer hardware. Fix: vLLM with continuous batching.</div>
        </div>
        <div class="revision-qa" style="margin-top:0.75rem">
          <div class="revision-q">4. "How do you detect semantic contradictions?"</div>
          <div class="revision-a">Honestly, we don't fully in the live path. Heuristic catches surface-level hallucinations. Semantic eval is offline only.</div>
        </div>
        <div class="revision-qa" style="margin-top:0.75rem">
          <div class="revision-q">5. "Can you defend the _LLMProxy pattern?"</div>
          <div class="revision-a">Prevents race condition: concurrent requests mutating shared model attribute. Thread lock + per-request proxy guarantees model isolation.</div>
        </div>
      </div>
    </div>

    <div style="text-align:center;margin-top:3rem">
      <svg width="200" height="100" viewBox="0 0 200 100" fill="none">
        <circle cx="100" cy="40" r="12" stroke="#E8A882" stroke-width="1.5" fill="none"/>
        <path d="M100 52 L100 80 M100 64 L85 76 M100 64 L115 76 M100 80 L92 98 M100 80 L108 98" stroke="#E8A882" stroke-width="1.5" stroke-linecap="round"/>
        <rect x="30" y="15" width="18" height="12" rx="2" stroke="#E8A882" stroke-width="0.8" fill="none" opacity="0.5"/>
        <rect x="55" y="8" width="18" height="12" rx="2" stroke="#E8A882" stroke-width="0.8" fill="none" opacity="0.5"/>
        <rect x="128" y="8" width="18" height="12" rx="2" stroke="#E8A882" stroke-width="0.8" fill="none" opacity="0.5"/>
        <rect x="155" y="15" width="18" height="12" rx="2" stroke="#E8A882" stroke-width="0.8" fill="none" opacity="0.5"/>
        <line x1="48" y1="21" x2="55" y2="16" stroke="#E8A882" stroke-width="0.5" opacity="0.4"/>
        <line x1="73" y1="14" x2="88" y2="32" stroke="#E8A882" stroke-width="0.5" opacity="0.4"/>
        <line x1="112" y1="32" x2="128" y2="14" stroke="#E8A882" stroke-width="0.5" opacity="0.4"/>
        <line x1="146" y1="16" x2="155" y2="21" stroke="#E8A882" stroke-width="0.5" opacity="0.4"/>
      </svg>
      <p style="font-family:var(--font-handwritten);font-size:1.4rem;color:var(--accent-core-soft);margin-top:0.5rem">"I don't need to memorize this project. I understand how it works."</p>
    </div>
  </div>
</section>

</div><!-- .page-wrapper -->

<footer style="text-align:center;padding:3rem;color:var(--text-light);font-size:var(--text-xs);font-family:var(--font-mono)">
  Enterprise Policy RAG — Interview Defense Playbook · Built from actual codebase analysis · 100 Technical Interview Questions
</footer>

<script src="script.js"></script>
</body>
</html>
"""

def main():
    target_path = r"c:\Users\jains\OneDrive\Desktop\Rag-chatbot\index.html"
    full_html = HTML_HEAD + "\n" + HTML_BODY_QA + "\n" + HTML_TAIL
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Successfully generated {target_path} (length: {len(full_html)} chars)")

if __name__ == "__main__":
    main()
