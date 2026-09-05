<div align="center">

# Aperture RAG — Grounded Answers for High-Stakes Documents

**A production-grade Retrieval-Augmented Generation platform that refuses to hallucinate.**
It pairs hybrid retrieval with an agentic *verify-and-retry* loop, extracts code verbatim, runs fully local (no data leaves your machine), and accounts for every millisecond of every query in a full-screen observability plane.

![Backend](https://img.shields.io/badge/Backend-FastAPI%20·%20Python%203.11-009688?logo=fastapi&logoColor=white)
![Frontend](https://img.shields.io/badge/Frontend-Next.js%2016%20·%20React%2018-000000?logo=next.js)
![Vectors](https://img.shields.io/badge/Retrieval-ChromaDB%20·%20BM25%20·%20RRF%20·%20Cross--Encoder-FF4F00)
![LLM](https://img.shields.io/badge/LLM-Ollama%20(local,%20private)-7C3AED?logo=ollama&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-132%20modules-2EA043)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)

*Local-first · privacy-preserving · fully instrumented · CPU-friendly*

</div>

---

## The problem I set out to solve

LLMs are fluent but unfaithful. In legal, HR, compliance, and technical-architecture domains, a confident-but-wrong answer is *worse* than no answer — and "the model made it up" is not an acceptable failure mode.

A naive RAG bolt-on (embed → top-k → stuff-the-prompt) still hallucinates. It retrieves the wrong chunks, silently mangles code formatting, serves stale cached answers after a document is deleted, and gives you zero visibility into *why* a given answer came out the way it did.

**Aperture RAG is built around one principle:** *every answer must be traceable to retrieved evidence, or the system says so explicitly.* It pairs a high-precision retrieval stack with a self-verification loop, treats ingestion and deletion as first-class data-lifecycle operations, and instruments the entire request path so you can inspect the 16 stages behind any response.

---

## What makes it interesting

> These are the parts I'd want a reviewer to read first. Each one is a deliberate design decision, not a library call.

- **🔁 Agentic verify-and-retry loop.** After generation, a 4-dimensional verifier scores the answer on **faithfulness, completeness, citation coverage, and coherence**. If it fails, an autonomous retry engine *widens retrieval and regenerates* (bounded cycles) instead of shipping an unsupported claim. Every grounding claim is labelled `SUPPORTED` / `UNSUPPORTED` / `INFERRED`.

- **🔍 Hybrid retrieval that actually reranks.** Dense vectors (`BAAI/bge-small-en-v1.5`) and sparse **BM25** are fused with **Reciprocal Rank Fusion**, then re-scored by a **cross-encoder** (`BAAI/bge-reranker-large`). Cross-page section expansion pulls adjacent context so a chunk boundary never severs an answer.

- **⚡ Non-blocking ingestion for files up to 100 MB.** Uploads return in **~50 ms** and run parse → chunk → embed → index on a background queue; the client polls a live per-stage progress endpoint. A 1.6 MB doc expands to **~9,000 chunks** — the point is the request never blocks on it.

- **🗑️ Delete means *gone*.** Removing a document cascades across **the vector store, BM25 postings, the docstore, the stored file, image assets, the vision cache, the semantic answer cache, and conversation state** — so a deleted document can never resurface through a cached answer.

- **📋 Verbatim code extraction.** Chunking preserves indentation (it no longer collapses whitespace), and the generation contract *requires* code to be reproduced character-for-character inside fenced ` ```language ` blocks — rendered in the UI with syntax highlighting, line numbers, and copy-to-clipboard.

- **📊 Zero-latency observability.** Every request emits a trace to an **async write-behind queue** backed by **SQLite in WAL mode** — telemetry never sits on the request's critical path. A **16-stage latency waterfall** breaks down intake, memory resolution, rewrite, dense/sparse search, RRF, reranking, vision, TTFT, and SSE streaming.

- **🎛️ A UI that respects the work.** Real-time **SSE token streaming** with sub-second time-to-first-token, a WebGL hero shared as one persistent layer across tabs (no re-mount, no black flash), and frame-rate-independent motion that holds 60 fps and honors `prefers-reduced-motion`.

---

## Architecture

```mermaid
flowchart TD
    UserClient["Next.js 16 Client · React 18 · SSE"] -->|REST / SSE| Gateway["FastAPI Gateway · request-id + telemetry context"]
    Gateway --> Memory["Session Memory Resolver"]
    Memory --> Router["Query Router · 5-type classifier"]

    Router -->|conversational bypass| Direct["Direct LLM synthesis"]
    Router -->|retrieval required| Cache["Semantic Answer Cache"]

    Cache -->|hit| Cached["Stream cached answer + trace"]
    Cache -->|miss| Hybrid["Dense (BGE) + Sparse (BM25)"]

    Hybrid --> RRF["Reciprocal Rank Fusion"]
    RRF --> Rerank["Cross-Encoder Reranker · BGE-large"]
    Rerank --> Evidence["Evidence gate · TEXT / CODE / DIAGRAM / TABLE"]

    Evidence -->|visual page| Vision["Vision VLM fallback · optional"]
    Evidence -->|context ready| Assembly["Context compression & assembly"]
    Vision --> Assembly

    Assembly --> Stream["LLM generation + SSE streaming"]
    Stream --> Verifier["4-D verifier · faithfulness · completeness · citations · coherence"]

    Verifier -->|failed| Retry["Autonomous retry & precision refinement"]
    Verifier -->|passed| Telemetry["Telemetry event hub"]
    Direct --> Telemetry
    Cached --> Telemetry
    Retry --> Telemetry

    Telemetry --> WriteBehind["Async write-behind queue"]
    WriteBehind --> SQLite["SQLite · WAL + indices"]
    SQLite --> AdminAPI["Admin REST · /summary /health /queries /errors"]
    AdminAPI --> Dashboard["Full-screen Observability dashboard"]
```

### The 16-stage request path

`request intake → session-memory resolution → query rewrite → dense search → BM25 search → RRF fusion → cross-encoder rerank → cross-page expansion → evidence classification → (optional) vision extraction → context assembly → prompt construction → time-to-first-token → LLM generation → 4-D verification → SSE token streaming`

Each stage is timed independently and surfaced in the trace drawer.

---

## The thought process behind it

The interesting decisions weren't *which* libraries to use — they were the trade-offs at each fork. Here's the reasoning I'd walk an interviewer through:

**Why hybrid retrieval instead of just vectors?**
Pure dense retrieval misses exact-match terms — policy codes, function names, statute numbers — where the embedding blurs the very token you're searching for. BM25 nails those. Fusing both with RRF gives me dense recall *and* lexical precision without hand-tuning a weight between two incomparable score scales (RRF only needs the ranks). The cross-encoder then does the expensive, accurate part on a small candidate set, where its cost is affordable.

**Why a verifier if generation is already grounded?**
Because "grounded in the prompt" and "faithful to the prompt" are different claims. Retrieval can pull *plausible* chunks that don't actually answer the question, and the model will happily paper over the gap. The verifier is a second, cheaper pass that grades the answer against its own cited evidence — and, crucially, is allowed to *reject and retry* with wider retrieval. That converts a silent failure into a bounded, observable one.

**Why local-first (Ollama) instead of a hosted API?**
The target domains are exactly the ones where you can't ship documents to a third party. Running the whole stack — LLM, embeddings, reranker, vision — on the host means the data never leaves. It also forced honest engineering: everything defaults to **CPU** and degrades gracefully. Vision (VLM inference) is config-gated precisely because it's minutes-per-page on CPU; text and code extraction work fully without it.

**Why treat ingestion and deletion as first-class lifecycle operations?**
A RAG system is only as trustworthy as its *freshness*. If deleting a document leaves its chunks in BM25, or its answer in the semantic cache, the system will confidently cite a file that no longer exists. So delete had to cascade across *every* store at once. And because embedding a large PDF takes minutes, upload had to be non-blocking with real progress — otherwise the UX pushes people toward tiny documents, which defeats the point.

**Why spend so much on observability?**
Because "the answer felt wrong" is un-debuggable. If I can't see that stage 7 (rerank) demoted the right chunk, or that TTFT ballooned because the model cold-started, I'm guessing. The 16-stage waterfall turns every complaint into a specific, reproducible span — and because telemetry writes behind an async queue, measuring the system never slows it down.

---

## Tech stack

| Layer | Technology | Why |
| :--- | :--- | :--- |
| **Frontend** | Next.js 16 (Turbopack), React 18, Tailwind, Framer Motion, WebGL | SSE streaming UI, glass-over-hero design, 60 fps motion |
| **API** | FastAPI, Uvicorn, Python 3.11, Pydantic v2 | Async gateway + SSE, typed contracts |
| **Retrieval** | ChromaDB (dense) · BM25 (sparse) · RRF | Hybrid recall with precision reranking |
| **Embeddings / Rerank** | `bge-small-en-v1.5` · `bge-reranker-large` | Dense representations + cross-encoder scoring |
| **Generation** | Ollama (`qwen2.5:7b` default) | Local, privacy-first — no data leaves the host |
| **Vision (optional)** | Local VLM (`Qwen3-VL`) | Diagram / code-screenshot / table understanding |
| **Observability** | SQLite (WAL) + async write-behind thread | Persistent traces off the request critical path |

**Runs CPU-first.** The reranker defaults to CPU and uses a GPU automatically when present (`RERANKER_DEVICE=cuda`). Vision is config-gated (`VISION_ENABLED`) because VLM inference on CPU is minutes-per-page.

---

## Quick start

> Copy-paste friendly. If you have the prerequisites, you'll be chatting with your own documents in about five minutes.

### 1. Prerequisites

- **Python 3.11+** and **Node 18+**
- **[Ollama](https://ollama.com)** installed and running
- Redis is **not** required — the background ingestion queue runs in-process.

### 2. Pull the models

```bash
ollama pull qwen2.5:7b          # required — the generation model
ollama pull nomic-embed-text    # optional embedding model (bge-small is bundled)
```

### 3. Backend

All backend commands run from **`company_policy_rag/`**.

```bash
cd company_policy_rag

python -m venv .venv
.venv\Scripts\activate                      # Windows
# source .venv/bin/activate                 # macOS / Linux

pip install -r requirements.txt

# PyTorch is intentionally NOT pinned — install the build for your hardware:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu    # CPU
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 # GPU (CUDA 12.4)
```

### 4. Configure

Copy the example env and adjust if needed — the defaults work out of the box:

```bash
cp .env.example .env
```

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen2.5:7b
VISION_ENABLED=false          # leave false unless you have a GPU
RERANKER_DEVICE=cpu           # set to cuda when a GPU is available
```

### 5. Frontend

```bash
cd frontend && npm install && cd ..
```

### 6. Run it

```bash
# Windows one-shot (starts both servers):
start_dev.bat
```

…or manually, in two terminals from `company_policy_rag/`:

```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

```bash
cd frontend && npm run dev
```

Open **http://localhost:3000**, upload a document, and ask it something. 🎉

<details>
<summary><b>Troubleshooting the first run</b></summary>

- **`torch` / `sentence-transformers` import error** → you skipped the PyTorch step in §3. Install it, then `pip install -r requirements.txt` again.
- **Connection refused on `:11434`** → Ollama isn't running. Start it and re-check with `ollama list`.
- **First query is slow** → the embedding and reranker models load lazily on the first request; subsequent queries are fast.
- **Scanned PDF indexes 0 chunks** → that's a pure-image page. It needs the vision path (`VISION_ENABLED=true`), which realistically wants a GPU.
</details>

---

## API surface (selected)

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `POST` | `/api/chat/stream` | SSE token streaming with live reasoning trace |
| `POST` | `/api/documents/upload` | Non-blocking ingest (returns immediately, then poll status) |
| `GET` | `/api/documents/{id}/status` | Live per-stage ingestion progress |
| `DELETE` | `/api/documents/{id}` | Cascade delete across DB, caches, and conversation state |
| `GET` | `/api/admin/observability` | Canonical summary: percentiles, waterfall, health, traces |
| `GET` | `/api/admin/observability/queries/{id}` | Full single-trace inspection |
| `POST` | `/api/admin/observability/clear` | Purge telemetry |

Interactive OpenAPI docs are served at **http://localhost:8000/docs**.

---

## Project structure

```
company_policy_rag/
├── backend/
│   ├── api/            # FastAPI gateway, routers, middleware
│   ├── rag/            # pipeline, router, verifier, semantic cache, prompts
│   ├── retrieval/      # hybrid search, BM25, RRF, cross-encoder reranker
│   ├── ingestion/      # loaders (PDF/DOCX/MD/HTML/CSV/JSON/TXT) + chunkers
│   ├── embeddings/     # ChromaDB vector store + embedding service
│   ├── services/       # document lifecycle, telemetry (SQLite WAL)
│   └── vision/         # optional VLM extraction + asset/vision caches
├── frontend/           # Next.js 16 app — SSE chat, observability dashboard
└── tests/              # 132 test modules: unit, edge-case, adversarial
```

---

## Testing & quality

- **132 test modules** across unit, boundary, and adversarial tiers (`pytest`), plus a frontend suite.
- Backend and frontend are fully type-checked (Pydantic v2 · TypeScript strict).

```bash
cd company_policy_rag && pytest             # backend
cd company_policy_rag/frontend && npm test  # frontend
```

---

## Roadmap

- GPU-gated vision path for image-embedded code/diagram extraction at query time
- Per-document scoped cache invalidation (today, deletion clears caches wholesale for correctness)
- Evaluation harness: faithfulness / retrieval-recall regression gates in CI

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
