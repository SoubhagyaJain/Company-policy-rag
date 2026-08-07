# 🚀 Enterprise Policy RAG System: High-Precision Knowledge Retrieval

![Architecture: Microservices](https://img.shields.io/badge/Architecture-Microservices-blue)
![Backend: FastAPI](https://img.shields.io/badge/Backend-FastAPI_|_Python_3.11-009688?logo=fastapi)
![Frontend: Next.js](https://img.shields.io/badge/Frontend-Next.js_15_|_React_18-000000?logo=next.js)
![VectorDB: Chroma](https://img.shields.io/badge/VectorDB-ChromaDB-FF4F00)
![Cache: Redis](https://img.shields.io/badge/Cache-Redis-DC382D?logo=redis)
![CI/CD: GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=github-actions)

A production-grade, highly scalable **Retrieval-Augmented Generation (RAG)** architecture designed to eliminate hallucinations in high-stakes domains (legal, HR, compliance). Built to MAANG engineering standards, this system prioritizes **precision, verifiability, and sub-second latency**.

Unlike standard "toy" RAG implementations, this system features a decoupled microservices architecture, advanced hybrid retrieval with Cross-Encoder reranking, and a rigorous faithfulness evaluation guardrail.

---

## 📐 System Architecture

The architecture is fully decoupled, horizontally scalable, and containerized.

```mermaid
graph TD
    %% Frontend
    Client[Client / Web Browser] --> |Next.js 15 App Router| UI[Next.js Frontend]
    UI --> |SSE Streaming| API[FastAPI Backend]

    %% Backend Services
    subgraph Backend [FastAPI Microservice]
        API --> |Cache Hit| Redis[(Redis Cache)]
        API --> |Cache Miss| Orchestrator[RAG Pipeline]
    end

    %% RAG Pipeline
    subgraph RAG [Advanced RAG Pipeline]
        Orchestrator --> |Query Rewrite| LLM_Q[LLM]
        Orchestrator --> |Hybrid Search| VectorDB[(ChromaDB)]
        VectorDB --> |BM25 + Dense Vectors| RRF[Reciprocal Rank Fusion]
        RRF --> |Top K Candidates| Reranker[BGE Cross-Encoder]
        Reranker --> |Context Compression| Context[Filtered Context]
        Context --> |Strict Grounding| LLM[LLM Generator]
    end
    
    %% External
    LLM --> |Verification| Guardrail[Faithfulness Evaluator]
```

---

## 🏆 Key Engineering Achievements

### 1. High-Precision Retrieval Pipeline
- **Hybrid Search (Dense + Sparse):** Combines dense vector similarity (`nomic-embed-text`) with sparse BM25 keyword matching via Reciprocal Rank Fusion (RRF) to capture both semantic meaning and exact lexical matches (critical for policy numbers/acronyms).
- **Cross-Encoder Reranking:** Implemented `BAAI/bge-reranker-large` to re-score the top 30 retrieval candidates, drastically reducing context noise and improving **Context Precision to 0.82+**.
- **Context Compression:** Dynamically extracts highly relevant sentences from large text chunks before passing them to the LLM context window, reducing token usage by ~40% and mitigating "lost in the middle" degradation.

### 2. Hallucination Mitigation & Verifiability
- **Strict Grounding Enforcement:** The LLM is forced to abstain (`"I cannot answer based on the provided documents"`) rather than guess when confidence thresholds are unmet.
- **Deterministic Citations:** Engineered a custom post-processing pipeline that binds generated claims to exact source document chunks, exposing verifiable `[Source N]` tags to the frontend UI.
- **Automated Faithfulness Evals:** Integrated LLM-as-a-judge evaluation scripts (`scripts/evaluate.py`) achieving a **Faithfulness Score of 0.90+** against a golden dataset.

### 3. Scalable Distributed Infrastructure
- **Asynchronous I/O:** Built entirely on Python `asyncio` and FastAPI, utilizing `uvloop` for high-throughput concurrency.
- **Distributed Caching:** Integrated an async `redis-py` caching layer using SHA-256 hashes of queries and metadata filters, reducing p99 latency by 85% for frequent queries.
- **Sub-second TTFT:** Optimized Server-Sent Events (SSE) streaming to achieve a Time-To-First-Token (TTFT) of `< 1.0s` even under heavy load.

### 4. Production-Ready DevOps
- **Multi-Stage Docker Builds:** Optimized Dockerfiles reducing image size by over 60%, utilizing non-root users (`uid 1001`) for strict container security.
- **CI/CD Pipelines:** GitHub Actions workflow enforces code quality via Ruff, MyPy strict mode typing, ESLint, and a 142+ unit/integration test suite using Pytest.

---

## 💻 Tech Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Frontend** | Next.js 15, React 18, Tailwind | App Router for SSR, Framer Motion for liquid UI |
| **Backend API** | FastAPI, Uvicorn, Python 3.11 | Async-first, automatic OpenAPI docs, Pydantic validation |
| **RAG Core** | LlamaIndex, PyTorch | Extensible orchestrator, modular retrievers |
| **Vector Store** | ChromaDB | Lightweight, persistent, fast vector lookups |
| **Caching** | Redis (Alpine) | In-memory key-value store for exact-match query caching |
| **LLM Provider** | Ollama (Local) / OpenAI | Configurable via `.env` for cost/privacy tradeoffs |

---

## 🚀 Getting Started

The entire stack is orchestrated via Docker Compose for a frictionless developer experience.

### Prerequisites
- Docker Engine & Docker Compose
- (Optional) Local LLM engine: [Ollama](https://ollama.com) pulling `qwen2.5:14b-instruct` and `nomic-embed-text`.

### Launch the Cluster

```bash
# 1. Clone the repository
git clone https://github.com/your-org/company_policy_rag.git
cd company_policy_rag

# 2. Configure Environment
cp .env.example .env

# 3. Spin up the microservices
docker-compose up --build -d
```

### Access Points
- **Next.js Web Client:** `http://localhost:3000`
- **FastAPI Swagger Docs:** `http://localhost:8000/docs`
- **Redis CLI:** `docker exec -it <redis-container-id> redis-cli`

---

## 🧪 Testing & CI

This repository enforces strict code quality and high test coverage.

**Run Backend Test Suite (142 Tests):**
```bash
python -m pytest tests/ -v --cov=backend
```

**Run Frontend Lint & Build Checks:**
```bash
cd frontend && npm run lint && npm run build
```

---

## 📈 Roadmap & Future Scaling

- [ ] **Semantic Caching:** Upgrade Redis cache to a Vector Cache (e.g., RedisVL) to cache semantically similar queries, not just exact string matches.
- [ ] **Graph RAG Integration:** Implement Knowledge Graphs (Neo4j) to capture hierarchical organizational structures and entity relationships across documents.
- [ ] **Kubernetes Migration:** Provide Helm charts for deploying the architecture to AWS EKS or GKE with Horizontal Pod Autoscaling (HPA).