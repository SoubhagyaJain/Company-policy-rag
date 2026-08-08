# Enterprise Policy RAG System: Complete Beginner's Prompt-by-Prompt Guide

> **Target System**: Production-Grade Enterprise Retrieval-Augmented Generation (RAG) Platform  
> **Tech Stack**: FastAPI (Python 3.11), PyTorch CUDA GPU, Ollama (`qwen2.5:7b`, `llama3.1:8b`), ChromaDB, BM25, BAAI Cross-Encoder Reranker (`bge-reranker-large`), Redis, Next.js 15 (React 19), Tailwind CSS, Docker Compose  
> **Target Path**: `c:\Users\jains\OneDrive\Desktop\Rag-chatbot\beginner_prompts_guide.md`

---

## Executive Overview & How to Use This Guide

### What is This Guide?
This guide is a complete, step-by-step master plan designed for developers and AI operators of any skill level. By following this document, you can copy-paste a carefully sequenced series of prompts into an AI coding assistant (such as **Claude 3.5 Sonnet**, **Antigravity**, **GPT-4o**, or **GitHub Copilot**) to build a complete, enterprise-ready **Enterprise Policy RAG System** completely from scratch.

### Architectural Overview
The target system is an end-to-end policy Q&A platform engineered for high accuracy, low latency, and production reliability.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Next.js 15 Frontend UI                             │
│ - Anthropic Cream Aesthetic (#FAF9F5) & Liquid Glass Styling               │
│ - Sub-1s SSE Token Streaming & Interactive Citation Drawer                  │
│ - Model Switcher (qwen2.5:7b vs llama3.1:8b) & Multi-Format Doc Manager     │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ HTTP / SSE Stream (POST fetch + ReadableStream)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FastAPI Backend API                              │
│ - Routes: /api/chat/stream, /api/documents, /api/admin/observability        │
│ - Async Processing, Structured Logging & Pydantic Validation                 │
└───────────────────┬─────────────────────────────────┬───────────────────────┘
                    │                                 │
                    ▼                                 ▼
┌───────────────────────────────────────┐ ┌───────────────────────────────────┐
│ Conversational Memory & Query Rewrite │ │ Multi-Format Ingestion & Chunking │
│ - Redis Sliding Window (Memory Dict)  │ │ - Loaders: PDF, DOCX, TXT, MD...  │
│ - Context-Aware Pronoun Resolution    │ │ - Chunkers: Heading, Parent-Child │
└───────────────────┬───────────────────┘ └─────────────────┬─────────────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Advanced Hybrid RAG Pipeline                          │
│ 1. Dense Vector Search (ChromaDB + sentence-transformers on CUDA GPU)       │
│ 2. Lexical Keyword Search (BM25Okapi Sparse Index)                           │
│ 3. Fusion: Reciprocal Rank Fusion (RRF with k=60)                           │
│ 4. Reranking: BAAI Cross-Encoder (bge-reranker-large loaded on CUDA GPU)    │
│ 5. Generation: Grounded Ollama LLM Execution & Citation Builder             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### How to Use the Prompts
1. **Sequential Execution**: Execute the prompts in exact numerical order (Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5). Do not skip phases.
2. **Copy and Paste**: Copy the exact text inside each prompt's code block and paste it directly into your AI assistant.
3. **Verify Each Step**: After your AI assistant generates the code, execute the provided terminal verification commands before proceeding to the next prompt.
4. **Clean Root Directory & Path Scoping**: Ensure you start with a clean project folder (e.g., `company_policy_rag/`). All prompt file paths (e.g. `backend/models/document.py`) are relative to your working project root directory `company_policy_rag/`. Do not create a nested `company_policy_rag/company_policy_rag/` directory.

---

## System Prerequisites & Local Setup

Before executing Phase 1, ensure your host environment meets the following software requirements:

* **Operating System**: Windows 11 / Linux (Ubuntu 22.04+) / macOS (Apple Silicon)
* **Python**: Version 3.11+ installed and added to system `PATH`
* **Node.js**: Version 20.x+ LTS and `npm` installed
* **NVIDIA GPU & Drivers** *(Recommended for GPU Acceleration)*: CUDA Toolkit 12.1 or 12.4 installed with PyTorch CUDA support.
* **Ollama Local LLM Runner**: Installed and running on `http://localhost:11434`.
  * Pull required models in your terminal:
    ```bash
    ollama pull qwen2.5:7b
    ollama pull llama3.1:8b
    ```
* **Docker & Docker Compose**: Installed for containerized deployment (Phase 5).

---

## Phase 1: Environment, CUDA GPU Foundation & Core Architecture

### Phase 1 - Prompt 1.1: Project Layout, Dependencies & PyTorch CUDA Device Foundation

#### Objective & Context
Initialize the standard multi-package workspace structure (`backend/`, `frontend/`, `shared/`, `tests/`, `docker/`), set up Python dependency management (`pyproject.toml`, `requirements.txt`), configure environment settings with Pydantic, and implement a PyTorch device utility that automatically detects and allocates CUDA GPU resources for local vector embeddings and reranking models.

#### Target Directory & File Paths
* `pyproject.toml`
* `requirements.txt`
* `.env.example`
* `.env`
* `backend/utils/device.py`
* `backend/utils/config.py`

#### Exact Copy-Paste Prompt Text
```text
I am building an Enterprise Policy RAG System. Please initialize Phase 1 by creating the core workspace directory layout, python package configuration, and CUDA GPU device initialization utilities.

Please create the following files with complete, robust code:

1. `pyproject.toml` and `requirements.txt`:
   Include exact dependencies for:
   - FastAPI (>=0.115.0), Uvicorn (>=0.32.0), Pydantic (>=2.9.0), pydantic-settings (>=2.5.0), python-multipart (>=0.0.9)
   - PyTorch (torch >=2.2.0), sentence-transformers (>=3.0.0), transformers (>=4.40.0), langchain (>=0.3.0), langchain-community (>=0.3.0)
   - ChromaDB (>=0.5.0, <1.0.0), rank-bm25 (>=0.2.2)
   - Redis (>=5.0.0), sse-starlette (>=2.1.0), httpx (>=0.27.0)
   - Document parsers: pypdf (>=5.0.0), pymupdf (>=1.24.0), python-docx (>=1.1.0), beautifulsoup4 (>=4.12.0)
   - Testing/Quality: pytest (>=8.0.0), pytest-asyncio (>=0.24.0), pyright, mypy

2. `.env.example` and `.env`:
   Define environment variables:
   - ENVIRONMENT=development
   - OLLAMA_BASE_URL=http://localhost:11434
   - OLLAMA_PRIMARY_MODEL=qwen2.5:7b
   - OLLAMA_SECONDARY_MODEL=llama3.1:8b
   - CHROMA_PERSIST_DIR=./storage/chroma_db
   - REDIS_URL=redis://localhost:6379/0
   - EMBEDDING_MODEL_NAME=sentence-transformers/bge-small-en-v1.5
   - RERANKER_MODEL_NAME=BAAI/bge-reranker-large
   - DEVICE=auto

3. `backend/utils/device.py`:
   Implement a thread-safe `get_compute_device()` function that detects whether CUDA GPU is available via PyTorch.
   - If CUDA is available, log GPU device name (e.g. NVIDIA GeForce RTX 3060) and return `torch.device("cuda")`.
   - If Apple MPS or CPU is detected, log and return appropriate device.
   - Provide helper functions `get_device_name()` and `clear_gpu_cache()`.

4. `backend/utils/config.py`:
   Create a `Settings` class using `pydantic_settings.BaseSettings` reading from `.env`.
   - Define all system parameters with type hints, defaults, and validation.
   - Include singleton instance getter `@lru_cache def get_settings() -> Settings`.

Ensure all files have complete imports, type annotations, and clear docstrings. Do not use pseudo-code or place TODO comments.
```

#### Verification & Terminal Test Commands
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
python -c "from backend.utils.device import get_compute_device; print('Compute device:', get_compute_device())"
python -c "from backend.utils.config import get_settings; print('Settings loaded for model:', get_settings().OLLAMA_PRIMARY_MODEL)"
```

---

### Phase 1 - Prompt 1.2: Core Domain Schemas & API Data Transfer Objects (DTOs)

#### Objective & Context
Define strong, type-safe Pydantic models for internal domain entities (Documents, Chunks, Search Results, Citations) and API request/response contracts (Chat Queries, Streaming Payloads, Document Metadata, Telemetry DTOs).

#### Target Directory & File Paths
* `backend/models/document.py`
* `backend/models/chunk.py`
* `backend/models/rag.py`
* `backend/models/api_dto.py`

#### Exact Copy-Paste Prompt Text
```text
Now implement the core domain schemas and data transfer objects (DTOs) for the backend. Create the following files in `backend/models/`:

1. `document.py`:
   Define Pydantic model `DocumentMetadata` and `Document`:
   - `id`: str (UUID)
   - `filename`: str
   - `file_type`: str (pdf, docx, txt, md, html, csv, json)
   - `file_size_bytes`: int
   - `content_hash`: str (SHA-256)
   - `created_at`: datetime
   - `chunk_count`: int
   - `metadata`: dict[str, Any]

2. `chunk.py`:
   Define Pydantic models `Chunk` and `ParentDocument`:
   - `Chunk`: `id`, `document_id`, `text`, `chunk_index`, `page_number` (optional int), `section_heading` (optional str), `section_path` (optional str), `strategy` (enum: recursive, semantic, markdown, heading, table, parent_child), `token_count`, `metadata`.
   - `ParentDocument`: `id`, `document_id`, `text`, `child_chunk_ids`, `section_heading`.

3. `rag.py`:
   Define model `SearchResult` and `Citation`:
   - `SearchResult`: `chunk` (Chunk), `dense_score` (float), `sparse_score` (float), `rrf_score` (float), `rerank_score` (float), `final_score` (float).
   - `Citation`: `citation_id` (int), `document_id` (str), `filename` (str), `page_number` (optional int), `section_heading` (optional str), `snippet` (str), `relevance_score` (float).

4. `api_dto.py`:
   Define API Request and Response DTOs:
   - `ChatRequest`: `message` (str), `session_id` (str), `model` (str, default "qwen2.5:7b"), `corpus_scope` (optional str), `temperature` (float = 0.1).
   - `ChatResponse`: `answer` (str), `session_id` (str), `citations` (list[Citation]), `latency_ms` (dict[str, float]), `model_used` (str).
   - `ObservabilityTelemetry`: `query`, `rewritten_query`, `retrieved_chunks` (list[dict]), `token_usage` (dict), `latency_ms` (dict).
   - `DocumentUploadResponse`: `document_id`, `filename`, `chunks_created`, `status`.

Ensure full Pydantic v2 compatibility (`Field`, `ConfigDict`, `field_validator`). No missing classes or placeholders.
```

#### Verification & Terminal Test Commands
```bash
python -c "from backend.models.api_dto import ChatRequest, ChatResponse; req = ChatRequest(message='What is PTO policy?', session_id='s1'); print('Schema valid:', req.model_dump())"
```

---

## Phase 2: Multi-Format Ingestion & Advanced Hybrid RAG Engine

### Phase 2 - Prompt 2.1: Multi-Format Document Loaders & Adaptive Chunking Strategy Engine

#### Objective & Context
Build specialized parsers for 7 distinct file formats (PDF, DOCX, TXT, Markdown, HTML, CSV, JSON) that extract structural hierarchy, headings, page numbers, and tables up to 100MB per file. Build an Adaptive Chunking Engine supporting Markdown-heading chunking, Table-aware code-fence chunking, Parent-Child hierarchical chunking, and Semantic distance chunking.

#### Target Directory & File Paths
* `backend/ingestion/loaders/base.py`
* `backend/ingestion/loaders/pdf_loader.py`
* `backend/ingestion/loaders/docx_loader.py`
* `backend/ingestion/loaders/text_md_loader.py`
* `backend/ingestion/loaders/html_csv_json_loader.py`
* `backend/ingestion/loaders/factory.py`
* `backend/ingestion/chunkers/adaptive_chunker.py`
* `backend/ingestion/pipeline.py`

#### Exact Copy-Paste Prompt Text
```text
Implement Phase 2 multi-format document ingestion and adaptive chunking engine in `backend/ingestion/`.

Please create:

1. `loaders/base.py`:
   Define abstract base class `BaseLoader` with abstract method `load(file_path: Path) -> LoadedDocument` containing full extracted text and detailed structural metadata.

2. Loaders (`pdf_loader.py`, `docx_loader.py`, `text_md_loader.py`, `html_csv_json_loader.py`):
   - PDF Loader: Use PyMuPDF (`fitz`) or `pypdf` to extract page-by-page text, preserving `page_number` in metadata and detecting headers.
   - DOCX Loader: Use `python-docx` to parse paragraphs, heading levels (H1-H6), and tables, building a section path.
   - Markdown Loader: Parse headings (`#`, `##`, `###`) and protect code blocks ('```').
   - HTML Loader: Use BeautifulSoup4 to extract clean structured text, preserving page titles and table elements.
   - CSV & JSON Loaders: Convert tabular rows into structured Markdown key-value pairs or Markdown tables so tabular context is preserved.

3. `loaders/factory.py`:
   Implement `LoaderFactory.get_loader(file_path: Path) -> BaseLoader` selecting parser based on file extension. Support file size validation up to 100 MB.

4. `chunkers/adaptive_chunker.py`:
   Implement `AdaptiveChunker` with strategies:
   - `RecursiveChunker`: Splits text by `["\n\n", "\n", ". ", " "]` with target 512 tokens, 64 token overlap.
   - `MarkdownHeadingChunker`: Splits by section headings, injecting section breadcrumbs (e.g. `[SECTION: HR Policies > PTO Accrual]`) onto every child chunk text.
   - `ParentChildChunker`: Creates large Parent blocks (1500-2000 tokens) stored in docstore, and splits them into smaller Child chunks (300-500 tokens) for vector indexing.
   - `TableAndCodeGuardChunker`: Ensures markdown tables and code fences are never cut mid-block.
   - `select_strategy(doc)`: Inspects document structure to select the best chunking strategy automatically.

5. `pipeline.py`:
   Implement `IngestionPipeline` which takes a file path, calculates SHA-256 hash, runs parser, executes adaptive chunker, and returns a list of `Chunk` objects ready for indexing. Include incremental indexing logic to skip unchanged file hashes.

Write complete Python code with error handling for corrupted files.
```

#### Verification & Terminal Test Commands
```bash
python -c "
from pathlib import Path
from backend.ingestion.pipeline import IngestionPipeline
import tempfile

with tempfile.NamedTemporaryFile(suffix='.md', mode='w', delete=False) as f:
    f.write('# Policy\n## Leave\nEmployees get 20 days paid leave annually.\n```python\n# code\npass\n```')
    f_path = Path(f.name)

pipeline = IngestionPipeline()
chunks = pipeline.process_file(f_path)
print(f'Processed {len(chunks)} chunks from test markdown file.')
f_path.unlink()
"
```

---

### Phase 2 - Prompt 2.2: Vector Store, BM25 Lexical Index, Hybrid RRF Search & BAAI Cross-Encoder Reranker

#### Objective & Context
Construct the hybrid retrieval engine combining ChromaDB dense vector search (`bge-small-en-v1.5` on PyTorch CUDA GPU) and BM25Okapi sparse keyword search. Implement Reciprocal Rank Fusion (RRF with $k=60$) to merge dense and sparse candidate pools, followed by a GPU-accelerated Cross-Encoder reranker (`BAAI/bge-reranker-large`) with score thresholding.

#### Target Directory & File Paths
* `backend/embeddings/vector_store.py`
* `backend/retrieval/bm25.py`
* `backend/retrieval/hybrid.py`
* `backend/retrieval/reranker.py`

#### Exact Copy-Paste Prompt Text
```text
Implement the advanced hybrid search and reranking engine in `backend/retrieval/` and `backend/embeddings/`.

Please create:

1. `backend/embeddings/vector_store.py`:
   Implement `ChromaVectorStore`:
   - Initialize persistent `chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)`.
   - Embed text using `SentenceTransformer("sentence-transformers/bge-small-en-v1.5", device=get_compute_device())`.
   - Methods: `add_chunks(chunks: list[Chunk])`, `delete_document(doc_id: str)`, `search(query: str, top_k: int = 30, metadata_filter: dict = None) -> list[SearchResult]`.

2. `backend/retrieval/bm25.py`:
   Implement `BM25Index`:
   - Use `rank_bm25.BM25Okapi` with stemming/tokenization over chunk texts.
   - Maintain persistent or in-memory inverse index mapping doc/chunk IDs.
   - Method: `search(query: str, top_k: int = 30) -> list[SearchResult]`.

3. `backend/retrieval/hybrid.py`:
   Implement `HybridRetriever`:
   - Executes `ChromaVectorStore.search` (top 30) and `BM25Index.search` (top 30) in parallel.
   - Combines results using Reciprocal Rank Fusion (RRF):
     $$RRF\_Score(d) = \frac{1}{60 + rank_{dense}(d)} + \frac{1}{60 + rank_{bm25}(d)}$$
   - Returns top 20 candidate chunks sorted by `rrf_score`.

4. `backend/retrieval/reranker.py`:
   Implement `CrossEncoderReranker`:
   - Load `SentenceTransformerRerank` (from `langchain` / `SentenceTransformers`) or `CrossEncoder("BAAI/bge-reranker-large", device=get_compute_device())`.
   - Method: `rerank(query: str, candidates: list[SearchResult], top_n: int = 6, min_relative_score: float = 0.40) -> list[SearchResult]`:
     1. Compute cross-encoder score for each (query, chunk.text) pair.
     2. Filter out chunks whose score is below `min_relative_score * max_score`.
     3. Sort by rerank score descending and return top_n chunks.

Provide robust exception handling and fallback if CUDA GPU VRAM is full.
```

#### Verification & Terminal Test Commands
```bash
python -c "
from backend.models.chunk import Chunk
from backend.embeddings.vector_store import ChromaVectorStore
from backend.retrieval.reranker import CrossEncoderReranker

print('Initializing vector store and reranker...')
store = ChromaVectorStore()
reranker = CrossEncoderReranker()
print('Vector store and reranker successfully initialized on GPU/CPU.')
"
```

---

### Phase 2 - Prompt 2.3: Grounded Citation Builder & RAG Pipeline Orchestrator

#### Objective & Context
Create the core RAG orchestration pipeline that coordinates multi-query processing, hybrid retrieval, cross-encoder reranking, prompt synthesis using local Ollama models (`qwen2.5:7b`), and structured citation extraction linking answer text directly to retrieved chunks using 1-based indexing (`[Source 1]`, `[Source 2]`).

#### Target Directory & File Paths
* `backend/rag/citations.py`
* `backend/rag/prompts.py`
* `backend/rag/pipeline.py`

#### Exact Copy-Paste Prompt Text
```text
Complete Phase 2 by implementing the grounded RAG pipeline and citation builder in `backend/rag/`.

Please create:

1. `prompts.py`:
   Define system prompts:
   - `SYSTEM_PROMPT_GROUNDED`: Instructs Ollama LLM to answer policy questions strictly based on the provided context chunks formatted as a 1-indexed list (`Source 1`, `Source 2`, ...). Requires inserting explicit inline citation tags `[Source 1]`, `[Source 2]` (using 1-based indexing) after statements backed by that chunk. Instructs model to explicitly state "I do not have sufficient information in the policy documents to answer this question" if the context does not contain the answer.

2. `citations.py`:
   Implement `CitationBuilder`:
   - Method `extract_citations(answer_text: str, retrieved_chunks: list[SearchResult]) -> list[Citation]`:
     1. Parses 1-indexed citation tags (e.g. `[Source 1]`, `[Source 2]`) from generated answer string using regular expressions.
     2. Maps 1-indexed tag numbers back to 0-indexed retrieved chunks array (`retrieved_chunks[N - 1]`) to retrieve exact chunk metadata (filename, page number, section heading, excerpt snippet, rerank relevance score).
     3. Returns clean list of structured `Citation` DTOs with 1-based `citation_id`.

3. `pipeline.py`:
   Implement `RAGPipeline`:
   - Methods:
     - `execute_query(query: str, corpus_scope: str = None, model: str = "qwen2.5:7b") -> ChatResponse`:
       1. Run `HybridRetriever.search(query)`.
       2. Run `CrossEncoderReranker.rerank(...)`.
       3. If parent-child chunking was used, fetch expanded parent text for synthesis context.
       4. Build system and user prompt with formatted 1-indexed context chunks.
       5. Call Ollama API (`http://localhost:11434/api/generate` or `api/chat`).
       6. Extract citations via `CitationBuilder`.
       7. Calculate execution timing breakdown (vector_ms, bm25_ms, rerank_ms, llm_ms, total_ms).
       8. Return `ChatResponse`.

Ensure complete implementation with type hints and error fallback handling.
```

#### Verification & Terminal Test Commands
```bash
python -c "from backend.rag.citations import CitationBuilder; print('CitationBuilder imported successfully.')"
```

---

## Phase 3: Conversational Memory, Query Rewriting & FastAPI SSE Streaming Server

### Phase 3 - Prompt 3.1: Redis Sliding-Window Memory & Context-Aware AI Query Rewriter

#### Objective & Context
Implement session-based conversational memory backed by Redis (with eager `.ping()` testing and a thread-safe in-memory dictionary fallback when Redis is offline). Implement an AI Query Rewriter that converts follow-up questions containing pronouns or ambiguous references (e.g., "What is its leave policy?") into self-contained, keyword-rich search queries.

#### Target Directory & File Paths
* `backend/services/memory_service.py`
* `backend/rag/query_rewrite.py`

#### Exact Copy-Paste Prompt Text
```text
Implement Phase 3 conversational memory and context-aware query rewriting in `backend/services/memory_service.py` and `backend/rag/query_rewrite.py`.

Please create:

1. `memory_service.py`:
   Implement `MemoryService`:
   - Instantiates Redis client via `redis.Redis.from_url(REDIS_URL)` and explicitly calls `self.redis_client.ping()` inside the `try...except` block during `__init__`. If `ping()` fails or raises a `ConnectionError`, immediately trigger the fallback to an in-memory dictionary (`dict[str, list[dict]]`) so offline Redis never causes unhandled runtime exceptions on subsequent requests.
   - Define async methods to ensure clean non-blocking invocation within FastAPI async generators:
     - `async def add_message(self, session_id: str, role: str, content: str) -> None`: Stores message in session history.
     - `async def get_history(self, session_id: str, max_messages: int = 10) -> list[dict]`: Retrieves last N messages (sliding window).
     - `async def clear_history(self, session_id: str) -> None`: Deletes session context.

2. `query_rewrite.py`:
   Implement `ContextAwareQueryRewriter`:
   - Method `rewrite_query(current_query: str, history: list[dict], model: str = "qwen2.5:7b") -> str`:
     1. If `history` is empty or query is self-contained, return `current_query` unchanged.
     2. Otherwise, construct a fast prompt to Ollama LLM providing the recent message history and the new user input.
     3. Instruct LLM to resolve pronouns (it, that, they, this policy) into explicit search entities (e.g. transform "What about remote work for them?" -> "What is the remote work policy for full time employees?").
     4. Return the rewritten search query.

Provide unit test verification methods inside the file under `if __name__ == '__main__':`.
```

#### Verification & Terminal Test Commands
```bash
python -c "
import asyncio
from backend.services.memory_service import MemoryService
from backend.rag.query_rewrite import ContextAwareQueryRewriter

async def test():
    mem = MemoryService()
    await mem.add_message('s1', 'user', 'What is the maternal leave policy?')
    await mem.add_message('s1', 'assistant', 'Maternal leave is 12 weeks paid.')
    history = await mem.get_history('s1')
    print('Session history retrieved:', len(history), 'messages.')

asyncio.run(test())
"
```

---

### Phase 3 - Prompt 3.2: FastAPI Streaming Web Server, SSE Engine & Admin Observability

#### Objective & Context
Construct the asynchronous FastAPI application server providing sub-1s initial token latency Server-Sent Events (`POST /api/chat/stream`), document management endpoints (`POST /api/documents/upload`, `GET /api/documents`), active model switcher endpoints (`GET /api/models`), and an Admin Observability endpoint (`GET /api/admin/observability`).

#### Target Directory & File Paths
* `backend/services/chat_service.py`
* `backend/services/document_service.py`
* `backend/api/routes/chat.py`
* `backend/api/routes/documents.py`
* `backend/api/routes/admin.py`
* `backend/api/routes/health.py`
* `backend/api/main.py`

#### Exact Copy-Paste Prompt Text
```text
Build the complete FastAPI backend server and streaming SSE routes in `backend/api/` and `backend/services/`.

Please create:

1. `services/chat_service.py`:
   Bridge RAG pipeline, memory service, and query rewriter.
   - Method `stream_chat(request: ChatRequest) -> AsyncGenerator[str, None]`:
     1. Await session history retrieval from MemoryService (`await memory_service.get_history(req.session_id)`).
     2. Rewrite user query via `ContextAwareQueryRewriter`.
     3. Perform hybrid search and cross-encoder reranking.
     4. Yield initial SSE event `event: start` with metadata and retrieved search chunks (TTFT < 1s).
     5. Stream generated LLM tokens from Ollama API as `event: chunk` SSE data frames.
     6. Parse final answer, yield `event: citation` with 1-indexed citations list, and `event: done` with total timing telemetry.
     7. Await saving user turn and assistant answer to MemoryService (`await memory_service.add_message(...)`).

2. `services/document_service.py`:
   Handle file upload up to 100MB, temporary saving, parsing via `IngestionPipeline`, vector index registration, and deletion.

3. Routes (`routes/chat.py`, `routes/documents.py`, `routes/admin.py`, `routes/health.py`):
   - `POST /api/chat/stream`: Accepts JSON body `ChatRequest` and returns `EventSourceResponse(chat_service.stream_chat(req))`.
   - `POST /api/documents/upload`: Accepts `UploadFile` (PDF, DOCX, TXT, MD, HTML, CSV, JSON up to 100MB). Uses `python-multipart` form parsing.
   - `GET /api/documents` & `DELETE /api/documents/{doc_id}`: Manage indexed documents.
   - `GET /api/models`: Returns available Ollama models (`qwen2.5:7b`, `llama3.1:8b`).
   - `GET /api/admin/observability`: Returns system metrics, active Chroma collection chunk counts, average query latencies, vector similarity scores, and rerank confidence score distributions.
   - `GET /api/health`: Health status of ChromaDB, Ollama, Redis, and PyTorch CUDA GPU.

4. `main.py`:
   Initialize FastAPI app with CORS middleware (`allow_origins=["*"]`), exception handlers, and route inclusion.

Write robust, production-grade async code.
```

#### Verification & Terminal Test Commands
```bash
# Terminal 1 (Start FastAPI backend server):
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 (Verify API endpoints):
curl http://localhost:8000/api/health
curl http://localhost:8000/api/models
```

---

## Phase 4: Anthropic-Inspired Next.js 15 UI Frontend

### Phase 4 - Prompt 4.1: Next.js 15 App Setup, Anthropic Cream Palette & Liquid Glass Styling

#### Objective & Context
Set up the Next.js 15 (React 19) App Router frontend with TypeScript and Tailwind CSS. Implement the requested Anthropic-inspired aesthetic featuring a soft cream background palette (`#FAF9F5`, `#F0EEE6`, `#191918`), liquid glass containers (`backdrop-blur-md bg-white/60 dark:bg-zinc-900/60 border border-amber-900/10`), elegant typography, and Framer Motion micro-animations.

#### Target Directory & File Paths
* `frontend/package.json`
* `frontend/tsconfig.json`
* `frontend/tailwind.config.js`
* `frontend/app/globals.css`
* `frontend/lib/theme.ts`
* `frontend/app/layout.tsx`
* `frontend/app/page.tsx`

#### Exact Copy-Paste Prompt Text
```text
Initialize Phase 4 by creating the Next.js 15 frontend project with an Anthropic-inspired design aesthetic in `frontend/`.

Please create:

1. `package.json` & `tsconfig.json`:
   Configure Next.js 15 (React 19), TypeScript, Tailwind CSS, Lucide React icons, Framer Motion (`framer-motion`), `react-markdown`, `remark-gfm`, `rehype-highlight`, and `clsx` / `tailwind-merge`.

2. `tailwind.config.js` & `app/globals.css`:
   - Color Palette Tokens:
     - Background Cream: `#FAF9F5` (Light mode), `#191918` (Dark mode)
     - Secondary Cream: `#F0EEE6` (Light mode), `#222220` (Dark mode)
     - Accent Warm Amber: `#D97706` / `#B45309`
     - Text Primary: `#2D2B28` / `#ECEAE5`
   - Custom Glass Utility Classes:
     - `.liquid-glass`: `backdrop-blur-md bg-white/60 dark:bg-zinc-900/60 border border-amber-900/10 shadow-sm rounded-2xl`
     - `.glass-card`: `backdrop-blur-sm bg-amber-50/40 dark:bg-zinc-800/40 border border-amber-900/5 rounded-xl`

3. `app/layout.tsx` & `app/page.tsx`:
   - Set up root HTML layout with serif headers font (`Newsreader` or `Playfair Display`) and sans-serif body font (`Inter` or `Plus Jakarta Sans`).
   - Include theme provider supporting light cream and dark charcoal modes.

Make sure Tailwind styles compile cleanly without missing variables.
```

#### Verification & Terminal Test Commands
```bash
cd frontend
npm install
npm run build
cd ..
```

---

### Phase 4 - Prompt 4.2: Streaming Chat Window, Citations Drawer, Model Switcher & Document Manager

#### Objective & Context
Build the complete interactive user interface: a custom React hook `useChatStream` for consuming SSE event streams over HTTP POST, a Markdown chat thread component with syntax highlighting, an interactive Citation Drawer that displays exact chunk sources and relevance scores using 1-based indexing, a Model Switcher dropdown (`qwen2.5:7b` vs `llama3.1:8b`), a multi-format Document Upload Manager (handling files up to 100MB), and a collapsible session sidebar.

#### Target Directory & File Paths
* `frontend/hooks/useChatStream.ts`
* `frontend/components/ChatWindow.tsx`
* `frontend/components/CitationCard.tsx`
* `frontend/components/ModelSwitcher.tsx`
* `frontend/components/DocumentManager.tsx`
* `frontend/components/Sidebar.tsx`

#### Exact Copy-Paste Prompt Text
```text
Build the interactive React UI components in `frontend/components/` and `frontend/hooks/`.

Please create:

1. `hooks/useChatStream.ts`:
   Custom React hook managing SSE stream connection to `POST /api/chat/stream`:
   - Uses `fetch` with `ReadableStream` (`response.body.getReader()`) or `@microsoft/fetch-event-source` to send `POST` requests with a JSON body (`ChatRequest`), avoiding standard browser `EventSource` which only supports `GET`.
   - Parses real-time SSE event frames (`event: start`, `event: chunk`, `event: citation`, `event: trace`, `event: done`).
   - Manages state: `messages`, `streamingAnswer`, `citations`, `isLoading`, `latencyMs`.

2. `components/ModelSwitcher.tsx`:
   Dropdown component placed in top bar allowing users to toggle between available local LLMs (`qwen2.5:7b` and `llama3.1:8b`). Passes selected model to `useChatStream`.

3. `components/CitationCard.tsx`:
   Interactive drawer/modal component:
   - Renders 1-indexed inline citation badges `[Source 1]`, `[Source 2]` (or `[1]`, `[2]`) corresponding to 1-based source chunk citations.
   - Clicking a citation opens a slide-over drawer showing document filename, page number, section breadcrumb, exact excerpt snippet, and rerank relevance percentage.

4. `components/DocumentManager.tsx`:
   Document upload modal supporting drag-and-drop file upload up to 100MB (PDF, DOCX, TXT, MD, HTML, CSV, JSON). Shows upload progress bar, indexed document list, chunk counts, and file delete buttons.

5. `components/Sidebar.tsx` & `components/ChatWindow.tsx`:
   - Collapsible sidebar for creating new chat sessions, viewing history, and clearing memory.
   - Main chat thread supporting markdown code blocks, streaming indicator, liquid glass input box, and quick suggested prompt pills.

Ensure modern React 19 functional component design with full TypeScript type safety.
```

#### Verification & Terminal Test Commands
```bash
cd frontend
npm run lint
npm run build
cd ..
```

---

## Phase 5: Production Containerization, Testing & Golden Evaluation Gate

### Phase 5 - Prompt 5.1: Multi-Stage Dockerfile & Production Docker Compose with GPU Support

#### Objective & Context
Containerize the entire application stack for production deployment. Create a multi-stage Dockerfile for FastAPI backend and Next.js frontend, and orchestrate services using Docker Compose with NVIDIA GPU container runtime reservation, Redis caching, and persistent volume storage.

#### Target Directory & File Paths
* `Dockerfile`
* `docker-compose.yml`
* `docker/entrypoint.sh`
* `.env.docker`

#### Exact Copy-Paste Prompt Text
````text
Implement Phase 5 production containerization by creating Docker configuration in project root directory.

Please create:

1. `Dockerfile`:
   Multi-stage build Dockerfile:
   - Backend Stage: Base `python:3.11-slim`. Install system dependencies (build-essential, git, libgomp1). Copy requirements, install dependencies, copy backend source code.
   - Frontend Stage: Base `node:20-alpine`. Install dependencies, build Next.js production bundle (`npm run build`), export production runner.

2. `docker-compose.yml`:
   Services configuration:
   - `backend`: FastAPI server running Uvicorn. Expose port 8000. Include NVIDIA GPU device reservation:
     ```yaml
     deploy:
       resources:
         reservations:
           devices:
             - driver: nvidia
               count: all
               capabilities: [gpu]
     ```
   - `frontend`: Next.js application container exposing port 3000.
   - `redis`: `redis:7-alpine` container exposing port 6379 with healthcheck.
   - `ollama`: `ollama/ollama:latest` container with GPU acceleration mounted to persistent volume `/root/.ollama`.

3. `docker/entrypoint.sh`:
   Startup shell script that waits for Redis and Ollama containers to be healthy, pulls `qwen2.5:7b` model if missing, runs database/index checks, and launches FastAPI server.

Ensure all ports, networks, and environment variables are properly mapped.
````

#### Verification & Terminal Test Commands
```bash
docker compose config
# Optional build test:
# docker compose build
```

---

### Phase 5 - Prompt 5.2: Type Safety, Automated Tests & Golden Evaluation Gate

#### Objective & Context
Enforce code quality across backend and frontend codebases (zero Pyright/mypy errors, zero ESLint errors). Implement unit and integration test suites using Pytest and Jest. Build an automated Golden Evaluation script that measures **Faithfulness (>=0.90)** and **Answer Relevancy (>=0.75)** against a golden dataset (`golden_subset_weak_guidebook.json`), serving as a strict quality gate.

#### Target Directory & File Paths
* `tests/unit/test_retrieval.py`
* `tests/unit/test_chunking.py`
* `tests/integration/test_api.py`
* `backend/evaluation/evaluator.py`
* `scripts/evaluate.py`
* `data/eval/golden_subset_weak_guidebook.json`

#### Exact Copy-Paste Prompt Text
```text
Finalize Phase 5 by writing unit/integration test suites, strict static type checks, and the automated Golden Evaluation Gate script.

Please create:

1. Tests (`tests/unit/test_retrieval.py`, `tests/unit/test_chunking.py`, `tests/integration/test_api.py`):
   - Pytest unit tests verifying chunking strategies, BM25 indexing, vector store queries, and hybrid RRF logic.
   - FastAPI integration tests using `httpx.AsyncClient` testing `/api/health`, `/api/models`, `/api/documents/upload`, and `/api/chat/stream`.

2. `backend/evaluation/evaluator.py` & `scripts/evaluate.py`:
   - Implement automated evaluation harness using local Ollama LLM as a judge:
     - `Faithfulness Metric`: Verifies generated claims against retrieved context (0.0 to 1.0 score). Target threshold: >= 0.90.
     - `Answer Relevancy Metric`: Verifies completeness and alignment of answer to original query (0.0 to 1.0 score). Target threshold: >= 0.75.
   - `scripts/evaluate.py`: Script that loads dataset `golden_subset_weak_guidebook.json`, executes test queries through `RAGPipeline`, computes average Faithfulness and Relevancy, prints score table, and exits with code 0 if thresholds pass, or code 1 if quality gate fails.

3. Type checking configuration:
   Ensure backend code passes `pyright` or `mypy` without errors, and frontend code passes `tsc --noEmit` and `eslint`.

Provide clean, reproducible python script execution commands.
```

#### Verification & Terminal Test Commands
```bash
# Run backend Pytest suite:
pytest tests/ -v

# Run backend type checking:
pyright backend/

# Run frontend linting & type checks:
cd frontend; npm run lint; npx tsc --noEmit; cd ..

# Run Golden Evaluation Quality Gate:
python scripts/evaluate.py --dataset data/eval/golden_subset_weak_guidebook.json
```

---

## Complete Verification & Acceptance Checklist

To verify that your newly generated system meets all requirements of the Enterprise Policy RAG System specification, verify each item in this final checklist:

| Quality Benchmark / Acceptance Requirement | Targeted Phase & Prompt | Terminal Verification Command | Expected Target Result |
|---|---|---|---|
| **CUDA PyTorch GPU Acceleration** | Phase 1 (Prompt 1.1) | `python -c "from backend.utils.device import get_compute_device; print(get_compute_device())"` | Prints `cuda` and detected GPU device name |
| **Multi-Format Document Upload (<=100MB)** | Phase 2 (Prompt 2.1) & Phase 3 (Prompt 3.2) | `curl -F "file=@sample.pdf" http://localhost:8000/api/documents/upload` | Returns HTTP 200 with `document_id` and `chunks_created` |
| **Hybrid Search (BM25 + Vector RRF)** | Phase 2 (Prompt 2.2) | `pytest tests/unit/test_retrieval.py` | All hybrid search and RRF tests pass |
| **BAAI Cross-Encoder Reranking** | Phase 2 (Prompt 2.2) | `curl http://localhost:8000/api/admin/observability` | Returns non-zero `rerank_score` values for top chunks |
| **Conversational Memory & Rewriting** | Phase 3 (Prompt 3.1) | `python backend/rag/query_rewrite.py` | Pronouns in follow-up queries resolved to explicit topics |
| **Sub-1s Initial Token Latency (SSE)** | Phase 3 (Prompt 3.2) & Phase 4 (Prompt 4.2) | `curl -N http://localhost:8000/api/chat/stream` | First `event: start` frame delivered within 1000ms |
| **Anthropic Cream Palette UI (#FAF9F5)** | Phase 4 (Prompt 4.1 & 4.2) | `cd frontend; npm run build; cd ..` | Next.js builds clean; UI displays cream liquid glass design |
| **Model Switcher (qwen2.5:7b / llama3.1:8b)**| Phase 4 (Prompt 4.2) | `curl http://localhost:8000/api/models` | Returns list containing both local Ollama models |
| **Faithfulness Score Quality Gate** | Phase 5 (Prompt 5.2) | `python scripts/evaluate.py --dataset data/eval/golden_subset_weak_guidebook.json` | Overall Faithfulness score **>= 0.90** |
| **Answer Relevancy Score Quality Gate** | Phase 5 (Prompt 5.2) | `python scripts/evaluate.py --dataset data/eval/golden_subset_weak_guidebook.json` | Overall Answer Relevancy score **>= 0.75** |
| **Containerized Deployment** | Phase 5 (Prompt 5.1) | `docker compose up --build` | FastAPI, Next.js, Redis, and Ollama services start cleanly |

---
*End of Beginner's Prompt-by-Prompt Guide.*
