# Company Policy RAG Command Reference

This file collects the main commands you will need to set up, run, develop, and test the `company_policy_rag` project.

> Run most commands from the `company_policy_rag/` folder unless stated otherwise.

---

## 1. Repository setup

### Clone the repository

```bash
cd /path/to/your/workspace
git clone https://github.com/SoubhagyaJain/Rag-chatbot.git
cd Rag-chatbot/company_policy_rag
```

### Create and activate a Python virtual environment

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Windows CMD
.\.venv\Scripts\activate.bat
# macOS / Linux
source .venv/bin/activate
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### Install optional reranker dependencies

If you want the high-precision reranker pipeline, install:

```bash
# CPU
pip install torch sentence-transformers llama-index-postprocessor-sbert-rerank

# GPU (example for CUDA 12.4)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install sentence-transformers llama-index-postprocessor-sbert-rerank
```

### Install frontend dependencies

From `company_policy_rag/`:

```bash
cd frontend
npm install
cd ..
```

---

## 2. Environment files

### Copy example env files

```bash
cp .env.example .env
cp .env.docker.example .env.docker
```

- `.env` is used for local Python/Streamlit runs.
- `.env.docker` is used for Docker Compose runs.

---

## 3. Document indexing

### Build or update the Chroma index

```bash
python scripts/index_documents.py
```

### Rebuild the full index from scratch

```bash
python scripts/index_documents.py --force
```

### Index only policy PDFs

```bash
python scripts/index_documents.py --policies-only
```

### Index only legal PDFs

```bash
python scripts/index_documents.py --legal-only
```

### Index one or more specific PDF files

```bash
python scripts/index_documents.py --file data/policies/employee_handbook.pdf
```

### Dry-run index discovery

```bash
python scripts/index_documents.py --dry-run
```

---

## 4. Run the application locally

### Run the Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Open: `http://localhost:8501`

### Run the FastAPI backend directly

If you want to run the backend without Docker:

```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

> Replace `backend.api.main:app` with the actual backend app path if your repository uses a different import path.

---

## 5. Frontend commands

From `company_policy_rag/frontend`:

```bash
npm run dev    # Start Next.js development server
npm run build  # Build the frontend for production
npm run start  # Run the built production frontend
npm run lint   # Run eslint checks
```

---

## 6. Docker commands

### Build and start the full local stack

From `company_policy_rag/`:

```bash
docker compose up --build -d
```

### Stop the Docker stack

```bash
docker compose down
```

### Index documents inside Docker

```bash
docker compose run --rm app python scripts/index_documents.py
```

### Force rebuild index inside Docker

```bash
docker compose run --rm app python scripts/index_documents.py --force
```

### Run evaluation inside Docker

```bash
docker compose run --rm app python scripts/evaluate.py
```

---

## 7. Evaluation and metrics

### Run full golden-set evaluation

```bash
python scripts/evaluate.py
```

### Run a quick smoke evaluation

```bash
python scripts/evaluate.py --max-samples 5
```

### Run evaluation without the LLM judge

```bash
python scripts/evaluate.py --no-judge
```

### Run retrieval-only evaluation (CI-style smoke)

```bash
python scripts/evaluate.py --retrieval-only
```

### Evaluate a specific corpus

```bash
python scripts/evaluate.py --corpus guidebook
python scripts/evaluate.py --corpus policy
```

### Evaluate using a custom dataset file

```bash
python scripts/evaluate.py --dataset data/eval/golden_dataset_guidebook.json
```

---

## 8. Diagnostics and debugging

### Diagnose Chroma index state

```bash
python scripts/diagnose_index.py
```

### Diagnose code validation for one case or question

```bash
python scripts/diagnose_code_validation.py --case-id <case_id>
python scripts/diagnose_code_validation.py --question "Describe company leave policy"
python scripts/diagnose_code_validation.py --question "Describe company leave policy" --dry-run
```

### Debug retrieval for one question

```bash
python scripts/debug_retrieval_case.py --question "What is the vacation policy?"
```

### Compare weak evaluation cases

```bash
python scripts/compare_weak_cases.py
```

### Compare human judge agreement results

```bash
python scripts/compare_human_judge.py
```

---

## 9. Testing

### Run unit and integration tests

```bash
pytest tests/ -v --tb=short -q --no-header
```

### Run a specific test file

```bash
pytest tests/unit/test_api_dependencies.py -q
```

### Run all tests with coverage (if supported)

```bash
python -m pytest tests/ -v --cov=backend
```

---

## 10. Git and repository operations

### Add and commit changes

```bash
git add .
git commit -m "Describe your change"
```

### Push current branch to origin

```bash
git push origin main
```

### Check Git status

```bash
git status --short --branch
```

---

## 11. Useful extra commands

### Pull required local LLM models

If you use Ollama locally:

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### Install PostHog-compatible telemetry dependency

```bash
pip install "posthog>=2.4.0,<3.0.0"
```

### Verify Python import path

```bash
python -c "from src.retriever import get_reranker; print('OK' if get_reranker() else 'FAILED')"
```

---

## 12. Command usage notes

- Use `python scripts/index_documents.py` before running the app or evaluation.
- Use `.env` for local development and `.env.docker` for Docker.
- Use `docker compose up --build -d` when you want a self-contained local stack with backend, frontend, Redis, and Chroma.
- Use `npm run dev` only from the `company_policy_rag/frontend` folder.
- The `streamlit` command runs the main UI from `app/streamlit_app.py`.

---

If you want, I can also add this command reference into `company_policy_rag/README.md` directly.
