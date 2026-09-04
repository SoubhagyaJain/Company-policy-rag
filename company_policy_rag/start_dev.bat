@echo off
echo Starting Company Policy RAG Backend and Frontend...

start cmd /k ".venv\Scripts\activate.bat && uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload"
start cmd /k "cd frontend && npm run dev"

echo Both servers are starting in separate windows.
echo Open http://localhost:3000 to view your chatbot!
