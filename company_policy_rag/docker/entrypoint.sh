#!/usr/bin/env sh
set -eu

OLLAMA_URL="${OLLAMA_BASE_URL:-http://host.docker.internal:11434}"
STREAMLIT_PORT="${STREAMLIT_SERVER_PORT:-8501}"
WAIT_SECONDS="${OLLAMA_WAIT_SECONDS:-60}"

echo "Waiting for Ollama at ${OLLAMA_URL} (up to ${WAIT_SECONDS}s)..."
elapsed=0
until curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; do
  if [ "$elapsed" -ge "$WAIT_SECONDS" ]; then
    echo "ERROR: Ollama not reachable at ${OLLAMA_URL}" >&2
    echo "Start Ollama on the host and pull models:" >&2
    echo "  ollama pull qwen2.5-coder-7b-policy" >&2
    echo "  ollama pull nomic-embed-text" >&2
    exit 1
  fi
  sleep 2
  elapsed=$((elapsed + 2))
done
echo "Ollama is ready."

if [ "${AUTO_INDEX_ON_START:-false}" = "true" ]; then
  echo "Checking Chroma index (AUTO_INDEX_ON_START=true)..."
  python - <<'PY'
import sys
sys.path.insert(0, "/app")
from src.indexing import index_exists, probe_chroma_index

probe = probe_chroma_index(clear_cache=True)
if probe["ready"]:
    print(f"Index ready: {probe['count']} chunks")
    sys.exit(0)
print(f"Index not ready ({probe.get('error') or 'empty'}) — running index_documents.py")
sys.exit(1)
PY
  if [ $? -ne 0 ]; then
    python scripts/index_documents.py
  fi
fi

export PYTHONPATH="/app${PYTHONPATH:+:$PYTHONPATH}"

exec streamlit run app/streamlit_app.py \
  --server.address=0.0.0.0 \
  --server.port="${STREAMLIT_PORT}" \
  --server.headless=true \
  --browser.gatherUsageStats=false