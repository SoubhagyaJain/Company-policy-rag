import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
RAG_DIR = ROOT_DIR / 'company_policy_rag'
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
