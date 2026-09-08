#!/usr/bin/env python3
"""Run the deterministic production regression suite used by RAG CI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CORE_TEST_PATHS = (
    "tests/test_conversation_interpreter.py",
    "tests/unit/test_conversation_quality_regression.py",
    "tests/test_adversarial_conversation_rag.py",
    "tests/test_conversation_aware_rag.py",
    "tests/test_evidence_continuity.py",
    "tests/test_memory.py",
    "tests/unit/test_clean_startup.py",
    "tests/unit/test_document_ingestion_hardening.py",
    "tests/unit/test_document_legacy_delete.py",
    "tests/test_document_upload.py",
    "tests/test_document_deduplication.py",
    "tests/test_production_observability.py",
    "tests/test_production_observability_full.py",
    "tests/unit/test_api_admin.py",
    "tests/unit/test_conversation_benchmark.py",
    "tests/unit/test_production_retrieval_smoke.py",
)


def main(extra_args: list[str] | None = None) -> int:
    missing = [path for path in CORE_TEST_PATHS if not (PROJECT_ROOT / path).is_file()]
    if missing:
        print("Core test manifest contains missing paths: " + ", ".join(missing), file=sys.stderr)
        return 2
    os.chdir(PROJECT_ROOT)
    return pytest.main([*CORE_TEST_PATHS, "-q", *(extra_args or [])])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
