#!/usr/bin/env python3
"""Run the deterministic production regression suite used by RAG CI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CORE_TEST_PATHS = (
    # SQLite catalog, durable recovery and process-death regressions (offline).
    "tests/unit/test_document_catalog.py",
    "tests/unit/test_ingestion_recovery.py",
    "tests/integration/test_ingestion_process_recovery.py",
    # Deterministic regressions for production-critical corrections.
    "tests/unit/test_deployment_persistence.py",
    "tests/unit/test_deletion_integrity.py",
    "tests/unit/test_canonical_answer_delivery.py",
    "tests/unit/test_embedding_integrity.py",
    "tests/unit/test_security_containment.py",
    "tests/unit/test_model_artifact_integrity.py",
    "tests/unit/test_export_merge_modelfile.py",
    "tests/unit/test_adversarial_merger_gguf.py",
    "tests/unit/test_ollama_registration.py",
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
    # Retrieval, context assembly, generation and citation correctness.
    "tests/test_downstream_evidence_loss.py",
    "tests/test_retrieval_experiment_flags.py",
    "tests/test_policy_block_gating.py",
    "tests/test_generation_cancellation.py",
    "tests/unit/test_llm_client.py",
    "tests/unit/test_retrieval_cache_invalidation.py",
    "tests/unit/test_citation_engine.py",
    "tests/unit/test_retry_engine_adversarial.py",
    "tests/unit/test_ingestion_telemetry_counts.py",
    "tests/unit/test_retrieval_eval.py",
    "tests/unit/test_answer_eval.py",
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
