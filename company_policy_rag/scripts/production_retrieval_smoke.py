#!/usr/bin/env python3
"""Run a self-contained smoke test through the production retrieval components.

The gate loads the public Markdown handbook, applies the production loader and
adaptive chunker, writes deterministic fallback embeddings to Chroma, builds
the production BM25 index, and evaluates the production hybrid RRF retriever.
It intentionally avoids network downloads so pull-request results are stable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.embeddings.embeddings import EmbeddingService  # noqa: E402
from backend.embeddings.vector_store import ChromaVectorStore  # noqa: E402
from backend.ingestion.chunkers.adaptive_chunker import AdaptiveChunker  # noqa: E402
from backend.ingestion.loaders.loader_factory import load_document  # noqa: E402
from backend.retrieval.bm25 import BM25SearchIndex  # noqa: E402
from backend.retrieval.hybrid import HybridRetriever  # noqa: E402
from backend.retrieval.vector import DenseVectorRetriever  # noqa: E402

DEFAULT_DATASET = PROJECT_ROOT / "data" / "eval" / "retrieval_smoke.json"


@dataclass(frozen=True)
class RetrievalCase:
    id: str
    query: str
    expected_section: str


def load_dataset(path: Path) -> tuple[dict[str, Any], list[RetrievalCase]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = [RetrievalCase(**case) for case in payload.get("cases", [])]
    if not cases:
        raise ValueError(f"Retrieval dataset contains no cases: {path}")
    return payload, cases


def _normalized(value: str | None) -> str:
    return " ".join(str(value or "").casefold().split())


def _is_relevant(hit: Any, expected_section: str) -> bool:
    expected = _normalized(expected_section)
    metadata = hit.chunk.metadata
    searchable = " ".join(
        _normalized(value)
        for value in (metadata.section_title, metadata.section_path, hit.chunk.text)
    )
    return expected in searchable


def run_smoke(dataset_path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    payload, cases = load_dataset(dataset_path)
    document_path = PROJECT_ROOT / str(payload["document"])
    if not document_path.is_file():
        raise FileNotFoundError(f"Demo document not found: {document_path}")

    file_hash = hashlib.sha256(document_path.read_bytes()).hexdigest()
    raw_documents = load_document(
        document_path,
        base_metadata={
            "document_id": "demo_northstar_handbook",
            "source_file": document_path.name,
            "file_path": str(document_path.relative_to(PROJECT_ROOT)),
            "file_hash": file_hash,
            "category": "policy",
        },
    )
    chunks = AdaptiveChunker(chunk_size=512, chunk_overlap=64).chunk(raw_documents)
    if not chunks:
        raise RuntimeError("Production chunker returned no chunks for the demo handbook.")

    # Exercise the production fallback that keeps retrieval available when a
    # local embedding model is absent. Marking initialization complete avoids
    # a network-dependent model lookup in CI.
    embedding_service = EmbeddingService(model_name="deterministic-ci", cache_enabled=False)
    embedding_service._model_loaded = True
    embedding_service._model = None
    for chunk, embedding in zip(
        chunks,
        embedding_service.embed_chunks([chunk.text for chunk in chunks]),
        strict=True,
    ):
        chunk.embedding = embedding

    temp_root = Path(tempfile.mkdtemp(prefix="rag-retrieval-smoke-"))
    started = time.perf_counter()
    try:
        vector_store = ChromaVectorStore(
            collection_name=f"retrieval_smoke_{uuid.uuid4().hex[:12]}",
            persist_dir=str(temp_root / "chroma"),
        )
        vector_store.add_chunks(chunks)
        bm25_index = BM25SearchIndex(storage_dir=str(temp_root / "bm25"))
        bm25_index.build_index(chunks)
        retriever = HybridRetriever(
            dense_retriever=DenseVectorRetriever(vector_store, embedding_service),
            bm25_index=bm25_index,
        )

        top_k = int(payload.get("top_k", 3))
        results: list[dict[str, Any]] = []
        reciprocal_ranks: list[float] = []
        hits = 0
        for case in cases:
            retrieved = retriever.retrieve(
                case.query,
                dense_top_k=max(top_k * 3, 10),
                bm25_top_k=max(top_k * 3, 10),
            )[:top_k]
            relevant_rank = next(
                (
                    rank
                    for rank, candidate in enumerate(retrieved, start=1)
                    if _is_relevant(candidate, case.expected_section)
                ),
                None,
            )
            if relevant_rank is not None:
                hits += 1
                reciprocal_ranks.append(1.0 / relevant_rank)
            else:
                reciprocal_ranks.append(0.0)

            results.append(
                {
                    **asdict(case),
                    "relevant_rank": relevant_rank,
                    "retrieved_sections": [
                        candidate.chunk.metadata.section_title
                        or candidate.chunk.metadata.section_path
                        or "Unsectioned"
                        for candidate in retrieved
                    ],
                }
            )

        hit_at_k = hits / len(cases)
        mean_reciprocal_rank = sum(reciprocal_ranks) / len(reciprocal_ranks)
        return {
            "schema_version": "1.0",
            "dataset": str(dataset_path.relative_to(PROJECT_ROOT)),
            "document": str(document_path.relative_to(PROJECT_ROOT)),
            "retrieval_path": "loader -> adaptive chunker -> Chroma + BM25 -> hybrid RRF",
            "embedding_mode": "production deterministic fallback",
            "case_count": len(cases),
            "chunk_count": len(chunks),
            "top_k": top_k,
            "metrics": {
                "hit_at_k": round(hit_at_k, 4),
                "mean_reciprocal_rank": round(mean_reciprocal_rank, 4),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            },
            "minimums": payload.get("minimums", {}),
            "cases": results,
        }
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assert-minimums", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_smoke(args.dataset.resolve())
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    if args.assert_minimums:
        metrics = result["metrics"]
        failures = [
            f"{name}={metrics.get(name, 0):.4f} < {floor:.4f}"
            for name, floor in result["minimums"].items()
            if float(metrics.get(name, 0)) < float(floor)
        ]
        if failures:
            print("Retrieval smoke gate failed: " + "; ".join(failures), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
