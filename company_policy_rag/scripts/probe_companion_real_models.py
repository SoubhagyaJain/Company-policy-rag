"""Real-model, isolated companion A/B and chat probe. Pass PDF and output directory."""

import json
import os
import sys
import time
import traceback
from pathlib import Path

root = Path(sys.argv[2]).resolve()
root.mkdir(parents=True, exist_ok=True)
os.environ["APP_STORAGE_DIR"] = str(root / "library")
os.environ["VISION_ENABLED"] = "false"
os.environ["SEMANTIC_CACHE_ENABLED"] = "false"
os.environ["PYTHONIOENCODING"] = "utf-8"
from fastapi.testclient import TestClient  # noqa: E402
from backend.api.main import app  # noqa: E402
from backend.api.dependencies import get_document_service, get_rag_pipeline  # noqa: E402
from backend.ingestion.loaders.loader_factory import _default_factory  # noqa: E402
from backend.ingestion.loaders.pdf import PDFLoader  # noqa: E402
from backend.retrieval.retrieval_cache import get_retrieval_cache  # noqa: E402
from backend.evaluation.retrieval_eval import stage_rankings  # noqa: E402
from src.config import settings  # noqa: E402

out = {
    "configuration": {
        "embedding_model": settings.embed_model,
        "llm_model": settings.llm_model,
        "vision_enabled": settings.vision_enabled,
        "reranker_enabled": settings.enable_reranker,
        "semantic_cache_enabled": settings.semantic_cache_enabled,
    },
    "retrieval": {},
    "conversation": [],
}


def save():
    (root / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def event(*a):
    print(*a, flush=True)


try:
    service = get_document_service()
    for loader in _default_factory.loaders:
        if isinstance(loader, PDFLoader):
            loader.image_asset_manager = service.image_asset_manager
    sample_pages = os.environ.get("COMPANION_EVAL_SAMPLE_PAGES")
    if sample_pages:
        import backend.services.document_service as document_module

        real_load = document_module.load_document
        selected = {int(value) for value in sample_pages.split(",")}

        def sample_load(*args, **kwargs):
            return [d for d in real_load(*args, **kwargs) if d.metadata.page_number in selected]

        document_module.load_document = sample_load
        out["configuration"]["sample_pages"] = sorted(selected)
        event("SAMPLE_PAGES", sorted(selected))
    client = TestClient(app)
    pdf = Path(sys.argv[1])
    event("MODELS", settings.embed_model, settings.llm_model)
    with pdf.open("rb") as f:
        response = client.post("/api/documents/upload", files={"file": (pdf.name, f, "application/pdf")})
    event("UPLOAD", response.status_code, response.text[:500])
    response.raise_for_status()
    doc_id = response.json()["document_id"]
    out["document_id"] = doc_id
    save()
    last = ""
    while True:
        status = service.get_ingestion_status(doc_id)
        current = f"{status.status} {status.progress} {status.current_stage}"
        if current != last:
            event("INGEST", current)
            last = current
        if status.status in ("READY", "FAILED", "REVIEW"):
            break
        time.sleep(2)
    out["ingestion"] = status.model_dump(mode="json")
    save()
    if status.status != "READY":
        raise RuntimeError("ingestion failed: " + str(status.error))
    chunks = list(service.docstore.values())
    event(
        "READY",
        len(chunks),
        service.vector_store.count(),
        sum(bool(c.metadata.extra.get("companion_navigation")) for c in chunks),
    )
    out["chunk_count"] = len(chunks)
    out["enriched_chunks"] = sum(bool(c.metadata.extra.get("companion_navigation")) for c in chunks)
    save()
    pipeline = get_rag_pipeline()
    index = service.bm25_index
    queries = [
        ("iron_law", "How does the iron law decompose execution time?", 45, "the data term"),
        (
            "selection",
            "What limits high quality data supply and motivates data selection?",
            440,
            "high-quality data supply",
        ),
        ("batching", "How does dynamic batching trade throughput against waiting time?", 730, "systems insight"),
        (
            "responsibility",
            "What role does engineering leadership play in responsible AI?",
            825,
            "responsible ai engineering",
        ),
        ("intensity", "How does arithmetic intensity affect accelerator bottlenecks?", 548, "arithmetic intensity"),
    ]
    labels = {}
    for qid, question, page, phrase in queries:
        matches = [c for c in chunks if c.metadata.page_number == page and phrase.casefold() in c.text.casefold()]
        labels[qid] = [c.id for c in matches]
        event("LABEL", qid, page, len(matches))
    out["labels"] = {
        qid: {"page": page, "phrase": phrase, "chunk_ids": labels[qid]} for qid, _, page, phrase in queries
    }
    save()
    original = index.metadata_fields
    baseline = tuple(x for x in original if x != "companion_navigation")
    for name, fields in [("before", baseline), ("after", original)]:
        index.reconfigure(metadata_fields=fields)
        get_retrieval_cache().clear()
        results = []
        for qid, question, page, phrase in queries:
            t = time.perf_counter()
            ctx = pipeline.run_retrieval_stages(question, filters={"document_id": doc_id})
            elapsed = (time.perf_counter() - t) * 1000
            ranks = stage_rankings(ctx.retrieval_stages or {})
            ids = ranks.get("fusion", [])
            target = set(labels[qid])
            rank = next((i for i, cid in enumerate(ids, 1) if cid in target), None)
            results.append(
                {
                    "id": qid,
                    "question": question,
                    "target_page": page,
                    "rank": rank,
                    "fusion_ids": ids[:10],
                    "fusion_pages": [
                        service.docstore[c].metadata.page_number for c in ids[:10] if c in service.docstore
                    ],
                    "latency_ms": round(elapsed, 2),
                    "stages": ctx.stage_timings,
                }
            )
            event("RETRIEVAL", name, qid, "rank", rank, "ms", round(elapsed, 2))
            out["retrieval"][name] = results
            save()
    index.reconfigure(metadata_fields=original)
    get_retrieval_cache().clear()
    turns = [
        "Explain the iron law of machine learning systems.",
        "What are its three terms, and what does each represent?",
        "Now explain dynamic batching in model serving.",
        "How does it affect latency and throughput?",
        "What does the book say about unicorn-powered quantum processors in 2045?",
    ]
    for i, question in enumerate(turns, 1):
        t = time.perf_counter()
        response = client.post(
            "/api/chat",
            json={
                "message": question,
                "session_id": "companion-real-model-test",
                "active_document_id": doc_id,
                "document_scope": "current_document",
                "model": settings.llm_model,
                "response_mode": "compact",
                "thinking_detail_level": "off",
            },
        )
        try:
            body = response.json()
        except Exception:
            body = {"raw": response.text}
        out["conversation"].append(
            {
                "turn": i,
                "question": question,
                "http_status": response.status_code,
                "wall_ms": round((time.perf_counter() - t) * 1000, 2),
                "response": body,
            }
        )
        save()
        event(
            "CHAT",
            i,
            response.status_code,
            "seconds",
            round(time.perf_counter() - t, 1),
            "citations",
            len(body.get("citations", [])),
            "answer",
            str(body.get("answer", ""))[:300],
        )
except Exception as exc:
    out["error"] = str(exc)
    out["traceback"] = traceback.format_exc()
    save()
    event("ERROR", out["traceback"])
    raise
finally:
    try:
        service.close()
    except Exception:
        pass
