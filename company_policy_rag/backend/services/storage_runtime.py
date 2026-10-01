"""Runtime probes for the Storage tab: RAM, VRAM attribution and loaded models.

Everything here reports what a runtime actually exposes. Where a figure cannot be
measured the field is ``None`` and the caller shows it as unavailable; nothing is
derived from guesses.
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime
from typing import Any

from backend.utils.logging import logger
from src.config import settings

# Ollama reports a pinned model (keep_alive=-1) with an expiry centuries away.
_PINNED_EXPIRY_YEARS = 50
_VISION_NAME_MARKERS = ("vl", "vision", "llava", "moondream")
_EMBED_NAME_MARKERS = ("embed", "bge-m3")

_weight_bytes_cache: dict[int, int | None] = {}


def system_memory() -> dict[str, int | None]:
    """RAM of this process and of the machine, or ``None`` fields without psutil."""
    try:
        import psutil

        virtual = psutil.virtual_memory()
        return {
            "process_rss_bytes": int(psutil.Process().memory_info().rss),
            "system_total_bytes": int(virtual.total),
            "system_available_bytes": int(virtual.available),
        }
    except Exception:
        return {"process_rss_bytes": None, "system_total_bytes": None, "system_available_bytes": None}


def torch_reserved_bytes() -> int | None:
    """VRAM this process's PyTorch allocator holds, without importing or initialising CUDA."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_initialized():
            return 0
        return int(torch.cuda.memory_reserved())
    except Exception:
        return None


def weight_bytes(model: Any) -> int | None:
    """Bytes of a loaded torch model's parameter tensors, measured once per model object."""
    if model is None:
        return None
    key = id(model)
    if key in _weight_bytes_cache:
        return _weight_bytes_cache[key]
    total: int | None = None
    for candidate in (model, getattr(model, "model", None)):
        parameters = getattr(candidate, "parameters", None)
        if not callable(parameters):
            continue
        try:
            total = sum(int(p.numel()) * int(p.element_size()) for p in parameters())
            break
        except Exception as exc:
            logger.debug("Could not measure model weights: %s", exc)
    _weight_bytes_cache[key] = total
    return total


def _device_of(model: Any, fallback: str | None = None) -> str | None:
    for candidate in (model, getattr(model, "model", None)):
        device = getattr(candidate, "device", None)
        if device is not None:
            return str(device).split(":")[0]
    return fallback


def _iso(epoch: float | None) -> str | None:
    return datetime.fromtimestamp(epoch, UTC).isoformat() if epoch else None


def model_purpose(name: str, families: list[str] | None = None) -> str:
    """Chat, Embedding or Vision, from the configured roles first and the model name second."""
    lower = name.lower()
    if lower == str(settings.llm_model).lower():
        return "Chat"
    if any(marker in lower for marker in _EMBED_NAME_MARKERS):
        return "Embedding"
    if any(marker in lower for marker in _VISION_NAME_MARKERS) or any(
        family in ("clip", "mllama") for family in families or []
    ):
        return "Vision"
    return "Chat"


def ollama_loaded(raw_models: list[dict[str, Any]], request_stats: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Shape Ollama's ``/api/ps`` rows, joined with request counts from telemetry."""
    this_year = time.gmtime().tm_year
    loaded = []
    for model in raw_models:
        name = str(model["name"])
        size = int(model.get("size") or 0)
        vram = int(model.get("size_vram") or 0)
        expires_at = str(model.get("expires_at") or "")
        expiry_year = expires_at[:4]
        pinned = expiry_year.isdigit() and int(expiry_year) - this_year >= _PINNED_EXPIRY_YEARS
        details = model.get("details") or {}
        stats = request_stats.get(name) or {}
        loaded.append(
            {
                "id": f"ollama:{name}",
                "name": name,
                "purpose": model_purpose(name, details.get("families")),
                "backend": "Ollama",
                "size_bytes": size,
                "vram_bytes": vram,
                "ram_bytes": max(0, size - vram),
                "gpu_pct": round(100 * vram / size, 1) if size else None,
                "context_length": model.get("context_length"),
                "pinned": pinned,
                "expires_at": None if pinned else (expires_at or None),
                "parameter_size": details.get("parameter_size"),
                "quantization": details.get("quantization_level"),
                # Ollama does not report when a model was loaded or how often it ran.
                "loaded_at": None,
                "last_request_at": stats.get("last"),
                "requests": stats.get("count"),
                "requests_source": "telemetry" if stats else None,
                "unloadable": True,
            }
        )
    return loaded


def in_process_models(embedding_service: Any) -> list[dict[str, Any]]:
    """Embedding, reranker and vision models held inside this backend process."""
    from backend.embeddings import embeddings as embedding_module
    from backend.retrieval import reranker as reranker_module
    from backend.vision.hf_vision_client import HFVisionClient

    models: list[dict[str, Any]] = []

    embedding_model = embedding_module._shared_embedding_model
    if embedding_model is not None:
        models.append(
            {
                "id": "embedding",
                "name": str(getattr(embedding_service, "model_name", "") or "Embedding model"),
                "purpose": "Embedding",
                "backend": "Sentence Transformers",
                "device": _device_of(embedding_model),
                "weight_bytes": weight_bytes(embedding_model),
                "loaded_at": _iso(embedding_module._shared_embedding_model_loaded_at),
                "unloadable": False,
            }
        )

    for key, model in list(reranker_module._shared_reranker_models.items()):
        if model is None:
            continue
        name, device, _max_length = key
        models.append(
            {
                "id": "reranker",
                "name": name,
                "purpose": "Reranker",
                "backend": "Sentence Transformers (cross-encoder)",
                "device": _device_of(model, device),
                "weight_bytes": weight_bytes(model),
                "loaded_at": _iso(reranker_module._shared_reranker_loaded_at.get(key)),
                "unloadable": False,
            }
        )

    vision = HFVisionClient.get_instance()
    if vision.is_loaded:
        models.append(
            {
                "id": "vision",
                "name": str(settings.vision_model),
                "purpose": "Vision",
                "backend": "Transformers",
                "device": vision.device,
                "weight_bytes": weight_bytes(vision.model),
                "loaded_at": _iso(vision.loaded_at),
                "unloadable": True,
            }
        )
    return models


def gpu_breakdown(
    gpu: dict[str, Any] | None,
    loaded: list[dict[str, Any]],
    backend_reserved_bytes: int | None,
) -> dict[str, Any] | None:
    """Attribute device-wide VRAM use to what each runtime reports holding.

    Ollama reports VRAM per model and PyTorch reports what this process has
    reserved; whatever the driver counts beyond those is left as one remainder.
    """
    if gpu is None:
        return None
    mb = 1024 * 1024
    used_bytes = int(gpu["used_mb"] * mb)
    segments = [
        {"id": m["id"], "label": m["name"], "kind": "model", "bytes": m["vram_bytes"], "source": "Ollama"}
        for m in loaded
        if m["vram_bytes"] > 0
    ]
    if backend_reserved_bytes:
        segments.append(
            {
                "id": "backend",
                "label": "Backend process (PyTorch)",
                "kind": "backend",
                "bytes": backend_reserved_bytes,
                "source": "torch.cuda.memory_reserved",
            }
        )
    attributed = sum(s["bytes"] for s in segments)
    segments.append(
        {
            "id": "other",
            "label": "Other processes, CUDA runtime and display",
            "kind": "other",
            "bytes": max(0, used_bytes - attributed),
            "source": "remainder of driver-reported use",
        }
    )
    return {
        **gpu,
        "segments": segments,
        "note": "Per-model VRAM comes from Ollama; the backend figure is what PyTorch has reserved. "
        "A split per model inside the backend process is not exposed by the runtime.",
    }
