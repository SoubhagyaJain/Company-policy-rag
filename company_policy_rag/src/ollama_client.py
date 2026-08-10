"""Ollama HTTP helpers shared by API, Streamlit, and chat service."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend.utils.logging import logger
from src.config import settings
from src.thinking_extract import is_reasoning_model

_EMBED_MARKERS = ("embed", "nomic-embed", "mxbai-embed", "bge-m3")


def filter_chat_models(names: list[str]) -> list[str]:
    """Exclude embedding-only models from LLM picker options."""
    chat: list[str] = []
    for name in names:
        lower = name.lower()
        if any(marker in lower for marker in _EMBED_MARKERS):
            continue
        chat.append(name)
    return sorted(chat)


def format_model_label(model_id: str) -> str:
    """Human-readable label for UI (qwen2.5:7b -> Qwen2.5 7B)."""
    base = model_id.split(":")[0]
    tag = model_id.split(":")[1] if ":" in model_id else ""
    label = base.replace("-", " ").replace("_", " ")
    parts = label.split()
    formatted = " ".join(p[:1].upper() + p[1:] for p in parts if p)
    if tag:
        formatted = f"{formatted} {tag.upper()}"
    return formatted.strip()


def unload_model(
    model_name: str,
    base_url: str | None = None,
    *,
    timeout: float = 5.0,
) -> bool:
    """Best-effort unload of a running Ollama model from VRAM."""
    if not model_name or not str(model_name).strip():
        return False

    url = (base_url or settings.ollama_base_url).rstrip("/") + "/api/generate"
    payload = json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8")
    req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")

    try:
        with urlopen(req, timeout=timeout) as response:
            response.read()
        logger.info("Unloaded Ollama model '%s' via keep_alive=0", model_name)
        return True
    except Exception as exc:
        logger.warning("Failed to unload Ollama model '%s': %s", model_name, exc)
        return False


def preload_model(
    model_name: str,
    base_url: str | None = None,
    *,
    timeout: float = 30.0,
) -> bool:
    """Best-effort preload of an Ollama model into VRAM."""
    if not model_name or not str(model_name).strip():
        return False

    url = (base_url or settings.ollama_base_url).rstrip("/") + "/api/generate"
    payload = json.dumps({"model": model_name, "keep_alive": -1}).encode("utf-8")
    req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")

    try:
        with urlopen(req, timeout=timeout) as response:
            response.read()
        logger.info("Preloaded Ollama model '%s' via keep_alive=-1", model_name)
        return True
    except Exception as exc:
        logger.warning("Failed to preload Ollama model '%s': %s", model_name, exc)
        return False


def probe_ollama_tags(
    base_url: str | None = None,
    *,
    timeout: float = 5.0,
) -> tuple[bool, list[str], str | None]:
    """Call Ollama GET /api/tags. Returns (ok, model_names, error_message)."""
    url = (base_url or settings.ollama_base_url).rstrip("/") + "/api/tags"
    try:
        with urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        return False, [], str(exc.reason if hasattr(exc, "reason") else exc)
    except (TimeoutError, json.JSONDecodeError, OSError) as exc:
        return False, [], str(exc)

    models = payload.get("models") or []
    names: list[str] = []
    for item in models:
        if isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
    return True, sorted(names), None


@lru_cache(maxsize=64)
def fetch_model_details(
    model_name: str,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Fetch Ollama POST /api/show details for a model."""
    url = (base_url or settings.ollama_base_url).rstrip("/") + "/api/show"
    body = json.dumps({"name": model_name}).encode("utf-8")
    try:
        req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=8.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (URLError, TimeoutError, json.JSONDecodeError, OSError):
        return {}


def _parse_param_size(details: dict[str, Any]) -> str | None:
    for key in ("parameter_size", "parameters"):
        val = details.get(key)
        if val:
            return str(val)
    params = details.get("details") or {}
    if isinstance(params, dict) and params.get("parameter_size"):
        return str(params["parameter_size"])
    return None


def _parse_quantization(details: dict[str, Any]) -> str | None:
    params = details.get("details") or {}
    if isinstance(params, dict) and params.get("quantization_level"):
        return str(params["quantization_level"])
    return None


def _parse_family(model_id: str, details: dict[str, Any]) -> str | None:
    model_info = details.get("model_info") or {}
    if isinstance(model_info, dict):
        for key in ("general.architecture", "family"):
            if model_info.get(key):
                return str(model_info[key])
    base = model_id.split(":")[0].lower()
    for fam in ("qwen", "llama", "deepseek", "mistral", "gemma", "phi"):
        if fam in base:
            return fam
    return None


def _param_size_numeric(param_size: str | None) -> float | None:
    if not param_size:
        return None
    lower = param_size.lower()
    for suffix, mult in (("b", 1.0), ("m", 0.001)):
        if suffix in lower:
            num = "".join(c for c in lower if c.isdigit() or c == ".")
            try:
                return float(num) * mult if suffix == "b" else float(num) * 0.001
            except ValueError:
                return None
    return None


def enrich_model_info(model_id: str, *, recommended: str | None = None) -> dict[str, Any]:
    """Build rich model metadata for UI selector."""
    details = fetch_model_details(model_id)
    param_size = _parse_param_size(details)
    quantization = _parse_quantization(details)
    family = _parse_family(model_id, details)
    badges: list[str] = []
    if is_reasoning_model(model_id):
        badges.append("Reasoning")
    size_num = _param_size_numeric(param_size)
    if size_num is not None and size_num <= 8.0:
        badges.append("Fast")
    if recommended and model_id == recommended:
        badges.append("Recommended")
    return {
        "id": model_id,
        "label": format_model_label(model_id),
        "family": family,
        "parameter_size": param_size,
        "quantization": quantization,
        "badges": badges,
    }


def list_enriched_models(
    names: list[str] | None = None,
    *,
    recommended: str | None = None,
) -> list[dict[str, Any]]:
    if names is None:
        ok, names, _ = probe_ollama_tags()
        if not ok:
            names = []
    chat_models = filter_chat_models(names)
    rec = recommended or settings.llm_model
    if rec not in chat_models:
        chat_models = [rec] + chat_models
    return [enrich_model_info(m, recommended=rec) for m in chat_models]