from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.api.dependencies import get_chat_service
from backend.models.api_dto import ModelInfo, ModelListResponse
from backend.services.chat_service import ChatService
from src.ollama_client import filter_chat_models, list_enriched_models, probe_ollama_tags

router = APIRouter(tags=["Models"])

# Default active model state
_current_active_model = "qwen2.5:7b"


class ModelSelectRequest(BaseModel):
    model: str = Field(..., description="Model ID to set active")


@router.get("/api/models", response_model=ModelListResponse)
def get_available_models() -> ModelListResponse:
    """List available LLM, Embedding, and Reranker model specifications."""
    ok, names, err = probe_ollama_tags()
    model_names = names if ok else ["qwen2.5:7b"]
    active_name = _current_active_model

    enriched = list_enriched_models(model_names, recommended=active_name)
    responses = []
    for item in enriched:
        info = ModelInfo(
            id=item["id"],
            name=item["label"],
            type="llm" if item["id"] in filter_chat_models(model_names) else "embedding" if "embed" in item["id"].lower() else "reranker",
            loaded=True,
            is_active=(item["id"] == active_name),
        )
        responses.append(info)

    return ModelListResponse(active_model=active_name, models=responses)


@router.post("/api/models/select")
@router.put("/api/models/active")
def select_active_model(
    req: ModelSelectRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> dict[str, str]:
    """Switch active LLM model and push it down into the live backend pipeline."""
    global _current_active_model

    ok, names, err = probe_ollama_tags()
    model_names = names if ok else ["qwen2.5:7b"]
    valid_ids = set(filter_chat_models(model_names))
    valid_ids.add("qwen2.5:7b")

    if req.model not in valid_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{req.model}' is not available. Installed Ollama models: {', '.join(sorted(valid_ids)) or 'none'}",
        )

    try:
        chat_service.set_active_model(req.model)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to switch model: {exc!s}",
        ) from exc

    _current_active_model = req.model
    return {"status": "switched", "active_model": _current_active_model}
