from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.models.api_dto import ModelInfo, ModelListResponse

router = APIRouter(tags=["Models"])

# Default active model state
_current_active_model = "qwen2.5:7b"

AVAILABLE_MODELS = [
    ModelInfo(id="qwen2.5:7b", name="Qwen 2.5 7B", type="llm", loaded=True, is_active=True),
    ModelInfo(id="BAAI/bge-small-en-v1.5", name="BGE Small Vector Embeddings", type="embedding", loaded=True, is_active=False),
    ModelInfo(id="BAAI/bge-reranker-large", name="BGE Cross-Encoder Reranker", type="reranker", loaded=True, is_active=False),
]


class ModelSelectRequest(BaseModel):
    model: str = Field(..., description="Model ID to set active")


@router.get("/api/models", response_model=ModelListResponse)
def get_available_models() -> ModelListResponse:
    """List available LLM, Embedding, and Reranker model specifications."""
    updated_models = [
        ModelInfo(
            id=m.id,
            name=m.name,
            type=m.type,
            loaded=m.loaded,
            is_active=(m.id == _current_active_model),
        )
        for m in AVAILABLE_MODELS
    ]
    return ModelListResponse(
        active_model=_current_active_model,
        models=updated_models,
    )


@router.post("/api/models/select")
@router.put("/api/models/active")
def select_active_model(req: ModelSelectRequest) -> dict[str, str]:
    """Switch active LLM model."""
    global _current_active_model
    valid_ids = [m.id for m in AVAILABLE_MODELS if m.type == "llm"]
    if req.model not in valid_ids and req.model != "qwen2.5:7b":
        raise HTTPException(status_code=400, detail=f"Model '{req.model}' is not available.")
    _current_active_model = req.model
    return {"status": "switched", "active_model": _current_active_model}
