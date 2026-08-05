from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from backend.api.dependencies import get_chat_service
from backend.models.api_dto import ChatRequest, ChatResponse
from backend.services.chat_service import ChatService

router = APIRouter(tags=["Chat"])


@router.post("/api/chat", response_model=ChatResponse)
def post_chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Synchronous RAG Chat query endpoint."""
    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chat message query cannot be empty.",
        )

    try:
        return chat_service.execute_query(request)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat query: {str(exc)}",
        )


@router.post("/api/chat/stream")
async def post_chat_stream(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """
    Sub-1s Time-To-First-Token (TTFT) Server-Sent Events (SSE) streaming chat endpoint.
    Emits structured events: start, chunk, citation, trace, done, error.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chat message query cannot be empty.",
        )

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
        "X-Accel-Buffering": "no",
    }

    return StreamingResponse(
        chat_service.stream_query(request),
        media_type="text/event-stream",
        headers=headers,
    )
