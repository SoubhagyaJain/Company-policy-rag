from __future__ import annotations

import asyncio
from fastapi import APIRouter, Depends, HTTPException, status, Request
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
            detail=f"Error processing chat query: {exc!s}",
        )


@router.post("/api/chat/stream")
async def post_chat_stream(
    request: Request,
    body: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """
    Sub-1s TTFT SSE streaming endpoint. Detects client disconnects and cancels generation.
    """
    if not body.message or not body.message.strip():
        raise HTTPException(status_code=400, detail="Chat message query cannot be empty.")

    async def sse_generator():
        # Cancellation token to pass into the background generation
        cancel_token = asyncio.Event()
        
        async def check_disconnect():
            """Continuously poll if the client dropped connection."""
            while True:
                if await request.is_disconnected():
                    cancel_token.set()
                    break
                await asyncio.sleep(0.2)
                
        # Launch disconnect monitor
        disconnect_task = asyncio.create_task(check_disconnect())
        
        try:
            # Yield events directly from the fully async ChatService
            async for event_str in chat_service.stream_query(body, cancel_token):
                yield event_str
                # Stop yielding if client dropped
                if cancel_token.is_set():
                    break
        finally:
            # Cleanup background task
            disconnect_task.cancel()

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(sse_generator(), media_type="text/event-stream", headers=headers)

@router.delete("/api/chat/session/{session_id}")
def delete_chat_session(
    session_id: str,
    chat_service: ChatService = Depends(get_chat_service)
):
    """Evict a session explicitly to prevent memory leaks."""
    chat_service.delete_session(session_id)
    return {"status": "success", "detail": f"Session {session_id} deleted."}
