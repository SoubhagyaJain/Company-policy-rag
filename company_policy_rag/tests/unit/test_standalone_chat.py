from unittest.mock import MagicMock

import pytest

from backend.models.api_dto import ChatRequest
from backend.models.rag import RAGResponse, RAGTrace
from backend.services.chat_service import ChatService


class Pipeline:
    def __init__(self):
        self.inputs = []

    def get_active_model(self):
        return "model"

    def query(self, **kwargs):
        self.inputs.append((kwargs["history"], kwargs["conversation_state"].model_copy(deep=True), kwargs))
        return RAGResponse(query=kwargs["user_query"], answer="answer", trace=RAGTrace(query=kwargs["user_query"]))

    async def stream_query(self, **kwargs):
        self.inputs.append((kwargs["history"], kwargs["conversation_state"].model_copy(deep=True), kwargs))
        yield {
            "type": "done",
            "answer": "answer",
            "citations": [],
            "trace": RAGTrace(query=kwargs["user_query"]),
            "context_chunks": [],
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_each_turn_is_independent_but_document_selection_is_retained(streaming):
    pipeline = Pipeline()
    service = ChatService(pipeline, MagicMock())
    first = ChatRequest(message="Explain leave", session_id="session", active_document_id="doc_123")
    second = ChatRequest(message="What about it?", session_id="session")
    for request in (first, second):
        if streaming:
            _ = [event async for event in service.stream_query(request)]
        else:
            service.execute_query(request)
    assert len(service._sessions["session"]) == 4
    assert len(pipeline.inputs) == 2
    for history, state, kwargs in pipeline.inputs:
        assert history == []
        assert state.turns == []
        assert state.active_topic is None
        assert state.previous_retrieved_chunks == []
        assert kwargs["active_document_id"] == "doc_123"
