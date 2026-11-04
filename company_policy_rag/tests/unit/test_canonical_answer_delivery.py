import json
from unittest.mock import MagicMock

import pytest

from backend.models.api_dto import ChatRequest
from backend.models.rag import Citation, RAGResponse, RAGTrace
from backend.services.chat_service import ChatService


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "draft,final", [("incorrect draft", "Corrected answer [Source 1]."), ("draft", ""), ("", "Final answer")]
)
async def test_final_answer_and_citations_are_canonical_everywhere(draft, final):
    citation = Citation(
        source_index=1,
        chunk_id="final",
        document_id="doc_123456789abc",
        source_file="policy.txt",
        snippet="Final evidence",
    )
    trace = RAGTrace(query="question", faithfulness_passed=True)

    class Pipeline:
        def get_active_model(self):
            return "active"

        def query(self, **kwargs):
            return RAGResponse(query="question", answer=final, citations=[citation], trace=trace, model="actual")

        async def stream_query(self, **kwargs):
            if draft:
                yield {"type": "token", "content": draft}
            yield {
                "type": "done",
                "answer": final,
                "citations": [citation],
                "trace": trace,
                "context_chunks": [],
                "model": "actual",
            }

    telemetry = MagicMock()
    telemetry.record_from_rag_response.return_value = {}
    service = ChatService(Pipeline(), telemetry)
    frames = [frame async for frame in service.stream_query(ChatRequest(message="question", session_id="stream"))]
    done = json.loads(next(frame.split("data: ", 1)[1] for frame in frames if frame.startswith("event: done")))
    assert done["answer"] == final
    assert done["model"] == "actual"
    assert done["citations"][0]["chunk_id"] == "final"
    assert service._sessions["stream"][-1]["content"] == final
    assert service.state_manager.get_state("stream").turns[-1].answer == final
    assert telemetry.record_from_rag_response.call_args.args[0].answer == final
    assert service.execute_query(ChatRequest(message="question", session_id="sync")).answer == final
