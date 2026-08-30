"""
Integration & Regression Test Suite for Document-Faithful Cross-Page Multimodal RAG.

Scenarios tested:
1. "How can I make X Writer Agent?" (Cross-page text on P63 + visual code on P64)
2. "How can I make X Analyst Agent?"
3. "Show me the X Writer Agent code."
4. "How is the Writer Task defined?"
5. "How does the X Crew work?"
6. Negative Grounding Test: "How do I configure the Quantum Flux Agent?" (Abstention)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.models.chunk import Chunk, ChunkMetadata, ContentType
from backend.models.rag import QueryCategory, ScoredChunk
from backend.rag.pipeline import RAGPipeline
from backend.vision.vision_service import VisualExtractionChunk


def _create_test_chunk(
    chunk_id: str,
    text: str,
    document_id: str,
    source_file: str,
    page_number: int,
    section_title: str,
    content_type: ContentType = ContentType.PROSE,
    extra: dict | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            document_id=document_id,
            source_file=source_file,
            file_path=f"storage/documents/{document_id}_{source_file}",
            file_hash=f"hash_{document_id}",
            document_type="pdf",
            chunk_strategy="adaptive",
            page_number=page_number,
            page_label=str(page_number),
            section_title=section_title,
            content_type=content_type,
            extra=extra or {},
        ),
    )


@pytest.fixture
def mock_vision_service(tmp_path: Path):
    vision = MagicMock()
    vision.is_available.return_value = (True, "Vision model 'Qwen3-VL-2B-Instruct' is available locally.")
    vision.vision_model = "Qwen3-VL-2B-Instruct"

    def fake_process_pdf_page_visuals(
        pdf_path, page_number, page_text="", document_id=None, section_title=None, continuation_cue=None, **kwargs
    ):
        if page_number == 64:
            code_text = (
                "```python\n"
                "writer_agent = Agent(\n"
                "    role='X Writer Agent',\n"
                "    goal='Draft engaging insights from analysis',\n"
                "    backstory='Expert copywriter specializing in viral summaries'\n"
                ")\n\n"
                "writer_task = Task(\n"
                "    description='Write a summary post from the analyst report',\n"
                "    expected_output='A clean 3-paragraph summary',\n"
                "    agent=writer_agent\n"
                ")\n"
                "```"
            )
            return [
                VisualExtractionChunk(
                    text=code_text,
                    content_type="code",
                    visual_type="code_screenshot",
                    page_number=64,
                    image_hash="hash_writer_code_64",
                    section_title=section_title or "X Writer Agent",
                    raw_code=code_text,
                )
            ]
        elif page_number == 62:
            code_text = (
                "```python\n"
                "class XAnalystAgent:\n"
                "    def __init__(self, bright_data_api_key: str):\n"
                "        self.api_key = bright_data_api_key\n\n"
                "    def analyze(self, posts: list[dict]) -> dict:\n"
                "        return {'sentiment': 'positive', 'count': len(posts)}\n"
                "```"
            )
            return [
                VisualExtractionChunk(
                    text=code_text,
                    content_type="code",
                    visual_type="code_screenshot",
                    page_number=62,
                    image_hash="hash_analyst_code_62",
                    section_title=section_title or "X Analyst Agent",
                    raw_code=code_text,
                )
            ]
        return []

    vision.process_pdf_page_visuals.side_effect = fake_process_pdf_page_visuals
    return vision


def test_scenario_1_x_writer_agent_cross_page_retrieval(tmp_path: Path, mock_vision_service):
    """
    Scenario 1:
    - Page 63: #6 X Writer Agent text ending with "Here's the code:"
    - Page 64: Visual code screenshot containing writer_agent = Agent(...) and writer_task = Task(...)
    - Query: "How can I make X Writer Agent?"
    - Verifies that lazy vision fallback inspects Page 64, extracts real code, and answers with 100% document fidelity.
    """
    doc_id = "doc_ai_agents_guidebook"
    filename = "AI_Agents_guidebook.pdf"

    # Create dummy physical file so _resolve_document_file_path succeeds
    dummy_pdf = tmp_path / f"{doc_id}_{filename}"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

    chunk_p63 = _create_test_chunk(
        chunk_id="chunk_p63_writer",
        text="#6 X Writer Agent\n\nThe Agent takes the output of the X Analyst agent and generates insights.\n\nHere's the code:",
        document_id=doc_id,
        source_file=filename,
        page_number=63,
        section_title="X Writer Agent",
        content_type=ContentType.PROSE,
    )
    chunk_p63.metadata.file_path = str(dummy_pdf)

    docstore = {"chunk_p63_writer": chunk_p63}

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [ScoredChunk(chunk=chunk_p63, score=0.92)]

    mock_llm = MagicMock()
    def fake_llm_complete(prompt: str, **kwargs) -> str:
        assert "writer_agent = Agent(" in prompt, "LLM prompt MUST contain real extracted code from Page 64!"
        assert "writer_task = Task(" in prompt, "LLM prompt MUST contain real task code from Page 64!"
        assert "def define_writer_agent" not in prompt
        return (
            "To make the X Writer Agent, use the code provided in the document [Source 1]:\n\n"
            "```python\n"
            "writer_agent = Agent(\n"
            "    role='X Writer Agent',\n"
            "    goal='Draft engaging insights from analysis',\n"
            "    backstory='Expert copywriter specializing in viral summaries'\n"
            ")\n\n"
            "writer_task = Task(\n"
            "    description='Write a summary post from the analyst report',\n"
            "    expected_output='A clean 3-paragraph summary',\n"
            "    agent=writer_agent\n"
            ")\n"
            "```\n\n"
            "The Writer Agent takes the output of the X Analyst agent and generates insights [Source 2]."
        )

    mock_llm.complete.side_effect = fake_llm_complete

    pipeline = RAGPipeline(
        hybrid_retriever=mock_retriever,
        docstore=docstore,
        llm=mock_llm,
        vision_service=mock_vision_service,
    )

    response = pipeline.query(
        user_query="How can I make X Writer Agent?",
        active_document_id=doc_id,
        active_document_name=filename,
    )

    assert response is not None
    assert "writer_agent = Agent" in response.answer
    assert "writer_task = Task" in response.answer
    assert "def define_writer_agent" not in response.answer
    assert "pass" not in response.answer
    assert response.trace.evidence_code_count >= 1
    assert response.trace.vision_fallback is True
    assert response.trace.adjacent_page_check is True


def test_generic_build_question_does_not_trigger_vision_without_visual_signal(tmp_path: Path):
    """A normal implementation question must stay on the fast text-RAG path."""
    chunk = _create_test_chunk(
        chunk_id="chunk_voice_rag_overview",
        text=(
            "A voice RAG agent combines speech recognition, retrieval, grounded "
            "generation, and text-to-speech response playback."
        ),
        document_id="doc_voice_rag",
        source_file="voice_rag_guide.pdf",
        page_number=4,
        section_title="Voice RAG architecture",
    )
    chunk.metadata.file_path = str(tmp_path / "voice_rag_guide.pdf")
    scored = ScoredChunk(chunk=chunk, score=0.91)

    vision = MagicMock()
    vision.vision_model = "Qwen3-VL-2B-Instruct"
    vision.image_asset_manager.get_page_assets_by_physical_page.return_value = []
    vision.image_asset_manager.get_page_assets.return_value = []

    pipeline = RAGPipeline(
        hybrid_retriever=MagicMock(),
        docstore={chunk.id: chunk},
        llm=MagicMock(),
        vision_service=vision,
    )
    augmented, telemetry = pipeline._apply_cross_page_vision_fallback_if_needed(
        chunks=[scored],
        user_query="How can I build a voice RAG agent for my RAG app?",
        intent=QueryCategory.IMPLEMENTATION,
    )

    assert augmented == [scored]
    assert telemetry["vision_fallback"] is False
    assert telemetry["vision_status"] == "SKIPPED_NO_VISUAL_SIGNAL"
    vision.process_pdf_page_visuals.assert_not_called()


def test_scenario_2_x_analyst_agent_implementation(tmp_path: Path, mock_vision_service):
    """
    Scenario 2:
    - Page 61: X Analyst Agent description (Bright Data API)
    - Page 62: Visual code screenshot
    - Query: "How can I make X Analyst Agent?"
    """
    doc_id = "doc_ai_agents_guidebook"
    filename = "AI_Agents_guidebook.pdf"
    dummy_pdf = tmp_path / f"{doc_id}_{filename}"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

    chunk_p61 = _create_test_chunk(
        chunk_id="chunk_p61_analyst",
        text="#5 X Analyst Agent\n\nThis agent analyzes posts scraped by Bright Data. Here's the code:",
        document_id=doc_id,
        source_file=filename,
        page_number=61,
        section_title="X Analyst Agent",
    )
    chunk_p61.metadata.file_path = str(dummy_pdf)

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [ScoredChunk(chunk=chunk_p61, score=0.93)]

    mock_llm = MagicMock()
    mock_llm.complete.return_value = (
        "To make the X Analyst Agent, initialize it with your Bright Data API key as shown in the document:\n\n"
        "```python\n"
        "class XAnalystAgent:\n"
        "    def __init__(self, bright_data_api_key: str):\n"
        "        self.api_key = bright_data_api_key\n"
        "``` [Source 1]"
    )

    pipeline = RAGPipeline(
        hybrid_retriever=mock_retriever,
        docstore={"chunk_p61_analyst": chunk_p61},
        llm=mock_llm,
        vision_service=mock_vision_service,
    )

    response = pipeline.query(
        user_query="How can I make X Analyst Agent?",
        active_document_id=doc_id,
    )

    assert "XAnalystAgent" in response.answer
    assert "bright_data_api_key" in response.answer
    assert response.trace.vision_fallback is True


def test_scenario_3_exact_mode_show_writer_code(tmp_path: Path, mock_vision_service):
    """
    Scenario 3:
    - Mode 1: EXACT mode
    - Query: "Show me the X Writer Agent code."
    """
    doc_id = "doc_ai_agents_guidebook"
    filename = "AI_Agents_guidebook.pdf"
    dummy_pdf = tmp_path / f"{doc_id}_{filename}"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

    chunk_p63 = _create_test_chunk(
        chunk_id="chunk_p63",
        text="#6 X Writer Agent. Here's the code:",
        document_id=doc_id,
        source_file=filename,
        page_number=63,
        section_title="X Writer Agent",
    )
    chunk_p63.metadata.file_path = str(dummy_pdf)

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [ScoredChunk(chunk=chunk_p63, score=0.95)]

    mock_llm = MagicMock()
    mock_llm.complete.return_value = (
        "```python\n"
        "writer_agent = Agent(\n"
        "    role='X Writer Agent',\n"
        "    goal='Draft engaging insights from analysis',\n"
        "    backstory='Expert copywriter specializing in viral summaries'\n"
        ")\n"
        "``` [Source 1]"
    )

    pipeline = RAGPipeline(
        hybrid_retriever=mock_retriever,
        docstore={"chunk_p63": chunk_p63},
        llm=mock_llm,
        vision_service=mock_vision_service,
    )

    response = pipeline.query(user_query="Show me the X Writer Agent code.")
    assert "writer_agent = Agent" in response.answer


def test_scenario_4_writer_task_definition(tmp_path: Path, mock_vision_service):
    """
    Scenario 4:
    - Query: "How is the Writer Task defined?"
    """
    doc_id = "doc_ai_agents_guidebook"
    filename = "AI_Agents_guidebook.pdf"
    dummy_pdf = tmp_path / f"{doc_id}_{filename}"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

    chunk_p63 = _create_test_chunk(
        chunk_id="chunk_p63",
        text="#6 X Writer Agent. Here's the code:",
        document_id=doc_id,
        source_file=filename,
        page_number=63,
        section_title="X Writer Agent",
    )
    chunk_p63.metadata.file_path = str(dummy_pdf)

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [ScoredChunk(chunk=chunk_p63, score=0.91)]

    mock_llm = MagicMock()
    mock_llm.complete.return_value = (
        "The Writer Task is defined in the document as follows:\n\n"
        "```python\n"
        "writer_task = Task(\n"
        "    description='Write a summary post from the analyst report',\n"
        "    expected_output='A clean 3-paragraph summary',\n"
        "    agent=writer_agent\n"
        ")\n"
        "``` [Source 1]"
    )

    pipeline = RAGPipeline(
        hybrid_retriever=mock_retriever,
        docstore={"chunk_p63": chunk_p63},
        llm=mock_llm,
        vision_service=mock_vision_service,
    )

    response = pipeline.query(user_query="How is the Writer Task defined?")
    assert "writer_task = Task" in response.answer
    assert "expected_output" in response.answer


def test_scenario_5_x_crew_architecture(tmp_path: Path, mock_vision_service):
    """
    Scenario 5:
    - Query: "How does the X Crew work?"
    """
    doc_id = "doc_ai_agents_guidebook"
    filename = "AI_Agents_guidebook.pdf"

    chunk_crew = _create_test_chunk(
        chunk_id="chunk_crew",
        text="The X Crew combines the X Analyst Agent (which pulls social data) and the X Writer Agent (which drafts posts).",
        document_id=doc_id,
        source_file=filename,
        page_number=60,
        section_title="X Crew Architecture",
    )

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [ScoredChunk(chunk=chunk_crew, score=0.94)]

    mock_llm = MagicMock()
    mock_llm.complete.return_value = (
        "Based on the document, the X Crew coordinates two specialized agents:\n"
        "1. The X Analyst Agent extracts and analyzes social posts [Source 1].\n"
        "2. The X Writer Agent takes those insights and drafts summary posts [Source 1]."
    )

    pipeline = RAGPipeline(
        hybrid_retriever=mock_retriever,
        docstore={"chunk_crew": chunk_crew},
        llm=mock_llm,
        vision_service=mock_vision_service,
    )

    response = pipeline.query(user_query="How does the X Crew work?")
    assert "X Analyst Agent" in response.answer
    assert "X Writer Agent" in response.answer


def test_scenario_6_negative_grounding_abstention():
    """
    Scenario 6: Negative grounding test.
    - Query: "How do I configure the Quantum Flux Agent?"
    - Expected: The system must NOT hallucinate fake Quantum Flux Agent code.
      It must return an unanswerable abstention message.
    """
    mock_retriever = MagicMock()
    # Return empty candidate list
    mock_retriever.retrieve.return_value = []

    pipeline = RAGPipeline(
        hybrid_retriever=mock_retriever,
        docstore={},
        llm=MagicMock(),
    )

    response = pipeline.query(user_query="How do I configure the Quantum Flux Agent?")

    assert "could not find" in response.answer.lower()
    assert "class QuantumFluxAgent" not in response.answer
    assert "def configure_quantum_flux" not in response.answer
    assert len(response.citations) == 0
