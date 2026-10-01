"""Regression tests for source precision and concise default answers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from backend.models.chunk import Chunk, ChunkMetadata, ContentType
from backend.models.conversation import AnswerMode
from backend.models.document import DocumentMetadata, DocumentType, RawDocument
from backend.models.rag import QueryCategory, ScoredChunk
from backend.ingestion.chunkers.base import BaseChunker
from backend.rag.evidence_gate import EvidenceSufficiencyGate, text_answers_without_visual
from backend.rag.citations import CitationEngine
from backend.rag.context_compression import ContextCompressor
from backend.rag.pipeline import (
    UNREAD_VISUAL_DIRECTIVE,
    RAGPipeline,
    _answer_matches_requested_enumeration,
    _extract_requested_numbered_list,
    _is_cacheable_grounded_answer,
    _is_degraded_or_abstention_answer,
)
from backend.rag.query_rewrite import QueryRewriter
from backend.vision.vision_service import VisionService


class _PassThroughChunker(BaseChunker):
    def chunk(self, documents: list[RawDocument]) -> list[Chunk]:
        return [
            self._create_chunk(document.content, document, index, "test")
            for index, document in enumerate(documents)
        ]


def _chunk(chunk_id: str, text: str, source_file: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=chunk_id,
            text=text,
            metadata=ChunkMetadata(
                document_id=f"doc_{chunk_id}",
                source_file=source_file,
                page_number=36,
                section_title="Levels of Agentic AI Systems",
                content_type=ContentType.PROSE,
            ),
        ),
        score=0.9,
        rerank_score=1.0,
    )


def test_voice_rag_rewrite_preserves_voice_intent() -> None:
    rewriter = QueryRewriter(enable_llm_rewrite=False)

    result = rewriter.rewrite("how can i make voice rag agent")

    # No corpus-specific vocabulary is appended, so another topic's terms
    # ("vector db context") cannot hijack the voice intent.
    assert result.rewritten_query == "how can i make voice rag agent"


def test_context_and_citations_dedupe_duplicate_document_uploads() -> None:
    passage = (
        "Agentic RAG uses a retriever agent to fetch context from a vector database "
        "and a writer agent to generate a grounded response."
    )
    chunks = [
        _chunk("one", passage, "AI Agents guidebook (1).pdf"),
        _chunk("two", passage, "AI_Agents_guidebook.pdf"),
    ]

    packed = ContextCompressor().pack_complementary_chunks(chunks, "agentic RAG", max_chunks=6)
    citations = CitationEngine().select_citations(
        "The workflow uses retrieval and writing [Source 1] [Source 2].",
        chunks,
        user_query="agentic RAG",
    )

    assert len(packed) == 1
    assert len(citations) == 1
    assert citations[0].source_file == "AI Agents guidebook (1).pdf"


def test_page_identity_falls_back_to_preserved_extra_metadata() -> None:
    metadata = ChunkMetadata(
        document_id="doc_book",
        source_file="book.pdf",
        page_number=71,
        extra={
            "internal_page_index": 70,
            "physical_page_number": 71,
            "display_page_number": 70,
            "page_label": "70",
        },
    )

    page_identity = metadata.get_page_identity()

    assert page_identity.physical_page_number == 71
    assert page_identity.display_page_number == 70
    assert page_identity.display_label == "70"


def test_chunking_preserves_page_identity_and_visual_assets() -> None:
    document = RawDocument(
        id="doc_book",
        content="Five techniques are depicted below.",
        metadata=DocumentMetadata(
            document_id="doc_book",
            source_file="book.pdf",
            file_path="uploads/book.pdf",
            file_hash="hash",
            document_type=DocumentType.PDF,
            page_number=71,
            internal_page_index=70,
            display_page_number=70,
            page_label="70",
            image_assets=[{"asset_id": "asset_70", "visual_type": "diagram_architecture"}],
        ),
    )

    chunk = _PassThroughChunker().chunk([document])[0]

    assert chunk.metadata.display_page_number == 70
    assert chunk.metadata.page_label == "70"
    assert chunk.metadata.visual_asset_ids == ["asset_70"]
    assert chunk.metadata.image_assets[0]["asset_id"] == "asset_70"


def test_visual_reference_is_not_treated_as_complete_text_evidence() -> None:
    scored = _chunk(
        "visual-list",
        "Five popular fine-tuning techniques are depicted below.",
        "AI Engineering Guidebook.pdf",
    )
    scored.chunk.metadata.page_number = 71

    result = EvidenceSufficiencyGate().evaluate(
        "What are the five fine-tuning techniques?",
        QueryCategory.FACTUAL,
        [scored],
    )

    assert result.is_sufficient is False
    assert "referenced_visual_content" in result.missing_evidence_types
    assert 71 in result.pages_to_inspect


def test_numbered_continuation_text_resolves_visual_list_reference() -> None:
    chunks = [
        _chunk(
            "visual-list-anchor",
            "Five popular fine-tuning techniques are depicted below.",
            "AI Engineering Guidebook.pdf",
        ),
        _chunk("visual-list-1", "1) LoRA", "AI Engineering Guidebook.pdf"),
        _chunk("visual-list-2", "2) LoRA-FA\n3) VeRA", "AI Engineering Guidebook.pdf"),
        _chunk("visual-list-3", "4) Delta-LoRA\n5) LoRA+", "AI Engineering Guidebook.pdf"),
    ]
    for page, scored in enumerate(chunks, start=71):
        scored.chunk.metadata.document_id = "doc_guide"
        scored.chunk.metadata.page_number = page
        scored.chunk.metadata.section_title = "LLM Fine-tuning Techniques"

    result = EvidenceSufficiencyGate().evaluate(
        "What are the five fine-tuning techniques?",
        QueryCategory.FACTUAL,
        chunks,
    )

    assert result.is_sufficient is True
    assert "referenced_visual_content" not in result.missing_evidence_types


_AGENTIC_RAG_PAGE = (
    "These systems retrieve once and generate once. If the retrieved context isn't "
    "enough, the LLM can not dynamically search for more information.\n"
    "The workflow of agentic RAG is depicted below:\n"
    "Steps 1-2) The user inputs the query, and an agent rewrites it.\n"
    "Step 3) Another agent decides whether it needs more details to answer the query.\n"
    "Step 10) A final agent checks if the answer is relevant to the query and context."
)


def test_unread_visual_does_not_block_text_that_answers_the_question() -> None:
    walkthrough = _chunk("agentic-rag", _AGENTIC_RAG_PAGE, "AI Engineering Guidebook.pdf")
    pointer_only = _chunk(
        "visual-list",
        "Five popular fine-tuning techniques are depicted below.",
        "AI Engineering Guidebook.pdf",
    )
    promised_list = _chunk(
        "visual-list-in-prose",
        "Traditional fine-tuning updates every weight, which is infeasible for large "
        "models because of the compute, memory, and storage each full copy needs. "
        "Five popular fine-tuning techniques are depicted below.",
        "AI Engineering Guidebook.pdf",
    )

    assert text_answers_without_visual("Compare naive RAG and Agentic RAG.", [walkthrough]) is True
    assert text_answers_without_visual("What are the five fine-tuning techniques?", [pointer_only]) is False
    # Surrounding prose does not supply labels that only the visual lists.
    assert text_answers_without_visual("What are the fine-tuning techniques?", [promised_list]) is False
    assert text_answers_without_visual("Why is traditional fine-tuning infeasible?", [promised_list]) is True


class _UnavailableVision:
    """Vision service whose model cannot run at query time (e.g. no free VRAM)."""

    vision_model = "Qwen3-VL-2B-Instruct"

    def __init__(self) -> None:
        self.image_asset_manager = MagicMock()
        self.image_asset_manager.get_page_assets_by_physical_page.return_value = []
        self.image_asset_manager.get_page_assets.return_value = []

    def is_query_time_available(self) -> tuple[bool, str]:
        return False, "CPU-only vision is disabled for interactive queries."


def _pipeline_without_vision(tmp_path: Path, text: str) -> tuple[RAGPipeline, MagicMock]:
    pdf = tmp_path / "guide.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")
    scored = _chunk("anchor", text, "guide.pdf")
    scored.chunk.metadata.document_id = "doc_guide"
    scored.chunk.metadata.file_path = str(pdf)
    scored.chunk.metadata.page_number = 130

    retriever = MagicMock()
    retriever.retrieve.return_value = [scored]
    llm = MagicMock()
    llm.complete.return_value = (
        "Traditional RAG retrieves once and generates once, while agentic RAG adds agents "
        "that rewrite the query and check the answer [Source 1]."
    )
    pipeline = RAGPipeline(
        hybrid_retriever=retriever,
        docstore={scored.chunk.id: scored.chunk},
        llm=llm,
        vision_service=_UnavailableVision(),
    )
    return pipeline, llm


def test_answer_is_generated_from_text_when_vision_is_unavailable(tmp_path: Path) -> None:
    pipeline, llm = _pipeline_without_vision(tmp_path, _AGENTIC_RAG_PAGE)

    response = pipeline.query(user_query="Compare naive RAG and Agentic RAG.")

    assert response.answer == llm.complete.return_value
    assert _is_degraded_or_abstention_answer(response.answer) is False
    assert UNREAD_VISUAL_DIRECTIVE in llm.complete.call_args.args[0]
    assert response.trace.vision_status == "DEGRADED"


def test_visual_only_answer_still_abstains_when_vision_is_unavailable(tmp_path: Path) -> None:
    pipeline, llm = _pipeline_without_vision(
        tmp_path, "Five popular fine-tuning techniques are depicted below."
    )

    response = pipeline.query(user_query="What are the five fine-tuning techniques?")

    assert _is_degraded_or_abstention_answer(response.answer) is True
    llm.complete.assert_not_called()


def test_degraded_visual_abstention_is_never_cacheable() -> None:
    answer = (
        "The retrieved text says the details are shown in a visual, but the visual "
        "labels could not be read reliably. I can't list them without guessing."
    )

    assert _is_degraded_or_abstention_answer(answer) is True
    assert _is_cacheable_grounded_answer(
        answer,
        has_citations=True,
        verifier_passed=True,
        evidence_sufficiency_passed=False,
        vision_status="DEGRADED",
        requires_visual_abstention=True,
    ) is False


def test_complete_grounded_answer_remains_cacheable() -> None:
    assert _is_cacheable_grounded_answer(
        "The five techniques are LoRA, LoRA-FA, VeRA, Delta-LoRA, and LoRA+.",
        has_citations=True,
        verifier_passed=True,
        evidence_sufficiency_passed=True,
        vision_status="READY",
    ) is True


def test_exact_numbered_list_is_extracted_without_adding_introductory_context() -> None:
    chunks = [
        _chunk(
            "list-page-one",
            "Traditional fine-tuning is infeasible.\n1) LoRA\nAdd low-rank matrices.",
            "guide.pdf",
        ),
        _chunk(
            "list-page-two",
            "2) LoRA-FA\nDetails.\n3) VeRA\nDetails.",
            "guide.pdf",
        ),
        _chunk(
            "list-page-three",
            "4) Delta-LoRA\nDetails.\n5) LoRA+\nDetails.",
            "guide.pdf",
        ),
    ]
    for page, scored in enumerate(chunks, start=72):
        scored.chunk.metadata.document_id = "doc_guide"
        scored.chunk.metadata.page_number = page

    result = _extract_requested_numbered_list(
        "what are the 5 techinque of llm fine tuning",
        chunks,
    )

    assert result == ["LoRA", "LoRA-FA", "VeRA", "Delta-LoRA", "LoRA+"]


def test_cached_numbered_answer_must_match_requested_count() -> None:
    wrong = "\n".join(
        [
            "1. Full Fine-tuning",
            "2. LoRA",
            "3. LoRA-FA",
            "4. VeRA",
            "5. Delta-LoRA",
            "6. LoRA+",
        ]
    )
    right = "\n".join(
        ["1. LoRA", "2. LoRA-FA", "3. VeRA", "4. Delta-LoRA", "5. LoRA+"]
    )

    query = "what are the 5 techniques of llm fine tuning"
    assert _answer_matches_requested_enumeration(query, wrong) is False
    assert _answer_matches_requested_enumeration(query, right) is True


def test_generic_continuation_cue_is_classified_as_diagram() -> None:
    service = VisionService()

    result = service.detect_visual_content(
        page_text="Five techniques are depicted below.",
        image_bytes=b"not-decoded-by-detection",
        image_count=1,
        page_number=71,
        continuation_cue="depicted below",
    )

    assert result.visual_type.value == "diagram_architecture"
