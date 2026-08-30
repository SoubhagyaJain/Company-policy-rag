"""Regression tests for source precision and concise default answers."""

from __future__ import annotations

from backend.models.chunk import Chunk, ChunkMetadata, ContentType
from backend.models.conversation import AnswerMode
from backend.models.document import DocumentMetadata, DocumentType, RawDocument
from backend.models.rag import QueryCategory, ScoredChunk
from backend.ingestion.chunkers.base import BaseChunker
from backend.rag.evidence_gate import EvidenceSufficiencyGate
from backend.rag.citations import CitationEngine
from backend.rag.context_compression import ContextCompressor
from backend.rag.pipeline import (
    _answer_matches_requested_enumeration,
    _enforce_direct_answer_length,
    _extract_requested_numbered_list,
    _is_cacheable_grounded_answer,
    _is_degraded_or_abstention_answer,
    _select_answer_token_budget,
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

    rewritten = result.rewritten_query.lower()
    assert "voice" in rewritten
    assert "speech-to-text" in rewritten
    assert "text-to-speech" in rewritten
    assert "vector db context" not in rewritten


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


def test_direct_answer_budget_is_compact_unless_detail_is_requested(monkeypatch) -> None:
    monkeypatch.setattr("backend.rag.pipeline.settings.max_new_tokens_direct", 192)
    monkeypatch.setattr("backend.rag.pipeline.settings.max_new_tokens_technical", 384)

    compact = _select_answer_token_budget(
        QueryCategory.IMPLEMENTATION,
        AnswerMode.DIRECT,
        "How can I make a voice RAG agent?",
    )
    detailed = _select_answer_token_budget(
        QueryCategory.IMPLEMENTATION,
        AnswerMode.DETAILED,
        "Explain in detail how to make a voice RAG agent",
    )

    assert compact == 192
    assert detailed == 384


def test_direct_answer_is_trimmed_at_a_sentence_boundary() -> None:
    verbose = " ".join(
        [
            "SFT uses labeled prompt and completion pairs.",
            "It updates weights against static examples.",
            "RFT explores outputs and scores them with a reward function.",
            "It learns online without requiring static labels.",
            "This extra sentence is adjacent detail that was not requested.",
        ]
        * 4
    )

    concise = _enforce_direct_answer_length(verbose, max_words=50)

    assert len(concise.split()) <= 50
    assert concise.endswith(".")


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
