"""Source fidelity regressions.

The fixture pages mirror a real failure: a guidebook lists "five chunking
strategies" as numbered headings spread over consecutive pages, each heading at
the foot of one page and its explanation on the next, with PDF ligatures in the
text. Retrieval returned the introducing page, a table of contents and a related
section; the model listed four strategies plus "REFRAG" from the contents page.

No test here needs a real model: a scripted one returns the bad answers.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.models.chunk import Chunk, ChunkMetadata, ContentType
from backend.models.conversation import ConversationRAGState
from backend.models.rag import Citation, ScoredChunk
from backend.rag import source_fidelity as sf
from backend.rag.context_compression import ContextCompressor
from backend.rag.pipeline import (
    GROUNDED_ANSWER_WRITER_PROMPT,
    SOURCE_FIDELITY_ANSWER_PROMPT,
    RAGPipeline,
)
from backend.rag.verifier import SelfReflectionVerifier
from src.config import settings

HEADER = "DailyDoseofDS.com \n"

PAGES = {
    5: HEADER
    + "RAG.............................................................. 105 \n"
    + "What is RAG?..................................................... 106 \n"
    + "5 chunking strategies for RAG.................................... 119 \n"
    + "REFRAG........................................................... 140 \n"
    + "Agentic RAG...................................................... 150 \n4",
    119: HEADER
    + "Once the most relevant chunks are re-ranked, they are fed into the LLM. \n"
    + "This model combines the user's original query with the retrieved chunks. \n118",
    120: HEADER
    + "Since the additional document(s) can be large, step 1 also involves chunking, \n"
    + "wherein a large document is divided into smaller/manageable pieces. \n"
    + "Here are ﬁve chunking strategies for RAG: \n \nLet’s understand them! \n"
    + "1) Fixed-size chunking \n119",
    121: HEADER
    + "Split the text into uniform segments based on a pre-deﬁned number of \n"
    + "characters, words, or tokens. \n"
    + "But this usually breaks sentences (or ideas) in between. \n"
    + "2) Semantic chunking \n \n"
    + "Segment the document based on meaningful units like sentences, paragraphs, or \n"
    + "thematic sections. \nNext, create embeddings for each segment. \n120",
    122: HEADER
    + "This continues until cosine similarity drops signiﬁcantly. \n"
    + "The moment it does, we start a new chunk and repeat. \n"
    + "A minor problem is that it depends on a threshold to determine if cosine \n"
    + "similarity has dropped. \n"
    + "3) Recursive chunking \n \n"
    + "First, chunk based on inherent separators like paragraphs, or sections. \n121",
    123: HEADER
    + "Next, paragraph 1 is further split into smaller chunks. \n"
    + "However, there is some extra overhead in terms of implementation. \n"
    + "4) Document structure-based chunking \n \n"
    + "It utilizes the inherent structure of documents, like headings, sections, or \n"
    + "paragraphs, to deﬁne chunk boundaries. \n122",
    124: HEADER
    + "That said, this approach assumes that the document has a clear structure, which \n"
    + "may not be true. \n"
    + "5) LLM-based chunking \n \n"
    + "Prompt the LLM to generate semantically isolated and meaningful chunks. \n123",
    125: HEADER
    + "Also, since LLMs typically have a limited context window, that is something to be \n"
    + "taken care of. \nEach technique has its own advantages and trade-oﬀs. \n124",
    141: HEADER
    + "Essentially, instead of feeding the LLM every chunk and every token, REFRAG \n"
    + "compresses and ﬁlters context at a vector level. \n"
    + "Chunk compression: each chunk is encoded into a single compressed embedding. \n140",
}

NAMES = [
    "Fixed-size chunking",
    "Semantic chunking",
    "Recursive chunking",
    "Document structure-based chunking",
    "LLM-based chunking",
]


def _chunk(page: int, text: str | None = None, document_id: str = "doc_guide") -> Chunk:
    return Chunk(
        id=f"{document_id}_p{page}",
        text=text if text is not None else PAGES[page],
        metadata=ChunkMetadata(
            document_id=document_id,
            source_file="guide.pdf",
            page_number=page,
            chunk_index=page,
            section_title="Generate the final response",
            content_type=ContentType.PROSE,
        ),
    )


def _scored(chunks: list[Chunk]) -> list[ScoredChunk]:
    return [ScoredChunk(chunk=chunk, score=1.0 - 0.01 * rank) for rank, chunk in enumerate(chunks)]


@pytest.fixture
def docstore() -> dict[str, Chunk]:
    return {chunk.id: chunk for chunk in (_chunk(page) for page in PAGES)}


@pytest.fixture
def order(docstore) -> sf.DocumentOrder:
    return sf.DocumentOrder(docstore)


def _context(docstore: dict[str, Chunk], pages: list[int]) -> list[ScoredChunk]:
    return _scored([docstore[f"doc_guide_p{page}"] for page in pages])


class _NoVision:
    vision_model = "none"

    def __init__(self) -> None:
        self.image_asset_manager = MagicMock()
        self.image_asset_manager.get_page_assets_by_physical_page.return_value = []
        self.image_asset_manager.get_page_assets.return_value = []

    def is_query_time_available(self) -> tuple[bool, str]:
        return False, "vision disabled in tests"


def _pipeline(tmp_path: Path, docstore: dict[str, Chunk], retrieved: list[int], answer: str):
    pdf = tmp_path / "guide.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")
    for chunk in docstore.values():
        chunk.metadata.file_path = str(pdf)
    retriever = MagicMock()
    retriever.retrieve.return_value = _context(docstore, retrieved)
    llm = MagicMock()
    llm.complete.return_value = answer
    pipeline = RAGPipeline(hybrid_retriever=retriever, docstore=docstore, llm=llm, vision_service=_NoVision())
    return pipeline, llm, retriever


# What retrieval really returned for the chunking question: the introducing page,
# an unrelated page, the page after the list, and the table of contents.
RETRIEVED = [120, 119, 125, 5, 141]


# ── 1. Explicit numbered lists ──────────────────────────────────────────────


def test_list_is_rebuilt_from_one_retrieved_page(docstore, order):
    structure = sf.find_structure("What are the five chunking strategies for RAG?", _context(docstore, [120]), order)

    assert structure is not None
    assert structure.labels == NAMES
    assert structure.complete
    assert structure.noun_phrase == "chunking strategies"
    assert [c.metadata.page_number for c in structure.chunks] == [120, 121, 122, 123, 124, 125]
    # Explanations are read across the page break that separates them from their heading.
    assert structure.items[1].excerpt.startswith("Segment the document based on meaningful units")
    assert "cosine similarity" not in structure.items[2].excerpt


def test_list_question_is_answered_from_the_source_not_the_model(tmp_path, docstore):
    pipeline, llm, _ = _pipeline(tmp_path, docstore, RETRIEVED, "1. Fixed-size chunking\n2. REFRAG")

    response = pipeline.query(user_query="What are the five chunking strategies for RAG?")

    listed = [line.split(". ", 1)[1].split(" [")[0] for line in response.answer.splitlines() if line[:1].isdigit()]
    assert listed == NAMES
    assert "REFRAG" not in response.answer
    llm.complete.assert_not_called()
    cited_pages = sorted({c.page_number for c in response.citations})
    assert cited_pages and set(cited_pages) <= {120, 121, 122, 123, 124}


def test_asker_may_use_a_different_noun_than_the_source(docstore, order):
    structure = sf.find_structure("What are the five chunking techniques?", _context(docstore, [120]), order)

    assert structure is not None and structure.labels == NAMES
    assert sf.structure_scope("What are the five chunking techniques?", structure) == "list"


def test_a_list_about_something_else_is_not_used(docstore, order):
    context = _context(docstore, [120, 121])

    assert sf.find_structure("What are the five fine-tuning techniques?", context, order) is None
    assert sf.find_structure("What are the three chunking strategies?", context, order) is None
    assert sf.find_structure("How does the LLM use retrieved context?", context, order) is None


def test_ligatures_do_not_hide_the_introducer(docstore):
    assert "ﬁve" in docstore["doc_guide_p120"].text
    assert "five chunking strategies" in sf.normalize(docstore["doc_guide_p120"].text)


# ── 2. Missing list members ─────────────────────────────────────────────────


def test_incomplete_list_is_reported_not_padded(tmp_path, docstore):
    for page in (123, 124, 125):
        del docstore[f"doc_guide_p{page}"]
    pipeline, llm, _ = _pipeline(tmp_path, docstore, [120, 119], "irrelevant")

    response = pipeline.query(user_query="What are the five chunking strategies for RAG?")

    assert "only 3 of them" in response.answer
    assert "remaining 2 are not in the retrieved evidence" in response.answer
    for name in NAMES[:3]:
        assert name in response.answer
    for name in NAMES[3:]:
        assert name not in response.answer
    llm.complete.assert_not_called()


def test_padded_list_is_caught(docstore, order):
    for page in (123, 124, 125):
        del docstore[f"doc_guide_p{page}"]
    partial = sf.find_structure(
        "What are the five chunking strategies?", _context(docstore, [120]), sf.DocumentOrder(docstore)
    )
    assert partial is not None and not partial.complete and len(partial.items) == 3

    padded = "1. Fixed-size chunking\n2. Semantic chunking\n3. Recursive chunking\n4. Sliding-window chunking\n5. Agentic chunking"
    missing, extra = sf.check_structure_answer(padded, partial)

    assert missing == []
    assert extra == ["Sliding-window chunking", "Agentic chunking"]


# ── 3. Neighbouring-section contamination ───────────────────────────────────


def test_contents_page_is_not_evidence(docstore):
    context = _context(docstore, RETRIEVED)

    kept = sf.drop_navigation_chunks(context)

    assert sf.is_navigation_chunk(PAGES[5])
    assert 5 not in [sc.chunk.metadata.page_number for sc in kept]
    # Never leave the model with nothing.
    only_contents = _context(docstore, [5])
    assert sf.drop_navigation_chunks(only_contents) == only_contents


def test_related_section_cannot_join_the_list(docstore, order):
    context = sf.drop_navigation_chunks(_context(docstore, RETRIEVED))
    structure = sf.find_structure("What are the five chunking strategies?", context, order)

    evidence = sf.apply_structure_to_context(context, structure, max_chunks=6)
    answer = (
        "1. Fixed-size chunking [Source 1]\n2. Recursive chunking [Source 3]\n"
        "3. Document structure-based chunking [Source 4]\n4. LLM-based chunking [Source 5]\n5. REFRAG [Source 1]"
    )
    findings = sf.check_answer(answer, evidence, "What are the five chunking strategies?", structure)

    # The list's own pages lead, in source order; the REFRAG page no longer fits.
    assert [sc.chunk.metadata.page_number for sc in evidence] == [120, 121, 122, 123, 124, 125]
    assert findings.missing_items == ["Semantic chunking"]
    assert findings.extra_items == ["REFRAG"]


# ── 4. Concept vs mechanism ─────────────────────────────────────────────────


def test_explanation_chunk_is_given_the_name_it_continues(docstore, order):
    context = _context(docstore, [122])

    labels = sf.continuation_labels(context, order)
    prompt_context = ContextCompressor().format_context_for_prompt(context, continuation_labels=labels)

    assert labels == {"doc_guide_p122": "2) Semantic chunking"}
    assert 'continues "2) Semantic chunking"' in prompt_context
    # Without the label the model only sees the mechanism.
    assert "Semantic chunking" not in ContextCompressor().format_context_for_prompt(context).split("\n", 1)[1][:60]


def test_no_label_when_an_unnumbered_heading_intervenes(order):
    previous = _chunk(300, HEADER + "1) First technique \nIt works well. \nA New Section \nThis section covers something else. \n299")
    following = _chunk(301, HEADER + "The new section continues here with plenty of additional explanatory words. \n300")
    local_order = sf.DocumentOrder({previous.id: previous, following.id: following})

    assert sf.continuation_labels(_scored([following]), local_order) == {}


def test_renamed_member_is_rewritten_from_the_source(tmp_path, docstore):
    renamed = (
        "1. **Fixed-size chunking**: uniform segments [Source 1]\n"
        "2. **Cosine Similarity Chunking**: merges segments while similarity is high [Source 3]\n"
        "3. **Recursive chunking**: separators first [Source 3]\n"
        "4. **Document structure-based chunking**: headings [Source 4]\n"
        "5. **LLM-based chunking**: the LLM decides [Source 5]"
    )
    pipeline, llm, _ = _pipeline(tmp_path, docstore, RETRIEVED, renamed)

    response = pipeline.query(user_query="Explain the five chunking strategies for RAG.")

    llm.complete.assert_called_once()
    assert "Cosine Similarity Chunking" not in response.answer
    for name in NAMES:
        assert f"**{name}**" in response.answer
    assert "Segment the document based on meaningful units" in response.answer
    # Each explanation cites the page that states it, so the rewritten answer verifies.
    assert response.trace.verification_report["passed"] is True
    assert response.trace.verification_report["citation_errors"] == []


def test_faithful_explanation_is_left_alone(tmp_path, docstore):
    faithful = "\n".join(
        f"{i}. **{name}**: as described in the guide [Source {i}]" for i, name in enumerate(NAMES, start=1)
    )
    pipeline, _llm, _ = _pipeline(tmp_path, docstore, RETRIEVED, faithful)

    response = pipeline.query(user_query="Explain the five chunking strategies for RAG.")

    assert response.answer == faithful


def test_question_that_only_involves_the_list_is_not_forced_into_it(tmp_path, docstore):
    answer = "Semantic chunking merges segments while their cosine similarity stays high [Source 2]."
    pipeline, llm, _ = _pipeline(tmp_path, docstore, [121, 122], answer)

    question = "Which of the chunking strategies depends on a cosine similarity threshold?"
    structure = sf.find_structure(question, _context(docstore, [120]), sf.DocumentOrder(docstore))
    response = pipeline.query(user_query=question)

    assert structure is not None and sf.structure_scope(question, structure) == "related"
    assert response.answer == answer
    llm.complete.assert_called_once()


# ── 5. Terminology ──────────────────────────────────────────────────────────


def test_terms_absent_from_the_evidence_are_named(docstore):
    evidence = PAGES[121] + PAGES[122]
    answer = (
        "Semantic chunking groups segments by cosine similarity. It is what LangChain calls "
        "SemanticChunker, and is often deployed on Kubernetes with a `max_chunk_size` setting."
    )

    flagged = sf.find_unsupported_terms(answer, evidence, "How does semantic chunking work?")

    assert "LangChain" in flagged and "SemanticChunker" in flagged
    assert "Kubernetes" in flagged and "max_chunk_size" in flagged
    assert not any("emantic" in term and term != "SemanticChunker" for term in flagged)


def test_source_and_question_terms_are_never_flagged():
    evidence = "REFRAG compresses context. The LLM reads fewer tokens with Chunk Compression."
    answer = "REFRAG uses Chunk Compression so the LLM reads fewer tokens, unlike Agentic RAG."

    assert sf.find_unsupported_terms(answer, evidence, "How does REFRAG compare with Agentic RAG?") == []


# ── 6. Numerical claims ─────────────────────────────────────────────────────

_LEAVE = "Full-time employees accrue 15 days of annual leave per year, effective 1 January 2026. Up to 40% may carry over."


def _leave_chunks() -> list[ScoredChunk]:
    return _scored([_chunk(1, _LEAVE, "doc_policy")])


@pytest.mark.parametrize(
    "answer",
    [
        "Full-time employees accrue 20 days of annual leave per year [Source 1].",
        "Full-time employees accrue 15 weeks of annual leave per year [Source 1].",
        "Up to 50% of annual leave may carry over [Source 1].",
    ],
)
def test_changed_numbers_and_units_fail_verification(answer):
    report = SelfReflectionVerifier().verify("How much annual leave do employees accrue?", answer, _leave_chunks(), [])

    assert report.passed is False
    assert report.unsupported_claims


def test_exact_numbers_pass_verification():
    answer = "Full-time employees accrue 15 days of annual leave per year, and up to 40% may carry over [Source 1]."
    citation = Citation(source_index=1, chunk_id="doc_policy_p1", document_id="doc_policy", source_file="guide.pdf", snippet=_LEAVE)

    report = SelfReflectionVerifier().verify("How much annual leave do employees accrue?", answer, _leave_chunks(), [citation])

    assert report.unsupported_claims == []
    assert report.unsupported_terms == []
    assert report.citation_errors == []


# ── 7. Unsupported questions ────────────────────────────────────────────────


def test_answer_from_model_memory_is_flagged(tmp_path, docstore):
    from_memory = (
        "Quantum annealing finds low-energy states using D-Wave hardware and the QUBO formulation, "
        "which is common in logistics optimisation [Source 1]."
    )
    pipeline, _llm, _ = _pipeline(tmp_path, docstore, [119], from_memory)

    response = pipeline.query(user_query="How does quantum annealing work?")
    report = response.trace.verification_report

    assert report["passed"] is False
    assert {"QUBO", "D-Wave"} <= set(report["unsupported_terms"])
    assert report["citation_errors"]
    # The answer itself is not rewritten: unsupported content is flagged, not removed.
    assert response.answer == from_memory


def test_the_prompt_forbids_filling_gaps_from_memory(tmp_path, docstore):
    pipeline, llm, _ = _pipeline(tmp_path, docstore, [119], "It is not in the retrieved evidence.")

    pipeline.query(user_query="How does quantum annealing work?")
    prompt = llm.complete.call_args.args[0]

    assert "say it is not in the retrieved evidence; never fill gaps from memory" in prompt
    assert "Before writing, check silently:" in prompt


# ── 8. Conflicting evidence ─────────────────────────────────────────────────

_OLD = "Full-time employees accrue 15 days of annual leave per year."
_NEW = "Full-time employees accrue 20 days of annual leave per year."


def _conflicting() -> list[ScoredChunk]:
    return _scored([_chunk(1, _OLD, "doc_policy"), _chunk(2, _NEW, "doc_policy")])


def test_disagreeing_sources_are_surfaced_in_the_prompt():
    conflicts = sf.find_numeric_conflicts(_conflicting())

    assert [(a, va, b, vb) for a, va, b, vb, _ in conflicts] == [(1, "15 day", 2, "20 day")]
    block = sf.format_conflict_block(conflicts)
    assert "SOURCES DISAGREE" in block and "[Source 1] says 15 day" in block and "[Source 2] says 20 day" in block


def test_silently_choosing_one_source_is_flagged():
    question = "How much annual leave do employees accrue?"
    one_sided = sf.check_answer("Employees accrue 15 days of annual leave [Source 1].", _conflicting(), question)
    both = sf.check_answer(
        "The sources disagree: 15 days [Source 1] and 20 days [Source 2] of annual leave.", _conflicting(), question
    )

    assert one_sided.unreported_conflicts and "Source 2 says 20 day" in one_sided.unreported_conflicts[0]
    assert both.unreported_conflicts == []


def test_agreeing_sources_are_not_a_conflict():
    same = _scored([_chunk(1, _OLD, "doc_policy"), _chunk(2, _OLD + " This applies from the first day.", "doc_policy")])

    assert sf.find_numeric_conflicts(same) == []


# ── 9. Multi-turn references ────────────────────────────────────────────────

_LIST_ANSWER = "The five chunking strategies are:\n\n" + "\n".join(
    f"{i}. {name} [Source {i}]" for i, name in enumerate(NAMES, start=1)
)


@pytest.mark.parametrize(
    "follow_up,resolved",
    [
        ("explain the second one", "explain Semantic chunking"),
        ("tell me more about the last one", "tell me more about LLM-based chunking"),
        ("how does number 3 work?", "how does Recursive chunking work?"),
        ("what are the drawbacks of the fourth strategy", "what are the drawbacks of Document structure-based chunking"),
    ],
)
def test_ordinal_reference_resolves_to_the_item_name(follow_up, resolved):
    assert sf.resolve_ordinal_reference(follow_up, _LIST_ANSWER) == resolved


def test_ordinal_words_without_a_previous_list_are_left_alone():
    assert sf.resolve_ordinal_reference("explain the second one", "A paragraph answer with no list.") is None
    assert sf.resolve_ordinal_reference("explain the second one", None) is None
    assert sf.resolve_ordinal_reference("what is semantic chunking", _LIST_ANSWER) is None
    assert sf.resolve_ordinal_reference("explain the ninth one", _LIST_ANSWER) is None


def test_follow_up_is_interpreted_with_the_resolved_name(tmp_path, docstore):
    pipeline, _llm, _ = _pipeline(tmp_path, docstore, [121, 122], "Semantic chunking groups segments [Source 1].")
    state = ConversationRAGState(conversation_id="c1", last_answer=_LIST_ANSWER)
    interpret = pipeline.conversation_interpreter.interpret
    seen: list[str] = []
    pipeline.conversation_interpreter.interpret = lambda question, s: (seen.append(question), interpret(question, s))[1]

    pipeline.query(user_query="explain the second one", conversation_state=state)

    assert seen == ["explain Semantic chunking"]


# ── 10. Citations that are related but do not support the claim ─────────────


def test_citation_must_state_the_sentence(docstore):
    context = _context(docstore, [121, 141])
    wrong = "Fixed-size chunking splits text into uniform segments of a pre-defined number of tokens [Source 2]."
    right = wrong.replace("[Source 2]", "[Source 1]")

    assert sf.find_citation_errors(right, context) == []
    errors = sf.find_citation_errors(wrong, context)
    assert len(errors) == 1 and errors[0].startswith("Source 2 does not state")


def test_unsupportive_citation_fails_verification(docstore):
    context = _context(docstore, [121, 141])
    answer = "Fixed-size chunking splits text into uniform segments of a pre-defined number of tokens [Source 2]."

    report = SelfReflectionVerifier().verify("What is fixed-size chunking?", answer, context, [])

    assert report.passed is False
    assert report.citation_errors and "Source 2" in report.citation_errors[0]


# ── Prompt and switch ───────────────────────────────────────────────────────


def test_prompt_adds_the_fidelity_checks_to_the_tested_rules():
    prompt = SOURCE_FIDELITY_ANSWER_PROMPT

    for rule in (
        "answer every part",
        "keep its items, names, count and order",
        "Never add, drop, rename, merge or split items",
        "A mechanism, example or metric is not the name of a technique or category",
        "Related is not relevant",
        "Copy numbers, units, dates and names exactly",
        "never fill gaps from memory",
        "Label inferences as inferences",
        "If sources disagree, report each version with its source",
    ):
        assert rule in prompt
    # The answer-writing rules were A/B-tested on the local model; restating them
    # as a numbered checklist made it append copied source headers to answers.
    # Every tested rule but the one the fidelity wording replaces stays verbatim.
    old_rules = [
        line for line in GROUNDED_ANSWER_WRITER_PROMPT.splitlines() if line.startswith("- ")
    ]
    kept = [line for line in old_rules if line in prompt]
    assert len(old_rules) - len(kept) == 1
    assert "If evidence is incomplete or conflicting" in (set(old_rules) - set(kept)).pop()
    # A weaker local model with a 4,096-token window: the additions stay small.
    assert len(prompt.split()) <= len(GROUNDED_ANSWER_WRITER_PROMPT.split()) + 110


def test_layer_can_be_switched_off(tmp_path, docstore, monkeypatch):
    monkeypatch.setattr(settings, "source_fidelity_enabled", False)
    model_answer = "1. Fixed-size chunking\n2. REFRAG [Source 1]"
    pipeline, llm, _ = _pipeline(tmp_path, docstore, RETRIEVED, model_answer)

    response = pipeline.query(user_query="What are the five chunking strategies for RAG?")

    assert response.answer == model_answer
    assert GROUNDED_ANSWER_WRITER_PROMPT.split("\n", 1)[0] in llm.complete.call_args.args[0]
    assert "SOURCE-DEFINED LIST" not in llm.complete.call_args.args[0]


def test_statement_about_the_evidence_is_not_a_citation_error(docstore):
    context = _context(docstore, [121, 141])
    abstention = (
        "The provided evidence does not contain any information regarding dental insurance plans. "
        "Therefore, based on the retrieved evidence, it cannot be stated which plan is offered [Source 1]."
    )

    assert sf.find_citation_errors(abstention, context) == []
