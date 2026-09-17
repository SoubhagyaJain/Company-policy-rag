"""Citation cards must match what the answer cites and how the prompt numbered sources."""

from __future__ import annotations

from backend.models.chunk import Chunk, ChunkMetadata
from backend.models.rag import ScoredChunk
from backend.rag.citations import CitationEngine


def _scored(index: int, rerank: float | None = None, score: float = 0.5, rank: int | None = None) -> ScoredChunk:
    chunk = Chunk(
        id=f"c{index}",
        text=f"Distinct passage number {index} " + "about leave policy " * 10,
        metadata=ChunkMetadata(document_id="doc_aaaaaaaaaaaa", source_file="handbook.pdf", page_number=4),
    )
    return ScoredChunk(chunk=chunk, score=score, rerank_score=rerank, rank=rank)


def test_every_cited_source_gets_a_card_beyond_max_citations() -> None:
    chunks = [_scored(i) for i in range(1, 7)]
    answer = "Leave accrues monthly [Source 1]. Carry-over is capped [Source 5]. Payout differs [Source 6]."

    citations = CitationEngine().select_citations(answer, chunks, max_citations=2)

    assert [c.source_index for c in citations] == [1, 5, 6]
    assert [c.chunk_id for c in citations] == ["c1", "c5", "c6"]


def test_distinct_passages_on_the_same_page_are_not_merged() -> None:
    chunks = [_scored(1), _scored(2)]
    citations = CitationEngine().select_citations("A [Source 1]. B [Source 2].", chunks, max_citations=4)
    assert len(citations) == 2
    assert {c.page_number for c in citations} == {4}


def test_fallback_citations_use_prompt_position_not_retrieval_rank() -> None:
    chunks = [_scored(1, score=0.9, rank=7), _scored(2, score=0.85, rank=3)]

    citations = CitationEngine().select_citations("An answer without tags.", chunks, max_citations=2)

    assert [c.source_index for c in citations] == [1, 2]


def test_probability_rerank_scores_are_not_squashed_again() -> None:
    engine = CitationEngine()
    high = engine.select_citations("x [Source 1]", [_scored(1, rerank=0.93)])[0]
    low = engine.select_citations("x [Source 1]", [_scored(1, rerank=0.12)])[0]
    assert high.relevance_score == 0.93
    assert low.relevance_score == 0.12


def test_parenthesised_and_grouped_source_tags_are_recognised() -> None:
    chunks = [_scored(i) for i in range(1, 6)]
    answer = "Interns get one day (Source 2). Staff get three [Source 1, 4]."

    citations = CitationEngine().select_citations(answer, chunks, max_citations=2)

    assert [c.source_index for c in citations] == [1, 2, 4]
    assert all(c.selection_reason == "cited_in_answer" for c in citations)
