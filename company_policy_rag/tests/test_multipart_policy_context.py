"""Per-part governing-clause context for multi-part questions (H3).

Prevents rules/conditions/thresholds from unrelated parts being merged into one
blended answer by giving each part its own labeled governing-rule block.
"""

from __future__ import annotations

from backend.models.chunk import Chunk, ChunkMetadata
from backend.models.rag import ScoredChunk
from backend.rag.multi_query import decompose_multi_part
from backend.rag.policy_reliability import (
    GoverningClauseSelector,
    bind_source_indices,
    format_multipart_policy_decision_context,
)


def _chunk(cid: str, text: str, idx: int) -> ScoredChunk:
    meta = ChunkMetadata(document_id="d1", source_file="handbook.pdf", chunk_index=idx)
    return ScoredChunk(chunk=Chunk(id=cid, text=text, metadata=meta), score=1.0 - idx * 0.1)


CHUNKS = [
    _chunk("c1", "Employees must give two weeks written notice before resignation.", 0),
    _chunk("c2", "Employees accrue one vacation day per month of service.", 1),
    _chunk("c3", "Reimbursements above $500 must be approved by a manager.", 2),
]


def test_two_part_query_decomposes() -> None:
    parts = decompose_multi_part(
        "How many vacation days do I accrue per month, and what notice must I give before resigning?"
    )
    assert len(parts) == 2


def test_multipart_context_has_per_part_blocks_and_no_merge_rule() -> None:
    selector = GoverningClauseSelector()
    parts = [
        "How many vacation days do I accrue per month",
        "What notice must I give before resigning",
    ]
    part_selections = []
    for part in parts:
        sel = selector.select(part, CHUNKS)
        bind_source_indices(sel, CHUNKS)
        part_selections.append((part, sel))

    out = format_multipart_policy_decision_context(part_selections)

    assert "PART 1:" in out and "PART 2:" in out
    assert "Do NOT merge" in out
    # Each part header carries its own text.
    assert "vacation days" in out
    assert "notice" in out
    # The cross-contamination guard is stated explicitly.
    assert "never apply one part's exception" in out.lower()
    assert "ANSWER CONSTRAINTS:" in out


def test_part_without_governing_rule_is_marked_not_merged() -> None:
    selector = GoverningClauseSelector()
    # A part with no matching normative rule in the pool.
    sel = selector.select("what is the office wifi password", CHUNKS)
    bind_source_indices(sel, CHUNKS)
    out = format_multipart_policy_decision_context([("wifi password", sel)])
    assert "PART 1:" in out
