"""BM25 performance/structure changes (L3).

Filter matching uses per-chunk metadata dicts precomputed once (no model_dump
per query), add_chunks extends the index incrementally, and removals keep the
entries / tokenized corpus / filter views in lockstep.
"""

from __future__ import annotations

from backend.models.chunk import Chunk, ChunkMetadata
from backend.retrieval.bm25 import BM25SearchIndex


def _chunk(cid: str, text: str, *, doc="d1", src="a.pdf", category="policy", extra=None) -> Chunk:
    meta = ChunkMetadata(
        document_id=doc,
        source_file=src,
        chunk_index=int(cid[-1]) if cid[-1].isdigit() else 0,
        category=category,
        extra=extra or {},
    )
    return Chunk(id=cid, text=text, metadata=meta)


def _index(chunks) -> BM25SearchIndex:
    idx = BM25SearchIndex(storage_dir="unused")
    idx.build_index(chunks)
    return idx


CHUNKS = [
    _chunk("c1", "vacation leave accrual policy for employees", doc="d1", src="a.pdf"),
    _chunk("c2", "resignation notice period two weeks", doc="d2", src="b.pdf"),
    _chunk("c3", "diagram of the approval workflow", doc="d1", src="a.pdf",
           extra={"visual_type": "diagram"}),
]


def test_meta_dicts_parallel_to_entries() -> None:
    idx = _index(CHUNKS)
    assert len(idx._meta_dicts) == len(idx.entries) == 3


def test_filter_on_field() -> None:
    idx = _index(CHUNKS)
    hits = idx.search("policy workflow", top_k=10, filters={"document_id": "d1"})
    assert hits
    assert all(h.chunk.metadata.document_id == "d1" for h in hits)


def test_filter_on_list_value() -> None:
    idx = _index(CHUNKS)
    hits = idx.search("policy notice", top_k=10, filters={"document_id": ["d1", "d2"]})
    got = {h.chunk.metadata.document_id for h in hits}
    assert got <= {"d1", "d2"} and got


def test_filter_on_extra_nested_key() -> None:
    idx = _index(CHUNKS)
    hits = idx.search("diagram workflow", top_k=10, filters={"visual_type": "diagram"})
    assert [h.chunk.id for h in hits] == ["c3"]


def test_no_filter_returns_matches() -> None:
    idx = _index(CHUNKS)
    hits = idx.search("policy", top_k=10)
    assert any(h.chunk.id == "c1" for h in hits)


def test_add_chunks_is_incremental_and_searchable() -> None:
    idx = _index(CHUNKS[:2])
    assert len(idx.entries) == 2
    added = idx.add_chunks([_chunk("c9", "remote work stipend reimbursement", doc="d3")])
    assert added == 1
    assert len(idx.entries) == len(idx._meta_dicts) == 3
    hits = idx.search("reimbursement stipend", top_k=5)
    assert any(h.chunk.id == "c9" for h in hits)


def test_add_chunks_skips_empty_text() -> None:
    idx = _index(CHUNKS)
    before = len(idx.entries)
    added = idx.add_chunks([_chunk("c0", "   ", src="", category="")])
    # A chunk with no searchable text is skipped.
    assert added == 0
    assert len(idx.entries) == before


def test_remove_keeps_structures_in_sync() -> None:
    idx = _index(CHUNKS)
    idx.remove_by_document_id("d1")
    assert len(idx.entries) == len(idx._tokenized_corpus) == len(idx._meta_dicts) == 1
    assert idx.entries[0].metadata.document_id == "d2"
    # Filtering still works after removal (meta_dicts stayed aligned).
    hits = idx.search("notice", top_k=5, filters={"document_id": "d2"})
    assert [h.chunk.id for h in hits] == ["c2"]


def test_remove_by_source_file() -> None:
    idx = _index(CHUNKS)
    idx.remove_by_source_file("a.pdf")
    assert {h.metadata.source_file for h in idx.entries} == {"b.pdf"}


def test_clear_resets_all() -> None:
    idx = _index(CHUNKS)
    idx.clear()
    assert idx.entries == [] and idx._meta_dicts == [] and idx._tokenized_corpus == []
    assert idx.search("policy") == []
