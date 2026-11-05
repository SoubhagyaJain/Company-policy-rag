import json
from types import SimpleNamespace as NS

from backend.ingestion.companion_metadata import enrich_chunks
from backend.models.chunk import Chunk, ChunkMetadata
from backend.retrieval.bm25 import BM25SearchIndex

HASH = "a" * 64


def fixture(tmp_path):
    data = {
        "metadata_schema_version": "1.0",
        "document": {"source_sha256": HASH, "page_count": 2},
        "chapters": [
            {"chapter_id": "c1", "title": "Systems", "source_location": {"pdf_page_start": 1, "pdf_page_end": 2}}
        ],
        "sections": [
            {
                "chapter_id": "c1",
                "section_id": "s1",
                "title": "Hidden signals",
                "source_location": {"pdf_page_start": 1, "pdf_page_end": 2},
            }
        ],
        "subsections": [],
    }
    (tmp_path / "metadata.json").write_text(json.dumps(data), encoding="utf-8")
    docs = [
        NS(content="Systems\nHidden signals", metadata=NS(page_number=1, total_pages=2, extra={})),
        NS(content="real source evidence", metadata=NS(page_number=2, total_pages=2, extra={})),
    ]
    chunks = [
        Chunk(
            id="a",
            text="real source evidence",
            metadata=ChunkMetadata(document_id="d", source_file="book.pdf", page_number=2),
        ),
        Chunk(
            id="b",
            text="other source evidence",
            metadata=ChunkMetadata(document_id="other", source_file="other.pdf", page_number=2),
        ),
    ]
    return data, docs, chunks


def test_enrichment_and_bm25_filters(tmp_path):
    _, docs, chunks = fixture(tmp_path)
    assert enrich_chunks(chunks[:1], docs, HASH, tmp_path) == 1
    assert chunks[0].text == "real source evidence"
    index = BM25SearchIndex(storage_dir=str(tmp_path / "index"))
    index.build_index(chunks)
    assert [s.chunk.id for s in index.search("hidden signals", filters={"document_id": "d"})] == ["a"]
    assert index.search("hidden signals", filters={"document_id": "other"}) == []
    index.save()
    restored = BM25SearchIndex(storage_dir=str(tmp_path / "index"))
    assert restored.load()
    assert restored.search("hidden signals")[0].chunk.id == "a"
    restored.remove_by_document_id("d")
    assert not restored.search("hidden signals")


def test_mismatch_unverified_and_duplicate(tmp_path):
    data, docs, chunks = fixture(tmp_path)
    assert enrich_chunks(chunks, docs, "b" * 64, tmp_path) == 0
    data["sections"][0]["title"] = "Invented Heading"
    (tmp_path / "metadata.json").write_text(json.dumps(data), encoding="utf-8")
    enrich_chunks(chunks[:1], docs, HASH, tmp_path)
    assert "Invented" not in chunks[0].metadata.extra["companion_navigation"]
    (tmp_path / "duplicate.json").write_text(json.dumps(data), encoding="utf-8")
    assert enrich_chunks(chunks[1:], docs, HASH, tmp_path) == 0


def test_invalid_structure_and_page_count(tmp_path):
    data, docs, chunks = fixture(tmp_path)
    data["sections"].append(data["sections"][0])
    (tmp_path / "metadata.json").write_text(json.dumps(data), encoding="utf-8")
    assert enrich_chunks(chunks, docs, HASH, tmp_path) == 0
    data["sections"].pop()
    data["sections"][0]["chapter_id"] = "missing"
    (tmp_path / "metadata.json").write_text(json.dumps(data), encoding="utf-8")
    assert enrich_chunks(chunks, docs, HASH, tmp_path) == 0
    data["sections"][0]["chapter_id"] = "c1"
    data["document"]["page_count"] = 3
    (tmp_path / "metadata.json").write_text(json.dumps(data), encoding="utf-8")
    assert enrich_chunks(chunks, docs, HASH, tmp_path) == 0
    (tmp_path / "metadata.json").write_text("{", encoding="utf-8")
    assert enrich_chunks(chunks, docs, HASH, tmp_path) == 0


def test_text_only_index_excludes_navigation(tmp_path):
    _, docs, chunks = fixture(tmp_path)
    enrich_chunks(chunks[:1], docs, HASH, tmp_path)
    index = BM25SearchIndex(storage_dir=str(tmp_path / "index"), metadata_fields="")
    index.build_index(chunks)
    assert not index.search("hidden signals")


def test_citation_uses_only_original_chunk_text_and_page(tmp_path):
    from backend.models.rag import ScoredChunk
    from backend.rag.citations import CitationEngine

    _, docs, chunks = fixture(tmp_path)
    enrich_chunks(chunks[:1], docs, HASH, tmp_path)
    scored = ScoredChunk(chunk=chunks[0], score=1.0)
    citation = CitationEngine()._build_citation_from_chunk(1, scored, "test")
    assert citation.snippet == "real source evidence"
    assert citation.page_number == 2
    assert citation.document_id == "d"
    assert "Hidden signals" not in citation.snippet
