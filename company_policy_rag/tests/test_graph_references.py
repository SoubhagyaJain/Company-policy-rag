"""Tests for cross-reference extraction, scoped resolution, and deferred repair.

The governing rule under test: a wrong cross-reference is worse than a missing
one. Every ambiguous case must produce no edge.
"""

from __future__ import annotations

from pathlib import Path

from backend.graph.builder import StructuralGraphBuilder
from backend.graph.models import EdgeType
from backend.graph.store import PolicyGraphStore
from backend.models.chunk import Chunk, ChunkMetadata


def make_chunk(document_id: str, index: int, text: str = "text", **meta: object) -> Chunk:
    return Chunk(
        text=text,
        metadata=ChunkMetadata(
            document_id=document_id,
            source_file=f"{document_id}.pdf",
            chunk_index=index,
            **meta,
        ),
    )


def reference_edges(store: PolicyGraphStore) -> list[tuple[str, str]]:
    return [
        (u, v)
        for u, v, k in store.graph.edges(keys=True)
        if k == EdgeType.REFERENCES.value
    ]


# ── extraction ──────────────────────────────────────────────────────────────


def test_extracts_section_symbol_reference() -> None:
    chunk = make_chunk("d1", 0, "Subject to §7.3 of this policy.")
    found = StructuralGraphBuilder._extract_references(chunk)
    assert ("§7.3", "7.3") in found


def test_extracts_spelled_section_and_clause_references() -> None:
    chunk = make_chunk("d1", 0, "See Section 4.2 and clause 9.1(b) below.")
    keys = {key for _, key in StructuralGraphBuilder._extract_references(chunk)}
    assert "4.2" in keys
    assert "9.1(b)" in keys


def test_extracts_policy_citation_with_its_own_namespace() -> None:
    chunk = make_chunk("d1", 0, "As required by Policy HR-104.")
    keys = {key for _, key in StructuralGraphBuilder._extract_references(chunk)}
    assert "policy::hr-104" in keys


def test_prose_without_references_yields_nothing() -> None:
    chunk = make_chunk("d1", 0, "Employees should contact their manager for details.")
    assert StructuralGraphBuilder._extract_references(chunk) == []


# ── resolution cascade ──────────────────────────────────────────────────────


def test_resolves_within_same_document(tmp_path: Path) -> None:
    chunks = [
        make_chunk("d1", 0, "Subject to the exceptions in Section 7.3.", section_number="4.2"),
        make_chunk("d1", 1, "Exceptions apply here.", section_number="7.3", clause_id="7.3"),
    ]
    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks(chunks)

    assert reference_edges(store) == [
        (f"chunk:{chunks[0].id}", "clause:d1:7.3"),
    ]
    assert store.pending_references == []


def test_same_document_wins_over_other_documents(tmp_path: Path) -> None:
    """A local §7.3 must win even when other documents also define a §7.3."""
    local = [
        make_chunk("d1", 0, "See Section 7.3.", section_number="4.2"),
        make_chunk("d1", 1, "Local exceptions.", section_number="7.3", clause_id="7.3"),
    ]
    other = [make_chunk("d2", 0, "Other exceptions.", section_number="7.3", clause_id="7.3")]

    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks(other)
    store.add_chunks(local)

    assert (f"chunk:{local[0].id}", "clause:d1:7.3") in reference_edges(store)


def test_ambiguous_target_across_documents_produces_no_edge(tmp_path: Path) -> None:
    """Two documents define §7.3, the referrer defines neither — refuse to guess."""
    referrer = make_chunk("d0", 0, "See Section 7.3.", section_number="1.1")
    a = make_chunk("d1", 0, "A exceptions.", section_number="7.3", clause_id="7.3")
    b = make_chunk("d2", 0, "B exceptions.", section_number="7.3", clause_id="7.3")

    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks([a])
    store.add_chunks([b])
    store.add_chunks([referrer])

    assert reference_edges(store) == []
    assert len(store.pending_references) == 1


def test_policy_scope_resolves_when_document_scope_has_no_candidate(tmp_path: Path) -> None:
    """Referrer's own document lacks §7.3; its policy sibling supplies it uniquely."""
    referrer = make_chunk("d1", 0, "See Section 7.3.", section_number="1.1", policy_id="HR-104")
    sibling = make_chunk(
        "d2", 0, "Exceptions.", section_number="7.3", clause_id="7.3", policy_id="HR-104"
    )
    unrelated = make_chunk(
        "d3", 0, "Unrelated 7.3.", section_number="7.3", clause_id="7.3", policy_id="FIN-900"
    )

    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks([sibling])
    store.add_chunks([unrelated])
    store.add_chunks([referrer])

    assert (f"chunk:{referrer.id}", "clause:d2:7.3") in reference_edges(store)


def test_globally_unique_target_resolves(tmp_path: Path) -> None:
    referrer = make_chunk("d1", 0, "See Section 7.3.", section_number="1.1")
    target = make_chunk("d2", 0, "Exceptions.", section_number="7.3", clause_id="7.3")

    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks([target])
    store.add_chunks([referrer])

    assert (f"chunk:{referrer.id}", "clause:d2:7.3") in reference_edges(store)


def test_policy_citation_resolves_to_policy_node(tmp_path: Path) -> None:
    referrer = make_chunk("d1", 0, "As required by Policy HR-104.")
    governed = make_chunk("d2", 0, "Body.", policy_id="HR-104")

    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks([governed])
    store.add_chunks([referrer])

    assert (f"chunk:{referrer.id}", "policy:hr-104") in reference_edges(store)


def test_self_reference_is_skipped(tmp_path: Path) -> None:
    """A clause restating its own number is noise, not a link."""
    chunk = make_chunk("d1", 0, "This Section 4.2 governs overtime.", section_number="4.2", clause_id="4.2")
    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks([chunk])

    assert reference_edges(store) == []
    assert store.pending_references == []


def test_unresolvable_reference_is_queued_not_guessed(tmp_path: Path) -> None:
    chunk = make_chunk("d1", 0, "See Section 99.9.", section_number="1.1")
    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks([chunk])

    assert reference_edges(store) == []
    assert [p["target_key"] for p in store.pending_references] == ["99.9"]


# ── deferred resolution and repair ──────────────────────────────────────────


def test_pending_reference_resolves_when_target_ingested_later(tmp_path: Path) -> None:
    """Cross-references must not silently depend on upload order."""
    referrer = make_chunk("d1", 0, "See Section 7.3.", section_number="1.1")
    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks([referrer])
    assert store.pending_references  # nothing to point at yet

    target = make_chunk("d2", 0, "Exceptions.", section_number="7.3", clause_id="7.3")
    store.add_chunks([target])

    assert (f"chunk:{referrer.id}", "clause:d2:7.3") in reference_edges(store)
    assert store.pending_references == []


def test_deleting_target_document_returns_reference_to_pending(tmp_path: Path) -> None:
    referrer = make_chunk("d1", 0, "See Section 7.3.", section_number="1.1")
    target = make_chunk("d2", 0, "Exceptions.", section_number="7.3", clause_id="7.3")

    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks([target])
    store.add_chunks([referrer])
    assert reference_edges(store)

    store.remove_by_document_id("d2")
    assert reference_edges(store) == []
    assert [p["target_key"] for p in store.pending_references] == ["7.3"]


def test_reupload_of_target_restores_the_reference_edge(tmp_path: Path) -> None:
    referrer = make_chunk("d1", 0, "See Section 7.3.", section_number="1.1")
    target = make_chunk("d2", 0, "Exceptions.", section_number="7.3", clause_id="7.3")

    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks([target])
    store.add_chunks([referrer])
    store.remove_by_document_id("d2")

    # Re-upload produces fresh chunk ids but identical clause node ids.
    reuploaded = make_chunk("d2", 0, "Exceptions.", section_number="7.3", clause_id="7.3")
    store.add_chunks([reuploaded])

    assert (f"chunk:{referrer.id}", "clause:d2:7.3") in reference_edges(store)
    assert store.pending_references == []


def test_deleting_source_document_drops_its_pending_entries(tmp_path: Path) -> None:
    referrer = make_chunk("d1", 0, "See Section 99.9.", section_number="1.1")
    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks([referrer])
    assert store.pending_references

    store.remove_by_document_id("d1")
    # Trigger a resolution pass; the orphaned entry must be discarded.
    store.add_chunks([make_chunk("d2", 0, "Unrelated.")])
    assert store.pending_references == []


def test_pending_references_survive_save_and_load(tmp_path: Path) -> None:
    referrer = make_chunk("d1", 0, "See Section 99.9.", section_number="1.1")
    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks([referrer])
    store.save()

    reloaded = PolicyGraphStore(storage_dir=tmp_path)
    assert reloaded.load() is True
    assert [p["target_key"] for p in reloaded.pending_references] == ["99.9"]
