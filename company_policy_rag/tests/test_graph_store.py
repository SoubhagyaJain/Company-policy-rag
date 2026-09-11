"""Tests for PolicyGraphStore persistence, versioning, removal and idempotency."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from backend.graph.models import NodeType
from backend.graph.store import GRAPH_FILE, PolicyGraphStore
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


def sample_chunks(document_id: str = "d1") -> list[Chunk]:
    return [
        make_chunk(
            document_id,
            0,
            "Overtime is paid per FLSA rules.",
            section_path="Pay > Overtime",
            section_number="4.2",
            clause_id="4.2",
            policy_id="HR-104",
            key_entities=["FLSA", "Overtime"],
        ),
        make_chunk(
            document_id,
            1,
            "Continued overtime discussion.",
            section_path="Pay > Overtime",
            section_number="4.2",
            policy_id="HR-104",
            key_entities=["Overtime"],
        ),
    ]


def test_save_load_round_trip(tmp_path: Path) -> None:
    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks(sample_chunks())
    before = store.stats()
    store.save()

    reloaded = PolicyGraphStore(storage_dir=tmp_path)
    assert reloaded.load() is True
    assert reloaded.stats() == before


def test_load_missing_file_returns_false(tmp_path: Path) -> None:
    assert PolicyGraphStore(storage_dir=tmp_path).load() is False


def test_load_corrupt_file_quarantines_and_returns_false(tmp_path: Path) -> None:
    (tmp_path / GRAPH_FILE).write_text("{not valid json", encoding="utf-8")

    store = PolicyGraphStore(storage_dir=tmp_path)
    assert store.load() is False
    # Corrupt graphs are preserved for debugging, never deleted.
    assert not (tmp_path / GRAPH_FILE).exists()
    assert list(tmp_path.glob(f"{GRAPH_FILE}.corrupt-*"))


def test_load_schema_mismatch_returns_false_without_quarantine(tmp_path: Path) -> None:
    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks(sample_chunks())
    store.save()

    target = tmp_path / GRAPH_FILE
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["graph"]["schema_version"] = PolicyGraphStore.SCHEMA_VERSION + 1
    target.write_text(json.dumps(payload), encoding="utf-8")

    fresh = PolicyGraphStore(storage_dir=tmp_path)
    assert fresh.load() is False
    # A future-version graph is readable data, not corruption — leave it in place.
    assert target.exists()


def test_save_leaves_no_temp_files(tmp_path: Path) -> None:
    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks(sample_chunks())
    store.save()
    assert not list(tmp_path.glob(".graph-*.tmp"))


def test_save_does_not_clobber_previous_on_failure(tmp_path: Path, monkeypatch) -> None:
    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks(sample_chunks())
    store.save()
    original = (tmp_path / GRAPH_FILE).read_text(encoding="utf-8")

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("backend.graph.store.os.replace", boom)
    store.add_chunks(sample_chunks("d2"))
    try:
        store.save()
    except OSError:
        pass

    # The previous graph survives a failed write intact.
    assert (tmp_path / GRAPH_FILE).read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".graph-*.tmp"))


def test_build_index_is_idempotent(tmp_path: Path) -> None:
    chunks = sample_chunks()
    store = PolicyGraphStore(storage_dir=tmp_path)
    store.build_index(chunks)
    first = (store.graph.number_of_nodes(), store.graph.number_of_edges())

    store.build_index(chunks)
    assert (store.graph.number_of_nodes(), store.graph.number_of_edges()) == first


def test_add_chunks_twice_does_not_duplicate_edges(tmp_path: Path) -> None:
    chunks = sample_chunks()
    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks(chunks)
    first = (store.graph.number_of_nodes(), store.graph.number_of_edges())

    store.add_chunks(chunks)
    assert (store.graph.number_of_nodes(), store.graph.number_of_edges()) == first


def test_graph_version_increments_on_mutation(tmp_path: Path) -> None:
    store = PolicyGraphStore(storage_dir=tmp_path)
    assert store.graph_version == 0
    store.add_chunks(sample_chunks())
    assert store.graph_version > 0

    before = store.graph_version
    store.remove_by_document_id("d1")
    assert store.graph_version > before


def test_remove_by_document_id_leaves_no_orphans(tmp_path: Path) -> None:
    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks(sample_chunks("d1"))
    store.remove_by_document_id("d1")

    assert store.graph.number_of_nodes() == 0
    assert store.graph.number_of_edges() == 0


def test_shared_entity_survives_until_last_document_removed(tmp_path: Path) -> None:
    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks(sample_chunks("d1"))
    store.add_chunks(sample_chunks("d2"))
    assert "entity:flsa" in store.graph

    d2_chunk_ids = {
        cid
        for _, data in store.graph.nodes(data=True)
        if data.get("document_id") == "d2"
        for cid in data.get("chunk_ids", [])
    }

    # A global entity shared by two documents must not vanish with the first, and
    # must keep only the surviving document's chunk ids.
    store.remove_by_document_id("d1")
    assert "entity:flsa" in store.graph
    remaining = store.graph.nodes["entity:flsa"]["chunk_ids"]
    assert remaining
    assert set(remaining) <= d2_chunk_ids

    store.remove_by_document_id("d2")
    assert "entity:flsa" not in store.graph


def test_remove_by_source_file(tmp_path: Path) -> None:
    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks(sample_chunks("d1"))
    store.remove_by_source_file("d1.pdf")
    assert store.graph.number_of_nodes() == 0


def test_clear_resets_to_empty_graph(tmp_path: Path) -> None:
    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks(sample_chunks())
    store.clear()
    assert store.graph.number_of_nodes() == 0
    assert store.graph.graph["schema_version"] == PolicyGraphStore.SCHEMA_VERSION


def test_node_types_present(tmp_path: Path) -> None:
    store = PolicyGraphStore(storage_dir=tmp_path)
    store.add_chunks(sample_chunks())
    types = {data["node_type"] for _, data in store.graph.nodes(data=True)}
    assert NodeType.DOCUMENT.value in types
    assert NodeType.CHUNK.value in types
    assert NodeType.SECTION.value in types
    assert NodeType.POLICY.value in types
    assert NodeType.ENTITY.value in types


def test_concurrent_add_and_read_is_safe(tmp_path: Path) -> None:
    """Ingestion runs on a worker thread while queries read; neither may corrupt."""
    store = PolicyGraphStore(storage_dir=tmp_path)
    errors: list[BaseException] = []

    def writer(doc_index: int) -> None:
        try:
            for i in range(10):
                store.add_chunks(sample_chunks(f"doc{doc_index}_{i}"))
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    def reader() -> None:
        try:
            for _ in range(50):
                with store.lock:
                    list(store.graph.nodes(data=True))
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(3)]
    threads += [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert store.graph.number_of_nodes() > 0
