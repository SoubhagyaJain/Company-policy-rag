"""
Hierarchical parent-child chunking with code-block protection.

Parents aggregate section/page context (stored in docstore, not embedded).
Children are retrieval units in Chroma; code blocks are never split mid-block.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode

from src.config import settings
from src.docstore import new_parent_id, register_parent_nodes
from src.utils import logger

_FENCE_RE = re.compile(r"(```[\w-]*\n.*?```)", re.DOTALL)


@dataclass
class HierarchicalNodes:
    """Parent nodes (docstore) + embeddable nodes (Chroma)."""

    parent_nodes: list[TextNode]
    embed_nodes: list[TextNode]


def _approx_token_count(text: str) -> int:
    """Fast token estimate when no tokenizer is wired."""
    return max(1, len(text) // 4)


def _child_splitter() -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        paragraph_separator="\n\n",
    )


def _split_into_segments(text: str, *, block_types: str = "") -> list[tuple[str, str]]:
    """
    Split text into (content_type, segment) tuples.

    content_type is 'code' or 'prose'. Marker block_types hint is a comma list.
    """
    if "code" in (block_types or "").lower() and not _FENCE_RE.search(text):
        if _approx_token_count(text) > 40:
            return [("code", text)]

    segments: list[tuple[str, str]] = []
    last = 0
    for match in _FENCE_RE.finditer(text):
        if match.start() > last:
            prose = text[last : match.start()].strip()
            if prose:
                segments.append(("prose", prose))
        segments.append(("code", match.group(1).strip()))
        last = match.end()

    tail = text[last:].strip()
    if tail:
        segments.append(("prose", tail))

    if not segments:
        segments.append(("prose", text))
    return segments


def _chunk_prose(
    text: str,
    base_meta: dict[str, Any],
    *,
    parent_id: str,
) -> list[TextNode]:
    splitter = _child_splitter()
    doc = Document(text=text, metadata=base_meta)
    nodes = splitter.get_nodes_from_documents([doc], show_progress=False)
    children: list[TextNode] = []
    for node in nodes:
        child_id = f"child_{uuid.uuid4().hex[:12]}"
        meta = dict(base_meta)
        meta.update(
            {
                "node_role": "child",
                "parent_id": parent_id,
                "content_type": "prose",
                "is_atomic": False,
            }
        )
        children.append(TextNode(text=node.text or "", metadata=meta, id_=child_id))
    return children


def _chunk_code(
    text: str,
    base_meta: dict[str, Any],
    *,
    parent_id: str,
) -> list[TextNode]:
    source_file = base_meta.get("source_file", "document")
    page = base_meta.get("display_page_number") or base_meta.get("page_label") or base_meta.get("page_number")
    page_label = f" p.{page}" if page else ""
    prefix = f"[CODE BLOCK — {source_file}{page_label}]\n"
    body = prefix + text

    max_tokens = settings.chunk_size
    if _approx_token_count(body) <= max_tokens:
        child_id = f"child_{uuid.uuid4().hex[:12]}"
        meta = dict(base_meta)
        meta.update(
            {
                "node_role": "child",
                "parent_id": parent_id,
                "content_type": "code",
                "is_atomic": True,
            }
        )
        return [TextNode(text=body, metadata=meta, id_=child_id)]

    # Oversized code: keep atomic (do not split inside code)
    child_id = f"child_{uuid.uuid4().hex[:12]}"
    meta = dict(base_meta)
    meta.update(
        {
            "node_role": "child",
            "parent_id": parent_id,
            "content_type": "code",
            "is_atomic": True,
        }
    )
    return [TextNode(text=body, metadata=meta, id_=child_id)]


def _build_parent_text(docs: list[Document]) -> str:
    parts: list[str] = []
    for doc in docs:
        text = (doc.text or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _group_documents_for_parents(documents: list[Document]) -> list[list[Document]]:
    """Group consecutive pages under the same section_path within parent size budget."""
    if not documents:
        return []

    sorted_docs = sorted(
        documents,
        key=lambda d: (
            d.metadata.get("source_file", ""),
            d.metadata.get("page_number") if isinstance(d.metadata.get("page_number"), int) else 0,
        ),
    )

    groups: list[list[Document]] = []
    current: list[Document] = []
    current_tokens = 0
    current_section = None

    for doc in sorted_docs:
        section = doc.metadata.get("section_path") or doc.metadata.get("section_title")
        text = doc.text or ""
        tokens = _approx_token_count(text)

        new_section = section != current_section and current
        over_budget = current and current_tokens + tokens > settings.parent_chunk_size

        if new_section or over_budget:
            groups.append(current)
            current = []
            current_tokens = 0

        current.append(doc)
        current_section = section
        current_tokens += tokens

    if current:
        groups.append(current)

    return groups


def documents_to_hierarchical_nodes(
    documents: list[Document],
    *,
    persist_parents: bool = True,
) -> HierarchicalNodes:
    """
    Convert page Documents into parent + child nodes.

    When ENABLE_HIERARCHICAL_CHUNKING=false, falls back to flat child-only chunking.
    """
    if not settings.enable_hierarchical_chunking:
        nodes = _child_splitter().get_nodes_from_documents(documents, show_progress=False)
        for node in nodes:
            meta = node.metadata or {}
            meta.setdefault("node_role", "child")
            meta.setdefault("content_type", "prose")
            node.metadata = meta
        return HierarchicalNodes(parent_nodes=[], embed_nodes=nodes)

    parent_nodes: list[TextNode] = []
    embed_nodes: list[TextNode] = []

    groups = _group_documents_for_parents(documents)
    for group in groups:
        if not group:
            continue

        parent_id = new_parent_id()
        parent_text = _build_parent_text(group)
        if not parent_text.strip():
            continue

        sample_meta = dict(group[0].metadata or {})
        parent_meta = {
            k: v
            for k, v in sample_meta.items()
            if k
            in (
                "source_file",
                "file_path",
                "file_hash",
                "document_type",
                "category",
                "section_path",
                "section_title",
                "section_number",
                "section_level",
                "page_number",
                "parser",
            )
        }
        parent_meta["node_role"] = "parent"
        parent_meta["child_ids"] = []

        children: list[TextNode] = []
        for doc in group:
            text = doc.text or ""
            if not text.strip():
                continue
            block_types = str(doc.metadata.get("block_types", ""))
            base_meta = dict(doc.metadata or {})
            for content_type, segment in _split_into_segments(text, block_types=block_types):
                if content_type == "code":
                    children.extend(
                        _chunk_code(segment, base_meta, parent_id=parent_id)
                    )
                else:
                    children.extend(
                        _chunk_prose(segment, base_meta, parent_id=parent_id)
                    )

        child_ids = [c.node_id for c in children if c.node_id]
        parent_meta["child_ids"] = child_ids
        parent_node = TextNode(text=parent_text, metadata=parent_meta, id_=parent_id)
        parent_nodes.append(parent_node)
        embed_nodes.extend(children)

    if persist_parents and parent_nodes:
        register_parent_nodes(parent_nodes)

    logger.info(
        "Hierarchical chunking: %d parents, %d embeddable children",
        len(parent_nodes),
        len(embed_nodes),
    )
    return HierarchicalNodes(parent_nodes=parent_nodes, embed_nodes=embed_nodes)