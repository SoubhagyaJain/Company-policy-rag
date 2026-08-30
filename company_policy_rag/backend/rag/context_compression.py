from __future__ import annotations

import re

from backend.models.chunk import Chunk, ContentType
from backend.models.rag import ScoredChunk
from backend.utils.logging import logger
from src.docstore import get_parent_node


class ContextCompressor:
    """
    Manages parent-child chunk expansion, complementary chunk packing,
    and prompt context budget formatting.
    """

    def __init__(self, max_token_budget: int = 4000, enable_parent_expansion: bool = True) -> None:
        self.max_token_budget = max_token_budget
        self.enable_parent_expansion = enable_parent_expansion

    def expand_to_parents(
        self,
        chunks: list[ScoredChunk],
        docstore: dict[str, Chunk] | None = None,
        enable_expansion: bool | None = None,
        active_document_id: str | None = None,
    ) -> list[ScoredChunk]:
        """
        If a child chunk has a parent_id, fetch the complete parent text from the docstore
        while maintaining strict single-document boundaries.
        """
        should_expand = self.enable_parent_expansion if enable_expansion is None else enable_expansion
        if not should_expand or not chunks:
            return chunks

        expanded: list[ScoredChunk] = []
        seen_parent_ids: set[str] = set()

        for sc in chunks:
            meta = sc.chunk.metadata
            parent_id = meta.parent_id

            if not parent_id:
                expanded.append(sc)
                continue

            if parent_id in seen_parent_ids:
                continue

            seen_parent_ids.add(parent_id)

            # Check local pipeline docstore first
            if docstore and parent_id in docstore:
                p_chunk = docstore[parent_id]
                p_doc_id = p_chunk.metadata.document_id
                if active_document_id and p_doc_id and p_doc_id != active_document_id:
                    continue
                expanded.append(
                    ScoredChunk(
                        chunk=p_chunk,
                        score=sc.score,
                        rerank_score=sc.rerank_score,
                        dense_score=sc.dense_score,
                        sparse_score=sc.sparse_score,
                        retrieval_source="parent_expansion",
                    )
                )
                continue

            # Check global docstore
            parent_node = get_parent_node(parent_id)
            if parent_node is not None:
                p_doc_id = parent_node.metadata.get("document_id")
                if active_document_id and p_doc_id and p_doc_id != active_document_id:
                    continue

                parent_chunk = Chunk(
                    id=parent_id,
                    text=parent_node.text,
                    metadata=meta.model_copy(
                        update={
                            "chunk_strategy": "parent_expanded",
                            "node_role": "parent",
                        }
                    ),
                )
                expanded.append(
                    ScoredChunk(
                        chunk=parent_chunk,
                        score=sc.score,
                        rerank_score=sc.rerank_score,
                        dense_score=sc.dense_score,
                        sparse_score=sc.sparse_score,
                        retrieval_source="parent_expansion",
                    )
                )
            else:
                expanded.append(sc)

        return expanded if expanded else chunks

    def expand_parent_context(
        self,
        chunks: list[ScoredChunk],
        active_document_id: str | None = None,
    ) -> list[ScoredChunk]:
        """Alias for backward compatibility."""
        return self.expand_to_parents(
            chunks=chunks,
            docstore=None,
            enable_expansion=True,
            active_document_id=active_document_id,
        )

    def pack_complementary_chunks(
        self,
        chunks: list[ScoredChunk],
        query: str,
        max_chunks: int = 6,
    ) -> list[ScoredChunk]:
        """
        Ensure balanced coverage across complementary categories (description, code, diagram/workflow, table, inputs/outputs)
        for procedural, building, or workflow queries to prevent redundant description chunks from crowding out visual/code evidence.
        """
        if not chunks:
            return chunks

        # The same PDF is often uploaded under more than one filename. Chunk
        # IDs differ in that case, so ID-only deduplication sends repeated text
        # to the model and produces duplicate/mixed source cards. Preserve the
        # highest-ranked occurrence of each substantive passage.
        unique_chunks: list[ScoredChunk] = []
        seen_content: set[str] = set()
        for sc in chunks:
            normalized = re.sub(r"\s+", " ", (sc.chunk.text or "")).strip().casefold()
            content_key = normalized if len(normalized) >= 80 else f"{sc.chunk.id}:{normalized}"
            if content_key in seen_content:
                continue
            seen_content.add(content_key)
            unique_chunks.append(sc)
        chunks = unique_chunks

        if len(chunks) <= max_chunks:
            return chunks

        query_lower = query.lower()
        is_building_or_workflow_query = any(
            k in query_lower
            for k in (
                "how to",
                "how can i",
                "make",
                "create",
                "build",
                "code",
                "implementation",
                "develop",
                "setup",
                "construct",
                "agent",
                "workflow",
                "architecture",
                "diagram",
                "flowchart",
                "pipeline",
                "process",
            )
        )

        if not is_building_or_workflow_query:
            return chunks[:max_chunks]

        prose_chunks: list[ScoredChunk] = []
        code_chunks: list[ScoredChunk] = []
        diagram_chunks: list[ScoredChunk] = []
        table_chunks: list[ScoredChunk] = []
        other_chunks: list[ScoredChunk] = []

        for sc in chunks:
            ct = str(sc.chunk.metadata.content_type).lower()
            extra = sc.chunk.metadata.extra or {}
            extra_ct = str(extra.get("content_type", "")).lower()
            v_type = str(extra.get("visual_type", "")).lower()

            is_code = (
                ct in ("contenttype.code", "code")
                or extra_ct == "code"
                or "```" in sc.chunk.text
                or getattr(sc.chunk.metadata, "has_code", False)
            )
            is_diagram = (
                "diagram" in ct
                or "diagram" in v_type
                or "workflow" in v_type
                or extra.get("is_visual_extraction")
            )
            is_table = (
                ct in ("contenttype.table", "table")
                or extra_ct == "table"
                or getattr(sc.chunk.metadata, "has_tables", False)
            )

            if is_diagram:
                diagram_chunks.append(sc)
            elif is_code:
                code_chunks.append(sc)
            elif is_table:
                table_chunks.append(sc)
            elif ct in ("contenttype.prose", "prose"):
                prose_chunks.append(sc)
            else:
                other_chunks.append(sc)

        selected: list[ScoredChunk] = []
        seen_ids: set[str] = set()

        def _add(sc_list: list[ScoredChunk], limit: int):
            count = 0
            for sc in sc_list:
                if sc.chunk.id not in seen_ids and count < limit:
                    selected.append(sc)
                    seen_ids.add(sc.chunk.id)
                    count += 1

        # 1. Top diagram / workflow chunks (critical for architectural queries)
        _add(diagram_chunks, limit=2)
        # 2. Top code chunks (vital for implementation queries)
        _add(code_chunks, limit=2)
        # 3. Top prose description chunks
        _add(prose_chunks, limit=2)
        # 4. Top table chunks
        _add(table_chunks, limit=1)
        # 5. Fill remaining budget from highest scored chunks
        for sc in chunks:
            if len(selected) >= max_chunks:
                break
            if sc.chunk.id not in seen_ids:
                selected.append(sc)
                seen_ids.add(sc.chunk.id)

        return selected if selected else chunks[:max_chunks]

    @staticmethod
    def estimate_chunk_tokens(sc: ScoredChunk) -> int:
        """Estimate a complete source block without cutting citation metadata or text."""
        meta = sc.chunk.metadata
        metadata_words = " ".join(
            str(value or "")
            for value in (
                meta.source_file,
                meta.section_title,
                meta.section_path,
                meta.get_page_identity().display_label,
                sc.chunk.id,
            )
        ).split()
        return max(1, int((len(sc.chunk.text.split()) + len(metadata_words) + 12) * 1.3))

    def pack_to_token_budget(
        self,
        chunks: list[ScoredChunk],
        max_token_budget: int,
    ) -> tuple[list[ScoredChunk], int]:
        """Pack whole ranked chunks into a request-scoped context budget."""
        selected: list[ScoredChunk] = []
        estimated_tokens = 0
        for sc in chunks:
            chunk_tokens = self.estimate_chunk_tokens(sc)
            if selected and estimated_tokens + chunk_tokens > max_token_budget:
                break
            selected.append(sc)
            estimated_tokens += chunk_tokens
            if estimated_tokens >= max_token_budget:
                break
        return selected, estimated_tokens

    def format_context_for_prompt(
        self,
        chunks: list[ScoredChunk],
        max_token_budget: int | None = None,
    ) -> str:
        """
        Format retrieved scored chunks into structured [Source N] and [VISUAL SOURCE N]
        context blocks for LLM synthesis with canonical human-readable page labels.
        """
        if not chunks:
            return "No relevant context found."

        context_blocks: list[str] = []
        total_estimated_tokens = 0
        effective_budget = (
            self.max_token_budget if max_token_budget is None else max(1, int(max_token_budget))
        )

        for idx, sc in enumerate(chunks, start=1):
            meta = sc.chunk.metadata
            extra = meta.extra or {}
            source_file = meta.source_file or "document"
            section = meta.section_title or meta.section_path or "General Section"

            # Resolve canonical display page label
            page_id = meta.get_page_identity()
            page_label_str = page_id.display_label

            is_visual = (
                extra.get("is_visual_extraction", False)
                or "diagram" in str(meta.content_type).lower()
                or extra.get("visual_type") in ("diagram_architecture", "code_screenshot", "table_data", "figure", "image")
            )

            if is_visual:
                visual_type = extra.get("visual_type", "diagram_architecture").upper()
                asset_id = extra.get("asset_id") or (meta.visual_asset_ids[0] if meta.visual_asset_ids else None) or "visual_asset"
                header = (
                    f"[VISUAL SOURCE {idx}] File: {source_file} | Section: {section} | "
                    f"Page: {page_label_str} | Evidence Type: {visual_type} | Visual Asset ID: {asset_id}"
                )
                block = f"{header}\nVisual Evidence:\n{sc.chunk.text.strip()}\n"
            else:
                c_type = "CODE" if ("```" in sc.chunk.text or meta.content_type == ContentType.CODE) else "TEXT"
                header = f"[Source {idx}] File: {source_file} | Section: {section} | Page: {page_label_str} | Evidence Type: {c_type}"
                block = f"{header}\n{sc.chunk.text.strip()}\n"

            # Token estimate: ~1.3 tokens per word
            block_tokens = int(len(block.split()) * 1.3)
            if total_estimated_tokens + block_tokens > effective_budget and context_blocks:
                logger.info("Context formatted clipped at %d sources (%d estimated tokens)", idx - 1, total_estimated_tokens)
                break

            context_blocks.append(block)
            total_estimated_tokens += block_tokens

        return "\n\n".join(context_blocks)
