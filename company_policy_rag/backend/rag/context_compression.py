from __future__ import annotations

from backend.models.chunk import Chunk, ChunkMetadata, ChunkRole
from backend.models.rag import ScoredChunk
from backend.utils.logging import logger


class ContextCompressor:
    """
    Handles context filtering, section deduplication, token budget enforcement,
    and Parent-Child Context Expansion.
    """

    def __init__(
        self,
        enable_parent_expansion: bool = True,
        max_token_budget: int = 4096,
    ) -> None:
        self.enable_parent_expansion = enable_parent_expansion
        self.max_token_budget = max_token_budget

    def expand_to_parents(
        self,
        chunks: list[ScoredChunk],
        docstore: dict[str, Chunk] | None = None,
        enable_expansion: bool | None = None,
    ) -> list[ScoredChunk]:
        """
        Replace child chunks with parent document sections when available,
        deduplicating by parent_id while retaining highest relevance score.
        """
        effective_expansion = self.enable_parent_expansion if enable_expansion is None else enable_expansion
        if not effective_expansion or not chunks:
            return chunks

        expanded: list[ScoredChunk] = []
        seen_parents: set[str] = set()

        for sc in chunks:
            meta = sc.chunk.metadata
            parent_id = meta.parent_id
            node_role = meta.node_role

            if node_role == ChunkRole.PARENT or not parent_id:
                expanded.append(sc)
                continue

            pid = str(parent_id)
            if pid in seen_parents:
                continue
            seen_parents.add(pid)

            parent_chunk = docstore.get(pid) if docstore else None
            if parent_chunk is not None:
                # Merge metadata
                new_meta_dict = meta.model_dump()
                new_meta_dict["node_role"] = str(ChunkRole.PARENT.value)
                new_meta_dict["extra"] = {
                    **(meta.extra or {}),
                    "expanded_from_child_id": sc.chunk.id,
                }
                new_chunk = Chunk(
                    id=parent_chunk.id,
                    text=parent_chunk.text,
                    metadata=ChunkMetadata(**new_meta_dict),
                    token_count=parent_chunk.token_count or len(parent_chunk.text.split()),
                )
                expanded.append(
                    ScoredChunk(
                        chunk=new_chunk,
                        score=sc.score,
                        rerank_score=sc.rerank_score,
                        sparse_score=sc.sparse_score,
                        dense_score=sc.dense_score,
                        rank=sc.rank,
                    )
                )
            else:
                expanded.append(sc)

        return expanded if expanded else chunks

    def pack_complementary_chunks(
        self,
        chunks: list[ScoredChunk],
        query: str,
        max_chunks: int = 5,
    ) -> list[ScoredChunk]:
        """
        Ensure balanced coverage across complementary categories (description, code, task/architecture, inputs/outputs)
        for procedural, building, or implementation queries to prevent redundant description chunks from crowding out code.
        """
        if not chunks or len(chunks) <= max_chunks:
            return chunks

        query_lower = query.lower()
        is_building_query = any(
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
            )
        )

        if not is_building_query:
            return chunks[:max_chunks]

        prose_chunks: list[ScoredChunk] = []
        code_chunks: list[ScoredChunk] = []
        table_chunks: list[ScoredChunk] = []
        other_chunks: list[ScoredChunk] = []

        for sc in chunks:
            ct = str(sc.chunk.metadata.content_type).lower()
            extra_ct = str(sc.chunk.metadata.extra.get("content_type", "")).lower()
            is_code = (
                ct in ("contenttype.code", "code")
                or extra_ct == "code"
                or "```" in sc.chunk.text
                or getattr(sc.chunk.metadata, "has_code", False)
            )
            is_table = (
                ct in ("contenttype.table", "table")
                or extra_ct == "table"
                or getattr(sc.chunk.metadata, "has_tables", False)
            )
            if is_code:
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

        # 1. Top 1-2 code chunks (vital for implementation queries)
        _add(code_chunks, limit=2)
        # 2. Top 1-2 prose description chunks
        _add(prose_chunks, limit=2)
        # 3. Top 1 table / diagram chunk
        _add(table_chunks, limit=1)
        # 4. Fill remaining budget from highest scored chunks
        for sc in chunks:
            if len(selected) >= max_chunks:
                break
            if sc.chunk.id not in seen_ids:
                selected.append(sc)
                seen_ids.add(sc.chunk.id)

        return selected if selected else chunks[:max_chunks]

    def format_context_for_prompt(self, chunks: list[ScoredChunk]) -> str:
        """
        Format retrieved scored chunks into structured [Source N] context blocks for LLM synthesis.
        Enforces token budget clipping.
        """
        if not chunks:
            return "No relevant context found."

        context_blocks: list[str] = []
        total_estimated_tokens = 0

        for idx, sc in enumerate(chunks, start=1):
            meta = sc.chunk.metadata
            source_file = meta.source_file or "document"
            section = meta.section_title or meta.section_path or "General Section"
            page_str = f" | Page {meta.page_number}" if meta.page_number is not None else ""

            header = f"[Source {idx}] File: {source_file} | Section: {section}{page_str}"
            block = f"{header}\n{sc.chunk.text.strip()}\n"

            # Rough token estimate: ~1.3 tokens per word
            block_tokens = int(len(block.split()) * 1.3)
            if total_estimated_tokens + block_tokens > self.max_token_budget and context_blocks:
                logger.info("Context formatted clipped at %d sources (%d estimated tokens)", idx - 1, total_estimated_tokens)
                break

            context_blocks.append(block)
            total_estimated_tokens += block_tokens

        return "\n\n".join(context_blocks)
