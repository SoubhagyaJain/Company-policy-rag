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
    ) -> list[ScoredChunk]:
        """
        Replace child chunks with parent document sections when available,
        deduplicating by parent_id while retaining highest relevance score.
        """
        if not self.enable_parent_expansion or not chunks:
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
