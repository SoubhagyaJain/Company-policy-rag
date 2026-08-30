from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from backend.models.rag import RetrievalStrategy


ResponseMode = Literal["compact", "standard", "detailed"]


@dataclass(frozen=True, slots=True)
class ResponseModeConfig:
    """Request-scoped evidence and generation limits for one answer depth."""

    retrieval_top_k: int
    rerank_top_k: int
    max_context_tokens: int
    max_output_tokens: int
    max_citations: int
    answer_style: str
    citation_style: str
    follow_up_style: str

    @property
    def prompt_instructions(self) -> str:
        return "\n".join(
            (
                f"Answer depth: {self.answer_style}",
                self.citation_style,
                self.follow_up_style,
            )
        )

    def apply_to(self, strategy: RetrievalStrategy) -> RetrievalStrategy:
        """Apply depth budgets without replacing intent-aware retrieval behavior."""
        configured = strategy.model_copy(deep=True)
        configured.dense_top_k = self.retrieval_top_k
        configured.bm25_top_k = self.retrieval_top_k
        configured.rerank_top_n = self.rerank_top_k
        return configured


# The local Qwen text model runs with a 4,096-token context window. These
# budgets reserve room for the grounding prompt and generated answer while
# still producing materially different evidence depth for each mode.
RESPONSE_MODES: dict[ResponseMode, ResponseModeConfig] = {
    "compact": ResponseModeConfig(
        retrieval_top_k=4,
        rerank_top_k=3,
        max_context_tokens=800,
        max_output_tokens=320,
        max_citations=2,
        answer_style=(
            "COMPACT. Lead with the answer and include only the essential facts. "
            "Aim for roughly 80-180 words for a normal question. Prefer one short "
            "paragraph or 3-6 bullets; omit background, recap, and repetition."
        ),
        citation_style=(
            "Use only the strongest retrieved evidence and cite the key claims that "
            "materially need support. Never omit a citation needed for truthfulness."
        ),
        follow_up_style=(
            "For a follow-up, resolve the reference from conversation history, then "
            "answer the current question directly from freshly retrieved evidence."
        ),
    ),
    "standard": ResponseModeConfig(
        retrieval_top_k=8,
        rerank_top_k=6,
        max_context_tokens=1_500,
        max_output_tokens=700,
        max_citations=4,
        answer_style=(
            "STANDARD. Give a balanced, focused explanation with the important reasoning. "
            "Aim for roughly 250-600 words when the question justifies it. Use short "
            "headings, examples, steps, or a comparison table only when they improve clarity."
        ),
        citation_style=(
            "Cite every major factual claim with the supporting retrieved source and keep "
            "citation coverage balanced across the answer."
        ),
        follow_up_style=(
            "Use conversation history to resolve references and intent, but ground the "
            "current answer in freshly retrieved evidence rather than repeating the prior answer."
        ),
    ),
    "detailed": ResponseModeConfig(
        retrieval_top_k=15,
        rerank_top_k=10,
        max_context_tokens=2_200,
        max_output_tokens=1_200,
        max_citations=6,
        answer_style=(
            "DETAILED. Provide a rigorous, comprehensive explanation when the evidence supports it. "
            "Explain what, how, and why; include definitions, mechanisms, relationships, examples, "
            "assumptions, edge cases, trade-offs, limitations, equations, or worked steps when relevant. "
            "Do not add repetition merely to make the answer longer."
        ),
        citation_style=(
            "Use dense, truthful citation coverage across major sections and technical claims. "
            "Every citation must come from retrieved evidence that supports the associated claim."
        ),
        follow_up_style=(
            "Resolve follow-up references from the conversation, retrieve fresh broad evidence, "
            "and deepen the answer without blindly injecting or restating the previous response. "
            "If the indexed evidence is insufficient for deeper coverage, say so explicitly."
        ),
    ),
}


def get_response_mode_config(mode: ResponseMode | str) -> ResponseModeConfig:
    """Resolve a validated response mode; callers must not silently fall back."""
    normalized = str(mode).strip().lower()
    if normalized not in RESPONSE_MODES:
        raise ValueError(f"Unsupported response mode: {mode!r}")
    return RESPONSE_MODES[cast(ResponseMode, normalized)]
