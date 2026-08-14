"""
Intelligent Query Router and Retrieval Strategy Selector.
Classifies incoming queries into 5 intent categories and maps them to dynamic retrieval configurations.
"""
from __future__ import annotations

import re
from typing import Any

from backend.models.rag import QueryCategory, QueryClassification, RetrievalStrategy
from backend.utils.logging import logger
from src.config import settings


class QueryRouter:
    """
    Intelligent Query Router classifying incoming user queries and selecting
    the optimal retrieval strategy according to PROJECT.md § Interface Contracts.
    """

    _COMPARISON_PATTERNS = re.compile(
        r"\b(compare|comparison|difference|differences|difference\s+between|versus|vs\.?|better\s+than|pros\s+and\s+cons|differ|distinguish|relative\s+to|in\s+contrast|contrast\s+between)\b",
        re.IGNORECASE,
    )
    _ENUMERATION_PATTERNS = re.compile(
        r"\b(list\s+all|list\s+the|list\s+of|give\s+me\s+all|what\s+are\s+all|all\s+eligible|all\s+available|all\s+paid|all\s+company|enumerate|provide\s+a\s+(full\s+)?list|complete\s+list|breakdown\s+of\s+all|every\s+single)\b|\b(list|enumerate)\b",
        re.IGNORECASE,
    )
    _PROCEDURAL_PATTERNS = re.compile(
        r"\b(how\s+to|how\s+do\s+i|how\s+can\s+i|how\s+should\s+i|step\s+by\s+step|step-by-step|steps\s+to|steps\s+for|procedure\s+for|procedure\s+to|process\s+for|process\s+to|guide\s+to|instructions\s+for|workflow\s+for|how\s+does\s+one)\b",
        re.IGNORECASE,
    )
    _CONVERSATIONAL_PATTERNS = re.compile(
        r"^(hi|hello|hey|heya|hiya|greetings|howdy|good\s+(morning|afternoon|evening|day)|thanks|thank\s+you|who\s+are\s+you|what\s+can\s+you\s+do|how\s+are\s+you|how\s+can\s+you\s+help|help)(\s+(there|everyone|all|assistant|bot|today|me))?[!?,. ]*(\s*(hi|hello|hey|how\s+are\s+you(\s+today)?|how\s+can\s+you\s+help(\s+me)?|who\s+are\s+you|what\s+can\s+you\s+do|how\s+can\s+you\s+assist(\s+me)?)[!?,. ]*)*$",
        re.IGNORECASE,
    )

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    def get_strategy_for_category(self, category: QueryCategory) -> RetrievalStrategy:
        """Return tuned retrieval parameters per query category."""
        if category == QueryCategory.FACTUAL:
            return RetrievalStrategy(
                name="factual_precision",
                dense_top_k=10,
                bm25_top_k=10,
                rrf_k=60,
                rerank_top_n=4,
                min_score_ratio=0.45,
                enable_multi_query=False,
                enable_parent_expansion=False,
                temperature=0.1,
            )
        elif category == QueryCategory.COMPARISON:
            return RetrievalStrategy(
                name="comparison_broad",
                dense_top_k=25,
                bm25_top_k=25,
                rrf_k=60,
                rerank_top_n=10,
                min_score_ratio=0.30,
                enable_multi_query=True,
                enable_parent_expansion=True,
                temperature=0.1,
            )
        elif category == QueryCategory.ENUMERATION:
            return RetrievalStrategy(
                name="enumeration_exhaustive",
                dense_top_k=30,
                bm25_top_k=30,
                rrf_k=60,
                rerank_top_n=12,
                min_score_ratio=0.25,
                enable_multi_query=True,
                enable_parent_expansion=False,
                temperature=0.1,
            )
        elif category == QueryCategory.PROCEDURAL:
            return RetrievalStrategy(
                name="procedural_workflow",
                dense_top_k=15,
                bm25_top_k=15,
                rrf_k=60,
                rerank_top_n=6,
                min_score_ratio=0.35,
                enable_multi_query=False,
                enable_parent_expansion=True,
                temperature=0.1,
            )
        elif category == QueryCategory.CONVERSATIONAL:
            return RetrievalStrategy(
                name="conversational_bypass",
                dense_top_k=0,
                bm25_top_k=0,
                rrf_k=0,
                rerank_top_n=0,
                min_score_ratio=0.0,
                enable_multi_query=False,
                enable_parent_expansion=False,
                temperature=0.7,
            )
        return RetrievalStrategy()

    def classify(self, query: str, history: list[dict[str, Any]] | None = None) -> QueryClassification:
        """Classify user query and return QueryClassification with category, confidence, strategy, and reasoning."""
        clean_q = (query or "").strip()

        if not clean_q:
            cat = QueryCategory.FACTUAL
            conf = 0.50
            reason = "Empty query defaulted to factual."
        elif self._COMPARISON_PATTERNS.search(clean_q):
            cat = QueryCategory.COMPARISON
            conf = 0.92
            reason = "Query contains comparative or differential keywords."
        elif self._ENUMERATION_PATTERNS.search(clean_q):
            cat = QueryCategory.ENUMERATION
            conf = 0.90
            reason = "Query requests enumeration or exhaustive listing."
        elif self._PROCEDURAL_PATTERNS.search(clean_q):
            cat = QueryCategory.PROCEDURAL
            conf = 0.90
            reason = "Query requests procedural instructions or workflow steps."
        elif len(clean_q.split()) <= 7 and self._CONVERSATIONAL_PATTERNS.match(clean_q):
            cat = QueryCategory.CONVERSATIONAL
            conf = 0.95
            reason = "Query matches conversational greeting or pleasantry pattern."
        else:
            cat = QueryCategory.FACTUAL
            conf = 0.85
            reason = "Direct factual policy inquiry."

        threshold = getattr(settings, "query_router_confidence_threshold", 0.70)
        if conf < threshold:
            strategy = RetrievalStrategy()
        else:
            strategy = self.get_strategy_for_category(cat)

        logger.debug(
            "Query '%s' classified as %s (confidence=%.2f, strategy=%s)",
            clean_q,
            cat.value,
            conf,
            strategy.name,
        )
        return QueryClassification(
            category=cat,
            confidence=conf,
            strategy=strategy,
            reasoning=reason,
        )
